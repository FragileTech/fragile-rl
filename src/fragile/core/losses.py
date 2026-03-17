import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .layers import FactorizedJumpOperator
from .layers.gauge import exp_map_zero, hyperbolic_distance, log_map_zero


def _project_to_ball(z: torch.Tensor, max_norm: float = 0.99, eps: float = 1e-6) -> torch.Tensor:
    """Project points to interior of the Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


class SupervisedTopologyLoss(nn.Module):
    """
    Supervised topology loss enforcing chart purity, balance, and separation.

    Cross-ref:
        - Definition 25.4.6 (Total Loss)
        - Section 7.8 (Router Weights)
    """

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin
        self.temperature = temperature

        # Learnable chart-to-class mapping (Definition 25.2.1)
        self.chart_to_class = nn.Parameter(torch.randn(num_charts, num_classes) * 0.01)

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: torch.Tensor,  # [B, N_c]
        y_true: torch.Tensor,  # [B] class labels
        z_latent: torch.Tensor | None = None,  # [B, D] optional for metric loss
    ) -> dict[str, torch.Tensor]:
        """
        Compute supervised topology losses.

        Returns dict with individual losses and total.
        """
        B = router_weights.shape[0]
        p_y_k = self.p_y_given_k  # [N_c, C]

        # === Route Alignment Loss (Definition 25.4.5) ===
        # P(Y|x) = sum_k w_k(x) * P(Y|K=k)
        p_y_x = torch.matmul(router_weights, p_y_k)  # [B, C]
        loss_route = F.nll_loss(torch.log(p_y_x + 1e-8), y_true)

        # === Purity Loss (Definition 25.4.1) ===
        # H(Y|K=k) for each chart
        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        # P(K=k) = average router weight
        p_k = router_weights.mean(dim=0)  # [N_c]
        # L_purity = sum_k P(K=k) * H(Y|K=k)
        loss_purity = (p_k * entropy_per_chart).sum()

        # === Balance Loss (Definition 25.4.3) ===
        # KL(p_k || Uniform) = sum_k p_k * log(p_k / (1/N_c))
        uniform = torch.ones_like(p_k) / self.num_charts
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()

        # === Metric Contrastive Loss (Definition 25.4.4) ===
        loss_metric = torch.tensor(0.0, device=router_weights.device)
        if self.lambda_metric > 0 and B > 1:
            # Router overlap as proxy for proximity
            overlap = torch.matmul(router_weights, router_weights.t())  # [B, B]

            # Class disagreement mask
            y_match = (y_true.unsqueeze(1) == y_true.unsqueeze(0)).float()
            y_diff = 1.0 - y_match

            # Penalize high overlap for different-class pairs
            pseudo_dist = 1.0 - overlap
            hinge = F.relu(self.margin - pseudo_dist)
            loss_metric = (y_diff * overlap * hinge**2).sum() / (y_diff.sum() + 1e-8)

        # === Total Loss ===
        loss_total = (
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return {
            "loss_total": loss_total,
            "loss_route": loss_route,
            "loss_purity": loss_purity,
            "loss_balance": loss_balance,
            "loss_metric": loss_metric,
        }


def compute_routing_entropy(router_weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute mean routing entropy (lower = sharper decisions)."""
    entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    return entropy.mean()


def compute_diversity_loss(
    router_weights: torch.Tensor, num_charts: int, eps: float = 1e-6
) -> torch.Tensor:
    """Prevent chart collapse by maximizing entropy of mean usage.

    loss_diversity = log(K) - H(K)
    - Returns 0 when uniform (all charts equally used)
    - Returns positive when collapsed (one chart dominates)

    Overhead: ~1% (simple statistics).
    """
    # Encourage uniform chart usage by maximizing entropy of mean routing.
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    log_K = float(np.log(num_charts))
    return log_K - H_K


def compute_codebook_centering_loss(codebook: torch.Tensor) -> torch.Tensor:
    """Encourage per-chart codebook deltas to be zero-mean.

    Args:
        codebook: [N_c, K, D] codebook deltas
    """
    codebook = _project_to_ball(codebook)
    centers_tan = log_map_zero(codebook).mean(dim=1)  # [N_c, D]
    return (centers_tan**2).sum(dim=1).mean()


def _as_tangent(z: torch.Tensor, assume_tangent: bool) -> torch.Tensor:
    """Return tangent vectors; map from ball if needed."""
    if assume_tangent:
        return z
    return log_map_zero(_project_to_ball(z))


def compute_variance_loss(
    z: torch.Tensor,
    target_std: float = 1.0,
    bundle_size: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Prevent latent collapse using invariant energy (trace of covariance).

    Uses per-bundle or global energy in the tangent space to avoid fixing a basis.
    """
    batch, dim = z.shape
    z_tan = log_map_zero(_project_to_ball(z))
    z_centered = z_tan - z_tan.mean(dim=0, keepdim=True)

    if bundle_size is not None and bundle_size > 0 and dim % bundle_size == 0:
        n_bundles = dim // bundle_size
        bundled = z_centered.reshape(batch, n_bundles, bundle_size)
        energy = (bundled**2).sum(dim=-1).mean(dim=0)
        target = (target_std**2) * bundle_size
        return F.relu(target - energy).mean()

    mean_energy = (z_centered**2).sum(dim=1).mean()
    target = (target_std**2) * dim
    return F.relu(target - mean_energy + eps)


def compute_separation_loss(
    z_geo: torch.Tensor,
    router_weights: torch.Tensor,
    num_charts: int,
    margin: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Force chart centers apart in latent space.

    Uses hinge loss: penalize if centers closer than margin.

    Overhead: ~1% (O(K^2) pairwise distances).
    """
    device = z_geo.device
    z_geo = _project_to_ball(z_geo)
    z_geo_tan = log_map_zero(z_geo)

    centers = []
    for i in range(num_charts):
        weights = router_weights[:, i : i + 1]
        weight_sum = weights.sum() + eps
        center_tan = (z_geo_tan * weights).sum(dim=0) / weight_sum
        centers.append(exp_map_zero(center_tan))
    centers_tensor = torch.stack(centers)

    loss_sep = torch.tensor(0.0, device=device)
    n_pairs = 0
    for i in range(num_charts):
        for j in range(i + 1, num_charts):
            dist = hyperbolic_distance(
                centers_tensor[i].unsqueeze(0),
                centers_tensor[j].unsqueeze(0),
            ).squeeze()
            loss_sep += F.relu(margin - dist)
            n_pairs += 1

    return loss_sep / max(n_pairs, 1)


def compute_chart_center_separation_loss(
    chart_centers: torch.Tensor,
    margin: float = 2.0,
) -> torch.Tensor:
    """Force chart center tokens apart in latent space.

    Uses hinge loss on pairwise distances between chart centers.
    """
    device = chart_centers.device
    chart_centers = _project_to_ball(chart_centers)
    loss_sep = torch.tensor(0.0, device=device)
    n_pairs = 0
    for i in range(chart_centers.shape[0]):
        for j in range(i + 1, chart_centers.shape[0]):
            dist = hyperbolic_distance(
                chart_centers[i].unsqueeze(0),
                chart_centers[j].unsqueeze(0),
            ).squeeze()
            loss_sep += F.relu(margin - dist)
            n_pairs += 1
    return loss_sep / max(n_pairs, 1)


def compute_residual_scale_loss(z_n: torch.Tensor, assume_tangent: bool = True) -> torch.Tensor:
    """Penalize residual gauge scale to preserve macro/meso hierarchy."""
    z_tan = _as_tangent(z_n, assume_tangent)
    return (z_tan**2).sum(dim=1).mean()


def compute_disentangle_loss(
    z_geo: torch.Tensor,
    router_weights: torch.Tensor,
    bundle_size: int | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Gauge coherence using invariant radial statistics.

    L = ||Cov(q(K|x), ||log_0(z)||)||^2_F (global or per-bundle norms).
    """
    device = z_geo.device
    z_tan = log_map_zero(_project_to_ball(z_geo))
    batch, dim = z_tan.shape

    if batch < 2:
        return torch.tensor(0.0, device=device)

    if bundle_size is not None and bundle_size > 0 and dim % bundle_size == 0:
        n_bundles = dim // bundle_size
        bundled = z_tan.reshape(batch, n_bundles, bundle_size)
        norms = torch.norm(bundled, dim=-1)
    else:
        norms = torch.norm(z_tan, dim=-1, keepdim=True)

    norms_centered = norms - norms.mean(dim=0, keepdim=True)
    w_centered = router_weights - router_weights.mean(dim=0, keepdim=True)

    norms_centered = torch.clamp(norms_centered, -100, 100)
    w_centered = torch.clamp(w_centered, -1, 1)

    cross_cov = (w_centered.T @ norms_centered) / max(batch - 1, 1)
    result = (cross_cov**2).sum()

    if torch.isnan(result) or torch.isinf(result):
        return torch.tensor(0.0, device=device, requires_grad=True)

    return result


def compute_orthogonality_loss(
    model: nn.Module,
    max_svd_dim: int = 64,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Penalize anisotropy using singular-value spread (basis-invariant).

    Uses log-variance of singular values. Skip large matrices by default.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n_layers = 0

    for name, param in model.named_parameters():
        if "weight" in name and param.dim() == 2:
            rows, cols = param.shape
            if max(rows, cols) > max_svd_dim:
                continue
            if param.numel() == 0 or not torch.isfinite(param).all():
                continue
            try:
                svals = torch.linalg.svdvals(param)
            except RuntimeError:
                continue
            if not torch.isfinite(svals).all():
                continue
            if svals.numel() < 2:
                continue
            svals = svals.clamp(min=eps)
            log_s = torch.log(svals)
            loss += log_s.var(unbiased=False)
            n_layers += 1

    return loss / max(n_layers, 1)


def compute_kl_prior_loss(
    z_n: torch.Tensor,
    z_tex: torch.Tensor,
    target_std: float = 1.0,
    center: bool = True,
) -> torch.Tensor:
    """Radial prior using invariant energy statistics.

    Matches expected ||z||^2 to target without fixing a basis.
    """

    def energy_loss(z: torch.Tensor) -> torch.Tensor:
        if center:
            z_centered = z - z.mean(dim=0, keepdim=True)
        else:
            z_centered = z
        dim = z_centered.shape[1]
        mean_energy = (z_centered**2).sum(dim=1).mean()
        target = (target_std**2) * dim
        return (mean_energy - target).pow(2)

    return energy_loss(z_n) + energy_loss(z_tex)


def compute_window_loss(
    router_weights: torch.Tensor,
    num_charts: int,
    eps_ground: float = 0.1,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict]:
    """Information-Stability Window (Theorem 15.1.3).

    Ensures chart assignment carries information about input:
    I(X;K) = H(K) - H(K|X) >= eps_ground

    Returns:
        loss: Penalty for insufficient grounding
        metrics: Dictionary with H(K), H(K|X), I(X;K)

    Overhead: ~2% (entropy statistics).
    """
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    H_K_given_X = -(router_weights * torch.log(router_weights + eps)).sum(dim=1).mean()
    I_XK = H_K - H_K_given_X

    # Penalize if I(X;K) < eps_ground (not enough information)
    loss_ground = F.relu(eps_ground - I_XK).pow(2)

    metrics = {
        "H_K": H_K.item(),
        "H_K_given_X": H_K_given_X.item(),
        "I_XK": I_XK.item(),
    }
    return loss_ground, metrics


def compute_code_entropy_loss(
    indices_stack: torch.Tensor,
    num_codes: int,
) -> torch.Tensor:
    """Maximize entropy of code usage within batch (micro-diversity).

    Prevents "index collapse" where a chart routes perfectly but
    maps every point to a single code index.

    Reference: Node 11 (ComplexCheck), Section 15.1 (Mixing Rate).

    Args:
        indices_stack: [B, N_charts] - code indices chosen per chart
        num_codes: Number of codes per chart

    Returns:
        loss: (max_entropy - H) where H is empirical code entropy

    Overhead: ~1% (just counting indices in batch).
    """
    device = indices_stack.device

    # Flatten all indices from all charts
    flat_indices = indices_stack.flatten()

    # Calculate empirical probabilities
    counts = torch.bincount(flat_indices, minlength=num_codes).float()
    probs = counts / (counts.sum() + 1e-6)

    # Filter zeros for log stability
    probs_nonzero = probs[probs > 0]

    # Entropy H(K_code)
    entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero + 1e-6))

    # Maximize entropy → minimize (max_entropy - H)
    max_entropy = math.log(num_codes)
    return torch.tensor(max_entropy, device=device) - entropy


def compute_per_chart_code_entropy_loss(
    indices_stack: torch.Tensor,
    K_chart: torch.Tensor,
    num_charts: int,
    num_codes: int,
) -> torch.Tensor:
    """Maximize code entropy WITHIN each chart separately.

    Unlike global code entropy, this ensures each chart uses
    all its codes uniformly, not just globally balanced.

    The global code entropy can be satisfied even if each chart
    only uses a subset of codes. Per-chart entropy forces every
    chart to utilize all its codes.

    Args:
        indices_stack: [B, num_charts] - code indices per chart
        K_chart: [B] - hard chart assignment for each sample
        num_charts: Number of charts
        num_codes: Codes per chart

    Returns:
        loss: Mean (max_entropy - H_c) across charts
    """
    device = indices_stack.device
    max_entropy = math.log(num_codes)
    total_loss = 0.0
    active_charts = 0

    for c in range(num_charts):
        mask = K_chart == c
        if mask.sum() < 2:  # Need samples to compute entropy
            continue

        # Get codes used by points assigned to this chart
        codes_in_chart = indices_stack[mask, c]

        # Compute entropy for this chart's code usage
        counts = torch.bincount(codes_in_chart, minlength=num_codes).float()
        probs = counts / (counts.sum() + 1e-6)
        probs_nonzero = probs[probs > 0]
        entropy = -torch.sum(probs_nonzero * torch.log(probs_nonzero + 1e-6))

        total_loss += max_entropy - entropy
        active_charts += 1

    if active_charts == 0:
        return torch.tensor(0.0, device=device)

    return total_loss / active_charts


def compute_jump_consistency_loss(
    jump_op: FactorizedJumpOperator,
    z_n_all_charts: torch.Tensor,
    router_weights: torch.Tensor,
) -> torch.Tensor:
    """Train Jump Operator on chart overlaps (vectorized).

    For pairs (i, j), if a point exists in both charts (w_i > 0 and w_j > 0),
    then Jump(i->j) applied to z_n_i should match z_n_j.

    The loss is weighted by the product of chart responsibilities,
    so we only learn transitions where evidence exists (overlap regions).

    Args:
        jump_op: The FactorizedJumpOperator module
        z_n_all_charts: [B, N_c, D] nuisance coords per chart (using each chart's best code)
        router_weights: [B, N_c] soft routing weights

    Returns:
        loss: Mean weighted MSE across all chart pairs
    """
    B, N_c, D = z_n_all_charts.shape
    device = z_n_all_charts.device

    if N_c < 2:
        return torch.tensor(0.0, device=device)

    # Build all N_c*(N_c-1) off-diagonal pair indices once
    src_list = []
    tgt_list = []
    for i in range(N_c):
        for j in range(N_c):
            if i != j:
                src_list.append(i)
                tgt_list.append(j)
    pair_src = torch.tensor(src_list, dtype=torch.long, device=device)  # [P]
    pair_tgt = torch.tensor(tgt_list, dtype=torch.long, device=device)  # [P]
    pair_src.shape[0]  # N_c * (N_c - 1)

    # Overlap weights for all pairs: w_i * w_j  -> [B, P]
    w_overlap = router_weights[:, pair_src] * router_weights[:, pair_tgt]

    # Mask out pairs with negligible total overlap across the batch
    pair_weight_sums = w_overlap.sum(dim=0)  # [P]
    active_mask = pair_weight_sums >= 1e-4  # [P]
    num_active = int(active_mask.sum().item())

    if num_active == 0:
        return torch.tensor(0.0, device=device)

    # Narrow to active pairs only
    active_src = pair_src[active_mask]  # [A]
    active_tgt = pair_tgt[active_mask]  # [A]
    w_active = w_overlap[:, active_mask]  # [B, A]
    A = active_src.shape[0]

    # Gather source and target coords: [B, A, D]
    z_sources = z_n_all_charts[:, active_src]
    z_targets = z_n_all_charts[:, active_tgt]

    # Flatten to [B*A, D] for a single jump_op call
    flat_src = z_sources.reshape(B * A, D)
    flat_src_idx = active_src.unsqueeze(0).expand(B, -1).reshape(B * A)
    flat_tgt_idx = active_tgt.unsqueeze(0).expand(B, -1).reshape(B * A)

    # Single batched forward pass through the jump operator
    z_pred_flat = jump_op(flat_src, flat_src_idx, flat_tgt_idx)  # [B*A, D]

    # Hyperbolic distance (vectorized)
    z_pred_flat = _project_to_ball(z_pred_flat)
    z_target_flat = _project_to_ball(z_targets.reshape(B * A, D))
    error_flat = hyperbolic_distance(z_pred_flat, z_target_flat).pow(2)  # [B*A]

    # Reshape and compute per-pair weighted loss
    error = error_flat.view(B, A)  # [B, A]
    w_sums = w_active.sum(dim=0)  # [A]
    pair_losses = (error * w_active).sum(dim=0) / (w_sums + 1e-6)  # [A]

    return pair_losses.mean()


def get_jump_weight_schedule(
    epoch: int,
    warmup_end: int = 50,
    ramp_end: int = 100,
    final_weight: float = 0.1,
) -> float:
    """Compute scheduled jump loss weight.

    Training schedule:
    - Warmup (0 to warmup_end): weight = 0 (let charts form)
    - Ramp (warmup_end to ramp_end): linear 0.01 -> final_weight
    - Full (ramp_end+): weight = final_weight

    Args:
        epoch: Current epoch
        warmup_end: Epoch when warmup ends
        ramp_end: Epoch when ramp ends
        final_weight: Final jump weight

    Returns:
        Current jump weight
    """
    if epoch < warmup_end:
        return 0.0
    if epoch < ramp_end:
        progress = (epoch - warmup_end) / (ramp_end - warmup_end)
        return 0.01 + progress * (final_weight - 0.01)
    return final_weight
