"""Hyperbolic loss functions for the TopoEncoder.

Consolidates all active losses (KEEP + NEW Poincaré-aware) into a single canonical module.
DROP-flagged losses (variance, separation, chart_center_sep, disentangle, kl_prior, orbit, vicreg)
remain in core/losses.py but have their config defaults set to zero weight.
"""

import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from .core.layers import FactorizedJumpOperator
from .core.layers.gauge import exp_map_zero, hyperbolic_distance, log_map_zero, mobius_add  # noqa: F401


# =============================================================================
# Helpers
# =============================================================================


def _project_to_ball(z: Tensor, max_norm: float = 0.99, eps: float = 1e-6) -> Tensor:
    """Project points to interior of the Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def _as_tangent(z: Tensor, assume_tangent: bool) -> Tensor:
    """Return tangent vectors; map from ball if needed."""
    if assume_tangent:
        return z
    return log_map_zero(_project_to_ball(z))


# =============================================================================
# KEEP losses (copied verbatim from core/losses.py)
# =============================================================================


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
    def p_y_given_k(self) -> Tensor:
        """P(Y|K) distribution [N_c, C]."""
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: Tensor,  # [B, N_c]
        y_true: Tensor,  # [B] class labels
        z_latent: Tensor | None = None,  # [B, D] optional for metric loss
    ) -> dict[str, Tensor]:
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


def compute_routing_entropy(router_weights: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute mean routing entropy (lower = sharper decisions)."""
    entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    return entropy.mean()


def compute_router_information_metrics(
    router_weights: Tensor,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute occupancy/conditional entropies and their mutual information."""
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    H_K_given_X = -(router_weights * torch.log(router_weights + eps)).sum(dim=1).mean()
    I_XK = H_K - H_K_given_X
    return {
        "H_K": H_K,
        "H_K_given_X": H_K_given_X,
        "I_XK": I_XK,
    }


def compute_router_sharpness_metrics(
    router_weights: Tensor,
) -> dict[str, Tensor]:
    """Summarize per-sample router sharpness from probabilities."""
    top2 = torch.topk(router_weights, k=min(2, router_weights.shape[-1]), dim=-1).values
    top1 = top2[:, 0]
    if top2.shape[-1] > 1:
        top2_prob = top2[:, 1]
    else:
        top2_prob = torch.zeros_like(top1)
    gap = top1 - top2_prob
    return {
        "top1_prob_mean": top1.mean(),
        "top1_prob_p10": torch.quantile(top1, 0.10),
        "top1_prob_p90": torch.quantile(top1, 0.90),
        "top2_prob_mean": top2_prob.mean(),
        "top1_gap_mean": gap.mean(),
    }


def compute_router_score_metrics(
    router_scores: Tensor,
) -> dict[str, Tensor]:
    """Summarize raw router-score geometry before softmax.

    These diagnostics are useful when probabilities saturate or flatten:
    they show whether the underlying logits still have meaningful separation.
    """
    top2 = torch.topk(router_scores, k=min(2, router_scores.shape[-1]), dim=-1).values
    top1 = top2[:, 0]
    if top2.shape[-1] > 1:
        top2_score = top2[:, 1]
    else:
        top2_score = torch.zeros_like(top1)
    gap = top1 - top2_score
    return {
        "score_gap_mean": gap.mean(),
        "score_gap_p50": torch.quantile(gap, 0.50),
        "score_gap_p90": torch.quantile(gap, 0.90),
        "score_gap_p99": torch.quantile(gap, 0.99),
        "score_std": router_scores.std(unbiased=False),
        "score_mean_abs": router_scores.abs().mean(),
    }


def compute_router_margin_loss(
    router_scores: Tensor,
    margin: float = 0.05,
) -> Tensor:
    """Enforce a positive score gap between the winning and runner-up charts.

    A hard Voronoi partition is only meaningful when the selected chart has a
    genuine margin over its competitors. This term acts directly on router
    scores, unlike entropy on probabilities which becomes first-order flat near
    a uniform softmax.
    """
    top2 = torch.topk(router_scores, k=min(2, router_scores.shape[-1]), dim=-1).values
    top1 = top2[:, 0]
    if top2.shape[-1] > 1:
        second = top2[:, 1]
    else:
        second = torch.zeros_like(top1)
    gap = top1 - second
    return F.relu(torch.as_tensor(margin, device=router_scores.device) - gap).mean()


def compute_hard_routing_nll(router_scores: Tensor) -> Tensor:
    """Maximize the Gibbs probability of the deterministic hard chart partition."""
    hard_labels = router_scores.detach().argmax(dim=-1)
    return F.cross_entropy(router_scores, hard_labels)


def compute_diversity_loss(
    router_weights: Tensor, num_charts: int, eps: float = 1e-6
) -> Tensor:
    """Prevent chart collapse by maximizing entropy of mean usage.

    loss_diversity = log(K) - H(K)
    - Returns 0 when uniform (all charts equally used)
    - Returns positive when collapsed (one chart dominates)

    Overhead: ~1% (simple statistics).
    """
    mean_usage = router_weights.mean(dim=0)
    H_K = -(mean_usage * torch.log(mean_usage + eps)).sum()
    log_K = float(np.log(num_charts))
    return log_K - H_K


def _entropy_band_loss(
    entropy: Tensor,
    h_low: float | None,
    h_high: float | None = None,
) -> Tensor:
    """Penalize entropy outside an optional target band."""
    loss = torch.zeros_like(entropy)
    if h_low is not None:
        loss = loss + F.relu(torch.as_tensor(h_low, device=entropy.device) - entropy).pow(2)
    if h_high is not None:
        loss = loss + F.relu(entropy - torch.as_tensor(h_high, device=entropy.device)).pow(2)
    return loss


def compute_chart_usage_band_loss(
    router_weights: Tensor,
    num_charts: int,
    h_low: float | None = None,
    h_high: float | None = None,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, float]]:
    """Encourage healthy chart occupancy using the hard/ST router.

    ``router_weights`` should be the encoder routing tensor from the forward pass.
    Under deterministic hard routing this tensor is straight-through:
    forward values are one-hot chart assignments while gradients flow through the
    underlying softmax scores. That gives the intended semantics for utilization:
    the loss sees actual chart occupancy, not diffuse soft marginals.
    """
    if h_low is None:
        h_low = math.log(max(0.9 * num_charts, 1.0))

    mean_usage = router_weights.mean(dim=0)
    entropy = -(mean_usage * torch.log(mean_usage + eps)).sum()
    loss = _entropy_band_loss(entropy, h_low=h_low, h_high=h_high)

    metrics = {
        "H_usage": entropy.item(),
        "usage_perplexity": float(torch.exp(entropy).item()),
        "usage_active": int((mean_usage > (1.0 / (2.0 * max(num_charts, 1)))).sum().item()),
    }
    return loss, metrics


def compute_sinkhorn_balanced_chart_loss(
    router_scores: Tensor,
    *,
    epsilon: float = 0.05,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> tuple[Tensor, dict[str, float]]:
    """Balance chart occupancy with an entropy-regularized OT assignment target.

    The returned target distribution is row-normalized from a Sinkhorn plan whose
    row marginals are uniform over samples and whose column marginals are uniform
    over charts. Minimizing the cross-entropy from the router scores to this
    detached target encourages globally balanced but locally sharp assignments.
    """
    if router_scores.ndim != 2:
        msg = "router_scores must have shape [B, K]."
        raise ValueError(msg)
    batch_size, num_charts = router_scores.shape
    if batch_size == 0 or num_charts == 0:
        zero = torch.tensor(0.0, device=router_scores.device)
        return zero, {
            "ot_target_top1_mean": 0.0,
            "ot_plan_col_l1": 0.0,
            "ot_plan_row_l1": 0.0,
        }

    log_r = torch.full(
        (batch_size,), -math.log(batch_size), device=router_scores.device, dtype=router_scores.dtype,
    )
    log_c = torch.full(
        (num_charts,), -math.log(num_charts), device=router_scores.device, dtype=router_scores.dtype,
    )
    log_kernel = router_scores / max(float(epsilon), 1e-6)

    u = torch.zeros_like(log_r)
    v = torch.zeros_like(log_c)
    for _ in range(max(int(num_iters), 1)):
        u = log_r - torch.logsumexp(log_kernel + v.unsqueeze(0), dim=1)
        v = log_c - torch.logsumexp(log_kernel + u.unsqueeze(1), dim=0)

    log_plan = log_kernel + u.unsqueeze(1) + v.unsqueeze(0)
    plan = torch.exp(log_plan)
    target = plan / plan.sum(dim=1, keepdim=True).clamp(min=eps)
    target_detached = target.detach()

    log_probs = F.log_softmax(router_scores, dim=-1)
    loss = -(target_detached * log_probs).sum(dim=-1).mean()

    row_target = 1.0 / max(batch_size, 1)
    col_target = 1.0 / max(num_charts, 1)
    metrics = {
        "ot_target_top1_mean": target_detached.max(dim=-1).values.mean().item(),
        "ot_plan_col_l1": (plan.sum(dim=0) - col_target).abs().sum().item(),
        "ot_plan_row_l1": (plan.sum(dim=1) - row_target).abs().sum().item(),
    }
    return loss, metrics


def compute_codebook_centering_loss(codebook: Tensor) -> Tensor:
    """Encourage per-chart codebook deltas to be zero-mean.

    Args:
        codebook: [N_c, K, D] codebook deltas
    """
    codebook = _project_to_ball(codebook)
    centers_tan = log_map_zero(codebook).mean(dim=1)  # [N_c, D]
    return (centers_tan**2).sum(dim=1).mean()


def compute_chart_center_mean_loss(chart_centers: Tensor) -> Tensor:
    """Anchor the atlas barycenter near the origin in tangent coordinates.

    This regularizes the global atlas frame without forcing individual chart
    centers to coincide. The tangent mean ``mean(log_0(c_k))`` is the natural
    origin-centered analogue of zero-centering the per-chart codebook deltas.
    """
    chart_centers = _project_to_ball(chart_centers)
    atlas_mean = log_map_zero(chart_centers).mean(dim=0)
    return atlas_mean.pow(2).sum()


def compute_chart_center_radius_loss(
    chart_centers: Tensor,
    radius_max: float,
    *,
    barrier_beta: float = 4.0,
) -> Tensor:
    """Keep chart centers inside a hyperbolic safe harbor.

    ``radius_max`` is interpreted in geodesic distance from the origin, not as
    a Euclidean ball norm. That avoids under-penalizing boundary drift.
    """
    if chart_centers.numel() == 0:
        return torch.tensor(0.0, device=chart_centers.device, dtype=chart_centers.dtype)

    chart_centers = _project_to_ball(chart_centers)
    origin = torch.zeros_like(chart_centers)
    radii = hyperbolic_distance(chart_centers, origin)
    beta = max(float(barrier_beta), 1e-6)
    barrier = (F.softplus(beta * (radii - radius_max)) - math.log(2.0)) / beta
    barrier = barrier.clamp_min(0.0)
    return barrier.pow(2).mean()


def compute_chart_center_separation_loss(
    chart_centers: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """Keep distinct chart anchors separated in hyperbolic geometry."""
    num_charts = chart_centers.shape[0]
    if num_charts < 2:
        return torch.tensor(0.0, device=chart_centers.device, dtype=chart_centers.dtype)

    chart_centers = _project_to_ball(chart_centers)
    ci = chart_centers.unsqueeze(1).expand(num_charts, num_charts, -1)
    cj = chart_centers.unsqueeze(0).expand(num_charts, num_charts, -1)
    distances = hyperbolic_distance(
        ci.reshape(num_charts * num_charts, -1),
        cj.reshape(num_charts * num_charts, -1),
    ).reshape(num_charts, num_charts)
    mask = torch.triu(
        torch.ones(num_charts, num_charts, device=chart_centers.device, dtype=torch.bool),
        diagonal=1,
    )
    return F.relu(margin - distances[mask]).pow(2).mean()


def compute_residual_scale_loss(z_n: Tensor, assume_tangent: bool = True) -> Tensor:
    """Penalize residual gauge scale to preserve macro/meso hierarchy."""
    z_tan = _as_tangent(z_n, assume_tangent)
    return (z_tan**2).sum(dim=1).mean()


def compute_window_loss(
    router_weights: Tensor,
    num_charts: int,
    eps_ground: float = 0.1,
    eps: float = 1e-6,
) -> tuple[Tensor, dict]:
    """Information-Stability Window (Theorem 15.1.3).

    Ensures chart assignment carries information about input:
    I(X;K) = H(K) - H(K|X) >= eps_ground

    Returns:
        loss: Penalty for insufficient grounding
        metrics: Dictionary with H(K), H(K|X), I(X;K)

    Overhead: ~2% (entropy statistics).
    """
    info = compute_router_information_metrics(router_weights, eps=eps)
    H_K = info["H_K"]
    H_K_given_X = info["H_K_given_X"]
    I_XK = info["I_XK"]

    # Penalize if I(X;K) < eps_ground (not enough information)
    loss_ground = F.relu(eps_ground - I_XK).pow(2)

    metrics = {
        "H_K": H_K.item(),
        "H_K_given_X": H_K_given_X.item(),
        "I_XK": I_XK.item(),
    }
    return loss_ground, metrics


def compute_code_usage_band_loss(
    v_local: Tensor,
    codebook: Tensor,
    router_weights: Tensor,
    *,
    hard_code_indices: Tensor | None = None,
    h_low: float | None = None,
    h_high: float | None = None,
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> tuple[Tensor, dict[str, float]]:
    """Encourage healthy per-chart code usage with straight-through assignments.

    The chart occupancy comes from ``router_weights`` and should therefore use the
    hard/ST encoder router. Code occupancy is formed from a straight-through code
    assignment computed from the same distances used by the VQ path.
    """
    num_charts, num_codes, _dim = codebook.shape
    if num_codes < 2:
        zero = torch.tensor(0.0, device=v_local.device)
        return zero, {
            "H_code_usage": 0.0,
            "code_usage_perplexity": 1.0,
            "active_code_charts": 0,
        }

    if h_low is None:
        h_low = math.log(max(0.75 * num_codes, 1.0))

    v_exp = _project_to_ball(v_local).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
    cb_exp = _project_to_ball(codebook).unsqueeze(0)  # [1, N_c, K, D]
    dist_sq = hyperbolic_distance(v_exp, cb_exp) ** 2  # [B, N_c, K]

    soft_assign = F.softmax(-dist_sq / max(temperature, 1e-6), dim=-1)
    hard_idx = hard_code_indices if hard_code_indices is not None else torch.argmax(soft_assign, dim=-1)
    hard_assign = F.one_hot(hard_idx, num_classes=num_codes).to(soft_assign.dtype)
    assign_st = hard_assign + soft_assign - soft_assign.detach()

    chart_code_mass = (assign_st * router_weights.unsqueeze(-1)).sum(dim=0)  # [N_c, K]
    chart_mass = chart_code_mass.sum(dim=-1)  # [N_c]
    active = chart_mass > eps
    if not active.any():
        zero = torch.tensor(0.0, device=v_local.device)
        return zero, {
            "H_code_usage": 0.0,
            "code_usage_perplexity": 1.0,
            "active_code_charts": 0,
        }

    usage_active = chart_code_mass[active] / chart_mass[active].unsqueeze(-1).clamp(min=eps)
    entropy = -(usage_active * torch.log(usage_active + eps)).sum(dim=-1)
    loss_per_chart = _entropy_band_loss(entropy, h_low=h_low, h_high=h_high)

    weights = chart_mass[active] / chart_mass[active].sum().clamp(min=eps)
    loss = (weights * loss_per_chart).sum()

    mean_entropy = entropy.mean()
    metrics = {
        "H_code_usage": mean_entropy.item(),
        "code_usage_perplexity": float(torch.exp(mean_entropy).item()),
        "active_code_charts": int(active.sum().item()),
    }
    return loss, metrics


def compute_code_entropy_loss(
    indices_stack: Tensor,
    num_codes: int,
) -> Tensor:
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
    indices_stack: Tensor,
    K_chart: Tensor,
    num_charts: int,
    num_codes: int,
) -> Tensor:
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


def compute_orthogonality_loss(
    model: nn.Module,
    max_svd_dim: int = 64,
    eps: float = 1e-6,
) -> Tensor:
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


def compute_jump_consistency_loss(
    jump_op: FactorizedJumpOperator,
    z_n_all_charts: Tensor,
    router_weights: Tensor,
) -> Tensor:
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
    if final_weight <= 0.0:
        return 0.0
    if ramp_end <= warmup_end:
        return final_weight
    if epoch < ramp_end:
        progress = (epoch - warmup_end) / (ramp_end - warmup_end)
        return 0.01 + progress * (final_weight - 0.01)
    return final_weight


# =============================================================================
# Generic schedule utility
# =============================================================================


def get_loss_schedule(
    epoch: int,
    warmup: int,
    ramp_end: int | None = None,
    final_weight: float = 1.0,
) -> float:
    """Generic warmup schedule. Returns multiplier in [0, final_weight]."""
    if epoch < warmup:
        return 0.0
    if ramp_end is None or epoch >= ramp_end:
        return final_weight
    progress = (epoch - warmup) / max(ramp_end - warmup, 1)
    return progress * final_weight


# =============================================================================
# MODIFIED: VQ geodesic loss (replaces tangent-space approximation)
# =============================================================================


def compute_vq_geodesic_loss(
    z_q_all: Tensor,       # [B, N_c, D] quantized codes
    v_local: Tensor,        # [B, D] encoder output
    router_weights: Tensor,  # [B, N_c] soft routing
    commitment_cost: float = 0.25,
) -> Tensor:
    """VQ loss using geodesic distance d_H instead of tangent-space approx."""
    z_q_proj = _project_to_ball(z_q_all)
    v_proj = _project_to_ball(v_local.unsqueeze(1).expand_as(z_q_all))

    # Codebook loss: codes -> encoder output
    d_codebook = hyperbolic_distance(z_q_all, v_proj.detach())  # [B, N_c]
    codebook_loss = (d_codebook ** 2 * router_weights.detach()).mean(0).sum()

    # Commitment loss: encoder -> codes (STE)
    d_commit = hyperbolic_distance(z_q_all.detach(), v_proj)    # [B, N_c]
    commitment = (d_commit ** 2 * router_weights.detach()).mean(0).sum()

    return codebook_loss + commitment_cost * commitment


# =============================================================================
# NEW: Hyperbolic uniformity loss
# =============================================================================


def compute_hyperbolic_uniformity_loss(z_geo: Tensor, eps: float = 1e-6) -> Tensor:
    """Repulsion loss encouraging uniform spread on the Poincare ball.

    O(B^2 D) complexity. Schedule: epoch 50+.

    tau_i = sqrt(D) * (1 - ||z_i||^2) / 2     # conformal temperature
    d_ij = hyperbolic_distance(z_i, z_j)        # pairwise geodesic
    L = log(mean_{i!=j} exp(-tau_i * d_ij))     # log-sum-exp repulsion
    """
    z = _project_to_ball(z_geo)
    B, D = z.shape
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Conformal temperature per point
    r2 = (z ** 2).sum(dim=-1)  # [B]
    tau = math.sqrt(D) * (1.0 - r2) / 2.0  # [B]
    tau = tau.clamp(min=eps)

    # Pairwise geodesic distances
    z_i = z.unsqueeze(1).expand(B, B, D)  # [B, B, D]
    z_j = z.unsqueeze(0).expand(B, B, D)  # [B, B, D]
    d_ij = hyperbolic_distance(
        z_i.reshape(B * B, D), z_j.reshape(B * B, D)
    ).reshape(B, B)  # [B, B]

    # Mask diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

    # Use tau_i for the row (source point)
    exponents = -tau.unsqueeze(1) * d_ij  # [B, B]
    exponents = exponents[mask].reshape(B, B - 1)

    # Log-mean-exp for numerical stability
    max_exp = exponents.max(dim=1, keepdim=True).values
    loss = (max_exp.squeeze(1) + torch.log(
        torch.exp(exponents - max_exp).mean(dim=1) + eps
    )).mean()

    return loss


# =============================================================================
# NEW: Hyperbolic contrastive loss
# =============================================================================


def compute_hyperbolic_contrastive_loss(
    z_geo: Tensor,
    labels: Tensor,
    margin: float = 2.0,
) -> Tensor:
    """Contrastive loss in geodesic space.

    O(B^2 D) complexity. Schedule: epoch 50+.

    d_ij = hyperbolic_distance(z_i, z_j)
    L_pos = mean_{y_i=y_j}(d_ij^2)
    L_neg = mean_{y_i!=y_j}(ReLU(margin - d_ij)^2)
    L = L_pos + L_neg
    """
    z = _project_to_ball(z_geo)
    B, D = z.shape
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Pairwise geodesic distances
    z_i = z.unsqueeze(1).expand(B, B, D).reshape(B * B, D)
    z_j = z.unsqueeze(0).expand(B, B, D).reshape(B * B, D)
    d_ij = hyperbolic_distance(z_i, z_j).reshape(B, B)  # [B, B]

    # Mask diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)
    y_match = (labels.unsqueeze(1) == labels.unsqueeze(0)) & mask
    y_diff = (labels.unsqueeze(1) != labels.unsqueeze(0)) & mask

    # Positive: pull same-class pairs together
    loss_pos = torch.tensor(0.0, device=z.device)
    if y_match.any():
        loss_pos = (d_ij[y_match] ** 2).mean()

    # Negative: push different-class pairs apart
    loss_neg = torch.tensor(0.0, device=z.device)
    if y_diff.any():
        loss_neg = (F.relu(margin - d_ij[y_diff]) ** 2).mean()

    return loss_pos + loss_neg


# =============================================================================
# NEW: Radial calibration loss
# =============================================================================


def compute_radial_calibration_loss(
    z_geo: Tensor,
    router_weights: Tensor,
    num_charts: int,
    *,
    center_points: Tensor | None = None,
    quality_target: Tensor | None = None,
    quality_mix: float = 0.0,
    quality_base_weight: float = 0.0,
    rho_max: float = 4.0,
    rho_band_width: float = 0.75,
    use_hyperbolic_radius: bool = False,
    eps: float = 1e-6,
) -> Tensor:
    """Calibrate radius to routing confidence and sample quality.

    O(BD) complexity.

    The intended use is with chart-local latents or with ``center_points`` set
    to the current chart-mixture barycenter so radius is earned by sample-local
    geometry instead of by pushing the whole atlas frame outward.

    r_i = ||z_i||
    H_i = -sum_k(w_ik * log(w_ik + eps))    # routing entropy
    target_i = 1 - H_i / log(num_charts)     # confident → 1, uncertain → 0
    When ``quality_target`` is provided, the confident shell is gated by
    per-sample quality so confident but inaccurate points are pulled inward.
    ``quality_base_weight`` adds a quality-driven basal shell before confidence
    sharpens it; this avoids the circular failure mode where zero confidence
    implies zero radial target everywhere and the router never develops a
    meaningful hard partition.
    With ``use_hyperbolic_radius=True``, a band loss is used instead of exact
    shell matching so high-quality samples can occupy a radial range rather than
    collapsing to a single shell.
    """
    z = _project_to_ball(z_geo)
    confidence = compute_routing_confidence(router_weights, num_charts, eps=eps)

    mix = min(max(float(quality_mix), 0.0), 1.0)
    if quality_target is None:
        radial_target = confidence
    else:
        quality = quality_target.clamp(0.0, 1.0)
        gated_target = confidence * ((1.0 - mix) + mix * quality)
        base_weight = min(max(float(quality_base_weight), 0.0), 1.0)
        radial_target = (1.0 - base_weight) * gated_target + base_weight * quality

    if center_points is not None:
        centers = _project_to_ball(center_points)
        rho = hyperbolic_distance(z, centers)
        r = None
    else:
        r = z.norm(dim=-1)  # [B]
        rho = 2.0 * torch.atanh(r.clamp(max=1.0 - eps))

    if not use_hyperbolic_radius:
        if r is None:
            r = torch.tanh(0.5 * rho)
        return ((r - radial_target) ** 2).mean()

    rho_cap = max(float(rho_max), eps)
    band = max(float(rho_band_width), 0.0)
    rho_target = radial_target * rho_cap
    rho_lo = (rho_target - band).clamp(min=0.0)
    rho_hi = (rho_target + band).clamp(max=rho_cap)
    return (F.relu(rho_lo - rho).pow(2) + F.relu(rho - rho_hi).pow(2)).mean()


def compute_routing_confidence(
    router_weights: Tensor,
    num_charts: int,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Map routing entropy to a confidence score in [0, 1]."""
    H = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    log_K = math.log(max(num_charts, 2))
    return (1.0 - H / log_K).clamp(0.0, 1.0)


def compute_error_quality_targets(
    per_sample_error: Tensor,
    *,
    alpha: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Turn detached per-sample errors into quality targets in [0, 1]."""
    if per_sample_error.ndim != 1:
        msg = "per_sample_error must have shape [B]"
        raise ValueError(msg)

    error = per_sample_error.detach()
    mean_error = error.mean().clamp_min(eps)
    return torch.exp(-float(alpha) * error / mean_error).clamp(0.0, 1.0)


def compute_rank_quality_targets(
    per_sample_error: Tensor,
) -> Tensor:
    """Turn per-sample errors into rank-based quality targets in [0, 1].

    Lower-error samples get higher quality, but the target is based on batch
    ordering instead of absolute scale. This is useful when we care more about
    "better than peers" than "close to zero error".
    """
    if per_sample_error.ndim != 1:
        msg = "per_sample_error must have shape [B]"
        raise ValueError(msg)

    error = per_sample_error.detach()
    if error.numel() <= 1:
        return torch.ones_like(error)

    order = torch.argsort(error)
    ranks = torch.empty_like(error)
    ranks[order] = torch.arange(
        error.numel(),
        device=error.device,
        dtype=error.dtype,
    )
    denom = max(error.numel() - 1, 1)
    return (1.0 - ranks / float(denom)).clamp(0.0, 1.0)


def mix_quality_targets(
    absolute_quality: Tensor,
    rank_quality: Tensor,
    *,
    rank_mix: float = 0.0,
) -> Tensor:
    """Blend absolute and rank-based quality targets into a single score."""
    mix = min(max(float(rank_mix), 0.0), 1.0)
    return ((1.0 - mix) * absolute_quality + mix * rank_quality).clamp(0.0, 1.0)


def combine_quality_targets(
    primary_quality: Tensor,
    secondary_quality: Tensor,
    *,
    primary_weight: float = 0.7,
) -> Tensor:
    """Combine two quality signals with a weighted average."""
    weight = min(max(float(primary_weight), 0.0), 1.0)
    return (weight * primary_quality + (1.0 - weight) * secondary_quality).clamp(0.0, 1.0)


def compute_confidence_calibration_loss(
    router_weights: Tensor,
    quality_target: Tensor,
    num_charts: int,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Align router confidence with a detached per-sample quality target."""
    confidence = compute_routing_confidence(router_weights, num_charts, eps=eps)
    return F.smooth_l1_loss(confidence, quality_target.clamp(0.0, 1.0))


# =============================================================================
# NEW: Pre-squash tangent barrier
# =============================================================================


def compute_v_tangent_barrier_loss(
    v_raw: Tensor,
    *,
    target_radius: float = 0.9,
    max_norm: float = 0.99,
) -> Tensor:
    """Penalize the pre-squash tangent norm once it enters the saturated tail."""
    if v_raw.numel() == 0:
        return torch.tensor(0.0, device=v_raw.device, dtype=v_raw.dtype)

    radius = min(max(float(target_radius), 0.0), float(max_norm) - 1e-4)
    tangent_target = math.atanh(radius)
    v_norm = v_raw.norm(dim=-1)
    return F.relu(v_norm - tangent_target).pow(2).mean()


# =============================================================================
# NEW: Codebook spread loss
# =============================================================================


def compute_codebook_spread_loss(
    codebook: Tensor,
    margin: float = 1.0,
) -> Tensor:
    """Encourage intra-chart codebook codes to be spread apart.

    O(N_c * K^2 * D) complexity. Schedule: epoch 0+.

    For each chart c:
        d_ij = hyperbolic_distance(codes_c[i], codes_c[j])
        L_c = mean(ReLU(margin - d_ij))   # hinge on all pairs
    L = mean_c(L_c)

    Args:
        codebook: [N_c, K, D] codebook parameters
        margin: minimum geodesic distance between codes
    """
    codebook_proj = _project_to_ball(codebook)  # [N_c, K, D]
    N_c, K, D = codebook_proj.shape
    device = codebook.device

    if K < 2:
        return torch.tensor(0.0, device=device)

    total_loss = torch.tensor(0.0, device=device)
    for c in range(N_c):
        codes_c = codebook_proj[c]  # [K, D]
        # All pairs
        ci = codes_c.unsqueeze(1).expand(K, K, D).reshape(K * K, D)
        cj = codes_c.unsqueeze(0).expand(K, K, D).reshape(K * K, D)
        d = hyperbolic_distance(ci, cj).reshape(K, K)  # [K, K]

        # Upper triangle (avoid double-counting and diagonal)
        mask = torch.triu(torch.ones(K, K, device=device, dtype=torch.bool), diagonal=1)
        d_pairs = d[mask]
        total_loss = total_loss + F.relu(margin - d_pairs).mean()

    return total_loss / N_c


# =============================================================================
# NEW: Symbol purity loss
# =============================================================================


def compute_symbol_purity_loss(
    K_chart: Tensor,
    indices_stack: Tensor,
    labels: Tensor,
    router_weights: Tensor,
    num_charts: int,
    num_codes: int,
    eps: float = 1e-6,
) -> Tensor:
    """Conditional entropy H(Y | chart, code) -- encourage pure symbols.

    Schedule: epoch 100+.

    For each (chart k, code c):
        mask = (K_chart == k) & (indices_stack[:, k] == c)
        P(y|k,c) = histogram of labels[mask] / count
        H(Y|k,c) = entropy of P(y|k,c)
        P(k,c) = count / total
    L = sum_{k,c} P(k,c) * H(Y|k,c)
    """
    device = K_chart.device
    B = K_chart.shape[0]
    num_classes = int(labels.max().item()) + 1

    total_loss = torch.tensor(0.0, device=device)
    total_count = 0

    for k in range(num_charts):
        for c in range(num_codes):
            mask = (K_chart == k) & (indices_stack[:, k] == c)
            count = mask.sum().item()
            if count < 2:
                continue

            # Label histogram for this (chart, code) symbol
            symbol_labels = labels[mask]
            counts = torch.bincount(symbol_labels, minlength=num_classes).float()
            probs = counts / (counts.sum() + eps)
            probs_nz = probs[probs > 0]
            H_yc = -(probs_nz * torch.log(probs_nz + eps)).sum()

            p_kc = count / B
            total_loss = total_loss + p_kc * H_yc
            total_count += 1

    if total_count == 0:
        return torch.tensor(0.0, device=device)

    return total_loss


# =============================================================================
# NEW: Symbol calibration loss
# =============================================================================


def compute_symbol_calibration_loss(
    z_geo: Tensor,
    K_chart: Tensor,
    indices_stack: Tensor,
    num_charts: int,
    num_codes: int,
) -> Tensor:
    """Encourage radial consistency within each symbol (chart, code).

    Schedule: epoch 100+.

    For each active (chart k, code c):
        r_kc = ||z_geo[mask]||    # radii in this symbol
        L_kc = Var(r_kc)
    L = mean over active symbols
    """
    z = _project_to_ball(z_geo)
    device = z.device

    total_var = torch.tensor(0.0, device=device)
    active = 0

    for k in range(num_charts):
        for c in range(num_codes):
            mask = (K_chart == k) & (indices_stack[:, k] == c)
            if mask.sum() < 2:
                continue
            r = z[mask].norm(dim=-1)  # radii
            total_var = total_var + r.var()
            active += 1

    if active == 0:
        return torch.tensor(0.0, device=device)

    return total_var / active


# =============================================================================
# Anti-collapse penalties (differentiable)
# =============================================================================


def compute_chart_collapse_penalty(
    router_weights: Tensor,
    num_charts: int,
) -> Tensor:
    """Direct penalty on chart usage concentration.

    penalty = max(p_k) - 1/K where p_k = mean chart probability across batch.
    Returns 0 when perfectly uniform, positive when one chart dominates.

    Fully differentiable through router_weights.
    """
    mean_usage = router_weights.mean(dim=0)  # [N_c]
    return mean_usage.max() - 1.0 / num_charts


def compute_code_collapse_penalty(
    v_local: Tensor,  # [B, D]
    codebook: Tensor,  # [N_c, K, D]
    router_weights: Tensor,  # [B, N_c]
    temperature: float = 1.0,
    eps: float = 1e-6,
) -> Tensor:
    """Differentiable penalty for code usage collapse.

    Computes soft code assignment probabilities from hyperbolic distances
    between v_local and codebook in the Poincaré ball, weighted by router.
    Penalizes low code entropy *within each chart* instead of building one
    global histogram over code indices shared across charts.

    Unlike per_chart_code_entropy (which uses bincount -> zero gradients),
    this stays differentiable through both the encoder outputs and the codebook.
    """
    N_c, K, _D = codebook.shape
    if K < 2:
        return torch.tensor(0.0, device=v_local.device)

    # Project both to Poincaré ball and compute hyperbolic distances [B, N_c, K]
    v_exp = _project_to_ball(v_local).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
    cb_exp = _project_to_ball(codebook).unsqueeze(0)  # [1, N_c, K, D]
    dist_sq = hyperbolic_distance(v_exp, cb_exp) ** 2  # [B, N_c, K]

    # Soft code assignments per chart
    soft_assign = F.softmax(-dist_sq / max(temperature, 1e-6), dim=-1)  # [B, N_c, K]

    # Weight by chart responsibility, but keep chart balancing separate from
    # code balancing by detaching the router here.
    w = router_weights.detach().unsqueeze(-1)  # [B, N_c, 1]
    chart_usage = (soft_assign * w).sum(dim=0)  # [N_c, K]
    chart_mass = chart_usage.sum(dim=-1)  # [N_c]
    active = chart_mass > eps
    if not active.any():
        return torch.tensor(0.0, device=v_local.device)

    usage_active = chart_usage[active] / chart_mass[active].unsqueeze(-1).clamp(min=eps)
    entropy = -(usage_active * torch.log(usage_active + eps)).sum(dim=-1)
    loss_per_chart = math.log(K) - entropy

    weights = chart_mass[active] / chart_mass[active].sum().clamp(min=eps)
    return (weights * loss_per_chart).sum()


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Helpers
    "_project_to_ball",
    "_as_tangent",
    # KEEP losses
    "SupervisedTopologyLoss",
    "compute_routing_entropy",
    "compute_router_information_metrics",
    "compute_router_sharpness_metrics",
    "compute_router_score_metrics",
    "compute_router_margin_loss",
    "compute_hard_routing_nll",
    "compute_diversity_loss",
    "compute_chart_usage_band_loss",
    "compute_sinkhorn_balanced_chart_loss",
    "compute_codebook_centering_loss",
    "compute_chart_center_mean_loss",
    "compute_chart_center_radius_loss",
    "compute_chart_center_separation_loss",
    "compute_residual_scale_loss",
    "compute_window_loss",
    "compute_code_usage_band_loss",
    "compute_code_entropy_loss",
    "compute_per_chart_code_entropy_loss",
    "compute_orthogonality_loss",
    "compute_jump_consistency_loss",
    "get_jump_weight_schedule",
    # Schedule
    "get_loss_schedule",
    # Modified VQ
    "compute_vq_geodesic_loss",
    # NEW losses
    "compute_hyperbolic_uniformity_loss",
    "compute_hyperbolic_contrastive_loss",
    "compute_routing_confidence",
    "compute_error_quality_targets",
    "compute_radial_calibration_loss",
    "compute_confidence_calibration_loss",
    "compute_v_tangent_barrier_loss",
    "compute_codebook_spread_loss",
    "compute_symbol_purity_loss",
    "compute_symbol_calibration_loss",
    # Anti-collapse penalties
    "compute_chart_collapse_penalty",
    "compute_code_collapse_penalty",
]
