"""Unused encoder losses — kept for future experiments.

These losses were part of the active encoder loss stack at some point but are
no longer called by any training script.  They are preserved here so they can
be revived without re-implementation.

See ``docs/source/0_architecture/encoder_losses.md`` for the full loss
inventory and call-site map.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F

from fragile.layers.gauge import (
    as_tangent,
    hyperbolic_distance,
    project_to_ball,
)


# =============================================================================
# SupervisedTopologyLoss (requires labels)
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


# =============================================================================
# Superseded chart / code balancing losses
# =============================================================================


def compute_diversity_loss(router_weights: Tensor, num_charts: int, eps: float = 1e-6) -> Tensor:
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
    _N_c, K, _D = codebook.shape
    if K < 2:
        return torch.tensor(0.0, device=v_local.device)

    # Project both to Poincaré ball and compute hyperbolic distances [B, N_c, K]
    v_exp = project_to_ball(v_local).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
    cb_exp = project_to_ball(codebook).unsqueeze(0)  # [1, N_c, K, D]
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
# Superseded code entropy losses (non-differentiable bincount path)
# =============================================================================


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


# =============================================================================
# Dropped regularizers
# =============================================================================


def compute_residual_scale_loss(z_n: Tensor, assume_tangent: bool = True) -> Tensor:
    """Penalize residual gauge scale to preserve macro/meso hierarchy."""
    z_tan = as_tangent(z_n, assume_tangent)
    return (z_tan**2).sum(dim=1).mean()


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


# =============================================================================
# VQ geodesic loss (not wired into Phase 1)
# =============================================================================


def compute_vq_geodesic_loss(
    z_q_all: Tensor,  # [B, N_c, D] quantized codes
    v_local: Tensor,  # [B, D] encoder output
    router_weights: Tensor,  # [B, N_c] soft routing
    commitment_cost: float = 0.25,
) -> Tensor:
    """VQ loss using geodesic distance d_H instead of tangent-space approx."""
    project_to_ball(z_q_all)
    v_proj = project_to_ball(v_local.unsqueeze(1).expand_as(z_q_all))

    # Codebook loss: codes -> encoder output
    d_codebook = hyperbolic_distance(z_q_all, v_proj.detach())  # [B, N_c]
    codebook_loss = (d_codebook**2 * router_weights.detach()).mean(0).sum()

    # Commitment loss: encoder -> codes (STE)
    d_commit = hyperbolic_distance(z_q_all.detach(), v_proj)  # [B, N_c]
    commitment = (d_commit**2 * router_weights.detach()).mean(0).sum()

    return codebook_loss + commitment_cost * commitment


# =============================================================================
# Supervised hyperbolic contrastive loss (requires labels)
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
    z = project_to_ball(z_geo)
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
# Symbol losses (require labels)
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
    z = project_to_ball(z_geo)
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
# Module exports
# =============================================================================

__all__ = [
    "SupervisedTopologyLoss",
    "compute_chart_collapse_penalty",
    "compute_code_collapse_penalty",
    "compute_code_entropy_loss",
    "compute_diversity_loss",
    "compute_hyperbolic_contrastive_loss",
    "compute_orthogonality_loss",
    "compute_per_chart_code_entropy_loss",
    "compute_residual_scale_loss",
    "compute_symbol_calibration_loss",
    "compute_symbol_purity_loss",
    "compute_vq_geodesic_loss",
    "get_loss_schedule",
]
