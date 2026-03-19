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
        """Initialize SupervisedTopologyLoss.

        Args:
            num_charts: Number of charts (atlas partitions) in the model.
            num_classes: Number of target classes for supervision.
            lambda_purity: Weight for the purity loss term (Definition 25.4.1).
            lambda_balance: Weight for the balance loss term (Definition 25.4.3).
            lambda_metric: Weight for the metric contrastive loss term
                (Definition 25.4.4).
            margin: Hinge margin for the metric contrastive loss.
            temperature: Temperature for softmax over the chart-to-class
                mapping.
        """
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
        """Compute the conditional probability distribution P(Y|K).

        Returns:
            Tensor: Softmax-normalized chart-to-class probabilities of shape
                ``[num_charts, num_classes]``.
        """
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        router_weights: Tensor,  # [B, N_c]
        y_true: Tensor,  # [B] class labels
        z_latent: Tensor | None = None,  # [B, D] optional for metric loss
    ) -> dict[str, Tensor]:
        """Compute supervised topology losses.

        Combines route alignment, purity, balance, and metric contrastive
        losses into a single total loss (Definition 25.4.6).

        Args:
            router_weights: Soft chart routing probabilities of shape
                ``[B, num_charts]``, where B is the batch size.
            y_true: Ground-truth class labels of shape ``[B]``.
            z_latent: Optional latent embeddings of shape ``[B, D]``. Currently
                unused but reserved for future metric loss variants.

        Returns:
            dict[str, Tensor]: Dictionary with the following scalar tensor
                entries:

                - ``"loss_total"``: Weighted sum of all sub-losses.
                - ``"loss_route"``: Route alignment NLL loss (Definition 25.4.5).
                - ``"loss_purity"``: Conditional entropy of labels given charts
                  (Definition 25.4.1).
                - ``"loss_balance"``: KL divergence of chart usage from uniform
                  (Definition 25.4.3).
                - ``"loss_metric"``: Hinge-based contrastive penalty on router
                  overlap for different-class pairs (Definition 25.4.4).
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

    Computes ``log(K) - H(K)`` where H(K) is the entropy of average chart
    usage across the batch. Returns 0 when usage is perfectly uniform and
    a positive value when one chart dominates.

    Args:
        router_weights: Soft routing probabilities of shape
            ``[B, num_charts]``.
        num_charts: Total number of charts K.
        eps: Small constant added inside log for numerical stability.

    Returns:
        Tensor: Scalar diversity loss ``log(K) - H(K)``.
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

    Computes ``max(p_k) - 1/K`` where ``p_k`` is the mean chart probability
    across the batch. Returns 0 when usage is perfectly uniform and a positive
    value when one chart dominates. Fully differentiable through
    ``router_weights``.

    Args:
        router_weights: Soft routing probabilities of shape
            ``[B, num_charts]``.
        num_charts: Total number of charts K.

    Returns:
        Tensor: Scalar collapse penalty.
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
    between ``v_local`` and ``codebook`` in the Poincare ball, weighted by
    the router. Penalizes low code entropy *within each chart* instead of
    building one global histogram over code indices shared across charts.

    Unlike ``compute_per_chart_code_entropy_loss`` (which uses bincount and
    therefore has zero gradients), this stays differentiable through both the
    encoder outputs and the codebook.

    Args:
        v_local: Encoder output embeddings of shape ``[B, D]``.
        codebook: Codebook embeddings of shape ``[num_charts, K, D]`` where
            K is the number of codes per chart.
        router_weights: Soft routing probabilities of shape
            ``[B, num_charts]``.
        temperature: Temperature scaling for the softmax over code distances.
        eps: Small constant for numerical stability.

    Returns:
        Tensor: Scalar code collapse penalty. Returns 0.0 when K < 2 or
            when no charts have non-negligible mass.
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

    Prevents "index collapse" where a chart routes perfectly but maps every
    point to a single code index. Uses ``bincount`` so gradients do not flow
    through this loss.

    Reference: Node 11 (ComplexCheck), Section 15.1 (Mixing Rate).

    Args:
        indices_stack: Code indices chosen per chart, of shape
            ``[B, num_charts]`` with integer values in ``[0, num_codes)``.
        num_codes: Number of codes per chart.

    Returns:
        Tensor: Scalar loss equal to ``log(num_codes) - H`` where H is the
            empirical entropy of the flattened code index distribution.
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
    """Maximize code entropy within each chart separately.

    Unlike global code entropy, this ensures each chart uses all its codes
    uniformly, not just that codes are globally balanced. The global code
    entropy can be satisfied even if each chart only uses a subset of codes;
    per-chart entropy forces every chart to utilize all its codes.

    Uses ``bincount`` so gradients do not flow through this loss.

    Args:
        indices_stack: Code indices per chart, of shape
            ``[B, num_charts]`` with integer values in ``[0, num_codes)``.
        K_chart: Hard chart assignment for each sample, of shape ``[B]``
            with integer values in ``[0, num_charts)``.
        num_charts: Total number of charts.
        num_codes: Number of codes per chart.

    Returns:
        Tensor: Scalar loss equal to the mean of ``log(num_codes) - H_c``
            across all active charts (those with at least 2 assigned
            samples). Returns 0.0 if no chart is active.
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
    """Penalize residual gauge scale to preserve macro/meso hierarchy.

    Computes the mean squared norm of the tangent representation of ``z_n``,
    encouraging small residuals in the gauge-equivariant decomposition.

    Args:
        z_n: Residual embeddings of shape ``[B, D]``.
        assume_tangent: If True, treat ``z_n`` as already living in the
            tangent space (skip projection). Passed through to
            ``as_tangent``.

    Returns:
        Tensor: Scalar mean squared tangent-space norm.
    """
    z_tan = as_tangent(z_n, assume_tangent)
    return (z_tan**2).sum(dim=1).mean()


def compute_orthogonality_loss(
    model: nn.Module,
    max_svd_dim: int = 64,
    eps: float = 1e-6,
) -> Tensor:
    """Penalize anisotropy using singular-value spread (basis-invariant).

    Iterates over all 2-D weight parameters in ``model``, computes the SVD,
    and returns the mean log-variance of singular values. Matrices larger
    than ``max_svd_dim`` along either axis are skipped to control cost.

    Args:
        model: PyTorch module whose weight matrices are regularized.
        max_svd_dim: Maximum allowed dimension (rows or cols) for a weight
            matrix to be included in the SVD computation.
        eps: Clamping floor for singular values before taking log.

    Returns:
        Tensor: Scalar mean log-variance of singular values across all
            eligible weight matrices. Returns 0.0 if no eligible layers
            are found.
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
    """VQ loss using geodesic (hyperbolic) distance instead of tangent-space approximation.

    Computes the sum of a codebook loss (pulling codes toward encoder outputs)
    and a commitment loss (pulling encoder outputs toward codes), both measured
    with squared hyperbolic distance in the Poincare ball and weighted by
    detached router probabilities.

    Args:
        z_q_all: Quantized code vectors of shape ``[B, num_charts, D]``.
        v_local: Encoder output embeddings of shape ``[B, D]``.
        router_weights: Soft routing probabilities of shape
            ``[B, num_charts]``.
        commitment_cost: Scalar multiplier for the commitment (encoder to
            code) loss term.

    Returns:
        Tensor: Scalar VQ loss equal to
            ``codebook_loss + commitment_cost * commitment_loss``.
    """
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
    """Contrastive loss in geodesic (hyperbolic) space.

    Pulls same-class pairs together and pushes different-class pairs apart
    using pairwise hyperbolic distances in the Poincare ball. Has O(B^2 D)
    complexity; recommended schedule is epoch 50+.

    Loss formula::

        d_ij = hyperbolic_distance(z_i, z_j)
        L_pos = mean_{y_i == y_j}(d_ij^2)
        L_neg = mean_{y_i != y_j}(ReLU(margin - d_ij)^2)
        L = L_pos + L_neg

    Args:
        z_geo: Embeddings of shape ``[B, D]`` to be projected onto the
            Poincare ball before distance computation.
        labels: Integer class labels of shape ``[B]``.
        margin: Minimum desired geodesic distance between embeddings of
            different classes.

    Returns:
        Tensor: Scalar contrastive loss. Returns 0.0 when B < 2.
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
    """Conditional entropy H(Y | chart, code) to encourage pure symbols.

    For each (chart k, code c) pair, computes the label entropy of samples
    assigned to that symbol and weights it by the symbol's empirical
    probability. Recommended schedule is epoch 100+.

    Args:
        K_chart: Hard chart assignment for each sample, of shape ``[B]``
            with integer values in ``[0, num_charts)``.
        indices_stack: Code indices per chart, of shape
            ``[B, num_charts]`` with integer values in ``[0, num_codes)``.
        labels: Ground-truth class labels of shape ``[B]``.
        router_weights: Soft routing probabilities of shape
            ``[B, num_charts]``. Not used in the current computation but
            accepted for API consistency.
        num_charts: Total number of charts.
        num_codes: Number of codes per chart.
        eps: Small constant for numerical stability in log and
            normalization.

    Returns:
        Tensor: Scalar purity loss equal to
            ``sum_{k,c} P(k,c) * H(Y | k, c)``. Returns 0.0 when no
            (chart, code) symbol has at least 2 samples.
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

    For each active (chart k, code c) pair, computes the variance of the
    Poincare-ball radii of the assigned embeddings. The loss is the mean
    variance across all active symbols. Recommended schedule is epoch 100+.

    Args:
        z_geo: Embeddings of shape ``[B, D]`` to be projected onto the
            Poincare ball.
        K_chart: Hard chart assignment for each sample, of shape ``[B]``
            with integer values in ``[0, num_charts)``.
        indices_stack: Code indices per chart, of shape
            ``[B, num_charts]`` with integer values in ``[0, num_codes)``.
        num_charts: Total number of charts.
        num_codes: Number of codes per chart.

    Returns:
        Tensor: Scalar calibration loss equal to the mean radius variance
            across active symbols. Returns 0.0 when no symbol has at least
            2 samples.
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
    """Generic warmup schedule returning a multiplier in [0, final_weight].

    Returns 0.0 for epochs before ``warmup``, linearly ramps from 0.0 to
    ``final_weight`` between ``warmup`` and ``ramp_end``, and returns
    ``final_weight`` for all epochs at or beyond ``ramp_end``. If
    ``ramp_end`` is None the ramp is skipped and ``final_weight`` is
    returned immediately after warmup.

    Args:
        epoch: Current training epoch (0-indexed).
        warmup: Epoch at which the loss first becomes non-zero.
        ramp_end: Epoch at which the ramp reaches ``final_weight``. If
            None, the multiplier jumps to ``final_weight`` right after
            warmup.
        final_weight: Maximum multiplier value at the end of the ramp.

    Returns:
        float: Loss weight multiplier for the current epoch.
    """
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
