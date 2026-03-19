"""Hyperbolic loss functions for the TopoEncoder.

Consolidates all active losses (KEEP + NEW Poincaré-aware) into a single canonical module.
DROP-flagged losses (variance, separation, chart_center_sep, disentangle, kl_prior, orbit, vicreg)
remain in core/losses.py but have their config defaults set to zero weight.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor
import torch.nn.functional as F

from fragile.layers import FactorizedJumpOperator
from fragile.layers.gauge import (  # noqa: F401  # noqa: F401
    as_tangent,
    exp_map_zero,
    hyperbolic_distance,
    log_map_zero,
    mobius_add,
    project_to_ball,
)


if TYPE_CHECKING:
    from fragile.vla.config import VLAConfig


def compute_routing_entropy(router_weights: Tensor, eps: float = 1e-6) -> Tensor:
    """Compute mean routing entropy (lower = sharper decisions).

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``,
            where ``B`` is batch size and ``K`` is the number of charts.
        eps: Small constant added inside the log for numerical stability.

    Returns:
        Scalar tensor with the mean entropy across the batch.
    """
    entropy = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    return entropy.mean()


def compute_router_information_metrics(
    router_weights: Tensor,
    eps: float = 1e-6,
) -> dict[str, Tensor]:
    """Compute occupancy/conditional entropies and their mutual information.

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        eps: Small constant added inside the log for numerical stability.

    Returns:
        Dictionary with the following keys:

        - ``"H_K"``: Scalar tensor, marginal occupancy entropy H(K).
        - ``"H_K_given_X"``: Scalar tensor, conditional entropy H(K|X).
        - ``"I_XK"``: Scalar tensor, mutual information I(X;K) = H(K) - H(K|X).
    """
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
    """Summarize per-sample router sharpness from probabilities.

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.

    Returns:
        Dictionary with the following keys:

        - ``"top1_prob_mean"``: Scalar tensor, mean of the top-1 probability.
        - ``"top1_prob_p10"``: Scalar tensor, 10th percentile of top-1 probability.
        - ``"top1_prob_p90"``: Scalar tensor, 90th percentile of top-1 probability.
        - ``"top2_prob_mean"``: Scalar tensor, mean of the runner-up probability.
        - ``"top1_gap_mean"``: Scalar tensor, mean gap between top-1 and top-2 probs.
    """
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

    Args:
        router_scores: Raw (pre-softmax) router logits of shape ``[B, K]``.

    Returns:
        Dictionary with the following keys:

        - ``"score_gap_mean"``: Scalar tensor, mean gap between top-1 and top-2 scores.
        - ``"score_gap_p50"``: Scalar tensor, 50th percentile of the score gap.
        - ``"score_gap_p90"``: Scalar tensor, 90th percentile of the score gap.
        - ``"score_gap_p99"``: Scalar tensor, 99th percentile of the score gap.
        - ``"score_std"``: Scalar tensor, standard deviation of all scores.
        - ``"score_mean_abs"``: Scalar tensor, mean absolute value of all scores.
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

    Args:
        router_scores: Raw (pre-softmax) router logits of shape ``[B, K]``.
        margin: Minimum desired gap between the top-1 and top-2 scores.

    Returns:
        Scalar tensor with the mean hinge penalty over the batch.
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
    """Maximize the Gibbs probability of the deterministic hard chart partition.

    Args:
        router_scores: Raw (pre-softmax) router logits of shape ``[B, K]``.

    Returns:
        Scalar tensor with the cross-entropy loss between the softmax scores
        and the argmax-derived hard labels.
    """
    hard_labels = router_scores.detach().argmax(dim=-1)
    return F.cross_entropy(router_scores, hard_labels)


def _entropy_band_loss(
    entropy: Tensor,
    h_low: float | None,
    h_high: float | None = None,
) -> Tensor:
    """Penalize entropy outside an optional target band.

    Applies a one-sided squared ReLU penalty for entropy below ``h_low``
    and/or above ``h_high``.

    Args:
        entropy: Entropy values, arbitrary shape.
        h_low: Lower entropy bound. If provided, values below this incur a
            squared penalty.
        h_high: Upper entropy bound. If provided, values above this incur a
            squared penalty.

    Returns:
        Tensor of the same shape as ``entropy`` with per-element penalty values.
    """
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

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        num_charts: Total number of charts ``K``.
        h_low: Lower bound for the occupancy entropy band. Defaults to
            ``log(0.9 * num_charts)`` if ``None``.
        h_high: Upper bound for the occupancy entropy band. ``None`` means no
            upper penalty.
        eps: Small constant added inside the log for numerical stability.

    Returns:
        Tuple of two elements:

        - **loss**: Scalar tensor with the entropy band penalty.
        - **metrics**: Dictionary with keys ``"H_usage"`` (occupancy entropy),
          ``"usage_perplexity"`` (exp of the entropy), and ``"usage_active"``
          (number of charts with non-negligible occupancy).
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

    Args:
        router_scores: Raw (pre-softmax) router logits of shape ``[B, K]``.
        epsilon: Entropy regularization strength for the Sinkhorn iterations.
        num_iters: Number of Sinkhorn iterations.
        eps: Small constant for numerical stability in normalization.

    Returns:
        Tuple of two elements:

        - **loss**: Scalar tensor with the cross-entropy loss from the router
          softmax to the detached Sinkhorn target.
        - **metrics**: Dictionary with keys ``"ot_target_top1_mean"`` (mean
          top-1 value of the target distribution), ``"ot_plan_col_l1"``
          (L1 deviation of column marginals from uniform), and
          ``"ot_plan_row_l1"`` (L1 deviation of row marginals from uniform).

    Raises:
        ValueError: If ``router_scores`` does not have exactly 2 dimensions.
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
        (batch_size,),
        -math.log(batch_size),
        device=router_scores.device,
        dtype=router_scores.dtype,
    )
    log_c = torch.full(
        (num_charts,),
        -math.log(num_charts),
        device=router_scores.device,
        dtype=router_scores.dtype,
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
        codebook: Codebook parameters of shape ``[N_c, K, D]``, where ``N_c``
            is the number of charts, ``K`` is the number of codes per chart,
            and ``D`` is the embedding dimension.

    Returns:
        Scalar tensor with the mean squared tangent-space center norm across
        charts.
    """
    codebook = project_to_ball(codebook)
    centers_tan = log_map_zero(codebook).mean(dim=1)  # [N_c, D]
    return (centers_tan**2).sum(dim=1).mean()


def compute_chart_center_mean_loss(chart_centers: Tensor) -> Tensor:
    """Anchor the atlas barycenter near the origin in tangent coordinates.

    This regularizes the global atlas frame without forcing individual chart
    centers to coincide. The tangent mean ``mean(log_0(c_k))`` is the natural
    origin-centered analogue of zero-centering the per-chart codebook deltas.

    Args:
        chart_centers: Chart center positions of shape ``[K, D]`` on the
            Poincare ball, where ``K`` is the number of charts and ``D`` is
            the embedding dimension.

    Returns:
        Scalar tensor with the squared norm of the tangent-space atlas
        barycenter.
    """
    chart_centers = project_to_ball(chart_centers)
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

    Args:
        chart_centers: Chart center positions of shape ``[K, D]`` on the
            Poincare ball.
        radius_max: Maximum allowed geodesic distance from the origin.
        barrier_beta: Steepness parameter for the softplus barrier function.

    Returns:
        Scalar tensor with the mean squared barrier penalty across chart
        centers.
    """
    if chart_centers.numel() == 0:
        return torch.tensor(0.0, device=chart_centers.device, dtype=chart_centers.dtype)

    chart_centers = project_to_ball(chart_centers)
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
    """Keep distinct chart anchors separated in hyperbolic geometry.

    Args:
        chart_centers: Chart center positions of shape ``[K, D]`` on the
            Poincare ball.
        margin: Minimum geodesic distance enforced between each pair of chart
            centers.

    Returns:
        Scalar tensor with the mean squared hinge penalty over all unique
        chart-center pairs.
    """
    num_charts = chart_centers.shape[0]
    if num_charts < 2:
        return torch.tensor(0.0, device=chart_centers.device, dtype=chart_centers.dtype)

    chart_centers = project_to_ball(chart_centers)
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


def compute_window_loss(
    router_weights: Tensor,
    eps_ground: float = 0.1,
    eps: float = 1e-6,
) -> tuple[Tensor, dict]:
    """Information-Stability Window (Theorem 15.1.3).

    Ensures chart assignment carries information about input:
    ``I(X;K) = H(K) - H(K|X) >= eps_ground``.

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        eps_ground: Minimum mutual information threshold below which the
            squared penalty activates.
        eps: Small constant added inside the log for numerical stability.

    Returns:
        Tuple of two elements:

        - **loss**: Scalar tensor with the squared ReLU penalty for
          insufficient mutual information.
        - **metrics**: Dictionary with keys ``"H_K"`` (marginal occupancy
          entropy), ``"H_K_given_X"`` (conditional entropy), and ``"I_XK"``
          (mutual information).
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

    Args:
        v_local: Chart-local latent vectors of shape ``[B, D]``.
        codebook: Codebook parameters of shape ``[N_c, K, D]``.
        router_weights: Per-sample routing weights of shape ``[B, N_c]``.
        hard_code_indices: Pre-computed hard code indices of shape ``[B, N_c]``.
            If ``None``, argmax of the soft assignment is used.
        h_low: Lower bound for the per-chart code-usage entropy band.
            Defaults to ``log(0.75 * K)`` if ``None``.
        h_high: Upper bound for the per-chart code-usage entropy band.
            ``None`` means no upper penalty.
        temperature: Temperature for the soft code assignment softmax.
        eps: Small constant for numerical stability.

    Returns:
        Tuple of two elements:

        - **loss**: Scalar tensor with the occupancy-weighted entropy band
          penalty across active charts.
        - **metrics**: Dictionary with keys ``"H_code_usage"`` (mean per-chart
          code entropy), ``"code_usage_perplexity"`` (exp of that entropy),
          and ``"active_code_charts"`` (number of charts with non-negligible
          occupancy).
    """
    _num_charts, num_codes, _dim = codebook.shape
    if num_codes < 2:
        zero = torch.tensor(0.0, device=v_local.device)
        return zero, {
            "H_code_usage": 0.0,
            "code_usage_perplexity": 1.0,
            "active_code_charts": 0,
        }

    if h_low is None:
        h_low = math.log(max(0.75 * num_codes, 1.0))

    v_exp = project_to_ball(v_local).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
    cb_exp = project_to_ball(codebook).unsqueeze(0)  # [1, N_c, K, D]
    dist_sq = hyperbolic_distance(v_exp, cb_exp) ** 2  # [B, N_c, K]

    soft_assign = F.softmax(-dist_sq / max(temperature, 1e-6), dim=-1)
    hard_idx = (
        hard_code_indices if hard_code_indices is not None else torch.argmax(soft_assign, dim=-1)
    )
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
        jump_op: The ``FactorizedJumpOperator`` module used to map nuisance
            coordinates from one chart to another.
        z_n_all_charts: Nuisance coordinates per chart of shape ``[B, N_c, D]``,
            where ``B`` is batch size, ``N_c`` is the number of charts, and
            ``D`` is the embedding dimension.
        router_weights: Soft routing weights of shape ``[B, N_c]``.

    Returns:
        Scalar tensor with the mean overlap-weighted squared hyperbolic
        distance across all active chart pairs.
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
    z_pred_flat = project_to_ball(z_pred_flat)
    z_target_flat = project_to_ball(z_targets.reshape(B * A, D))
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
        epoch: Current training epoch number.
        warmup_end: Epoch at which the warmup phase ends and the ramp begins.
        ramp_end: Epoch at which the ramp phase ends and the full weight is
            applied.
        final_weight: Target jump loss weight after the ramp completes.

    Returns:
        The scheduled jump loss weight as a float for the given epoch.
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
# NEW: Hyperbolic uniformity loss
# =============================================================================


def compute_hyperbolic_uniformity_loss(z_geo: Tensor, eps: float = 1e-6) -> Tensor:
    """Repulsion loss encouraging uniform spread on the Poincare ball.

    O(B^2 D) complexity. Schedule: epoch 50+.

    Uses a conformal-temperature-weighted log-sum-exp repulsion kernel:
    ``tau_i = sqrt(D) * (1 - ||z_i||^2) / 2``,
    ``L = mean_i log(mean_{j!=i} exp(-tau_i * d_ij))``.

    Args:
        z_geo: Latent embeddings of shape ``[B, D]`` on the Poincare ball.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor with the mean log-sum-exp repulsion loss across all
        samples.
    """
    z = project_to_ball(z_geo)
    B, D = z.shape
    if B < 2:
        return torch.tensor(0.0, device=z.device)

    # Conformal temperature per point
    r2 = (z**2).sum(dim=-1)  # [B]
    tau = math.sqrt(D) * (1.0 - r2) / 2.0  # [B]
    tau = tau.clamp(min=eps)

    # Pairwise geodesic distances
    z_i = z.unsqueeze(1).expand(B, B, D)  # [B, B, D]
    z_j = z.unsqueeze(0).expand(B, B, D)  # [B, B, D]
    d_ij = hyperbolic_distance(z_i.reshape(B * B, D), z_j.reshape(B * B, D)).reshape(
        B, B
    )  # [B, B]

    # Mask diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=z.device)

    # Use tau_i for the row (source point)
    exponents = -tau.unsqueeze(1) * d_ij  # [B, B]
    exponents = exponents[mask].reshape(B, B - 1)

    # Log-mean-exp for numerical stability
    max_exp = exponents.max(dim=1, keepdim=True).values
    return (
        max_exp.squeeze(1) + torch.log(torch.exp(exponents - max_exp).mean(dim=1) + eps)
    ).mean()


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

    When ``quality_target`` is provided, the confident shell is gated by
    per-sample quality so confident but inaccurate points are pulled inward.
    ``quality_base_weight`` adds a quality-driven basal shell before confidence
    sharpens it; this avoids the circular failure mode where zero confidence
    implies zero radial target everywhere and the router never develops a
    meaningful hard partition.
    With ``use_hyperbolic_radius=True``, a band loss is used instead of exact
    shell matching so high-quality samples can occupy a radial range rather than
    collapsing to a single shell.

    Args:
        z_geo: Latent embeddings of shape ``[B, D]`` on the Poincare ball.
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        num_charts: Total number of charts ``K``.
        center_points: Optional per-sample chart-mixture barycenters of shape
            ``[B, D]``. When provided, radius is measured as the hyperbolic
            distance from each point to its center instead of from the origin.
        quality_target: Optional per-sample quality scores of shape ``[B]`` in
            ``[0, 1]`` that gate the radial target.
        quality_mix: Interpolation weight in ``[0, 1]`` blending pure
            confidence with quality-gated confidence.
        quality_base_weight: Weight in ``[0, 1]`` for an unconditional
            quality-driven basal radial target.
        rho_max: Maximum geodesic distance used to scale the radial target
            when ``use_hyperbolic_radius`` is ``True``.
        rho_band_width: Half-width of the acceptable radial band when
            ``use_hyperbolic_radius`` is ``True``.
        use_hyperbolic_radius: If ``True``, use a band loss in geodesic
            distance instead of Euclidean shell matching.
        eps: Small constant for numerical stability.

    Returns:
        Scalar tensor with the mean squared radial calibration penalty.
    """
    z = project_to_ball(z_geo)
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
        centers = project_to_ball(center_points)
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
    """Map routing entropy to a confidence score in [0, 1].

    Confidence is defined as ``1 - H / log(K)`` where ``H`` is the per-sample
    routing entropy and ``K`` is the number of charts.

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        num_charts: Total number of charts ``K``.
        eps: Small constant added inside the log for numerical stability.

    Returns:
        Tensor of shape ``[B]`` with per-sample confidence values clamped to
        ``[0, 1]``.
    """
    H = -(router_weights * torch.log(router_weights + eps)).sum(dim=1)
    log_K = math.log(max(num_charts, 2))
    return (1.0 - H / log_K).clamp(0.0, 1.0)


def compute_error_quality_targets(
    per_sample_error: Tensor,
    *,
    alpha: float = 2.0,
    eps: float = 1e-6,
) -> Tensor:
    """Turn detached per-sample errors into quality targets in [0, 1].

    Quality is computed as ``exp(-alpha * error / mean_error)`` so that
    low-error samples receive quality close to 1.

    Args:
        per_sample_error: Per-sample reconstruction or VQ errors of shape
            ``[B]``.
        alpha: Steepness parameter controlling how quickly quality decays
            with increasing error.
        eps: Small constant to avoid division by zero in mean error.

    Returns:
        Tensor of shape ``[B]`` with per-sample quality values clamped to
        ``[0, 1]``.

    Raises:
        ValueError: If ``per_sample_error`` does not have exactly 1 dimension.
    """
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

    Args:
        per_sample_error: Per-sample errors of shape ``[B]``.

    Returns:
        Tensor of shape ``[B]`` with rank-based quality values in ``[0, 1]``.
        The sample with the lowest error receives quality 1.0 and the highest
        receives 0.0.

    Raises:
        ValueError: If ``per_sample_error`` does not have exactly 1 dimension.
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
    """Blend absolute and rank-based quality targets into a single score.

    Args:
        absolute_quality: Absolute quality scores of shape ``[B]`` in
            ``[0, 1]``.
        rank_quality: Rank-based quality scores of shape ``[B]`` in
            ``[0, 1]``.
        rank_mix: Interpolation weight in ``[0, 1]``. A value of 0 yields
            pure absolute quality; 1 yields pure rank quality.

    Returns:
        Tensor of shape ``[B]`` with blended quality values clamped to
        ``[0, 1]``.
    """
    mix = min(max(float(rank_mix), 0.0), 1.0)
    return ((1.0 - mix) * absolute_quality + mix * rank_quality).clamp(0.0, 1.0)


def combine_quality_targets(
    primary_quality: Tensor,
    secondary_quality: Tensor,
    *,
    primary_weight: float = 0.7,
) -> Tensor:
    """Combine two quality signals with a weighted average.

    Args:
        primary_quality: Primary quality scores of shape ``[B]`` in ``[0, 1]``.
        secondary_quality: Secondary quality scores of shape ``[B]`` in
            ``[0, 1]``.
        primary_weight: Weight in ``[0, 1]`` for the primary signal. The
            secondary signal receives ``1 - primary_weight``.

    Returns:
        Tensor of shape ``[B]`` with combined quality values clamped to
        ``[0, 1]``.
    """
    weight = min(max(float(primary_weight), 0.0), 1.0)
    return (weight * primary_quality + (1.0 - weight) * secondary_quality).clamp(0.0, 1.0)


def compute_confidence_calibration_loss(
    router_weights: Tensor,
    quality_target: Tensor,
    num_charts: int,
    *,
    eps: float = 1e-6,
) -> Tensor:
    """Align router confidence with a detached per-sample quality target.

    Args:
        router_weights: Per-sample routing probability vectors of shape ``[B, K]``.
        quality_target: Detached per-sample quality scores of shape ``[B]`` in
            ``[0, 1]``.
        num_charts: Total number of charts ``K``.
        eps: Small constant for numerical stability in confidence computation.

    Returns:
        Scalar tensor with the smooth L1 loss between routing confidence and
        the quality target.
    """
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
    """Penalize the pre-squash tangent norm once it enters the saturated tail.

    Args:
        v_raw: Pre-squash (tangent-space) latent vectors of shape
            ``[B, D]`` or ``[..., D]``.
        target_radius: Desired maximum Euclidean radius in the Poincare ball.
            Converted to a tangent-space threshold via ``atanh``.
        max_norm: Upper bound used to keep ``target_radius`` within a safe
            range for the ``atanh`` conversion.

    Returns:
        Scalar tensor with the mean squared ReLU penalty for norms exceeding
        the tangent-space threshold.
    """
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

    For each chart c, a hinge penalty is applied to all unique code pairs
    whose geodesic distance falls below the margin.

    Args:
        codebook: Codebook parameters of shape ``[N_c, K, D]``, where ``N_c``
            is the number of charts, ``K`` is the number of codes per chart,
            and ``D`` is the embedding dimension.
        margin: Minimum geodesic distance enforced between every pair of codes
            within the same chart.

    Returns:
        Scalar tensor with the mean hinge penalty averaged across charts.
    """
    codebook_proj = project_to_ball(codebook)  # [N_c, K, D]
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
# Functions extracted from vla/losses.py
# =============================================================================


def _deterministic_st_router_weights(router_scores: torch.Tensor) -> torch.Tensor:
    """Build deterministic straight-through one-hot router weights from scores.

    The forward pass produces one-hot vectors (argmax), while gradients flow
    through the underlying softmax via the straight-through estimator.

    Args:
        router_scores: Raw (pre-softmax) router logits of shape ``[B, K]``.

    Returns:
        Tensor of shape ``[B, K]`` with one-hot forward values and softmax
        gradients.
    """
    soft = F.softmax(router_scores, dim=-1)
    one_hot = F.one_hot(router_scores.argmax(dim=-1), router_scores.shape[-1]).to(soft.dtype)
    return one_hot + soft - soft.detach()


def orthogonality_loss(zn: torch.Tensor, ztex: torch.Tensor) -> torch.Tensor:
    """Penalize correlation between z_n and z_tex.

    Uses squared cosine similarity (normalized dot product) so the loss is
    scale-invariant.  This prevents explosions when tangent-vector norms grow
    as points migrate toward the Poincaré boundary (expected behaviour from
    radial calibration).

    If dimensions match: squared cosine similarity, meaned over batch.
    If dimensions differ: Frobenius norm of cross-correlation matrix
    (columns pre-normalized).

    Args:
        zn: Navigational latent vectors of shape ``[B, D1]`` in tangent space.
        ztex: Texture latent vectors of shape ``[B, D2]`` in tangent space.

    Returns:
        Scalar tensor with the decorrelation loss, bounded in ``[0, 1]``.
        When ``D1 == D2``, this is the mean squared cosine similarity. When
        ``D1 != D2``, this is the mean squared entry of the column-normalized
        cross-correlation matrix.
    """
    if zn.shape[-1] == ztex.shape[-1]:
        # Squared cosine similarity per sample, mean over batch
        zn_n = F.normalize(zn, dim=-1)
        ztex_n = F.normalize(ztex, dim=-1)
        return ((zn_n * ztex_n).sum(dim=-1) ** 2).mean()

    zn_c = zn - zn.mean(dim=0, keepdim=True)
    ztex_c = ztex - ztex.mean(dim=0, keepdim=True)
    # Normalize columns so the cross-correlation is scale-invariant
    zn_c = F.normalize(zn_c, dim=0)
    ztex_c = F.normalize(ztex_c, dim=0)
    C = zn_c.T @ ztex_c  # [D1, D2], entries in [-1, 1]
    return (C**2).mean()


def compute_phase1_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    vq_loss: torch.Tensor,
    enc_router_weights: torch.Tensor,
    dec_router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    encoder: torch.nn.Module,
    config: VLAConfig,
    *,
    router_reg_weights: torch.Tensor | None = None,
    usage_router_weights: torch.Tensor | None = None,
    c_bar: torch.Tensor | None = None,
    v_local: torch.Tensor | None = None,
    indices_stack: torch.Tensor | None = None,
    router_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Assemble Phase 1 (encoder-only) loss from all active terms.

    Gathers reconstruction, VQ, routing, chart-usage, uniformity, radial
    calibration, codebook, and consistency losses according to the weights
    specified in ``config``. Losses that do not flow through ``z_n`` are
    accumulated in ``base_loss``; those that do are in ``zn_reg_loss``.

    Args:
        x: Original input tensor of shape ``[B, ...]``.
        x_recon: Reconstructed input tensor of shape ``[B, ...]``.
        vq_loss: Scalar VQ commitment loss from the encoder forward pass.
        enc_router_weights: Encoder routing weights of shape ``[B, K]``.
        dec_router_weights: Decoder routing weights of shape ``[B, K]``.
        z_geo: Geometric latent embeddings of shape ``[B, D]`` on the
            Poincare ball.
        encoder: The encoder module (or a wrapper whose ``.encoder`` attribute
            holds the atlas encoder).
        config: VLA configuration object carrying all loss weights and
            hyper-parameters.
        router_reg_weights: Soft router weights used for regularization terms.
            If ``None``, fetched from the atlas encoder's cached attribute.
        usage_router_weights: Router weights used for chart/code usage
            measurement. Defaults to ``enc_router_weights`` if ``None``.
        c_bar: Per-sample chart-mixture barycenters of shape ``[B, D]``.
            If ``None``, fetched from the atlas encoder's cached attribute.
        v_local: Chart-local latent vectors of shape ``[B, D]``. If ``None``,
            fetched from the atlas encoder's cached attribute.
        indices_stack: Hard code indices of shape ``[B, N_c]``. If ``None``,
            fetched from the atlas encoder's cached attribute.
        router_scores: Raw pre-softmax router logits of shape ``[B, K]``.
            If ``None``, fetched from the atlas encoder's cached attribute.

    Returns:
        Tuple of three elements:

        - **base_loss**: Scalar tensor aggregating all loss terms that do NOT
          flow through ``z_n`` (reconstruction, VQ, routing, codebook, etc.).
        - **zn_reg_loss**: Scalar tensor aggregating loss terms that flow
          through ``z_geo`` / ``z_n`` (uniformity, radial calibration).
        - **metrics**: Dictionary mapping metric names to float values for
          logging. Includes at minimum ``"total"``, ``"recon"``, ``"vq"``,
          ``"entropy"``, ``"consistency"``, and default-zero entries for all
          optional loss terms.
    """
    metrics: dict[str, float] = {}
    atlas_encoder = getattr(encoder, "encoder", encoder)
    if router_reg_weights is None:
        router_reg_weights = getattr(
            atlas_encoder,
            "_last_soft_router_weights_live",
            enc_router_weights,
        )
    if usage_router_weights is None:
        usage_router_weights = enc_router_weights
    if c_bar is None:
        c_bar = getattr(atlas_encoder, "_last_c_bar", None)
    if v_local is None:
        v_local = getattr(atlas_encoder, "_last_v_local", None)
    v_raw = getattr(atlas_encoder, "_last_v_raw", None)
    if indices_stack is None:
        indices_stack = getattr(atlas_encoder, "_last_indices_stack", None)
    if router_scores is None:
        router_scores = getattr(atlas_encoder, "_last_router_scores_live", None)

    loss_x = x
    loss_x_recon = x_recon
    loss_space_pair = getattr(encoder, "loss_space_pair", None)
    if callable(loss_space_pair):
        loss_x, loss_x_recon = loss_space_pair(x, x_recon)
        metrics["recon_raw"] = F.mse_loss(x_recon, x).item()

    # Feature reconstruction (MSE)
    loss_recon = F.mse_loss(loss_x_recon, loss_x)
    base_loss = config.w_feature_recon * loss_recon
    metrics["recon"] = loss_recon.item()
    per_sample_recon_error = (
        F
        .mse_loss(
            loss_x_recon,
            loss_x,
            reduction="none",
        )
        .reshape(x.shape[0], -1)
        .mean(dim=1)
    )
    recon_quality_abs = compute_error_quality_targets(
        per_sample_recon_error,
        alpha=config.radial_quality_alpha,
    )
    recon_quality_rank = compute_rank_quality_targets(per_sample_recon_error)
    recon_quality = mix_quality_targets(
        recon_quality_abs,
        recon_quality_rank,
        rank_mix=config.radial_quality_rank_mix,
    )
    vq_quality = torch.ones_like(recon_quality)
    has_vq_quality = False
    if (
        hasattr(atlas_encoder, "codebook")
        and v_local is not None
        and indices_stack is not None
        and getattr(config, "radial_vq_alpha", 0.0) > 0
    ):
        selected_chart = torch.argmax(enc_router_weights.detach(), dim=1)
        selected_code = indices_stack.gather(1, selected_chart.unsqueeze(1)).squeeze(1)
        codebook_proj = project_to_ball(atlas_encoder.codebook)
        selected_codes = codebook_proj[selected_chart, selected_code]
        per_sample_vq_error = hyperbolic_distance(v_local, selected_codes).pow(2)
        vq_quality_abs = compute_error_quality_targets(
            per_sample_vq_error,
            alpha=config.radial_vq_alpha,
        )
        vq_quality_rank = compute_rank_quality_targets(per_sample_vq_error)
        vq_quality = mix_quality_targets(
            vq_quality_abs,
            vq_quality_rank,
            rank_mix=config.radial_quality_rank_mix,
        )
        has_vq_quality = True
    if has_vq_quality:
        quality_target = combine_quality_targets(
            recon_quality,
            vq_quality,
            primary_weight=config.radial_recon_quality_weight,
        )
    else:
        quality_target = recon_quality
    quality_mix = min(max(float(config.radial_quality_mix), 0.0), 1.0)
    quality_base_weight = min(
        max(float(getattr(config, "radial_quality_base_weight", 0.0)), 0.0), 1.0
    )
    routing_confidence = compute_routing_confidence(
        router_reg_weights.detach(),
        config.num_charts,
    )
    gated_radial_target = routing_confidence * ((1.0 - quality_mix) + quality_mix * quality_target)
    radial_target = (
        1.0 - quality_base_weight
    ) * gated_radial_target + quality_base_weight * quality_target
    metrics["recon_quality_mean"] = recon_quality.mean().item()
    metrics["vq_quality_mean"] = vq_quality.mean().item()
    metrics["combined_quality_mean"] = quality_target.mean().item()
    metrics["routing_confidence_mean"] = routing_confidence.mean().item()
    metrics["radial_target_mean"] = radial_target.mean().item()

    # VQ loss (from encoder)
    base_loss = base_loss + config.w_vq * vq_loss
    metrics["vq"] = vq_loss.item()

    # Local routing entropy: minimize H(K|X) for high-confidence assignments.
    loss_entropy = compute_routing_entropy(router_reg_weights)
    base_loss = base_loss + config.w_entropy * loss_entropy
    metrics["entropy"] = loss_entropy.item()
    if config.w_router_margin > 0 and router_scores is not None:
        loss_margin = compute_router_margin_loss(
            router_scores,
            margin=config.router_margin_target,
        )
        base_loss = base_loss + config.w_router_margin * loss_margin
        metrics["router_margin"] = loss_margin.item()
    else:
        metrics["router_margin"] = 0.0
    if config.w_hard_routing_nll > 0 and router_scores is not None:
        loss_hard_nll = compute_hard_routing_nll(router_scores)
        base_loss = base_loss + config.w_hard_routing_nll * loss_hard_nll
        metrics["hard_routing_nll"] = loss_hard_nll.item()
    else:
        metrics["hard_routing_nll"] = 0.0
    info_metrics = compute_router_information_metrics(router_reg_weights)
    metrics.update({k: v.item() for k, v in info_metrics.items()})
    sharpness_metrics = compute_router_sharpness_metrics(router_reg_weights)
    metrics.update({k: v.item() for k, v in sharpness_metrics.items()})

    if config.w_chart_ot > 0:
        if router_scores is None:
            router_scores = torch.log(router_reg_weights.clamp(min=1e-8))
        loss_chart_ot, chart_ot_metrics = compute_sinkhorn_balanced_chart_loss(
            router_scores,
            epsilon=config.chart_ot_epsilon,
            num_iters=config.chart_ot_iters,
        )
        base_loss = base_loss + config.w_chart_ot * loss_chart_ot
        metrics["chart_ot"] = loss_chart_ot.item()
        metrics.update(chart_ot_metrics)
    else:
        metrics["chart_ot"] = 0.0

    # Chart usage should reflect the actual forward assignment, so it is
    # measured on the hard/ST router tensor rather than the live soft router.
    loss_chart_usage, chart_usage_metrics = compute_chart_usage_band_loss(
        usage_router_weights,
        config.num_charts,
        h_low=config.chart_usage_entropy_low,
        h_high=config.chart_usage_entropy_high,
    )
    base_loss = base_loss + config.w_diversity * loss_chart_usage
    metrics["chart_usage"] = loss_chart_usage.item()
    metrics.update(chart_usage_metrics)

    # z_n regularization terms (flow through z_geo -> z_n)
    zn_reg_loss = torch.zeros((), device=x.device)

    # Hyperbolic uniformity
    if config.w_uniformity > 0:
        loss_unif = compute_hyperbolic_uniformity_loss(z_geo)
        zn_reg_loss = zn_reg_loss + config.w_uniformity * loss_unif
        metrics["uniformity"] = loss_unif.item()

    # Radial calibration
    if config.w_radial_calibration > 0:
        radial_latent = z_geo
        radial_center = c_bar
        if radial_center is None and v_local is not None:
            radial_latent = v_local
        if radial_center is not None:
            metrics["local_radius_mean"] = (
                hyperbolic_distance(
                    project_to_ball(z_geo),
                    project_to_ball(radial_center),
                )
                .mean()
                .item()
            )
        elif v_local is not None:
            origin = torch.zeros_like(v_local)
            metrics["local_radius_mean"] = (
                hyperbolic_distance(
                    project_to_ball(v_local),
                    origin,
                )
                .mean()
                .item()
            )
        else:
            metrics["local_radius_mean"] = 0.0
        loss_radcal = compute_radial_calibration_loss(
            radial_latent,
            router_reg_weights.detach(),
            config.num_charts,
            center_points=radial_center.detach() if radial_center is not None else None,
            quality_target=quality_target.detach(),
            quality_mix=quality_mix,
            quality_base_weight=quality_base_weight,
            rho_max=config.radial_calibration_rho_max,
            rho_band_width=config.radial_calibration_band_width,
            use_hyperbolic_radius=True,
        )
        zn_reg_loss = zn_reg_loss + config.w_radial_calibration * loss_radcal
        metrics["radial_cal"] = loss_radcal.item()
    else:
        metrics["local_radius_mean"] = 0.0

    if config.w_confidence_calibration > 0:
        loss_confcal = compute_confidence_calibration_loss(
            router_reg_weights,
            quality_target.detach(),
            config.num_charts,
        )
        base_loss = base_loss + config.w_confidence_calibration * loss_confcal
        metrics["confidence_calibration"] = loss_confcal.item()

    if config.w_v_tangent_barrier > 0 and v_raw is not None:
        loss_v_tangent = compute_v_tangent_barrier_loss(
            v_raw,
            target_radius=config.v_tangent_barrier_radius,
        )
        base_loss = base_loss + config.w_v_tangent_barrier * loss_v_tangent
        metrics["v_tangent_barrier"] = loss_v_tangent.item()

    # Codebook spread
    if config.w_codebook_spread > 0 and hasattr(atlas_encoder, "codebook"):
        codebook = atlas_encoder.codebook
        loss_spread = compute_codebook_spread_loss(
            codebook,
            margin=config.w_codebook_spread_margin,
        )
        base_loss = base_loss + config.w_codebook_spread * loss_spread
        metrics["codebook_spread"] = loss_spread.item()

    if config.w_codebook_center > 0 and hasattr(atlas_encoder, "codebook"):
        codebook = atlas_encoder.codebook
        loss_center = compute_codebook_centering_loss(codebook)
        base_loss = base_loss + config.w_codebook_center * loss_center
        metrics["codebook_center"] = loss_center.item()

    if hasattr(atlas_encoder, "chart_centers"):
        chart_centers = atlas_encoder.chart_centers
        if config.w_chart_center_mean > 0:
            loss_chart_center_mean = compute_chart_center_mean_loss(chart_centers)
            base_loss = base_loss + config.w_chart_center_mean * loss_chart_center_mean
            metrics["chart_center_mean"] = loss_chart_center_mean.item()
        if config.w_chart_center_radius > 0:
            loss_chart_center_radius = compute_chart_center_radius_loss(
                chart_centers,
                radius_max=config.chart_center_radius_max,
            )
            base_loss = base_loss + config.w_chart_center_radius * loss_chart_center_radius
            metrics["chart_center_radius"] = loss_chart_center_radius.item()
        if config.w_chart_center_sep > 0:
            loss_chart_center_sep = compute_chart_center_separation_loss(
                chart_centers,
                margin=config.chart_center_sep_margin,
            )
            base_loss = base_loss + config.w_chart_center_sep * loss_chart_center_sep
            metrics["chart_center_sep"] = loss_chart_center_sep.item()

    # Code usage is measured with a straight-through code assignment so the
    # forward value reflects actual selected codes while gradients still flow.
    if config.w_code_collapse > 0 and hasattr(atlas_encoder, "codebook"):
        if v_local is None:
            msg = "Phase 1 code-usage loss requires the encoder's chart-local latent."
            raise RuntimeError(msg)
        codebook = atlas_encoder.codebook
        loss_code_usage, code_usage_metrics = compute_code_usage_band_loss(
            v_local,
            codebook,
            usage_router_weights,
            hard_code_indices=indices_stack,
            h_low=config.code_usage_entropy_low,
            h_high=config.code_usage_entropy_high,
            temperature=config.w_code_collapse_temperature,
        )
        base_loss = base_loss + config.w_code_collapse * loss_code_usage
        metrics["code_usage"] = loss_code_usage.item()
        metrics.update(code_usage_metrics)

    # Window loss
    if config.w_window > 0:
        loss_window, _ = compute_window_loss(
            router_reg_weights,
            eps_ground=config.w_window_eps_ground,
        )
        base_loss = base_loss + config.w_window * loss_window
        metrics["window"] = loss_window.item()
    else:
        metrics["window"] = 0.0

    # Encoder-decoder routing consistency
    if config.w_consistency > 0:
        eps = 1e-6
        kl = (
            (
                enc_router_weights
                * torch.log((enc_router_weights + eps) / (dec_router_weights + eps))
            )
            .sum(dim=-1)
            .mean()
        )
        base_loss = base_loss + config.w_consistency * kl
        metrics["consistency"] = kl.item()
    else:
        metrics["consistency"] = 0.0

    metrics.setdefault("uniformity", 0.0)
    metrics.setdefault("radial_cal", 0.0)
    metrics.setdefault("confidence_calibration", 0.0)
    metrics.setdefault("router_margin", 0.0)
    metrics.setdefault("hard_routing_nll", 0.0)
    metrics.setdefault("v_tangent_barrier", 0.0)
    metrics.setdefault("codebook_spread", 0.0)
    metrics.setdefault("codebook_center", 0.0)
    metrics.setdefault("chart_center_mean", 0.0)
    metrics.setdefault("chart_center_radius", 0.0)
    metrics.setdefault("chart_center_sep", 0.0)
    metrics.setdefault("chart_ot", 0.0)
    metrics.setdefault("ot_target_top1_mean", 0.0)
    metrics.setdefault("ot_plan_col_l1", 0.0)
    metrics.setdefault("ot_plan_row_l1", 0.0)
    metrics.setdefault("code_usage", 0.0)
    metrics.setdefault("H_code_usage", 0.0)
    metrics.setdefault("code_usage_perplexity", 1.0)
    metrics.setdefault("active_code_charts", 0.0)
    metrics.setdefault("recon_quality_mean", 0.0)
    metrics.setdefault("vq_quality_mean", 0.0)
    metrics.setdefault("combined_quality_mean", 0.0)
    metrics.setdefault("routing_confidence_mean", 0.0)
    metrics.setdefault("radial_target_mean", 0.0)
    metrics.setdefault("local_radius_mean", 0.0)

    total = base_loss + zn_reg_loss
    metrics["total"] = total.item()
    return base_loss, zn_reg_loss, metrics


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    # Extracted from vla/losses.py
    "_deterministic_st_router_weights",
    "as_tangent",
    "combine_quality_targets",
    # Chart center losses
    "compute_chart_center_mean_loss",
    "compute_chart_center_radius_loss",
    "compute_chart_center_separation_loss",
    # Chart / code balancing
    "compute_chart_usage_band_loss",
    "compute_code_usage_band_loss",
    "compute_codebook_centering_loss",
    "compute_codebook_spread_loss",
    # Routing losses
    "compute_confidence_calibration_loss",
    "compute_error_quality_targets",
    "compute_hard_routing_nll",
    # Geometry losses
    "compute_hyperbolic_uniformity_loss",
    # Jump
    "compute_jump_consistency_loss",
    # Phase 1 assembly
    "compute_phase1_loss",
    "compute_radial_calibration_loss",
    # Quality targets
    "compute_rank_quality_targets",
    "compute_router_information_metrics",
    "compute_router_margin_loss",
    "compute_router_score_metrics",
    "compute_router_sharpness_metrics",
    "compute_routing_confidence",
    "compute_routing_entropy",
    "compute_sinkhorn_balanced_chart_loss",
    "compute_v_tangent_barrier_loss",
    "compute_window_loss",
    "get_jump_weight_schedule",
    "mix_quality_targets",
    "orthogonality_loss",
    "project_to_ball",
]
