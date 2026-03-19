"""Critic / value-field losses for geometric Dreamer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from fragile.layers.gauge import poincare_log_map


if TYPE_CHECKING:
    from fragile.rl.config import DreamerConfig


# ---------------------------------------------------------------------------
# Shared helpers (small, duplicated where needed across loss modules)
# ---------------------------------------------------------------------------


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one.

    Args:
        values: Tensor of arbitrary shape containing the values to average.
        mask: Tensor broadcastable to ``values`` with 1.0 for valid entries
            and 0.0 for entries to ignore.

    Returns:
        A scalar tensor with the masked mean. Returns zero when the mask
        sum is zero (denominator is clamped to 1).
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standard deviation over entries where ``mask`` is one.

    Args:
        values: Tensor of arbitrary shape containing the values.
        mask: Tensor broadcastable to ``values`` with 1.0 for valid entries
            and 0.0 for entries to ignore.

    Returns:
        A scalar tensor with the population standard deviation (biased) of
        the masked entries. Returns a zero scalar when no entries are valid.
    """
    valid = values.reshape(-1)[mask.reshape(-1).bool()]
    if valid.numel() == 0:
        return values.new_zeros(())
    return valid.std(unbiased=False)


def _masked_corrcoef(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson correlation over masked entries with a safe zero fallback.

    Args:
        left: Tensor of arbitrary shape containing the first variable.
        right: Tensor of the same shape as ``left`` containing the second
            variable.
        mask: Tensor broadcastable to ``left`` and ``right`` with 1.0 for
            valid entries and 0.0 for entries to ignore.

    Returns:
        A scalar tensor with the Pearson correlation coefficient between
        the masked elements of ``left`` and ``right``. Returns zero when
        fewer than two valid entries exist or when either variable has
        near-zero variance.
    """
    valid = mask.reshape(-1).bool()
    left_valid = left.reshape(-1)[valid]
    right_valid = right.reshape(-1)[valid]
    if left_valid.numel() < 2:
        return left.new_zeros(())
    left_centered = left_valid - left_valid.mean()
    right_centered = right_valid - right_valid.mean()
    denom = left_centered.pow(2).mean().sqrt() * right_centered.pow(2).mean().sqrt()
    if float(denom.detach()) <= 1e-8:
        return left.new_zeros(())
    return (left_centered * right_centered).mean() / denom


def _masked_sign_agreement(
    left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Fraction of masked entries with matching signs.

    Args:
        left: Tensor of arbitrary shape containing the first variable.
        right: Tensor of the same shape as ``left`` containing the second
            variable.
        mask: Tensor broadcastable to ``left`` and ``right`` with 1.0 for
            valid entries and 0.0 for entries to ignore.

    Returns:
        A scalar tensor with the fraction of valid entries where
        ``sign(left) == sign(right)``. Returns zero when no valid entries
        exist.
    """
    valid = mask.reshape(-1).bool()
    left_valid = left.reshape(-1)[valid]
    right_valid = right.reshape(-1)[valid]
    if left_valid.numel() == 0:
        return left.new_zeros(())
    return left_valid.sign().eq(right_valid.sign()).to(left_valid.dtype).mean()


def _masked_support_fraction(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> torch.Tensor:
    """Fraction of masked entries with magnitude above ``threshold``.

    Args:
        values: Tensor of arbitrary shape containing the values to check.
        mask: Tensor broadcastable to ``values`` with 1.0 for valid entries
            and 0.0 for entries to ignore.
        threshold: Minimum absolute value for an entry to be counted as
            part of the support.

    Returns:
        A scalar tensor with the fraction of valid entries whose absolute
        value is at least ``threshold``. Returns zero when no valid entries
        exist.
    """
    valid = mask.reshape(-1).bool()
    value_valid = values.reshape(-1)[valid]
    if value_valid.numel() == 0:
        return values.new_zeros(())
    return (value_valid.abs() >= float(threshold)).to(value_valid.dtype).mean()


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    """Quantile over masked values with a safe fallback for empty masks.

    Args:
        values: Tensor of arbitrary shape containing the values.
        mask: Tensor broadcastable to ``values`` with 1.0 for valid entries
            and 0.0 for entries to ignore.
        quantile: Quantile level in ``[0, 1]`` to compute. Values outside
            this range are clipped.

    Returns:
        A scalar tensor with the requested quantile of the masked entries.
        Returns zero when no valid entries exist, or the single value when
        exactly one entry is valid.
    """
    valid = values[mask.reshape(-1).bool()]
    if valid.numel() == 0:
        return values.new_zeros(())
    if valid.numel() == 1:
        return valid[0]
    q = float(np.clip(quantile, 0.0, 1.0))
    return torch.quantile(valid, q)


def _metric_inverse_scale(metric: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Return the inverse conformal metric scale ``lambda(z)^{-2}``.

    Args:
        metric: A conformal metric module that exposes a
            ``conformal_factor(z)`` method and an optional ``epsilon``
            attribute.
        z: Latent-space coordinates of shape ``(..., D)`` where ``D`` is
            the latent dimension.

    Returns:
        A tensor of shape ``(..., 1)`` containing ``1 / (lambda(z)^2 + eps)``
        at each point.
    """
    cf = metric.conformal_factor(z)
    epsilon = getattr(metric, "epsilon", 1e-8)
    return 1.0 / (cf.pow(2) + epsilon)


def _metric_covector_norm_sq(
    metric: nn.Module,
    z: torch.Tensor,
    covector: torch.Tensor,
) -> torch.Tensor:
    """Covector norm squared under the inverse conformal metric.

    Computes ``g^{ij}(z) * covector_i * covector_j`` where the inverse
    metric is ``lambda(z)^{-2} * delta^{ij}``.

    Args:
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Latent-space coordinates of shape ``(N, D)``.
        covector: Covector field of shape ``(N, D)``.

    Returns:
        A tensor of shape ``(N,)`` with the squared covector norm at each
        point.
    """
    return _metric_inverse_scale(metric, z).squeeze(-1) * covector.pow(2).sum(dim=-1)


def _metric_covector_pair(
    metric: nn.Module,
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Pair two covectors under the inverse conformal metric.

    Computes ``g^{ij}(z) * left_i * right_j``.

    Args:
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Latent-space coordinates of shape ``(N, D)``.
        left: First covector field of shape ``(N, D)``.
        right: Second covector field of shape ``(N, D)``.

    Returns:
        A tensor of shape ``(N,)`` with the metric pairing at each point.
    """
    return _metric_inverse_scale(metric, z).squeeze(-1) * (left * right).sum(dim=-1)


def _metric_vector_norm_sq(
    metric: nn.Module,
    z: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Vector norm squared under the conformal metric.

    Computes ``g_{ij}(z) * vector^i * vector^j`` where the metric is
    ``lambda(z)^2 * delta_{ij}``.

    Args:
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Latent-space coordinates of shape ``(N, D)``.
        vector: Tangent vector field of shape ``(N, D)``.

    Returns:
        A tensor of shape ``(N,)`` with the squared vector norm at each
        point.
    """
    return vector.pow(2).sum(dim=-1) / _metric_inverse_scale(metric, z).squeeze(-1).clamp_min(1e-8)


def _exact_increment_observability_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metric_prefix: str,
    support_threshold: float,
) -> dict[str, float]:
    """Detailed observability metrics for exact-increment supervision.

    Computes a set of diagnostic statistics (standard deviations, sign
    agreement, correlation, support fraction, positive fraction) between
    predicted and target exact increments.

    Args:
        pred: Predicted exact-increment values (flat or batched tensor).
        target: Target exact-increment values with the same shape as
            ``pred``.
        mask: Valid-entry mask broadcastable to ``pred`` and ``target``.
        metric_prefix: String prefix prepended to every key in the
            returned dictionary (e.g. ``"critic"``).
        support_threshold: Minimum absolute value for the support
            fraction metric.

    Returns:
        A dictionary mapping metric names to float values. Keys include
        ``{metric_prefix}/exact_increment_pred_std``,
        ``{metric_prefix}/exact_increment_target_std``,
        ``{metric_prefix}/exact_increment_sign_acc``,
        ``{metric_prefix}/exact_increment_corr``,
        ``{metric_prefix}/exact_increment_support_frac``, and
        ``{metric_prefix}/exact_increment_positive_frac``.
    """
    positive_mask = (
        (target.reshape(-1) >= float(support_threshold)).to(target.dtype).reshape_as(target)
    )
    return {
        f"{metric_prefix}/exact_increment_pred_std": float(_masked_std(pred, mask).detach()),
        f"{metric_prefix}/exact_increment_target_std": float(_masked_std(target, mask).detach()),
        f"{metric_prefix}/exact_increment_sign_acc": float(
            _masked_sign_agreement(pred, target, mask).detach(),
        ),
        f"{metric_prefix}/exact_increment_corr": float(
            _masked_corrcoef(pred, target, mask).detach()
        ),
        f"{metric_prefix}/exact_increment_support_frac": float(
            _masked_support_fraction(target, mask, threshold=support_threshold).detach(),
        ),
        f"{metric_prefix}/exact_increment_positive_frac": float(
            _masked_mean(positive_mask, mask).detach(),
        ),
    }


def _linear_warmup_scale(epoch: int, warmup_epochs: int) -> float:
    """Linear warmup from zero to one over ``warmup_epochs`` epochs.

    Args:
        epoch: Current training epoch (zero-indexed).
        warmup_epochs: Number of epochs over which the scale ramps from
            zero to one. If zero or negative, returns 1.0 immediately.

    Returns:
        A float in ``[0.0, 1.0]`` representing the warmup multiplier for
        the current epoch.
    """
    if warmup_epochs <= 0:
        return 1.0
    return min(max((epoch + 1) / warmup_epochs, 0.0), 1.0)


def _target_normalization_scale(
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    quantile: float,
    min_scale: float,
    template: torch.Tensor,
) -> torch.Tensor:
    """Estimate a robust target scale for normalized exact-field training.

    Concatenates the absolute values of all target tensors, computes the
    requested quantile over the valid entries, and clamps the result to a
    minimum value to avoid division by near-zero scales.

    Args:
        targets: List of target tensors whose absolute values form the
            scale estimate.
        masks: List of masks (one per target) indicating valid entries.
        quantile: Quantile level in ``[0, 1]`` used to estimate the scale.
        min_scale: Minimum allowed scale value (clamped from below).
        template: A tensor used to infer device and dtype when ``targets``
            is empty.

    Returns:
        A scalar tensor with the estimated normalization scale, at least
        ``max(min_scale, 1e-8)``.
    """
    if not targets:
        return template.new_tensor(max(float(min_scale), 1e-8))
    stacked_targets = torch.cat([target.abs().reshape(-1) for target in targets])
    stacked_masks = torch.cat([mask.reshape(-1) for mask in masks])
    return _masked_quantile(stacked_targets, stacked_masks, float(quantile)).clamp_min(
        max(float(min_scale), 1e-8),
    )


# ---------------------------------------------------------------------------
# Multi-step target building
# ---------------------------------------------------------------------------


def _discounted_return_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted replay return-to-go until the first terminal step.

    Iterates backwards through the time dimension, accumulating
    ``r_t + gamma * G_{t+1}`` and resetting at terminal states.

    Args:
        rewards: Reward tensor of shape ``(B, T)`` where ``B`` is the
            batch size and ``T`` is the sequence length.
        dones: Terminal-state indicator tensor of shape ``(B, T)`` with
            1.0 at episode boundaries and 0.0 otherwise.
        gamma: Discount factor in ``[0, 1]``.

    Returns:
        A tensor of shape ``(B, T)`` with the discounted return-to-go at
        each time step.
    """
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running * (1.0 - dones[:, t])
        returns[:, t] = running
    return returns


def _discounted_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted cumulative reward over a fixed imagined horizon.

    Computes ``sum_t gamma^t * rewards[:, t]`` for ``t`` in
    ``[0, horizon)``.

    Args:
        rewards: Reward tensor of shape ``(B, H)`` where ``H`` is the
            imagined horizon length.
        gamma: Discount factor in ``[0, 1]``.

    Returns:
        A tensor of shape ``(B,)`` with the discounted cumulative reward
        for each batch element.
    """
    horizon = rewards.shape[1]
    if horizon == 0:
        return rewards.sum(dim=1)
    exponents = torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discounts = torch.pow(rewards.new_full((), gamma), exponents).unsqueeze(0)
    return (rewards * discounts).sum(dim=1)


def _multistep_horizon_ladder(max_horizon: int) -> list[int]:
    """Return a logarithmic horizon ladder with the final horizon included.

    Builds a sorted list of unique horizons starting at 1 and doubling
    until ``max_horizon`` is reached, always including ``max_horizon``
    itself.

    Args:
        max_horizon: Maximum horizon step. Clamped to at least 1.

    Returns:
        A sorted list of unique integer horizons forming a logarithmic
        ladder from 1 up to and including ``max_horizon``.
    """
    max_horizon = max(int(max_horizon), 1)
    horizons = [1]
    step = 1
    while step < max_horizon:
        step *= 2
        horizons.append(min(step, max_horizon))
    if horizons[-1] != max_horizon:
        horizons.append(max_horizon)
    return sorted(set(horizons))


def _multistep_discounted_targets(
    one_step_targets: torch.Tensor,
    continuation: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    horizon: int,
) -> list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Build discounted k-step targets and masks from one-step conservative targets.

    For each selected horizon ``k`` on the logarithmic ladder, accumulates
    discounted one-step targets over ``k`` steps while tracking
    continuation products and validity masks.

    Args:
        one_step_targets: One-step conservative target tensor of shape
            ``(B, T)``.
        continuation: Continuation (1 - done) tensor of shape ``(B, T)``
            used to gate multi-step accumulation across episode boundaries.
        valid_mask: Binary validity mask of shape ``(B, T)`` indicating
            which time steps carry usable data.
        gamma: Discount factor in ``[0, 1]``.
        horizon: Maximum number of steps to look ahead. Clamped to
            ``[1, T]``.

    Returns:
        A list of tuples ``(k, target_k, continuation_k, valid_k)`` where:
            k: The step horizon (int).
            target_k: Discounted k-step target tensor of shape
                ``(B, T - k + 1)``.
            continuation_k: Product of continuation flags over the k
                steps, shape ``(B, T - k + 1)``.
            valid_k: Product of validity masks over the k steps, shape
                ``(B, T - k + 1)``.
    """
    _, T = one_step_targets.shape
    max_horizon = max(1, min(int(horizon), T))
    selected_steps = set(_multistep_horizon_ladder(max_horizon))
    discounted_targets = one_step_targets
    continuation_prod = continuation
    valid_prod = valid_mask
    gamma_power = float(gamma)
    outputs: list[tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    for step in range(1, max_horizon + 1):
        seq_len = T - step + 1
        if step in selected_steps:
            outputs.append(
                (
                    step,
                    discounted_targets[:, :seq_len],
                    continuation_prod[:, :seq_len],
                    valid_prod[:, :seq_len],
                ),
            )
        if step == max_horizon:
            break
        discounted_targets = (
            discounted_targets[:, :-1]
            + gamma_power * continuation_prod[:, :-1] * one_step_targets[:, step:]
        )
        continuation_prod = continuation_prod[:, :-1] * continuation[:, step:]
        valid_prod = valid_prod[:, :-1] * valid_mask[:, step:]
        gamma_power *= float(gamma)
    return outputs


# ---------------------------------------------------------------------------
# Gradient observability
# ---------------------------------------------------------------------------


def _weighted_loss_grad_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> float:
    """Return the L2 norm of gradients induced by ``loss`` on ``parameters``.

    Uses ``torch.autograd.grad`` with ``retain_graph=True`` so the
    computation graph remains available for subsequent backward passes.

    Args:
        loss: Scalar loss tensor whose gradients are to be measured.
        parameters: List of model parameters with respect to which
            gradients are computed.

    Returns:
        A float with the L2 norm of the concatenated gradient vectors.
        Returns 0.0 if ``loss`` does not require grad, ``parameters`` is
        empty, or all computed gradients are ``None``.
    """
    if not torch.is_tensor(loss) or not loss.requires_grad or not parameters:
        return 0.0
    grads = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    norms = [grad.detach().norm() for grad in grads if grad is not None]
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), p=2).detach())


# ---------------------------------------------------------------------------
# Critic losses
# ---------------------------------------------------------------------------


def _critic_stage_scales(config: DreamerConfig, epoch: int) -> dict[str, float]:
    """Return epoch-dependent scales for critic formation vs shaping terms.

    Each loss component has its own warmup schedule read from
    ``config``; this function evaluates all of them at the given epoch.

    Args:
        config: Dreamer configuration object containing per-component
            warmup epoch counts (e.g.
            ``screened_poisson_warmup_epochs``,
            ``critic_covector_warmup_epochs``, etc.).
        epoch: Current training epoch (zero-indexed).

    Returns:
        A dictionary mapping component names (``"exact_increment"``,
        ``"poisson"``, ``"covector"``, ``"stiffness"``,
        ``"macro_pullback"``, ``"on_policy"``) to their warmup scale
        floats in ``[0.0, 1.0]``.
    """
    return {
        "exact_increment": 1.0,
        "poisson": _linear_warmup_scale(epoch, int(config.screened_poisson_warmup_epochs)),
        "covector": _linear_warmup_scale(epoch, int(config.critic_covector_warmup_epochs)),
        "stiffness": _linear_warmup_scale(epoch, int(config.critic_stiffness_warmup_epochs)),
        "macro_pullback": _linear_warmup_scale(
            epoch, int(config.critic_macro_pullback_warmup_epochs)
        ),
        "on_policy": _linear_warmup_scale(epoch, int(config.critic_on_policy_warmup_epochs)),
    }


def _critic_grad_observability_metrics(
    *,
    config: DreamerConfig,
    epoch: int,
    update_idx: int,
    parameters: list[torch.nn.Parameter],
    value_loss: torch.Tensor,
    exact_loss: torch.Tensor,
    poisson_loss: torch.Tensor,
    covector_loss: torch.Tensor,
    stiffness_loss: torch.Tensor,
    macro_pullback_loss: torch.Tensor,
    on_policy_loss: torch.Tensor,
) -> dict[str, float]:
    """Measure which critic losses dominate the value-field gradient.

    Computes per-component gradient norms at a configurable frequency
    (``config.critic_grad_metrics_every``). On non-reporting steps,
    returns all-zero metrics to avoid the cost of extra backward passes.

    Args:
        config: Dreamer configuration object. Controls the reporting
            frequency via ``critic_grad_metrics_every``.
        epoch: Current training epoch (unused but accepted for a uniform
            interface).
        update_idx: Current optimisation step index within the epoch.
            Metrics are only computed when
            ``update_idx % config.critic_grad_metrics_every == 0``.
        parameters: List of critic parameters to differentiate against.
        value_loss: Combined (total) critic loss scalar.
        exact_loss: Exact-increment loss scalar.
        poisson_loss: Screened-Poisson loss scalar.
        covector_loss: Covector-alignment loss scalar.
        stiffness_loss: Stiffness loss scalar.
        macro_pullback_loss: Macro-covector pullback loss scalar.
        on_policy_loss: On-policy loss scalar.

    Returns:
        A dictionary mapping gradient-norm metric names (e.g.
        ``"critic/grad_value"``, ``"critic/grad_exact_increment"``, etc.)
        to float values.
    """
    zero_metrics = {
        "critic/grad_value": 0.0,
        "critic/grad_exact_increment": 0.0,
        "critic/grad_poisson": 0.0,
        "critic/grad_covector_align": 0.0,
        "critic/grad_stiffness": 0.0,
        "critic/grad_macro_pullback": 0.0,
        "critic/grad_on_policy": 0.0,
    }
    every = int(config.critic_grad_metrics_every)
    if every <= 0 or update_idx % every != 0 or not parameters:
        return zero_metrics
    del epoch
    return {
        "critic/grad_value": _weighted_loss_grad_norm(value_loss, parameters),
        "critic/grad_exact_increment": _weighted_loss_grad_norm(exact_loss, parameters),
        "critic/grad_poisson": _weighted_loss_grad_norm(poisson_loss, parameters),
        "critic/grad_covector_align": _weighted_loss_grad_norm(covector_loss, parameters),
        "critic/grad_stiffness": _weighted_loss_grad_norm(stiffness_loss, parameters),
        "critic/grad_macro_pullback": _weighted_loss_grad_norm(macro_pullback_loss, parameters),
        "critic/grad_on_policy": _weighted_loss_grad_norm(on_policy_loss, parameters),
    }


def _critic_stiffness_loss(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    z: torch.Tensor,
    exact_covector: torch.Tensor,
    replay_valid: torch.Tensor,
    stiffness_scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep the conservative critic field steep enough to drive control.

    Penalises points where the covector norm falls below a target
    stiffness scale, encouraging the value field to maintain sufficient
    gradient magnitude for effective policy optimisation.

    Args:
        config: Dreamer configuration object. Uses
            ``critic_stiffness_min`` as the minimum allowed stiffness
            target.
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Latent-space coordinates of shape ``(N, D)``.
        exact_covector: Exact covector field of shape ``(N, D)``.
        replay_valid: Binary mask of shape ``(N,)`` or ``(N, 1)``
            indicating valid replay entries.
        stiffness_scale: Optional externally supplied stiffness target.
            When ``None``, the minimum from ``config`` is used.

    Returns:
        A tuple of (loss, metrics) where:
            loss: Scalar stiffness loss tensor.
            metrics: Dictionary of diagnostic floats including the loss
                value, covector norm mean, stiffness target, deficit
                mean, and a binary certified indicator.
    """
    exact_covector_norm = _metric_covector_norm_sq(metric, z, exact_covector).sqrt()
    exact_covector_norm_mean = _masked_mean(exact_covector_norm, replay_valid.reshape(-1))
    if stiffness_scale is None:
        stiffness_scale_t = exact_covector_norm_mean.new_tensor(
            max(float(config.critic_stiffness_min), 1e-8),
        )
    else:
        stiffness_scale_t = torch.as_tensor(
            stiffness_scale,
            device=exact_covector_norm_mean.device,
            dtype=exact_covector_norm_mean.dtype,
        ).clamp_min(max(float(config.critic_stiffness_min), 1e-8))
    stiffness_deficit = (stiffness_scale_t - exact_covector_norm).clamp(
        min=0.0
    ) / stiffness_scale_t
    L_critic_stiffness = _masked_mean(stiffness_deficit.pow(2), replay_valid.reshape(-1))
    metrics = {
        "critic/L_stiffness": float(L_critic_stiffness.detach()),
        "critic/exact_covector_norm_mean": float(exact_covector_norm_mean.detach()),
        "critic/stiffness_target": float(stiffness_scale_t.detach()),
        "critic/stiffness_deficit_mean": float(
            _masked_mean(stiffness_deficit, replay_valid.reshape(-1)).detach(),
        ),
        "critic/stiffness_certified": (
            1.0
            if float(exact_covector_norm_mean.detach()) >= float(stiffness_scale_t.detach())
            else 0.0
        ),
    }
    return L_critic_stiffness, metrics


def _macro_covector_pullback_loss(
    *,
    metric: nn.Module,
    z: torch.Tensor,
    macro_covector: torch.Tensor,
    exact_covector: torch.Tensor,
    replay_valid: torch.Tensor,
    metric_prefix: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align the continuous exact field with the lifted symbolic control field.

    Measures the metric-norm of the difference between the macro
    (symbolic) covector and the exact (learned) covector, normalised by a
    robust target scale.

    Args:
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Latent-space coordinates of shape ``(N, D)``.
        macro_covector: Lifted symbolic covector field of shape
            ``(N, D)``.
        exact_covector: Learned exact covector field of shape ``(N, D)``.
        replay_valid: Binary mask of shape ``(N,)`` or ``(N, 1)``
            indicating valid replay entries.
        metric_prefix: String prefix for metric dictionary keys.

    Returns:
        A tuple of (loss, metrics) where:
            loss: Scalar pullback alignment loss tensor.
            metrics: Dictionary of diagnostic floats including the loss
                value, absolute error, covector norm means, and the
                target normalisation scale.
    """
    valid_flat = replay_valid.reshape(-1)
    macro_norm = _metric_covector_norm_sq(metric, z, macro_covector).sqrt()
    exact_norm = _metric_covector_norm_sq(metric, z, exact_covector).sqrt()
    diff_norm = _metric_covector_norm_sq(
        metric,
        z,
        macro_covector - exact_covector,
    ).sqrt()
    target_scale = _target_normalization_scale(
        [macro_norm.detach(), exact_norm.detach()],
        [valid_flat, valid_flat],
        quantile=0.75,
        min_scale=1e-3,
        template=macro_norm,
    )
    loss = _masked_mean((diff_norm / target_scale).pow(2), valid_flat)
    metrics = {
        f"{metric_prefix}/L_covector_pullback": float(loss.detach()),
        f"{metric_prefix}/covector_pullback_abs_err": float(
            _masked_mean(diff_norm, valid_flat).detach(),
        ),
        f"{metric_prefix}/covector_norm_mean": float(
            _masked_mean(macro_norm, valid_flat).detach()
        ),
        f"{metric_prefix}/covector_target_norm_mean": float(
            _masked_mean(exact_norm, valid_flat).detach(),
        ),
        f"{metric_prefix}/covector_target_scale": float(target_scale.detach()),
    }
    return loss, metrics


def _critic_covector_alignment_loss(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    z: torch.Tensor,
    z_next: torch.Tensor,
    value_current: torch.Tensor,
    exact_covector: torch.Tensor,
    reward_conservative_target: torch.Tensor,
    continuation: torch.Tensor,
    gamma: float,
    replay_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Align ``dV`` with the discounted local exact increment along replay geodesics.

    Computes the predicted reward from a first-order Taylor expansion of
    the value function along the Poincare log-map displacement, then
    penalises deviation from the conservative reward target. Also derives
    an adaptive stiffness scale from the ratio of target reward magnitude
    to displacement norm.

    Args:
        config: Dreamer configuration object. Uses
            ``critic_target_scale_quantile``,
            ``critic_target_scale_min``,
            ``critic_stiffness_target_scale``,
            ``critic_stiffness_quantile``,
            ``critic_stiffness_min``, and
            ``critic_stiffness_target_max``.
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z: Current latent-space coordinates of shape ``(N, D)``.
        z_next: Next-step latent-space coordinates of shape ``(N, D)``.
        value_current: Current value predictions of shape ``(N,)`` or
            ``(N, 1)``.
        exact_covector: Exact covector field of shape ``(N, D)``.
        reward_conservative_target: Conservative reward targets of shape
            ``(N,)`` or ``(N, 1)``.
        continuation: Continuation flags (1 - done) of shape ``(N,)`` or
            ``(N, 1)``.
        gamma: Discount factor in ``[0, 1]``.
        replay_valid: Binary mask of shape ``(N,)`` or ``(N, 1)``
            indicating valid replay entries.

    Returns:
        A tuple of (loss, stiffness_scale, metrics) where:
            loss: Scalar covector-alignment loss tensor.
            stiffness_scale: Detached adaptive stiffness scale tensor
                derived from reward-to-displacement ratios.
            metrics: Dictionary of diagnostic floats including alignment
                error, predicted/target reward means, displacement norm,
                target scale, and the adaptive stiffness target.
    """
    displacement = poincare_log_map(z.detach(), z_next.detach())
    local_value_delta = (exact_covector * displacement).sum(dim=-1)
    continuation_scale = float(gamma) * continuation.reshape(-1)
    value_current_flat = value_current.reshape(-1)
    predicted_reward = value_current_flat - continuation_scale * (
        value_current_flat + local_value_delta
    )
    target_reward = reward_conservative_target.reshape(-1)
    replay_valid_flat = replay_valid.reshape(-1)
    target_scale = _target_normalization_scale(
        [target_reward],
        [replay_valid_flat],
        quantile=config.critic_target_scale_quantile,
        min_scale=config.critic_target_scale_min,
        template=predicted_reward,
    )
    L_critic_covector_align = _masked_mean(
        ((predicted_reward - target_reward) / target_scale).pow(2),
        replay_valid_flat,
    )
    displacement_norm = _metric_vector_norm_sq(metric, z.detach(), displacement).sqrt()
    reward_scale = target_reward.detach().abs() / (displacement_norm.detach() + 1e-8)
    stiffness_scale = (
        float(config.critic_stiffness_target_scale)
        * _masked_quantile(
            reward_scale, replay_valid_flat, float(config.critic_stiffness_quantile)
        )
    ).clamp_min(max(float(config.critic_stiffness_min), 1e-8))
    stiffness_max = float(config.critic_stiffness_target_max)
    if stiffness_max > 0.0:
        stiffness_scale = stiffness_scale.clamp(max=stiffness_max)
    metrics = {
        "critic/L_covector_align": float(L_critic_covector_align.detach()),
        "critic/covector_align_abs_err": float(
            _masked_mean((predicted_reward - target_reward).abs(), replay_valid_flat).detach(),
        ),
        "critic/covector_predicted_reward_mean": float(
            _masked_mean(predicted_reward, replay_valid_flat).detach(),
        ),
        "critic/covector_target_reward_mean": float(
            _masked_mean(target_reward, replay_valid_flat).detach(),
        ),
        "critic/displacement_norm_mean": float(
            _masked_mean(displacement_norm, replay_valid_flat).detach(),
        ),
        "critic/covector_target_scale": float(target_scale.detach()),
        "critic/stiffness_target_adaptive": float(stiffness_scale.detach()),
    }
    return L_critic_covector_align, stiffness_scale.detach(), metrics


def _critic_exact_increment_loss(
    *,
    reward_conservative_pred: torch.Tensor,
    reward_conservative_target: torch.Tensor,
    replay_valid: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise the exact discounted value increment ``V_t - gamma V_{t+1}`` directly.

    Computes a scale-normalised MSE between the predicted and target
    conservative reward increments, plus detailed observability metrics.

    Args:
        reward_conservative_pred: Predicted conservative reward of shape
            ``(N,)`` or ``(N, 1)``.
        reward_conservative_target: Target conservative reward of shape
            ``(N,)`` or ``(N, 1)``.
        replay_valid: Binary mask of shape ``(N,)`` or ``(N, 1)``
            indicating valid replay entries.

    Returns:
        A tuple of (loss, metrics) where:
            loss: Scalar normalised MSE loss tensor.
            metrics: Dictionary of diagnostic floats including the loss
                value, absolute error, predicted/target means, target
                scale, and observability statistics (std, sign accuracy,
                correlation, support fraction, positive fraction).
    """
    pred = reward_conservative_pred.reshape(-1)
    target = reward_conservative_target.reshape(-1)
    replay_valid_flat = replay_valid.reshape(-1)
    target_scale = _target_normalization_scale(
        [target],
        [replay_valid_flat],
        quantile=0.75,
        min_scale=1e-3,
        template=pred,
    )
    loss = _masked_mean(((pred - target) / target_scale).pow(2), replay_valid_flat)
    support_threshold = max(0.1 * float(target_scale.detach()), 1e-6)
    metrics = {
        "critic/L_exact_increment": float(loss.detach()),
        "critic/exact_increment_abs_err": float(
            _masked_mean((pred - target).abs(), replay_valid_flat).detach(),
        ),
        "critic/exact_increment_pred_mean": float(_masked_mean(pred, replay_valid_flat).detach()),
        "critic/exact_increment_target_mean": float(
            _masked_mean(target, replay_valid_flat).detach(),
        ),
        "critic/exact_increment_target_scale": float(target_scale.detach()),
    }
    metrics.update(
        _exact_increment_observability_metrics(
            pred=pred,
            target=target,
            mask=replay_valid_flat,
            metric_prefix="critic",
            support_threshold=support_threshold,
        ),
    )
    return loss, metrics


def _multistep_exact_increment_loss(
    *,
    value_seq: torch.Tensor,
    reward_conservative_targets: torch.Tensor,
    continuation: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    horizon: int,
    decay: float,
    metric_prefix: str,
    target_scale_quantile: float = 0.75,
    target_scale_min: float = 1e-3,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise exact discounted value increments over multiple horizons.

    For each horizon ``k`` on the logarithmic ladder, computes
    ``V_t - gamma^k * continuation_k * V_{t+k}`` and penalises its
    deviation from the corresponding discounted target. Losses at
    different horizons are combined with exponentially decaying weights.

    Args:
        value_seq: Value predictions of shape ``(B, T+1)`` covering all
            time steps including the final bootstrap step.
        reward_conservative_targets: One-step conservative reward targets
            of shape ``(B, T)``.
        continuation: Continuation flags (1 - done) of shape ``(B, T)``.
        valid_mask: Binary validity mask of shape ``(B, T)``.
        gamma: Discount factor in ``[0, 1]``.
        horizon: Maximum look-ahead horizon.
        decay: Exponential decay base applied to the weight of each
            successive horizon step.
        metric_prefix: String prefix for metric dictionary keys.
        target_scale_quantile: Quantile used for target normalisation
            scale estimation.
        target_scale_min: Minimum allowed target normalisation scale.

    Returns:
        A tuple of (loss, metrics) where:
            loss: Scalar weighted multi-horizon loss tensor.
            metrics: Dictionary of diagnostic floats including the
                combined loss, absolute error, predicted/target means,
                target scale, number of horizons used, and observability
                statistics.
    """
    losses: list[torch.Tensor] = []
    weights: list[float] = []
    abs_err_terms: list[torch.Tensor] = []
    pred_mean_terms: list[torch.Tensor] = []
    target_mean_terms: list[torch.Tensor] = []
    target_samples: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    pred_samples: list[torch.Tensor] = []
    horizon_weights_sum = 0.0
    target_sequences = _multistep_discounted_targets(
        reward_conservative_targets,
        continuation,
        valid_mask,
        gamma,
        horizon,
    )
    for step, target_k, continuation_k, valid_k in target_sequences:
        seq_len = target_k.shape[1]
        pred_k = (
            value_seq[:, :seq_len]
            - (float(gamma) ** step) * continuation_k * value_seq[:, step : step + seq_len]
        )
        weight = float(decay) ** (step - 1)
        pred_samples.append(pred_k)
        target_samples.append(target_k)
        target_masks.append(valid_k)
        weights.append(weight)
        horizon_weights_sum += weight
    if not target_samples:
        zero = value_seq.new_zeros(())
        metrics = {
            f"{metric_prefix}/L_exact_increment": 0.0,
            f"{metric_prefix}/exact_increment_abs_err": 0.0,
            f"{metric_prefix}/exact_increment_pred_mean": 0.0,
            f"{metric_prefix}/exact_increment_target_mean": 0.0,
            f"{metric_prefix}/exact_increment_target_scale": 0.0,
            f"{metric_prefix}/exact_increment_horizon_used": 0.0,
        }
        return zero, metrics
    target_scale = _target_normalization_scale(
        target_samples,
        target_masks,
        quantile=target_scale_quantile,
        min_scale=target_scale_min,
        template=value_seq,
    )
    for (step, target_k, continuation_k, valid_k), weight in zip(
        target_sequences, weights, strict=False
    ):
        seq_len = target_k.shape[1]
        pred_k = (
            value_seq[:, :seq_len]
            - (float(gamma) ** step) * continuation_k * value_seq[:, step : step + seq_len]
        )
        losses.append(weight * _masked_mean(((pred_k - target_k) / target_scale).pow(2), valid_k))
        abs_err_terms.append(weight * _masked_mean((pred_k - target_k).abs(), valid_k))
        pred_mean_terms.append(weight * _masked_mean(pred_k, valid_k))
        target_mean_terms.append(weight * _masked_mean(target_k, valid_k))
    total_weight = max(horizon_weights_sum, 1e-8)
    loss = torch.stack(losses).sum() / total_weight
    pred_concat = torch.cat([pred.reshape(-1) for pred in pred_samples])
    target_concat = torch.cat([target.reshape(-1) for target in target_samples])
    mask_concat = torch.cat([mask.reshape(-1) for mask in target_masks])
    support_threshold = max(0.1 * float(target_scale.detach()), 1e-6)
    metrics = {
        f"{metric_prefix}/L_exact_increment": float(loss.detach()),
        f"{metric_prefix}/exact_increment_abs_err": float(
            (torch.stack(abs_err_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/exact_increment_pred_mean": float(
            (torch.stack(pred_mean_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/exact_increment_target_mean": float(
            (torch.stack(target_mean_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/exact_increment_target_scale": float(target_scale.detach()),
        f"{metric_prefix}/exact_increment_horizon_used": float(len(losses)),
    }
    metrics.update(
        _exact_increment_observability_metrics(
            pred=pred_concat,
            target=target_concat,
            mask=mask_concat,
            metric_prefix=metric_prefix,
            support_threshold=support_threshold,
        ),
    )
    return loss, metrics


def _multistep_covector_alignment_loss(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    z_seq: torch.Tensor,
    value_seq: torch.Tensor,
    exact_covector_seq: torch.Tensor,
    continuation: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    horizon: int,
    decay: float,
    metric_prefix: str,
    reward_conservative_targets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Align ``dV`` with multi-step exact increments along replay or policy geodesics.

    Extends the single-step covector alignment to multiple horizons on
    the logarithmic ladder. At each horizon ``k``, computes a first-order
    Taylor prediction of the k-step value difference using Poincare
    log-map displacements and the exact covector, then penalises
    deviation from the discounted target. Also estimates an adaptive
    stiffness scale from displacement norms.

    Args:
        config: Dreamer configuration object. Uses
            ``critic_target_scale_quantile``,
            ``critic_target_scale_min``,
            ``critic_stiffness_target_scale``,
            ``critic_stiffness_quantile``,
            ``critic_stiffness_min``, and
            ``critic_stiffness_target_max``.
        metric: A conformal metric module (see ``_metric_inverse_scale``).
        z_seq: Latent-space coordinate sequence of shape
            ``(B, T+1, D)``.
        value_seq: Value predictions of shape ``(B, T+1)``.
        exact_covector_seq: Exact covector field sequence of shape
            ``(B, T+1, D)``.
        continuation: Continuation flags (1 - done) of shape ``(B, T)``.
        valid_mask: Binary validity mask of shape ``(B, T)``.
        gamma: Discount factor in ``[0, 1]``.
        horizon: Maximum look-ahead horizon.
        decay: Exponential decay base for horizon weighting.
        metric_prefix: String prefix for metric dictionary keys.
        reward_conservative_targets: Optional one-step conservative
            reward targets of shape ``(B, T)``. When provided, multi-step
            targets are built from these; otherwise, targets are derived
            from the detached value sequence.

    Returns:
        A tuple of (loss, stiffness_scale, metrics) where:
            loss: Scalar weighted multi-horizon covector alignment loss
                tensor.
            stiffness_scale: Detached adaptive stiffness scale tensor
                derived from reward-to-displacement ratios across all
                horizons.
            metrics: Dictionary of diagnostic floats including the
                combined loss, absolute error, predicted/target reward
                means, displacement norm, target scale, adaptive
                stiffness target, and number of horizons used.
    """
    B, T_plus_1, latent_dim = z_seq.shape
    del B, latent_dim
    T = T_plus_1 - 1
    max_horizon = max(1, min(int(horizon), T))
    if reward_conservative_targets is not None:
        target_sequences = _multistep_discounted_targets(
            reward_conservative_targets,
            continuation,
            valid_mask,
            gamma,
            max_horizon,
        )
    else:
        target_sequences = []
        continuation_prod = continuation
        valid_prod = valid_mask
        for step in range(1, max_horizon + 1):
            seq_len = T - step + 1
            target_k = (
                value_seq[:, :seq_len]
                - (float(gamma) ** step)
                * continuation_prod[:, :seq_len]
                * value_seq[:, step : step + seq_len]
            ).detach()
            target_sequences.append(
                (
                    step,
                    target_k,
                    continuation_prod[:, :seq_len],
                    valid_prod[:, :seq_len],
                ),
            )
            if step == max_horizon:
                break
            continuation_prod = continuation_prod[:, :-1] * continuation[:, step:]
            valid_prod = valid_prod[:, :-1] * valid_mask[:, step:]

    losses: list[torch.Tensor] = []
    abs_err_terms: list[torch.Tensor] = []
    pred_mean_terms: list[torch.Tensor] = []
    target_mean_terms: list[torch.Tensor] = []
    disp_mean_terms: list[torch.Tensor] = []
    stiffness_samples: list[torch.Tensor] = []
    stiffness_masks: list[torch.Tensor] = []
    target_samples: list[torch.Tensor] = []
    target_masks: list[torch.Tensor] = []
    total_weight = 0.0

    for _, target_k, _, valid_k in target_sequences:
        target_samples.append(target_k)
        target_masks.append(valid_k)
    if not target_samples:
        zero = value_seq.new_zeros(())
        metrics = {
            f"{metric_prefix}/L_covector_align": 0.0,
            f"{metric_prefix}/covector_align_abs_err": 0.0,
            f"{metric_prefix}/covector_predicted_reward_mean": 0.0,
            f"{metric_prefix}/covector_target_reward_mean": 0.0,
            f"{metric_prefix}/covector_target_scale": 0.0,
            f"{metric_prefix}/displacement_norm_mean": 0.0,
            f"{metric_prefix}/stiffness_target_adaptive": float(config.critic_stiffness_min),
            f"{metric_prefix}/covector_horizon_used": 0.0,
        }
        return zero, value_seq.new_tensor(float(config.critic_stiffness_min)), metrics
    target_scale = _target_normalization_scale(
        target_samples,
        target_masks,
        quantile=config.critic_target_scale_quantile
        if metric_prefix.startswith("critic")
        else 0.75,
        min_scale=config.critic_target_scale_min if metric_prefix.startswith("critic") else 1e-3,
        template=value_seq,
    )
    for step, target_k, continuation_k, valid_k in target_sequences:
        seq_len = target_k.shape[1]
        z_curr = z_seq[:, :seq_len].reshape(-1, z_seq.shape[-1])
        z_future = z_seq[:, step : step + seq_len].reshape(-1, z_seq.shape[-1])
        displacement = poincare_log_map(z_curr.detach(), z_future.detach()).reshape(
            target_k.shape[0],
            seq_len,
            -1,
        )
        local_value_delta = (exact_covector_seq[:, :seq_len] * displacement).sum(dim=-1)
        predicted_k = value_seq[:, :seq_len] - (float(gamma) ** step) * continuation_k * (
            value_seq[:, :seq_len] + local_value_delta
        )
        weight = float(decay) ** (step - 1)
        losses.append(
            weight * _masked_mean(((predicted_k - target_k) / target_scale).pow(2), valid_k)
        )
        abs_err_terms.append(weight * _masked_mean((predicted_k - target_k).abs(), valid_k))
        pred_mean_terms.append(weight * _masked_mean(predicted_k, valid_k))
        target_mean_terms.append(weight * _masked_mean(target_k, valid_k))
        displacement_norm = (
            _metric_vector_norm_sq(
                metric,
                z_curr.detach(),
                displacement.reshape(-1, displacement.shape[-1]),
            )
            .sqrt()
            .reshape_as(target_k)
        )
        disp_mean_terms.append(weight * _masked_mean(displacement_norm, valid_k))
        stiffness_samples.append(
            (target_k.detach().abs() / (displacement_norm.detach() + 1e-8)).reshape(-1)
        )
        stiffness_masks.append(valid_k.reshape(-1))
        total_weight += weight

    total_weight = max(total_weight, 1e-8)
    loss = torch.stack(losses).sum() / total_weight
    reward_scale = torch.cat(stiffness_samples)
    reward_scale_mask = torch.cat(stiffness_masks)
    stiffness_scale = (
        float(config.critic_stiffness_target_scale)
        * _masked_quantile(
            reward_scale,
            reward_scale_mask,
            float(config.critic_stiffness_quantile),
        )
    ).clamp_min(max(float(config.critic_stiffness_min), 1e-8))
    stiffness_max = float(config.critic_stiffness_target_max)
    if stiffness_max > 0.0:
        stiffness_scale = stiffness_scale.clamp(max=stiffness_max)
    metrics = {
        f"{metric_prefix}/L_covector_align": float(loss.detach()),
        f"{metric_prefix}/covector_align_abs_err": float(
            (torch.stack(abs_err_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/covector_predicted_reward_mean": float(
            (torch.stack(pred_mean_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/covector_target_reward_mean": float(
            (torch.stack(target_mean_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/covector_target_scale": float(target_scale.detach()),
        f"{metric_prefix}/displacement_norm_mean": float(
            (torch.stack(disp_mean_terms).sum() / total_weight).detach(),
        ),
        f"{metric_prefix}/stiffness_target_adaptive": float(stiffness_scale.detach()),
        f"{metric_prefix}/covector_horizon_used": float(len(losses)),
    }
    return loss, stiffness_scale.detach(), metrics
