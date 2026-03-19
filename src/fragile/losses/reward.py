"""Reward decomposition losses for geometric Dreamer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import nn


if TYPE_CHECKING:
    from fragile.rl.config import DreamerConfig


# ---------------------------------------------------------------------------
# Shared helpers (small, duplicated where needed across loss modules)
# ---------------------------------------------------------------------------


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one.

    Args:
        values: Tensor of arbitrary shape containing the values to average.
        mask: Tensor broadcastable to ``values`` with ones indicating which
            entries to include and zeros for entries to ignore.

    Returns:
        Scalar tensor with the masked mean. When the mask sums to zero the
        denominator is clamped to one to avoid division by zero.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _metric_inverse_scale(metric: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Return the inverse conformal metric scale ``lambda(z)^{-2}``.

    Args:
        metric: Neural-network module exposing a ``conformal_factor(z)`` method
            and an optional ``epsilon`` attribute used for numerical stability.
        z: Latent-state tensor of shape ``(*, latent_dim)`` at which the
            conformal factor is evaluated.

    Returns:
        Tensor of the same leading shape as ``z`` containing the inverse
        squared conformal factor ``1 / (lambda(z)^2 + epsilon)``.
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

    Computes ``lambda(z)^{-2} * ||covector||^2`` where the Euclidean norm is
    taken over the last dimension of ``covector``.

    Args:
        metric: Neural-network module exposing a ``conformal_factor(z)`` method
            (forwarded to :func:`_metric_inverse_scale`).
        z: Latent-state tensor of shape ``(*, latent_dim)``.
        covector: Covector tensor of shape ``(*, latent_dim)`` whose squared
            norm is to be computed.

    Returns:
        Tensor of shape ``(*)`` containing the squared covector norm at each
        point, weighted by the inverse conformal scale.
    """
    return _metric_inverse_scale(metric, z).squeeze(-1) * covector.pow(2).sum(dim=-1)


# ---------------------------------------------------------------------------
# Reward losses
# ---------------------------------------------------------------------------


def _reward_nonconservative_gate(
    config: DreamerConfig,
    *,
    exact_covector_norm_mean: torch.Tensor,
    force_rel_err_mean: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gate residual reward until the exact field is non-flat and force-consistent.

    The gate is the product of a stiffness factor (ramps from 0 to 1 as the
    exact covector norm grows relative to ``config.critic_stiffness_min``) and
    a force-consistency factor (decays exponentially with the relative force
    error).

    Args:
        config: Dreamer configuration object providing
            ``critic_stiffness_min`` and
            ``reward_nonconservative_force_err_scale``.
        exact_covector_norm_mean: Scalar tensor with the mean norm of the
            exact (conservative) reward covector field.
        force_rel_err_mean: Scalar tensor with the mean relative error
            between the predicted and true force fields.

    Returns:
        A tuple of two elements:
            gate: Scalar tensor in ``[0, 1]`` that multiplicatively gates the
                non-conservative reward contribution.
            metrics: Dictionary of monitoring scalars keyed by metric name,
                including the overall gate value and its stiffness and force
                sub-factors.
    """
    stiffness_scale = max(float(config.critic_stiffness_min), 1e-8)
    stiffness_factor = (exact_covector_norm_mean / stiffness_scale).clamp(0.0, 1.0)
    force_factor = torch.exp(
        -float(config.reward_nonconservative_force_err_scale) * force_rel_err_mean.clamp(min=0.0),
    )
    gate = (stiffness_factor * force_factor).clamp(0.0, 1.0)
    metrics = {
        "wm/reward_nonconservative_gate": float(gate.detach()),
        "wm/reward_nonconservative_gate_stiffness": float(stiffness_factor.detach()),
        "wm/reward_nonconservative_gate_force": float(force_factor.detach()),
    }
    return gate, metrics


def _reward_conservative_preference_losses(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    z: torch.Tensor,
    reward_conservative: torch.Tensor,
    reward_nonconservative: torch.Tensor,
    reward_form_cov: torch.Tensor,
    replay_valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Bias the reward split toward the exact sector before using the residual.

    Computes two regularisation losses that encourage the non-conservative
    (residual) reward component to remain small relative to the conservative
    (exact) component: a covector-norm penalty and a budget-excess penalty.

    Args:
        config: Dreamer configuration object providing
            ``reward_nonconservative_budget_floor`` and
            ``reward_nonconservative_budget_ratio``.
        metric: Neural-network module exposing a ``conformal_factor(z)``
            method, used to compute the metric-weighted covector norm.
        z: Latent-state tensor of shape ``(batch, latent_dim)``.
        reward_conservative: Conservative (exact / potential-derived) reward
            tensor of shape ``(batch, 1)``.
        reward_nonconservative: Non-conservative (residual) reward tensor of
            shape ``(batch,)``.
        reward_form_cov: Covector field of the non-conservative reward form,
            tensor of shape ``(batch, latent_dim)``.
        replay_valid: Boolean-like mask tensor of shape ``(batch,)`` indicating
            which time-steps carry valid replay data.

    Returns:
        A tuple of three elements:
            L_reward_nonconservative_norm: Scalar loss penalising the
                metric-weighted squared norm of the non-conservative reward
                covector field.
            L_reward_nonconservative_budget: Scalar loss penalising the
                squared excess of the non-conservative reward magnitude beyond
                an adaptive budget derived from the conservative reward.
            metrics: Dictionary of monitoring scalars keyed by metric name,
                including both loss values, the mean budget, the mean excess,
                and the masked fraction of residual reward.
    """
    reward_cons_mag = reward_conservative.detach().squeeze(-1).abs()
    reward_noncons_mag = reward_nonconservative.abs()
    budget = (
        float(config.reward_nonconservative_budget_floor)
        + float(config.reward_nonconservative_budget_ratio) * reward_cons_mag
    )
    reward_residual_excess = (reward_noncons_mag - budget).clamp(min=0.0)
    reward_form_norm_sq = _metric_covector_norm_sq(metric, z, reward_form_cov)
    L_reward_nonconservative_norm = _masked_mean(
        reward_form_norm_sq,
        replay_valid.reshape(-1),
    )
    L_reward_nonconservative_budget = _masked_mean(
        reward_residual_excess.pow(2),
        replay_valid,
    )
    residual_frac = _masked_mean(
        reward_noncons_mag / (reward_noncons_mag + reward_cons_mag + 1e-8),
        replay_valid,
    )
    metrics = {
        "wm/L_reward_nonconservative_norm": float(L_reward_nonconservative_norm),
        "wm/L_reward_nonconservative_budget": float(L_reward_nonconservative_budget),
        "wm/reward_nonconservative_budget_mean": float(_masked_mean(budget, replay_valid)),
        "wm/reward_nonconservative_excess_mean": float(
            _masked_mean(reward_residual_excess, replay_valid),
        ),
        "wm/reward_nonconservative_frac_masked": float(residual_frac),
    }
    return L_reward_nonconservative_norm, L_reward_nonconservative_budget, metrics
