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
    """Mean over entries where ``mask`` is one."""
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _metric_inverse_scale(metric: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Return the inverse conformal metric scale ``lambda(z)^{-2}``."""
    cf = metric.conformal_factor(z)
    epsilon = getattr(metric, "epsilon", 1e-8)
    return 1.0 / (cf.pow(2) + epsilon)


def _metric_covector_norm_sq(
    metric: nn.Module,
    z: torch.Tensor,
    covector: torch.Tensor,
) -> torch.Tensor:
    """Covector norm under the inverse conformal metric."""
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
    """Gate residual reward until the exact field is non-flat and force-consistent."""
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
    """Bias the reward split toward the exact sector before using the residual."""
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
