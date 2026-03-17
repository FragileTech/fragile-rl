"""Lambda-return (TD-lambda / GAE) computation for imagined trajectories."""

from __future__ import annotations

import torch


def compute_lambda_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    lambda_gae: float = 0.95,
) -> torch.Tensor:
    r"""Compute lambda-returns from reward-aligned bootstrap values.

    .. math::
        G_t^\lambda = r_t + \gamma\bigl[(1-\lambda)\,V^\text{boot}_{t}
                      + \lambda\,G_{t+1}^\lambda\bigr]

    Args:
        rewards: [B, H] rewards at each imagined step.
        values: [B, H] bootstrap values aligned with each reward step.
            In a model-based rollout this is typically ``V(s_{t+1})``.
        gamma: Discount factor.
        lambda_gae: GAE lambda for bias-variance trade-off.

    Returns:
        returns: [B, H] lambda-return targets (detached from value graph).
    """
    B, H = rewards.shape
    device = rewards.device
    returns = torch.zeros(B, H, device=device)

    # Bootstrap from the next-state value aligned with the final reward.
    next_value = values[:, -1]
    returns[:, -1] = rewards[:, -1] + gamma * next_value

    for t in reversed(range(H - 1)):
        next_val = values[:, t]
        returns[:, t] = (
            rewards[:, t]
            + gamma * (1.0 - lambda_gae) * next_val
            + gamma * lambda_gae * returns[:, t + 1]
        )

    return returns
