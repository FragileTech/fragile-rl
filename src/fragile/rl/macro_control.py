"""Off-policy control primitives for standalone macro RL.

This module keeps control deliberately simple for the first macro RL loop:

- the observation macro state is a soft belief over atlas symbols,
- the action is a discrete macro action symbol,
- the controller is a small Q-learning head over that symbolic space.

There is no planning here yet; the coarse Markov model is trained in parallel
for future use, while action selection is handled by standard Q-learning.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn
import torch.nn.functional as F


class MacroQNetwork(nn.Module):
    """Table-style Q network evaluated under a soft symbolic state belief.

    The network stores one Q value per hard `(state, action)` pair, then uses
    the current soft state belief to compute expected Q values for all actions.
    This keeps the controller tied to the actual atlas symbols instead of
    inventing a second learned tokenization inside the RL head.
    """

    def __init__(self, num_states: int, num_actions: int) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        if self.num_states <= 0 or self.num_actions <= 0:
            msg = "num_states and num_actions must both be positive."
            raise ValueError(msg)
        self.q_table = nn.Parameter(torch.zeros(self.num_states, self.num_actions))

    def forward(self, state_probs: torch.Tensor) -> torch.Tensor:
        """Return Q values for every discrete action under a soft state belief."""
        if state_probs.shape[-1] != self.num_states:
            msg = "state_probs has the wrong number of symbolic states."
            raise ValueError(msg)
        return torch.einsum("...s,sa->...a", state_probs.to(self.q_table), self.q_table)

    def q_for_actions(
        self,
        state_probs: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Gather Q values for the chosen discrete macro actions."""
        q_values = self(state_probs)
        if q_values.shape[:-1] != action_idx.shape:
            msg = "action_idx must match the leading shape of state_probs."
            raise ValueError(msg)
        return q_values.gather(dim=-1, index=action_idx.long().unsqueeze(-1)).squeeze(-1)


def epsilon_greedy_macro_action(
    q_network: MacroQNetwork,
    state_probs: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[int, torch.Tensor, bool]:
    """Sample one macro action with epsilon-greedy exploration."""
    action_idx, q_values, was_random = epsilon_greedy_macro_actions(
        q_network,
        state_probs,
        epsilon=epsilon,
    )
    return int(action_idx[0].item()), q_values[0], bool(was_random[0].item())


def epsilon_greedy_macro_actions(
    q_network: MacroQNetwork,
    state_probs: torch.Tensor,
    *,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample one epsilon-greedy macro action for every row in a batched belief."""
    if state_probs.dim() == 1:
        state_probs = state_probs.unsqueeze(0)
    if state_probs.dim() != 2:
        msg = "state_probs must have shape [S] or [B, S] for action selection."
        raise ValueError(msg)

    q_values = q_network(state_probs)
    greedy_actions = q_values.argmax(dim=-1)
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        was_random = torch.zeros_like(greedy_actions, dtype=torch.bool)
        return greedy_actions, q_values.detach(), was_random

    batch_size = state_probs.shape[0]
    device = q_values.device
    was_random = torch.rand(batch_size, device=device) < epsilon
    random_actions = torch.randint(
        q_network.num_actions,
        size=(batch_size,),
        device=device,
    )
    action_idx = torch.where(was_random, random_actions, greedy_actions)
    return action_idx, q_values.detach(), was_random


def compute_q_learning_loss(
    q_network: MacroQNetwork,
    target_q_network: MacroQNetwork,
    state_probs: torch.Tensor,
    action_idx: torch.Tensor,
    rewards: torch.Tensor,
    continuation: torch.Tensor,
    next_state_probs: torch.Tensor,
    *,
    gamma: float,
    loss_type: Literal["huber", "mse"] = "huber",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute one-step off-policy TD loss in symbolic state space."""
    if state_probs.dim() != 2 or next_state_probs.dim() != 2:
        msg = "state_probs and next_state_probs must have shape [B, S]."
        raise ValueError(msg)
    if state_probs.shape != next_state_probs.shape:
        msg = "state_probs and next_state_probs must have the same shape."
        raise ValueError(msg)
    if action_idx.dim() != 1 or rewards.dim() != 1 or continuation.dim() != 1:
        msg = "action_idx, rewards, and continuation must all be rank-1 tensors."
        raise ValueError(msg)
    if action_idx.shape[0] != state_probs.shape[0]:
        msg = "Batch size mismatch between state_probs and action_idx."
        raise ValueError(msg)

    q_values = q_network(state_probs)
    chosen_q = q_values.gather(dim=-1, index=action_idx.long().unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        next_q = target_q_network(next_state_probs)
        next_max_q = next_q.max(dim=-1).values
        td_target = rewards + float(gamma) * continuation * next_max_q

    if loss_type == "huber":
        loss = F.smooth_l1_loss(chosen_q, td_target)
    elif loss_type == "mse":
        loss = F.mse_loss(chosen_q, td_target)
    else:
        msg = "loss_type must be one of {'huber', 'mse'}."
        raise ValueError(msg)

    sorted_q = q_values.sort(dim=-1, descending=True).values
    top1 = sorted_q[:, 0]
    top2 = sorted_q[:, 1] if q_values.shape[-1] > 1 else torch.zeros_like(top1)

    metrics = {
        "q/loss": float(loss.detach()),
        "q/value_mean": float(chosen_q.mean().detach()),
        "q/target_mean": float(td_target.mean().detach()),
        "q/td_abs": float((chosen_q - td_target).abs().mean().detach()),
        "q/q_mean": float(q_values.mean().detach()),
        "q/q_max": float(q_values.max(dim=-1).values.mean().detach()),
        "q/action_gap": float((top1 - top2).mean().detach()),
    }
    return loss, metrics


def hard_update_target(target_network: nn.Module, source_network: nn.Module) -> None:
    """Copy all parameters from `source_network` into `target_network`."""
    target_network.load_state_dict(source_network.state_dict())


@torch.no_grad()
def soft_update_target(
    target_network: nn.Module,
    source_network: nn.Module,
    tau: float,
) -> None:
    """Exponential moving-average update for target-network parameters."""
    tau = float(tau)
    if not 0.0 < tau <= 1.0:
        msg = "tau must lie in the interval (0, 1]."
        raise ValueError(msg)
    for target_param, source_param in zip(
        target_network.parameters(),
        source_network.parameters(),
        strict=True,
    ):
        target_param.data.lerp_(source_param.data, tau)


__all__ = [
    "MacroQNetwork",
    "compute_q_learning_loss",
    "epsilon_greedy_macro_action",
    "epsilon_greedy_macro_actions",
    "hard_update_target",
    "soft_update_target",
]
