"""Unit tests for standalone macro-control primitives."""

from __future__ import annotations

import torch

from fragile.rl.macro_control import (
    compute_q_learning_loss,
    epsilon_greedy_macro_action,
    epsilon_greedy_macro_actions,
    hard_update_target,
    MacroQNetwork,
    soft_update_target,
)


def _normalized(x: torch.Tensor) -> torch.Tensor:
    return x / x.sum(dim=-1, keepdim=True).clamp_min(1e-6)


def test_macro_q_learning_loss_backprops() -> None:
    torch.manual_seed(3)
    q_network = MacroQNetwork(num_states=5, num_actions=3)
    target_q = MacroQNetwork(num_states=5, num_actions=3)
    hard_update_target(target_q, q_network)

    state_probs = _normalized(torch.rand(4, 5))
    next_state_probs = _normalized(torch.rand(4, 5))
    action_idx = torch.tensor([0, 2, 1, 0], dtype=torch.long)
    rewards = torch.tensor([1.0, 0.5, -0.25, 0.0])
    continuation = torch.tensor([1.0, 1.0, 0.0, 1.0])

    loss, metrics = compute_q_learning_loss(
        q_network,
        target_q,
        state_probs,
        action_idx,
        rewards,
        continuation,
        next_state_probs,
        gamma=0.99,
        loss_type="huber",
    )
    loss.backward()

    assert loss.ndim == 0
    assert q_network.q_table.grad is not None
    assert metrics["q/loss"] >= 0.0
    assert "q/action_gap" in metrics


def test_epsilon_greedy_macro_action_returns_valid_action() -> None:
    q_network = MacroQNetwork(num_states=4, num_actions=3)
    with torch.no_grad():
        q_network.q_table.zero_()
        q_network.q_table[:, 2] = 1.0

    state_probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
    action_idx, q_values, was_random = epsilon_greedy_macro_action(
        q_network,
        state_probs,
        epsilon=0.0,
    )

    assert action_idx == 2
    assert q_values.shape == (3,)
    assert was_random is False


def test_epsilon_greedy_macro_actions_batches_actions() -> None:
    q_network = MacroQNetwork(num_states=4, num_actions=3)
    with torch.no_grad():
        q_network.q_table.zero_()
        q_network.q_table[:, 1] = 0.5
        q_network.q_table[:, 2] = 1.0

    state_probs = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    action_idx, q_values, was_random = epsilon_greedy_macro_actions(
        q_network,
        state_probs,
        epsilon=0.0,
    )

    assert action_idx.tolist() == [2, 2]
    assert q_values.shape == (2, 3)
    assert was_random.shape == (2,)
    assert not bool(was_random.any())


def test_soft_update_target_moves_parameters() -> None:
    source = MacroQNetwork(num_states=3, num_actions=2)
    target = MacroQNetwork(num_states=3, num_actions=2)
    with torch.no_grad():
        source.q_table.fill_(2.0)
        target.q_table.zero_()

    soft_update_target(target, source, tau=0.25)

    assert torch.allclose(target.q_table, torch.full_like(target.q_table, 0.5))
