"""Tests for Phase 1 OT balancing and adaptive regulation."""

from types import SimpleNamespace

import torch

from fragile.losses.encoder import compute_sinkhorn_balanced_chart_loss
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    update_phase1_adaptive_state,
)


def test_sinkhorn_chart_loss_prefers_balanced_scores() -> None:
    """Balanced router scores should fit the OT assignment target better."""
    balanced_scores = torch.tensor(
        [
            [5.0, 0.0],
            [4.0, 0.0],
            [0.0, 5.0],
            [0.0, 4.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    collapsed_scores = torch.tensor(
        [
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
            [5.0, 0.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )

    loss_balanced, metrics_balanced = compute_sinkhorn_balanced_chart_loss(
        balanced_scores,
        epsilon=0.1,
        num_iters=30,
    )
    loss_collapsed, _metrics_collapsed = compute_sinkhorn_balanced_chart_loss(
        collapsed_scores,
        epsilon=0.1,
        num_iters=30,
    )

    assert loss_balanced.item() < loss_collapsed.item()
    assert metrics_balanced["ot_plan_col_l1"] < 1e-3
    assert metrics_balanced["ot_target_top1_mean"] > 0.9

    loss_collapsed.backward()
    assert collapsed_scores.grad is not None
    assert collapsed_scores.grad.abs().sum().item() > 0.0


def test_phase1_adaptive_state_gates_code_usage_until_charts_are_alive() -> None:
    """Code-usage pressure should stay off until chart entropy clears the gate."""
    args = SimpleNamespace(
        phase1_adaptive_multipliers=True,
        phase1_multiplier_max=8.0,
        phase1_multiplier_decay=0.05,
        conf_target_top1=0.55,
        conf_multiplier_lr=1.5,
        chart_multiplier_lr=1.0,
        chart_usage_h_low=1.95,
        chart_usage_h_high=2.05,
        chart_ot_i_target=0.35,
        chart_ot_multiplier_lr=1.0,
        code_usage_gate_h=1.25,
        code_usage_ramp_epochs=10,
        code_usage_h_low=2.3,
        code_usage_h_high=2.65,
        code_multiplier_lr=0.5,
    )

    state = init_phase1_adaptive_state(args)
    assert state is not None
    assert state.code_usage_scale == 0.0

    update_phase1_adaptive_state(
        state,
        args,
        train_metrics={},
        eval_metrics={
            "hard_entropy": 0.2,
            "soft_I_XK": 0.0,
            "soft_top1_prob_mean": 0.125,
            "code_entropy_mean_active": 0.0,
        },
        epoch=0,
    )
    assert state.code_usage_scale == 0.0
    assert state.code_gate_open_epoch is None

    update_phase1_adaptive_state(
        state,
        args,
        train_metrics={},
        eval_metrics={
            "hard_entropy": 1.5,
            "soft_I_XK": 0.4,
            "soft_top1_prob_mean": 0.6,
            "code_entropy_mean_active": 0.0,
        },
        epoch=1,
    )
    assert state.code_gate_open_epoch == 1
    assert state.code_usage_scale > 0.0
