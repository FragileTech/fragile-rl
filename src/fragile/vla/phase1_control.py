"""Phase 1 routing regulators for representation-only training."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Phase1AdaptiveState:
    """State for Phase 1 adaptive multipliers."""

    entropy_scale: float = 1.0
    chart_usage_scale: float = 1.0
    chart_ot_scale: float = 1.0
    code_usage_scale: float = 0.0
    code_gate_open_epoch: int | None = None


def _band_violation(value: float, low: float | None, high: float | None) -> float:
    violation = 0.0
    if low is not None:
        violation += max(0.0, low - value)
    if high is not None:
        violation += max(0.0, value - high)
    return violation


def _exp_update(
    value: float,
    signal: float,
    lr: float,
    *,
    min_value: float = 0.1,
    max_value: float = 8.0,
) -> float:
    updated = value * math.exp(lr * signal)
    return max(min_value, min(max_value, updated))


def _decay_toward(value: float, target: float, decay: float) -> float:
    return target + (value - target) * max(0.0, 1.0 - decay)


def init_phase1_adaptive_state(args) -> Phase1AdaptiveState | None:
    """Initialize controller state when adaptive multipliers are enabled."""
    if not getattr(args, "phase1_adaptive_multipliers", False):
        return None
    code_scale = 0.0 if getattr(args, "code_usage_gate_h", 0.0) > 0 else 1.0
    return Phase1AdaptiveState(code_usage_scale=code_scale)


def phase1_effective_weight_scales(args, state: Phase1AdaptiveState | None) -> dict[str, float]:
    """Return current weight scales for Phase 1 losses."""
    if state is None:
        return {
            "entropy_scale": 1.0,
            "chart_usage_scale": 1.0,
            "chart_ot_scale": 1.0,
            "code_usage_scale": 1.0,
        }
    return {
        "entropy_scale": state.entropy_scale,
        "chart_usage_scale": state.chart_usage_scale,
        "chart_ot_scale": state.chart_ot_scale,
        "code_usage_scale": state.code_usage_scale,
    }


def update_phase1_adaptive_state(
    state: Phase1AdaptiveState | None,
    args,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    epoch: int,
) -> None:
    """Update adaptive multipliers from routing diagnostics."""
    if state is None:
        return

    max_scale = float(getattr(args, "phase1_multiplier_max", 8.0))
    decay = float(getattr(args, "phase1_multiplier_decay", 0.05))

    top1_target = float(getattr(args, "conf_target_top1", 0.55))
    conf_error = max(0.0, top1_target - float(eval_metrics.get("soft_top1_prob_mean", 0.0)))
    if conf_error > 0.0:
        state.entropy_scale = _exp_update(
            state.entropy_scale,
            conf_error,
            float(getattr(args, "conf_multiplier_lr", 1.5)),
            max_value=max_scale,
        )
    else:
        state.entropy_scale = _decay_toward(state.entropy_scale, 1.0, decay)

    hard_entropy = float(eval_metrics.get("hard_entropy", 0.0))
    chart_violation = _band_violation(
        hard_entropy,
        getattr(args, "chart_usage_h_low", None),
        getattr(args, "chart_usage_h_high", None),
    )
    if chart_violation > 0.0:
        state.chart_usage_scale = _exp_update(
            state.chart_usage_scale,
            chart_violation,
            float(getattr(args, "chart_multiplier_lr", 1.0)),
            max_value=max_scale,
        )
    else:
        state.chart_usage_scale = _decay_toward(state.chart_usage_scale, 1.0, decay)

    soft_i_xk = float(eval_metrics.get("soft_I_XK", 0.0))
    ot_target = float(getattr(args, "chart_ot_i_target", 0.35))
    ot_violation = max(0.0, ot_target - soft_i_xk)
    if ot_violation > 0.0:
        state.chart_ot_scale = _exp_update(
            state.chart_ot_scale,
            ot_violation,
            float(getattr(args, "chart_ot_multiplier_lr", 1.0)),
            max_value=max_scale,
        )
    else:
        state.chart_ot_scale = _decay_toward(state.chart_ot_scale, 1.0, decay)

    code_gate_h = float(getattr(args, "code_usage_gate_h", 0.0))
    if code_gate_h > 0.0 and hard_entropy < code_gate_h:
        state.code_usage_scale = 0.0
        state.code_gate_open_epoch = None
        return

    if state.code_gate_open_epoch is None:
        state.code_gate_open_epoch = epoch

    ramp_epochs = max(int(getattr(args, "code_usage_ramp_epochs", 1)), 1)
    ramp_progress = min((epoch - state.code_gate_open_epoch + 1) / ramp_epochs, 1.0)

    code_entropy = float(eval_metrics.get("code_entropy_mean_active", 0.0))
    code_violation = _band_violation(
        code_entropy,
        getattr(args, "code_usage_h_low", None),
        getattr(args, "code_usage_h_high", None),
    )
    target_scale = ramp_progress
    if code_violation > 0.0:
        target_scale = min(
            ramp_progress
            * _exp_update(
                max(state.code_usage_scale, 1e-3),
                code_violation,
                float(getattr(args, "code_multiplier_lr", 0.5)),
                min_value=1e-3,
                max_value=max_scale,
            ),
            max_scale,
        )
    state.code_usage_scale = _decay_toward(state.code_usage_scale, target_scale, decay)
