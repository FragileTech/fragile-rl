"""Geometric Dreamer: Phase 4 model-based RL on the Poincare ball.

Trains a two-manifold Dreamer whose deployed motor variable is the canonical
action latent on the action manifold, not a separately reconstructed control
covector.

The encoder uses the same Phase 1 reconstruction/atlas objective as
``train_joint.py`` plus observation-world-model closure/zeno synchronization.
The observation world model is the only learned Markov predictor; the action
topoencoder remains an action codec and supervision anchor, not a second
transition model. All RL heads consume detached latents; they never update the
topoencoder directly.

Usage:
    uv run python -m fragile.rl.train_dreamer
    uv run python -m fragile.rl.train_dreamer --domain walker --task walk
"""

from __future__ import annotations

import argparse
import contextlib
import copy
from dataclasses import asdict, fields
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import FactorizedJumpOperator, TopoEncoder
from fragile.layers.gauge import hyperbolic_distance, poincare_log_map, project_to_ball
from fragile.losses.macro import zeno_loss
from fragile.losses.old_macro import (
    compute_enclosure_loss,
    EnclosureProbe,
    grl_alpha_schedule,
)
from fragile.losses.world_model import (
    compute_dynamics_chart_loss,
    compute_dynamics_geodesic_loss,
    compute_energy_conservation_loss,
    compute_hodge_consistency_loss,
    compute_momentum_regularization,
    compute_screened_poisson_loss,
)
from fragile.vla.config import VLAConfig
from fragile.vla.covariant_world_model import GeometricWorldModel
from fragile.vla.optim import build_encoder_param_groups
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    update_phase1_adaptive_state,
)
from fragile.vla.shared_dyn.encoder import SharedDynTopoEncoder
from fragile.vla.train_joint import (
    _compute_encoder_losses,
    _get_hard_routing_tau,
    _phase1_grad_breakdown,
    _use_hard_routing,
)

from .action_manifold import symbolize_latent_with_atlas
from .actor import GeometricActor
from .boundary import (
    critic_value,
)
from .config import DreamerConfig
from .env_helpers import (
    _apply_task_preset,
    _build_episode_dict,
    _flatten_obs,
    _infer_action_dim,
    _make_env,
    _sample_collection_action,
    ObservationNormalizer,
)
from .replay_buffer import SequenceReplayBuffer
from .reward_head import RewardHead


try:
    from fragile.mlflow_logging import (
        end_mlflow_run,
        log_mlflow_metrics,
    )
except ImportError:
    log_mlflow_metrics = None  # type: ignore[assignment]
    end_mlflow_run = None  # type: ignore[assignment]

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False


def _rollout_routing_tau(hard_routing: bool, hard_routing_tau: float) -> float:
    """Preserve the configured routing temperature for rollouts and evaluation."""
    del hard_routing
    return hard_routing_tau


def _structured_state_from_encoder_output(
    enc_out: tuple[torch.Tensor, ...],
) -> dict[str, torch.Tensor]:
    """Pack the encoder's `(K, z_n, z_geo)` outputs into a named state dict."""
    return {
        "chart_idx": enc_out[0],
        "code_idx": enc_out[1],
        "z_n": enc_out[2],
        "router_weights": enc_out[4],
        "z_geo": enc_out[5],
        "z_q": enc_out[11],
    }


def _optimizer_parameters(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    """Return unique optimizer parameters in stable order."""
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if id(param) in seen:
                continue
            seen.add(id(param))
            params.append(param)
    return params


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Return the first param-group learning rate for logging."""
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0]["lr"])


def _shared_value_field(world_model: nn.Module, critic: nn.Module) -> bool:
    """Return whether the critic is the world-model value branch."""
    wm = _unwrap_compiled_module(world_model)
    critic_mod = _unwrap_compiled_module(critic)
    return critic_mod is getattr(wm, "potential_net", None)


class MacroValueModel(nn.Module):
    """Symbolic control head defined on top of the closure-consistent state/action dynamics."""

    def __init__(
        self,
        num_charts: int,
        codes_per_chart: int,
        num_action_charts: int,
        action_codes_per_chart: int,
    ) -> None:
        super().__init__()
        self.num_charts = int(num_charts)
        self.codes_per_chart = int(codes_per_chart)
        self.num_states = self.num_charts * self.codes_per_chart
        self.num_action_charts = int(num_action_charts)
        self.action_codes_per_chart = int(action_codes_per_chart)
        self.num_actions = self.num_action_charts * self.action_codes_per_chart
        self.state_action_q = nn.Embedding(self.num_states * self.num_actions, 1)
        self.state_action_reward = nn.Embedding(self.num_states * self.num_actions, 1)
        nn.init.zeros_(self.state_action_q.weight)
        nn.init.zeros_(self.state_action_reward.weight)

    def _default_action_probs(self, state_probs: torch.Tensor) -> torch.Tensor:
        return state_probs.new_full(
            (state_probs.shape[0], self.num_actions),
            1.0 / max(self.num_actions, 1),
        )

    def reward_from_state_action_idx(
        self,
        state_idx: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> torch.Tensor:
        sa_idx = state_idx.long() * self.num_actions + action_idx.long()
        return self.state_action_reward(sa_idx).squeeze(-1)

    def reward_from_probs(
        self, state_probs: torch.Tensor, action_probs: torch.Tensor
    ) -> torch.Tensor:
        reward_table = self.state_action_reward.weight.view(self.num_states, self.num_actions)
        return torch.einsum("bi,ij,bj->b", state_probs, reward_table, action_probs)

    def q_from_state_action_idx(
        self,
        state_idx: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> torch.Tensor:
        sa_idx = state_idx.long() * self.num_actions + action_idx.long()
        return self.state_action_q(sa_idx).squeeze(-1)

    def q_from_probs(self, state_probs: torch.Tensor, action_probs: torch.Tensor) -> torch.Tensor:
        q_table = self.state_action_q.weight.view(self.num_states, self.num_actions)
        return torch.einsum("bi,ij,bj->b", state_probs, q_table, action_probs)

    def value_from_probs(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if action_probs is None:
            action_probs = self._default_action_probs(state_probs)
        return self.q_from_probs(state_probs, action_probs)

    def q_from_transition(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
        next_state_probs: torch.Tensor,
        *,
        next_action_probs: torch.Tensor | None = None,
        gamma: float,
        continuation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        reward = self.reward_from_probs(state_probs, action_probs)
        next_value = self.value_from_probs(next_state_probs, next_action_probs)
        if continuation is None:
            continuation_scale = next_value.new_ones(next_value.shape)
        else:
            continuation_scale = continuation.reshape_as(next_value).to(next_value)
        return reward + float(gamma) * continuation_scale * next_value


def _state_index(
    chart_idx: torch.Tensor, code_idx: torch.Tensor, codes_per_chart: int
) -> torch.Tensor:
    """Flatten `(chart, code)` symbolic state indices."""
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _soft_symbolic_state_distribution(
    atlas_model: nn.Module,
    z_latent: torch.Tensor,
    *,
    router_weights_override: torch.Tensor | None = None,
    hard_routing_tau: float = 1.0,
    code_tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return a soft symbolic state distribution for latent pullback/value lifting."""
    obs_info = symbolize_latent_with_atlas(
        atlas_model,
        z_latent,
        router_weights_override=router_weights_override,
        hard_routing=False,
        hard_routing_tau=hard_routing_tau,
    )
    atlas = getattr(atlas_model, "encoder", atlas_model)
    codebook = project_to_ball(atlas.codebook).to(device=z_latent.device, dtype=z_latent.dtype)
    v_local = obs_info["v_local"]
    B, K, C = v_local.shape[0], codebook.shape[0], codebook.shape[1]
    flat_v = v_local[:, None, None, :].expand(B, K, C, -1).reshape(-1, z_latent.shape[-1])
    flat_code = codebook[None].expand(B, -1, -1, -1).reshape(-1, z_latent.shape[-1])
    code_dist = hyperbolic_distance(flat_v, flat_code).reshape(B, K, C)
    code_logits = -(code_dist.square()) / max(float(code_tau), 1e-6)
    code_probs = F.softmax(code_logits, dim=-1)
    state_probs = (obs_info["router_weights"].unsqueeze(-1) * code_probs).reshape(B, K * C)
    state_probs = state_probs / state_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return {
        **obs_info,
        "code_probs": code_probs,
        "state_probs": state_probs,
        "state_value_entropy": -(state_probs * state_probs.clamp(min=1e-8).log()).sum(dim=-1),
    }


def _symbolic_transition_supervision_losses(
    *,
    state_probs: torch.Tensor,
    code_probs: torch.Tensor,
    target_charts: torch.Tensor,
    target_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    codes_per_chart: int,
    metric_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return replay-supervised code and full-symbol losses for a symbolic transition."""
    flat_state_probs = state_probs.reshape(-1, state_probs.shape[-1])
    flat_code_probs = code_probs.reshape(-1, code_probs.shape[-2], code_probs.shape[-1])
    flat_target_charts = target_charts.reshape(-1).long()
    flat_target_codes = target_codes.reshape(-1).long()
    flat_target_state = _state_index(flat_target_charts, flat_target_codes, codes_per_chart)
    flat_valid = valid_mask.reshape(-1).to(flat_state_probs)
    batch_idx = torch.arange(flat_target_state.shape[0], device=flat_state_probs.device)

    target_chart_code_probs = flat_code_probs[batch_idx, flat_target_charts]
    target_code_log_prob = (
        target_chart_code_probs
        .gather(
            1,
            flat_target_codes.unsqueeze(-1),
        )
        .squeeze(-1)
        .clamp(min=1e-8)
        .log()
    )
    target_state_log_prob = (
        flat_state_probs
        .gather(
            1,
            flat_target_state.unsqueeze(-1),
        )
        .squeeze(-1)
        .clamp(min=1e-8)
        .log()
    )

    L_code = _masked_mean(-target_code_log_prob, flat_valid)
    L_symbol = _masked_mean(-target_state_log_prob, flat_valid)

    code_pred_target_chart = target_chart_code_probs.argmax(dim=-1)
    symbol_pred = flat_state_probs.argmax(dim=-1)
    symbol_pred_chart = torch.div(symbol_pred, codes_per_chart, rounding_mode="floor")
    symbol_pred_code = symbol_pred.remainder(codes_per_chart)

    symbol_entropy = -(flat_state_probs * flat_state_probs.clamp(min=1e-8).log()).sum(dim=-1)
    metrics = {
        f"{metric_prefix}/L_code": float(L_code.detach()),
        f"{metric_prefix}/L_symbol": float(L_symbol.detach()),
        f"{metric_prefix}/code_nll": float(L_code.detach()),
        f"{metric_prefix}/symbol_nll": float(L_symbol.detach()),
        f"{metric_prefix}/code_acc": float(
            _masked_mean(
                (code_pred_target_chart == flat_target_codes).to(flat_state_probs.dtype),
                flat_valid,
            ).detach(),
        ),
        f"{metric_prefix}/chart_acc_from_symbol": float(
            _masked_mean(
                (symbol_pred_chart == flat_target_charts).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/symbol_acc": float(
            _masked_mean(
                (symbol_pred == flat_target_state).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/symbol_code_acc": float(
            _masked_mean(
                (symbol_pred_code == flat_target_codes).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/state_entropy": float(_masked_mean(symbol_entropy, flat_valid).detach()),
    }
    return L_code, L_symbol, metrics


def _action_symbolic_distribution_from_actor(
    action_state: dict[str, torch.Tensor],
    *,
    code_tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return a soft symbolic action distribution from actor chart/code logits."""
    chart_probs = F.softmax(action_state["action_chart_logits"], dim=-1)
    code_logits = action_state["action_code_logits"] / max(float(code_tau), 1e-6)
    code_probs = F.softmax(code_logits, dim=-1)
    state_probs = (chart_probs.unsqueeze(-1) * code_probs).reshape(chart_probs.shape[0], -1)
    state_probs = state_probs / state_probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    return {
        "chart_probs": chart_probs,
        "code_probs": code_probs,
        "state_probs": state_probs,
        "state_value_entropy": -(state_probs * state_probs.clamp(min=1e-8).log()).sum(dim=-1),
    }


def _macro_state_transition_distribution(
    atlas_model: nn.Module,
    world_model: GeometricWorldModel | nn.Module,
    z_latent: torch.Tensor,
    action_canonical: torch.Tensor,
    chart_probs: torch.Tensor,
    *,
    hard_routing_tau: float = 1.0,
    code_tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return the existing macro symbolic transition kernel from the world model."""
    wm = _unwrap_compiled_module(world_model)
    p_latent = wm.momentum_init(z_latent)
    step_out = wm._rollout_transition(
        z_latent,
        p_latent,
        action_canonical,
        chart_probs,
        track_energy=False,
    )
    next_symbolic = _soft_symbolic_state_distribution(
        atlas_model,
        step_out["z"],
        router_weights_override=step_out["rw"],
        hard_routing_tau=hard_routing_tau,
        code_tau=code_tau,
    )
    return {
        "next_z": step_out["z"],
        "next_router_weights": next_symbolic["router_weights"],
        "next_chart_idx": next_symbolic["chart_idx"],
        "next_code_idx": next_symbolic["code_idx"],
        "next_z_n": next_symbolic["z_n"],
        "next_code_probs": next_symbolic["code_probs"],
        "next_state_probs": next_symbolic["state_probs"],
        "next_state_entropy": next_symbolic["state_value_entropy"],
        "next_chart_entropy": -(
            next_symbolic["router_weights"] * next_symbolic["router_weights"].clamp(min=1e-8).log()
        ).sum(dim=-1),
    }


def _macro_transition_observability_metrics(
    *,
    macro_critic: MacroValueModel,
    state_idx: torch.Tensor,
    next_state_probs: torch.Tensor,
    reward_pred: torch.Tensor,
    reward_target: torch.Tensor,
    next_value: torch.Tensor,
    bootstrap_term: torch.Tensor,
    valid_mask: torch.Tensor,
    metric_prefix: str,
) -> dict[str, float]:
    """Observability metrics for the transition-backed symbolic Bellman backbone."""
    next_state_probs_flat = next_state_probs.reshape(-1, macro_critic.num_states)
    state_idx_flat = state_idx.reshape(-1).long()
    reward_pred_flat = reward_pred.reshape(-1)
    reward_target_flat = reward_target.reshape(-1)
    next_value_flat = next_value.reshape(-1)
    bootstrap_term_flat = bootstrap_term.reshape(-1)
    valid_flat = valid_mask.reshape(-1)
    next_state_entropy = -(
        next_state_probs_flat * next_state_probs_flat.clamp(min=1e-8).log()
    ).sum(dim=-1)
    next_state_top1 = next_state_probs_flat.max(dim=-1).values
    self_transition_prob = next_state_probs_flat.gather(1, state_idx_flat.unsqueeze(-1)).squeeze(
        -1
    )
    positive_chart_mask = (
        macro_critic.state_action_q.weight
        .detach()
        .view(macro_critic.num_states, macro_critic.num_actions)
        .amax(dim=-1)
        > 0.0
    ).to(next_state_probs_flat)
    positive_value_mass = (next_state_probs_flat * positive_chart_mask.unsqueeze(0)).sum(dim=-1)
    return {
        f"{metric_prefix}/reward_pred_mean": float(
            _masked_mean(reward_pred_flat, valid_flat).detach()
        ),
        f"{metric_prefix}/reward_target_mean": float(
            _masked_mean(reward_target_flat, valid_flat).detach(),
        ),
        f"{metric_prefix}/reward_abs_err": float(
            _masked_mean((reward_pred_flat - reward_target_flat).abs(), valid_flat).detach(),
        ),
        f"{metric_prefix}/value_next_mean": float(
            _masked_mean(next_value_flat, valid_flat).detach()
        ),
        f"{metric_prefix}/bootstrap_term_mean": float(
            _masked_mean(bootstrap_term_flat, valid_flat).detach(),
        ),
        f"{metric_prefix}/next_state_entropy": float(
            _masked_mean(next_state_entropy, valid_flat).detach(),
        ),
        f"{metric_prefix}/next_state_top1_prob": float(
            _masked_mean(next_state_top1, valid_flat).detach(),
        ),
        f"{metric_prefix}/self_transition_prob": float(
            _masked_mean(self_transition_prob, valid_flat).detach(),
        ),
        f"{metric_prefix}/next_state_positive_value_mass": float(
            _masked_mean(positive_value_mass, valid_flat).detach(),
        ),
    }


def _macro_transition_sharpening_gate(
    config: DreamerConfig,
    *,
    obs_state_acc: float,
    enclosure_defect_acc: float,
    enclosure_defect_ce: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Sharpen macro transitions only when symbolic closure is already grounded."""
    chance = 1.0 / max(int(config.num_charts), 1)
    acc_target = max(float(config.macro_transition_closure_acc_target), chance + 1e-6)
    obs_state_acc_t = template.new_tensor(obs_state_acc)
    enclosure_defect_acc_t = template.new_tensor(enclosure_defect_acc)
    enclosure_defect_ce_t = template.new_tensor(enclosure_defect_ce)

    obs_factor = ((obs_state_acc_t - chance) / (acc_target - chance)).clamp(0.0, 1.0)
    defect_acc_factor = torch.exp(
        -float(config.macro_transition_enclosure_defect_acc_scale)
        * enclosure_defect_acc_t.clamp(min=0.0),
    )
    defect_ce_factor = torch.exp(
        -float(config.macro_transition_enclosure_defect_ce_scale)
        * enclosure_defect_ce_t.clamp(min=0.0),
    )
    closure_gate = (obs_factor * defect_acc_factor * defect_ce_factor).clamp(0.0, 1.0)
    metrics = {
        "macro/transition_sharpen_gate": float(closure_gate.detach()),
        "macro/transition_sharpen_obs_factor": float(obs_factor.detach()),
        "macro/transition_sharpen_defect_acc_factor": float(defect_acc_factor.detach()),
        "macro/transition_sharpen_defect_ce_factor": float(defect_ce_factor.detach()),
    }
    return closure_gate, metrics


def _macro_pullback_value_covector(
    atlas_model: nn.Module,
    macro_critic: MacroValueModel,
    z_latent: torch.Tensor,
    *,
    action_state_probs: torch.Tensor | None = None,
    router_weights_override: torch.Tensor | None = None,
    hard_routing_tau: float = 1.0,
    code_tau: float = 1.0,
    create_graph: bool = False,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    """Lift the macro value table back to latent space and differentiate it."""
    z_req = z_latent.detach().requires_grad_(True)
    router_override = (
        router_weights_override.detach() if router_weights_override is not None else None
    )
    symbolic = _soft_symbolic_state_distribution(
        atlas_model,
        z_req,
        router_weights_override=router_override,
        hard_routing_tau=hard_routing_tau,
        code_tau=code_tau,
    )
    macro_value = macro_critic.value_from_probs(
        symbolic["state_probs"],
        action_state_probs.detach() if action_state_probs is not None else None,
    ).unsqueeze(-1)
    macro_covector = torch.autograd.grad(
        macro_value.sum(),
        z_req,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    if detach:
        macro_value = macro_value.detach()
        macro_covector = macro_covector.detach()
        symbolic = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in symbolic.items()
        }
    return {
        **symbolic,
        "macro_value": macro_value,
        "macro_covector": macro_covector,
    }


def _macro_control_pullback_covector(
    atlas_model: nn.Module,
    world_model: GeometricWorldModel | nn.Module,
    macro_critic: MacroValueModel,
    actor: GeometricActor | None,
    z_latent: torch.Tensor,
    *,
    action_state_probs: torch.Tensor,
    action_canonical: torch.Tensor,
    gamma: float,
    continuation: torch.Tensor | None = None,
    router_weights_override: torch.Tensor | None = None,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
    code_tau: float = 1.0,
    create_graph: bool = False,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    """Lift the transition-backed symbolic control backbone back to latent space."""
    z_req = z_latent.detach().requires_grad_(True)
    router_override = (
        router_weights_override.detach() if router_weights_override is not None else None
    )
    state_symbolic = _soft_symbolic_state_distribution(
        atlas_model,
        z_req,
        router_weights_override=router_override,
        hard_routing_tau=hard_routing_tau,
        code_tau=code_tau,
    )
    action_state_probs_t = action_state_probs.detach()
    action_canonical_t = action_canonical.detach()
    transition = _macro_state_transition_distribution(
        atlas_model,
        world_model,
        z_req,
        action_canonical_t,
        state_symbolic["router_weights"],
        hard_routing_tau=hard_routing_tau,
        code_tau=code_tau,
    )
    if actor is not None:
        with torch.no_grad():
            next_action_state = actor(
                transition["next_chart_idx"],
                transition["next_code_idx"],
                transition["next_z_n"],
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        next_action_state_probs = _action_symbolic_distribution_from_actor(
            next_action_state,
            code_tau=code_tau,
        )["state_probs"].detach()
    else:
        next_action_state_probs = action_state_probs_t
    macro_reward = macro_critic.reward_from_probs(
        state_symbolic["state_probs"],
        action_state_probs_t,
    ).unsqueeze(-1)
    macro_control = macro_critic.q_from_transition(
        state_symbolic["state_probs"],
        action_state_probs_t,
        transition["next_state_probs"],
        next_action_probs=next_action_state_probs,
        gamma=gamma,
        continuation=continuation,
    ).unsqueeze(-1)
    macro_covector = torch.autograd.grad(
        macro_control.sum(),
        z_req,
        create_graph=create_graph,
        retain_graph=create_graph or not detach,
    )[0]
    if detach:
        macro_reward = macro_reward.detach()
        macro_control = macro_control.detach()
        macro_covector = macro_covector.detach()
        transition = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in transition.items()
        }
        state_symbolic = {
            key: value.detach() if torch.is_tensor(value) else value
            for key, value in state_symbolic.items()
        }
    return {
        **state_symbolic,
        **transition,
        "next_action_state_probs": next_action_state_probs,
        "macro_reward": macro_reward,
        "macro_control": macro_control,
        "macro_covector": macro_covector,
    }


def _conservative_reward_from_value(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
    z_next: torch.Tensor,
    rw_next: torch.Tensor,
    gamma: float,
    continuation: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the exact discounted reward increment from the value field."""
    v_curr = critic_value(critic, z, rw)
    v_next = critic_value(critic, z_next, rw_next)
    continuation_scale = gamma if continuation is None else gamma * continuation
    reward_conservative = v_curr - continuation_scale * v_next
    return reward_conservative, v_curr, v_next


def _value_covector_from_critic(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
    *,
    create_graph: bool = False,
    detach: bool = True,
) -> torch.Tensor:
    """Return the exact state covector field ``dV`` induced by the critic."""
    z_req = z if z.requires_grad else z.detach().requires_grad_(True)
    value = critic_value(critic, z_req, rw.detach()).sum()
    value_covector = torch.autograd.grad(
        value,
        z_req,
        create_graph=create_graph,
        retain_graph=create_graph,
    )[0]
    return value_covector.detach() if detach else value_covector


def _parameter_grad_norm(parameters: list[torch.nn.Parameter]) -> float:
    """Compute an unclipped gradient norm for diagnostics."""
    grads = [p.grad.detach().norm() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0
    return float(torch.norm(torch.stack(grads), p=2))


@contextlib.contextmanager
def _temporary_requires_grad(
    modules: list[tuple[nn.Module, bool]],
) -> None:
    """Temporarily override ``requires_grad`` for a list of modules."""
    saved_states: list[tuple[nn.Parameter, bool]] = []
    for module, requires_grad in modules:
        for param in module.parameters():
            saved_states.append((param, param.requires_grad))
            param.requires_grad_(requires_grad)
    try:
        yield
    finally:
        for param, requires_grad in saved_states:
            param.requires_grad_(requires_grad)


def _make_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    min_lr: float,
) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    """Cosine-anneal from the optimizer base LR to ``min_lr`` across the run."""
    t_max = max(int(total_epochs) - 1, 1)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=min_lr,
    )


def _maybe_compile_module(module: nn.Module, enabled: bool, mode: str) -> nn.Module:
    """Optionally wrap a module with ``torch.compile`` without changing defaults."""
    if not enabled or not hasattr(torch, "compile"):
        return module
    return torch.compile(module, mode=mode)


def _unwrap_compiled_module(module: nn.Module) -> nn.Module:
    """Return the original module behind ``torch.compile`` wrappers."""
    return getattr(module, "_orig_mod", module)


def _mark_compile_step_begin() -> None:
    """Hint TorchInductor/CUDAGraphs that a new training step is starting."""
    compiler = getattr(torch, "compiler", None)
    mark_step = getattr(compiler, "cudagraph_mark_step_begin", None)
    if callable(mark_step):
        mark_step()


def _policy_action(
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    obs_info: dict[str, torch.Tensor],
    *,
    hard_routing: bool = True,
    hard_routing_tau: float = -1.0,
) -> dict[str, torch.Tensor]:
    """Decode the canonical motor-side action state and discard execution texture."""
    action_state = actor(
        obs_info["chart_idx"],
        obs_info["code_idx"],
        obs_info["z_n"],
        hard_routing=hard_routing,
        hard_routing_tau=hard_routing_tau,
    )
    with torch.inference_mode():
        action_mean, _, _ = action_model.decoder(
            action_state["action_z_geo"].detach(),
            router_weights=action_state["action_router_weights"].detach(),
            routing_tau=hard_routing_tau,
        )
    return {
        "action": action_mean.detach(),
        "action_mean": action_mean.detach(),
        "action_canonical": action_state["action_z_geo"].detach(),
        "action_latent": action_state["action_z_geo"].detach(),
        "action_latent_mean": action_state["action_z_geo"].detach(),
        "action_router_weights": action_state["action_router_weights"].detach(),
        "action_chart_idx": action_state["action_chart_idx"].detach(),
        "action_code_idx": action_state["action_code_idx"].detach(),
        "action_code_latent": action_state["action_z_q"].detach(),
        "action_z_n": action_state["action_z_n"].detach(),
        "action_chart_logits": action_state["action_chart_logits"],
        "action_code_logits": action_state["action_code_logits"],
    }


def _collect_episode(
    env,
    actor: GeometricActor | None,
    action_model: SharedDynTopoEncoder | None,
    encoder: nn.Module,
    device: torch.device,
    control_dim: int,
    num_action_charts: int,
    obs_normalizer: ObservationNormalizer | None = None,
    action_repeat: int = 1,
    max_steps: int = 1000,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
    sigma_motor: float = 0.0,
) -> dict[str, np.ndarray]:
    """Collect a single episode.  Uses random actions if the policy is absent."""
    obs_list, act_list, rew_list, done_list = [], [], [], []
    action_mean_list = []
    action_latent_list, action_router_weight_list = [], []
    action_chart_idx_list, action_code_idx_list, action_code_latent_list = [], [], []
    time_step = env.reset()
    action_spec = env.action_spec()
    step = 0
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)

    while not time_step.last() and step < max_steps:
        obs = _flatten_obs(time_step)
        obs_list.append(obs)

        if actor is None or action_model is None:
            action = np.random.uniform(
                action_spec.minimum,
                action_spec.maximum,
                size=action_spec.shape,
            ).astype(np.float32)
            action_mean = action.copy()
            action_latent = np.zeros(control_dim, dtype=np.float32)
            action_router_weights = np.zeros(num_action_charts, dtype=np.float32)
            action_chart_idx = np.int64(0)
            action_code_idx = np.int64(0)
            action_code_latent = np.zeros(control_dim, dtype=np.float32)
        else:
            obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            with torch.no_grad():
                enc_out = encoder.encoder(
                    obs_t,
                    routing_tau=routing_tau,
                )
                obs_info = _structured_state_from_encoder_output(enc_out)
            action_out = _policy_action(
                actor,
                action_model,
                obs_info,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action_mean = action_out["action_mean"].squeeze(0).cpu().numpy()
            action = _sample_collection_action(
                action_mean,
                action_min=np.asarray(action_spec.minimum, dtype=np.float32),
                action_max=np.asarray(action_spec.maximum, dtype=np.float32),
                sigma_motor=sigma_motor,
            )
            action_latent = action_out["action_canonical"].squeeze(0).cpu().numpy()
            action_router_weights = action_out["action_router_weights"].squeeze(0).cpu().numpy()
            action_chart_idx = np.int64(action_out["action_chart_idx"].item())
            action_code_idx = np.int64(action_out["action_code_idx"].item())
            action_code_latent = action_out["action_code_latent"].squeeze(0).cpu().numpy()
            action_mean = np.clip(action_mean, action_spec.minimum, action_spec.maximum)

        total_reward = 0.0
        for _ in range(action_repeat):
            time_step = env.step(action)
            total_reward += time_step.reward or 0.0
            if time_step.last():
                break

        act_list.append(action)
        action_mean_list.append(action_mean)
        action_latent_list.append(action_latent)
        action_router_weight_list.append(action_router_weights)
        action_chart_idx_list.append(action_chart_idx)
        action_code_idx_list.append(action_code_idx)
        action_code_latent_list.append(action_code_latent)
        rew_list.append(np.float32(total_reward))
        done_list.append(np.float32(time_step.last()))
        step += 1

    # Append final observation
    obs_list.append(_flatten_obs(time_step))

    return _build_episode_dict(
        obs_list,
        act_list,
        rew_list,
        done_list,
        action_mean_list,
        action_latent_list,
        action_router_weight_list,
        action_chart_idx_list,
        action_code_idx_list,
        action_code_latent_list,
    )


def _collect_parallel_episodes(
    env: object,  # VectorizedDMControlEnv or similar
    actor: GeometricActor | None,
    action_model: SharedDynTopoEncoder | None,
    encoder: nn.Module,
    device: torch.device,
    control_dim: int,
    num_action_charts: int,
    *,
    num_episodes: int,
    obs_normalizer: ObservationNormalizer | None = None,
    action_repeat: int = 1,
    max_steps: int = 1000,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
    sigma_motor: float = 0.0,
) -> list[dict[str, np.ndarray]]:
    """Collect multiple episodes in parallel with batched policy inference."""
    if num_episodes < 1:
        return []
    if num_episodes > env.n_workers:
        msg = f"num_episodes={num_episodes} exceeds available workers={env.n_workers}"
        raise ValueError(msg)

    action_spec = env.action_space
    action_min = np.asarray(action_spec.minimum, dtype=np.float32)
    action_max = np.asarray(action_spec.maximum, dtype=np.float32)
    action_shape = tuple(action_spec.shape)
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)
    env_indices = np.arange(num_episodes, dtype=int)
    observations = env.reset_batch(env_indices=env_indices).astype(np.float32, copy=False)

    obs_lists = [[observations[i].copy()] for i in range(num_episodes)]
    act_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    rew_lists: list[list[np.float32]] = [[] for _ in range(num_episodes)]
    done_lists: list[list[np.float32]] = [[] for _ in range(num_episodes)]
    action_mean_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    action_latent_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    action_router_weight_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]
    action_chart_lists: list[list[np.int64]] = [[] for _ in range(num_episodes)]
    action_code_lists: list[list[np.int64]] = [[] for _ in range(num_episodes)]
    action_code_latent_lists: list[list[np.ndarray]] = [[] for _ in range(num_episodes)]

    active = np.ones(num_episodes, dtype=bool)
    step_counts = np.zeros(num_episodes, dtype=np.int32)

    while active.any():
        active_indices = np.flatnonzero(active)
        active_obs = observations[active_indices]

        if actor is None or action_model is None:
            actions = np.random.uniform(
                action_min,
                action_max,
                size=(len(active_indices), *action_shape),
            ).astype(np.float32)
            action_means = actions.copy()
            action_latents = np.zeros((len(active_indices), control_dim), dtype=np.float32)
            action_router_weights = np.zeros(
                (len(active_indices), num_action_charts),
                dtype=np.float32,
            )
            action_chart_idx = np.zeros(len(active_indices), dtype=np.int64)
            action_code_idx = np.zeros(len(active_indices), dtype=np.int64)
            action_code_latents = np.zeros((len(active_indices), control_dim), dtype=np.float32)
        else:
            obs_t = torch.from_numpy(active_obs).to(device)
            if obs_normalizer is not None:
                obs_t = obs_normalizer.normalize_tensor(obs_t)
            with torch.no_grad():
                enc_out = encoder.encoder(
                    obs_t,
                    routing_tau=routing_tau,
                )
                obs_info = _structured_state_from_encoder_output(enc_out)
            action_out = _policy_action(
                actor,
                action_model,
                obs_info,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action_means = action_out["action_mean"].cpu().numpy()
            actions = _sample_collection_action(
                action_means,
                action_min=action_min,
                action_max=action_max,
                sigma_motor=sigma_motor,
            )
            action_latents = action_out["action_canonical"].cpu().numpy()
            action_router_weights = action_out["action_router_weights"].cpu().numpy()
            action_chart_idx = (
                action_out["action_chart_idx"].cpu().numpy().astype(np.int64, copy=False)
            )
            action_code_idx = (
                action_out["action_code_idx"].cpu().numpy().astype(np.int64, copy=False)
            )
            action_code_latents = action_out["action_code_latent"].cpu().numpy()
            action_means = np.clip(action_means, action_min, action_max)

        next_obs, rewards, dones, _trunc, _infos = env.step_actions_batch(
            actions.astype(np.float64, copy=False),
            dt=np.full(len(active_indices), action_repeat, dtype=int),
            env_indices=active_indices,
        )
        next_obs = next_obs.astype(np.float32, copy=False)

        for row, episode_idx in enumerate(active_indices):
            act_lists[episode_idx].append(actions[row].astype(np.float32, copy=False))
            action_mean_lists[episode_idx].append(
                action_means[row].astype(np.float32, copy=False),
            )
            action_latent_lists[episode_idx].append(
                action_latents[row].astype(np.float32, copy=False),
            )
            action_router_weight_lists[episode_idx].append(
                action_router_weights[row].astype(np.float32, copy=False),
            )
            action_chart_lists[episode_idx].append(np.int64(action_chart_idx[row]))
            action_code_lists[episode_idx].append(np.int64(action_code_idx[row]))
            action_code_latent_lists[episode_idx].append(
                action_code_latents[row].astype(np.float32, copy=False),
            )
            rew_lists[episode_idx].append(np.float32(rewards[row]))
            done_lists[episode_idx].append(np.float32(dones[row]))
            step_counts[episode_idx] += 1
            observations[episode_idx] = next_obs[row]
            obs_lists[episode_idx].append(next_obs[row].copy())
            if dones[row] or step_counts[episode_idx] >= max_steps:
                active[episode_idx] = False

    return [
        _build_episode_dict(
            obs_lists[i],
            act_lists[i],
            rew_lists[i],
            done_lists[i],
            action_mean_lists[i],
            action_latent_lists[i],
            action_router_weight_lists[i],
            action_chart_lists[i],
            action_code_lists[i],
            action_code_latent_lists[i],
        )
        for i in range(num_episodes)
    ]


def _eval_policy(
    env,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    encoder: nn.Module,
    device: torch.device,
    action_repeat: int,
    num_episodes: int,
    max_steps: int,
    obs_normalizer: ObservationNormalizer | None = None,
    hard_routing: bool = True,
    hard_routing_tau: float = 1.0,
) -> dict[str, float]:
    """Run evaluation episodes and return summary stats."""
    rewards, lengths = [], []
    action_spec = env.action_spec()
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)
    for _ in range(num_episodes):
        time_step = env.reset()
        ep_reward, ep_len = 0.0, 0
        while not time_step.last() and ep_len < max_steps:
            obs = _flatten_obs(time_step)
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
                if obs_normalizer is not None:
                    obs_t = obs_normalizer.normalize_tensor(obs_t)
                enc_out = encoder.encoder(
                    obs_t,
                    routing_tau=routing_tau,
                )
                obs_info = _structured_state_from_encoder_output(enc_out)
            action_out = _policy_action(
                actor,
                action_model,
                obs_info,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action = action_out["action"].squeeze(0).cpu().numpy()
            action = np.clip(action, action_spec.minimum, action_spec.maximum)
            for _ in range(action_repeat):
                time_step = env.step(action)
                ep_reward += time_step.reward or 0.0
                if time_step.last():
                    break
            ep_len += 1
        rewards.append(ep_reward)
        lengths.append(ep_len)
    return {
        "eval/reward_mean": float(np.mean(rewards)),
        "eval/reward_std": float(np.std(rewards)),
        "eval/length_mean": float(np.mean(lengths)),
    }


# ---------------------------------------------------------------------------
# Imagination
# ---------------------------------------------------------------------------


def _imagine(
    obs_model: TopoEncoder,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
    gamma: float,
    *,
    hard_routing: bool = True,
    hard_routing_tau: float = -1.0,
    reward_curl_batch_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Roll out the canonical action manifold in latent space.

    Returns:
        z_states: [B, H, D] policy states before each action
        rw_states: [B, H, K] router weights before each action
        z_traj: [B, H, D]
        rw_traj: [B, H, K]
        action_canonicals: [B, H, D]
        action_latents: [B, H, D]
        action_router_weights: [B, H, K_a]
        actions: [B, H, A]
        rewards: [B, H]
        reward_conservative: [B, H]
        reward_nonconservative: [B, H]
        reward_curl_norm: [B, H]
        reward_curl_valid: [B, H]
        phi_eff: [B, H, 1]
    """
    z, rw = z_0, rw_0
    p = world_model.momentum_init(z_0)  # [B, D]
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)

    z_state_list, rw_state_list = [], []
    z_list, rw_list, action_canonical_list, action_list = [], [], [], []
    action_latent_list, action_router_list = [], []
    r_list, r_cons_list, r_noncons_list, r_curl_list, r_curl_valid_list = [], [], [], [], []
    policy_chart_acc_list, policy_router_sync_list = [], []
    force_rel_err_list, hodge_cons_exact_list = [], []
    phi_list = []

    for _t in range(horizon):
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            router_weights_override=rw,
            hard_routing=hard_routing,
            hard_routing_tau=routing_tau,
        )
        action_out = _policy_action(
            actor,
            action_model,
            obs_info,
            hard_routing=hard_routing,
            hard_routing_tau=routing_tau,
        )
        z_state_list.append(z.detach())
        rw_state_list.append(rw.detach())
        action_canonical_list.append(action_out["action_canonical"])
        action_latent_list.append(action_out["action_latent"])
        action_router_list.append(action_out["action_router_weights"])
        action_list.append(action_out["action_mean"])
        z_curr = z
        rw_curr = rw
        force_diag = _conservative_force_diagnostics(
            world_model,
            z.detach(),
            p.detach(),
            rw.detach(),
            action_out["action_canonical"].detach(),
        )

        with torch.no_grad():
            step_out = world_model._rollout_transition(
                z,
                p,
                action_out["action_canonical"],
                rw,
                track_energy=False,
            )
            z_next = step_out["z"]
            p_next = step_out["p"]
            rw_next = step_out["rw"]
            phi_eff = step_out["phi_eff"]
            reward_conservative, _v_curr, _v_next = _conservative_reward_from_value(
                critic,
                z_curr,
                rw_curr,
                z_next,
                rw_next,
                gamma,
            )
            z = z_next
            p = p_next
            rw = rw_next
            next_obs_info = symbolize_latent_with_atlas(
                obs_model,
                z_next,
                router_weights_override=rw_next,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            chart_target = next_obs_info["chart_idx"].long()
            policy_chart_acc_list.append(
                (step_out["chart_logits"].argmax(dim=-1) == chart_target).float(),
            )
            policy_router_sync_list.append(
                (rw_next - next_obs_info["router_weights"]).abs().mean(dim=-1),
            )
        exact_covector = _value_covector_from_critic(critic, z_curr, rw_curr)
        rw_curr_detached = rw_curr.detach()

        def exact_covector_fn(z_req: torch.Tensor) -> torch.Tensor:
            return _value_covector_from_critic(
                critic,
                z_req,
                rw_curr_detached[: z_req.shape[0]],
                create_graph=True,
                detach=False,
            )

        with torch.no_grad():
            reward_info = reward_head.decompose(
                z_curr,
                rw_curr,
                action_out["action_latent"],
                action_out["action_router_weights"],
                action_out["action_code_latent"],
                action_canonical=action_out["action_canonical"],
                exact_covector=exact_covector,
                compute_curl=False,
            )
            r_noncons = reward_info["reward_nonconservative"]
            r_hat = reward_conservative + r_noncons  # [B, 1]

        reward_curl = reward_head.reward_curl(
            z_curr,
            rw_curr,
            action_out["action_latent"],
            action_out["action_router_weights"],
            action_out["action_code_latent"],
            exact_covector=exact_covector,
            exact_covector_fn=exact_covector_fn,
            max_batch=reward_curl_batch_limit,
        )
        reward_curl_norm = z_curr.new_zeros(z_curr.shape[0])
        reward_curl_valid = torch.zeros(z_curr.shape[0], dtype=torch.bool, device=z_curr.device)
        if reward_curl.numel() > 0:
            eval_batch = reward_curl.shape[0]
            reward_curl_norm[:eval_batch] = torch.linalg.norm(reward_curl, dim=(-2, -1))
            reward_curl_valid[:eval_batch] = True

        z_list.append(z.detach())
        rw_list.append(rw.detach())
        r_list.append(r_hat.squeeze(-1))
        r_cons_list.append(reward_conservative.squeeze(-1))
        r_noncons_list.append(r_noncons.squeeze(-1))
        r_curl_list.append(reward_curl_norm)
        r_curl_valid_list.append(reward_curl_valid)
        force_rel_err_list.append(force_diag["force_rel_err"])
        hodge_cons_exact_list.append(force_diag["hodge_exact"]["conservative_ratio"])
        phi_list.append(phi_eff)

    return {
        "z_states": torch.stack(z_state_list, dim=1),  # [B, H, D]
        "rw_states": torch.stack(rw_state_list, dim=1),  # [B, H, K]
        "z_traj": torch.stack(z_list, dim=1),  # [B, H, D]
        "rw_traj": torch.stack(rw_list, dim=1),  # [B, H, K]
        "action_canonicals": torch.stack(action_canonical_list, dim=1),  # [B, H, D]
        "action_latents": torch.stack(action_latent_list, dim=1),  # [B, H, D]
        "action_router_weights": torch.stack(action_router_list, dim=1),  # [B, H, K_a]
        "actions": torch.stack(action_list, dim=1),  # [B, H, A]
        "rewards": torch.stack(r_list, dim=1),  # [B, H]
        "reward_conservative": torch.stack(r_cons_list, dim=1),  # [B, H]
        "reward_nonconservative": torch.stack(r_noncons_list, dim=1),  # [B, H]
        "reward_curl_norm": torch.stack(r_curl_list, dim=1),  # [B, H]
        "reward_curl_valid": torch.stack(r_curl_valid_list, dim=1),  # [B, H]
        "policy_chart_acc": torch.stack(policy_chart_acc_list, dim=1),  # [B, H]
        "policy_router_sync": torch.stack(policy_router_sync_list, dim=1),  # [B, H]
        "policy_force_rel_err": torch.stack(force_rel_err_list, dim=1),  # [B, H]
        "policy_hodge_conservative_exact": torch.stack(hodge_cons_exact_list, dim=1),  # [B, H]
        "phi_eff": torch.stack(phi_list, dim=1),  # [B, H, 1]
    }


# ---------------------------------------------------------------------------
# VLAConfig bridge — create Phase 1 config from DreamerConfig
# ---------------------------------------------------------------------------


def _phase1_config(
    config: DreamerConfig,
    phase1_state=None,
    *,
    input_dim: int | None = None,
    num_charts: int | None = None,
    codes_per_chart: int | None = None,
) -> VLAConfig:
    """Build a VLAConfig with Phase 1 weights matching DreamerConfig."""
    scales = phase1_effective_weight_scales(config, phase1_state)
    return VLAConfig(
        input_dim=config.obs_dim if input_dim is None else input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts if num_charts is None else num_charts,
        codes_per_chart=config.codes_per_chart if codes_per_chart is None else codes_per_chart,
        w_feature_recon=config.w_feature_recon,
        w_vq=config.w_vq,
        w_entropy=config.w_entropy * scales["entropy_scale"],
        w_diversity=config.w_diversity * scales["chart_usage_scale"],
        chart_usage_entropy_low=config.chart_usage_h_low,
        chart_usage_entropy_high=config.chart_usage_h_high,
        w_chart_ot=config.w_chart_ot * scales["chart_ot_scale"],
        chart_ot_epsilon=config.chart_ot_epsilon,
        chart_ot_iters=config.chart_ot_iters,
        w_uniformity=config.w_uniformity,
        w_radial_calibration=config.w_radial_calibration,
        w_confidence_calibration=config.w_confidence_calibration,
        w_hard_routing_nll=config.w_hard_routing_nll,
        w_router_margin=config.w_router_margin,
        router_margin_target=config.router_margin_target,
        radial_quality_alpha=config.radial_quality_alpha,
        radial_vq_alpha=config.radial_vq_alpha,
        radial_quality_rank_mix=config.radial_quality_rank_mix,
        radial_recon_quality_weight=config.radial_recon_quality_weight,
        radial_quality_mix=config.radial_quality_mix,
        radial_quality_base_weight=config.radial_quality_base_weight,
        radial_calibration_rho_max=config.radial_calibration_rho_max,
        radial_calibration_band_width=config.radial_calibration_band_width,
        w_v_tangent_barrier=config.w_v_tangent_barrier,
        w_codebook_spread=config.w_codebook_spread,
        w_codebook_center=config.w_codebook_center,
        w_chart_center_mean=config.w_chart_center_mean,
        w_chart_center_radius=config.w_chart_center_radius,
        chart_center_radius_max=config.chart_center_radius_max,
        w_chart_center_sep=config.w_chart_center_sep,
        chart_center_sep_margin=config.chart_center_sep_margin,
        w_chart_collapse=config.w_chart_collapse,
        w_code_collapse=config.w_code_collapse * scales["code_usage_scale"],
        code_usage_entropy_low=config.code_usage_h_low,
        code_usage_entropy_high=config.code_usage_h_high,
        w_code_collapse_temperature=config.code_usage_temperature,
        w_window=config.w_window,
        w_window_eps_ground=config.w_window_eps_ground,
        w_consistency=config.w_consistency,
        w_perp=config.w_perp,
        lr_chart_centers_scale=config.lr_chart_centers_scale,
        lr_codebook_scale=config.lr_codebook_scale,
    )


# ---------------------------------------------------------------------------
# Atlas alignment
# ---------------------------------------------------------------------------


@torch.no_grad()
def _bind_chart_tokenizer_centers(chart_tok: nn.Module, centers: torch.Tensor) -> None:
    """Copy projected chart centers into a chart tokenizer and freeze them."""
    safe_centers = project_to_ball(centers.detach())
    chart_tok.chart_centers.copy_(
        safe_centers.to(
            device=chart_tok.chart_centers.device,
            dtype=chart_tok.chart_centers.dtype,
        ),
    )
    chart_tok.chart_centers.requires_grad_(False)


@torch.no_grad()
def _sync_rl_atlas(
    model: TopoEncoder,
    action_model: TopoEncoder,
    world_model: GeometricWorldModel,
    critic: nn.Module,
    actor: GeometricActor,
    reward_head: RewardHead,
    actor_old: GeometricActor | None = None,
) -> None:
    """Keep RL consumers bound to the observation or action atlas they read from."""
    obs_centers = getattr(model.encoder, "chart_centers", None)
    action_centers = getattr(action_model.encoder, "chart_centers", None)
    obs_codebook = getattr(model.encoder, "codebook", None)
    action_codebook = getattr(action_model.encoder, "codebook", None)
    if (
        obs_centers is None
        or action_centers is None
        or obs_codebook is None
        or action_codebook is None
    ):
        return

    obs_centers = project_to_ball(obs_centers.detach())
    action_centers = project_to_ball(action_centers.detach())
    obs_codebook = project_to_ball(obs_codebook.detach())
    action_codebook = project_to_ball(action_codebook.detach())
    world_model_mod = _unwrap_compiled_module(world_model)
    critic_mod = _unwrap_compiled_module(critic)
    reward_head_mod = _unwrap_compiled_module(reward_head)

    world_model_mod.bind_chart_centers(obs_centers, freeze=True)
    actor.bind_action_atlas(action_centers, action_codebook)
    if actor_old is not None:
        actor_old.bind_action_atlas(action_centers, action_codebook)
    if hasattr(critic_mod, "chart_tok"):
        _bind_chart_tokenizer_centers(critic_mod.chart_tok, obs_centers)
    _bind_chart_tokenizer_centers(reward_head_mod.action_chart_tok, action_centers)


# ---------------------------------------------------------------------------
# Per-chart diagnostics
# ---------------------------------------------------------------------------


def _per_chart_diagnostics(
    K_chart: torch.Tensor,
    K_code: torch.Tensor,
    router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> dict[str, float]:
    """Compute per-chart metrics from flattened encoder outputs."""
    metrics: dict[str, float] = {}

    K_chart = K_chart.long()
    K_code = K_code.long()

    counts = torch.bincount(K_chart, minlength=num_charts).to(dtype=z_geo.dtype)
    total = counts.sum().clamp(min=1.0)
    fracs = counts / total

    for k, frac in enumerate(fracs.tolist()):
        metrics[f"chart/{k}/usage"] = frac

    # Usage entropy (higher = more balanced)
    fracs_safe = fracs.clamp(min=1e-8)
    usage_entropy = -(fracs_safe * fracs_safe.log()).sum()
    metrics["chart/usage_entropy"] = float(usage_entropy)
    metrics["chart/usage_max_frac"] = float(fracs.max())
    metrics["chart/active_charts"] = float((counts > 0).sum())

    z_norms = z_geo.norm(dim=-1)
    z_norm_sums = torch.bincount(K_chart, weights=z_norms, minlength=num_charts)
    z_norm_means = z_norm_sums / counts.clamp(min=1.0)

    flat_code_idx = K_chart * codes_per_chart + K_code
    code_counts = (
        torch
        .bincount(
            flat_code_idx,
            minlength=num_charts * codes_per_chart,
        )
        .to(dtype=z_geo.dtype)
        .reshape(num_charts, codes_per_chart)
    )
    code_mass = code_counts.sum(dim=-1, keepdim=True).clamp(min=1.0)
    code_probs = code_counts / code_mass
    code_entropy = -(code_probs * code_probs.clamp(min=1e-8).log()).sum(dim=-1)
    code_perplexity = code_entropy.exp()
    active_codes = (code_counts > 0).sum(dim=-1).to(dtype=z_geo.dtype)
    active_symbols_total = float(active_codes.sum())

    for k in range(num_charts):
        metrics[f"chart/{k}/z_norm"] = float(z_norm_means[k])
        metrics[f"chart/{k}/code_entropy"] = float(code_entropy[k])
        metrics[f"chart/{k}/code_perplexity"] = float(code_perplexity[k])
        metrics[f"chart/{k}/active_codes"] = float(active_codes[k])

    # Router confidence: mean of max router weight (1.0 = perfectly confident)
    metrics["chart/router_confidence"] = float(router_weights.max(dim=-1).values.mean())

    # Top-2 gap
    if num_charts >= 2:
        top2 = router_weights.topk(2, dim=-1).values  # [B, 2]
        metrics["chart/top2_gap"] = float((top2[:, 0] - top2[:, 1]).mean())

    metrics["chart/active_symbols"] = active_symbols_total
    metrics["chart/active_symbol_fraction"] = active_symbols_total / (num_charts * codes_per_chart)

    return metrics


def _phase1_controller_metrics(
    metrics: dict[str, float],
    *,
    enc_prefix: str,
    chart_prefix: str,
    num_charts: int,
) -> dict[str, float]:
    """Map per-manifold Dreamer diagnostics to Phase 1 controller inputs."""

    def _active_code_entropies(prefix: str, num_charts: int) -> list[float]:
        entropies: list[float] = []
        for chart_idx in range(num_charts):
            active_codes = metrics.get(f"{prefix}/{chart_idx}/active_codes", 0.0)
            if active_codes > 0.0:
                entropies.append(metrics.get(f"{prefix}/{chart_idx}/code_entropy", 0.0))
        return entropies

    active_code_entropies = _active_code_entropies(chart_prefix, num_charts)

    return {
        "soft_top1_prob_mean": float(metrics.get(f"{enc_prefix}/top1_prob_mean", 0.0)),
        "soft_I_XK": float(metrics.get(f"{enc_prefix}/I_XK", 0.0)),
        "hard_entropy": float(metrics.get(f"{chart_prefix}/usage_entropy", 0.0)),
        "code_entropy_mean_active": (
            float(np.mean(active_code_entropies)) if active_code_entropies else 0.0
        ),
    }


def _transition_valid_mask(dones: torch.Tensor) -> torch.Tensor:
    """Mask valid replay transitions, excluding padded steps after termination."""
    if dones.numel() == 0:
        return dones
    prefix = torch.cat([torch.ones_like(dones[:, :1]), 1.0 - dones[:, :-1]], dim=1)
    return torch.cumprod(prefix, dim=1)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one."""
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Standard deviation over entries where ``mask`` is one."""
    valid = values.reshape(-1)[mask.reshape(-1).bool()]
    if valid.numel() == 0:
        return values.new_zeros(())
    return valid.std(unbiased=False)


def _masked_corrcoef(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pearson correlation over masked entries with a safe zero fallback."""
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
    """Fraction of masked entries with matching signs."""
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
    """Fraction of masked entries with magnitude above ``threshold``."""
    valid = mask.reshape(-1).bool()
    value_valid = values.reshape(-1)[valid]
    if value_valid.numel() == 0:
        return values.new_zeros(())
    return (value_valid.abs() >= float(threshold)).to(value_valid.dtype).mean()


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


def _metric_covector_pair(
    metric: nn.Module,
    z: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
) -> torch.Tensor:
    """Pair two covectors under the inverse conformal metric."""
    return _metric_inverse_scale(metric, z).squeeze(-1) * (left * right).sum(dim=-1)


def _metric_vector_norm_sq(
    metric: nn.Module,
    z: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Vector norm under the conformal metric."""
    return vector.pow(2).sum(dim=-1) / _metric_inverse_scale(metric, z).squeeze(-1).clamp_min(1e-8)


def _masked_quantile(values: torch.Tensor, mask: torch.Tensor, quantile: float) -> torch.Tensor:
    """Quantile over masked values with a safe fallback for empty masks."""
    valid = values[mask.reshape(-1).bool()]
    if valid.numel() == 0:
        return values.new_zeros(())
    if valid.numel() == 1:
        return valid[0]
    q = float(np.clip(quantile, 0.0, 1.0))
    return torch.quantile(valid, q)


def _exact_increment_observability_metrics(
    *,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    metric_prefix: str,
    support_threshold: float,
) -> dict[str, float]:
    """Detailed observability metrics for exact-increment supervision."""
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
    """Linear warmup from zero to one over ``warmup_epochs`` epochs."""
    if warmup_epochs <= 0:
        return 1.0
    return min(max((epoch + 1) / warmup_epochs, 0.0), 1.0)


def _critic_stage_scales(config: DreamerConfig, epoch: int) -> dict[str, float]:
    """Return epoch-dependent scales for critic formation vs shaping terms."""
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


def _weighted_loss_grad_norm(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
) -> float:
    """Return the L2 norm of gradients induced by ``loss`` on ``parameters``."""
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
    """Measure which critic losses dominate the value-field gradient."""
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


def _conservative_force_diagnostics(
    world_model: GeometricWorldModel,
    z: torch.Tensor,
    p: torch.Tensor,
    rw: torch.Tensor,
    action_canonical: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compare the direct conservative field to the exact scalar-gradient field."""
    if not hasattr(world_model, "potential_net"):
        zeros = z.new_zeros(z.shape[0])
        ones = z.new_ones(z.shape[0])
        return {
            "direct_terms": {},
            "exact_terms": {},
            "hodge_direct": {
                "conservative_ratio": ones,
                "solenoidal_ratio": zeros,
                "harmonic_ratio": zeros,
            },
            "hodge_exact": {
                "conservative_ratio": ones,
                "solenoidal_ratio": zeros,
                "harmonic_ratio": zeros,
            },
            "force_err_sq": zeros,
            "task_force_err_sq": zeros,
            "risk_force_err_sq": zeros,
            "force_rel_err": zeros,
            "task_force_rel_err": zeros,
            "risk_force_rel_err": zeros,
        }
    potential_net = world_model.potential_net
    direct_terms = potential_net.force_terms(z, rw)
    exact_terms = potential_net.exact_force_terms(z, rw)
    u_pi = world_model.control_lift(action_canonical)
    direct_kick = direct_terms["force"] - u_pi
    exact_kick = exact_terms["force"] - u_pi
    if world_model.F_max > 0:
        direct_kick = (
            world_model.F_max
            * direct_kick
            / (world_model.F_max + direct_kick.norm(dim=-1, keepdim=True))
        )
        exact_kick = (
            world_model.F_max
            * exact_kick
            / (world_model.F_max + exact_kick.norm(dim=-1, keepdim=True))
        )
    curl_F = None
    if world_model.curl_net is not None:
        curl_F = world_model.curl_net(z, action_canonical)
    p_minus_direct = p - (world_model.dt / 2.0) * direct_kick
    p_plus_direct, _ = world_model._boris_rotation(
        p_minus_direct,
        z,
        action_canonical,
        curl_F=curl_F,
    )
    p_minus_exact = p - (world_model.dt / 2.0) * exact_kick
    p_plus_exact, _ = world_model._boris_rotation(
        p_minus_exact,
        z,
        action_canonical,
        curl_F=curl_F,
    )
    f_sol_direct = (p_plus_direct - p_minus_direct) / max(world_model.dt, 1e-8)
    f_sol_exact = (p_plus_exact - p_minus_exact) / max(world_model.dt, 1e-8)
    hodge_direct = world_model.hodge_decomposer(
        direct_terms["force"] + f_sol_direct,
        direct_terms["force"],
        f_sol_direct,
    )
    hodge_exact = world_model.hodge_decomposer(
        exact_terms["force"] + f_sol_exact,
        exact_terms["force"],
        f_sol_exact,
    )
    total_err_sq = _metric_covector_norm_sq(
        world_model.metric,
        z,
        direct_terms["force"] - exact_terms["force"],
    )
    task_err_sq = _metric_covector_norm_sq(
        world_model.metric,
        z,
        direct_terms["task_force"] - exact_terms["task_force"],
    )
    risk_err_sq = _metric_covector_norm_sq(
        world_model.metric,
        z,
        direct_terms["risk_force"] - exact_terms["risk_force"],
    )
    total_exact_sq = _metric_covector_norm_sq(
        world_model.metric, z, exact_terms["force"]
    ).clamp_min(1e-8)
    task_exact_sq = _metric_covector_norm_sq(
        world_model.metric,
        z,
        exact_terms["task_force"],
    ).clamp_min(1e-8)
    risk_exact_sq = _metric_covector_norm_sq(
        world_model.metric,
        z,
        exact_terms["risk_force"],
    ).clamp_min(1e-8)
    return {
        "direct_terms": direct_terms,
        "exact_terms": exact_terms,
        "hodge_direct": hodge_direct,
        "hodge_exact": hodge_exact,
        "force_err_sq": total_err_sq,
        "task_force_err_sq": task_err_sq,
        "risk_force_err_sq": risk_err_sq,
        "force_rel_err": (total_err_sq.sqrt() / total_exact_sq.sqrt()).detach(),
        "task_force_rel_err": (task_err_sq.sqrt() / task_exact_sq.sqrt()).detach(),
        "risk_force_rel_err": (risk_err_sq.sqrt() / risk_exact_sq.sqrt()).detach(),
    }


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


def _critic_stiffness_loss(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    z: torch.Tensor,
    exact_covector: torch.Tensor,
    replay_valid: torch.Tensor,
    stiffness_scale: torch.Tensor | float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep the conservative critic field steep enough to drive control."""
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
    """Align the continuous exact field with the lifted symbolic control field."""
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
    """Align ``dV`` with the discounted local exact increment along replay geodesics."""
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
    """Supervise the exact discounted value increment ``V_t - γ V_{t+1}`` directly."""
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


def _hodge_conservative_preference_losses(
    config: DreamerConfig,
    *,
    hodge_conservative_ratio: torch.Tensor,
    hodge_solenoidal_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Bias dynamics toward conservative explanation once harmonic residue is small."""
    conservative_target = float(config.hodge_conservative_target)
    conservative_deficit = (conservative_target - hodge_conservative_ratio).clamp(min=0.0)
    L_hodge_conservative_margin = conservative_deficit.pow(2).mean()
    L_hodge_solenoidal = hodge_solenoidal_ratio.pow(2).mean()
    metrics = {
        "wm/L_hodge_conservative_margin": float(L_hodge_conservative_margin),
        "wm/L_hodge_solenoidal": float(L_hodge_solenoidal),
        "geometric/hodge_conservative_deficit": float(conservative_deficit.mean()),
        "geometric/hodge_conservative_target": conservative_target,
    }
    return L_hodge_conservative_margin, L_hodge_solenoidal, metrics


def _discounted_return_to_go(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted replay return-to-go until the first terminal step."""
    B, T = rewards.shape
    returns = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(T)):
        running = rewards[:, t] + gamma * running * (1.0 - dones[:, t])
        returns[:, t] = running
    return returns


def _discounted_sum(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """Discounted cumulative reward over a fixed imagined horizon."""
    horizon = rewards.shape[1]
    if horizon == 0:
        return rewards.sum(dim=1)
    exponents = torch.arange(horizon, device=rewards.device, dtype=rewards.dtype)
    discounts = torch.pow(rewards.new_full((), gamma), exponents).unsqueeze(0)
    return (rewards * discounts).sum(dim=1)


def _multistep_horizon_ladder(max_horizon: int) -> list[int]:
    """Return a logarithmic horizon ladder with the final horizon included."""
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
    """Build discounted k-step targets and masks from one-step conservative targets."""
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


def _target_normalization_scale(
    targets: list[torch.Tensor],
    masks: list[torch.Tensor],
    *,
    quantile: float,
    min_scale: float,
    template: torch.Tensor,
) -> torch.Tensor:
    """Estimate a robust target scale for normalized exact-field training."""
    if not targets:
        return template.new_tensor(max(float(min_scale), 1e-8))
    stacked_targets = torch.cat([target.abs().reshape(-1) for target in targets])
    stacked_masks = torch.cat([mask.reshape(-1) for mask in masks])
    return _masked_quantile(stacked_targets, stacked_masks, float(quantile)).clamp_min(
        max(float(min_scale), 1e-8),
    )


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
    """Supervise exact discounted value increments over multiple horizons."""
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
    """Align ``dV`` with multi-step exact increments along replay or policy geodesics."""
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


def _collect_policy_state_rollout(
    obs_model: TopoEncoder,
    world_model: GeometricWorldModel,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
    *,
    hard_routing: bool,
    hard_routing_tau: float,
) -> dict[str, torch.Tensor]:
    """Roll out policy-visited latent states without reward diagnostics."""
    z = z_0
    rw = rw_0
    p = world_model.momentum_init(z_0)
    z_seq = [z.detach()]
    rw_seq = [rw.detach()]
    action_state_prob_seq: list[torch.Tensor] = []
    action_chart_prob_seq: list[torch.Tensor] = []
    action_canonical_seq: list[torch.Tensor] = []
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)
    with torch.no_grad():
        for _ in range(horizon):
            obs_info = symbolize_latent_with_atlas(
                obs_model,
                z,
                router_weights_override=rw,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action_out = _policy_action(
                actor,
                action_model,
                obs_info,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
            )
            action_symbolic = _action_symbolic_distribution_from_actor(action_out)
            action_state_prob_seq.append(action_symbolic["state_probs"].detach())
            action_chart_prob_seq.append(action_symbolic["chart_probs"].detach())
            action_canonical_seq.append(action_out["action_canonical"].detach())
            step_out = world_model._rollout_transition(
                z,
                p,
                action_out["action_canonical"],
                rw,
                track_energy=False,
            )
            z = step_out["z"]
            p = step_out["p"]
            rw = step_out["rw"]
            z_seq.append(z.detach())
            rw_seq.append(rw.detach())
    return {
        "z_seq": torch.stack(z_seq, dim=1),
        "rw_seq": torch.stack(rw_seq, dim=1),
        "action_state_probs": torch.stack(action_state_prob_seq, dim=1)
        if action_state_prob_seq
        else z_0.new_zeros(
            z_0.shape[0],
            0,
            actor.num_action_charts * actor.action_codes_per_chart,
        ),
        "action_chart_probs": torch.stack(action_chart_prob_seq, dim=1)
        if action_chart_prob_seq
        else z_0.new_zeros(z_0.shape[0], 0, actor.num_action_charts),
        "action_canonical_seq": torch.stack(action_canonical_seq, dim=1)
        if action_canonical_seq
        else z_0.new_zeros(z_0.shape[0], 0, z_0.shape[-1]),
    }


def _actor_return_trust(
    config: DreamerConfig,
    *,
    chart_acc: float,
    force_rel_err: float,
    policy_sync_err: float,
    hodge_conservative: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify how much imagined return the actor is allowed to trust."""
    chance = 1.0 / max(int(config.num_charts), 1)
    chart_target = max(float(config.actor_return_chart_acc_target), chance + 1e-6)
    chart_acc_t = template.new_tensor(chart_acc)
    force_rel_err_t = template.new_tensor(force_rel_err)
    policy_sync_err_t = template.new_tensor(policy_sync_err)
    hodge_conservative_t = template.new_tensor(hodge_conservative)

    chart_factor = ((chart_acc_t - chance) / (chart_target - chance)).clamp(0.0, 1.0)
    force_factor = torch.exp(
        -float(config.actor_return_force_err_scale) * force_rel_err_t.clamp(min=0.0),
    )
    policy_sync_factor = torch.exp(
        -float(config.actor_return_policy_sync_scale) * policy_sync_err_t.clamp(min=0.0),
    )
    conservative_factor = hodge_conservative_t.clamp(0.0, 1.0)
    trust = (chart_factor * force_factor * policy_sync_factor * conservative_factor).clamp(
        0.0, 1.0
    )
    trust_metrics = {
        "actor/return_trust": float(trust.detach()),
        "actor/return_trust_chart": float(chart_factor.detach()),
        "actor/return_trust_force": float(force_factor.detach()),
        "actor/return_trust_sync": float(policy_sync_factor.detach()),
        "actor/return_trust_conservative_exact": float(conservative_factor.detach()),
    }
    return trust, trust_metrics


def _categorical_entropy_varentropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return categorical entropy and varentropy from logits."""
    probs = F.softmax(logits, dim=-1)
    surprisal = -probs.clamp_min(1e-8).log()
    entropy = (probs * surprisal).sum(dim=-1)
    varentropy = (probs * (surprisal - entropy.unsqueeze(-1)).pow(2)).sum(dim=-1)
    return entropy, varentropy


def _actor_curiosity_closure_gate(
    config: DreamerConfig,
    *,
    obs_state_acc: float,
    enclosure_defect_acc: float,
    enclosure_defect_ce: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gate curiosity by grounded symbolic closure rather than raw novelty."""
    chance = 1.0 / max(int(config.num_charts), 1)
    acc_target = max(float(config.actor_curiosity_closure_acc_target), chance + 1e-6)
    obs_state_acc_t = template.new_tensor(obs_state_acc)
    enclosure_defect_acc_t = template.new_tensor(enclosure_defect_acc)
    enclosure_defect_ce_t = template.new_tensor(enclosure_defect_ce)

    obs_factor = ((obs_state_acc_t - chance) / (acc_target - chance)).clamp(0.0, 1.0)
    defect_acc_factor = torch.exp(
        -float(config.actor_curiosity_enclosure_defect_acc_scale)
        * enclosure_defect_acc_t.clamp(min=0.0),
    )
    defect_ce_factor = torch.exp(
        -float(config.actor_curiosity_enclosure_defect_ce_scale)
        * enclosure_defect_ce_t.clamp(min=0.0),
    )
    closure_gate = (obs_factor * defect_acc_factor * defect_ce_factor).clamp(0.0, 1.0)
    metrics = {
        "actor/curiosity_closure_gate": float(closure_gate.detach()),
        "actor/curiosity_closure_obs_factor": float(obs_factor.detach()),
        "actor/curiosity_closure_defect_acc_factor": float(defect_acc_factor.detach()),
        "actor/curiosity_closure_defect_ce_factor": float(defect_ce_factor.detach()),
    }
    return closure_gate, metrics


def _exact_control_gate(
    config: DreamerConfig,
    *,
    exact_increment_abs_err: float,
    exact_increment_target_mean: float,
    on_policy_covector_align_abs_err: float,
    on_policy_covector_target_mean: float,
    on_policy_exact_covector_norm_mean: float,
    on_policy_stiffness_target: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify whether the exact conservative field is calibrated enough to drive control."""
    exact_increment_rel = exact_increment_abs_err / max(abs(exact_increment_target_mean), 1e-6)
    on_policy_covector_rel = on_policy_covector_align_abs_err / max(
        abs(on_policy_covector_target_mean),
        1e-6,
    )
    calibration_ratio = on_policy_exact_covector_norm_mean / max(on_policy_stiffness_target, 1e-6)

    exact_increment_rel_t = template.new_tensor(exact_increment_rel)
    on_policy_covector_rel_t = template.new_tensor(on_policy_covector_rel)
    calibration_ratio_t = template.new_tensor(calibration_ratio)

    exact_increment_factor = torch.exp(
        -float(config.actor_return_exact_increment_rel_scale)
        * exact_increment_rel_t.clamp(min=0.0),
    )
    on_policy_covector_factor = torch.exp(
        -float(config.actor_return_exact_covector_rel_scale)
        * on_policy_covector_rel_t.clamp(min=0.0),
    )
    calibration_factor = calibration_ratio_t.clamp(0.0, 1.0)
    exact_control_gate = (
        exact_increment_factor * on_policy_covector_factor * calibration_factor
    ).clamp(0.0, 1.0)
    metrics = {
        "actor/exact_control_gate": float(exact_control_gate.detach()),
        "actor/exact_control_increment_rel": float(exact_increment_rel_t.detach()),
        "actor/exact_control_covector_rel": float(on_policy_covector_rel_t.detach()),
        "actor/exact_control_calibration_ratio": float(calibration_ratio_t.detach()),
        "actor/exact_control_increment_factor": float(exact_increment_factor.detach()),
        "actor/exact_control_covector_factor": float(on_policy_covector_factor.detach()),
        "actor/exact_control_calibration_factor": float(calibration_factor.detach()),
    }
    return exact_control_gate, metrics


def _macro_control_gate(
    config: DreamerConfig,
    *,
    macro_exact_increment_abs_err: float,
    macro_exact_increment_target_mean: float,
    macro_on_policy_pullback_abs_err: float,
    macro_on_policy_value_std: float,
    macro_on_policy_exact_increment_pred_abs_mean: float,
    macro_target_scale: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify whether the symbolic macro value is calibrated enough to stage actor RL."""
    if (
        max(
            float(config.w_macro_value),
            float(config.w_macro_exact_increment),
            float(config.w_macro_pullback),
            float(config.w_macro_on_policy_pullback),
        )
        <= 0.0
    ):
        one = template.new_ones(())
        metrics = {
            "actor/macro_control_gate": 1.0,
            "actor/macro_control_increment_rel": 0.0,
            "actor/macro_control_pullback_rel": 0.0,
            "actor/macro_control_calibration_ratio": 1.0,
            "actor/macro_control_signal_scale": 1.0,
            "actor/macro_control_increment_factor": 1.0,
            "actor/macro_control_pullback_factor": 1.0,
            "actor/macro_control_calibration_factor": 1.0,
        }
        return one, metrics
    macro_increment_rel = macro_exact_increment_abs_err / max(
        abs(macro_exact_increment_target_mean), 1e-6
    )
    macro_pullback_rel = macro_on_policy_pullback_abs_err / max(abs(macro_target_scale), 1e-6)
    macro_signal_scale = max(
        abs(macro_on_policy_exact_increment_pred_abs_mean),
        abs(macro_on_policy_value_std),
    )
    macro_calibration_ratio = macro_signal_scale / max(abs(macro_target_scale), 1e-6)

    macro_increment_rel_t = template.new_tensor(macro_increment_rel)
    macro_pullback_rel_t = template.new_tensor(macro_pullback_rel)
    macro_calibration_ratio_t = template.new_tensor(macro_calibration_ratio)

    macro_increment_factor = torch.exp(
        -float(config.actor_return_macro_increment_rel_scale)
        * macro_increment_rel_t.clamp(min=0.0),
    )
    macro_pullback_factor = torch.exp(
        -float(config.actor_return_macro_pullback_rel_scale) * macro_pullback_rel_t.clamp(min=0.0),
    )
    macro_calibration_factor = macro_calibration_ratio_t.clamp(0.0, 1.0)
    macro_control_gate = (
        macro_increment_factor * macro_pullback_factor * macro_calibration_factor
    ).clamp(0.0, 1.0)
    metrics = {
        "actor/macro_control_gate": float(macro_control_gate.detach()),
        "actor/macro_control_increment_rel": float(macro_increment_rel_t.detach()),
        "actor/macro_control_pullback_rel": float(macro_pullback_rel_t.detach()),
        "actor/macro_control_calibration_ratio": float(macro_calibration_ratio_t.detach()),
        "actor/macro_control_signal_scale": float(macro_signal_scale),
        "actor/macro_control_increment_factor": float(macro_increment_factor.detach()),
        "actor/macro_control_pullback_factor": float(macro_pullback_factor.detach()),
        "actor/macro_control_calibration_factor": float(macro_calibration_factor.detach()),
    }
    return macro_control_gate, metrics


def _actor_supervise_scale(
    config: DreamerConfig,
    *,
    epoch: int,
    actor_return_gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Schedule replay-action supervision as a bootstrap term, not a permanent target."""
    gate = actor_return_gate.detach().clamp(0.0, 1.0)
    min_scale = float(np.clip(config.actor_supervise_min_scale, 0.0, 1.0))
    warmup_epochs = max(int(config.actor_supervise_warmup_epochs), 0)
    decay_epochs = max(int(config.actor_supervise_decay_epochs), 0)
    if epoch < warmup_epochs:
        warmup_scale = 1.0
    elif decay_epochs <= 0:
        warmup_scale = min_scale
    else:
        decay_progress = min(max((epoch - warmup_epochs + 1) / decay_epochs, 0.0), 1.0)
        warmup_scale = 1.0 - (1.0 - min_scale) * decay_progress
    gate_scale = float(1.0 - gate)
    scale = max(min_scale, warmup_scale * gate_scale)
    scale_t = gate.new_tensor(scale)
    metrics = {
        "actor/supervise_scale": float(scale_t.detach()),
        "actor/supervise_warmup_scale": float(warmup_scale),
        "actor/supervise_gate_scale": gate_scale,
    }
    return scale_t, metrics


def _scheduled_sigma_motor(
    config: DreamerConfig,
    *,
    epoch: int,
    exact_control_gate: float,
) -> tuple[float, dict[str, float]]:
    """Schedule thermal motor exploration with optional exact-field-aware cooling."""
    sigma_final = max(float(config.sigma_motor), 0.0)
    sigma_init = (
        float(config.sigma_motor_init) if float(config.sigma_motor_init) > 0.0 else sigma_final
    )
    anneal_epochs = max(int(config.sigma_motor_anneal_epochs), 0)
    epoch_progress = 1.0 if anneal_epochs <= 0 else min(max((epoch + 1) / anneal_epochs, 0.0), 1.0)
    exact_target = max(float(config.sigma_motor_exact_gate_target), 0.0)
    exact_progress = (
        1.0 if exact_target <= 0.0 else min(max(exact_control_gate / exact_target, 0.0), 1.0)
    )
    cooling_progress = min(epoch_progress, exact_progress)
    sigma = sigma_init + (sigma_final - sigma_init) * cooling_progress
    metrics = {
        "policy/sigma_motor": float(sigma),
        "policy/sigma_motor_init": float(sigma_init),
        "policy/sigma_motor_final": float(sigma_final),
        "policy/sigma_motor_epoch_progress": float(epoch_progress),
        "policy/sigma_motor_exact_progress": float(exact_progress),
    }
    return float(sigma), metrics


def _actor_old_policy_kl_losses(
    actor_out: dict[str, torch.Tensor],
    actor_old_out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize large discrete policy changes relative to the previous actor."""
    old_chart_probs = F.softmax(actor_old_out["action_chart_logits"].detach(), dim=-1)
    new_chart_log_probs = F.log_softmax(actor_out["action_chart_logits"], dim=-1)
    L_chart_kl = F.kl_div(new_chart_log_probs, old_chart_probs, reduction="batchmean")

    old_code_probs = F.softmax(actor_old_out["action_code_logits"].detach(), dim=-1)
    new_code_log_probs = F.log_softmax(actor_out["action_code_logits"], dim=-1)
    code_kl_per_chart = (
        old_code_probs * (old_code_probs.clamp_min(1e-8).log() - new_code_log_probs)
    ).sum(dim=-1)
    L_code_kl = (old_chart_probs * code_kl_per_chart).sum(dim=-1).mean()
    return L_chart_kl, L_code_kl


def _actor_state_metric(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    state_z_geo: torch.Tensor,
    actor_out: dict[str, torch.Tensor],
    obs_z_n: torch.Tensor,
    target_chart_idx: torch.Tensor,
    target_code_idx: torch.Tensor,
    exact_covector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build the diagonal state-space metric proxy used by the actor update."""
    sample_idx = torch.arange(obs_z_n.shape[0], device=obs_z_n.device)
    chart_log_probs = F.log_softmax(actor_out["action_chart_logits"], dim=-1)
    chart_logp = chart_log_probs[sample_idx, target_chart_idx.long()]
    code_logits = actor_out["action_code_logits"][sample_idx, target_chart_idx.long()]
    code_log_probs = F.log_softmax(code_logits, dim=-1)
    code_logp = code_log_probs[sample_idx, target_code_idx.long()]
    log_prob = chart_logp + code_logp
    state_score = torch.autograd.grad(
        log_prob.sum(),
        obs_z_n,
        retain_graph=True,
        create_graph=False,
    )[0]
    metric_scale = _metric_inverse_scale(metric, state_z_geo.detach()).mean()
    fisher_diag = metric_scale * state_score.detach().pow(2).mean(dim=0)
    fisher_scale = max(float(config.actor_metric_fisher_scale), 0.0)
    fisher_diag_scaled = fisher_scale * fisher_diag
    value_diag = metric_scale * exact_covector.detach().pow(2).mean(dim=0)
    metric_diag = value_diag + fisher_diag_scaled + float(config.actor_metric_epsilon)
    metric_inv = metric_diag.reciprocal()
    alpha = value_diag.mean().sqrt()
    beta_pi_raw = fisher_diag.mean().sqrt()
    beta_pi = fisher_diag_scaled.mean().sqrt()
    scale_barrier = (beta_pi - alpha).clamp(min=0.0) / (beta_pi + 1e-8)
    scale_trust = torch.exp(-scale_barrier)
    scale_certified = bool(float(alpha.detach()) > float(beta_pi.detach()))
    metrics = {
        "actor/state_alpha": float(alpha.detach()),
        "actor/state_beta_pi": float(beta_pi.detach()),
        "actor/state_beta_pi_raw": float(beta_pi_raw.detach()),
        "actor/state_scale_barrier": float(scale_barrier.detach()),
        "actor/state_scale_trust": float(scale_trust.detach()),
        "actor/state_metric_mean": float(metric_diag.mean().detach()),
        "actor/state_metric_inv_mean": float(metric_inv.mean().detach()),
        "actor/state_scale_certified": 1.0 if scale_certified else 0.0,
    }
    return (
        metric_diag.detach(),
        metric_inv.detach(),
        scale_certified,
        scale_trust.detach(),
        scale_barrier.detach(),
        metrics,
    )


def _relative_trust_region_scale(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    *,
    kappa: float,
    epsilon_theta: float,
) -> tuple[float, float, float, float]:
    """Apply the parameter-space Mach limit to the current gradients."""
    grads = [param for param in parameters if param.grad is not None]
    if not grads or kappa <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    grad_norm_t = torch.norm(torch.stack([param.grad.detach().norm() for param in grads]), p=2)
    param_norm_t = torch.norm(torch.stack([param.detach().norm() for param in grads]), p=2)
    base_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    step_norm_t = grad_norm_t * base_lr
    max_step_t = float(kappa) * (param_norm_t + float(epsilon_theta))
    scale = min(1.0, float((max_step_t / (step_norm_t + 1e-12)).detach()))
    if scale < 1.0:
        for param in grads:
            param.grad.mul_(scale)
    return (
        scale,
        float(param_norm_t.detach()),
        float(step_norm_t.detach()),
        float(max_step_t.detach()),
    )


def _world_model_closure_losses(
    config: DreamerConfig,
    world_model: GeometricWorldModel,
    enclosure_probe: EnclosureProbe,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    chart_embed_t: torch.Tensor,
    z_tex_t: torch.Tensor,
    action_canonicals: torch.Tensor,
    code_t: torch.Tensor,
    target_charts: torch.Tensor,
    target_codes: torch.Tensor,
    update_idx: int,
    *,
    zeno_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Measure closure and router smoothness from the observation Markov model."""
    zero = z_0.new_zeros(())
    if action_canonicals.shape[1] == 0 or target_charts.shape[1] == 0:
        return (
            zero,
            zero,
            zero,
            zero,
            {
                "closure/obs_state_acc": 0.0,
                "closure/obs_symbol_acc": 0.0,
                "closure/chart_entropy": 0.0,
                "closure/enclosure_acc_full": 0.0,
                "closure/enclosure_acc_base": 0.0,
                "closure/enclosure_defect_acc": 0.0,
                "closure/enclosure_defect_ce": 0.0,
                "closure/grl_alpha": 0.0,
            },
        )

    wm_out = world_model(z_0, action_canonicals, rw_0)
    chart_logits_all = wm_out["chart_logits"]
    T_sync = min(chart_logits_all.shape[1], target_charts.shape[1])
    if T_sync == 0:
        return (
            zero,
            zero,
            zero,
            zero,
            {
                "closure/obs_state_acc": 0.0,
                "closure/obs_symbol_acc": 0.0,
                "closure/chart_entropy": 0.0,
                "closure/enclosure_acc_full": 0.0,
                "closure/enclosure_acc_base": 0.0,
                "closure/enclosure_defect_acc": 0.0,
                "closure/enclosure_defect_ce": 0.0,
                "closure/grl_alpha": 0.0,
            },
        )

    chart_logits = chart_logits_all[:, :T_sync]
    target = target_charts[:, :T_sync].detach()
    chart_probs = F.softmax(chart_logits, dim=-1)
    prev_rw = torch.cat([rw_0.unsqueeze(1), chart_probs[:, :-1]], dim=1)
    L_closure_obs = compute_dynamics_chart_loss(chart_logits, target)
    L_obs_zeno = zeno_loss(
        chart_probs.reshape(-1, chart_probs.shape[-1]),
        prev_rw.reshape(-1, prev_rw.shape[-1]),
        mode=zeno_mode,
    )
    grl_alpha = grl_alpha_schedule(
        update_idx,
        warmup_steps=config.enclosure_grl_warmup_updates,
        max_alpha=config.enclosure_grl_alpha_max,
    )
    enclosure_probe.grl.alpha.copy_(enclosure_probe.grl.alpha.new_tensor(grl_alpha))
    L_enclosure, L_enclosure_probe, enclosure_metrics = compute_enclosure_loss(
        enclosure_probe,
        chart_embed_t[:, :T_sync].reshape(-1, chart_embed_t.shape[-1]),
        action_canonicals[:, :T_sync].reshape(-1, action_canonicals.shape[-1]),
        z_tex_t[:, :T_sync].reshape(-1, z_tex_t.shape[-1]),
        target_charts[:, :T_sync].reshape(-1),
        K_code_t=code_t[:, :T_sync].reshape(-1),
        K_code_tp1=target_codes[:, :T_sync].reshape(-1),
        codes_per_chart=config.codes_per_chart,
    )
    metrics = {
        "closure/obs_state_acc": float((chart_logits.argmax(dim=-1) == target).float().mean()),
        "closure/obs_symbol_acc": enclosure_metrics["acc_full"],
        "closure/chart_entropy": float(
            -(chart_probs * chart_probs.clamp(min=1e-8).log()).sum(dim=-1).mean(),
        ),
        "closure/enclosure_acc_full": enclosure_metrics["acc_full"],
        "closure/enclosure_acc_base": enclosure_metrics["acc_base"],
        "closure/enclosure_defect_acc": enclosure_metrics["defect_acc"],
        "closure/enclosure_defect_ce": enclosure_metrics["defect_ce"],
        "closure/grl_alpha": grl_alpha,
    }
    return L_closure_obs, L_obs_zeno, L_enclosure, L_enclosure_probe, metrics


def _should_run_actor_update(
    config: DreamerConfig,
    *,
    epoch: int,
    update_idx: int,
) -> bool:
    """Return whether the imagined-return actor step should run."""
    if (
        max(
            float(config.w_actor_return),
            float(config.w_actor_curiosity),
        )
        <= 0.0
    ):
        return False
    if config.actor_return_update_every <= 0:
        return False
    if config.actor_return_horizon <= 0:
        return False
    if epoch < config.actor_return_warmup_epochs:
        return False
    return update_idx % config.actor_return_update_every == 0


def _imagine_actor_return(
    config: DreamerConfig | None,
    obs_model: TopoEncoder,
    world_model: GeometricWorldModel,
    reward_head: RewardHead,
    critic: nn.Module,
    macro_critic: MacroValueModel | None,
    actor: GeometricActor,
    action_model: SharedDynTopoEncoder,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    horizon: int,
    gamma: float,
    *,
    hard_routing: bool = True,
    hard_routing_tau: float = -1.0,
    reward_curl_batch_limit: int | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiate the actor objective through canonical-action imagination."""
    z = z_0
    rw = rw_0
    p = world_model.momentum_init(z_0)
    routing_tau = _rollout_routing_tau(hard_routing, hard_routing_tau)

    z_state_list: list[torch.Tensor] = []
    z_traj_list: list[torch.Tensor] = []
    reward_list: list[torch.Tensor] = []
    reward_cons_list: list[torch.Tensor] = []
    reward_noncons_list: list[torch.Tensor] = []
    reward_form_cov_list: list[torch.Tensor] = []
    exact_covector_list: list[torch.Tensor] = []
    macro_covector_list: list[torch.Tensor] = []
    macro_value_list: list[torch.Tensor] = []
    macro_reward_list: list[torch.Tensor] = []
    action_list: list[torch.Tensor] = []
    action_canonical_list: list[torch.Tensor] = []
    rw_state_list: list[torch.Tensor] = []
    rw_traj_list: list[torch.Tensor] = []
    chart_acc_list: list[torch.Tensor] = []
    chart_ce_list: list[torch.Tensor] = []
    router_sync_list: list[torch.Tensor] = []
    force_rel_err_list: list[torch.Tensor] = []
    hodge_cons_list: list[torch.Tensor] = []
    reward_noncons_gate_list: list[torch.Tensor] = []
    policy_sync_loss_list: list[torch.Tensor] = []
    curiosity_list: list[torch.Tensor] = []
    curiosity_entropy_list: list[torch.Tensor] = []
    curiosity_varentropy_list: list[torch.Tensor] = []

    for _ in range(horizon):
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            router_weights_override=rw,
            hard_routing=hard_routing,
            hard_routing_tau=routing_tau,
        )
        action_state = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=hard_routing,
            hard_routing_tau=routing_tau,
        )
        action_mean, _, _ = action_model.decoder(
            action_state["action_z_geo"],
            router_weights=action_state["action_router_weights"],
            routing_tau=routing_tau,
        )
        z_state_list.append(z)
        rw_state_list.append(rw.detach())
        force_diag = _conservative_force_diagnostics(
            world_model,
            z.detach(),
            p.detach(),
            rw.detach(),
            action_state["action_z_geo"].detach(),
        )
        step_out = world_model._rollout_transition(
            z,
            p,
            action_state["action_z_geo"],
            rw,
            track_energy=False,
        )
        z_next = step_out["z"]
        rw_next = step_out["rw"]
        reward_conservative, _v_curr, _v_next = _conservative_reward_from_value(
            critic,
            z,
            rw,
            z_next,
            rw_next,
            gamma,
        )
        exact_covector = _value_covector_from_critic(critic, z, rw)
        if macro_critic is not None:
            action_symbolic = _action_symbolic_distribution_from_actor(action_state)
            macro_pullback = _macro_control_pullback_covector(
                obs_model,
                world_model,
                macro_critic,
                actor,
                z,
                action_state_probs=action_symbolic["state_probs"].detach(),
                action_canonical=action_state["action_z_geo"],
                gamma=gamma,
                continuation=reward_conservative.new_ones(reward_conservative.shape),
                router_weights_override=rw,
                hard_routing=hard_routing,
                hard_routing_tau=routing_tau,
                create_graph=False,
                detach=True,
            )
            macro_covector = macro_pullback["macro_covector"]
            macro_value = macro_pullback["macro_control"]
            macro_reward = macro_pullback["macro_reward"].squeeze(-1)
        else:
            macro_value = reward_conservative.new_zeros(reward_conservative.shape)
            macro_covector = exact_covector.new_zeros(exact_covector.shape)
            macro_reward = reward_conservative.squeeze(-1).new_zeros(
                reward_conservative.squeeze(-1).shape,
            )
        reward_info = reward_head.decompose(
            z,
            rw,
            action_state["action_z_geo"],
            action_state["action_router_weights"],
            action_state["action_z_q"],
            action_canonical=action_state["action_z_geo"],
            exact_covector=exact_covector,
            compute_curl=False,
            curl_batch_limit=reward_curl_batch_limit,
        )
        reward_nonconservative_raw = reward_info["reward_nonconservative"].squeeze(-1)
        reward_nonconservative_gate = reward_nonconservative_raw.new_ones(())
        if config is not None:
            reward_nonconservative_gate, _ = _reward_nonconservative_gate(
                config,
                exact_covector_norm_mean=_metric_covector_norm_sq(
                    world_model.metric,
                    z.detach(),
                    exact_covector.detach(),
                )
                .sqrt()
                .mean(),
                force_rel_err_mean=force_diag["force_rel_err"].mean(),
            )
        reward_nonconservative = reward_nonconservative_gate.detach() * reward_nonconservative_raw
        reward_total = reward_conservative.squeeze(-1) + reward_nonconservative
        next_obs_info = symbolize_latent_with_atlas(
            obs_model,
            z_next.detach(),
            router_weights_override=rw_next.detach(),
            hard_routing=hard_routing,
            hard_routing_tau=routing_tau,
        )
        chart_target = next_obs_info["chart_idx"].detach().long()
        chart_logits = step_out["chart_logits"]
        chart_entropy, chart_varentropy = _categorical_entropy_varentropy(chart_logits)
        max_chart_entropy = max(float(np.log(max(chart_logits.shape[-1], 1))), 1e-6)
        curiosity_step = (chart_varentropy / (max_chart_entropy**2)).clamp(min=0.0)
        chart_acc_list.append((chart_logits.argmax(dim=-1) == chart_target).float())
        chart_ce_list.append(F.cross_entropy(chart_logits, chart_target, reduction="none"))
        router_sync = (
            (rw_next.detach() - next_obs_info["router_weights"].detach()).abs().mean(dim=-1)
        )
        router_sync_list.append(router_sync)
        force_rel_err_list.append(force_diag["force_rel_err"])
        hodge_cons_list.append(force_diag["hodge_exact"]["conservative_ratio"])
        reward_noncons_gate_list.append(
            reward_nonconservative_gate.expand_as(reward_nonconservative)
        )
        policy_sync_loss_list.append(chart_ce_list[-1] + router_sync)
        curiosity_list.append(curiosity_step)
        curiosity_entropy_list.append(chart_entropy / max_chart_entropy)
        curiosity_varentropy_list.append(curiosity_step)
        reward_list.append(reward_total)
        reward_cons_list.append(reward_conservative.squeeze(-1))
        reward_noncons_list.append(reward_nonconservative)
        reward_form_cov_list.append(reward_info["reward_form_cov"])
        exact_covector_list.append(exact_covector)
        macro_covector_list.append(macro_covector)
        macro_value_list.append(macro_value.squeeze(-1))
        macro_reward_list.append(macro_reward)
        action_list.append(action_mean)
        action_canonical_list.append(action_state["action_z_geo"])
        z = z_next
        p = step_out["p"]
        rw = rw_next
        z_traj_list.append(z)
        rw_traj_list.append(rw.detach())

    rewards = torch.stack(reward_list, dim=1)
    reward_conservative = torch.stack(reward_cons_list, dim=1)
    reward_nonconservative = torch.stack(reward_noncons_list, dim=1)
    reward_macro = torch.stack(macro_reward_list, dim=1)
    curiosity = torch.stack(curiosity_list, dim=1)
    objective_conservative = _discounted_sum(reward_conservative, gamma)
    objective_nonconservative = _discounted_sum(reward_nonconservative, gamma)
    objective_macro = _discounted_sum(reward_macro, gamma)
    objective_curiosity = _discounted_sum(curiosity, gamma)
    objective = objective_conservative + objective_nonconservative
    return {
        "z_states": torch.stack(z_state_list, dim=1),
        "z_traj": torch.stack(z_traj_list, dim=1),
        "rewards": rewards,
        "reward_conservative": reward_conservative,
        "reward_nonconservative": reward_nonconservative,
        "reward_macro": reward_macro,
        "curiosity": curiosity,
        "curiosity_entropy": torch.stack(curiosity_entropy_list, dim=1),
        "curiosity_varentropy": torch.stack(curiosity_varentropy_list, dim=1),
        "reward_form_cov": torch.stack(reward_form_cov_list, dim=1),
        "exact_covector": torch.stack(exact_covector_list, dim=1),
        "macro_covector": torch.stack(macro_covector_list, dim=1),
        "macro_value": torch.stack(macro_value_list, dim=1),
        "objective": objective,
        "objective_conservative": objective_conservative,
        "objective_nonconservative": objective_nonconservative,
        "objective_macro": objective_macro,
        "objective_curiosity": objective_curiosity,
        "actions": torch.stack(action_list, dim=1),
        "action_canonicals": torch.stack(action_canonical_list, dim=1),
        "rw_states": torch.stack(rw_state_list, dim=1),
        "rw_traj": torch.stack(rw_traj_list, dim=1),
        "policy_chart_acc": torch.stack(chart_acc_list, dim=1),
        "policy_chart_ce": torch.stack(chart_ce_list, dim=1),
        "policy_router_sync": torch.stack(router_sync_list, dim=1),
        "policy_force_rel_err": torch.stack(force_rel_err_list, dim=1),
        "policy_hodge_conservative_exact": torch.stack(hodge_cons_list, dim=1),
        "policy_reward_nonconservative_gate": torch.stack(reward_noncons_gate_list, dim=1),
        "policy_sync_loss": torch.stack(policy_sync_loss_list, dim=1),
    }


def _value_calibration_error(
    values: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    num_bins: int = 8,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Compute an ECE-style calibration error for scalar values."""
    valid = mask.bool()
    if not valid.any():
        zero = values.new_zeros(())
        return zero, zero, 0.0

    v = values[valid]
    t = targets[valid]
    v_min = v.min()
    v_max = v.max()
    if (v_max - v_min).abs() < 1e-8:
        gap = (v.mean() - t.mean()).abs()
        return gap, gap, 1.0

    edges = torch.linspace(v_min, v_max, num_bins + 1, device=v.device, dtype=v.dtype)
    cal_err = values.new_zeros(())
    cal_max = values.new_zeros(())
    nonempty = 0
    for idx in range(num_bins):
        lower = edges[idx]
        upper = edges[idx + 1]
        if idx == num_bins - 1:
            in_bin = (v >= lower) & (v <= upper)
        else:
            in_bin = (v >= lower) & (v < upper)
        if not in_bin.any():
            continue
        gap = (v[in_bin].mean() - t[in_bin].mean()).abs()
        cal_err = cal_err + in_bin.float().mean() * gap
        cal_max = torch.maximum(cal_max, gap)
        nonempty += 1

    return cal_err, cal_max, float(nonempty)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def _train_step(
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    action_model: TopoEncoder,
    action_jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel,
    enclosure_probe: EnclosureProbe,
    reward_head: RewardHead,
    critic: nn.Module,
    macro_critic: MacroValueModel,
    actor: GeometricActor,
    actor_old: GeometricActor,
    optimizer_enc: torch.optim.Optimizer,
    optimizer_wm: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer | None,
    optimizer_boundary: torch.optim.Optimizer,
    optimizer_enclosure: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: DreamerConfig,
    phase1_cfg: VLAConfig,
    action_phase1_cfg: VLAConfig,
    epoch: int,
    current_hard_routing: bool,
    current_tau: float,
    update_idx: int = 0,
    obs_normalizer: ObservationNormalizer | None = None,
    compute_diagnostics: bool = True,
    optimizer_macro: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    """One training iteration for the two-manifold Dreamer."""
    metrics: dict[str, float] = {}
    step_t0 = time.perf_counter()
    _mark_compile_step_begin()

    obs_raw = batch["obs"]
    obs = obs_normalizer.normalize_tensor(obs_raw) if obs_normalizer is not None else obs_raw
    action_seq = batch.get("action_means", batch["actions"])
    actions = action_seq[:, :-1]
    rewards = batch["rewards"][:, :-1]
    replay_dones = batch["dones"][:, :-1]
    B, T, _A = actions.shape
    H_wm = min(T, max(1, int(config.wm_prediction_horizon), int(config.imagination_horizon)))
    shared_critic = _shared_value_field(world_model, critic)
    replay_valid = _transition_valid_mask(replay_dones)

    t_section = time.perf_counter()
    flat_obs = obs.reshape(B * (T + 1), -1)
    flat_actions = action_seq.reshape(B * (T + 1), -1)

    def _encode_sequence(
        x: torch.Tensor,
        topo_model: TopoEncoder,
        topo_jump_op: FactorizedJumpOperator,
        phase_cfg: VLAConfig,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, float],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        (
            base_loss,
            zn_reg_loss,
            enc_metrics,
            z_geo_flat,
            enc_w_flat,
            K_ch_flat,
            z_n_flat,
            z_tex_flat,
            c_bar_flat,
            K_code_flat,
            z_q_flat,
            v_local_flat,
        ) = _compute_encoder_losses(
            x,
            topo_model,
            topo_jump_op,
            config,
            epoch,
            routing_tau=current_tau,
            phase1_config=phase_cfg,
        )
        return (
            base_loss,
            zn_reg_loss,
            enc_metrics,
            z_geo_flat.reshape(B, T + 1, -1),
            enc_w_flat.reshape(B, T + 1, -1),
            K_ch_flat.reshape(B, T + 1),
            K_code_flat.reshape(B, T + 1),
            z_n_flat.reshape(B, T + 1, -1),
            z_tex_flat.reshape(B, T + 1, -1),
            c_bar_flat.reshape(B, T + 1, -1),
            z_q_flat.reshape(B, T + 1, -1),
            v_local_flat.reshape(B, T + 1, -1),
        )

    optimizer_enc.zero_grad()
    optimizer_enclosure.zero_grad()
    macro_optimizer = optimizer_macro if optimizer_macro is not None else optimizer_wm
    macro_optimizer.zero_grad()

    encode_context = torch.no_grad() if config.freeze_encoder else contextlib.nullcontext()
    with encode_context:
        (
            base_loss_obs,
            zn_reg_loss_obs,
            enc_metrics_obs,
            z_all,
            rw_all,
            K_all,
            K_code_all,
            zn_all,
            z_tex_all,
            c_bar_all,
            _z_q_all,
            _v_local_all,
        ) = _encode_sequence(flat_obs, model, jump_op, phase1_cfg)
        (
            base_loss_action,
            zn_reg_loss_action,
            enc_metrics_action,
            action_z_all,
            action_rw_all,
            action_K_all,
            action_K_code_all,
            action_zn_all,
            _action_z_tex_all,
            _action_c_bar_all,
            action_z_q_all,
            _action_v_local_all,
        ) = _encode_sequence(flat_actions, action_model, action_jump_op, action_phase1_cfg)

    z_prev = z_all[:, :-1]
    rw_prev = rw_all[:, :-1]
    zn_prev = zn_all[:, :-1]
    action_z_prev = action_z_all[:, :-1]
    action_rw_prev = action_rw_all[:, :-1]
    action_z_q_prev = action_z_q_all[:, :-1]
    action_zn_prev = action_zn_all[:, :-1]

    H_closure = min(int(config.wm_prediction_horizon), T)
    world_model_mod = _unwrap_compiled_module(world_model)
    with _temporary_requires_grad([(world_model_mod, False)]):
        (
            L_closure_obs,
            L_obs_zeno,
            L_enclosure,
            L_enclosure_probe,
            closure_metrics,
        ) = _world_model_closure_losses(
            config,
            world_model,
            enclosure_probe,
            z_all[:, 0],
            rw_all[:, 0],
            c_bar_all[:, :-1],
            z_tex_all[:, :-1],
            action_z_all[:, :-1][:, :H_closure],
            K_code_all[:, :-1],
            K_all[:, 1 : H_closure + 1],
            K_code_all[:, 1 : H_closure + 1],
            update_idx,
            zeno_mode=config.zeno_mode,
        )
    L_closure = (
        config.w_dyn_transition * L_closure_obs
        + config.w_zeno * L_obs_zeno
        + config.w_enclosure * L_enclosure
    )

    L_obs_total = base_loss_obs + zn_reg_loss_obs
    L_action_total = base_loss_action + zn_reg_loss_action
    if config.freeze_encoder:
        L_enc_total = config.encoder_loss_scale * L_closure
    else:
        L_enc_total = config.encoder_loss_scale * (L_obs_total + L_action_total + L_closure)
    probe_loss_total = config.w_enclosure_probe * L_enclosure_probe
    if L_enc_total.requires_grad or probe_loss_total.requires_grad:
        (L_enc_total + probe_loss_total).backward()
        enc_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_enc), config.grad_clip)
        optimizer_enc.step()
        enclosure_grad = nn.utils.clip_grad_norm_(
            _optimizer_parameters(optimizer_enclosure),
            config.grad_clip,
        )
        optimizer_enclosure.step()
    else:
        enc_grad = 0.0
        enclosure_grad = 0.0
    metrics["enc/grad_norm"] = float(enc_grad)
    metrics["closure/probe_grad_norm"] = float(enclosure_grad)
    if not config.freeze_encoder:
        for key, value in _phase1_grad_breakdown(model).items():
            metrics[f"enc/{key}"] = value
        for key, value in _phase1_grad_breakdown(action_model).items():
            metrics[f"enc_action/{key}"] = value

    for key, value in enc_metrics_obs.items():
        metrics[f"enc/{key}"] = value
    for key, value in enc_metrics_action.items():
        metrics[f"enc_action/{key}"] = value
    metrics["enc/L_total"] = float(config.encoder_loss_scale * L_obs_total)
    metrics["enc_action/L_total"] = float(config.encoder_loss_scale * L_action_total)
    metrics["closure/L_total"] = float(L_closure)
    metrics["closure/L_obs_state"] = float(L_closure_obs)
    metrics["closure/L_obs_zeno"] = float(L_obs_zeno)
    metrics["closure/L_enclosure"] = float(L_enclosure)
    metrics["closure/L_enclosure_probe"] = float(L_enclosure_probe)
    metrics["closure/obs_state_acc"] = closure_metrics["closure/obs_state_acc"]
    metrics["closure/obs_symbol_acc"] = closure_metrics["closure/obs_symbol_acc"]
    metrics["closure/chart_entropy"] = closure_metrics["closure/chart_entropy"]
    metrics["closure/enclosure_acc_full"] = closure_metrics["closure/enclosure_acc_full"]
    metrics["closure/enclosure_acc_base"] = closure_metrics["closure/enclosure_acc_base"]
    metrics["closure/enclosure_defect_acc"] = closure_metrics["closure/enclosure_defect_acc"]
    metrics["closure/enclosure_defect_ce"] = closure_metrics["closure/enclosure_defect_ce"]
    metrics["closure/grl_alpha"] = closure_metrics["closure/grl_alpha"]
    metrics["time/encoder"] = time.perf_counter() - t_section

    t_section = time.perf_counter()
    _sync_rl_atlas(model, action_model, world_model, critic, actor, reward_head, actor_old)
    metrics["time/atlas_sync"] = time.perf_counter() - t_section

    if compute_diagnostics:
        t_section = time.perf_counter()
        metrics.update(
            _per_chart_diagnostics(
                K_all.reshape(-1),
                K_code_all.reshape(-1),
                rw_all.reshape(-1, config.num_charts),
                z_all.reshape(-1, config.latent_dim),
                config.num_charts,
                config.codes_per_chart,
            ),
        )
        action_chart_diag = _per_chart_diagnostics(
            action_K_all.reshape(-1),
            action_K_code_all.reshape(-1),
            action_rw_all.reshape(-1, config.num_action_charts),
            action_z_all.reshape(-1, config.latent_dim),
            config.num_action_charts,
            config.action_codes_per_chart,
        )
        metrics.update({f"action_chart/{k}": v for k, v in action_chart_diag.items()})
        metrics["time/chart_diag"] = time.perf_counter() - t_section

    z_0 = z_all[:, 0].detach()
    rw_0 = rw_all[:, 0].detach()
    z_targets = z_all[:, 1:].detach()
    z_prev = z_all[:, :-1].detach()
    rw_prev = rw_all[:, :-1].detach()
    action_z_prev = action_z_all[:, :-1].detach()
    action_rw_prev = action_rw_all[:, :-1].detach()
    action_z_q_prev = action_z_q_all[:, :-1].detach()
    action_canonical_model = action_z_prev

    t_section = time.perf_counter()
    optimizer_wm.zero_grad()
    wm_out = world_model(z_0, action_canonical_model[:, :H_wm], rw_0)
    z_pred = wm_out["z_trajectory"]
    T_wm = min(z_pred.shape[1], H_wm)
    z_tgt_wm = z_targets[:, :T_wm]
    L_geo = compute_dynamics_geodesic_loss(z_pred, z_tgt_wm)
    target_charts = K_all.detach()[:, 1 : T_wm + 1]
    target_codes = K_code_all.detach()[:, 1 : T_wm + 1]
    wm_valid = replay_valid[:, :T_wm]
    L_chart = compute_dynamics_chart_loss(wm_out["chart_logits"][:, :T_wm], target_charts)
    chart_probs = F.softmax(wm_out["chart_logits"][:, :T_wm], dim=-1)
    wm_symbolic = _soft_symbolic_state_distribution(
        model,
        z_pred[:, :T_wm].reshape(-1, config.latent_dim),
        router_weights_override=chart_probs.reshape(-1, config.num_charts),
        hard_routing_tau=current_tau,
    )
    wm_state_probs = wm_symbolic["state_probs"].reshape(
        B, T_wm, config.num_charts * config.codes_per_chart
    )
    wm_code_probs = wm_symbolic["code_probs"].reshape(
        B, T_wm, config.num_charts, config.codes_per_chart
    )
    L_wm_code, L_wm_symbol, wm_symbol_metrics = _symbolic_transition_supervision_losses(
        state_probs=wm_state_probs,
        code_probs=wm_code_probs,
        target_charts=target_charts,
        target_codes=target_codes,
        valid_mask=wm_valid,
        codes_per_chart=config.codes_per_chart,
        metric_prefix="wm",
    )
    with torch.no_grad():
        wm_pred_state = wm_state_probs.detach().argmax(dim=-1)
        wm_pred_code = wm_pred_state.remainder(config.codes_per_chart)
    metrics["wm/L_geodesic"] = float(L_geo)
    metrics["wm/L_chart"] = float(L_chart)
    metrics["wm/L_code"] = float(L_wm_code.detach())
    metrics["wm/L_symbol"] = float(L_wm_symbol.detach())
    metrics["wm/chart_acc"] = float(
        _masked_mean(
            (wm_out["chart_logits"][:, :T_wm].argmax(dim=-1) == target_charts).to(
                chart_probs.dtype
            ),
            wm_valid,
        ).detach()
    )
    metrics["wm/code_acc_model_chart"] = float(
        _masked_mean(
            (wm_pred_code == target_codes).to(chart_probs.dtype),
            wm_valid,
        ).detach()
    )
    metrics.update(wm_symbol_metrics)
    metrics["wm/chart_entropy"] = float(
        _masked_mean(
            -(chart_probs * chart_probs.clamp(min=1e-8).log()).sum(dim=-1),
            wm_valid,
        ).detach(),
    )

    z_prev_flat = z_prev.reshape(-1, config.latent_dim)
    rw_prev_flat = rw_prev.reshape(-1, config.num_charts)
    K_prev_flat = K_all[:, :-1].reshape(-1)
    K_code_prev_flat = K_code_all[:, :-1].reshape(-1)
    zn_prev_flat = zn_prev.reshape(-1, config.latent_dim)
    z_next_flat = z_targets.reshape(-1, config.latent_dim)
    rw_next_flat = rw_all[:, 1:].detach().reshape(-1, config.num_charts)
    action_z_prev_flat = action_z_prev.reshape(-1, config.latent_dim)
    action_rw_prev_flat = action_rw_prev.reshape(-1, config.num_action_charts)
    action_K_prev_flat = action_K_all[:, :-1].reshape(-1)
    action_K_code_prev_flat = action_K_code_all[:, :-1].reshape(-1)
    action_zn_prev_flat = action_zn_prev.reshape(-1, config.latent_dim)
    action_z_q_prev_flat = action_z_q_prev.reshape(-1, config.latent_dim)
    action_canonical_flat = action_canonical_model.reshape(-1, config.latent_dim)
    continuation_flat = (1.0 - replay_dones).reshape(-1, 1)
    continuation_seq = (1.0 - replay_dones).detach()
    reward_conservative_flat, value_prev_flat, _value_next_flat = _conservative_reward_from_value(
        critic,
        z_prev_flat,
        rw_prev_flat,
        z_next_flat,
        rw_next_flat,
        config.gamma,
        continuation=continuation_flat,
    )
    exact_covector = _value_covector_from_critic(critic, z_prev_flat, rw_prev_flat)
    exact_covector_fn = None
    if compute_diagnostics:
        rw_prev_flat_detached = rw_prev_flat.detach()

        def exact_covector_fn(z_req: torch.Tensor) -> torch.Tensor:
            return _value_covector_from_critic(
                critic,
                z_req,
                rw_prev_flat_detached[: z_req.shape[0]],
                create_graph=True,
                detach=False,
            )

    reward_info = reward_head.decompose(
        z_prev_flat,
        rw_prev_flat,
        action_z_prev_flat,
        action_rw_prev_flat,
        action_z_q_prev_flat,
        action_canonical=action_canonical_flat,
        exact_covector=exact_covector,
        exact_covector_fn=exact_covector_fn,
        compute_curl=compute_diagnostics,
        curl_batch_limit=config.reward_curl_batch_limit,
    )
    z_prev_flat_critic = z_prev_flat.detach()
    z_next_flat_critic = z_next_flat.detach()
    rw_prev_flat_critic = rw_prev_flat.detach()
    z_all_critic = z_all.detach()
    rw_all_critic = rw_all.detach()
    exact_covector_train = None
    if (
        config.w_critic_stiffness > 0.0
        or config.w_critic_covector_align > 0.0
        or config.w_macro_covector_pullback > 0.0
        or config.w_macro_on_policy_covector_pullback > 0.0
    ):
        exact_covector_train = _value_covector_from_critic(
            critic,
            z_prev_flat_critic,
            rw_prev_flat_critic,
            create_graph=True,
            detach=False,
        )
    force_consistency_diag = _conservative_force_diagnostics(
        world_model_mod,
        z_prev_flat.detach(),
        world_model_mod.momentum_init(z_prev_flat.detach()).detach(),
        rw_prev_flat.detach(),
        action_canonical_flat.detach(),
    )
    rollout_rw_hodge = torch.cat([rw_0.unsqueeze(1), chart_probs[:, :-1].detach()], dim=1)
    rollout_z_hodge = torch.cat([z_0.unsqueeze(1), z_pred[:, :-1].detach()], dim=1)
    rollout_p0_hodge = world_model_mod.momentum_init(z_0.detach()).detach()
    rollout_p_hodge = torch.cat(
        [rollout_p0_hodge.unsqueeze(1), wm_out["momenta"][:, :-1].detach()], dim=1
    )
    rollout_force_diag = _conservative_force_diagnostics(
        world_model_mod,
        rollout_z_hodge.reshape(-1, config.latent_dim),
        rollout_p_hodge.reshape(-1, config.latent_dim),
        rollout_rw_hodge.reshape(-1, config.num_charts),
        action_canonical_model[:, :T_wm].detach().reshape(-1, config.latent_dim),
    )
    r_noncons_flat = reward_info["reward_nonconservative"]
    r_pred = (reward_conservative_flat + r_noncons_flat).reshape(B, T)
    r_cons = reward_conservative_flat.reshape(B, T, 1)
    r_noncons = r_noncons_flat.reshape(B, T)
    rho_r = reward_info["reward_density"].reshape(B, T, 1)
    reward_form_cov = reward_info["reward_form_cov"]
    reward_form_cov_raw = reward_info["reward_form_cov_raw"]
    reward_form_exact_component = reward_info["reward_form_exact_component"]
    exact_covector_norm = _metric_covector_norm_sq(
        world_model_mod.metric,
        z_prev_flat.detach(),
        exact_covector,
    ).sqrt()
    exact_covector_norm_mean = _masked_mean(exact_covector_norm, replay_valid.reshape(-1))
    force_rel_err_mean = _masked_mean(
        force_consistency_diag["force_rel_err"],
        replay_valid.reshape(-1),
    )
    reward_nonconservative_gate, reward_nonconservative_gate_metrics = (
        _reward_nonconservative_gate(
            config,
            exact_covector_norm_mean=exact_covector_norm_mean,
            force_rel_err_mean=force_rel_err_mean,
        )
    )
    r_noncons_effective = reward_nonconservative_gate.detach() * r_noncons
    reward_residual_target = reward_nonconservative_gate.detach() * (
        rewards - r_cons.detach().squeeze(-1)
    )
    replay_cons_target = rewards - r_noncons_effective.detach()
    L_reward = _masked_mean((r_pred - rewards).pow(2), replay_valid)
    L_reward_nonconservative = _masked_mean(
        (r_noncons - reward_residual_target).pow(2),
        replay_valid,
    )
    L_reward_conservative_match = _masked_mean(
        (r_cons.squeeze(-1) - replay_cons_target.detach()).pow(2),
        replay_valid,
    )
    reward_form_dot_exact = (reward_form_cov_raw * exact_covector).sum(dim=-1)
    reward_form_norm_sq = reward_form_cov_raw.pow(2).sum(dim=-1)
    exact_covector_norm_sq = exact_covector.pow(2).sum(dim=-1)
    # The latent Poincare metric is conformal, so the scalar G^{ij}(z) factor cancels in
    # this pointwise cosine between covectors and the Euclidean formula is equivalent.
    reward_exact_cos2 = reward_form_dot_exact.pow(2) / (
        reward_form_norm_sq * exact_covector_norm_sq + 1e-8
    )
    L_reward_exact_orth = _masked_mean(reward_exact_cos2, replay_valid.reshape(-1))
    (
        L_reward_nonconservative_norm,
        L_reward_nonconservative_budget,
        reward_preference_metrics,
    ) = _reward_conservative_preference_losses(
        config,
        metric=world_model_mod.metric,
        z=z_prev_flat.detach(),
        reward_conservative=r_cons,
        reward_nonconservative=r_noncons,
        reward_form_cov=reward_form_cov,
        replay_valid=replay_valid,
    )
    L_force_exact = _masked_mean(
        force_consistency_diag["force_err_sq"],
        replay_valid.reshape(-1),
    )
    L_force_task_exact = _masked_mean(
        force_consistency_diag["task_force_err_sq"],
        replay_valid.reshape(-1),
    )
    L_force_risk_exact = _masked_mean(
        force_consistency_diag["risk_force_err_sq"],
        replay_valid.reshape(-1),
    )
    metrics["wm/L_reward"] = float(L_reward)
    metrics["wm/L_reward_nonconservative"] = float(L_reward_nonconservative)
    metrics["wm/L_reward_conservative_match"] = float(L_reward_conservative_match)
    metrics["wm/L_reward_exact_orth"] = float(L_reward_exact_orth)
    metrics["wm/L_force_exact"] = float(L_force_exact)
    metrics["wm/L_force_task_exact"] = float(L_force_task_exact)
    metrics["wm/L_force_risk_exact"] = float(L_force_risk_exact)
    metrics.update(reward_preference_metrics)
    metrics.update(reward_nonconservative_gate_metrics)
    metrics["wm/reward_nonconservative_frac_masked"] = float(
        _masked_mean(
            r_noncons_effective.abs()
            / (r_noncons_effective.abs() + r_cons.detach().squeeze(-1).abs() + 1e-8),
            replay_valid,
        ),
    )
    metrics["wm/action_canonical_norm_mean"] = float(action_canonical_flat.norm(dim=-1).mean())
    metrics["wm/reward_conservative_mean"] = float(r_cons.mean())
    metrics["wm/reward_nonconservative_mean"] = float(r_noncons.mean())
    metrics["wm/reward_nonconservative_effective_mean"] = float(r_noncons_effective.mean())
    metrics["wm/reward_residual_target_mean"] = float(reward_residual_target.mean())
    metrics["wm/reward_exact_cos2_mean"] = float(
        _masked_mean(reward_exact_cos2, replay_valid.reshape(-1))
    )
    metrics["wm/reward_nonconservative_frac"] = float(
        r_noncons_effective.abs().mean() / (r_pred.abs().mean() + 1e-8)
    )
    metrics["wm/reward_density_mean"] = float(rho_r.mean())
    metrics["wm/reward_form_metric_norm_mean"] = float(
        _metric_covector_norm_sq(world_model_mod.metric, z_prev_flat.detach(), reward_form_cov)
        .sqrt()
        .mean()
    )
    metrics["wm/reward_form_raw_metric_norm_mean"] = float(
        _metric_covector_norm_sq(
            world_model_mod.metric,
            z_prev_flat.detach(),
            reward_form_cov_raw,
        )
        .sqrt()
        .mean()
    )
    metrics["wm/reward_form_exact_leakage_metric_mean"] = float(
        _metric_covector_norm_sq(
            world_model_mod.metric,
            z_prev_flat.detach(),
            reward_form_exact_component,
        )
        .sqrt()
        .mean()
    )
    metrics["wm/reward_exact_covector_norm_mean"] = float(exact_covector_norm_mean)
    metrics["wm/force_exact_rel_mean"] = float(force_rel_err_mean)
    metrics["wm/force_task_exact_rel_mean"] = float(
        force_consistency_diag["task_force_rel_err"].mean()
    )
    metrics["wm/force_risk_exact_rel_mean"] = float(
        force_consistency_diag["risk_force_rel_err"].mean()
    )
    if reward_info["reward_curl"].numel() > 0:
        metrics["wm/reward_curl_norm_mean"] = float(
            torch.linalg.norm(reward_info["reward_curl"], dim=(-2, -1)).mean()
        )
    else:
        metrics["wm/reward_curl_norm_mean"] = 0.0

    L_momentum = compute_momentum_regularization(wm_out["momenta"], wm_out["z_trajectory"])
    L_energy = (
        wm_out["energy_var"]
        if "energy_var" in wm_out
        else compute_energy_conservation_loss(
            wm_out["phi_eff"],
            wm_out["momenta"],
            wm_out["z_trajectory"],
        )
    )
    hodge_conservative_ratio = wm_out["hodge_conservative_ratio"].mean()
    hodge_solenoidal_ratio = wm_out["hodge_solenoidal_ratio"].mean()
    hodge_harmonic_ratio = wm_out["hodge_harmonic_ratio"].mean()
    L_hodge = compute_hodge_consistency_loss(wm_out["hodge_harmonic_forces"])
    (
        L_hodge_conservative_margin,
        L_hodge_solenoidal,
        hodge_preference_metrics,
    ) = _hodge_conservative_preference_losses(
        config,
        hodge_conservative_ratio=wm_out["hodge_conservative_ratio"],
        hodge_solenoidal_ratio=wm_out["hodge_solenoidal_ratio"],
    )
    metrics["wm/L_momentum"] = float(L_momentum)
    metrics["wm/L_energy"] = float(L_energy)
    metrics["wm/L_hodge"] = float(L_hodge)
    metrics.update(hodge_preference_metrics)
    L_wm_core = (
        config.w_dynamics * (L_geo + L_chart)
        + config.w_wm_code * L_wm_code
        + config.w_wm_symbol * L_wm_symbol
        + config.w_reward * L_reward
        + config.w_reward_conservative_match * L_reward_conservative_match
        + config.w_reward_nonconservative * L_reward_nonconservative
        + config.w_reward_exact_orth * L_reward_exact_orth
        + config.w_exact_force_consistency * L_force_exact
        + config.w_exact_task_force_consistency * L_force_task_exact
        + config.w_exact_risk_force_consistency * L_force_risk_exact
        + config.w_reward_nonconservative_norm * L_reward_nonconservative_norm
        + config.w_reward_nonconservative_budget * L_reward_nonconservative_budget
        + config.w_momentum_reg * L_momentum
        + config.w_energy_conservation * L_energy
        + config.w_hodge * L_hodge
        + config.w_hodge_conservative_margin * L_hodge_conservative_margin
        + config.w_hodge_solenoidal * L_hodge_solenoidal
    )

    critic_t0 = time.perf_counter()
    replay_values = value_prev_flat.reshape(B, T)
    replay_rtg = _discounted_return_to_go(replay_cons_target, replay_dones, config.gamma)
    replay_target_scale = _target_normalization_scale(
        [replay_rtg],
        [replay_valid],
        quantile=config.critic_target_scale_quantile,
        min_scale=config.critic_target_scale_min,
        template=replay_values,
    )
    replay_reward_support_threshold = max(0.1 * float(replay_target_scale.detach()), 1e-6)
    metrics["critic/replay_reward_mean"] = float(
        _masked_mean(replay_cons_target, replay_valid).detach()
    )
    metrics["critic/replay_return_mean"] = float(_masked_mean(replay_rtg, replay_valid).detach())
    metrics["critic/replay_target_scale"] = float(replay_target_scale.detach())
    metrics["critic/replay_reward_support_frac"] = float(
        _masked_support_fraction(
            replay_cons_target,
            replay_valid,
            threshold=replay_reward_support_threshold,
        ).detach(),
    )
    metrics["critic/replay_reward_positive_frac"] = float(
        _masked_mean(
            (replay_cons_target >= replay_reward_support_threshold).to(replay_cons_target.dtype),
            replay_valid,
        ).detach(),
    )
    replay_gap = replay_values - replay_rtg
    value_abs_err = _masked_mean(replay_gap.abs(), replay_valid)
    L_value = _masked_mean((replay_gap / replay_target_scale).pow(2), replay_valid)
    replay_chart_idx_seq = K_all.detach()[:, :-1]
    replay_code_idx_seq = K_code_all.detach()[:, :-1]
    replay_state_idx_seq = _state_index(
        replay_chart_idx_seq,
        replay_code_idx_seq,
        config.codes_per_chart,
    )
    replay_next_chart_idx_seq = K_all.detach()[:, 1:]
    replay_next_code_idx_seq = K_code_all.detach()[:, 1:]
    replay_next_state_idx_seq = _state_index(
        replay_next_chart_idx_seq,
        replay_next_code_idx_seq,
        config.codes_per_chart,
    )
    replay_action_chart_idx_seq = action_K_all.detach()[:, :-1]
    replay_action_code_idx_seq = action_K_code_all.detach()[:, :-1]
    replay_action_state_idx_seq = _state_index(
        replay_action_chart_idx_seq,
        replay_action_code_idx_seq,
        config.action_codes_per_chart,
    )
    with torch.no_grad():
        replay_next_actor_out = actor(
            replay_next_chart_idx_seq.reshape(-1),
            replay_next_code_idx_seq.reshape(-1),
            zn_all[:, 1:].detach().reshape(-1, config.latent_dim),
            hard_routing=current_hard_routing,
            hard_routing_tau=current_tau,
        )
        replay_next_action_state_probs = _action_symbolic_distribution_from_actor(
            replay_next_actor_out,
        )["state_probs"].reshape(B, T, macro_critic.num_actions)
    replay_macro_transition = _macro_state_transition_distribution(
        model,
        world_model_mod,
        z_prev_flat.detach(),
        action_canonical_flat.detach(),
        rw_prev_flat.detach(),
        hard_routing_tau=current_tau,
    )
    replay_macro_next_state_probs = replay_macro_transition["next_state_probs"].reshape(
        B,
        T,
        macro_critic.num_states,
    )
    macro_q_state_action_seq = macro_critic.q_from_state_action_idx(
        replay_state_idx_seq,
        replay_action_state_idx_seq,
    )
    macro_reward_seq = macro_critic.reward_from_state_action_idx(
        replay_state_idx_seq,
        replay_action_state_idx_seq,
    )
    macro_value_next_seq = macro_critic.value_from_probs(
        replay_macro_next_state_probs.reshape(-1, macro_critic.num_states),
        replay_next_action_state_probs.reshape(-1, macro_critic.num_actions),
    ).reshape(B, T)
    macro_bootstrap_seq = config.gamma * continuation_seq * macro_value_next_seq.detach()
    macro_td_target = replay_cons_target.detach() + macro_bootstrap_seq
    macro_target_scale = _target_normalization_scale(
        [replay_cons_target.detach(), macro_td_target.detach(), macro_q_state_action_seq.detach()],
        [replay_valid, replay_valid, replay_valid],
        quantile=config.macro_target_scale_quantile,
        min_scale=config.macro_target_scale_min,
        template=macro_q_state_action_seq,
    )
    macro_gap = macro_q_state_action_seq - macro_td_target
    L_macro_value = _masked_mean((macro_gap / macro_target_scale).pow(2), replay_valid)
    replay_macro_log_probs = replay_macro_next_state_probs.clamp(min=1e-8).log()
    replay_macro_transition_ce = F.nll_loss(
        replay_macro_log_probs.reshape(-1, macro_critic.num_states),
        replay_next_state_idx_seq.reshape(-1),
        reduction="none",
    ).reshape(B, T)
    L_macro_transition = _masked_mean(replay_macro_transition_ce, replay_valid)
    macro_transition_sharpen_gate, macro_transition_sharpen_metrics = (
        _macro_transition_sharpening_gate(
            config,
            obs_state_acc=closure_metrics["closure/obs_state_acc"],
            enclosure_defect_acc=closure_metrics["closure/enclosure_defect_acc"],
            enclosure_defect_ce=closure_metrics["closure/enclosure_defect_ce"],
            template=macro_q_state_action_seq,
        )
    )
    metrics.update(macro_transition_sharpen_metrics)
    replay_macro_next_state_entropy = replay_macro_transition["next_state_entropy"].reshape(B, T)
    macro_transition_entropy_norm = replay_macro_next_state_entropy / max(
        float(np.log(max(macro_critic.num_states, 2))),
        1.0,
    )
    L_macro_transition_entropy = macro_transition_sharpen_gate * _masked_mean(
        macro_transition_entropy_norm,
        replay_valid,
    )
    replay_action_state_probs = F.one_hot(
        replay_action_state_idx_seq.reshape(-1),
        num_classes=macro_critic.num_actions,
    ).to(dtype=action_canonical_flat.dtype, device=action_canonical_flat.device)
    replay_symbolic_soft = _macro_control_pullback_covector(
        model,
        world_model_mod,
        macro_critic,
        actor,
        z_prev_flat.detach(),
        action_state_probs=replay_action_state_probs.detach(),
        action_canonical=action_canonical_flat.detach(),
        gamma=config.gamma,
        continuation=continuation_flat.detach(),
        router_weights_override=rw_prev_flat.detach(),
        hard_routing=current_hard_routing,
        hard_routing_tau=current_tau,
        create_graph=config.w_macro_covector_pullback > 0.0,
        detach=config.w_macro_covector_pullback <= 0.0,
    )
    macro_reward_pullback_replay = replay_symbolic_soft["macro_reward"].reshape(B, T)
    L_macro_pullback = _masked_mean(
        ((macro_reward_pullback_replay - replay_cons_target.detach()) / macro_target_scale).pow(2),
        replay_valid,
    )
    L_critic_macro_pullback = _masked_mean(
        (
            (
                reward_conservative_flat.squeeze(-1).reshape(B, T)
                - macro_reward_pullback_replay.detach()
            )
            / macro_target_scale
        ).pow(2),
        replay_valid,
    )
    L_macro_covector_pullback = replay_gap.new_zeros(())
    L_critic_macro_covector_pullback = replay_gap.new_zeros(())
    metrics["macro/L_covector_pullback"] = 0.0
    metrics["macro/covector_pullback_abs_err"] = 0.0
    metrics["macro/covector_norm_mean"] = 0.0
    metrics["macro/covector_target_norm_mean"] = 0.0
    metrics["macro/covector_target_scale"] = 0.0
    metrics["critic/L_macro_covector_pullback"] = 0.0
    if exact_covector_train is not None and (
        config.w_macro_covector_pullback > 0.0 or config.w_macro_pullback > 0.0
    ):
        replay_macro_covector = replay_symbolic_soft["macro_covector"]
        if config.w_macro_covector_pullback > 0.0:
            L_macro_covector_pullback, macro_covector_pullback_metrics = (
                _macro_covector_pullback_loss(
                    metric=world_model_mod.metric,
                    z=z_prev_flat.detach(),
                    macro_covector=replay_macro_covector,
                    exact_covector=exact_covector_train.detach(),
                    replay_valid=replay_valid,
                    metric_prefix="macro",
                )
            )
            metrics.update(macro_covector_pullback_metrics)
        if config.w_macro_pullback > 0.0:
            L_critic_macro_covector_pullback, _ = _macro_covector_pullback_loss(
                metric=world_model_mod.metric,
                z=z_prev_flat.detach(),
                macro_covector=replay_macro_covector.detach(),
                exact_covector=exact_covector_train,
                replay_valid=replay_valid,
                metric_prefix="critic",
            )
            metrics["critic/L_macro_covector_pullback"] = float(
                L_critic_macro_covector_pullback.detach(),
            )
    L_poisson = compute_screened_poisson_loss(
        critic,
        z_prev,
        None,
        rw_prev,
        reward_density=rho_r,
        kappa=config.screened_poisson_kappa,
    )
    critic_stage_scales = _critic_stage_scales(config, epoch)
    poisson_weight = config.w_screened_poisson * critic_stage_scales["poisson"]
    covector_weight = config.w_critic_covector_align * critic_stage_scales["covector"]
    stiffness_weight = config.w_critic_stiffness * critic_stage_scales["stiffness"]
    macro_pullback_weight = config.w_macro_pullback * critic_stage_scales["macro_pullback"]
    macro_covector_pullback_weight = (
        config.w_macro_covector_pullback * critic_stage_scales["macro_pullback"]
    )
    on_policy_covector_weight = (
        config.w_critic_on_policy_covector_align
        * critic_stage_scales["covector"]
        * critic_stage_scales["on_policy"]
    )
    on_policy_stiffness_weight = (
        config.w_critic_on_policy_stiffness
        * critic_stage_scales["stiffness"]
        * critic_stage_scales["on_policy"]
    )
    macro_on_policy_pullback_weight = (
        config.w_macro_on_policy_pullback
        * critic_stage_scales["macro_pullback"]
        * critic_stage_scales["on_policy"]
    )
    macro_on_policy_covector_pullback_weight = (
        config.w_macro_on_policy_covector_pullback
        * critic_stage_scales["macro_pullback"]
        * critic_stage_scales["on_policy"]
    )
    critic_multistep_horizon = max(int(config.critic_multistep_horizon), 1)
    critic_multistep_decay = max(float(config.critic_multistep_decay), 0.0)
    replay_value_seq = None
    L_macro_exact_increment = replay_gap.new_zeros(())
    L_critic_exact_increment = replay_gap.new_zeros(())
    if config.w_critic_exact_increment > 0.0 and critic_multistep_horizon > 1:
        replay_value_seq = critic_value(
            critic,
            z_all_critic.reshape(-1, config.latent_dim),
            rw_all_critic.reshape(-1, config.num_charts),
        ).reshape(B, T + 1)
        L_critic_exact_increment, critic_exact_increment_metrics = _multistep_exact_increment_loss(
            value_seq=replay_value_seq,
            reward_conservative_targets=replay_cons_target,
            continuation=continuation_seq,
            valid_mask=replay_valid,
            gamma=config.gamma,
            horizon=critic_multistep_horizon,
            decay=critic_multistep_decay,
            metric_prefix="critic",
            target_scale_quantile=config.critic_target_scale_quantile,
            target_scale_min=config.critic_target_scale_min,
        )
    else:
        L_critic_exact_increment, critic_exact_increment_metrics = _critic_exact_increment_loss(
            reward_conservative_pred=reward_conservative_flat.squeeze(-1),
            reward_conservative_target=replay_cons_target,
            replay_valid=replay_valid,
        )
    metrics.update(critic_exact_increment_metrics)
    L_macro_exact_increment, macro_exact_increment_metrics = _critic_exact_increment_loss(
        reward_conservative_pred=macro_reward_seq,
        reward_conservative_target=replay_cons_target,
        replay_valid=replay_valid,
    )
    macro_exact_increment_metrics = {
        key.replace("critic/", "macro/"): value
        for key, value in macro_exact_increment_metrics.items()
    }
    metrics.update(macro_exact_increment_metrics)
    metrics["macro/L_value"] = float(L_macro_value.detach())
    metrics["macro/value_abs_err"] = float(_masked_mean(macro_gap.abs(), replay_valid).detach())
    metrics["macro/target_scale"] = float(macro_target_scale.detach())
    metrics["macro/L_transition"] = float(L_macro_transition.detach())
    metrics["macro/L_transition_entropy"] = float(L_macro_transition_entropy.detach())
    metrics["macro/transition_ce"] = float(
        _masked_mean(replay_macro_transition_ce, replay_valid).detach()
    )
    metrics["macro/transition_acc"] = float(
        _masked_mean(
            (replay_macro_next_state_probs.argmax(dim=-1) == replay_next_state_idx_seq).to(
                replay_macro_next_state_probs.dtype
            ),
            replay_valid,
        ).detach(),
    )
    metrics.update(
        _macro_transition_observability_metrics(
            macro_critic=macro_critic,
            state_idx=replay_state_idx_seq,
            next_state_probs=replay_macro_next_state_probs,
            reward_pred=macro_reward_seq,
            reward_target=replay_cons_target.detach(),
            next_value=macro_value_next_seq.detach(),
            bootstrap_term=macro_bootstrap_seq,
            valid_mask=replay_valid,
            metric_prefix="macro",
        ),
    )
    replay_macro_chart_entropy = replay_symbolic_soft["state_value_entropy"]
    metrics["macro/replay_state_entropy"] = float(
        _masked_mean(replay_macro_chart_entropy, replay_valid.reshape(-1)).detach(),
    )
    metrics["macro/L_pullback"] = float(L_macro_pullback.detach())
    metrics["macro/pullback_abs_err"] = float(
        _masked_mean(
            (macro_reward_pullback_replay - replay_cons_target.detach()).abs(),
            replay_valid,
        ).detach(),
    )
    metrics["critic/L_macro_pullback"] = float(L_critic_macro_pullback.detach())
    L_critic_covector_align = replay_gap.new_zeros(())
    L_critic_stiffness = replay_gap.new_zeros(())
    L_critic_covector_align_on_policy = replay_gap.new_zeros(())
    L_critic_stiffness_on_policy = replay_gap.new_zeros(())
    L_macro_on_policy_pullback = replay_gap.new_zeros(())
    L_critic_macro_pullback_on_policy = replay_gap.new_zeros(())
    L_macro_on_policy_covector_pullback = replay_gap.new_zeros(())
    L_critic_macro_covector_pullback_on_policy = replay_gap.new_zeros(())
    metrics["critic/on_policy/L_covector_align"] = 0.0
    metrics["critic/on_policy/L_stiffness"] = 0.0
    metrics["critic/on_policy/covector_align_abs_err"] = 0.0
    metrics["critic/on_policy/covector_predicted_reward_mean"] = 0.0
    metrics["critic/on_policy/covector_target_reward_mean"] = 0.0
    metrics["critic/on_policy/displacement_norm_mean"] = 0.0
    metrics["critic/on_policy/stiffness_target_adaptive"] = float(config.critic_stiffness_min)
    metrics["critic/on_policy/covector_horizon_used"] = 0.0
    metrics["critic/on_policy/exact_covector_norm_mean"] = 0.0
    metrics["critic/on_policy/stiffness_target"] = float(config.critic_stiffness_min)
    metrics["critic/on_policy/stiffness_certified"] = 0.0
    metrics["critic/on_policy/calibration_ratio"] = 0.0
    metrics["critic/on_policy/batch_size"] = 0.0
    metrics["macro/on_policy/L_pullback"] = 0.0
    metrics["macro/on_policy/L_covector_pullback"] = 0.0
    metrics["macro/on_policy/pullback_abs_err"] = 0.0
    metrics["macro/on_policy/covector_pullback_abs_err"] = 0.0
    metrics["macro/on_policy/covector_norm_mean"] = 0.0
    metrics["macro/on_policy/value_std"] = 0.0
    metrics["macro/on_policy/exact_increment_abs_err"] = 0.0
    metrics["macro/on_policy/exact_increment_pred_abs_mean"] = 0.0
    metrics["macro/on_policy/exact_increment_target_mean"] = 0.0
    metrics["macro/on_policy/target_scale"] = float(macro_target_scale.detach())
    metrics["critic/on_policy/L_macro_pullback"] = 0.0
    metrics["critic/on_policy/L_macro_covector_pullback"] = 0.0
    if exact_covector_train is not None:
        exact_covector_train_seq = exact_covector_train.reshape(B, T, config.latent_dim)
        if critic_multistep_horizon > 1:
            if replay_value_seq is None:
                replay_value_seq = critic_value(
                    critic,
                    z_all_critic.reshape(-1, config.latent_dim),
                    rw_all_critic.reshape(-1, config.num_charts),
                ).reshape(B, T + 1)
            (
                L_critic_covector_align,
                critic_stiffness_target,
                critic_covector_metrics,
            ) = _multistep_covector_alignment_loss(
                config,
                metric=world_model_mod.metric,
                z_seq=z_all_critic,
                value_seq=replay_value_seq,
                exact_covector_seq=exact_covector_train_seq,
                reward_conservative_targets=replay_cons_target,
                continuation=continuation_seq,
                valid_mask=replay_valid,
                gamma=config.gamma,
                horizon=critic_multistep_horizon,
                decay=critic_multistep_decay,
                metric_prefix="critic",
            )
        else:
            (
                L_critic_covector_align,
                critic_stiffness_target,
                critic_covector_metrics,
            ) = _critic_covector_alignment_loss(
                config,
                metric=world_model_mod.metric,
                z=z_prev_flat_critic,
                z_next=z_next_flat_critic,
                value_current=replay_values.reshape(-1, 1),
                exact_covector=exact_covector_train,
                reward_conservative_target=replay_cons_target,
                continuation=continuation_flat,
                gamma=config.gamma,
                replay_valid=replay_valid,
            )
        metrics.update(critic_covector_metrics)
        L_critic_stiffness, critic_stiffness_metrics = _critic_stiffness_loss(
            config,
            metric=world_model_mod.metric,
            z=z_prev_flat_critic,
            exact_covector=exact_covector_train,
            replay_valid=replay_valid,
            stiffness_scale=critic_stiffness_target,
        )
        metrics.update(critic_stiffness_metrics)
        metrics["critic/calibration_ratio"] = float(
            critic_stiffness_metrics["critic/exact_covector_norm_mean"]
            / max(critic_stiffness_metrics["critic/stiffness_target"], 1e-6),
        )
        critic_on_policy_horizon = min(
            max(int(config.critic_on_policy_horizon), 0),
            max(int(config.actor_return_horizon), int(config.imagination_horizon)),
        )
        if critic_on_policy_horizon > 0 and (
            config.w_critic_on_policy_covector_align > 0.0
            or config.w_critic_on_policy_stiffness > 0.0
        ):
            critic_on_policy_batch = (
                min(B, int(config.critic_on_policy_batch_size))
                if int(config.critic_on_policy_batch_size) > 0
                else B
            )
            if critic_on_policy_batch > 0:
                if critic_on_policy_batch < B:
                    critic_policy_idx = torch.randperm(B, device=z_0.device)[
                        :critic_on_policy_batch
                    ]
                    z_policy_0 = z_0.detach()[critic_policy_idx]
                    rw_policy_0 = rw_0.detach()[critic_policy_idx]
                else:
                    z_policy_0 = z_0.detach()
                    rw_policy_0 = rw_0.detach()
                actor_eval_modules = [
                    (actor, False),
                    (action_model, False),
                    (_unwrap_compiled_module(world_model), False),
                ]
                with _temporary_requires_grad(actor_eval_modules):
                    policy_rollout = _collect_policy_state_rollout(
                        model,
                        world_model_mod,
                        actor,
                        action_model,
                        z_policy_0,
                        rw_policy_0,
                        critic_on_policy_horizon,
                        hard_routing=current_hard_routing,
                        hard_routing_tau=current_tau,
                    )
                policy_z_seq = policy_rollout["z_seq"]
                policy_rw_seq = policy_rollout["rw_seq"]
                policy_values_seq = critic_value(
                    critic,
                    policy_z_seq.reshape(-1, config.latent_dim),
                    policy_rw_seq.reshape(-1, config.num_charts),
                ).reshape(policy_z_seq.shape[0], policy_z_seq.shape[1])
                policy_exact_covector = _value_covector_from_critic(
                    critic,
                    policy_z_seq[:, :-1].reshape(-1, config.latent_dim),
                    policy_rw_seq[:, :-1].reshape(-1, config.num_charts),
                    create_graph=True,
                    detach=False,
                ).reshape(policy_z_seq.shape[0], critic_on_policy_horizon, config.latent_dim)
                policy_valid = torch.ones(
                    policy_z_seq.shape[0],
                    critic_on_policy_horizon,
                    device=policy_z_seq.device,
                    dtype=policy_z_seq.dtype,
                )
                (
                    L_critic_covector_align_on_policy,
                    critic_on_policy_stiffness_target,
                    critic_on_policy_metrics,
                ) = _multistep_covector_alignment_loss(
                    config,
                    metric=world_model_mod.metric,
                    z_seq=policy_z_seq,
                    value_seq=policy_values_seq,
                    exact_covector_seq=policy_exact_covector,
                    reward_conservative_targets=None,
                    continuation=policy_valid,
                    valid_mask=policy_valid,
                    gamma=config.gamma,
                    horizon=critic_on_policy_horizon,
                    decay=max(float(config.critic_on_policy_decay), 0.0),
                    metric_prefix="critic/on_policy",
                )
                metrics.update(critic_on_policy_metrics)
                (
                    L_critic_stiffness_on_policy,
                    critic_on_policy_stiffness_metrics,
                ) = _critic_stiffness_loss(
                    config,
                    metric=world_model_mod.metric,
                    z=policy_z_seq[:, :-1].reshape(-1, config.latent_dim),
                    exact_covector=policy_exact_covector.reshape(-1, config.latent_dim),
                    replay_valid=policy_valid,
                    stiffness_scale=critic_on_policy_stiffness_target,
                )
                metrics["critic/on_policy/L_stiffness"] = float(
                    L_critic_stiffness_on_policy.detach(),
                )
                metrics["critic/on_policy/exact_covector_norm_mean"] = (
                    critic_on_policy_stiffness_metrics["critic/exact_covector_norm_mean"]
                )
                metrics["critic/on_policy/stiffness_target"] = critic_on_policy_stiffness_metrics[
                    "critic/stiffness_target"
                ]
                metrics["critic/on_policy/stiffness_certified"] = (
                    critic_on_policy_stiffness_metrics["critic/stiffness_certified"]
                )
                metrics["critic/on_policy/calibration_ratio"] = float(
                    critic_on_policy_stiffness_metrics["critic/exact_covector_norm_mean"]
                    / max(critic_on_policy_stiffness_metrics["critic/stiffness_target"], 1e-6),
                )
                metrics["critic/on_policy/batch_size"] = float(critic_on_policy_batch)
                macro_on_policy_horizon = min(
                    max(int(config.macro_on_policy_horizon), 0),
                    critic_on_policy_horizon,
                )
                if macro_on_policy_horizon > 0 and (
                    config.w_macro_on_policy_pullback > 0.0
                    or config.w_macro_pullback > 0.0
                    or config.w_macro_on_policy_covector_pullback > 0.0
                ):
                    policy_macro_control = _macro_control_pullback_covector(
                        model,
                        world_model_mod,
                        macro_critic,
                        actor,
                        policy_z_seq[:, :macro_on_policy_horizon]
                        .reshape(-1, config.latent_dim)
                        .detach(),
                        action_state_probs=policy_rollout["action_state_probs"][
                            :, :macro_on_policy_horizon
                        ]
                        .reshape(-1, macro_critic.num_actions)
                        .detach(),
                        action_canonical=policy_rollout["action_canonical_seq"][
                            :, :macro_on_policy_horizon
                        ]
                        .reshape(-1, config.latent_dim)
                        .detach(),
                        gamma=config.gamma,
                        continuation=policy_valid[:, :macro_on_policy_horizon]
                        .reshape(-1, 1)
                        .detach(),
                        router_weights_override=policy_rw_seq[:, :macro_on_policy_horizon]
                        .reshape(-1, config.num_charts)
                        .detach(),
                        hard_routing=current_hard_routing,
                        hard_routing_tau=current_tau,
                        create_graph=config.w_macro_on_policy_covector_pullback > 0.0,
                        detach=config.w_macro_on_policy_covector_pullback <= 0.0,
                    )
                    policy_macro_q_seq = policy_macro_control["macro_control"].reshape(
                        policy_z_seq.shape[0],
                        macro_on_policy_horizon,
                    )
                    policy_macro_reward_seq = policy_macro_control["macro_reward"].reshape(
                        policy_z_seq.shape[0],
                        macro_on_policy_horizon,
                    )
                    policy_value_seq_trunc = policy_values_seq[:, : macro_on_policy_horizon + 1]
                    policy_macro_exact_target = (
                        policy_value_seq_trunc[:, :-1].detach()
                        - float(config.gamma) * policy_value_seq_trunc[:, 1:].detach()
                    )
                    policy_macro_scale = _target_normalization_scale(
                        [policy_macro_exact_target.detach(), policy_macro_q_seq.detach()],
                        [
                            policy_valid[:, :macro_on_policy_horizon],
                            policy_valid[:, :macro_on_policy_horizon],
                        ],
                        quantile=config.macro_target_scale_quantile,
                        min_scale=config.macro_target_scale_min,
                        template=policy_macro_q_seq,
                    )
                    L_macro_on_policy_pullback = _masked_mean(
                        (
                            (
                                policy_macro_reward_seq
                                - (
                                    policy_value_seq_trunc[:, :-1].detach()
                                    - float(config.gamma) * policy_value_seq_trunc[:, 1:].detach()
                                )
                            )
                            / policy_macro_scale
                        ).pow(2),
                        policy_valid[:, :macro_on_policy_horizon],
                    )
                    L_critic_macro_pullback_on_policy = _masked_mean(
                        (
                            (
                                (
                                    policy_value_seq_trunc[:, :-1]
                                    - float(config.gamma) * policy_value_seq_trunc[:, 1:].detach()
                                )
                                - policy_macro_reward_seq.detach()
                            )
                            / policy_macro_scale
                        ).pow(2),
                        policy_valid[:, :macro_on_policy_horizon],
                    )
                    policy_macro_exact_pred = policy_macro_reward_seq
                    metrics["macro/on_policy/L_pullback"] = float(
                        L_macro_on_policy_pullback.detach(),
                    )
                    metrics["macro/on_policy/pullback_abs_err"] = float(
                        _masked_mean(
                            (policy_macro_reward_seq - policy_macro_exact_target).abs(),
                            policy_valid[:, :macro_on_policy_horizon],
                        ).detach(),
                    )
                    metrics["macro/on_policy/value_std"] = float(
                        policy_macro_q_seq.detach().std(unbiased=False),
                    )
                    metrics["macro/on_policy/exact_increment_abs_err"] = float(
                        _masked_mean(
                            (policy_macro_exact_pred - policy_macro_exact_target).abs(),
                            policy_valid[:, :macro_on_policy_horizon],
                        ).detach(),
                    )
                    metrics["macro/on_policy/exact_increment_pred_abs_mean"] = float(
                        _masked_mean(
                            policy_macro_exact_pred.abs(),
                            policy_valid[:, :macro_on_policy_horizon],
                        ).detach(),
                    )
                    metrics["macro/on_policy/exact_increment_target_mean"] = float(
                        _masked_mean(
                            policy_macro_exact_target,
                            policy_valid[:, :macro_on_policy_horizon],
                        ).detach(),
                    )
                    metrics["macro/on_policy/target_scale"] = float(policy_macro_scale.detach())
                    metrics["critic/on_policy/L_macro_pullback"] = float(
                        L_critic_macro_pullback_on_policy.detach(),
                    )
                    if exact_covector_train is not None and (
                        config.w_macro_on_policy_covector_pullback > 0.0
                        or config.w_macro_on_policy_pullback > 0.0
                    ):
                        policy_macro_covector = policy_macro_control["macro_covector"].reshape(
                            policy_z_seq.shape[0],
                            macro_on_policy_horizon,
                            config.latent_dim,
                        )
                        if config.w_macro_on_policy_covector_pullback > 0.0:
                            (
                                L_macro_on_policy_covector_pullback,
                                macro_on_policy_covector_metrics,
                            ) = _macro_covector_pullback_loss(
                                metric=world_model_mod.metric,
                                z=policy_z_seq[:, :macro_on_policy_horizon]
                                .reshape(-1, config.latent_dim)
                                .detach(),
                                macro_covector=policy_macro_covector.reshape(
                                    -1, config.latent_dim
                                ),
                                exact_covector=policy_exact_covector[:, :macro_on_policy_horizon]
                                .detach()
                                .reshape(
                                    -1,
                                    config.latent_dim,
                                ),
                                replay_valid=policy_valid[:, :macro_on_policy_horizon],
                                metric_prefix="macro/on_policy",
                            )
                            metrics.update(macro_on_policy_covector_metrics)
                        if config.w_macro_on_policy_pullback > 0.0:
                            L_critic_macro_covector_pullback_on_policy, _ = (
                                _macro_covector_pullback_loss(
                                    metric=world_model_mod.metric,
                                    z=policy_z_seq[:, :macro_on_policy_horizon]
                                    .reshape(-1, config.latent_dim)
                                    .detach(),
                                    macro_covector=policy_macro_covector.detach().reshape(
                                        -1,
                                        config.latent_dim,
                                    ),
                                    exact_covector=policy_exact_covector[
                                        :, :macro_on_policy_horizon
                                    ].reshape(-1, config.latent_dim),
                                    replay_valid=policy_valid[:, :macro_on_policy_horizon],
                                    metric_prefix="critic/on_policy",
                                )
                            )
                            metrics["critic/on_policy/L_macro_covector_pullback"] = float(
                                L_critic_macro_covector_pullback_on_policy.detach(),
                            )
    else:
        critic_exact_covector_norm_mean = float(
            _metric_covector_norm_sq(world_model_mod.metric, z_prev_flat.detach(), exact_covector)
            .sqrt()
            .mean(),
        )
        metrics["critic/L_covector_align"] = 0.0
        metrics["critic/covector_align_abs_err"] = 0.0
        metrics["critic/covector_predicted_reward_mean"] = 0.0
        metrics["critic/covector_target_reward_mean"] = 0.0
        metrics["critic/displacement_norm_mean"] = 0.0
        metrics["critic/stiffness_target_adaptive"] = float(config.critic_stiffness_min)
        metrics["critic/L_stiffness"] = 0.0
        metrics["critic/exact_covector_norm_mean"] = critic_exact_covector_norm_mean
        metrics["critic/stiffness_target"] = float(config.critic_stiffness_min)
        metrics["critic/stiffness_certified"] = (
            1.0 if critic_exact_covector_norm_mean >= float(config.critic_stiffness_min) else 0.0
        )
        metrics["critic/calibration_ratio"] = float(
            critic_exact_covector_norm_mean / max(float(config.critic_stiffness_min), 1e-6),
        )
        metrics["critic/on_policy/L_covector_align"] = 0.0
        metrics["critic/on_policy/L_stiffness"] = 0.0
        metrics["critic/on_policy/covector_align_abs_err"] = 0.0
        metrics["critic/on_policy/covector_predicted_reward_mean"] = 0.0
        metrics["critic/on_policy/covector_target_reward_mean"] = 0.0
        metrics["critic/on_policy/displacement_norm_mean"] = 0.0
        metrics["critic/on_policy/stiffness_target_adaptive"] = float(config.critic_stiffness_min)
        metrics["critic/on_policy/covector_horizon_used"] = 0.0
        metrics["critic/on_policy/exact_covector_norm_mean"] = 0.0
        metrics["critic/on_policy/stiffness_target"] = float(config.critic_stiffness_min)
        metrics["critic/on_policy/stiffness_certified"] = 0.0
        metrics["critic/on_policy/calibration_ratio"] = 0.0
        metrics["critic/on_policy/batch_size"] = 0.0
        metrics["macro/on_policy/L_pullback"] = 0.0
        metrics["macro/on_policy/pullback_abs_err"] = 0.0
        metrics["macro/on_policy/value_std"] = 0.0
        metrics["macro/on_policy/exact_increment_abs_err"] = 0.0
        metrics["macro/on_policy/exact_increment_target_mean"] = 0.0
        metrics["macro/on_policy/target_scale"] = float(macro_target_scale.detach())
        metrics["critic/on_policy/L_macro_pullback"] = 0.0
    value_loss_term = config.w_critic * L_value
    exact_loss_term = config.w_critic_exact_increment * L_critic_exact_increment
    poisson_loss_term = poisson_weight * L_poisson
    covector_loss_term = covector_weight * L_critic_covector_align
    stiffness_loss_term = stiffness_weight * L_critic_stiffness
    macro_pullback_loss_term = (
        macro_pullback_weight * L_critic_macro_pullback
        + macro_covector_pullback_weight * L_critic_macro_covector_pullback
    )
    on_policy_loss_term = (
        on_policy_covector_weight * L_critic_covector_align_on_policy
        + on_policy_stiffness_weight * L_critic_stiffness_on_policy
        + macro_on_policy_pullback_weight * L_critic_macro_pullback_on_policy
        + macro_on_policy_covector_pullback_weight * L_critic_macro_covector_pullback_on_policy
    )
    L_critic = (
        value_loss_term
        + exact_loss_term
        + poisson_loss_term
        + covector_loss_term
        + stiffness_loss_term
        + macro_pullback_loss_term
        + on_policy_loss_term
    )
    L_macro = (
        config.w_macro_value * L_macro_value
        + config.w_macro_exact_increment * L_macro_exact_increment
        + config.w_macro_pullback * L_macro_pullback
        + config.w_macro_covector_pullback * L_macro_covector_pullback
        + config.w_macro_on_policy_pullback * L_macro_on_policy_pullback
        + config.w_macro_on_policy_covector_pullback * L_macro_on_policy_covector_pullback
    )
    L_wm_total = (
        L_wm_core
        + config.w_macro_transition * L_macro_transition
        + config.w_macro_transition_entropy * L_macro_transition_entropy
    )
    metrics["time/critic"] = time.perf_counter() - critic_t0
    metrics["critic/L_critic"] = float(L_critic)
    metrics["macro/L_macro"] = float(L_macro.detach())
    metrics["wm/L_total"] = float(L_wm_total.detach())
    metrics["critic/L_value"] = float(L_value)
    metrics["critic/L_poisson"] = float(L_poisson)
    metrics["critic/poisson_weight"] = float(poisson_weight)
    metrics["critic/weight_exact_increment"] = float(config.w_critic_exact_increment)
    metrics["critic/weight_covector_align"] = float(covector_weight)
    metrics["critic/weight_stiffness"] = float(stiffness_weight)
    metrics["critic/weight_on_policy_covector_align"] = float(on_policy_covector_weight)
    metrics["critic/weight_on_policy_stiffness"] = float(on_policy_stiffness_weight)
    metrics["critic/weight_macro_pullback"] = float(macro_pullback_weight)
    metrics["critic/weight_macro_covector_pullback"] = float(macro_covector_pullback_weight)
    metrics["critic/weight_macro_on_policy_pullback"] = float(macro_on_policy_pullback_weight)
    metrics["critic/weight_macro_on_policy_covector_pullback"] = float(
        macro_on_policy_covector_pullback_weight,
    )
    metrics["critic/stage_exact_increment"] = critic_stage_scales["exact_increment"]
    metrics["critic/stage_poisson"] = critic_stage_scales["poisson"]
    metrics["critic/stage_covector"] = critic_stage_scales["covector"]
    metrics["critic/stage_stiffness"] = critic_stage_scales["stiffness"]
    metrics["critic/stage_macro_pullback"] = critic_stage_scales["macro_pullback"]
    metrics["critic/stage_on_policy"] = critic_stage_scales["on_policy"]
    metrics["critic/value_abs_err"] = float(value_abs_err.detach())
    metrics["geometric/hodge_conservative_direct"] = float(hodge_conservative_ratio.detach())
    metrics["geometric/hodge_solenoidal_direct"] = float(hodge_solenoidal_ratio.detach())
    metrics["geometric/hodge_harmonic_direct"] = float(hodge_harmonic_ratio.detach())
    metrics["geometric/hodge_conservative_exact"] = float(
        rollout_force_diag["hodge_exact"]["conservative_ratio"].mean(),
    )
    metrics["geometric/hodge_solenoidal_exact"] = float(
        rollout_force_diag["hodge_exact"]["solenoidal_ratio"].mean(),
    )
    metrics["geometric/hodge_harmonic_exact"] = float(
        rollout_force_diag["hodge_exact"]["harmonic_ratio"].mean(),
    )
    critic_grad_metrics = _critic_grad_observability_metrics(
        config=config,
        epoch=epoch,
        update_idx=update_idx,
        parameters=[param for param in critic.parameters() if param.requires_grad],
        value_loss=value_loss_term,
        exact_loss=exact_loss_term,
        poisson_loss=poisson_loss_term,
        covector_loss=covector_loss_term,
        stiffness_loss=stiffness_loss_term,
        macro_pullback_loss=macro_pullback_loss_term,
        on_policy_loss=on_policy_loss_term,
    )
    metrics.update(critic_grad_metrics)

    L_macro.backward()
    macro_grad = nn.utils.clip_grad_norm_(list(macro_critic.parameters()), config.grad_clip)
    macro_optimizer.step()
    metrics["macro/grad_norm"] = float(macro_grad)

    if shared_critic:
        (L_wm_total + L_critic).backward()
        critic_grad = _parameter_grad_norm(list(critic.parameters()))
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()
    else:
        L_wm_total.backward()
        wm_grad = nn.utils.clip_grad_norm_(_optimizer_parameters(optimizer_wm), config.grad_clip)
        optimizer_wm.step()
        if optimizer_critic is not None:
            optimizer_critic.zero_grad()
            L_critic.backward()
            critic_grad = nn.utils.clip_grad_norm_(
                list(critic.parameters()),
                config.grad_clip,
            )
            optimizer_critic.step()
        else:
            critic_grad = 0.0
    metrics["wm/grad_norm"] = float(wm_grad)
    metrics["critic/grad_norm"] = float(critic_grad)
    metrics["time/world_model"] = time.perf_counter() - t_section

    t_section = time.perf_counter()
    optimizer_boundary.zero_grad()
    actor_modules = [
        (_unwrap_compiled_module(world_model), False),
        (_unwrap_compiled_module(reward_head), False),
        (_unwrap_compiled_module(critic), False),
        (action_model, False),
        (actor, True),
    ]
    with _temporary_requires_grad(actor_modules):
        actor_obs_z = zn_prev_flat.detach().clone().requires_grad_(True)
        actor_out = actor(
            K_prev_flat.detach(),
            K_code_prev_flat.detach(),
            actor_obs_z,
            hard_routing=current_hard_routing,
            hard_routing_tau=current_tau,
        )
        with torch.no_grad():
            actor_old_out = actor_old(
                K_prev_flat.detach(),
                K_code_prev_flat.detach(),
                zn_prev_flat.detach(),
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )
        L_actor_chart = F.cross_entropy(
            actor_out["action_chart_logits"],
            action_K_prev_flat.detach().long(),
        )
        sample_idx = torch.arange(
            actor_out["action_code_logits"].shape[0], device=actor_obs_z.device
        )
        selected_code_logits = actor_out["action_code_logits"][
            sample_idx,
            action_K_prev_flat.detach().long(),
        ]
        L_actor_code = F.cross_entropy(
            selected_code_logits,
            action_K_code_prev_flat.detach().long(),
        )
        L_actor_zn = F.smooth_l1_loss(
            actor_out["action_z_n"],
            action_zn_prev_flat.detach(),
        )
        L_actor_supervise_raw = L_actor_chart + L_actor_code + L_actor_zn
        (
            _actor_metric_diag,
            actor_metric_inv,
            _actor_scale_certified,
            actor_scale_trust,
            actor_scale_barrier,
            actor_metric_metrics,
        ) = _actor_state_metric(
            config,
            metric=world_model_mod.metric,
            state_z_geo=z_prev_flat.detach(),
            actor_out=actor_out,
            obs_z_n=actor_obs_z,
            target_chart_idx=action_K_prev_flat.detach(),
            target_code_idx=action_K_code_prev_flat.detach(),
            exact_covector=exact_covector,
        )
        metrics.update(actor_metric_metrics)
        L_actor_old_policy_geodesic = hyperbolic_distance(
            actor_out["action_z_geo"],
            actor_old_out["action_z_geo"].detach(),
        ).mean()
        actor_old_policy_geodesic_dist = hyperbolic_distance(
            actor_out["action_z_geo"].detach(),
            actor_old_out["action_z_geo"].detach(),
        ).mean()
        actor_old_policy_chart_kl, actor_old_policy_code_kl = _actor_old_policy_kl_losses(
            actor_out,
            actor_old_out,
        )
        L_actor_old_policy_chart_kl = (
            config.w_actor_old_policy_chart_kl * actor_old_policy_chart_kl
        )
        L_actor_old_policy_code_kl = config.w_actor_old_policy_code_kl * actor_old_policy_code_kl

        actor_update_due = _should_run_actor_update(config, epoch=epoch, update_idx=update_idx)
        L_actor_return = actor_out["action_z_n"].new_zeros(())
        L_actor_natural = actor_out["action_z_n"].new_zeros(())
        L_actor_scale = actor_out["action_z_n"].new_zeros(())
        L_actor_sync = actor_out["action_z_n"].new_zeros(())
        L_actor_stiffness = actor_out["action_z_n"].new_zeros(())
        L_actor_curiosity = actor_out["action_z_n"].new_zeros(())
        actor_return = actor_out["action_z_n"].new_zeros(())
        actor_return_conservative = actor_out["action_z_n"].new_zeros(())
        actor_return_nonconservative = actor_out["action_z_n"].new_zeros(())
        actor_return_macro = actor_out["action_z_n"].new_zeros(())
        actor_curiosity = actor_out["action_z_n"].new_zeros(())
        actor_return_trust = actor_out["action_z_n"].new_zeros(())
        actor_return_gate = actor_out["action_z_n"].new_zeros(())
        actor_exact_control_gate = actor_out["action_z_n"].new_zeros(())
        actor_macro_control_gate = actor_out["action_z_n"].new_zeros(())
        actor_macro_control_weight = actor_out["action_z_n"].new_zeros(())
        actor_curiosity_gate = actor_out["action_z_n"].new_zeros(())
        actor_curiosity_closure_gate = actor_out["action_z_n"].new_zeros(())
        actor_supervise_scale = actor_out["action_z_n"].new_ones(())
        actor_natural_objective = actor_out["action_z_n"].new_zeros(())
        actor_gauge_norm = actor_out["action_z_n"].new_zeros(())
        actor_macro_covector_norm = actor_out["action_z_n"].new_zeros(())
        actor_stiffness_trust = actor_out["action_z_n"].new_zeros(())
        actor_reward_mean = 0.0
        actor_action_canonical_norm_mean = 0.0
        actor_action_abs_mean = 0.0
        actor_action_std = 0.0
        actor_action_saturation_frac = 0.0
        actor_imagined_replay_reward_gap = 0.0
        actor_imagined_replay_return_gap = 0.0
        actor_router_drift = 0.0
        actor_policy_sync_mean = 0.0
        actor_policy_chart_ce = 0.0
        actor_policy_chart_acc = 0.0
        actor_policy_force_rel_err = 0.0
        actor_policy_hodge_cons_exact = 0.0
        actor_policy_reward_noncons_gate = 0.0
        actor_return_applied = False
        actor_curiosity_applied = False
        actor_stiffness_certified = False
        actor_chart_acc = float(
            (actor_out["action_chart_idx"].detach() == action_K_prev_flat.detach()).float().mean()
        )
        actor_code_acc = float(
            (actor_out["action_code_idx"].detach() == action_K_code_prev_flat.detach())
            .float()
            .mean()
        )
        actor_symbol_acc = float(
            (
                (actor_out["action_chart_idx"].detach() == action_K_prev_flat.detach())
                & (actor_out["action_code_idx"].detach() == action_K_code_prev_flat.detach())
            )
            .float()
            .mean()
        )
        actor_batch_size = 0
        metrics["actor/return_trust"] = 0.0
        metrics["actor/return_trust_chart"] = 0.0
        metrics["actor/return_trust_force"] = 0.0
        metrics["actor/return_trust_sync"] = 0.0
        metrics["actor/return_trust_conservative_exact"] = 0.0
        metrics["actor/exact_control_gate"] = 0.0
        metrics["actor/exact_control_increment_rel"] = 0.0
        metrics["actor/exact_control_covector_rel"] = 0.0
        metrics["actor/exact_control_calibration_ratio"] = 0.0
        metrics["actor/exact_control_increment_factor"] = 0.0
        metrics["actor/exact_control_covector_factor"] = 0.0
        metrics["actor/exact_control_calibration_factor"] = 0.0
        metrics["actor/macro_control_gate"] = 0.0
        metrics["actor/macro_control_increment_rel"] = 0.0
        metrics["actor/macro_control_pullback_rel"] = 0.0
        metrics["actor/macro_control_calibration_ratio"] = 0.0
        metrics["actor/macro_control_signal_scale"] = 0.0
        metrics["actor/macro_control_increment_factor"] = 0.0
        metrics["actor/macro_control_pullback_factor"] = 0.0
        metrics["actor/macro_control_calibration_factor"] = 0.0
        metrics["actor/curiosity_gate"] = 0.0
        metrics["actor/curiosity_closure_gate"] = 0.0
        metrics["actor/curiosity_closure_obs_factor"] = 0.0
        metrics["actor/curiosity_closure_defect_acc_factor"] = 0.0
        metrics["actor/curiosity_closure_defect_ce_factor"] = 0.0
        if actor_update_due:
            actor_batch_size = min(int(config.actor_return_batch_size), B)
            if actor_batch_size > 0:
                if actor_batch_size < B:
                    actor_idx = torch.randperm(B, device=z_all.device)[:actor_batch_size]
                    z_actor = z_0[actor_idx]
                    rw_actor = rw_0[actor_idx]
                else:
                    z_actor = z_0
                    rw_actor = rw_0
                actor_rollout = _imagine_actor_return(
                    config,
                    model,
                    _unwrap_compiled_module(world_model),
                    _unwrap_compiled_module(reward_head),
                    critic,
                    macro_critic,
                    actor,
                    action_model,
                    z_actor,
                    rw_actor,
                    horizon=config.actor_return_horizon,
                    gamma=config.gamma,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                    reward_curl_batch_limit=config.reward_curl_batch_limit,
                )
                actor_return_conservative = actor_rollout["objective_conservative"].mean()
                actor_return_nonconservative = actor_rollout["objective_nonconservative"].mean()
                actor_return_macro = actor_rollout["objective_macro"].mean()
                actor_curiosity = actor_rollout["objective_curiosity"].mean()
                actor_router_drift = float(
                    (actor_rollout["rw_traj"].detach() - actor_rollout["rw_states"].detach())
                    .abs()
                    .mean()
                )
                actor_policy_chart_acc = float(actor_rollout["policy_chart_acc"].detach().mean())
                actor_policy_chart_ce = float(actor_rollout["policy_chart_ce"].detach().mean())
                actor_policy_sync_mean = float(actor_rollout["policy_router_sync"].detach().mean())
                actor_policy_force_rel_err = float(
                    actor_rollout["policy_force_rel_err"].detach().mean(),
                )
                actor_policy_hodge_cons_exact = float(
                    actor_rollout["policy_hodge_conservative_exact"].detach().mean(),
                )
                actor_policy_reward_noncons_gate = float(
                    actor_rollout["policy_reward_nonconservative_gate"].detach().mean(),
                )
                actor_return_trust, trust_metrics = _actor_return_trust(
                    config,
                    chart_acc=actor_policy_chart_acc,
                    force_rel_err=actor_policy_force_rel_err,
                    policy_sync_err=actor_policy_sync_mean,
                    hodge_conservative=actor_policy_hodge_cons_exact,
                    template=actor_return_conservative,
                )
                metrics.update(trust_metrics)
                actor_exact_control_gate, exact_control_metrics = _exact_control_gate(
                    config,
                    exact_increment_abs_err=metrics["critic/exact_increment_abs_err"],
                    exact_increment_target_mean=metrics["critic/exact_increment_target_mean"],
                    on_policy_covector_align_abs_err=metrics[
                        "critic/on_policy/covector_align_abs_err"
                    ],
                    on_policy_covector_target_mean=metrics[
                        "critic/on_policy/covector_target_reward_mean"
                    ],
                    on_policy_exact_covector_norm_mean=metrics[
                        "critic/on_policy/exact_covector_norm_mean"
                    ],
                    on_policy_stiffness_target=metrics["critic/on_policy/stiffness_target"],
                    template=actor_return_conservative,
                )
                metrics.update(exact_control_metrics)
                actor_macro_control_gate, macro_control_metrics = _macro_control_gate(
                    config,
                    macro_exact_increment_abs_err=metrics["macro/exact_increment_abs_err"],
                    macro_exact_increment_target_mean=metrics["macro/exact_increment_target_mean"],
                    macro_on_policy_pullback_abs_err=metrics["macro/on_policy/pullback_abs_err"],
                    macro_on_policy_value_std=metrics["macro/on_policy/value_std"],
                    macro_on_policy_exact_increment_pred_abs_mean=metrics[
                        "macro/on_policy/exact_increment_pred_abs_mean"
                    ],
                    macro_target_scale=metrics["macro/on_policy/target_scale"],
                    template=actor_return_conservative,
                )
                metrics.update(macro_control_metrics)
                actor_curiosity_closure_gate, curiosity_closure_metrics = (
                    _actor_curiosity_closure_gate(
                        config,
                        obs_state_acc=metrics["closure/obs_state_acc"],
                        enclosure_defect_acc=metrics["closure/enclosure_defect_acc"],
                        enclosure_defect_ce=metrics["closure/enclosure_defect_ce"],
                        template=actor_return_conservative,
                    )
                )
                metrics.update(curiosity_closure_metrics)
                conservative_control_gate = (
                    actor_return_trust * actor_exact_control_gate * actor_macro_control_gate
                ).clamp(0.0, 1.0)
                actor_curiosity_gate = (
                    actor_return_trust * actor_curiosity_closure_gate * actor_scale_trust
                ).clamp(0.0, 1.0) * (1.0 - conservative_control_gate.detach()).clamp(0.0, 1.0)
                nonconservative_weight = conservative_control_gate.pow(
                    config.actor_return_nonconservative_power,
                )
                actor_macro_control_weight = float(
                    config.actor_macro_backbone_weight
                ) * actor_macro_control_gate.pow(float(config.actor_macro_backbone_power))
                actor_return = (
                    actor_return_conservative
                    + actor_macro_control_weight * actor_return_macro
                    + nonconservative_weight * actor_return_nonconservative
                )
                rollout_velocity = actor_rollout["z_traj"].reshape(
                    -1, config.latent_dim
                ) - actor_rollout["z_states"].reshape(-1, config.latent_dim)
                rollout_exact_covector = actor_rollout["exact_covector"].reshape(
                    -1, config.latent_dim
                )
                rollout_macro_covector = actor_rollout["macro_covector"].reshape(
                    -1, config.latent_dim
                )
                rollout_reward_form_cov = actor_rollout["reward_form_cov"].reshape(
                    -1,
                    config.latent_dim,
                )
                rollout_state_z = actor_rollout["z_states"].reshape(-1, config.latent_dim)
                gauge_covector = (
                    rollout_exact_covector
                    + actor_macro_control_weight * rollout_macro_covector
                    - nonconservative_weight * rollout_reward_form_cov
                )
                actor_natural_objective = (
                    -((gauge_covector * rollout_velocity) * actor_metric_inv.unsqueeze(0))
                    .sum(dim=-1)
                    .mean()
                )
                actor_macro_covector_norm = (
                    _metric_covector_norm_sq(
                        world_model_mod.metric,
                        rollout_state_z.detach(),
                        rollout_macro_covector,
                    )
                    .sqrt()
                    .mean()
                )
                actor_gauge_norm = (
                    _metric_covector_norm_sq(
                        world_model_mod.metric,
                        rollout_state_z.detach(),
                        gauge_covector,
                    )
                    .sqrt()
                    .mean()
                )
                L_actor_sync = config.w_actor_wm_sync * actor_rollout["policy_sync_loss"].mean()
                actor_stiffness_scale = max(float(config.actor_stiffness_min), 1e-8)
                actor_stiffness_deficit = (actor_stiffness_scale - actor_gauge_norm).clamp(
                    min=0.0
                ) / actor_stiffness_scale
                actor_stiffness_trust = torch.exp(-actor_stiffness_deficit)
                L_actor_stiffness = config.w_actor_stiffness * (actor_stiffness_deficit.pow(2))
                L_actor_scale = config.w_actor_scale_barrier * actor_scale_barrier.pow(2)
                L_actor_curiosity = (
                    -config.w_actor_curiosity * actor_curiosity_gate * actor_curiosity
                )
                actor_curiosity_applied = (
                    float(actor_curiosity_gate.detach()) > 0.0
                    and float(config.w_actor_curiosity) > 0.0
                )
                actor_stiffness_certified = (
                    float(actor_gauge_norm.detach()) >= actor_stiffness_scale
                )
                if float(actor_return_trust.detach()) >= config.actor_return_trust_min:
                    actor_return_gate = (
                        conservative_control_gate * actor_scale_trust * actor_stiffness_trust
                    ).clamp(0.0, 1.0)
                    L_actor_return = -config.w_actor_return * actor_return_gate * actor_return
                    L_actor_natural = (
                        -config.w_actor_natural
                        * actor_return_gate.pow(config.actor_return_exact_control_power)
                        * actor_natural_objective
                    )
                    actor_return_applied = float(actor_return_gate.detach()) > 0.0
                action_samples = actor_rollout["actions"].detach()
                actor_reward_mean = float(actor_rollout["rewards"].detach().mean())
                actor_action_canonical_norm_mean = float(
                    actor_rollout["action_canonicals"].detach().norm(dim=-1).mean()
                )
                actor_action_abs_mean = float(action_samples.abs().mean())
                actor_action_std = float(action_samples.std(unbiased=False))
                actor_action_saturation_frac = float((action_samples.abs() >= 0.95).float().mean())
                actor_imagined_replay_reward_gap = (
                    actor_reward_mean - metrics["critic/replay_reward_mean"]
                )
                actor_imagined_replay_return_gap = (
                    float(actor_return.detach()) - metrics["critic/replay_return_mean"]
                )
            else:
                actor_update_due = False
        actor_supervise_scale, actor_supervise_metrics = _actor_supervise_scale(
            config,
            epoch=epoch,
            actor_return_gate=torch.maximum(actor_return_gate, actor_curiosity_gate),
        )
        metrics.update(actor_supervise_metrics)
        L_actor_supervise = (
            config.w_action_latent_supervise * actor_supervise_scale * L_actor_supervise_raw
        )
        L_actor_total = (
            L_actor_supervise
            + config.w_actor_geodesic * L_actor_old_policy_geodesic
            + L_actor_old_policy_chart_kl
            + L_actor_old_policy_code_kl
            + L_actor_scale
            + L_actor_sync
            + L_actor_stiffness
            + L_actor_curiosity
            + L_actor_natural
            + L_actor_return
        )
        L_actor_total.backward()
    actor_params = _optimizer_parameters(optimizer_boundary)
    actor_trust_region_scale, actor_param_norm, actor_step_norm, actor_max_step = (
        _relative_trust_region_scale(
            optimizer_boundary,
            actor_params,
            kappa=config.actor_trust_region_kappa,
            epsilon_theta=config.actor_trust_region_eps,
        )
    )
    actor_grad = nn.utils.clip_grad_norm_(actor.parameters(), config.grad_clip)
    optimizer_boundary.step()
    actor_old.load_state_dict(actor.state_dict())
    metrics["actor/L_total"] = float(L_actor_total.detach())
    metrics["actor/L_supervise_raw"] = float(L_actor_supervise_raw.detach())
    metrics["actor/L_supervise"] = float(L_actor_supervise.detach())
    metrics["actor/L_old_policy_geodesic"] = float(L_actor_old_policy_geodesic.detach())
    metrics["actor/L_old_policy_chart_kl"] = float(L_actor_old_policy_chart_kl.detach())
    metrics["actor/L_old_policy_code_kl"] = float(L_actor_old_policy_code_kl.detach())
    metrics["actor/L_scale"] = float(L_actor_scale.detach())
    metrics["actor/L_natural"] = float(L_actor_natural.detach())
    metrics["actor/L_sync"] = float(L_actor_sync.detach())
    metrics["actor/L_stiffness"] = float(L_actor_stiffness.detach())
    metrics["actor/L_curiosity"] = float(L_actor_curiosity.detach())
    metrics["actor/L_chart"] = float(L_actor_chart.detach())
    metrics["actor/L_code"] = float(L_actor_code.detach())
    metrics["actor/L_zn"] = float(L_actor_zn.detach())
    metrics["actor/L_return"] = float(L_actor_return.detach())
    metrics["actor/return_mean"] = float(actor_return.detach())
    metrics["actor/return_conservative_mean"] = float(actor_return_conservative.detach())
    metrics["actor/return_nonconservative_mean"] = float(actor_return_nonconservative.detach())
    metrics["actor/return_macro_mean"] = float(actor_return_macro.detach())
    metrics["actor/curiosity_mean"] = float(actor_curiosity.detach())
    metrics["actor/natural_objective_mean"] = float(actor_natural_objective.detach())
    metrics["actor/gauge_covector_norm_mean"] = float(actor_gauge_norm.detach())
    metrics["actor/macro_covector_norm_mean"] = float(actor_macro_covector_norm.detach())
    metrics["actor/old_policy_geodesic_mean"] = float(actor_old_policy_geodesic_dist.detach())
    metrics["actor/old_policy_chart_kl"] = float(actor_old_policy_chart_kl.detach())
    metrics["actor/old_policy_code_kl"] = float(actor_old_policy_code_kl.detach())
    metrics["actor/reward_mean"] = actor_reward_mean
    metrics["actor/action_canonical_norm_mean"] = actor_action_canonical_norm_mean
    metrics["actor/action_abs_mean"] = actor_action_abs_mean
    metrics["actor/action_std"] = actor_action_std
    metrics["actor/action_saturation_frac"] = actor_action_saturation_frac
    metrics["actor/imagined_replay_reward_gap"] = actor_imagined_replay_reward_gap
    metrics["actor/imagined_replay_return_gap"] = actor_imagined_replay_return_gap
    metrics["actor/chart_acc"] = actor_chart_acc
    metrics["actor/code_acc"] = actor_code_acc
    metrics["actor/symbol_acc"] = actor_symbol_acc
    metrics["actor/policy_chart_acc"] = actor_policy_chart_acc
    metrics["actor/policy_chart_ce"] = actor_policy_chart_ce
    metrics["actor/policy_router_sync_mean"] = actor_policy_sync_mean
    metrics["actor/policy_force_rel_err_mean"] = actor_policy_force_rel_err
    metrics["actor/policy_hodge_conservative_exact_mean"] = actor_policy_hodge_cons_exact
    metrics["actor/policy_reward_nonconservative_gate_mean"] = actor_policy_reward_noncons_gate
    metrics["actor/grad_norm"] = float(actor_grad)
    metrics["actor/trust_region_scale"] = actor_trust_region_scale
    metrics["actor/param_norm"] = actor_param_norm
    metrics["actor/step_norm"] = actor_step_norm
    metrics["actor/max_step_norm"] = actor_max_step
    metrics["actor/update_due"] = 1.0 if actor_update_due else 0.0
    metrics["actor/update_applied"] = 1.0
    metrics["actor/return_applied"] = 1.0 if actor_return_applied else 0.0
    metrics["actor/horizon"] = float(config.actor_return_horizon if actor_update_due else 0.0)
    metrics["actor/batch_size"] = float(actor_batch_size)
    metrics["actor/router_drift"] = actor_router_drift
    metrics["actor/return_trust_min"] = float(config.actor_return_trust_min)
    metrics["actor/return_trust_used"] = float(actor_return_trust.detach())
    metrics["actor/return_gate"] = float(actor_return_gate.detach())
    metrics["actor/macro_control_gate"] = float(actor_macro_control_gate.detach())
    metrics["actor/macro_backbone_weight"] = float(actor_macro_control_weight.detach())
    metrics["actor/curiosity_gate"] = float(actor_curiosity_gate.detach())
    metrics["actor/curiosity_closure_gate"] = float(actor_curiosity_closure_gate.detach())
    metrics["actor/curiosity_applied"] = 1.0 if actor_curiosity_applied else 0.0
    metrics["actor/regime_bc_weight"] = float(
        (config.w_action_latent_supervise * actor_supervise_scale).detach(),
    )
    metrics["actor/regime_rl_weight"] = float(actor_return_gate.detach())
    metrics["actor/stiffness_trust"] = float(actor_stiffness_trust.detach())
    metrics["actor/stiffness_certified"] = 1.0 if actor_stiffness_certified else 0.0
    metrics["time/actor"] = time.perf_counter() - t_section

    if compute_diagnostics:
        t_section = time.perf_counter()
        imagination = _imagine(
            model,
            _unwrap_compiled_module(world_model),
            _unwrap_compiled_module(reward_head),
            critic,
            actor,
            action_model,
            z_0,
            rw_0,
            config.imagination_horizon,
            config.gamma,
            reward_curl_batch_limit=config.reward_curl_batch_limit,
            hard_routing=current_hard_routing,
            hard_routing_tau=current_tau,
        )
        imag_rw_states = imagination["rw_states"]
        imag_action_canonicals = imagination["action_canonicals"]
        imag_action_latents = imagination["action_latents"]
        imag_action_router = imagination["action_router_weights"]
        imag_rewards = imagination["rewards"]
        imag_reward_cons = imagination["reward_conservative"]
        imag_reward_noncons = imagination["reward_nonconservative"]
        imag_reward_curl = imagination["reward_curl_norm"]
        imag_reward_curl_valid = imagination["reward_curl_valid"]
        imag_z = imagination["z_traj"]
        imag_rw = imagination["rw_traj"]
        imag_actions = imagination["actions"]
        discounted_rewards = _discounted_sum(imag_rewards, config.gamma)
        discounted_nonconservative = _discounted_sum(imag_reward_noncons, config.gamma)
        discounted_exact_boundary = _discounted_sum(imag_reward_cons, config.gamma)
        terminal_value = critic_value(critic, imag_z[:, -1], imag_rw[:, -1]).squeeze(-1)
        full_return = discounted_rewards
        replay_values_all = critic_value(
            critic,
            z_all.reshape(-1, config.latent_dim),
            rw_all.reshape(-1, config.num_charts),
        ).reshape(B, T + 1)
        replay_values_diag = replay_values_all[:, :-1]
        replay_next_values = replay_values_all[:, 1:]
        replay_delta = (
            replay_cons_target
            + config.gamma * (1.0 - replay_dones) * replay_next_values
            - replay_values_diag
        )
        cal_err, cal_max, cal_bins = _value_calibration_error(
            replay_values_diag,
            replay_rtg,
            replay_valid,
        )
        imag_action_router_entropy = -(
            imag_action_router * imag_action_router.clamp(min=1e-8).log()
        ).sum(dim=-1)
        metrics["policy/action_canonical_norm_mean"] = float(
            imag_action_canonicals.norm(dim=-1).mean()
        )
        metrics["policy/action_abs_mean"] = float(imag_actions.abs().mean())
        metrics["policy/action_latent_norm_mean"] = float(imag_action_latents.norm(dim=-1).mean())
        metrics["policy/action_router_entropy"] = float(imag_action_router_entropy.mean())
        metrics["imagination/reward_mean"] = float(imag_rewards.mean())
        metrics["imagination/reward_std"] = float(imag_rewards.std(unbiased=False))
        metrics["imagination/reward_conservative_mean"] = float(imag_reward_cons.mean())
        metrics["imagination/reward_nonconservative_mean"] = float(imag_reward_noncons.mean())
        if imag_reward_curl_valid.any():
            metrics["imagination/reward_curl_norm_mean"] = float(
                _masked_mean(imag_reward_curl, imag_reward_curl_valid.float())
            )
        else:
            metrics["imagination/reward_curl_norm_mean"] = 0.0
        metrics["imagination/reward_curl_eval_frac"] = float(imag_reward_curl_valid.float().mean())
        metrics["imagination/discounted_reward_mean"] = float(discounted_rewards.mean())
        metrics["imagination/nonconservative_return_mean"] = float(
            discounted_nonconservative.mean()
        )
        metrics["imagination/exact_boundary_mean"] = float(discounted_exact_boundary.mean())
        metrics["imagination/terminal_value_mean"] = float(terminal_value.mean())
        metrics["imagination/return_mean"] = float(full_return.mean())
        metrics["imagination/full_return_mean"] = float(full_return.mean())
        metrics["imagination/router_entropy"] = float(
            -(imag_rw_states * imag_rw_states.clamp(min=1e-8).log()).sum(dim=-1).mean()
        )
        metrics["imagination/router_drift"] = float((imag_rw - imag_rw_states).abs().mean())
        metrics["imagination/policy_chart_acc"] = float(imagination["policy_chart_acc"].mean())
        metrics["imagination/policy_router_sync_mean"] = float(
            imagination["policy_router_sync"].mean(),
        )
        metrics["imagination/policy_force_rel_err_mean"] = float(
            imagination["policy_force_rel_err"].mean(),
        )
        metrics["imagination/hodge_conservative_exact_mean"] = float(
            imagination["policy_hodge_conservative_exact"].mean(),
        )
        metrics["critic/value_bias"] = float(_masked_mean(replay_gap, replay_valid))
        metrics["critic/value_mean"] = float(_masked_mean(replay_values_diag, replay_valid))
        metrics["critic/replay_bellman_abs"] = float(
            _masked_mean(replay_delta.abs(), replay_valid)
        )
        bellman_centered = replay_delta - _masked_mean(replay_delta, replay_valid)
        metrics["critic/replay_bellman_std"] = float(
            (_masked_mean(bellman_centered.pow(2), replay_valid) + 1e-8).sqrt()
        )
        metrics["critic/replay_rtg_abs_err"] = float(_masked_mean(replay_gap.abs(), replay_valid))
        metrics["critic/replay_calibration_err"] = float(cal_err)
        metrics["critic/replay_calibration_max"] = float(cal_max)
        metrics["critic/replay_calibration_bins"] = cal_bins
        metrics["train/replay_horizon"] = float(T)
        metrics["train/wm_horizon"] = float(T_wm)
        metrics["geometric/z_norm_mean"] = float(z_all.norm(dim=-1).mean())
        metrics["geometric/z_norm_max"] = float(z_all.norm(dim=-1).max())
        metrics["geometric/action_z_norm_mean"] = float(action_z_all.norm(dim=-1).mean())
        metrics["geometric/jump_frac"] = float(wm_out["jumped"].float().mean())
        metrics["geometric/energy_var"] = float(wm_out["energy_var"])

        obs_centers = project_to_ball(model.encoder.chart_centers.detach())
        action_centers = project_to_ball(action_model.encoder.chart_centers.detach())
        world_model_mod = _unwrap_compiled_module(world_model)
        reward_head_mod = _unwrap_compiled_module(reward_head)
        metrics["chart/wm_center_drift"] = float(
            (
                obs_centers
                - project_to_ball(world_model_mod.potential_net.chart_tok.chart_centers.detach())
            )
            .norm(dim=-1)
            .mean()
        )
        metrics["action_chart/actor_center_drift"] = float(
            (action_centers - project_to_ball(actor.action_chart_centers.detach()))
            .norm(dim=-1)
            .mean()
        )
        metrics["action_chart/reward_center_drift"] = float(
            (
                action_centers
                - project_to_ball(reward_head_mod.action_chart_tok.chart_centers.detach())
            )
            .norm(dim=-1)
            .mean()
        )
        metrics["time/diagnostics"] = time.perf_counter() - t_section
    else:
        metrics["time/diagnostics"] = 0.0

    metrics["time/step"] = time.perf_counter() - step_t0
    return metrics


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------


def train(config: DreamerConfig) -> None:
    """Run Geometric Dreamer training."""
    device = torch.device(config.device)
    print(f"Device: {device}")
    if config.matmul_precision:
        torch.set_float32_matmul_precision(config.matmul_precision)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.manual_seed(config.seed)
    obs_normalizer: ObservationNormalizer | None = None

    applied_preset, preset_changes = _apply_task_preset(config)
    if applied_preset is not None and preset_changes:
        change_items = ", ".join(
            f"{name}={old}->{new}" for name, (old, new) in sorted(preset_changes.items())
        )
        print(f"Applied task preset {applied_preset}: {change_items}")

    # --- MLflow ---
    mlflow_enabled = False
    if config.mlflow and MLFLOW_AVAILABLE and mlflow is not None:
        if config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
        mlflow.set_experiment(config.mlflow_experiment)
        mlflow.start_run()
        try:
            safe = {k: str(v) for k, v in asdict(config).items()}
            mlflow.log_params(safe)
        except Exception:
            pass
        mlflow_enabled = True

    # --- Environment ---
    env = _make_env(config.domain, config.task)
    collect_env = None
    time_step = env.reset()
    actual_obs_dim = len(_flatten_obs(time_step))
    actual_action_dim = _infer_action_dim(env)
    if actual_obs_dim != config.obs_dim:
        print(f"Overriding obs_dim: {config.obs_dim} -> {actual_obs_dim}")
        config.obs_dim = actual_obs_dim
    if actual_action_dim != config.action_dim:
        print(f"Overriding action_dim: {config.action_dim} -> {actual_action_dim}")
        config.action_dim = actual_action_dim

    print(
        f"Environment: {config.domain}-{config.task}  "
        f"obs_dim={config.obs_dim}  action_dim={config.action_dim}"
    )

    # --- Models ---
    model = SharedDynTopoEncoder(
        input_dim=config.obs_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
        commitment_beta=config.commitment_beta,
        codebook_loss_weight=config.codebook_loss_weight,
    ).to(device)
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
    ).to(device)
    action_model = SharedDynTopoEncoder(
        input_dim=config.action_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
        commitment_beta=config.commitment_beta,
        codebook_loss_weight=config.codebook_loss_weight,
    ).to(device)
    action_jump_op = FactorizedJumpOperator(
        num_charts=config.num_action_charts,
        latent_dim=config.latent_dim,
    ).to(device)

    world_model = GeometricWorldModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        control_dim=config.latent_dim,
        num_charts=config.num_charts,
        d_model=config.d_model,
        hidden_dim=config.hidden_dim,
        dt=config.wm_dt,
        gamma_friction=config.wm_gamma_friction,
        T_c=config.wm_T_c,
        alpha_potential=config.wm_alpha_potential,
        beta_curl=config.wm_beta_curl,
        gamma_risk=config.wm_gamma_risk,
        use_boris=config.wm_use_boris,
        use_jump=config.wm_use_jump,
        n_refine_steps=config.wm_n_refine_steps,
        jump_beta=config.wm_jump_beta,
        min_length=config.wm_min_length,
        risk_metric_alpha=config.wm_risk_metric_alpha,
    ).to(device)

    actor = GeometricActor(
        latent_dim=config.latent_dim,
        num_obs_charts=config.num_charts,
        obs_codes_per_chart=config.codes_per_chart,
        num_action_charts=config.num_action_charts,
        action_codes_per_chart=config.action_codes_per_chart,
        d_model=config.d_model,
    ).to(device)
    actor_old = copy.deepcopy(actor).to(device)
    actor_old.eval()
    for param in actor_old.parameters():
        param.requires_grad_(False)
    critic = world_model.potential_net
    macro_critic = MacroValueModel(
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        num_action_charts=config.num_action_charts,
        action_codes_per_chart=config.action_codes_per_chart,
    ).to(device)

    reward_head = RewardHead(
        potential_net=world_model.potential_net,
        num_action_charts=config.num_action_charts,
        d_model=config.d_model,
    ).to(device)
    enclosure_probe = EnclosureProbe(
        chart_dim=config.latent_dim,
        action_dim=config.latent_dim,
        ztex_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        hidden_dim=config.enclosure_hidden_dim,
        alpha=config.enclosure_grl_alpha_max,
    ).to(device)

    world_model = _maybe_compile_module(
        world_model,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    reward_head = _maybe_compile_module(
        reward_head,
        enabled=config.torch_compile,
        mode=config.torch_compile_mode,
    )
    shared_critic = _shared_value_field(world_model, critic)

    # --- Parameter counts ---
    def _count(m: nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    print(f"Encoder:     {_count(model.encoder):,} params")
    print(f"Decoder:     {_count(model.decoder):,} params")
    print(f"Jump op:     {_count(jump_op):,} params")
    print(f"Act enc:     {_count(action_model.encoder):,} params")
    print(f"Act dec:     {_count(action_model.decoder):,} params")
    print(f"Act jump:    {_count(action_jump_op):,} params")
    print(f"World model: {_count(world_model):,} params")
    print(f"Actor:       {_count(actor):,} params")
    print(f"Macro critic:{_count(macro_critic):,} params")
    print(f"Enclosure:   {_count(enclosure_probe):,} params")
    if shared_critic:
        print(f"Value field: shared in world model ({_count(critic):,} params)")
    else:
        print(f"Critic:      {_count(critic):,} params")
    print(f"Reward head: {_count(reward_head):,} params  (chart_tok/z_embed shared)")

    # --- Optimizers ---
    encoder_groups = build_encoder_param_groups(
        model,
        jump_op,
        base_lr=config.lr_encoder,
        lr_chart_centers_scale=config.lr_chart_centers_scale,
        lr_codebook_scale=config.lr_codebook_scale,
    )
    encoder_groups.extend(
        build_encoder_param_groups(
            action_model,
            action_jump_op,
            base_lr=config.lr_encoder,
            lr_chart_centers_scale=config.lr_chart_centers_scale,
            lr_codebook_scale=config.lr_codebook_scale,
        ),
    )
    optimizer_enc = torch.optim.Adam(encoder_groups)

    # WM optimizer: world_model + reward_head (minus shared params)
    reward_head_mod = _unwrap_compiled_module(reward_head)
    reward_shared_ids = {
        *(id(p) for p in reward_head_mod.chart_tok.parameters()),
        *(id(p) for p in reward_head_mod.z_embed.parameters()),
    }
    reward_own_params = [p for p in reward_head_mod.parameters() if id(p) not in reward_shared_ids]
    wm_param_groups = [
        {"params": world_model.parameters(), "lr": config.lr_wm},
        {"params": reward_own_params, "lr": config.lr_wm},
    ]
    optimizer_wm = torch.optim.Adam(wm_param_groups)
    optimizer_macro = torch.optim.Adam(
        macro_critic.parameters(),
        lr=config.lr_wm * float(config.macro_lr_multiplier),
    )

    optimizer_boundary = torch.optim.Adam(
        actor.parameters(),
        lr=config.lr_actor,
    )
    optimizer_enclosure = torch.optim.Adam(
        enclosure_probe.parameters(),
        lr=config.lr_wm,
    )
    scheduler_enc = _make_cosine_scheduler(optimizer_enc, config.total_epochs, config.lr_min)
    scheduler_wm = _make_cosine_scheduler(optimizer_wm, config.total_epochs, config.lr_min)
    scheduler_macro = _make_cosine_scheduler(
        optimizer_macro,
        config.total_epochs,
        config.lr_min,
    )
    scheduler_boundary = _make_cosine_scheduler(
        optimizer_boundary,
        config.total_epochs,
        config.lr_min,
    )
    scheduler_enclosure = _make_cosine_scheduler(
        optimizer_enclosure,
        config.total_epochs,
        config.lr_min,
    )
    optimizer_critic = None
    scheduler_critic = None
    if not shared_critic:
        optimizer_critic = torch.optim.Adam(
            critic.parameters(),
            lr=config.lr_wm,
        )
        scheduler_critic = _make_cosine_scheduler(
            optimizer_critic,
            config.total_epochs,
            config.lr_min,
        )

    # --- Phase 1 config for encoder losses ---
    phase1_state_obs = init_phase1_adaptive_state(config)
    phase1_state_action = init_phase1_adaptive_state(config)
    phase1_cfg = _phase1_config(config, phase1_state_obs)
    action_phase1_cfg = _phase1_config(
        config,
        phase1_state_action,
        input_dim=config.action_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
    )

    # --- Load checkpoint ---
    start_epoch = 0
    if config.load_checkpoint and os.path.exists(config.load_checkpoint):
        ckpt = torch.load(config.load_checkpoint, map_location=device, weights_only=False)
        required_keys = [
            "encoder",
            "decoder",
            "jump_op",
            "action_model",
            "action_jump_op",
            "world_model",
            "actor",
            "actor_old",
            "macro_critic",
            "reward_head",
            "enclosure_probe",
            "optimizer_enc",
            "optimizer_wm",
            "optimizer_macro",
            "optimizer_boundary",
            "optimizer_enclosure",
            "scheduler_enc",
            "scheduler_wm",
            "scheduler_macro",
            "scheduler_boundary",
            "scheduler_enclosure",
        ]
        if not shared_critic:
            required_keys.extend(["critic", "optimizer_critic", "scheduler_critic"])
        missing_keys = [key for key in required_keys if key not in ckpt]
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            msg = (
                f"Checkpoint {config.load_checkpoint} is missing required keys: {missing}. "
                "Dreamer checkpoint loading is strict and does not support legacy partial state."
            )
            raise KeyError(msg)
        if config.normalize_observations and ckpt.get("obs_normalizer") is not None:
            obs_normalizer = ObservationNormalizer.from_state_dict(
                ckpt["obs_normalizer"],
                device=device,
            )
        model.encoder.load_state_dict(ckpt["encoder"])
        model.decoder.load_state_dict(ckpt["decoder"])
        jump_op.load_state_dict(ckpt["jump_op"])
        action_model.load_state_dict(ckpt["action_model"])
        action_jump_op.load_state_dict(ckpt["action_jump_op"])
        _unwrap_compiled_module(world_model).load_state_dict(ckpt["world_model"])
        actor.load_state_dict(ckpt["actor"])
        actor_old.load_state_dict(ckpt["actor_old"])
        macro_critic.load_state_dict(ckpt["macro_critic"])
        if not shared_critic:
            _unwrap_compiled_module(critic).load_state_dict(ckpt["critic"])
        _unwrap_compiled_module(reward_head).load_state_dict(ckpt["reward_head"])
        enclosure_probe.load_state_dict(ckpt["enclosure_probe"])
        optimizer_enc.load_state_dict(ckpt["optimizer_enc"])
        optimizer_wm.load_state_dict(ckpt["optimizer_wm"])
        optimizer_macro.load_state_dict(ckpt["optimizer_macro"])
        optimizer_boundary.load_state_dict(ckpt["optimizer_boundary"])
        optimizer_enclosure.load_state_dict(ckpt["optimizer_enclosure"])
        if optimizer_critic is not None:
            optimizer_critic.load_state_dict(ckpt["optimizer_critic"])
        scheduler_enc.load_state_dict(ckpt["scheduler_enc"])
        scheduler_wm.load_state_dict(ckpt["scheduler_wm"])
        scheduler_macro.load_state_dict(ckpt["scheduler_macro"])
        scheduler_boundary.load_state_dict(ckpt["scheduler_boundary"])
        scheduler_enclosure.load_state_dict(ckpt["scheduler_enclosure"])
        if scheduler_critic is not None:
            scheduler_critic.load_state_dict(ckpt["scheduler_critic"])
        start_epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint from {config.load_checkpoint} (epoch {start_epoch})")
    actor_old.eval()
    for param in actor_old.parameters():
        param.requires_grad_(False)

    if config.freeze_encoder:
        model.encoder.eval()
        model.decoder.eval()
        action_model.encoder.eval()
        action_model.decoder.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for p in action_model.parameters():
            p.requires_grad_(False)
        jump_op.eval()
        action_jump_op.eval()
        for p in jump_op.parameters():
            p.requires_grad_(False)
        for p in action_jump_op.parameters():
            p.requires_grad_(False)
        print("Observation and action topoencoders frozen.")

        _sync_rl_atlas(model, action_model, world_model, critic, actor, reward_head, actor_old)
    print("Bound RL chart centers to encoder atlas.")

    # --- Replay buffer ---
    buffer = SequenceReplayBuffer(
        capacity=config.buffer_capacity,
        seq_len=config.seq_len,
    )

    # --- Seed episodes (random policy) ---
    print(f"Collecting {config.seed_episodes} seed episodes...")
    seed_episodes_data: list[dict[str, np.ndarray]] = []
    seed_count = 0
    while seed_count < config.seed_episodes:
        batch_episodes = min(
            config.seed_episodes - seed_count,
            config.collect_n_env_workers if collect_env is not None else 1,
        )
        if collect_env is not None:
            episodes = _collect_parallel_episodes(
                collect_env,
                None,
                None,
                model,
                device,
                config.latent_dim,
                config.num_action_charts,
                num_episodes=batch_episodes,
                obs_normalizer=obs_normalizer,
                action_repeat=config.action_repeat,
                max_steps=config.max_episode_steps,
                hard_routing=config.hard_routing,
                hard_routing_tau=config.hard_routing_tau,
            )
        else:
            episodes = [
                _collect_episode(
                    env,
                    None,
                    None,
                    model,
                    device,
                    config.latent_dim,
                    config.num_action_charts,
                    obs_normalizer=obs_normalizer,
                    action_repeat=config.action_repeat,
                    max_steps=config.max_episode_steps,
                    hard_routing=config.hard_routing,
                    hard_routing_tau=config.hard_routing_tau,
                ),
            ]
        for ep in episodes:
            buffer.add_episode(ep)
            seed_episodes_data.append(ep)
            seed_count += 1
            ep_r = ep["rewards"].sum()
            print(
                f"  Seed {seed_count}/{config.seed_episodes}: reward={ep_r:.1f}  len={len(ep['obs'])}"
            )

    if config.normalize_observations and obs_normalizer is None:
        if not seed_episodes_data:
            msg = (
                "Observation normalization requires at least one seed episode or checkpoint stats."
            )
            raise ValueError(msg)
        obs_normalizer = ObservationNormalizer.from_episodes(
            seed_episodes_data,
            device=device,
            min_std=config.obs_norm_min_std,
        )
        std_cpu = obs_normalizer.std.detach().cpu()
        print(
            "Observation normalization: "
            f"min_std={std_cpu.min().item():.4f}  "
            f"mean_std={std_cpu.mean().item():.4f}  "
            f"max_std={std_cpu.max().item():.4f}",
        )

    # --- Main training loop ---
    total_env_steps = buffer.total_steps
    episode_rewards: list[float] = []
    best_eval_reward = -float("inf")
    last_exact_control_gate = 0.0

    tpb = config.batch_size * max(config.seq_len, 1)
    est_upd = (
        config.updates_per_epoch
        if config.updates_per_epoch > 0
        else max(1, buffer.total_steps // tpb)
    )
    print(
        f"\nStarting training for {config.total_epochs} epochs "
        f"(~{est_upd} updates/epoch, buffer={buffer.total_steps} steps, "
        f"capacity={config.buffer_capacity})"
    )
    print("=" * 80)

    for epoch in range(start_epoch, config.total_epochs):
        t0 = time.perf_counter()
        collect_time = 0.0
        current_hard_routing = _use_hard_routing(config, epoch)
        current_tau = _get_hard_routing_tau(config, epoch, config.total_epochs)
        current_sigma_motor, sigma_metrics = _scheduled_sigma_motor(
            config,
            epoch=epoch,
            exact_control_gate=last_exact_control_gate,
        )

        # --- Data collection ---
        if epoch % config.collect_every == 0:
            collect_t0 = time.perf_counter()
            if collect_env is not None:
                episodes = _collect_parallel_episodes(
                    collect_env,
                    actor,
                    action_model,
                    model,
                    device,
                    config.latent_dim,
                    config.num_action_charts,
                    num_episodes=config.collect_n_env_workers,
                    obs_normalizer=obs_normalizer,
                    action_repeat=config.action_repeat,
                    max_steps=config.max_episode_steps,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                    sigma_motor=current_sigma_motor,
                )
            else:
                episodes = [
                    _collect_episode(
                        env,
                        actor,
                        action_model,
                        model,
                        device,
                        config.latent_dim,
                        config.num_action_charts,
                        obs_normalizer=obs_normalizer,
                        action_repeat=config.action_repeat,
                        max_steps=config.max_episode_steps,
                        hard_routing=current_hard_routing,
                        hard_routing_tau=current_tau,
                        sigma_motor=current_sigma_motor,
                    ),
                ]
            for ep in episodes:
                buffer.add_episode(ep)
                ep_reward = ep["rewards"].sum()
                episode_rewards.append(ep_reward)
                total_env_steps += len(ep["obs"])
            collect_time += time.perf_counter() - collect_t0

        # --- Training steps (multiple updates per epoch) ---
        if not config.freeze_encoder:
            model.train()
            jump_op.train()
            action_model.train()
            action_jump_op.train()
        world_model.train()
        actor.train()
        critic.train()
        macro_critic.train()
        reward_head.train()

        tokens_per_batch = config.batch_size * max(config.seq_len, 1)
        if config.updates_per_epoch > 0:
            n_updates = config.updates_per_epoch
        else:
            n_updates = max(1, buffer.total_steps // tokens_per_batch)

        phase1_cfg = _phase1_config(config, phase1_state_obs)
        action_phase1_cfg = _phase1_config(
            config,
            phase1_state_action,
            input_dim=config.action_dim,
            num_charts=config.num_action_charts,
            codes_per_chart=config.action_codes_per_chart,
        )

        metrics_accum: dict[str, list[float]] = {}
        sample_time_total = 0.0
        update_time_total = 0.0
        for u in range(n_updates):
            sample_t0 = time.perf_counter()
            batch = buffer.sample(config.batch_size, device=device)
            sample_time_total += time.perf_counter() - sample_t0
            step_t0 = time.perf_counter()
            compute_diagnostics = (
                config.diagnostics_every_updates <= 1
                or ((u + 1) % config.diagnostics_every_updates == 0)
                or (u == n_updates - 1)
            )

            step_metrics = _train_step(
                model,
                jump_op,
                action_model,
                action_jump_op,
                world_model,
                enclosure_probe,
                reward_head,
                critic,
                macro_critic,
                actor,
                actor_old,
                optimizer_enc,
                optimizer_wm,
                optimizer_critic,
                optimizer_boundary,
                optimizer_enclosure,
                batch,
                config,
                phase1_cfg,
                action_phase1_cfg,
                epoch,
                current_hard_routing,
                current_tau,
                update_idx=epoch * max(n_updates, 1) + u,
                obs_normalizer=obs_normalizer,
                compute_diagnostics=compute_diagnostics,
                optimizer_macro=optimizer_macro,
            )
            update_time_total += time.perf_counter() - step_t0
            for k, v in step_metrics.items():
                metrics_accum.setdefault(k, []).append(v)

        # Average metrics across updates
        metrics = {k: float(np.mean(v)) for k, v in metrics_accum.items()}
        metrics.update(sigma_metrics)
        metrics["train/updates"] = float(n_updates)
        metrics["time/sample"] = sample_time_total / max(n_updates, 1)
        metrics["time/sample_total"] = sample_time_total
        metrics["time/update_total"] = update_time_total
        metrics["time/collection"] = collect_time
        metrics["time/diagnostics_interval"] = float(max(1, config.diagnostics_every_updates))
        controller_metrics_obs = _phase1_controller_metrics(
            metrics,
            enc_prefix="enc",
            chart_prefix="chart",
            num_charts=config.num_charts,
        )
        controller_metrics_action = _phase1_controller_metrics(
            metrics,
            enc_prefix="enc_action",
            chart_prefix="action_chart",
            num_charts=config.num_action_charts,
        )
        update_phase1_adaptive_state(
            phase1_state_obs,
            config,
            metrics,
            controller_metrics_obs,
            epoch,
        )
        update_phase1_adaptive_state(
            phase1_state_action,
            config,
            metrics,
            controller_metrics_action,
            epoch,
        )
        current_scales_obs = phase1_effective_weight_scales(config, phase1_state_obs)
        current_scales_action = phase1_effective_weight_scales(config, phase1_state_action)
        for name, value in current_scales_obs.items():
            metrics[f"phase1_obs/{name}"] = value
        for name, value in current_scales_action.items():
            metrics[f"phase1_action/{name}"] = value
        metrics["phase1/entropy_scale"] = 0.5 * (
            current_scales_obs["entropy_scale"] + current_scales_action["entropy_scale"]
        )
        metrics["phase1/chart_usage_scale"] = 0.5 * (
            current_scales_obs["chart_usage_scale"] + current_scales_action["chart_usage_scale"]
        )
        metrics["phase1/chart_ot_scale"] = 0.5 * (
            current_scales_obs["chart_ot_scale"] + current_scales_action["chart_ot_scale"]
        )
        metrics["phase1/code_usage_scale"] = 0.5 * (
            current_scales_obs["code_usage_scale"] + current_scales_action["code_usage_scale"]
        )
        last_exact_control_gate = float(metrics.get("actor/exact_control_gate", 0.0))

        dt = time.perf_counter() - t0

        # --- Logging ---
        if epoch % config.log_every == 0:
            metrics["env/total_steps"] = total_env_steps
            metrics["env/buffer_episodes"] = buffer.num_episodes
            metrics["env/buffer_steps"] = buffer.total_steps
            metrics["train/wall_time"] = dt
            metrics["train/lr_encoder"] = _current_lr(optimizer_enc)
            metrics["train/lr_wm"] = _current_lr(optimizer_wm)
            metrics["train/lr_macro"] = _current_lr(optimizer_macro)
            metrics["train/lr_boundary"] = _current_lr(optimizer_boundary)
            metrics["train/lr_critic"] = (
                _current_lr(optimizer_critic)
                if optimizer_critic is not None
                else _current_lr(optimizer_wm)
            )
            if episode_rewards:
                metrics["env/last_episode_reward"] = episode_rewards[-1]
                metrics["env/mean_episode_reward_20"] = float(np.mean(episode_rewards[-20:]))

            # Console output
            hdr = f"E{epoch:04d} [{n_updates}upd]"
            line1_keys = [
                ("ep_rew", "env/last_episode_reward"),
                ("rew_20", "env/mean_episode_reward_20"),
                ("enc_o", "enc/L_total"),
                ("enc_a", "enc_action/L_total"),
                ("close", "closure/L_total"),
                ("L_geo", "wm/L_geodesic"),
                ("L_rew", "wm/L_reward"),
                ("L_crit", "critic/L_critic"),
                ("L_act", "actor/L_total"),
                ("lr_a", "train/lr_boundary"),
            ]
            line2_keys = [
                ("recon", "enc/recon"),
                ("vq", "enc/vq"),
                ("a_recon", "enc_action/recon"),
                ("a_vq", "enc_action/vq"),
                ("cl_obs", "closure/L_obs_state"),
                ("cl_z", "closure/L_obs_zeno"),
                ("cl_en", "closure/L_enclosure"),
                ("code_H", "enc/H_code_usage"),
                ("act_H", "action_chart/usage_entropy"),
                ("enc_gn", "enc/grad_norm"),
                ("act_gn", "actor/grad_norm"),
            ]
            line3_keys = [
                ("a_can", "policy/action_canonical_norm_mean"),
                ("a_lat", "policy/action_latent_norm_mean"),
                ("a_ent", "policy/action_router_entropy"),
                ("im_rew", "imagination/reward_mean"),
                ("im_ret", "imagination/return_mean"),
                ("act_ret", "actor/return_mean"),
                ("value", "critic/value_mean"),
                ("wm_gn", "wm/grad_norm"),
            ]
            line4_keys = [
                ("z_norm", "geometric/z_norm_mean"),
                ("az_norm", "geometric/action_z_norm_mean"),
                ("jump", "geometric/jump_frac"),
                ("cons", "geometric/hodge_conservative_exact"),
                ("sol", "geometric/hodge_solenoidal_exact"),
                ("e_var", "geometric/energy_var"),
                ("ch_ent", "chart/usage_entropy"),
                ("ach_ent", "action_chart/usage_entropy"),
                ("wm_ctr", "chart/wm_center_drift"),
                ("a_ctr", "action_chart/actor_center_drift"),
            ]
            line5_keys = [
                ("obj", "imagination/return_mean"),
                ("dret", "imagination/discounted_reward_mean"),
                ("term", "imagination/terminal_value_mean"),
                ("exbd", "imagination/exact_boundary_mean"),
                ("chart_acc", "wm/chart_acc"),
                ("code_acc", "wm/code_acc"),
                ("sym_acc", "wm/symbol_acc"),
                ("rw_sync", "imagination/policy_router_sync_mean"),
                ("f_err", "imagination/policy_force_rel_err_mean"),
                ("a_can", "policy/action_canonical_norm_mean"),
            ]
            line6_keys = [
                ("sig", "policy/sigma_motor"),
                ("col", "time/collection"),
                ("smp", "time/sample"),
                ("enc_t", "time/encoder"),
                ("wm_t", "time/world_model"),
                ("crt_t", "time/critic"),
                ("act_t", "time/actor"),
                ("diag_t", "time/diagnostics"),
            ]
            line7_keys = [
                ("rfrac", "wm/reward_nonconservative_frac_masked"),
                ("bell", "critic/replay_bellman_abs"),
                ("bell_s", "critic/replay_bellman_std"),
                ("rtg_e", "critic/replay_rtg_abs_err"),
                ("cal_e", "critic/replay_calibration_err"),
                ("trust", "actor/return_trust_used"),
                ("curg", "actor/curiosity_gate"),
                ("cur", "actor/curiosity_mean"),
                ("sync", "actor/L_sync"),
                ("a_sup", "actor/L_supervise"),
                ("a_bc", "actor/supervise_scale"),
                ("a_rl", "actor/return_gate"),
                ("a_opg", "actor/L_old_policy_geodesic"),
                ("ret_upd", "actor/return_applied"),
            ]
            line8_keys = [
                ("c_cov", "critic/exact_covector_norm_mean"),
                ("c_cal", "critic/calibration_ratio"),
                ("c_stf", "critic/stiffness_certified"),
                ("m_cal", "actor/macro_control_calibration_ratio"),
                ("mgate", "actor/macro_control_gate"),
                ("a_al", "actor/state_alpha"),
                ("a_be", "actor/state_beta_pi"),
                ("a_br", "actor/state_beta_pi_raw"),
                ("a_sc", "actor/state_scale_trust"),
                ("a_stf", "actor/stiffness_trust"),
                ("xgate", "actor/exact_control_gate"),
                ("a_gau", "actor/gauge_covector_norm_mean"),
            ]
            line9_keys = [
                ("ex_ps", "critic/exact_increment_pred_std"),
                ("ex_ts", "critic/exact_increment_target_std"),
                ("ex_sa", "critic/exact_increment_sign_acc"),
                ("ex_cr", "critic/exact_increment_corr"),
                ("ex_sf", "critic/exact_increment_support_frac"),
                ("m_re", "macro/reward_abs_err"),
                ("m_bs", "macro/bootstrap_term_mean"),
                ("m_ne", "macro/next_state_entropy"),
                ("m_t1", "macro/next_state_top1_prob"),
                ("m_sf", "macro/self_transition_prob"),
                ("a_sat", "actor/action_saturation_frac"),
                ("a_std", "actor/action_std"),
                ("i_gap", "actor/imagined_replay_return_gap"),
            ]
            line10_keys = [
                ("m_tx", "macro/transition_acc"),
                ("m_ce", "macro/transition_ce"),
                ("m_sh", "macro/transition_sharpen_gate"),
                ("m_te", "macro/L_transition_entropy"),
                ("m_ts", "macro/target_scale"),
                ("op_ts", "macro/on_policy/target_scale"),
            ]
            line11_keys = [
                ("s_po", "critic/stage_poisson"),
                ("s_cv", "critic/stage_covector"),
                ("s_st", "critic/stage_stiffness"),
                ("s_mp", "critic/stage_macro_pullback"),
                ("s_op", "critic/stage_on_policy"),
                ("g_ex", "critic/grad_exact_increment"),
                ("g_po", "critic/grad_poisson"),
                ("g_cv", "critic/grad_covector_align"),
                ("g_st", "critic/grad_stiffness"),
                ("g_mp", "critic/grad_macro_pullback"),
                ("g_op", "critic/grad_on_policy"),
            ]

            def _fmt(pairs):
                return "  ".join(
                    f"{label}={metrics[key]:.4f}" for label, key in pairs if key in metrics
                )

            print(f"{hdr}  {_fmt(line1_keys)}  dt={dt:.2f}s")
            print(f"  {'':4s}  {_fmt(line2_keys)}")
            print(f"  {'':4s}  {_fmt(line3_keys)}")
            print(f"  {'':4s}  {_fmt(line4_keys)}")
            print(f"  {'':4s}  {_fmt(line5_keys)}")
            print(f"  {'':4s}  {_fmt(line6_keys)}")
            print(f"  {'':4s}  {_fmt(line7_keys)}")
            print(f"  {'':4s}  {_fmt(line8_keys)}")
            print(f"  {'':4s}  {_fmt(line9_keys)}")
            print(f"  {'':4s}  {_fmt(line10_keys)}")
            print(f"  {'':4s}  {_fmt(line11_keys)}")

            # Per-chart usage line
            usage_parts = []
            for k in range(config.num_charts):
                key = f"chart/{k}/usage"
                if key in metrics:
                    usage_parts.append(f"c{k}={metrics[key]:.2f}")
            if usage_parts:
                active_charts = round(metrics.get("chart/active_charts", 0.0))
                print(
                    f"  {'':4s}  charts: {active_charts}/{config.num_charts} active  "
                    f"{' '.join(usage_parts)}",
                )

            action_usage_parts = []
            for k in range(config.num_action_charts):
                key = f"action_chart/{k}/usage"
                if key in metrics:
                    action_usage_parts.append(f"a{k}={metrics[key]:.2f}")
            if action_usage_parts:
                active_action_charts = round(metrics.get("action_chart/active_charts", 0.0))
                print(
                    f"  {'':4s}  action charts: {active_action_charts}/{config.num_action_charts} active  "
                    f"{' '.join(action_usage_parts)}",
                )

            symbol_parts = []
            for k in range(config.num_charts):
                active_key = f"chart/{k}/active_codes"
                entropy_key = f"chart/{k}/code_entropy"
                if active_key in metrics and entropy_key in metrics:
                    active_codes = round(metrics[active_key])
                    symbol_parts.append(
                        f"c{k}={active_codes}/{config.codes_per_chart}(H={metrics[entropy_key]:.2f})",
                    )
            if symbol_parts:
                active_symbols = round(metrics.get("chart/active_symbols", 0.0))
                total_symbols = config.num_charts * config.codes_per_chart
                print(
                    f"  {'':4s}  symbols: {active_symbols}/{total_symbols} active  "
                    f"{' '.join(symbol_parts)}",
                )

            action_symbol_parts = []
            for k in range(config.num_action_charts):
                active_key = f"action_chart/{k}/active_codes"
                entropy_key = f"action_chart/{k}/code_entropy"
                if active_key in metrics and entropy_key in metrics:
                    active_codes = round(metrics[active_key])
                    action_symbol_parts.append(
                        f"a{k}={active_codes}/{config.action_codes_per_chart}(H={metrics[entropy_key]:.2f})",
                    )
            if action_symbol_parts:
                active_action_symbols = round(metrics.get("action_chart/active_symbols", 0.0))
                total_action_symbols = config.num_action_charts * config.action_codes_per_chart
                print(
                    f"  {'':4s}  action symbols: {active_action_symbols}/{total_action_symbols} active  "
                    f"{' '.join(action_symbol_parts)}",
                )

            # MLflow
            if mlflow_enabled and log_mlflow_metrics is not None:
                prefixed = {f"phase4/{k}": v for k, v in metrics.items()}
                log_mlflow_metrics(prefixed, step=epoch, enabled=True)

        # --- Evaluation ---
        if epoch % config.eval_every == 0 and epoch > 0:
            model.encoder.eval()
            action_model.eval()
            actor.eval()
            critic.eval()
            macro_critic.eval()
            eval_metrics = _eval_policy(
                env,
                actor,
                action_model,
                model,
                device,
                obs_normalizer=obs_normalizer,
                action_repeat=config.action_repeat,
                num_episodes=config.eval_episodes,
                max_steps=config.max_episode_steps,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )
            print(
                f"  EVAL  reward={eval_metrics['eval/reward_mean']:.1f} "
                f"+/- {eval_metrics['eval/reward_std']:.1f}  "
                f"len={eval_metrics['eval/length_mean']:.0f}"
            )
            if mlflow_enabled and log_mlflow_metrics is not None:
                log_mlflow_metrics(
                    {f"phase4/{k}": v for k, v in eval_metrics.items()},
                    step=epoch,
                    enabled=True,
                )
            if eval_metrics["eval/reward_mean"] > best_eval_reward:
                best_eval_reward = eval_metrics["eval/reward_mean"]
                _save_checkpoint(
                    os.path.join(config.checkpoint_dir, "best.pt"),
                    model,
                    jump_op,
                    action_model,
                    action_jump_op,
                    world_model,
                    actor,
                    actor_old,
                    critic,
                    macro_critic,
                    reward_head,
                    enclosure_probe,
                    optimizer_enc,
                    optimizer_wm,
                    optimizer_boundary,
                    optimizer_critic,
                    optimizer_enclosure,
                    scheduler_enc,
                    scheduler_wm,
                    scheduler_boundary,
                    scheduler_critic,
                    scheduler_enclosure,
                    epoch,
                    config,
                    eval_metrics,
                    obs_normalizer=obs_normalizer,
                    optimizer_macro=optimizer_macro,
                    scheduler_macro=scheduler_macro,
                )
                print(f"  New best: {best_eval_reward:.1f}")

        # --- Checkpoint ---
        if epoch % config.checkpoint_every == 0 and epoch > 0:
            _save_checkpoint(
                os.path.join(config.checkpoint_dir, f"epoch_{epoch:05d}.pt"),
                model,
                jump_op,
                action_model,
                action_jump_op,
                world_model,
                actor,
                actor_old,
                critic,
                macro_critic,
                reward_head,
                enclosure_probe,
                optimizer_enc,
                optimizer_wm,
                optimizer_boundary,
                optimizer_critic,
                optimizer_enclosure,
                scheduler_enc,
                scheduler_wm,
                scheduler_boundary,
                scheduler_critic,
                scheduler_enclosure,
                epoch,
                config,
                metrics,
                obs_normalizer=obs_normalizer,
                optimizer_macro=optimizer_macro,
                scheduler_macro=scheduler_macro,
            )

        if not config.freeze_encoder:
            scheduler_enc.step()
        scheduler_wm.step()
        scheduler_macro.step()
        scheduler_boundary.step()
        scheduler_enclosure.step()
        if scheduler_critic is not None:
            scheduler_critic.step()

    print("=" * 80)
    print("Training complete.")
    if collect_env is not None:
        collect_env.close()
    if mlflow_enabled and end_mlflow_run is not None:
        end_mlflow_run(enabled=True)


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    model: TopoEncoder,
    jump_op: nn.Module,
    action_model: nn.Module,
    action_jump_op: nn.Module,
    world_model: nn.Module,
    actor: nn.Module,
    actor_old: nn.Module,
    critic: nn.Module,
    macro_critic: nn.Module,
    reward_head: nn.Module,
    enclosure_probe: nn.Module,
    optimizer_enc: torch.optim.Optimizer,
    optimizer_wm: torch.optim.Optimizer,
    optimizer_boundary: torch.optim.Optimizer,
    optimizer_critic: torch.optim.Optimizer | None,
    optimizer_enclosure: torch.optim.Optimizer,
    scheduler_enc: torch.optim.lr_scheduler.LRScheduler,
    scheduler_wm: torch.optim.lr_scheduler.LRScheduler,
    scheduler_boundary: torch.optim.lr_scheduler.LRScheduler,
    scheduler_critic: torch.optim.lr_scheduler.LRScheduler | None,
    scheduler_enclosure: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    config: DreamerConfig,
    metrics: dict | None = None,
    obs_normalizer: ObservationNormalizer | None = None,
    optimizer_macro: torch.optim.Optimizer | None = None,
    scheduler_macro: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "epoch": epoch,
        "config": asdict(config),
        "encoder": {k: v.cpu() for k, v in model.encoder.state_dict().items()},
        "decoder": {k: v.cpu() for k, v in model.decoder.state_dict().items()},
        "jump_op": {k: v.cpu() for k, v in jump_op.state_dict().items()},
        "action_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_model).state_dict().items()
        },
        "action_jump_op": {
            k: v.cpu() for k, v in _unwrap_compiled_module(action_jump_op).state_dict().items()
        },
        "world_model": {
            k: v.cpu() for k, v in _unwrap_compiled_module(world_model).state_dict().items()
        },
        "actor": {k: v.cpu() for k, v in _unwrap_compiled_module(actor).state_dict().items()},
        "actor_old": {
            k: v.cpu() for k, v in _unwrap_compiled_module(actor_old).state_dict().items()
        },
        "critic": {k: v.cpu() for k, v in _unwrap_compiled_module(critic).state_dict().items()},
        "macro_critic": {
            k: v.cpu() for k, v in _unwrap_compiled_module(macro_critic).state_dict().items()
        },
        "reward_head": {
            k: v.cpu() for k, v in _unwrap_compiled_module(reward_head).state_dict().items()
        },
        "enclosure_probe": {
            k: v.cpu() for k, v in _unwrap_compiled_module(enclosure_probe).state_dict().items()
        },
        "optimizer_enc": optimizer_enc.state_dict(),
        "optimizer_wm": optimizer_wm.state_dict(),
        "optimizer_boundary": optimizer_boundary.state_dict(),
        "optimizer_enclosure": optimizer_enclosure.state_dict(),
        "scheduler_enc": scheduler_enc.state_dict(),
        "scheduler_wm": scheduler_wm.state_dict(),
        "scheduler_boundary": scheduler_boundary.state_dict(),
        "scheduler_enclosure": scheduler_enclosure.state_dict(),
        "metrics": metrics or {},
        "obs_normalizer": None if obs_normalizer is None else obs_normalizer.state_dict(),
    }
    if optimizer_critic is not None:
        state["optimizer_critic"] = optimizer_critic.state_dict()
    if optimizer_macro is not None:
        state["optimizer_macro"] = optimizer_macro.state_dict()
    if scheduler_critic is not None:
        state["scheduler_critic"] = scheduler_critic.state_dict()
    if scheduler_macro is not None:
        state["scheduler_macro"] = scheduler_macro.state_dict()
    torch.save(state, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> DreamerConfig:
    import dataclasses

    parser = argparse.ArgumentParser(description="Geometric Dreamer training")
    for f in fields(DreamerConfig):
        # Resolve default (handle default_factory for fields like `device`)
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            continue  # skip fields with no default
        if f.type == "bool":
            parser.add_argument(f"--{f.name}", action="store_true", default=default)
            parser.add_argument(f"--no-{f.name}", dest=f.name, action="store_false")
        else:
            parser.add_argument(f"--{f.name}", type=type(default), default=default)
    args = parser.parse_args()
    return DreamerConfig(**{f.name: getattr(args, f.name) for f in fields(DreamerConfig)})


if __name__ == "__main__":
    cfg = _parse_args()
    train(cfg)
