"""dm_control environment helpers for Geometric Dreamer training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .config import DreamerConfig


# ---------------------------------------------------------------------------
# Environment creation & observation utilities
# ---------------------------------------------------------------------------


def _make_env(domain: str, task: str):
    """Create a dm_control environment."""
    from dm_control import suite

    return suite.load(domain_name=domain, task_name=task)


def _flatten_obs(time_step) -> np.ndarray:
    """Flatten dm_control observation OrderedDict to a single vector."""
    parts = []
    for v in time_step.observation.values():
        v = np.asarray(v, dtype=np.float32).flatten()
        parts.append(v)
    return np.concatenate(parts)


def _infer_action_dim(env) -> int:
    """Infer the flattened continuous action dimension from an env or wrapper."""
    action_spec = None
    if hasattr(env, "action_spec"):
        action_spec = env.action_spec()
    elif hasattr(env, "action_space"):
        action_spec = env.action_space
    if action_spec is None:
        msg = "Environment does not expose action_spec() or action_space."
        raise AttributeError(msg)
    shape = tuple(getattr(action_spec, "shape", ()))
    if not shape:
        msg = "Environment action spec does not expose a valid shape."
        raise ValueError(msg)
    action_dim = int(np.prod(shape))
    if action_dim <= 0:
        msg = f"Environment action spec has invalid flattened dimension {action_dim}."
        raise ValueError(msg)
    return action_dim


# ---------------------------------------------------------------------------
# Task presets
# ---------------------------------------------------------------------------


def _apply_task_preset(
    config: DreamerConfig,
) -> tuple[str | None, dict[str, tuple[object, object]]]:
    """Apply environment-specific defaults while respecting explicit user overrides."""
    preset_name = str(getattr(config, "task_preset", "auto") or "auto").strip().lower()
    if preset_name in {"", "none", "off", "false"}:
        return None, {}
    if preset_name == "auto":
        if (config.domain, config.task) == ("cartpole", "swingup"):
            preset_name = "cartpole_swingup"
        elif (config.domain, config.task) == ("cartpole", "balance"):
            preset_name = "cartpole_balance"
        else:
            return None, {}
    if preset_name not in {"cartpole_swingup", "cartpole_balance"}:
        msg = f"Unknown task preset: {config.task_preset}"
        raise ValueError(msg)

    defaults = DreamerConfig()
    changes: dict[str, tuple[object, object]] = {}
    old_num_charts = config.num_charts
    old_num_action_charts = config.num_action_charts
    old_num_action_macros = config.num_action_macros
    old_codes_per_chart = config.codes_per_chart
    old_action_codes_per_chart = config.action_codes_per_chart

    def _maybe_override(name: str, value: object) -> None:
        current = getattr(config, name)
        default = getattr(defaults, name)
        if current == default and current != value:
            setattr(config, name, value)
            changes[name] = (current, value)

    def _maybe_override_from(
        name: str, value: object, allowed_currents: tuple[object, ...]
    ) -> None:
        current = getattr(config, name)
        if current in allowed_currents and current != value:
            setattr(config, name, value)
            changes[name] = (current, value)

    _maybe_override("latent_dim", 8)
    _maybe_override("num_charts", 4)
    _maybe_override("codes_per_chart", 8)
    _maybe_override("d_model", 64)
    _maybe_override("hidden_dim", 128)
    _maybe_override("max_episode_steps", 200)
    _maybe_override("batch_size", 8)
    _maybe_override("seq_len", 32)
    _maybe_override("imagination_horizon", 8)
    _maybe_override("actor_return_horizon", 8)
    _maybe_override("hard_routing", True)
    _maybe_override("hard_routing_warmup_epochs", 0)
    _maybe_override("hard_routing_tau", 1.0)
    _maybe_override("hard_routing_tau_end", 1.0)
    _maybe_override("hard_routing_tau_anneal_epochs", 0)
    _maybe_override("w_entropy", 0.05)
    _maybe_override("w_diversity", 2.0)
    _maybe_override("chart_multiplier_lr", 1.5)
    _maybe_override("phase1_multiplier_max", 12.0)
    _maybe_override("w_reward_nonconservative_norm", 0.1)
    _maybe_override("w_reward_nonconservative_budget", 0.25)
    _maybe_override("reward_nonconservative_budget_ratio", 0.05)
    _maybe_override("reward_nonconservative_budget_floor", 0.001)
    _maybe_override("w_wm_code", 0.25)
    _maybe_override("w_wm_symbol", 0.5)
    _maybe_override("w_reward_exact_orth", 0.1)
    _maybe_override("w_reward_conservative_match", 10.0)
    _maybe_override("w_screened_poisson", 2.0)
    _maybe_override("screened_poisson_warmup_epochs", 10)
    _maybe_override("w_critic", 1.0)
    _maybe_override("w_critic_exact_increment", 1.0)
    _maybe_override("w_critic_stiffness", 5.0)
    _maybe_override("w_critic_covector_align", 5.0)
    _maybe_override("critic_covector_warmup_epochs", 5)
    _maybe_override("critic_stiffness_warmup_epochs", 10)
    _maybe_override("critic_macro_pullback_warmup_epochs", 8)
    _maybe_override("critic_on_policy_warmup_epochs", 8)
    _maybe_override("critic_grad_metrics_every", 1)
    _maybe_override("w_macro_value", 0.25)
    _maybe_override("w_macro_exact_increment", 0.5)
    _maybe_override("w_macro_pullback", 0.25)
    _maybe_override("w_macro_covector_pullback", 0.1)
    _maybe_override("w_macro_on_policy_pullback", 0.1)
    _maybe_override("w_macro_on_policy_covector_pullback", 0.05)
    _maybe_override("w_macro_transition", 0.25)
    _maybe_override("w_macro_transition_entropy", 0.01)
    _maybe_override("macro_multistep_horizon", 4)
    _maybe_override("macro_multistep_decay", 0.8)
    _maybe_override("macro_on_policy_horizon", 4)
    _maybe_override("macro_on_policy_batch_size", 4)
    _maybe_override("macro_target_scale_quantile", 0.75)
    _maybe_override("macro_target_scale_min", 1e-3)
    _maybe_override("macro_transition_closure_acc_target", 0.5)
    _maybe_override("macro_transition_enclosure_defect_acc_scale", 4.0)
    _maybe_override("macro_transition_enclosure_defect_ce_scale", 1.0)
    _maybe_override("critic_stiffness_min", 0.001)
    _maybe_override("critic_stiffness_target_max", 0.05)
    _maybe_override("actor_return_chart_acc_target", 0.5)
    _maybe_override("actor_return_update_every", 2)
    _maybe_override("actor_return_warmup_epochs", 2)
    _maybe_override("actor_metric_fisher_scale", 0.01)
    _maybe_override("actor_stiffness_min", 0.001)
    _maybe_override("actor_supervise_warmup_epochs", 2)
    _maybe_override("actor_supervise_decay_epochs", 20)
    _maybe_override("actor_supervise_min_scale", 0.05)
    _maybe_override("w_actor_old_policy_chart_kl", 0.01)
    _maybe_override("w_actor_old_policy_code_kl", 0.01)
    _maybe_override("collect_every", 1)
    _maybe_override("collect_n_env_workers", 4)
    _maybe_override("eval_every", 10)
    _maybe_override("checkpoint_every", 25)
    _maybe_override("actor_return_exact_increment_rel_scale", 1.0)
    _maybe_override("actor_return_exact_covector_rel_scale", 1.0)
    _maybe_override("actor_return_exact_control_power", 1.0)
    _maybe_override("actor_macro_backbone_weight", 0.25)
    _maybe_override("actor_macro_backbone_power", 1.0)
    _maybe_override("critic_on_policy_decay", 1.0)
    _maybe_override("actor_curiosity_closure_acc_target", 0.5)
    _maybe_override("actor_curiosity_enclosure_defect_acc_scale", 4.0)
    _maybe_override("actor_curiosity_enclosure_defect_ce_scale", 1.0)
    _maybe_override("macro_lr_multiplier", 3.0)

    if preset_name == "cartpole_balance":
        _maybe_override("seed_episodes", 8)
        _maybe_override("critic_multistep_horizon", 4)
        _maybe_override("critic_multistep_decay", 0.75)
        _maybe_override("w_critic_on_policy_covector_align", 2.0)
        _maybe_override("w_critic_on_policy_stiffness", 1.0)
        _maybe_override("critic_on_policy_horizon", 4)
        _maybe_override("critic_on_policy_batch_size", 4)
        _maybe_override("w_macro_covector_pullback", 0.1)
        _maybe_override("w_macro_on_policy_covector_pullback", 0.05)
        _maybe_override("actor_macro_backbone_weight", 0.25)
        _maybe_override("w_macro_on_policy_pullback", 0.1)
        _maybe_override("w_macro_transition", 0.25)
        _maybe_override("w_macro_transition_entropy", 0.01)
        _maybe_override("w_wm_code", 0.5)
        _maybe_override("w_wm_symbol", 1.0)
        _maybe_override("critic_covector_warmup_epochs", 3)
        _maybe_override("critic_stiffness_warmup_epochs", 6)
        _maybe_override("critic_macro_pullback_warmup_epochs", 5)
        _maybe_override("critic_on_policy_warmup_epochs", 5)
        _maybe_override("sigma_motor", 0.1)
        _maybe_override("sigma_motor_init", 0.15)
        _maybe_override("sigma_motor_anneal_epochs", 20)
        _maybe_override("sigma_motor_exact_gate_target", 0.35)
        _maybe_override("w_actor_curiosity", 0.05)
    else:
        _maybe_override_from("seed_episodes", 24, (defaults.seed_episodes, 8))
        _maybe_override_from("codes_per_chart", 16, (defaults.codes_per_chart, 8))
        _maybe_override_from(
            "action_codes_per_chart",
            16,
            (defaults.action_codes_per_chart, 8),
        )
        _maybe_override_from("seq_len", 64, (defaults.seq_len, 32))
        _maybe_override_from("max_episode_steps", 500, (defaults.max_episode_steps, 200))
        _maybe_override_from("imagination_horizon", 12, (defaults.imagination_horizon, 8))
        _maybe_override_from("actor_return_horizon", 12, (defaults.actor_return_horizon, 8))
        _maybe_override_from(
            "critic_multistep_horizon", 16, (defaults.critic_multistep_horizon, 4)
        )
        _maybe_override_from(
            "critic_multistep_decay", 0.8, (defaults.critic_multistep_decay, 0.75)
        )
        _maybe_override_from(
            "w_critic_on_policy_covector_align",
            5.0,
            (defaults.w_critic_on_policy_covector_align, 2.0),
        )
        _maybe_override_from(
            "w_critic_on_policy_stiffness",
            2.0,
            (defaults.w_critic_on_policy_stiffness, 1.0),
        )
        _maybe_override_from(
            "critic_on_policy_horizon", 12, (defaults.critic_on_policy_horizon, 4)
        )
        _maybe_override_from(
            "critic_on_policy_batch_size", 8, (defaults.critic_on_policy_batch_size, 4)
        )
        _maybe_override_from("critic_on_policy_decay", 0.9, (defaults.critic_on_policy_decay, 1.0))
        _maybe_override_from("w_macro_value", 0.5, (defaults.w_macro_value, 0.25))
        _maybe_override_from(
            "w_macro_exact_increment", 1.0, (defaults.w_macro_exact_increment, 0.5)
        )
        _maybe_override_from("w_macro_pullback", 0.5, (defaults.w_macro_pullback, 0.25))
        _maybe_override_from(
            "w_macro_covector_pullback",
            0.5,
            (defaults.w_macro_covector_pullback, 0.1),
        )
        _maybe_override_from(
            "w_macro_on_policy_pullback", 0.25, (defaults.w_macro_on_policy_pullback, 0.1)
        )
        _maybe_override_from(
            "w_macro_on_policy_covector_pullback",
            0.25,
            (defaults.w_macro_on_policy_covector_pullback, 0.05),
        )
        _maybe_override_from("w_macro_transition", 0.75, (defaults.w_macro_transition, 0.25))
        _maybe_override_from(
            "w_macro_transition_entropy",
            0.05,
            (defaults.w_macro_transition_entropy, 0.01),
        )
        _maybe_override_from("w_wm_code", 0.75, (defaults.w_wm_code, 0.25, 0.5))
        _maybe_override_from("w_wm_symbol", 1.5, (defaults.w_wm_symbol, 0.5, 1.0))
        _maybe_override_from(
            "screened_poisson_warmup_epochs",
            20,
            (defaults.screened_poisson_warmup_epochs, 10),
        )
        _maybe_override_from(
            "critic_covector_warmup_epochs", 10, (defaults.critic_covector_warmup_epochs, 5, 3)
        )
        _maybe_override_from(
            "critic_stiffness_warmup_epochs", 20, (defaults.critic_stiffness_warmup_epochs, 10, 6)
        )
        _maybe_override_from(
            "critic_macro_pullback_warmup_epochs",
            15,
            (defaults.critic_macro_pullback_warmup_epochs, 8, 5),
        )
        _maybe_override_from(
            "critic_on_policy_warmup_epochs",
            15,
            (defaults.critic_on_policy_warmup_epochs, 8, 5),
        )
        _maybe_override_from("macro_multistep_horizon", 16, (defaults.macro_multistep_horizon, 4))
        _maybe_override_from("macro_multistep_decay", 0.85, (defaults.macro_multistep_decay, 0.8))
        _maybe_override_from("macro_on_policy_horizon", 12, (defaults.macro_on_policy_horizon, 4))
        _maybe_override_from(
            "macro_on_policy_batch_size", 8, (defaults.macro_on_policy_batch_size, 4)
        )
        _maybe_override_from(
            "actor_macro_backbone_weight",
            1.0,
            (defaults.actor_macro_backbone_weight, 0.25),
        )
        _maybe_override("sigma_motor", 0.2)
        _maybe_override_from("sigma_motor_init", 0.5, (defaults.sigma_motor_init, 0.15))
        _maybe_override_from(
            "sigma_motor_anneal_epochs", 60, (defaults.sigma_motor_anneal_epochs, 20)
        )
        _maybe_override_from(
            "sigma_motor_exact_gate_target", 0.45, (defaults.sigma_motor_exact_gate_target, 0.35)
        )
        _maybe_override("w_actor_curiosity", 0.2)
        _maybe_override_from("macro_lr_multiplier", 10.0, (defaults.macro_lr_multiplier, 3.0))

    chart_entropy_max = float(np.log(max(config.num_charts, 1)))
    _maybe_override("chart_usage_h_low", 0.6 * chart_entropy_max)
    _maybe_override("chart_usage_h_high", 0.95 * chart_entropy_max)

    if old_num_action_charts in {defaults.num_action_charts, old_num_charts}:
        if config.num_action_charts != config.num_charts:
            changes["num_action_charts"] = (config.num_action_charts, config.num_charts)
            config.num_action_charts = config.num_charts
    if old_num_action_macros in {
        defaults.num_action_macros,
        old_num_action_charts,
        old_num_charts,
    }:
        if config.num_action_macros != config.num_action_charts:
            changes["num_action_macros"] = (config.num_action_macros, config.num_action_charts)
            config.num_action_macros = config.num_action_charts
    if old_action_codes_per_chart in {defaults.action_codes_per_chart, old_codes_per_chart}:
        if config.action_codes_per_chart != config.codes_per_chart:
            changes["action_codes_per_chart"] = (
                config.action_codes_per_chart,
                config.codes_per_chart,
            )
            config.action_codes_per_chart = config.codes_per_chart

    return preset_name, changes


# ---------------------------------------------------------------------------
# Action sampling
# ---------------------------------------------------------------------------


def _sample_collection_action(
    action_mean: np.ndarray,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    sigma_motor: float,
) -> np.ndarray:
    """Sample thermal motor exploration around the deterministic action mean."""
    action = np.clip(action_mean, action_min, action_max).astype(np.float32, copy=False)
    if sigma_motor <= 0.0:
        return action
    noise = np.random.normal(loc=0.0, scale=float(sigma_motor), size=action.shape).astype(
        np.float32
    )
    return np.clip(action + noise, action_min, action_max).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Observation normalization
# ---------------------------------------------------------------------------


@dataclass
class ObservationNormalizer:
    """Fixed per-dimension affine observation normalization."""

    mean: torch.Tensor
    std: torch.Tensor
    min_std: float = 1e-3

    @classmethod
    def from_episodes(
        cls,
        episodes: list[dict[str, np.ndarray]],
        device: torch.device,
        *,
        min_std: float = 1e-3,
    ) -> ObservationNormalizer:
        if not episodes:
            msg = "Need at least one episode to estimate observation normalization stats."
            raise ValueError(msg)
        obs = np.concatenate([episode["obs"] for episode in episodes], axis=0).astype(np.float32)
        mean = torch.from_numpy(obs.mean(axis=0)).to(device=device)
        std = torch.from_numpy(obs.std(axis=0)).to(device=device).clamp(min=min_std)
        return cls(mean=mean, std=std, min_std=min_std)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict[str, torch.Tensor | float],
        device: torch.device,
    ) -> ObservationNormalizer:
        min_std = float(state_dict.get("min_std", 1e-3))
        mean = torch.as_tensor(state_dict["mean"], device=device, dtype=torch.float32)
        std = torch.as_tensor(state_dict["std"], device=device, dtype=torch.float32).clamp(
            min=min_std,
        )
        return cls(mean=mean, std=std, min_std=min_std)

    def state_dict(self) -> dict[str, torch.Tensor | float]:
        return {
            "mean": self.mean.detach().cpu(),
            "std": self.std.detach().cpu(),
            "min_std": float(self.min_std),
        }

    def normalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = self.std.to(device=obs.device, dtype=obs.dtype)
        return (obs - mean) / std

    def denormalize_tensor(self, obs: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=obs.device, dtype=obs.dtype)
        std = self.std.to(device=obs.device, dtype=obs.dtype)
        return obs * std + mean

    def normalize_numpy(self, obs: np.ndarray) -> np.ndarray:
        mean = self.mean.detach().cpu().numpy()
        std = self.std.detach().cpu().numpy()
        return ((obs - mean) / std).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Episode dict packing
# ---------------------------------------------------------------------------


def _build_episode_dict(
    obs_list: list[np.ndarray],
    act_list: list[np.ndarray],
    rew_list: list[np.float32],
    done_list: list[np.float32],
    action_mean_list: list[np.ndarray],
    action_latent_list: list[np.ndarray],
    action_router_weight_list: list[np.ndarray],
    action_chart_idx_list: list[np.int64],
    action_code_idx_list: list[np.int64],
    action_code_latent_list: list[np.ndarray],
) -> dict[str, np.ndarray]:
    """Pack per-step episode traces into the replay-buffer episode format."""
    return {
        "obs": np.stack(obs_list),
        "actions": np.stack([*act_list, np.zeros_like(act_list[0])]),
        "action_means": np.stack([*action_mean_list, np.zeros_like(action_mean_list[0])]),
        "action_latents": np.stack(
            [*action_latent_list, np.zeros_like(action_latent_list[0])],
        ),
        "action_router_weights": np.stack(
            [*action_router_weight_list, np.zeros_like(action_router_weight_list[0])],
        ),
        "action_charts": np.array([*action_chart_idx_list, 0], dtype=np.int64),
        "action_codes": np.array([*action_code_idx_list, 0], dtype=np.int64),
        "action_code_latents": np.stack(
            [*action_code_latent_list, np.zeros_like(action_code_latent_list[0])],
        ),
        "rewards": np.array([*rew_list, 0.0], dtype=np.float32),
        "dones": np.array([*done_list, 1.0], dtype=np.float32),
    }
