"""Environment interaction helpers for standalone macro RL."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from .env_helpers import _flatten_obs, ObservationNormalizer
from .macro_control import (
    epsilon_greedy_macro_actions,
    MacroQNetwork,
)
from .macro_data import (
    action_symbol_to_continuous,
    ActionPrototypeTable,
    build_macro_episode_dict,
    symbolize_observations,
)


if TYPE_CHECKING:
    from fragile.agent import FragileAgent


def _action_usage_metrics(
    action_indices: list[int],
    num_actions: int,
) -> dict[str, float]:
    """Summarize how many discrete macro actions were used in one rollout set."""
    if not action_indices:
        return {
            "action_usage_active": 0.0,
            "action_usage_perplexity": 0.0,
        }
    counts = torch.bincount(
        torch.tensor(action_indices, dtype=torch.long), minlength=num_actions
    ).float()
    probs = counts / counts.sum().clamp(min=1.0)
    active = float((counts > 0).sum().item())
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    return {
        "action_usage_active": active,
        "action_usage_perplexity": float(math.exp(entropy.item())),
    }


def _new_rollout_state(env) -> dict[str, Any]:
    """Allocate per-environment rollout buffers for synchronous collection."""
    return {
        "time_step": env.reset(),
        "obs_list": [],
        "act_list": [],
        "rew_list": [],
        "done_list": [],
        "action_indices": [],
        "obs_state_indices": [],
        "random_action_count": 0,
        "step_count": 0,
    }


def _finalize_rollout(
    rollout: dict[str, Any],
    q_network: MacroQNetwork | None,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Convert one in-progress rollout buffer into replay and logging payloads."""
    time_step = rollout["time_step"]
    rollout["obs_list"].append(_flatten_obs(time_step).astype(np.float32, copy=False))
    episode = build_macro_episode_dict(
        rollout["obs_list"],
        rollout["act_list"],
        rollout["rew_list"],
        rollout["done_list"],
    )
    metrics = {
        "return": float(sum(float(reward) for reward in rollout["rew_list"])),
        "length": float(len(rollout["rew_list"])),
        "random_action_frac": (
            float(rollout["random_action_count"]) / float(max(len(rollout["rew_list"]), 1))
        ),
        "action_indices": list(rollout["action_indices"]),
        "obs_state_indices": list(rollout["obs_state_indices"]),
    }
    metrics.update(
        _action_usage_metrics(
            rollout["action_indices"],
            0 if q_network is None else q_network.num_actions,
        )
    )
    return episode, metrics


def collect_macro_episodes_batched(
    envs: list[Any],
    agent: FragileAgent | None,
    q_network: MacroQNetwork | None,
    action_prototypes: ActionPrototypeTable | None,
    *,
    num_episodes: int,
    device: torch.device,
    obs_normalizer: ObservationNormalizer | None = None,
    epsilon: float = 1.0,
    action_repeat: int = 1,
    max_steps: int = 1000,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
    sigma_motor: float = 0.0,
    use_inference_mode: bool = True,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, float]]]:
    """Collect several episodes synchronously while batching model inference."""
    if num_episodes <= 0:
        return [], []
    if not envs:
        msg = "envs must contain at least one environment."
        raise ValueError(msg)

    action_spec = envs[0].action_spec()
    action_min = np.asarray(action_spec.minimum, dtype=np.float32)
    action_max = np.asarray(action_spec.maximum, dtype=np.float32)

    started = 0
    finished = 0
    active_rollouts: list[dict[str, Any] | None] = [None] * len(envs)
    for slot in range(min(len(envs), int(num_episodes))):
        active_rollouts[slot] = _new_rollout_state(envs[slot])
        started += 1

    episodes: list[dict[str, np.ndarray]] = []
    infos: list[dict[str, float]] = []

    while finished < int(num_episodes):
        active_slots = [idx for idx, rollout in enumerate(active_rollouts) if rollout is not None]
        if not active_slots:
            break

        if agent is None or q_network is None:
            action_idx_batch: list[int] = [-1] * len(active_slots)
            random_batch: list[bool] = [True] * len(active_slots)
            state_idx_batch: list[int | None] = [None] * len(active_slots)
        else:
            obs_batch = np.stack(
                [
                    _flatten_obs(active_rollouts[slot]["time_step"]).astype(np.float32, copy=False)
                    for slot in active_slots
                ],
                axis=0,
            )
            obs_tensor = torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32)
            context = torch.inference_mode() if use_inference_mode else torch.no_grad()
            with context:
                obs_symbol = symbolize_observations(
                    agent,
                    obs_tensor,
                    obs_normalizer=obs_normalizer,
                    routing_tau=routing_tau,
                    macro_chart_tau=macro_chart_tau,
                    macro_code_tau=macro_code_tau,
                )
                action_idx_tensor, _q_values, random_tensor = epsilon_greedy_macro_actions(
                    q_network,
                    obs_symbol["state_probs_valid"],
                    epsilon=epsilon,
                )
            action_idx_batch = action_idx_tensor.detach().cpu().tolist()
            random_batch = random_tensor.detach().cpu().tolist()
            state_idx_batch = obs_symbol["state_idx_valid"].detach().cpu().tolist()

        for local_idx, slot in enumerate(active_slots):
            rollout = active_rollouts[slot]
            if rollout is None:
                continue

            obs = _flatten_obs(rollout["time_step"]).astype(np.float32, copy=False)
            rollout["obs_list"].append(obs.copy())

            if agent is None or q_network is None:
                action = np.random.uniform(action_min, action_max, size=action_spec.shape).astype(
                    np.float32,
                )
                rollout["random_action_count"] += 1
            else:
                state_idx = state_idx_batch[local_idx]
                if state_idx is not None:
                    rollout["obs_state_indices"].append(int(state_idx))
                action_idx = int(action_idx_batch[local_idx])
                rollout["action_indices"].append(action_idx)
                rollout["random_action_count"] += int(bool(random_batch[local_idx]))
                action = action_symbol_to_continuous(
                    action_idx,
                    action_prototypes,
                    action_min=action_min,
                    action_max=action_max,
                    sigma_motor=sigma_motor,
                )

            total_reward = 0.0
            time_step = rollout["time_step"]
            for _ in range(max(int(action_repeat), 1)):
                time_step = envs[slot].step(action)
                total_reward += float(time_step.reward or 0.0)
                if time_step.last():
                    break

            rollout["time_step"] = time_step
            rollout["act_list"].append(action.astype(np.float32, copy=False))
            rollout["rew_list"].append(np.float32(total_reward))
            rollout["done_list"].append(np.float32(time_step.last()))
            rollout["step_count"] += 1

            if time_step.last() or rollout["step_count"] >= max_steps:
                episode, info = _finalize_rollout(rollout, q_network)
                episodes.append(episode)
                infos.append(info)
                finished += 1
                if started < int(num_episodes):
                    active_rollouts[slot] = _new_rollout_state(envs[slot])
                    started += 1
                else:
                    active_rollouts[slot] = None

    return episodes, infos


def collect_macro_episode(
    env,
    agent: FragileAgent | None,
    q_network: MacroQNetwork | None,
    action_prototypes: ActionPrototypeTable | None,
    *,
    device: torch.device,
    obs_normalizer: ObservationNormalizer | None = None,
    epsilon: float = 1.0,
    action_repeat: int = 1,
    max_steps: int = 1000,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
    sigma_motor: float = 0.0,
    use_inference_mode: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Collect one episode with epsilon-greedy macro control or random actions."""
    episodes, infos = collect_macro_episodes_batched(
        [env],
        agent,
        q_network,
        action_prototypes,
        num_episodes=1,
        device=device,
        obs_normalizer=obs_normalizer,
        epsilon=epsilon,
        action_repeat=action_repeat,
        max_steps=max_steps,
        routing_tau=routing_tau,
        macro_chart_tau=macro_chart_tau,
        macro_code_tau=macro_code_tau,
        sigma_motor=sigma_motor,
        use_inference_mode=use_inference_mode,
    )
    return episodes[0], infos[0]


def evaluate_macro_policy(
    env,
    agent: FragileAgent,
    q_network: MacroQNetwork,
    action_prototypes: ActionPrototypeTable | None,
    *,
    device: torch.device,
    obs_normalizer: ObservationNormalizer | None,
    num_episodes: int,
    action_repeat: int,
    max_steps: int,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
    use_inference_mode: bool = True,
) -> dict[str, float]:
    """Run deterministic evaluation episodes for the current macro policy."""
    episodes, infos = collect_macro_episodes_batched(
        [env],
        agent,
        q_network,
        action_prototypes,
        num_episodes=max(int(num_episodes), 1),
        device=device,
        obs_normalizer=obs_normalizer,
        epsilon=0.0,
        action_repeat=action_repeat,
        max_steps=max_steps,
        routing_tau=routing_tau,
        macro_chart_tau=macro_chart_tau,
        macro_code_tau=macro_code_tau,
        sigma_motor=0.0,
        use_inference_mode=use_inference_mode,
    )
    del episodes

    returns = [info["return"] for info in infos]
    lengths = [info["length"] for info in infos]
    action_indices: list[int] = []
    for info in infos:
        action_indices.extend(int(idx) for idx in info.get("action_indices", []))

    summary = {
        "eval/return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval/return_std": float(np.std(returns)) if returns else 0.0,
        "eval/length_mean": float(np.mean(lengths)) if lengths else 0.0,
    }
    summary.update({
        f"eval/{key}": value
        for key, value in _action_usage_metrics(action_indices, q_network.num_actions).items()
    })
    return summary


__all__ = [
    "collect_macro_episode",
    "collect_macro_episodes_batched",
    "evaluate_macro_policy",
]
