"""Replay and symbol-wrangling helpers for standalone macro RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from fragile.losses.markov_model import soft_macro_state_distribution

from .env_helpers import _sample_collection_action, ObservationNormalizer
from .replay_buffer import SequenceReplayBuffer


if TYPE_CHECKING:
    from fragile.agent import FragileAgent


def _scatter_valid(
    values: torch.Tensor,
    valid_idx: torch.Tensor,
    leading_shape: tuple[int, ...],
    *,
    fill_value: float | int = 0,
) -> torch.Tensor:
    """Scatter valid rows back onto a `[B, T, ...]` grid."""
    flat_size = 1
    for dim in leading_shape:
        flat_size *= int(dim)
    full = values.new_full((flat_size, *values.shape[1:]), fill_value)
    full[valid_idx] = values
    return full.reshape(*leading_shape, *values.shape[1:])


def _flatten_selected(values: torch.Tensor, valid_idx: torch.Tensor) -> torch.Tensor:
    """Flatten batch/time axes and keep only rows selected by `valid_idx`."""
    return values.reshape(-1, *values.shape[2:])[valid_idx]


def _ensure_sequence_input(x: torch.Tensor) -> torch.Tensor:
    """Convert `[D]` or `[B, D]` inputs into the `[B, T, D]` agent format."""
    if x.dim() == 1:
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:
        return x.unsqueeze(1)
    if x.dim() == 3:
        return x
    msg = "Inputs must have shape [D], [B, D], or [B, T, D]."
    raise ValueError(msg)


def _symbolize_encoded(
    encoded: dict[str, torch.Tensor],
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chart_tau: float,
    code_tau: float,
) -> dict[str, torch.Tensor]:
    """Attach a soft chart/code state distribution to an encoded latent batch."""
    symbol_valid = soft_macro_state_distribution(
        encoded["z_geo_valid"],
        chart_centers,
        codebook,
        chart_tau=chart_tau,
        code_tau=code_tau,
    )
    codes_per_chart = int(codebook.shape[1])
    state_idx_valid = symbol_valid["state_idx"]
    chart_idx_valid = torch.div(state_idx_valid, codes_per_chart, rounding_mode="floor")
    code_idx_valid = state_idx_valid.remainder(codes_per_chart)

    leading_shape = tuple(int(dim) for dim in encoded["mask"].shape)
    valid_idx = encoded["valid_idx"]
    return {
        "state_probs_valid": symbol_valid["state_probs"],
        "state_idx_valid": state_idx_valid,
        "chart_idx_valid": chart_idx_valid,
        "code_idx_valid": code_idx_valid,
        "state_probs": _scatter_valid(symbol_valid["state_probs"], valid_idx, leading_shape),
        "state_idx": _scatter_valid(state_idx_valid, valid_idx, leading_shape),
        "chart_idx": _scatter_valid(chart_idx_valid, valid_idx, leading_shape),
        "code_idx": _scatter_valid(code_idx_valid, valid_idx, leading_shape),
        "chart_centers": symbol_valid["chart_centers"],
        "codebook": symbol_valid["codebook"],
        "state_points": symbol_valid["state_points"],
        "state_tangent_points": symbol_valid["state_tangent_points"],
    }


@dataclass
class ActionPrototypeTable:
    """Continuous action prototypes attached to discrete action symbols."""

    means: torch.Tensor
    counts: torch.Tensor
    valid: torch.Tensor

    def to(self, device: torch.device | str) -> ActionPrototypeTable:
        """Move the prototype table to another device."""
        return ActionPrototypeTable(
            means=self.means.to(device),
            counts=self.counts.to(device),
            valid=self.valid.to(device),
        )

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Serialize the prototype table for checkpoints."""
        return {
            "means": self.means.detach().cpu(),
            "counts": self.counts.detach().cpu(),
            "valid": self.valid.detach().cpu(),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, torch.Tensor]) -> ActionPrototypeTable:
        """Restore a prototype table from checkpoint tensors."""
        return cls(
            means=torch.as_tensor(state["means"], dtype=torch.float32),
            counts=torch.as_tensor(state["counts"], dtype=torch.float32),
            valid=torch.as_tensor(state["valid"], dtype=torch.bool),
        )


def build_macro_episode_dict(
    obs_list: list[np.ndarray],
    act_list: list[np.ndarray],
    rew_list: list[np.float32],
    done_list: list[np.float32],
) -> dict[str, np.ndarray]:
    """Pack one raw environment rollout into the replay-buffer episode format.

    Observations have length `T + 1`, while actions/rewards/dones have length
    `T`. The replay buffer expects aligned leading axes, so the final action is
    padded by repeating the last real action instead of inserting an unrelated
    zero vector that would distort action reconstruction losses.
    """
    if not obs_list:
        msg = "obs_list must contain at least one observation."
        raise ValueError(msg)
    if len(obs_list) != len(act_list) + 1:
        msg = "obs_list must have exactly one more entry than act_list."
        raise ValueError(msg)
    if len(act_list) != len(rew_list) or len(act_list) != len(done_list):
        msg = "act_list, rew_list, and done_list must have matching lengths."
        raise ValueError(msg)
    if not act_list:
        msg = "Need at least one action to build a replay episode."
        raise ValueError(msg)

    final_action = np.asarray(act_list[-1], dtype=np.float32)
    return {
        "obs": np.stack(obs_list).astype(np.float32, copy=False),
        "actions": np.stack([*act_list, final_action]).astype(np.float32, copy=False),
        "rewards": np.asarray([*rew_list, 0.0], dtype=np.float32),
        "dones": np.asarray([*done_list, 1.0], dtype=np.float32),
    }


def transition_valid_mask(dones: torch.Tensor) -> torch.Tensor:
    """Return the valid one-step transition mask for replay samples."""
    if dones.dim() < 2 or dones.shape[-1] < 2:
        msg = "dones must have shape [B, T] with T >= 2."
        raise ValueError(msg)
    # Replay windows are sampled from fully valid contiguous sub-sequences, so
    # every adjacent `(t, t+1)` pair is a legitimate transition.
    return torch.ones(
        *dones.shape[:-1], dones.shape[-1] - 1, device=dones.device, dtype=torch.bool
    )


def action_stats_from_episodes(
    episodes: list[dict[str, np.ndarray]],
    *,
    min_std: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dataset-level mean/std over real action steps from replay episodes."""
    real_actions = [
        episode["actions"][:-1] for episode in episodes if episode["actions"].shape[0] > 1
    ]
    if not real_actions:
        msg = "Need at least one non-empty episode to estimate action stats."
        raise ValueError(msg)
    flat = np.concatenate(real_actions, axis=0).astype(np.float32, copy=False)
    mean = torch.from_numpy(flat.mean(axis=0))
    std = torch.from_numpy(flat.std(axis=0)).clamp(min=float(min_std))
    return mean, std


def symbolize_observations(
    agent: FragileAgent,
    obs: torch.Tensor,
    *,
    obs_normalizer: ObservationNormalizer | None = None,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
) -> dict[str, torch.Tensor]:
    """Encode observations and attach their soft macro-state distribution."""
    obs = _ensure_sequence_input(obs)
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize_tensor(obs)
    encoded = agent.encode_observations(obs, routing_tau=routing_tau)
    out = _symbolize_encoded(
        encoded,
        agent.obs_encoder.encoder.chart_centers,
        agent.obs_encoder.encoder.codebook,
        chart_tau=macro_chart_tau,
        code_tau=macro_code_tau,
    )
    out["encoded"] = encoded
    return out


def symbolize_actions(
    agent: FragileAgent,
    actions: torch.Tensor,
    *,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
) -> dict[str, torch.Tensor]:
    """Encode actions and attach their soft macro-state distribution."""
    actions = _ensure_sequence_input(actions)
    encoded = agent.encode_actions(actions, routing_tau=routing_tau)
    out = _symbolize_encoded(
        encoded,
        agent.act_encoder.encoder.chart_centers,
        agent.act_encoder.encoder.codebook,
        chart_tau=macro_chart_tau,
        code_tau=macro_code_tau,
    )
    out["encoded"] = encoded
    return out


def prepare_macro_transition_batch(
    agent: FragileAgent,
    replay_batch: dict[str, torch.Tensor],
    *,
    obs_normalizer: ObservationNormalizer | None,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
) -> dict[str, Any]:
    """Turn one replay sample into the symbolic tensors used by RL updates."""
    obs = replay_batch["obs"]
    act = replay_batch["actions"]
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize_tensor(obs)
    mask = torch.ones(obs.shape[:2], device=obs.device, dtype=torch.bool)
    forward = agent.forward_batch(
        obs,
        act,
        mask=mask,
        routing_tau=routing_tau,
        macro_chart_tau=macro_chart_tau,
        macro_code_tau=macro_code_tau,
        compute_macro=True,
    )
    return prepare_macro_transition_batch_from_forward(forward, replay_batch)


def prepare_macro_transition_batch_from_forward(
    forward: dict[str, Any],
    replay_batch: dict[str, torch.Tensor],
) -> dict[str, Any]:
    """Turn one already-computed agent forward pass into RL replay tensors."""
    transitions = forward["transitions"]
    reward_t = _flatten_selected(replay_batch["rewards"][:, :-1], transitions["valid_idx"])
    done_t = _flatten_selected(replay_batch["dones"][:, :-1], transitions["valid_idx"])
    continuation_t = 1.0 - done_t

    obs_geometry = {
        "chart_centers": forward["macro"]["obs"]["chart_centers"],
        "codebook": forward["macro"]["obs"]["codebook"],
        "state_points": forward["macro"]["obs"]["state_points"],
        "state_tangent_points": forward["macro"]["obs"]["state_tangent_points"],
    }
    act_geometry = {
        "chart_centers": forward["macro"]["act"]["chart_centers"],
        "codebook": forward["macro"]["act"]["codebook"],
        "state_points": forward["macro"]["act"]["state_points"],
        "state_tangent_points": forward["macro"]["act"]["state_tangent_points"],
    }

    return {
        "forward": forward,
        "valid_mask": transition_valid_mask(replay_batch["dones"]),
        "obs_state_probs_t": transitions["obs_state_probs_t_valid"],
        "obs_state_probs_tp1": transitions["obs_state_probs_tp1_valid"],
        "act_state_probs_t": transitions["act_state_probs_t_valid"],
        "obs_state_idx_t": transitions["obs_state_idx_t_valid"],
        "obs_state_idx_tp1": transitions["obs_state_idx_tp1_valid"],
        "act_state_idx_t": transitions["act_state_idx_t_valid"],
        "obs_chart_tp1": transitions["obs_chart_tp1_valid"],
        "obs_code_tp1": transitions["obs_code_tp1_valid"],
        "reward_t": reward_t,
        "done_t": done_t,
        "continuation_t": continuation_t,
        "obs_geometry": obs_geometry,
        "act_geometry": act_geometry,
    }


def fit_action_symbol_prototypes(
    agent: FragileAgent,
    episodes: list[dict[str, np.ndarray]],
    *,
    device: torch.device | str,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
    min_count: int = 1,
    chunk_size: int = 4096,
) -> ActionPrototypeTable:
    """Estimate one continuous prototype action per learned macro action symbol."""
    real_actions = [
        episode["actions"][:-1] for episode in episodes if episode["actions"].shape[0] > 1
    ]
    action_dim = agent.config.act_encoder.input_dim
    num_actions = agent.num_act_states
    if not real_actions:
        return ActionPrototypeTable(
            means=torch.zeros(num_actions, action_dim),
            counts=torch.zeros(num_actions),
            valid=torch.zeros(num_actions, dtype=torch.bool),
        )

    flat_actions = torch.from_numpy(np.concatenate(real_actions, axis=0)).to(
        device=device,
        dtype=torch.float32,
    )
    sums = torch.zeros(num_actions, action_dim, device=device)
    counts = torch.zeros(num_actions, device=device)

    with torch.inference_mode():
        for start in range(0, flat_actions.shape[0], max(int(chunk_size), 1)):
            action_chunk = flat_actions[start : start + max(int(chunk_size), 1)]
            symbolized = symbolize_actions(
                agent,
                action_chunk,
                routing_tau=routing_tau,
                macro_chart_tau=macro_chart_tau,
                macro_code_tau=macro_code_tau,
            )
            state_idx = symbolized["state_idx_valid"]
            sums.index_add_(0, state_idx, action_chunk)
            counts.index_add_(0, state_idx, torch.ones_like(state_idx, dtype=sums.dtype))

    means = torch.zeros_like(sums)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(-1)
    valid = counts >= float(min_count)
    return ActionPrototypeTable(
        means=means.detach().cpu(),
        counts=counts.detach().cpu(),
        valid=valid.detach().cpu(),
    )


def update_action_symbol_prototypes(
    agent: FragileAgent,
    episodes: list[dict[str, np.ndarray]],
    current: ActionPrototypeTable | None,
    *,
    device: torch.device | str,
    routing_tau: float,
    macro_chart_tau: float,
    macro_code_tau: float,
    min_count: int = 1,
    ema: float = 0.9,
    chunk_size: int = 4096,
) -> ActionPrototypeTable:
    """Refresh action prototypes from replay with optional EMA smoothing."""
    fresh = fit_action_symbol_prototypes(
        agent,
        episodes,
        device=device,
        routing_tau=routing_tau,
        macro_chart_tau=macro_chart_tau,
        macro_code_tau=macro_code_tau,
        min_count=min_count,
        chunk_size=chunk_size,
    )
    if current is None:
        return fresh
    if current.means.shape != fresh.means.shape:
        return fresh

    ema = float(ema)
    if not 0.0 <= ema <= 1.0:
        msg = "ema must lie in [0, 1]."
        raise ValueError(msg)

    current_cpu = current.to("cpu")
    means = current_cpu.means.clone()
    update_mask = fresh.valid
    means[update_mask] = (
        ema * current_cpu.means[update_mask] + (1.0 - ema) * fresh.means[update_mask]
    )
    counts = torch.where(update_mask, fresh.counts, current_cpu.counts)
    valid = current_cpu.valid | fresh.valid
    return ActionPrototypeTable(means=means, counts=counts, valid=valid)


def update_action_symbol_prototypes_from_rollouts(
    episodes: list[dict[str, np.ndarray]],
    infos: list[dict[str, Any]],
    current: ActionPrototypeTable | None,
    *,
    num_actions: int,
    action_dim: int,
    min_count: int = 1,
    ema: float = 0.9,
) -> ActionPrototypeTable | None:
    """Update action prototypes from newly collected labeled rollouts only.

    The online control loop already knows which discrete macro action it chose.
    Reusing those labels avoids re-encoding the full replay buffer each epoch.
    """
    if not episodes or not infos:
        return current

    if len(episodes) != len(infos):
        msg = "episodes and infos must have the same length."
        raise ValueError(msg)
    if num_actions <= 0 or action_dim <= 0:
        msg = "num_actions and action_dim must both be positive."
        raise ValueError(msg)

    ema = float(ema)
    if not 0.0 <= ema <= 1.0:
        msg = "ema must lie in [0, 1]."
        raise ValueError(msg)

    sums = torch.zeros(num_actions, action_dim, dtype=torch.float32)
    counts = torch.zeros(num_actions, dtype=torch.float32)
    for episode, info in zip(episodes, infos, strict=False):
        action_idx = np.asarray(info.get("action_indices", []), dtype=np.int64)
        if action_idx.size == 0:
            continue
        actions = np.asarray(episode["actions"][:-1], dtype=np.float32)
        length = min(int(action_idx.shape[0]), int(actions.shape[0]))
        if length <= 0:
            continue
        action_tensor = torch.from_numpy(actions[:length])
        index_tensor = torch.from_numpy(action_idx[:length]).to(dtype=torch.long)
        valid_rows = (index_tensor >= 0) & (index_tensor < num_actions)
        if not bool(valid_rows.any()):
            continue
        action_tensor = action_tensor[valid_rows]
        index_tensor = index_tensor[valid_rows]
        sums.index_add_(0, index_tensor, action_tensor)
        counts.index_add_(0, index_tensor, torch.ones_like(index_tensor, dtype=torch.float32))

    if not bool((counts > 0).any()):
        return current

    means = torch.zeros_like(sums)
    nonzero = counts > 0
    means[nonzero] = sums[nonzero] / counts[nonzero].unsqueeze(-1)

    if current is None:
        valid = counts >= float(min_count)
        return ActionPrototypeTable(
            means=means,
            counts=counts,
            valid=valid,
        )

    if current.means.shape != means.shape:
        msg = "current prototype table shape does not match num_actions/action_dim."
        raise ValueError(msg)

    current_cpu = current.to("cpu")
    updated_means = current_cpu.means.clone()
    update_mask = counts > 0
    updated_means[update_mask] = (
        ema * current_cpu.means[update_mask] + (1.0 - ema) * means[update_mask]
    )
    updated_counts = current_cpu.counts + counts
    updated_valid = updated_counts >= float(min_count)
    return ActionPrototypeTable(
        means=updated_means,
        counts=updated_counts,
        valid=updated_valid,
    )


def action_symbol_to_continuous(
    action_idx: int | torch.Tensor,
    prototypes: ActionPrototypeTable | None,
    *,
    action_min: np.ndarray,
    action_max: np.ndarray,
    sigma_motor: float,
) -> np.ndarray:
    """Map a discrete macro action symbol to an executable continuous action."""
    if torch.is_tensor(action_idx):
        action_idx = int(action_idx.item())
    if prototypes is None or action_idx < 0 or action_idx >= prototypes.means.shape[0]:
        action_mean = np.random.uniform(action_min, action_max).astype(np.float32)
    elif bool(prototypes.valid[action_idx].item()):
        action_mean = (
            prototypes.means[action_idx].detach().cpu().numpy().astype(np.float32, copy=False)
        )
    else:
        action_mean = np.random.uniform(action_min, action_max).astype(np.float32)
    return _sample_collection_action(
        action_mean,
        action_min=action_min,
        action_max=action_max,
        sigma_motor=sigma_motor,
    )


def replay_buffer_state(buffer: SequenceReplayBuffer) -> dict[str, Any]:
    """Serialize replay-buffer episodes for checkpointing."""
    episodes = []
    for episode in buffer._episodes:  # noqa: SLF001 - internal storage is the replay state.
        episodes.append({key: np.array(value, copy=True) for key, value in episode.items()})
    return {
        "capacity": int(buffer.capacity),
        "seq_len": int(buffer.seq_len),
        "episodes": episodes,
    }


def replay_buffer_from_state(state: dict[str, Any]) -> SequenceReplayBuffer:
    """Restore a replay buffer from checkpointed episode arrays."""
    buffer = SequenceReplayBuffer(capacity=int(state["capacity"]), seq_len=int(state["seq_len"]))
    for episode in state.get("episodes", []):
        buffer.add_episode({key: np.array(value, copy=True) for key, value in episode.items()})
    return buffer


__all__ = [
    "ActionPrototypeTable",
    "action_stats_from_episodes",
    "action_symbol_to_continuous",
    "build_macro_episode_dict",
    "fit_action_symbol_prototypes",
    "prepare_macro_transition_batch",
    "prepare_macro_transition_batch_from_forward",
    "replay_buffer_from_state",
    "replay_buffer_state",
    "symbolize_actions",
    "symbolize_observations",
    "transition_valid_mask",
    "update_action_symbol_prototypes",
    "update_action_symbol_prototypes_from_rollouts",
]
