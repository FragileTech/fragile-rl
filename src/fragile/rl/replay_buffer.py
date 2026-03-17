"""Simple sequence replay buffer for Dreamer-style training."""

from __future__ import annotations

import numpy as np
import torch


class SequenceReplayBuffer:
    """Stores full episodes and samples contiguous sub-sequences.

    Each episode is a dict of numpy arrays with a leading time axis.
    Sampling returns a batch of sub-sequences of length ``seq_len``.
    """

    def __init__(self, capacity: int, seq_len: int) -> None:
        self.capacity = capacity
        self.seq_len = seq_len
        self._episodes: list[dict[str, np.ndarray]] = []
        self._valid_starts: list[np.ndarray] = []
        self._total_steps = 0
        self._total_windows = 0

    @property
    def total_steps(self) -> int:
        return self._total_steps

    @property
    def num_episodes(self) -> int:
        return len(self._episodes)

    def add_episode(self, episode: dict[str, np.ndarray]) -> None:
        """Add an episode dict with keys: obs, actions, rewards, dones."""
        ep_len = len(episode["obs"])
        self._episodes.append(episode)
        if ep_len >= self.seq_len + 1:
            starts = np.arange(ep_len - self.seq_len, dtype=np.int64)
        else:
            starts = np.empty(0, dtype=np.int64)
        self._valid_starts.append(starts)
        self._total_steps += ep_len
        self._total_windows += int(starts.size)
        # Evict oldest episodes if over capacity
        while self._total_steps > self.capacity and len(self._episodes) > 1:
            removed = self._episodes.pop(0)
            removed_starts = self._valid_starts.pop(0)
            self._total_steps -= len(removed["obs"])
            self._total_windows -= int(removed_starts.size)

    def sample(
        self, batch_size: int, device: torch.device | str = "cpu",
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of sub-sequences.

        Returns dict with:
            obs: [B, T, obs_dim]
            actions: [B, T, action_dim]
            rewards: [B, T]
            dones: [B, T]
        """
        if not self._episodes:
            msg = "Cannot sample from an empty replay buffer."
            raise ValueError(msg)

        if self._total_windows > 0:
            counts = np.fromiter(
                (starts.size for starts in self._valid_starts),
                dtype=np.int64,
                count=len(self._valid_starts),
            )
            cumulative = np.cumsum(counts)
            draws = np.random.randint(0, self._total_windows, size=batch_size)
            ep_indices = np.searchsorted(cumulative, draws, side="right")
            prev_cumulative = np.zeros_like(draws)
            valid_rows = ep_indices > 0
            prev_cumulative[valid_rows] = cumulative[ep_indices[valid_rows] - 1]
            start_indices = draws - prev_cumulative
            starts = np.fromiter(
                (
                    self._valid_starts[int(ep_idx)][int(start_idx)]
                    for ep_idx, start_idx in zip(ep_indices, start_indices, strict=False)
                ),
                dtype=np.int64,
                count=batch_size,
            )
            max_len = self.seq_len + 1
        else:
            ep_indices = np.random.randint(0, len(self._episodes), size=batch_size)
            starts = np.zeros(batch_size, dtype=np.int64)
            max_len = max(len(self._episodes[int(ep_idx)]["obs"]) for ep_idx in ep_indices)

        first_episode = self._episodes[int(ep_indices[0])]

        def _shape_and_dtype(key: str):
            for episode in self._episodes:
                if key in episode:
                    return episode[key].shape[1:], episode[key].dtype
            return None, None

        obs_batch = np.zeros(
            (batch_size, max_len, *first_episode["obs"].shape[1:]),
            dtype=first_episode["obs"].dtype,
        )
        act_batch = np.zeros(
            (batch_size, max_len, *first_episode["actions"].shape[1:]),
            dtype=first_episode["actions"].dtype,
        )
        rew_batch = np.zeros((batch_size, max_len), dtype=first_episode["rewards"].dtype)
        done_batch = np.zeros((batch_size, max_len), dtype=first_episode["dones"].dtype)

        action_mean_shape, action_mean_dtype = _shape_and_dtype("action_means")
        action_latent_shape, action_latent_dtype = _shape_and_dtype("action_latents")
        action_router_shape, action_router_dtype = _shape_and_dtype("action_router_weights")
        action_chart_shape, action_chart_dtype = _shape_and_dtype("action_charts")
        action_code_shape, action_code_dtype = _shape_and_dtype("action_codes")
        action_code_latent_shape, action_code_latent_dtype = _shape_and_dtype("action_code_latents")

        action_mean_batch = None
        if action_mean_shape is not None and action_mean_dtype is not None:
            action_mean_batch = np.zeros(
                (batch_size, max_len, *action_mean_shape),
                dtype=action_mean_dtype,
            )

        action_latent_batch = None
        if action_latent_shape is not None and action_latent_dtype is not None:
            action_latent_batch = np.zeros(
                (batch_size, max_len, *action_latent_shape),
                dtype=action_latent_dtype,
            )

        action_router_batch = None
        if action_router_shape is not None and action_router_dtype is not None:
            action_router_batch = np.zeros(
                (batch_size, max_len, *action_router_shape),
                dtype=action_router_dtype,
            )

        action_chart_batch = None
        if action_chart_shape is not None and action_chart_dtype is not None:
            action_chart_batch = np.zeros(
                (batch_size, max_len, *action_chart_shape),
                dtype=action_chart_dtype,
            )

        action_code_batch = None
        if action_code_shape is not None and action_code_dtype is not None:
            action_code_batch = np.zeros(
                (batch_size, max_len, *action_code_shape),
                dtype=action_code_dtype,
            )

        action_code_latent_batch = None
        if action_code_latent_shape is not None and action_code_latent_dtype is not None:
            action_code_latent_batch = np.zeros(
                (batch_size, max_len, *action_code_latent_shape),
                dtype=action_code_latent_dtype,
            )

        for row, (ep_idx, start) in enumerate(zip(ep_indices, starts, strict=False)):
            episode = self._episodes[int(ep_idx)]
            start_i = int(start)
            end_i = min(start_i + max_len, len(episode["obs"]))
            length = end_i - start_i
            sl = slice(start_i, end_i)
            obs_batch[row, :length] = episode["obs"][sl]
            act_batch[row, :length] = episode["actions"][sl]
            rew_batch[row, :length] = episode["rewards"][sl]
            done_batch[row, :length] = episode["dones"][sl]
            if action_mean_batch is not None:
                action_mean_batch[row, :length] = episode.get("action_means", episode["actions"])[sl]
            if action_latent_batch is not None and "action_latents" in episode:
                action_latent_batch[row, :length] = episode["action_latents"][sl]
            if action_router_batch is not None and "action_router_weights" in episode:
                action_router_batch[row, :length] = episode["action_router_weights"][sl]
            if action_chart_batch is not None and "action_charts" in episode:
                action_chart_batch[row, :length] = episode["action_charts"][sl]
            if action_code_batch is not None and "action_codes" in episode:
                action_code_batch[row, :length] = episode["action_codes"][sl]
            if action_code_latent_batch is not None and "action_code_latents" in episode:
                action_code_latent_batch[row, :length] = episode["action_code_latents"][sl]

        batch = {
            "obs": torch.from_numpy(obs_batch).to(device=device, dtype=torch.float32),
            "actions": torch.from_numpy(act_batch).to(device=device, dtype=torch.float32),
            "rewards": torch.from_numpy(rew_batch).to(device=device, dtype=torch.float32),
            "dones": torch.from_numpy(done_batch).to(device=device, dtype=torch.float32),
        }
        if action_mean_batch is not None:
            batch["action_means"] = torch.from_numpy(action_mean_batch).to(
                device=device,
                dtype=torch.float32,
            )
        if action_latent_batch is not None:
            batch["action_latents"] = torch.from_numpy(action_latent_batch).to(
                device=device,
                dtype=torch.float32,
            )
        if action_router_batch is not None:
            batch["action_router_weights"] = torch.from_numpy(action_router_batch).to(
                device=device,
                dtype=torch.float32,
            )
        if action_chart_batch is not None:
            batch["action_charts"] = torch.from_numpy(action_chart_batch).to(
                device=device,
                dtype=torch.long,
            )
        if action_code_batch is not None:
            batch["action_codes"] = torch.from_numpy(action_code_batch).to(
                device=device,
                dtype=torch.long,
            )
        if action_code_latent_batch is not None:
            batch["action_code_latents"] = torch.from_numpy(action_code_latent_batch).to(
                device=device,
                dtype=torch.float32,
            )
        return batch
