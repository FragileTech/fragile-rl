"""Unit tests for standalone macro-RL data helpers."""

from __future__ import annotations

import numpy as np
import torch

from fragile.agent import FragileAgent, FragileAgentConfig
from fragile.rl.env_helpers import ObservationNormalizer
from fragile.rl.macro_data import (
    action_stats_from_episodes,
    build_macro_episode_dict,
    fit_action_symbol_prototypes,
    prepare_macro_transition_batch,
    prepare_macro_transition_batch_from_forward,
    transition_valid_mask,
    update_action_symbol_prototypes_from_rollouts,
)
from fragile.rl.replay_buffer import SequenceReplayBuffer
from fragile.vla.config import VLAConfig


def _tiny_agent() -> FragileAgent:
    cfg = FragileAgentConfig(
        obs_encoder=VLAConfig(
            input_dim=3,
            feature_dim=3,
            hidden_dim=24,
            latent_dim=4,
            num_charts=2,
            codes_per_chart=2,
            chart_ot_iters=4,
        ),
        act_encoder=VLAConfig(
            input_dim=2,
            feature_dim=2,
            hidden_dim=24,
            latent_dim=4,
            num_charts=2,
            codes_per_chart=2,
            chart_ot_iters=4,
            input_affine_enabled=True,
        ),
        enclosure_hidden_dim=16,
        markov_hidden_dim=16,
    )
    return FragileAgent(cfg)


def _make_episode(offset: float) -> dict[str, np.ndarray]:
    obs = [
        np.array([0.0, 0.1, 0.2], dtype=np.float32) + offset,
        np.array([0.2, 0.3, 0.4], dtype=np.float32) + offset,
        np.array([0.4, 0.5, 0.6], dtype=np.float32) + offset,
    ]
    act = [
        np.array([0.1, -0.2], dtype=np.float32) + offset,
        np.array([0.3, -0.1], dtype=np.float32) + offset,
    ]
    rew = [np.float32(1.0 + offset), np.float32(0.5 + offset)]
    done = [np.float32(0.0), np.float32(1.0)]
    return build_macro_episode_dict(obs, act, rew, done)


def test_build_macro_episode_dict_repeats_last_action() -> None:
    episode = _make_episode(0.0)
    assert episode["obs"].shape == (3, 3)
    assert episode["actions"].shape == (3, 2)
    assert np.allclose(episode["actions"][-1], episode["actions"][-2])
    assert episode["rewards"][-1] == 0.0
    assert episode["dones"][-1] == 1.0


def test_prepare_macro_transition_batch_and_prototypes() -> None:
    torch.manual_seed(5)
    agent = _tiny_agent()
    episodes = [_make_episode(0.0), _make_episode(0.1)]
    replay = SequenceReplayBuffer(capacity=32, seq_len=1)
    for episode in episodes:
        replay.add_episode(episode)

    obs_normalizer = ObservationNormalizer.from_episodes(episodes, torch.device("cpu"))
    act_mean, act_std = action_stats_from_episodes(episodes)
    agent.act_encoder.set_io_affine_stats(act_mean, act_std, learnable=False)

    prototypes = fit_action_symbol_prototypes(
        agent,
        episodes,
        device=torch.device("cpu"),
        routing_tau=-1.0,
        macro_chart_tau=1.0,
        macro_code_tau=1.0,
        min_count=1,
    )
    batch = replay.sample(batch_size=2, device="cpu")
    macro_batch = prepare_macro_transition_batch(
        agent,
        batch,
        obs_normalizer=obs_normalizer,
        routing_tau=-1.0,
        macro_chart_tau=1.0,
        macro_code_tau=1.0,
    )

    assert prototypes.means.shape == (agent.num_act_states, agent.config.act_encoder.input_dim)
    assert prototypes.counts.shape == (agent.num_act_states,)
    assert transition_valid_mask(batch["dones"]).shape == (2, 1)
    assert macro_batch["obs_state_probs_t"].shape[-1] == agent.num_obs_states
    assert macro_batch["act_state_probs_t"].shape[-1] == agent.num_act_states
    assert macro_batch["reward_t"].shape == macro_batch["continuation_t"].shape

    obs = obs_normalizer.normalize_tensor(batch["obs"])
    forward = agent.forward_batch(
        obs,
        batch["actions"],
        mask=torch.ones(obs.shape[:2], dtype=torch.bool),
        routing_tau=-1.0,
        macro_chart_tau=1.0,
        macro_code_tau=1.0,
        compute_macro=True,
    )
    from_forward = prepare_macro_transition_batch_from_forward(forward, batch)

    assert torch.allclose(
        macro_batch["obs_state_probs_t"],
        from_forward["obs_state_probs_t"],
        atol=2e-3,
        rtol=2e-3,
    )
    assert torch.equal(macro_batch["act_state_idx_t"], from_forward["act_state_idx_t"])
    assert torch.allclose(macro_batch["reward_t"], from_forward["reward_t"])


def test_online_action_prototype_update_uses_new_rollouts_only() -> None:
    episode = _make_episode(0.0)
    info = {"action_indices": [0, 3]}

    prototypes = update_action_symbol_prototypes_from_rollouts(
        [episode],
        [info],
        current=None,
        num_actions=4,
        action_dim=2,
        min_count=1,
        ema=0.9,
    )

    assert prototypes is not None
    assert prototypes.valid.tolist() == [True, False, False, True]
    assert torch.allclose(prototypes.means[0], torch.tensor([0.1, -0.2]))
    assert torch.allclose(prototypes.means[3], torch.tensor([0.3, -0.1]))
