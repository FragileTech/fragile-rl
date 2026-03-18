"""Current-path tests for the theory-aligned Dreamer RL stack."""

from __future__ import annotations

import copy
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from fragile.layers import FactorizedJumpOperator
from fragile.layers.gauge import poincare_exp_map, project_to_ball
from fragile.rl import train_dreamer
from fragile.rl.train_dreamer import _sync_rl_atlas


B = 4
D = 8
A = 6
K = 4
CODES_PER_CHART = 4
D_MODEL = 32
OBS_DIM = 24
H_IMAGINATION = 3
H_WM = 2


def _random_action_form_inputs(
    z: torch.Tensor,
    *,
    num_action_charts: int = K,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_z = torch.randn_like(z) * 0.2
    action_rw = torch.softmax(
        torch.randn(z.shape[0], num_action_charts, device=z.device, dtype=z.dtype),
        dim=-1,
    )
    action_code_z = torch.randn_like(z) * 0.2
    return action_z, action_rw, action_code_z


def _make_replay_episode(length: int) -> dict[str, np.ndarray]:
    router = np.random.rand(length, K).astype(np.float32)
    router /= router.sum(axis=-1, keepdims=True)
    return {
        "obs": np.random.randn(length, OBS_DIM).astype(np.float32),
        "actions": np.random.randn(length, A).astype(np.float32),
        "action_means": np.random.randn(length, A).astype(np.float32),
        "action_latents": np.random.randn(length, D).astype(np.float32),
        "action_router_weights": router,
        "action_charts": np.random.randint(0, K, size=length, dtype=np.int64),
        "action_codes": np.random.randint(0, CODES_PER_CHART, size=length, dtype=np.int64),
        "action_code_latents": np.random.randn(length, D).astype(np.float32),
        "rewards": np.random.randn(length).astype(np.float32),
        "dones": np.zeros(length, dtype=np.float32),
    }


def _make_training_batch(device: torch.device) -> dict[str, torch.Tensor]:
    time_dim = H_IMAGINATION + 3
    rewards = torch.randn(B, time_dim, device=device) * 0.1
    dones = torch.zeros(B, time_dim, device=device)
    return {
        "obs": torch.randn(B, time_dim, OBS_DIM, device=device),
        "actions": torch.randn(B, time_dim, A, device=device) * 0.2,
        "action_means": torch.randn(B, time_dim, A, device=device) * 0.2,
        "rewards": rewards,
        "dones": dones,
    }


def _make_optimizers(
    obs_model: nn.Module,
    jump_op: nn.Module,
    action_model: nn.Module,
    action_jump_op: nn.Module,
    world_model: nn.Module,
    macro_critic: nn.Module,
    reward_head: nn.Module,
    actor: nn.Module,
    enclosure_probe: nn.Module,
) -> tuple[
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
]:
    reward_shared_ids = {
        *(id(param) for param in reward_head.chart_tok.parameters()),
        *(id(param) for param in reward_head.z_embed.parameters()),
    }
    reward_own_params = [
        param for param in reward_head.parameters() if id(param) not in reward_shared_ids
    ]
    optimizer_enc = torch.optim.Adam(
        list(obs_model.parameters())
        + list(jump_op.parameters())
        + list(action_model.parameters())
        + list(action_jump_op.parameters()),
        lr=1e-3,
    )
    optimizer_wm = torch.optim.Adam(
        list(world_model.parameters()) + list(macro_critic.parameters()) + reward_own_params,
        lr=1e-3,
    )
    optimizer_boundary = torch.optim.Adam(actor.parameters(), lr=1e-3)
    optimizer_enclosure = torch.optim.Adam(enclosure_probe.parameters(), lr=1e-3)
    return optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure


def _make_constant_policy_output(
    z_in: torch.Tensor,
    *,
    action_value: float = 0.25,
    code_idx: int = 1,
) -> dict[str, torch.Tensor]:
    batch = z_in.shape[0]
    action_router = torch.zeros(batch, K, device=z_in.device, dtype=z_in.dtype)
    action_router[:, 0] = 1.0
    return {
        "action": torch.full((batch, A), action_value, device=z_in.device, dtype=z_in.dtype),
        "action_mean": torch.full((batch, A), action_value, device=z_in.device, dtype=z_in.dtype),
        "action_canonical": torch.full((batch, D), 0.15, device=z_in.device, dtype=z_in.dtype),
        "action_latent": torch.full((batch, D), 0.15, device=z_in.device, dtype=z_in.dtype),
        "action_latent_mean": torch.full((batch, D), 0.15, device=z_in.device, dtype=z_in.dtype),
        "action_router_weights": action_router,
        "action_chart_idx": torch.zeros(batch, device=z_in.device, dtype=torch.long),
        "action_code_idx": torch.full((batch,), code_idx, device=z_in.device, dtype=torch.long),
        "action_code_latent": torch.full((batch, D), 0.05, device=z_in.device, dtype=z_in.dtype),
    }


class FakeTimeStep:
    def __init__(self, obs, reward: float = 0.0, last: bool = False):
        self.observation = {"obs": np.asarray(obs, dtype=np.float32)}
        self.reward = reward
        self._last = last

    def last(self) -> bool:
        return self._last


class FakeActionSpec:
    def __init__(self, action_dim: int = A):
        self.minimum = -np.ones(action_dim, dtype=np.float32)
        self.maximum = np.ones(action_dim, dtype=np.float32)
        self.shape = (action_dim,)


class SingleEpisodeEnv:
    def __init__(
        self,
        *,
        start_obs: list[float] | np.ndarray,
        next_obs: list[float] | np.ndarray,
        reward: float = 1.0,
        action_dim: int = A,
    ):
        self.start_obs = np.asarray(start_obs, dtype=np.float32)
        self.next_obs = np.asarray(next_obs, dtype=np.float32)
        self.reward = reward
        self._step = 0
        self._action_spec = FakeActionSpec(action_dim)

    def reset(self):
        self._step = 0
        return FakeTimeStep(self.start_obs, last=False)

    def action_spec(self):
        return self._action_spec

    def step(self, action):
        del action
        self._step += 1
        return FakeTimeStep(self.next_obs, reward=self.reward, last=self._step >= 1)


class ParallelEnv:
    def __init__(self):
        self.n_workers = 2
        self.action_space = FakeActionSpec(action_dim=A)
        self.steps = np.zeros(2, dtype=np.int32)
        self.batch_calls: list[list[int]] = []

    def reset_batch(self, env_indices=None, seeds=None):
        del seeds
        indices = np.asarray(env_indices, dtype=int)
        self.steps[indices] = 0
        return np.stack(
            [np.full(OBS_DIM, float(idx) + 0.1, dtype=np.float32) for idx in indices],
        )

    def step_actions_batch(self, actions, dt=None, env_indices=None):
        del actions, dt
        indices = np.asarray(env_indices, dtype=int)
        self.batch_calls.append(indices.tolist())
        obs = []
        rewards = np.zeros(len(indices), dtype=np.float32)
        dones = np.zeros(len(indices), dtype=bool)
        truncated = np.zeros(len(indices), dtype=bool)
        infos = [{} for _ in range(len(indices))]
        for row, idx in enumerate(indices):
            self.steps[idx] += 1
            rewards[row] = float(idx + 1)
            done_after = 1 if idx == 0 else 2
            dones[row] = self.steps[idx] >= done_after
            obs.append(np.full(OBS_DIM, float(idx + self.steps[idx]), dtype=np.float32))
        return np.stack(obs), rewards, dones, truncated, infos


class RecordingEncoder:
    def __init__(self, *, latent_dim: int = D, num_charts: int = K):
        self.latent_dim = latent_dim
        self.num_charts = num_charts
        self.calls: list[tuple[bool, float]] = []
        self.obs_seen: list[torch.Tensor] = []

    def __call__(self, obs_t, *, routing_tau=1.0, **_kwargs):
        self.calls.append((False, routing_tau))
        self.obs_seen.append(obs_t.clone())
        batch = obs_t.shape[0]
        rw = torch.zeros(batch, self.num_charts, device=obs_t.device, dtype=obs_t.dtype)
        rw[:, 0] = 1.0
        z = torch.full((batch, self.latent_dim), 0.25, device=obs_t.device, dtype=obs_t.dtype)
        z_q = torch.full_like(z, 0.05)
        out = [None] * 12
        out[4] = rw
        out[5] = z
        out[11] = z_q
        return tuple(out)


class FakeEncoderModel:
    def __init__(self, *, latent_dim: int = D, num_charts: int = K):
        self.encoder = RecordingEncoder(latent_dim=latent_dim, num_charts=num_charts)


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z(device):
    return torch.randn(B, D, device=device) * 0.2


@pytest.fixture
def rw(device):
    return torch.softmax(torch.randn(B, K, device=device), dim=-1)


@pytest.fixture
def actor():
    from fragile.rl.actor import GeometricActor

    return GeometricActor(
        latent_dim=D,
        num_obs_charts=K,
        obs_codes_per_chart=CODES_PER_CHART,
        num_action_charts=K,
        action_codes_per_chart=CODES_PER_CHART,
        d_model=D_MODEL,
    )


@pytest.fixture
def actor_old(actor):
    old_actor = copy.deepcopy(actor)
    old_actor.eval()
    for param in old_actor.parameters():
        param.requires_grad_(False)
    return old_actor


@pytest.fixture
def standalone_critic():
    from fragile.rl.critic import GeometricCritic

    return GeometricCritic(D, K, D_MODEL)


@pytest.fixture
def obs_model():
    from fragile.vla.shared_dyn.encoder import SharedDynTopoEncoder

    return SharedDynTopoEncoder(
        input_dim=OBS_DIM,
        hidden_dim=D_MODEL,
        latent_dim=D,
        num_charts=K,
        codes_per_chart=CODES_PER_CHART,
    )


@pytest.fixture
def action_model():
    from fragile.vla.shared_dyn.encoder import SharedDynTopoEncoder

    return SharedDynTopoEncoder(
        input_dim=A,
        hidden_dim=D_MODEL,
        latent_dim=D,
        num_charts=K,
        codes_per_chart=CODES_PER_CHART,
    )


@pytest.fixture
def jump_op():

    return FactorizedJumpOperator(num_charts=K, latent_dim=D)


@pytest.fixture
def action_jump_op():

    return FactorizedJumpOperator(num_charts=K, latent_dim=D)


@pytest.fixture
def world_model():
    from fragile.vla.covariant_world_model import GeometricWorldModel

    return GeometricWorldModel(
        latent_dim=D,
        action_dim=A,
        control_dim=D,
        num_charts=K,
        d_model=D_MODEL,
        hidden_dim=D_MODEL,
        n_refine_steps=2,
        use_boris=False,
        use_jump=False,
    )


@pytest.fixture
def critic(world_model):
    return world_model.potential_net


@pytest.fixture
def macro_critic():
    from fragile.rl.train_dreamer import MacroValueModel

    return MacroValueModel(K, CODES_PER_CHART, K, CODES_PER_CHART)


@pytest.fixture
def reward_head(world_model):
    from fragile.rl.reward_head import RewardHead

    return RewardHead(world_model.potential_net, K, D_MODEL)


@pytest.fixture
def enclosure_probe():
    from fragile.losses.old_macro import EnclosureProbe

    return EnclosureProbe(
        chart_dim=D,
        action_dim=D,
        ztex_dim=D,
        num_charts=K,
        codes_per_chart=CODES_PER_CHART,
        hidden_dim=D_MODEL,
        alpha=1.0,
    )


@pytest.fixture
def config(tmp_path):
    from fragile.rl.config import DreamerConfig

    return DreamerConfig(
        device="cpu",
        obs_dim=OBS_DIM,
        action_dim=A,
        latent_dim=D,
        num_charts=K,
        num_action_charts=K,
        d_model=D_MODEL,
        hidden_dim=D_MODEL,
        codes_per_chart=CODES_PER_CHART,
        action_codes_per_chart=CODES_PER_CHART,
        wm_prediction_horizon=H_WM,
        imagination_horizon=H_IMAGINATION,
        batch_size=B,
        seq_len=6,
        total_epochs=2,
        checkpoint_dir=str(tmp_path / "ckpt"),
        normalize_observations=False,
        actor_return_horizon=2,
        actor_return_batch_size=2,
        actor_return_update_every=1,
        actor_return_warmup_epochs=0,
        reward_curl_batch_limit=2,
    )


@pytest.fixture
def phase1_cfg(config):
    from fragile.rl.train_dreamer import _phase1_config

    return _phase1_config(config)


@pytest.fixture
def action_phase1_cfg(config):
    from fragile.rl.train_dreamer import _phase1_config

    return _phase1_config(
        config,
        input_dim=config.action_dim,
        num_charts=config.num_action_charts,
        codes_per_chart=config.action_codes_per_chart,
    )


class TestGeometricActor:
    def test_forward_outputs_structured_action_state(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        obs_info = symbolize_latent_with_atlas(
            obs_model, z, hard_routing=False, hard_routing_tau=1.0
        )
        out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        assert out["action_chart_logits"].shape == (B, K)
        assert out["action_code_logits"].shape == (B, K, CODES_PER_CHART)
        assert out["action_z_n"].shape == (B, D)
        assert out["action_z_geo"].shape == (B, D)
        assert (out["action_z_geo"].norm(dim=-1) < 1.0).all()

    def test_sample_and_mode_latents_are_deterministic(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        actor.eval()
        obs_info = symbolize_latent_with_atlas(
            obs_model, z, hard_routing=False, hard_routing_tau=1.0
        )
        sample = actor.sample_latent(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        mode = actor.mode_latent(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
        )
        torch.testing.assert_close(sample["action_z_geo"], mode["action_z_geo"], atol=1e-6, rtol=0)
        torch.testing.assert_close(
            mode["action_z_geo"],
            actor.mode_latent(obs_info["chart_idx"], obs_info["code_idx"], obs_info["z_n"])[
                "action_z_geo"
            ],
            atol=1e-6,
            rtol=0,
        )

    def test_gradients_flow_through_actor(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        obs_info = symbolize_latent_with_atlas(
            obs_model, z, hard_routing=False, hard_routing_tau=1.0
        )
        action_out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        action_out["action_z_geo"].mean().backward()
        grad_norms = [
            param.grad.norm().item() for param in actor.parameters() if param.grad is not None
        ]
        assert grad_norms
        assert any(norm > 0.0 for norm in grad_norms)


class TestGeometricCritic:
    def test_forward_shape_and_finiteness(self, standalone_critic, z, rw):
        value = standalone_critic(z, rw)
        assert value.shape == (B, 1)
        assert torch.isfinite(value).all()

    def test_gradient_flows_to_state(self, standalone_critic, z, rw):
        z_req = z.clone().requires_grad_(True)
        standalone_critic(z_req, rw).sum().backward()
        assert z_req.grad is not None
        assert torch.isfinite(z_req.grad).all()


class TestBoundaryGeometry:
    def test_raise_lower_roundtrip(self, z):
        from fragile.rl.boundary import lower_control, raise_control

        control_tan = torch.randn_like(z)
        control_cov = lower_control(z, control_tan)
        control_tan_rt = raise_control(z, control_cov)
        torch.testing.assert_close(control_tan_rt, control_tan, atol=1e-5, rtol=1e-5)

    def test_critic_control_field_matches_metric_raise(self, standalone_critic, z, rw):
        from fragile.rl.boundary import critic_control_field, lower_control

        control_cov, control_tan, value = critic_control_field(standalone_critic, z, rw)
        assert control_cov.shape == (B, D)
        assert control_tan.shape == (B, D)
        assert value.shape == (B, 1)
        torch.testing.assert_close(
            lower_control(z, control_tan), control_cov, atol=1e-5, rtol=1e-5
        )


class TestRewardHead:
    def test_decompose_uses_current_action_manifold_inputs(self, reward_head, z, rw):
        action_z, action_rw, action_code_z = _random_action_form_inputs(z)
        control = torch.randn_like(z)
        info = reward_head.decompose(
            z,
            rw,
            action_z,
            action_rw,
            action_code_z,
            action_canonical=control,
            exact_covector=torch.randn_like(z),
        )
        assert info["reward_nonconservative"].shape == (B, 1)
        assert info["reward_density"].shape == (B, 1)
        assert info["reward_form_cov"].shape == (B, D)
        assert info["reward_form_cov_raw"].shape == (B, D)
        assert info["reward_form_exact_component"].shape == (B, D)

    def test_chart_and_embedding_are_shared_with_potential_net(self, reward_head, world_model):
        assert reward_head.chart_tok is world_model.potential_net.chart_tok
        assert reward_head.z_embed is world_model.potential_net.z_embed


class TestPolicyAction:
    def test_outputs_action_manifold_schema(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
        rw,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _policy_action, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        out = _policy_action(
            actor,
            action_model,
            obs_info,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        assert out["action"].shape == (B, A)
        assert out["action_mean"].shape == (B, A)
        assert out["action_canonical"].shape == (B, D)
        assert out["action_latent"].shape == (B, D)
        assert out["action_latent_mean"].shape == (B, D)
        assert out["action_router_weights"].shape == (B, K)
        assert out["action_chart_idx"].shape == (B,)
        assert out["action_code_idx"].shape == (B,)
        assert out["action_code_latent"].shape == (B, D)
        torch.testing.assert_close(out["action"], out["action_mean"])
        torch.testing.assert_close(out["action_canonical"], out["action_latent"])
        torch.testing.assert_close(out["action_latent"], out["action_latent_mean"])
        torch.testing.assert_close(
            out["action_router_weights"].sum(dim=-1),
            torch.ones(B, device=out["action_router_weights"].device),
        )

    def test_preserves_actor_structured_action_state(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _policy_action, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        actor.eval()
        obs_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        actor_out = actor(
            obs_info["chart_idx"],
            obs_info["code_idx"],
            obs_info["z_n"],
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        out = _policy_action(
            actor,
            action_model,
            obs_info,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        torch.testing.assert_close(out["action_canonical"], actor_out["action_z_geo"])
        torch.testing.assert_close(out["action_latent"], actor_out["action_z_geo"])
        torch.testing.assert_close(
            out["action_router_weights"], actor_out["action_router_weights"]
        )
        torch.testing.assert_close(out["action_code_latent"], actor_out["action_z_q"])
        torch.testing.assert_close(out["action_chart_idx"], actor_out["action_chart_idx"])
        torch.testing.assert_close(out["action_code_idx"], actor_out["action_code_idx"])


class TestReplayBuffer:
    def test_samples_current_action_manifold_schema(self):
        from fragile.rl.replay_buffer import SequenceReplayBuffer

        buffer = SequenceReplayBuffer(capacity=10_000, seq_len=3)
        buffer.add_episode(_make_replay_episode(10))
        batch = buffer.sample(2, device="cpu")
        assert batch["obs"].shape == (2, 4, OBS_DIM)
        assert batch["actions"].shape == (2, 4, A)
        assert batch["action_latents"].shape == (2, 4, D)
        assert batch["action_router_weights"].shape == (2, 4, K)
        assert batch["action_charts"].shape == (2, 4)
        assert batch["action_codes"].shape == (2, 4)
        assert batch["action_code_latents"].shape == (2, 4, D)

    def test_short_episode_fallback_keeps_schema(self):
        from fragile.rl.replay_buffer import SequenceReplayBuffer

        buffer = SequenceReplayBuffer(capacity=10_000, seq_len=100)
        buffer.add_episode(_make_replay_episode(5))
        batch = buffer.sample(2, device="cpu")
        assert batch["obs"].shape[0] == 2
        assert batch["action_router_weights"].shape[1] == 5


class TestDreamerHelpers:
    def test_build_episode_dict_appends_zero_tail(self):
        from fragile.rl.train_dreamer import _build_episode_dict

        obs = [np.ones(OBS_DIM, dtype=np.float32), np.full(OBS_DIM, 2.0, dtype=np.float32)]
        actions = [np.ones(A, dtype=np.float32)]
        rewards = [np.float32(1.0)]
        dones = [np.float32(0.0)]
        action_means = [np.full(A, 0.5, dtype=np.float32)]
        action_latents = [np.full(D, 0.1, dtype=np.float32)]
        action_router = [np.eye(K, dtype=np.float32)[0]]
        action_charts = [np.int64(2)]
        action_codes = [np.int64(3)]
        action_code_latents = [np.full(D, 0.2, dtype=np.float32)]

        episode = _build_episode_dict(
            obs,
            actions,
            rewards,
            dones,
            action_means,
            action_latents,
            action_router,
            action_charts,
            action_codes,
            action_code_latents,
        )

        assert episode["obs"].shape == (2, OBS_DIM)
        assert episode["actions"].shape == (2, A)
        assert episode["action_latents"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert episode["actions"][-1].sum() == 0.0
        assert episode["action_latents"][-1].sum() == 0.0
        assert episode["rewards"][-1] == 0.0
        assert episode["dones"][-1] == 1.0

    def test_optimizer_parameters_deduplicate_shared_tensors(self):
        from fragile.rl.train_dreamer import _optimizer_parameters

        param = nn.Parameter(torch.randn(3))
        optimizer = SimpleNamespace(param_groups=[{"params": [param, param]}])
        params = _optimizer_parameters(optimizer)
        assert params == [param]

    def test_rollout_routing_tau_preserves_configured_temperature(self):
        from fragile.rl.train_dreamer import _rollout_routing_tau

        assert _rollout_routing_tau(True, 0.7) == pytest.approx(0.7)
        assert _rollout_routing_tau(False, 0.7) == pytest.approx(0.7)

    def test_sample_collection_action_adds_thermal_noise(self, monkeypatch):
        from fragile.rl import train_dreamer

        monkeypatch.setattr(
            train_dreamer.np.random,
            "normal",
            lambda *_args, **kwargs: np.full(kwargs["size"], kwargs["scale"], dtype=np.float32),
        )
        action = train_dreamer._sample_collection_action(
            np.array([0.1], dtype=np.float32),
            action_min=np.array([-1.0], dtype=np.float32),
            action_max=np.array([1.0], dtype=np.float32),
            sigma_motor=0.2,
        )

        np.testing.assert_allclose(action, np.array([0.3], dtype=np.float32))


class TestObservationNormalizer:
    def test_round_trip_tensor(self, device):
        from fragile.rl.train_dreamer import ObservationNormalizer

        normalizer = ObservationNormalizer(
            mean=torch.arange(OBS_DIM, device=device, dtype=torch.float32),
            std=torch.full((OBS_DIM,), 2.0, device=device),
        )
        obs = torch.randn(3, OBS_DIM, device=device)
        denorm = normalizer.denormalize_tensor(normalizer.normalize_tensor(obs))
        torch.testing.assert_close(denorm, obs)

    def test_from_episodes_clamps_small_std(self, device):
        from fragile.rl.train_dreamer import ObservationNormalizer

        episodes = [
            {"obs": np.ones((3, OBS_DIM), dtype=np.float32)},
            {"obs": np.ones((2, OBS_DIM), dtype=np.float32)},
        ]
        normalizer = ObservationNormalizer.from_episodes(episodes, device=device, min_std=0.25)
        assert float(normalizer.std.min()) == pytest.approx(0.25)


class TestConfigAndParseArgs:
    def test_config_canonicalizes_action_atlas_sizes(self):
        from fragile.rl.config import DreamerConfig

        cfg = DreamerConfig(
            num_charts=5, num_action_charts=0, action_codes_per_chart=0, codes_per_chart=7
        )
        assert cfg.num_action_charts == 5
        assert cfg.num_action_macros == 5
        assert cfg.action_codes_per_chart == 7

    def test_apply_cartpole_task_preset_overrides_default_sized_config(self):
        from fragile.rl.config import DreamerConfig
        from fragile.rl.train_dreamer import _apply_task_preset

        cfg = DreamerConfig(domain="cartpole", task="swingup")
        preset_name, changes = _apply_task_preset(cfg)

        assert preset_name == "cartpole_swingup"
        assert cfg.latent_dim == 8
        assert cfg.num_charts == 4
        assert cfg.num_action_charts == 4
        assert cfg.num_action_macros == 4
        assert cfg.codes_per_chart == 16
        assert cfg.action_codes_per_chart == 16
        assert cfg.d_model == 64
        assert cfg.hidden_dim == 128
        assert cfg.max_episode_steps == 500
        assert cfg.seed_episodes == 24
        assert cfg.batch_size == 8
        assert cfg.seq_len == 64
        assert cfg.imagination_horizon == 12
        assert cfg.actor_return_horizon == 12
        assert cfg.hard_routing
        assert cfg.hard_routing_warmup_epochs == 0
        assert cfg.hard_routing_tau == pytest.approx(1.0)
        assert cfg.hard_routing_tau_end == pytest.approx(1.0)
        assert cfg.hard_routing_tau_anneal_epochs == 0
        assert cfg.w_entropy == pytest.approx(0.05)
        assert cfg.w_diversity == pytest.approx(2.0)
        assert cfg.w_critic == pytest.approx(1.0)
        assert cfg.w_screened_poisson == pytest.approx(2.0)
        assert cfg.screened_poisson_warmup_epochs == 20
        assert cfg.w_reward_conservative_match == pytest.approx(10.0)
        assert cfg.w_critic_stiffness == pytest.approx(5.0)
        assert cfg.w_critic_exact_increment == pytest.approx(1.0)
        assert cfg.w_critic_covector_align == pytest.approx(5.0)
        assert cfg.critic_covector_warmup_epochs == 10
        assert cfg.critic_stiffness_warmup_epochs == 20
        assert cfg.critic_macro_pullback_warmup_epochs == 15
        assert cfg.critic_on_policy_warmup_epochs == 15
        assert cfg.critic_grad_metrics_every == 1
        assert cfg.critic_multistep_horizon == 16
        assert cfg.critic_multistep_decay == pytest.approx(0.8)
        assert cfg.w_critic_on_policy_covector_align == pytest.approx(5.0)
        assert cfg.w_critic_on_policy_stiffness == pytest.approx(2.0)
        assert cfg.critic_on_policy_horizon == 12
        assert cfg.critic_on_policy_batch_size == 8
        assert cfg.critic_on_policy_decay == pytest.approx(0.9)
        assert cfg.w_macro_value == pytest.approx(0.5)
        assert cfg.w_macro_exact_increment == pytest.approx(1.0)
        assert cfg.w_macro_pullback == pytest.approx(0.5)
        assert cfg.w_macro_covector_pullback == pytest.approx(0.5)
        assert cfg.w_macro_on_policy_pullback == pytest.approx(0.25)
        assert cfg.w_macro_on_policy_covector_pullback == pytest.approx(0.25)
        assert cfg.macro_multistep_horizon == 16
        assert cfg.macro_on_policy_horizon == 12
        assert cfg.critic_stiffness_min == pytest.approx(0.001)
        assert cfg.critic_stiffness_target_max == pytest.approx(0.05)
        assert cfg.w_actor_curiosity == pytest.approx(0.2)
        assert cfg.reward_nonconservative_budget_ratio == pytest.approx(0.05)
        assert cfg.actor_return_chart_acc_target == pytest.approx(0.5)
        assert cfg.actor_return_update_every == 2
        assert cfg.actor_return_warmup_epochs == 2
        assert cfg.actor_metric_fisher_scale == pytest.approx(0.01)
        assert cfg.actor_stiffness_min == pytest.approx(0.001)
        assert cfg.actor_supervise_warmup_epochs == 2
        assert cfg.actor_supervise_decay_epochs == 20
        assert cfg.actor_supervise_min_scale == pytest.approx(0.05)
        assert cfg.actor_macro_backbone_weight == pytest.approx(1.0)
        assert cfg.w_actor_old_policy_chart_kl == pytest.approx(0.01)
        assert cfg.w_actor_old_policy_code_kl == pytest.approx(0.01)
        assert cfg.collect_every == 1
        assert cfg.collect_n_env_workers == 4
        assert cfg.eval_every == 10
        assert cfg.checkpoint_every == 25
        assert cfg.sigma_motor == pytest.approx(0.2)
        assert cfg.sigma_motor_init == pytest.approx(0.5)
        assert cfg.sigma_motor_anneal_epochs == 60
        assert cfg.sigma_motor_exact_gate_target == pytest.approx(0.45)
        assert cfg.chart_usage_h_low is not None
        assert cfg.chart_usage_h_high is not None
        assert "num_charts" in changes

    def test_apply_cartpole_task_preset_respects_explicit_user_overrides(self):
        from fragile.rl.config import DreamerConfig
        from fragile.rl.train_dreamer import _apply_task_preset

        cfg = DreamerConfig(
            domain="cartpole",
            task="swingup",
            latent_dim=12,
            num_charts=3,
            num_action_charts=3,
            hard_routing_warmup_epochs=7,
            w_entropy=0.2,
        )
        preset_name, _changes = _apply_task_preset(cfg)

        assert preset_name == "cartpole_swingup"
        assert cfg.latent_dim == 12
        assert cfg.num_charts == 3
        assert cfg.num_action_charts == 3
        assert cfg.hard_routing_warmup_epochs == 7
        assert cfg.w_entropy == pytest.approx(0.2)
        assert cfg.codes_per_chart == 16
        assert cfg.chart_usage_h_low is not None

    def test_apply_cartpole_balance_task_preset_uses_small_control_defaults(self):
        from fragile.rl.config import DreamerConfig
        from fragile.rl.train_dreamer import _apply_task_preset

        cfg = DreamerConfig(domain="cartpole", task="balance")
        preset_name, _changes = _apply_task_preset(cfg)

        assert preset_name == "cartpole_balance"
        assert cfg.latent_dim == 8
        assert cfg.num_charts == 4
        assert cfg.w_critic_exact_increment == pytest.approx(1.0)
        assert cfg.critic_multistep_horizon == 4
        assert cfg.w_critic_on_policy_covector_align == pytest.approx(2.0)
        assert cfg.w_critic_on_policy_stiffness == pytest.approx(1.0)
        assert cfg.critic_on_policy_horizon == 4
        assert cfg.critic_on_policy_batch_size == 4
        assert cfg.w_macro_value == pytest.approx(0.25)
        assert cfg.w_macro_exact_increment == pytest.approx(0.5)
        assert cfg.w_macro_pullback == pytest.approx(0.25)
        assert cfg.w_macro_covector_pullback == pytest.approx(0.1)
        assert cfg.w_macro_on_policy_pullback == pytest.approx(0.1)
        assert cfg.w_macro_on_policy_covector_pullback == pytest.approx(0.05)
        assert cfg.macro_multistep_horizon == 4
        assert cfg.macro_on_policy_horizon == 4
        assert cfg.actor_macro_backbone_weight == pytest.approx(0.25)
        assert cfg.sigma_motor == pytest.approx(0.1)
        assert cfg.sigma_motor_init == pytest.approx(0.15)
        assert cfg.sigma_motor_anneal_epochs == 20
        assert cfg.sigma_motor_exact_gate_target == pytest.approx(0.35)
        assert cfg.w_actor_curiosity == pytest.approx(0.05)

    def test_parse_args_uses_current_cli_schema(self, monkeypatch):
        from fragile.rl.train_dreamer import _parse_args

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--action_dim",
                "3",
                "--latent_dim",
                "5",
                "--num_charts",
                "6",
            ],
        )
        cfg = _parse_args()
        assert cfg.action_dim == 3
        assert cfg.latent_dim == 5
        assert cfg.num_charts == 6

    def test_infer_action_dim_supports_dm_control_and_vectorized_specs(self):
        from fragile.rl.train_dreamer import _infer_action_dim

        class FakeDmEnv:
            def action_spec(self):
                return SimpleNamespace(shape=(3,))

        class FakeVectorizedEnv:
            action_space = SimpleNamespace(shape=(2, 2))

        assert _infer_action_dim(FakeDmEnv()) == 3
        assert _infer_action_dim(FakeVectorizedEnv()) == 4

    def test_train_overrides_action_dim_from_environment(self, monkeypatch, tmp_path):
        from fragile.rl import train_dreamer
        from fragile.rl.config import DreamerConfig

        cfg = DreamerConfig(
            device="cpu",
            obs_dim=OBS_DIM,
            action_dim=6,
            latent_dim=D,
            num_charts=K,
            num_action_charts=K,
            d_model=D_MODEL,
            hidden_dim=D_MODEL,
            codes_per_chart=CODES_PER_CHART,
            action_codes_per_chart=CODES_PER_CHART,
            checkpoint_dir=str(tmp_path / "ckpt"),
            collect_n_env_workers=1,
            normalize_observations=False,
        )

        class FakeEnv:
            def reset(self):
                return SimpleNamespace(observation={"obs": np.zeros(OBS_DIM, dtype=np.float32)})

            def action_spec(self):
                return FakeActionSpec(action_dim=3)

        created_input_dims: list[int] = []

        class StopAfterActionModelError(RuntimeError):
            pass

        class FakeTopoEncoder:
            def __init__(self, *, input_dim, **kwargs):
                del kwargs
                created_input_dims.append(input_dim)
                self.encoder = SimpleNamespace()
                self.decoder = SimpleNamespace()
                if len(created_input_dims) == 2:
                    raise StopAfterActionModelError()

            def to(self, _device):
                return self

        class FakeJumpOp:
            def __init__(self, **kwargs):
                del kwargs

            def to(self, _device):
                return self

        monkeypatch.setattr(train_dreamer, "_make_env", lambda _domain, _task: FakeEnv())
        monkeypatch.setattr(train_dreamer, "SharedDynTopoEncoder", FakeTopoEncoder)
        monkeypatch.setattr(train_dreamer, "FactorizedJumpOperator", FakeJumpOp)

        with pytest.raises(StopAfterActionModelError):
            train_dreamer.train(cfg)

        assert cfg.action_dim == 3
        assert created_input_dims == [OBS_DIM, 3]


class TestActorStateMetric:
    def test_scale_certification_uses_scaled_fisher_term(self, device, config, world_model):
        from fragile.rl import train_dreamer

        batch = 2
        obs_z_n = torch.zeros(batch, D, device=device, requires_grad=True)
        zeros = torch.zeros(batch, device=device)
        chart_logits = torch.stack(
            [obs_z_n[:, 0], -obs_z_n[:, 0], zeros, zeros],
            dim=-1,
        )
        code_base = torch.stack(
            [obs_z_n[:, 1], -obs_z_n[:, 1], zeros, zeros],
            dim=-1,
        )
        actor_out = {
            "action_chart_logits": chart_logits,
            "action_code_logits": code_base.unsqueeze(1).expand(-1, K, -1),
        }
        state_z_geo = torch.zeros(batch, D, device=device)
        target_chart_idx = torch.zeros(batch, device=device, dtype=torch.long)
        target_code_idx = torch.zeros(batch, device=device, dtype=torch.long)
        exact_covector = torch.full((batch, D), 1e-3, device=device)

        cfg_zero = copy.deepcopy(config)
        cfg_zero.actor_metric_fisher_scale = 0.0
        _, _, scale_cert_zero, _scale_trust_zero, _scale_barrier_zero, metrics_zero = (
            train_dreamer._actor_state_metric(
                cfg_zero,
                metric=world_model.metric,
                state_z_geo=state_z_geo,
                actor_out=actor_out,
                obs_z_n=obs_z_n,
                target_chart_idx=target_chart_idx,
                target_code_idx=target_code_idx,
                exact_covector=exact_covector,
            )
        )

        cfg_full = copy.deepcopy(config)
        cfg_full.actor_metric_fisher_scale = 1.0
        _, _, scale_cert_full, _scale_trust_full, _scale_barrier_full, metrics_full = (
            train_dreamer._actor_state_metric(
                cfg_full,
                metric=world_model.metric,
                state_z_geo=state_z_geo,
                actor_out=actor_out,
                obs_z_n=obs_z_n,
                target_chart_idx=target_chart_idx,
                target_code_idx=target_code_idx,
                exact_covector=exact_covector,
            )
        )

        assert scale_cert_zero
        assert metrics_zero["actor/state_beta_pi"] == pytest.approx(0.0)
        assert metrics_zero["actor/state_beta_pi_raw"] > 0.0
        assert not scale_cert_full
        assert metrics_full["actor/state_beta_pi"] == pytest.approx(
            metrics_full["actor/state_beta_pi_raw"],
        )


class TestActorBootstrapAndTrustRegion:
    def test_categorical_entropy_varentropy_filters_white_noise(self, device):
        from fragile.rl import train_dreamer

        uniform_logits = torch.zeros(2, 4, device=device)
        structured_logits = torch.tensor(
            [[12.0, -12.0, -12.0, -12.0], [3.0, 3.0, -3.0, -3.0]],
            device=device,
        )

        entropy_uniform, varentropy_uniform = train_dreamer._categorical_entropy_varentropy(
            uniform_logits,
        )
        entropy_structured, varentropy_structured = train_dreamer._categorical_entropy_varentropy(
            structured_logits,
        )

        assert float(entropy_uniform.mean()) > float(entropy_structured.mean())
        assert float(varentropy_uniform.mean()) == pytest.approx(0.0, abs=1e-7)
        assert float(varentropy_structured[0]) < 1e-6
        assert float(varentropy_structured[1]) > 0.0

    def test_curiosity_closure_gate_prefers_grounded_dynamics(self, config):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        template = torch.tensor(0.0)
        gate_good, metrics_good = train_dreamer._actor_curiosity_closure_gate(
            cfg,
            obs_state_acc=0.9,
            enclosure_defect_acc=0.0,
            enclosure_defect_ce=0.0,
            template=template,
        )
        gate_bad, metrics_bad = train_dreamer._actor_curiosity_closure_gate(
            cfg,
            obs_state_acc=0.2,
            enclosure_defect_acc=0.5,
            enclosure_defect_ce=2.0,
            template=template,
        )

        assert float(gate_good) > float(gate_bad)
        assert metrics_good["actor/curiosity_closure_obs_factor"] > 0.0
        assert metrics_bad["actor/curiosity_closure_defect_ce_factor"] < 1.0

    def test_exact_control_gate_favors_better_exact_calibration(self, config):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        template = torch.tensor(0.0)
        gate_good, metrics_good = train_dreamer._exact_control_gate(
            cfg,
            exact_increment_abs_err=0.1,
            exact_increment_target_mean=1.0,
            on_policy_covector_align_abs_err=0.1,
            on_policy_covector_target_mean=1.0,
            on_policy_exact_covector_norm_mean=0.02,
            on_policy_stiffness_target=0.01,
            template=template,
        )
        gate_bad, metrics_bad = train_dreamer._exact_control_gate(
            cfg,
            exact_increment_abs_err=1.0,
            exact_increment_target_mean=1.0,
            on_policy_covector_align_abs_err=1.0,
            on_policy_covector_target_mean=1.0,
            on_policy_exact_covector_norm_mean=0.001,
            on_policy_stiffness_target=0.01,
            template=template,
        )

        assert float(gate_good) > float(gate_bad)
        assert metrics_good["actor/exact_control_calibration_ratio"] > 1.0
        assert metrics_bad["actor/exact_control_calibration_ratio"] < 0.2

    def test_macro_control_gate_uses_macro_increment_signal(self, config):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        cfg.w_macro_value = 1.0
        template = torch.tensor(0.0)
        gate_good, metrics_good = train_dreamer._macro_control_gate(
            cfg,
            macro_exact_increment_abs_err=0.1,
            macro_exact_increment_target_mean=1.0,
            macro_on_policy_pullback_abs_err=0.1,
            macro_on_policy_value_std=0.01,
            macro_on_policy_exact_increment_pred_abs_mean=0.8,
            macro_target_scale=1.0,
            template=template,
        )
        gate_bad, metrics_bad = train_dreamer._macro_control_gate(
            cfg,
            macro_exact_increment_abs_err=1.0,
            macro_exact_increment_target_mean=1.0,
            macro_on_policy_pullback_abs_err=1.0,
            macro_on_policy_value_std=0.01,
            macro_on_policy_exact_increment_pred_abs_mean=0.02,
            macro_target_scale=1.0,
            template=template,
        )

        assert float(gate_good) > float(gate_bad)
        assert metrics_good["actor/macro_control_signal_scale"] == pytest.approx(0.8)
        assert metrics_bad["actor/macro_control_calibration_ratio"] < 0.1

    def test_macro_transition_sharpening_gate_tracks_closure_quality(self, config):
        from fragile.rl import train_dreamer

        template = torch.tensor(0.0)
        gate_good, metrics_good = train_dreamer._macro_transition_sharpening_gate(
            config,
            obs_state_acc=0.95,
            enclosure_defect_acc=0.01,
            enclosure_defect_ce=0.01,
            template=template,
        )
        gate_bad, metrics_bad = train_dreamer._macro_transition_sharpening_gate(
            config,
            obs_state_acc=0.2,
            enclosure_defect_acc=0.5,
            enclosure_defect_ce=1.0,
            template=template,
        )

        assert float(gate_good) > float(gate_bad)
        assert (
            metrics_good["macro/transition_sharpen_gate"]
            > metrics_bad["macro/transition_sharpen_gate"]
        )

    def test_critic_stage_scales_delay_shaping_terms(self, config):
        from fragile.rl import train_dreamer

        config.screened_poisson_warmup_epochs = 10
        config.critic_covector_warmup_epochs = 8
        config.critic_stiffness_warmup_epochs = 12
        config.critic_macro_pullback_warmup_epochs = 6
        config.critic_on_policy_warmup_epochs = 4

        early = train_dreamer._critic_stage_scales(config, epoch=0)
        late = train_dreamer._critic_stage_scales(config, epoch=11)

        assert early["exact_increment"] == pytest.approx(1.0)
        assert early["poisson"] < late["poisson"]
        assert early["covector"] < late["covector"]
        assert early["stiffness"] < late["stiffness"]
        assert early["macro_pullback"] < late["macro_pullback"]
        assert early["on_policy"] < late["on_policy"]
        assert late["poisson"] == pytest.approx(1.0)
        assert late["covector"] == pytest.approx(1.0)
        assert late["stiffness"] == pytest.approx(1.0)
        assert late["macro_pullback"] == pytest.approx(1.0)
        assert late["on_policy"] == pytest.approx(1.0)

    def test_transition_backed_macro_q_uses_next_state_distribution(self, macro_critic):
        state_probs = torch.zeros(1, macro_critic.num_states, dtype=torch.float32)
        action_probs = torch.zeros(1, macro_critic.num_actions, dtype=torch.float32)
        next_state_probs = torch.zeros(1, macro_critic.num_states, dtype=torch.float32)
        next_action_probs = torch.zeros(1, macro_critic.num_actions, dtype=torch.float32)
        state_probs[0, 0] = 0.25
        state_probs[0, 5] = 0.75
        action_probs[0, 0] = 0.4
        action_probs[0, 5] = 0.6
        next_state_probs[0, 0] = 0.1
        next_state_probs[0, 1] = 0.2
        next_state_probs[0, 4] = 0.3
        next_state_probs[0, 5] = 0.4
        next_action_probs[0, 0] = 0.25
        next_action_probs[0, 5] = 0.75

        with torch.no_grad():
            macro_critic.state_action_q.weight.zero_()
            q_table = torch.zeros(macro_critic.num_states, macro_critic.num_actions)
            q_table[0, 0] = 1.0
            q_table[0, 5] = 1.5
            q_table[1, 0] = 2.0
            q_table[1, 5] = 2.5
            q_table[4, 0] = 3.0
            q_table[4, 5] = 3.5
            q_table[5, 0] = 4.0
            q_table[5, 5] = 4.5
            macro_critic.state_action_q.weight[:, 0] = q_table.reshape(-1)
            reward_table = torch.zeros(macro_critic.num_states, macro_critic.num_actions)
            reward_table[0, 0] = 0.5
            reward_table[0, 5] = 0.7
            reward_table[5, 0] = 1.1
            reward_table[5, 5] = 1.3
            macro_critic.state_action_reward.weight[:, 0] = reward_table.reshape(-1)

        reward = macro_critic.reward_from_probs(state_probs, action_probs)
        q_value = macro_critic.q_from_transition(
            state_probs,
            action_probs,
            next_state_probs,
            next_action_probs=next_action_probs,
            gamma=0.5,
        )
        expected_reward = 0.25 * (0.4 * 0.5 + 0.6 * 0.7) + 0.75 * (0.4 * 1.1 + 0.6 * 1.3)
        expected_next_value = (
            0.1 * (0.25 * 1.0 + 0.75 * 1.5)
            + 0.2 * (0.25 * 2.0 + 0.75 * 2.5)
            + 0.3 * (0.25 * 3.0 + 0.75 * 3.5)
            + 0.4 * (0.25 * 4.0 + 0.75 * 4.5)
        )

        assert reward.item() == pytest.approx(expected_reward)
        assert q_value.item() == pytest.approx(expected_reward + 0.5 * expected_next_value)

    def test_macro_transition_observability_metrics_report_bellman_structure(self, macro_critic):
        from fragile.rl import train_dreamer

        with torch.no_grad():
            macro_critic.state_action_q.weight.zero_()
            q_table = torch.zeros(macro_critic.num_states, macro_critic.num_actions)
            q_table[0, 0] = 1.0
            q_table[5, 0] = -0.5
            q_table[1, 0] = 0.25
            q_table[9, 0] = -0.25
            macro_critic.state_action_q.weight[:, 0] = q_table.reshape(-1)

        metrics = train_dreamer._macro_transition_observability_metrics(
            macro_critic=macro_critic,
            state_idx=torch.tensor([[0, 5]], dtype=torch.long),
            next_state_probs=torch.nn.functional.one_hot(
                torch.tensor([[0, 5]], dtype=torch.long),
                num_classes=macro_critic.num_states,
            ).to(torch.float32)
            * torch.tensor([[[0.8], [0.4]]], dtype=torch.float32)
            + torch.nn.functional.one_hot(
                torch.tensor([[1, 1]], dtype=torch.long),
                num_classes=macro_critic.num_states,
            ).to(torch.float32)
            * torch.tensor([[[0.2], [0.3]]], dtype=torch.float32)
            + torch.nn.functional.one_hot(
                torch.tensor([[0, 8]], dtype=torch.long),
                num_classes=macro_critic.num_states,
            ).to(torch.float32)
            * torch.tensor([[[0.0], [0.2]]], dtype=torch.float32)
            + torch.nn.functional.one_hot(
                torch.tensor([[0, 9]], dtype=torch.long),
                num_classes=macro_critic.num_states,
            ).to(torch.float32)
            * torch.tensor([[[0.0], [0.1]]], dtype=torch.float32),
            reward_pred=torch.tensor([[0.9, -0.1]], dtype=torch.float32),
            reward_target=torch.tensor([[1.1, -0.2]], dtype=torch.float32),
            next_value=torch.tensor([[0.7, -0.4]], dtype=torch.float32),
            bootstrap_term=torch.tensor([[0.35, -0.2]], dtype=torch.float32),
            valid_mask=torch.ones(1, 2, dtype=torch.float32),
            metric_prefix="macro",
        )

        assert metrics["macro/reward_pred_mean"] == pytest.approx(0.4)
        assert metrics["macro/reward_target_mean"] == pytest.approx(0.45)
        assert metrics["macro/reward_abs_err"] == pytest.approx(0.15)
        assert metrics["macro/value_next_mean"] == pytest.approx(0.15)
        assert metrics["macro/bootstrap_term_mean"] == pytest.approx(0.075)
        assert metrics["macro/next_state_top1_prob"] == pytest.approx(0.6)
        assert metrics["macro/self_transition_prob"] == pytest.approx(0.6)
        assert metrics["macro/next_state_positive_value_mass"] == pytest.approx(0.65)

    def test_scheduled_sigma_motor_keeps_exploration_high_until_exact_gate_improves(self, config):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        cfg.sigma_motor = 0.1
        cfg.sigma_motor_init = 0.4
        cfg.sigma_motor_anneal_epochs = 10
        cfg.sigma_motor_exact_gate_target = 0.5

        sigma_early, _metrics_early = train_dreamer._scheduled_sigma_motor(
            cfg,
            epoch=0,
            exact_control_gate=0.0,
        )
        sigma_mid, _metrics_mid = train_dreamer._scheduled_sigma_motor(
            cfg,
            epoch=8,
            exact_control_gate=0.1,
        )
        sigma_ready, metrics_ready = train_dreamer._scheduled_sigma_motor(
            cfg,
            epoch=8,
            exact_control_gate=0.8,
        )

        assert sigma_early == pytest.approx(0.4)
        assert sigma_mid > sigma_ready
        assert metrics_ready["policy/sigma_motor_exact_progress"] == pytest.approx(1.0)
        assert sigma_ready < sigma_early

    def test_actor_supervise_scale_decays_with_epoch_and_return_gate(self, config):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        cfg.actor_supervise_warmup_epochs = 2
        cfg.actor_supervise_decay_epochs = 4
        cfg.actor_supervise_min_scale = 0.1

        scale_early, metrics_early = train_dreamer._actor_supervise_scale(
            cfg,
            epoch=0,
            actor_return_gate=torch.tensor(0.0),
        )
        scale_mid, _metrics_mid = train_dreamer._actor_supervise_scale(
            cfg,
            epoch=3,
            actor_return_gate=torch.tensor(0.0),
        )
        scale_gate, metrics_gate = train_dreamer._actor_supervise_scale(
            cfg,
            epoch=3,
            actor_return_gate=torch.tensor(0.8),
        )
        scale_floor, metrics_floor = train_dreamer._actor_supervise_scale(
            cfg,
            epoch=20,
            actor_return_gate=torch.tensor(1.0),
        )

        assert float(scale_early) == pytest.approx(1.0)
        assert metrics_early["actor/supervise_scale"] == pytest.approx(1.0)
        assert 0.1 < float(scale_mid) < 1.0
        assert float(scale_gate) < float(scale_mid)
        assert metrics_gate["actor/supervise_gate_scale"] == pytest.approx(0.2)
        assert float(scale_floor) == pytest.approx(0.1)
        assert metrics_floor["actor/supervise_scale"] == pytest.approx(0.1)

    def test_old_policy_kl_losses_vanish_when_logits_match(self, device):
        from fragile.rl import train_dreamer

        chart_logits = torch.tensor(
            [[1.0, -1.0, 0.0, 0.5], [0.3, -0.2, 0.4, -0.1]],
            device=device,
        )
        code_logits = torch.randn(2, K, CODES_PER_CHART, device=device)
        actor_out = {
            "action_chart_logits": chart_logits.clone(),
            "action_code_logits": code_logits.clone(),
        }
        actor_old_out = {
            "action_chart_logits": chart_logits.clone(),
            "action_code_logits": code_logits.clone(),
        }

        chart_kl, code_kl = train_dreamer._actor_old_policy_kl_losses(actor_out, actor_old_out)

        assert float(chart_kl) == pytest.approx(0.0, abs=1e-7)
        assert float(code_kl) == pytest.approx(0.0, abs=1e-7)

    def test_old_policy_kl_losses_increase_when_logits_change(self, device):
        from fragile.rl import train_dreamer

        actor_old_out = {
            "action_chart_logits": torch.tensor(
                [[1.0, -1.0, 0.0, 0.5], [0.3, -0.2, 0.4, -0.1]],
                device=device,
            ),
            "action_code_logits": torch.randn(2, K, CODES_PER_CHART, device=device),
        }
        actor_out = {
            "action_chart_logits": actor_old_out["action_chart_logits"] * -1.5,
            "action_code_logits": actor_old_out["action_code_logits"] * -0.5,
        }

        chart_kl, code_kl = train_dreamer._actor_old_policy_kl_losses(actor_out, actor_old_out)

        assert float(chart_kl) > 0.0
        assert float(code_kl) > 0.0


class TestCriticStiffness:
    def test_critic_stiffness_loss_penalizes_flat_exact_field(self, config, world_model):
        from fragile.rl import train_dreamer

        z = torch.zeros(4, D, dtype=torch.float32)
        exact_covector = torch.full((4, D), 1e-4, dtype=torch.float32, requires_grad=True)
        replay_valid = torch.ones(2, 2, dtype=torch.float32)
        config.critic_stiffness_min = 0.01

        loss, metrics = train_dreamer._critic_stiffness_loss(
            config,
            metric=world_model.metric,
            z=z,
            exact_covector=exact_covector,
            replay_valid=replay_valid,
        )

        assert loss.item() > 0.0
        assert metrics["critic/exact_covector_norm_mean"] < config.critic_stiffness_min
        assert metrics["critic/stiffness_certified"] == 0.0

    def test_critic_stiffness_loss_uses_adaptive_target(self, config, world_model):
        from fragile.rl import train_dreamer

        z = torch.zeros(4, D, dtype=torch.float32)
        exact_covector = torch.full((4, D), 5e-4, dtype=torch.float32, requires_grad=True)
        replay_valid = torch.ones(2, 2, dtype=torch.float32)

        loss, metrics = train_dreamer._critic_stiffness_loss(
            config,
            metric=world_model.metric,
            z=z,
            exact_covector=exact_covector,
            replay_valid=replay_valid,
            stiffness_scale=torch.tensor(0.02),
        )

        assert loss.item() > 0.0
        assert metrics["critic/stiffness_target"] == pytest.approx(0.02)

    def test_critic_covector_alignment_loss_matches_constructed_transition(
        self,
        config,
        world_model,
        device,
    ):

        z = torch.zeros(4, D, device=device)
        displacement = torch.zeros(4, D, device=device)
        displacement[:, 0] = 0.05
        z_next = poincare_exp_map(z, displacement)
        exact_covector = torch.zeros(4, D, device=device)
        exact_covector[:, 0] = -2.0
        exact_covector.requires_grad_()
        value_current = torch.zeros(4, 1, device=device)
        reward_target = torch.full((2, 2), 0.1, device=device)
        replay_valid = torch.ones(2, 2, device=device)
        continuation = torch.ones(2, 2, 1, device=device)

        loss, stiffness_scale, metrics = train_dreamer._critic_covector_alignment_loss(
            config,
            metric=world_model.metric,
            z=z,
            z_next=z_next,
            value_current=value_current,
            exact_covector=exact_covector,
            reward_conservative_target=reward_target,
            continuation=continuation,
            gamma=config.gamma,
            replay_valid=replay_valid,
        )

        assert float(loss) < 2e-4
        assert metrics["critic/covector_align_abs_err"] < 1e-2
        assert float(stiffness_scale) >= float(config.critic_stiffness_min)

    def test_critic_exact_increment_loss_matches_discounted_target(self):
        from fragile.rl import train_dreamer

        reward_pred = torch.tensor([[0.2, 0.4], [0.1, -0.2]], dtype=torch.float32)
        reward_target = reward_pred.clone()
        replay_valid = torch.ones_like(reward_pred)

        loss, metrics = train_dreamer._critic_exact_increment_loss(
            reward_conservative_pred=reward_pred,
            reward_conservative_target=reward_target,
            replay_valid=replay_valid,
        )

        assert float(loss) == pytest.approx(0.0)
        assert metrics["critic/exact_increment_abs_err"] == pytest.approx(0.0)

    def test_critic_exact_increment_loss_reports_observability_stats(self):
        from fragile.rl import train_dreamer

        reward_pred = torch.tensor([[1.0, -0.5, 2.0]], dtype=torch.float32)
        reward_target = reward_pred.clone()
        replay_valid = torch.ones_like(reward_pred)

        _, metrics = train_dreamer._critic_exact_increment_loss(
            reward_conservative_pred=reward_pred,
            reward_conservative_target=reward_target,
            replay_valid=replay_valid,
        )

        assert metrics["critic/exact_increment_pred_std"] == pytest.approx(
            float(reward_pred.reshape(-1).std(unbiased=False)),
        )
        assert metrics["critic/exact_increment_target_std"] == pytest.approx(
            float(reward_target.reshape(-1).std(unbiased=False)),
        )
        assert metrics["critic/exact_increment_sign_acc"] == pytest.approx(1.0)
        assert metrics["critic/exact_increment_corr"] == pytest.approx(1.0)
        assert metrics["critic/exact_increment_support_frac"] == pytest.approx(1.0)
        assert metrics["critic/exact_increment_positive_frac"] == pytest.approx(2.0 / 3.0)

    def test_multistep_exact_increment_loss_matches_two_step_value_telescoping(self):
        from fragile.rl import train_dreamer

        value_seq = torch.tensor([[1.0, 0.8, 0.5]], dtype=torch.float32)
        reward_target = torch.tensor([[0.6, 0.55]], dtype=torch.float32)
        continuation = torch.ones_like(reward_target)
        valid = torch.ones_like(reward_target)

        loss, metrics = train_dreamer._multistep_exact_increment_loss(
            value_seq=value_seq,
            reward_conservative_targets=reward_target,
            continuation=continuation,
            valid_mask=valid,
            gamma=0.5,
            horizon=2,
            decay=1.0,
            metric_prefix="critic",
        )

        assert float(loss) == pytest.approx(0.0, abs=1e-7)
        assert metrics["critic/exact_increment_horizon_used"] == pytest.approx(2.0)

    def test_multistep_covector_alignment_loss_matches_two_step_transition(
        self, config, world_model
    ):
        from fragile.rl import train_dreamer

        cfg = copy.deepcopy(config)
        cfg.gamma = 1.0
        z_seq = torch.zeros(1, 3, D, dtype=torch.float32)
        z_seq[:, 1, 0] = 0.05
        z_seq[:, 2, 0] = 0.10
        value_seq = torch.tensor([[0.0, -0.1, -0.2]], dtype=torch.float32)
        exact_covector_seq = torch.zeros(1, 2, D, dtype=torch.float32)
        exact_covector_seq[:, :, 0] = -2.0
        exact_covector_seq.requires_grad_()
        reward_target = torch.tensor([[0.1, 0.1]], dtype=torch.float32)
        continuation = torch.ones_like(reward_target)
        valid = torch.ones_like(reward_target)

        loss, stiffness_scale, metrics = train_dreamer._multistep_covector_alignment_loss(
            cfg,
            metric=world_model.metric,
            z_seq=z_seq,
            value_seq=value_seq,
            exact_covector_seq=exact_covector_seq,
            reward_conservative_targets=reward_target,
            continuation=continuation,
            valid_mask=valid,
            gamma=1.0,
            horizon=2,
            decay=1.0,
            metric_prefix="critic",
        )

        assert float(loss) < 1e-4
        assert metrics["critic/covector_horizon_used"] == pytest.approx(2.0)
        assert float(stiffness_scale) >= float(cfg.critic_stiffness_min)


class TestAtlasSync:
    def test_sync_rl_atlas_binds_obs_and_action_atlases(
        self,
        obs_model,
        action_model,
        world_model,
        standalone_critic,
        actor,
        reward_head,
    ):

        with torch.no_grad():
            obs_model.encoder.chart_centers.copy_(
                torch.randn_like(obs_model.encoder.chart_centers) * 0.4
            )
            action_model.encoder.chart_centers.copy_(
                torch.randn_like(action_model.encoder.chart_centers) * 0.4
            )
            world_model.potential_net.chart_tok.chart_centers.zero_()
            standalone_critic.chart_tok.chart_centers.zero_()
            actor.action_chart_centers.zero_()
            reward_head.action_chart_tok.chart_centers.zero_()

        _sync_rl_atlas(
            obs_model,
            action_model,
            world_model,
            standalone_critic,
            actor,
            reward_head,
        )

        expected_obs = project_to_ball(obs_model.encoder.chart_centers.detach())
        expected_action = project_to_ball(action_model.encoder.chart_centers.detach())
        torch.testing.assert_close(
            project_to_ball(world_model.potential_net.chart_tok.chart_centers.detach()),
            expected_obs,
        )
        torch.testing.assert_close(
            project_to_ball(standalone_critic.chart_tok.chart_centers.detach()),
            expected_obs,
        )
        torch.testing.assert_close(
            project_to_ball(actor.action_chart_centers.detach()),
            expected_action,
        )
        torch.testing.assert_close(
            project_to_ball(reward_head.action_chart_tok.chart_centers.detach()),
            expected_action,
        )
        assert not actor.action_chart_centers.requires_grad
        assert not reward_head.action_chart_tok.chart_centers.requires_grad


class TestRolloutCollection:
    def test_collect_episode_without_policy_zeros_canonical_action_state(self):
        from fragile.rl.train_dreamer import _collect_episode

        env = SingleEpisodeEnv(start_obs=[0.1] * OBS_DIM, next_obs=[0.2] * OBS_DIM)
        episode = _collect_episode(
            env,
            None,
            None,
            SimpleNamespace(),
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            action_repeat=1,
            max_steps=1,
        )
        assert episode["obs"].shape == (2, OBS_DIM)
        assert episode["action_latents"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert np.allclose(episode["action_latents"], 0.0)

    def test_collect_episode_normalizes_obs_and_uses_current_schema(self, monkeypatch):
        from fragile.rl import train_dreamer

        env = SingleEpisodeEnv(start_obs=[3.0] * OBS_DIM, next_obs=[4.0] * OBS_DIM)
        model = FakeEncoderModel()
        normalizer = train_dreamer.ObservationNormalizer(
            mean=torch.ones(OBS_DIM),
            std=torch.full((OBS_DIM,), 2.0),
            min_std=1e-3,
        )

        def fake_policy_action(
            _actor,
            _action_model,
            obs_info,
            **_kwargs,
        ):
            return _make_constant_policy_output(obs_info["z_geo"])

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        episode = train_dreamer._collect_episode(
            env,
            object(),
            object(),
            model,
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            obs_normalizer=normalizer,
            action_repeat=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert model.encoder.calls == [(False, 0.7)]
        torch.testing.assert_close(model.encoder.obs_seen[0], torch.ones(1, OBS_DIM))
        assert episode["action_latents"].shape == (2, D)
        assert episode["action_router_weights"].shape == (2, K)
        assert episode["action_charts"][0] == 0
        assert episode["action_codes"][0] == 1
        assert episode["action_latents"][0, 0] == pytest.approx(0.15)

    def test_collect_parallel_episodes_returns_replay_compatible_schema(self, monkeypatch):
        from fragile.rl import train_dreamer

        env = ParallelEnv()
        model = FakeEncoderModel()

        def fake_policy_action(
            _actor,
            _action_model,
            obs_info,
            **_kwargs,
        ):
            return _make_constant_policy_output(obs_info["z_geo"])

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        episodes = train_dreamer._collect_parallel_episodes(
            env,
            object(),
            object(),
            model,
            torch.device("cpu"),
            control_dim=D,
            num_action_charts=K,
            num_episodes=2,
            action_repeat=1,
            max_steps=2,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert len(episodes) == 2
        assert episodes[0]["obs"].shape == (2, OBS_DIM)
        assert episodes[1]["obs"].shape == (3, OBS_DIM)
        assert episodes[0]["action_latents"].shape == (2, D)
        assert episodes[1]["action_router_weights"].shape == (3, K)
        assert episodes[1]["action_code_latents"].shape == (3, D)
        assert episodes[1]["dones"][-1] == pytest.approx(1.0)
        assert env.batch_calls == [[0, 1], [1]]

    def test_eval_policy_uses_deterministic_mean_actions(self, monkeypatch):
        from fragile.rl import train_dreamer

        env = SingleEpisodeEnv(start_obs=[0.1] * OBS_DIM, next_obs=[0.2] * OBS_DIM)
        model = FakeEncoderModel()

        def fake_policy_action(
            _actor,
            _action_model,
            obs_info,
            **_kwargs,
        ):
            return _make_constant_policy_output(obs_info["z_geo"], action_value=0.44)

        monkeypatch.setattr(train_dreamer, "_policy_action", fake_policy_action)
        metrics = train_dreamer._eval_policy(
            env,
            object(),
            object(),
            model,
            torch.device("cpu"),
            action_repeat=1,
            num_episodes=1,
            max_steps=1,
            hard_routing=True,
            hard_routing_tau=0.7,
        )

        assert model.encoder.calls == [(False, 0.7)]
        assert metrics["eval/reward_mean"] == pytest.approx(1.0)


class TestImagination:
    def test_outputs_current_shapes_and_exact_split(
        self,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
        rw,
    ):
        from fragile.rl.train_dreamer import _imagine, _sync_rl_atlas

        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        out = _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            z,
            rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
            reward_curl_batch_limit=2,
        )
        assert out["z_states"].shape == (B, H_IMAGINATION, D)
        assert out["rw_states"].shape == (B, H_IMAGINATION, K)
        assert out["z_traj"].shape == (B, H_IMAGINATION, D)
        assert out["rw_traj"].shape == (B, H_IMAGINATION, K)
        assert out["action_canonicals"].shape == (B, H_IMAGINATION, D)
        assert out["action_latents"].shape == (B, H_IMAGINATION, D)
        assert out["action_router_weights"].shape == (B, H_IMAGINATION, K)
        assert out["actions"].shape == (B, H_IMAGINATION, A)
        assert out["reward_conservative"].shape == (B, H_IMAGINATION)
        assert out["reward_nonconservative"].shape == (B, H_IMAGINATION)
        assert out["reward_curl_norm"].shape == (B, H_IMAGINATION)
        assert out["reward_curl_valid"].shape == (B, H_IMAGINATION)
        assert out["phi_eff"].shape == (B, H_IMAGINATION, 1)
        assert not out["actions"].requires_grad
        assert not out["z_traj"].requires_grad
        torch.testing.assert_close(
            out["rewards"],
            out["reward_conservative"] + out["reward_nonconservative"],
        )

    def test_no_jump_world_model_keeps_router_weights(
        self,
        monkeypatch,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
        rw,
    ):
        from fragile.rl.train_dreamer import _imagine, _sync_rl_atlas

        def bad_chart_logits(self, z_in, control_in, rw_in):
            del z_in, control_in, rw_in
            logits = torch.full((B, K), -50.0)
            logits[:, -1] = 50.0
            return logits

        monkeypatch.setattr(
            world_model.chart_predictor,
            "forward",
            bad_chart_logits.__get__(
                world_model.chart_predictor, type(world_model.chart_predictor)
            ),
        )
        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        out = _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            z,
            rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
        )
        expected = rw.unsqueeze(1).expand_as(out["rw_traj"])
        torch.testing.assert_close(out["rw_traj"], expected)

    def test_imagination_uses_world_model_chart_state_for_actor_inputs(
        self,
        monkeypatch,
        obs_model,
        action_model,
        world_model,
        critic,
        actor,
        reward_head,
        z,
    ):
        from fragile.rl.action_manifold import symbolize_latent_with_atlas
        from fragile.rl.train_dreamer import _imagine, _sync_rl_atlas

        initial_info = symbolize_latent_with_atlas(
            obs_model,
            z,
            hard_routing=False,
            hard_routing_tau=1.0,
        )
        forced_chart_idx = (initial_info["chart_idx"] + 1) % K
        forced_rw = torch.nn.functional.one_hot(
            forced_chart_idx,
            num_classes=K,
        ).to(dtype=z.dtype, device=z.device)

        seen_chart_idx: list[torch.Tensor] = []
        original_forward = actor.forward

        def recording_forward(obs_chart_idx, obs_code_idx, obs_z_n, **kwargs):
            seen_chart_idx.append(obs_chart_idx.detach().clone())
            return original_forward(obs_chart_idx, obs_code_idx, obs_z_n, **kwargs)

        monkeypatch.setattr(actor, "forward", recording_forward)
        _sync_rl_atlas(obs_model, action_model, world_model, critic, actor, reward_head)
        _imagine(
            obs_model,
            world_model,
            reward_head,
            critic,
            actor,
            action_model,
            z,
            forced_rw,
            horizon=H_IMAGINATION,
            gamma=0.99,
        )

        assert seen_chart_idx
        for chart_idx in seen_chart_idx:
            torch.testing.assert_close(chart_idx, forced_chart_idx)

    def test_actor_return_optimizes_full_reward_over_canonical_actions(self, monkeypatch):
        from fragile.rl import train_dreamer

        class FakeActor(nn.Module):
            def forward(
                self,
                obs_chart_idx,
                obs_code_idx,
                obs_z_n,
                *,
                hard_routing=False,
                hard_routing_tau=1.0,
            ):
                del obs_chart_idx, obs_code_idx, hard_routing, hard_routing_tau
                batch = obs_z_n.shape[0]
                chart_logits = torch.zeros(batch, K, device=obs_z_n.device)
                chart_logits[:, 0] = 1.0
                code_logits = torch.zeros(batch, K, CODES_PER_CHART, device=obs_z_n.device)
                code_logits[:, :, 0] = 1.0
                return {
                    "action_chart_logits": chart_logits,
                    "action_chart_idx": torch.zeros(
                        batch, dtype=torch.long, device=obs_z_n.device
                    ),
                    "action_code_logits": code_logits,
                    "action_code_idx": torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                    "action_z_n": torch.full_like(obs_z_n, 0.2),
                    "action_z_q": torch.zeros_like(obs_z_n),
                    "action_z_geo": torch.full_like(obs_z_n, 0.2),
                    "action_router_weights": torch.nn.functional.one_hot(
                        torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                        num_classes=K,
                    ).to(dtype=obs_z_n.dtype),
                }

        class FakeActionModel(nn.Module):
            def decoder(
                self,
                z_geo,
                *,
                router_weights,
                routing_tau=1.0,
            ):
                del router_weights, routing_tau
                return z_geo[:, :A], None, None

        class FakeWorldModel:
            def momentum_init(self, z_0):
                return torch.zeros_like(z_0)

            def _rollout_transition(self, z, p, action_canonical, rw, track_energy=False):
                del action_canonical, track_energy
                chart_logits = torch.zeros(z.shape[0], K, device=z.device, dtype=z.dtype)
                chart_logits[:, 0] = 1.0
                return {
                    "z": z + 0.01,
                    "p": p,
                    "rw": rw,
                    "phi_eff": torch.zeros(z.shape[0], 1),
                    "chart_logits": chart_logits,
                }

        class FakeRewardHead:
            def decompose(
                self,
                z,
                rw,
                action_z,
                action_rw,
                action_code_z,
                action_canonical,
                *,
                exact_covector=None,
                compute_curl=False,
                curl_batch_limit=None,
            ):
                del (
                    z,
                    rw,
                    action_z,
                    action_rw,
                    action_code_z,
                    action_canonical,
                    exact_covector,
                    compute_curl,
                    curl_batch_limit,
                )
                return {
                    "reward_nonconservative": torch.full((2, 1), 3.0),
                    "reward_form_cov": torch.zeros(2, D),
                }

        class FakeCritic(nn.Module):
            def task_value(self, z, rw):
                del rw
                return z[:, :1]

        def fake_symbolize_latent_with_atlas(_atlas, z_in, **_kwargs):
            batch = z_in.shape[0]
            router = torch.zeros(batch, K, device=z_in.device, dtype=z_in.dtype)
            router[:, 0] = 1.0
            return {
                "z_geo": z_in,
                "router_weights": router,
                "chart_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "code_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "z_q": torch.zeros_like(z_in),
                "z_n": torch.zeros_like(z_in),
            }

        monkeypatch.setattr(
            train_dreamer, "symbolize_latent_with_atlas", fake_symbolize_latent_with_atlas
        )

        def zero_covector(_critic, z_in, _rw, **_kwargs):
            return torch.zeros_like(z_in)

        monkeypatch.setattr(train_dreamer, "_value_covector_from_critic", zero_covector)

        def constant_conservative_reward(_critic, z_curr, _rw_curr, _z_next, _rw_next, _gamma):
            reward = torch.full(
                (z_curr.shape[0], 1), 2.0, device=z_curr.device, dtype=z_curr.dtype
            )
            return reward, reward, reward

        monkeypatch.setattr(
            train_dreamer,
            "_conservative_reward_from_value",
            constant_conservative_reward,
        )

        z_0 = torch.randn(2, D) * 0.1
        rw_0 = torch.softmax(torch.randn(2, K), dim=-1)
        out = train_dreamer._imagine_actor_return(
            None,
            object(),
            FakeWorldModel(),
            FakeRewardHead(),
            FakeCritic(),
            None,
            FakeActor(),
            FakeActionModel(),
            z_0,
            rw_0,
            horizon=3,
            gamma=0.5,
        )

        expected_objective = torch.full((2,), 5.0 * (1.0 + 0.5 + 0.25))
        torch.testing.assert_close(out["objective"], expected_objective)
        assert out["rewards"].shape == (2, 3)
        assert out["reward_conservative"].shape == (2, 3)
        assert out["reward_nonconservative"].shape == (2, 3)
        assert out["reward_macro"].shape == (2, 3)
        assert out["actions"].shape == (2, 3, A)
        assert out["action_canonicals"].shape == (2, 3, D)

    def test_actor_return_rollout_respects_explicit_routing_schedule(self, monkeypatch):
        from fragile.rl import train_dreamer

        routing_calls: list[tuple[str, bool, float]] = []

        class FakeActor(nn.Module):
            def forward(
                self,
                obs_chart_idx,
                obs_code_idx,
                obs_z_n,
                *,
                hard_routing=False,
                hard_routing_tau=1.0,
            ):
                del obs_chart_idx, obs_code_idx
                routing_calls.append(("actor", hard_routing, hard_routing_tau))
                batch = obs_z_n.shape[0]
                chart_logits = torch.zeros(batch, K, device=obs_z_n.device)
                chart_logits[:, 0] = 1.0
                code_logits = torch.zeros(batch, K, CODES_PER_CHART, device=obs_z_n.device)
                code_logits[:, :, 0] = 1.0
                return {
                    "action_chart_logits": chart_logits,
                    "action_chart_idx": torch.zeros(
                        batch, dtype=torch.long, device=obs_z_n.device
                    ),
                    "action_code_logits": code_logits,
                    "action_code_idx": torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                    "action_z_n": torch.full_like(obs_z_n, 0.2),
                    "action_z_q": torch.zeros_like(obs_z_n),
                    "action_z_geo": torch.full_like(obs_z_n, 0.2),
                    "action_router_weights": torch.nn.functional.one_hot(
                        torch.zeros(batch, dtype=torch.long, device=obs_z_n.device),
                        num_classes=K,
                    ).to(dtype=obs_z_n.dtype),
                }

        class FakeActionModel(nn.Module):
            def decoder(
                self,
                z_geo,
                *,
                router_weights,
                routing_tau=1.0,
            ):
                del router_weights
                routing_calls.append(("decoder", routing_tau))
                return z_geo[:, :A], None, None

        class FakeWorldModel:
            def momentum_init(self, z_0):
                return torch.zeros_like(z_0)

            def _rollout_transition(self, z, p, action_canonical, rw, track_energy=False):
                del action_canonical, track_energy
                chart_logits = torch.zeros(z.shape[0], K, device=z.device, dtype=z.dtype)
                chart_logits[:, 0] = 1.0
                return {
                    "z": z + 0.01,
                    "p": p,
                    "rw": rw,
                    "phi_eff": torch.zeros(z.shape[0], 1),
                    "chart_logits": chart_logits,
                }

        class FakeRewardHead:
            def decompose(
                self,
                z,
                rw,
                action_z,
                action_rw,
                action_code_z,
                action_canonical,
                *,
                exact_covector=None,
                compute_curl=False,
                curl_batch_limit=None,
            ):
                del (
                    z,
                    rw,
                    action_z,
                    action_rw,
                    action_code_z,
                    action_canonical,
                    exact_covector,
                    compute_curl,
                    curl_batch_limit,
                )
                return {
                    "reward_nonconservative": torch.full((2, 1), 1.0),
                    "reward_form_cov": torch.zeros(2, D),
                }

        class FakeCritic(nn.Module):
            def task_value(self, z, rw):
                del rw
                return z[:, :1]

        def fake_symbolize_latent_with_atlas(_atlas, z_in, **kwargs):
            routing_calls.append(
                ("symbolize", kwargs["hard_routing"], kwargs["hard_routing_tau"]),
            )
            batch = z_in.shape[0]
            router = torch.zeros(batch, K, device=z_in.device, dtype=z_in.dtype)
            router[:, 0] = 1.0
            return {
                "z_geo": z_in,
                "router_weights": router,
                "chart_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "code_idx": torch.zeros(batch, dtype=torch.long, device=z_in.device),
                "z_q": torch.zeros_like(z_in),
                "z_n": torch.zeros_like(z_in),
            }

        monkeypatch.setattr(
            train_dreamer, "symbolize_latent_with_atlas", fake_symbolize_latent_with_atlas
        )
        monkeypatch.setattr(
            train_dreamer,
            "_value_covector_from_critic",
            lambda _critic, z_in, _rw, **_kwargs: torch.zeros_like(z_in),
        )
        monkeypatch.setattr(
            train_dreamer,
            "_conservative_reward_from_value",
            lambda _critic, z_curr, _rw_curr, _z_next, _rw_next, _gamma: (
                torch.ones((z_curr.shape[0], 1), device=z_curr.device, dtype=z_curr.dtype),
                torch.ones((z_curr.shape[0], 1), device=z_curr.device, dtype=z_curr.dtype),
                torch.ones((z_curr.shape[0], 1), device=z_curr.device, dtype=z_curr.dtype),
            ),
        )

        z_0 = torch.randn(2, D) * 0.1
        rw_0 = torch.softmax(torch.randn(2, K), dim=-1)
        train_dreamer._imagine_actor_return(
            None,
            object(),
            FakeWorldModel(),
            FakeRewardHead(),
            FakeCritic(),
            None,
            FakeActor(),
            FakeActionModel(),
            z_0,
            rw_0,
            horizon=2,
            gamma=0.5,
            hard_routing=False,
            hard_routing_tau=0.7,
        )

        assert routing_calls == [
            ("symbolize", False, 0.7),
            ("actor", False, 0.7),
            ("decoder", 0.7),
            ("symbolize", False, 0.7),
            ("symbolize", False, 0.7),
            ("actor", False, 0.7),
            ("decoder", 0.7),
            ("symbolize", False, 0.7),
        ]


class TestTrainStep:
    def test_updates_parameters_and_reports_reward_split_metrics(
        self,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        world_model,
        critic,
        macro_critic,
        reward_head,
        actor,
        actor_old,
        enclosure_probe,
    ):
        from fragile.rl.train_dreamer import _train_step

        batch = _make_training_batch(device)
        optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            world_model,
            macro_critic,
            reward_head,
            actor,
            enclosure_probe,
        )
        actor_before = {name: param.detach().clone() for name, param in actor.named_parameters()}
        reward_before = {
            name: param.detach().clone()
            for name, param in reward_head.named_parameters()
            if "chart_tok" not in name and "z_embed" not in name
        }

        metrics = _train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            optimizer_enclosure,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        expected_keys = {
            "enc/L_total",
            "closure/L_enclosure",
            "wm/L_reward",
            "wm/L_reward_nonconservative",
            "wm/L_reward_exact_orth",
            "wm/L_code",
            "wm/L_symbol",
            "wm/L_force_exact",
            "wm/L_reward_nonconservative_norm",
            "wm/L_reward_nonconservative_budget",
            "wm/reward_form_exact_leakage_metric_mean",
            "wm/L_hodge_conservative_margin",
            "wm/L_hodge_solenoidal",
            "wm/code_acc",
            "wm/symbol_acc",
            "macro/L_transition",
            "critic/L_critic",
            "critic/stage_poisson",
            "critic/stage_covector",
            "critic/stage_stiffness",
            "critic/stage_macro_pullback",
            "critic/stage_on_policy",
            "critic/grad_exact_increment",
            "critic/grad_poisson",
            "critic/grad_covector_align",
            "critic/grad_stiffness",
            "critic/grad_macro_pullback",
            "critic/grad_on_policy",
            "actor/L_total",
            "actor/L_return",
            "actor/L_supervise_raw",
            "actor/L_supervise",
            "actor/L_old_policy_geodesic",
            "actor/L_old_policy_chart_kl",
            "actor/L_old_policy_code_kl",
            "actor/L_natural",
            "actor/L_sync",
            "actor/L_stiffness",
            "actor/symbol_acc",
            "actor/supervise_scale",
            "actor/update_applied",
            "time/step",
        }
        assert expected_keys.issubset(metrics)
        assert metrics["actor/update_applied"] == 1.0
        assert np.isfinite([metrics[key] for key in expected_keys]).all()

        actor_changed = any(
            not torch.equal(actor_before[name], param.detach())
            for name, param in actor.named_parameters()
        )
        reward_changed = any(
            not torch.equal(reward_before[name], param.detach())
            for name, param in reward_head.named_parameters()
            if name in reward_before
        )
        assert actor_changed
        assert reward_changed

    def test_gates_imagined_return_when_trust_is_low(
        self,
        monkeypatch,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        world_model,
        critic,
        macro_critic,
        reward_head,
        actor,
        actor_old,
        enclosure_probe,
    ):
        from fragile.rl import train_dreamer

        batch = _make_training_batch(device)
        optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            world_model,
            macro_critic,
            reward_head,
            actor,
            enclosure_probe,
        )
        actor_before = {name: param.detach().clone() for name, param in actor.named_parameters()}

        def zero_trust(*_args, template, **_kwargs):
            zero = template.new_zeros(())
            return zero, {
                "actor/return_trust": 0.0,
                "actor/return_trust_chart": 0.0,
                "actor/return_trust_force": 0.0,
                "actor/return_trust_sync": 0.0,
                "actor/return_trust_conservative_exact": 0.0,
            }

        monkeypatch.setattr(train_dreamer, "_actor_return_trust", zero_trust)

        metrics = train_dreamer._train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            optimizer_enclosure,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        assert metrics["actor/update_applied"] == 1.0
        assert metrics["actor/return_applied"] == 0.0
        assert metrics["actor/L_return"] == pytest.approx(0.0)
        assert metrics["actor/return_trust_used"] == pytest.approx(0.0)
        actor_changed = any(
            not torch.equal(actor_before[name], param.detach())
            for name, param in actor.named_parameters()
        )
        assert actor_changed

    def test_reward_preference_losses_penalize_oversized_residual(self, config, world_model):
        from fragile.rl import train_dreamer

        reward_conservative = torch.tensor(
            [[[0.8], [0.4]], [[0.6], [0.2]]],
            dtype=torch.float32,
        )
        reward_nonconservative = torch.tensor(
            [[0.7, 0.3], [0.5, 0.4]],
            dtype=torch.float32,
        )
        reward_form_cov = torch.full((4, D), 0.5, dtype=torch.float32)
        replay_valid = torch.ones(2, 2, dtype=torch.float32)

        norm_loss, budget_loss, metrics = train_dreamer._reward_conservative_preference_losses(
            config,
            metric=world_model.metric,
            z=torch.zeros(4, D, dtype=torch.float32),
            reward_conservative=reward_conservative,
            reward_nonconservative=reward_nonconservative,
            reward_form_cov=reward_form_cov,
            replay_valid=replay_valid,
        )

        assert norm_loss.item() > 0.0
        assert budget_loss.item() > 0.0
        assert metrics["wm/reward_nonconservative_frac_masked"] > 0.0
        assert metrics["wm/reward_nonconservative_excess_mean"] > 0.0

    def test_exact_hodge_diagnostics_use_rollout_momentum(
        self,
        monkeypatch,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        world_model,
        critic,
        macro_critic,
        reward_head,
        actor,
        actor_old,
        enclosure_probe,
    ):
        from fragile.rl import train_dreamer

        batch = _make_training_batch(device)
        config.w_actor_return = 0.0
        optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            world_model,
            macro_critic,
            reward_head,
            actor,
            enclosure_probe,
        )

        original_forward = world_model.forward

        def marked_forward(self, z_0, action_canonicals, router_weights_0):
            out = original_forward(z_0, action_canonicals, router_weights_0)
            marked_momenta = out["momenta"].clone()
            marked_momenta[:, 0, :] = 0.777
            out["momenta"] = marked_momenta
            return out

        monkeypatch.setattr(
            world_model,
            "forward",
            marked_forward.__get__(world_model, type(world_model)),
        )

        captured_p: list[torch.Tensor] = []

        def fake_force_diag(_world_model, z, p, rw, action_canonical):
            del _world_model, rw, action_canonical
            captured_p.append(p.detach().clone())
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

        monkeypatch.setattr(train_dreamer, "_conservative_force_diagnostics", fake_force_diag)

        train_dreamer._train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            optimizer_enclosure,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        assert len(captured_p) == 2
        expected_horizon = max(config.wm_prediction_horizon, config.imagination_horizon)
        rollout_p = captured_p[1].reshape(B, expected_horizon, D)
        torch.testing.assert_close(
            rollout_p[:, 1, :],
            torch.full((B, D), 0.777, device=rollout_p.device, dtype=rollout_p.dtype),
        )

    def test_hodge_preference_losses_penalize_solenoidal_dominance(self, config):
        from fragile.rl import train_dreamer

        hodge_conservative = torch.full((2, H_WM), 0.05, dtype=torch.float32)
        hodge_solenoidal = torch.full((2, H_WM), 0.9, dtype=torch.float32)

        margin_loss, sol_loss, metrics = train_dreamer._hodge_conservative_preference_losses(
            config,
            hodge_conservative_ratio=hodge_conservative,
            hodge_solenoidal_ratio=hodge_solenoidal,
        )

        assert margin_loss.item() > 0.0
        assert sol_loss.item() > 0.0
        assert metrics["geometric/hodge_conservative_deficit"] == pytest.approx(0.2)
        assert metrics["geometric/hodge_conservative_target"] == pytest.approx(
            config.hodge_conservative_target,
        )

    def test_frozen_encoder_keeps_encoder_weights_fixed(
        self,
        device,
        config,
        phase1_cfg,
        action_phase1_cfg,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        world_model,
        critic,
        macro_critic,
        reward_head,
        actor,
        actor_old,
        enclosure_probe,
    ):
        from fragile.rl.train_dreamer import _train_step

        config.freeze_encoder = True
        batch = _make_training_batch(device)
        optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            world_model,
            macro_critic,
            reward_head,
            actor,
            enclosure_probe,
        )
        encoder_before = {
            name: param.detach().clone() for name, param in obs_model.encoder.named_parameters()
        }

        metrics = _train_step(
            obs_model,
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
            None,
            optimizer_boundary,
            optimizer_enclosure,
            batch,
            config,
            phase1_cfg,
            action_phase1_cfg,
            epoch=0,
            current_hard_routing=False,
            current_tau=1.0,
            update_idx=0,
            compute_diagnostics=False,
        )

        assert "enc/L_total" in metrics
        for name, param in obs_model.encoder.named_parameters():
            torch.testing.assert_close(param.detach(), encoder_before[name])


class TestCheckpoint:
    def test_save_checkpoint_stores_current_modules_and_normalizer(
        self,
        tmp_path,
        config,
        obs_model,
        jump_op,
        action_model,
        action_jump_op,
        world_model,
        critic,
        macro_critic,
        reward_head,
        actor,
        actor_old,
        enclosure_probe,
    ):
        from fragile.rl.train_dreamer import _save_checkpoint, ObservationNormalizer

        optimizer_enc, optimizer_wm, optimizer_boundary, optimizer_enclosure = _make_optimizers(
            obs_model,
            jump_op,
            action_model,
            action_jump_op,
            world_model,
            macro_critic,
            reward_head,
            actor,
            enclosure_probe,
        )
        scheduler_enc = torch.optim.lr_scheduler.LambdaLR(optimizer_enc, lr_lambda=lambda _: 1.0)
        scheduler_wm = torch.optim.lr_scheduler.LambdaLR(optimizer_wm, lr_lambda=lambda _: 1.0)
        scheduler_boundary = torch.optim.lr_scheduler.LambdaLR(
            optimizer_boundary,
            lr_lambda=lambda _: 1.0,
        )
        scheduler_enclosure = torch.optim.lr_scheduler.LambdaLR(
            optimizer_enclosure,
            lr_lambda=lambda _: 1.0,
        )
        normalizer = ObservationNormalizer(
            mean=torch.zeros(OBS_DIM),
            std=torch.ones(OBS_DIM),
            min_std=1e-3,
        )
        path = tmp_path / "dreamer.pt"
        _save_checkpoint(
            str(path),
            obs_model,
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
            None,
            optimizer_enclosure,
            scheduler_enc,
            scheduler_wm,
            scheduler_boundary,
            None,
            scheduler_enclosure,
            epoch=3,
            config=config,
            metrics={"wm/L_reward": 1.0},
            obs_normalizer=normalizer,
        )

        checkpoint = torch.load(path, map_location="cpu")
        assert checkpoint["epoch"] == 3
        assert "action_model" in checkpoint
        assert "world_model" in checkpoint
        assert "actor_old" in checkpoint
        assert "macro_critic" in checkpoint
        assert "reward_head" in checkpoint
        assert "enclosure_probe" in checkpoint
        assert "optimizer_wm" in checkpoint
        assert "optimizer_enclosure" in checkpoint
        assert "scheduler_boundary" in checkpoint
        assert "scheduler_enclosure" in checkpoint
        assert checkpoint["metrics"]["wm/L_reward"] == 1.0
        assert checkpoint["obs_normalizer"]["min_std"] == pytest.approx(1e-3)

    def test_train_rejects_legacy_partial_checkpoint(self, tmp_path, config, monkeypatch):
        from fragile.rl import train_dreamer

        checkpoint_path = tmp_path / "legacy.pt"
        torch.save({"encoder": {}}, checkpoint_path)
        config.load_checkpoint = str(checkpoint_path)
        config.collect_n_env_workers = 1

        def fake_make_env(domain, task):
            del domain, task
            return SingleEpisodeEnv(
                start_obs=np.zeros(OBS_DIM, dtype=np.float32),
                next_obs=np.zeros(OBS_DIM, dtype=np.float32),
                reward=0.0,
                action_dim=A,
            )

        monkeypatch.setattr(train_dreamer, "_make_env", fake_make_env)

        with pytest.raises(KeyError, match="missing required keys"):
            train_dreamer.train(config)
