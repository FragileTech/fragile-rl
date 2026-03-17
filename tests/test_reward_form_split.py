"""Focused regression tests for the exact/non-exact reward split."""

from __future__ import annotations

import pytest
import torch


B = 4
D = 8
A = 6
K = 4
D_MODEL = 32


@pytest.fixture
def z() -> torch.Tensor:
    """Random latent points inside the Poincare ball."""
    return torch.randn(B, D) * 0.3


@pytest.fixture
def rw() -> torch.Tensor:
    """Soft chart assignments."""
    return torch.softmax(torch.randn(B, K), dim=-1)


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
def reward_head(world_model):
    from fragile.rl.reward_head import RewardHead

    return RewardHead(world_model.potential_net, K, D_MODEL)


def _random_action_form_inputs(
    z: torch.Tensor,
    reward_head,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    action_z = torch.randn_like(z)
    num_action_charts = reward_head.action_chart_tok.chart_embeddings.shape[0]
    action_rw = torch.softmax(torch.randn(z.shape[0], num_action_charts, device=z.device), dim=-1)
    action_code_z = torch.randn_like(z)
    return action_z, action_rw, action_code_z


def test_reward_form_is_independent_of_control(reward_head, z, rw) -> None:
    reward_head.eval()
    action_z, action_rw, action_code_z = _random_action_form_inputs(z, reward_head)
    control_a = torch.randn_like(z)
    control_b = torch.randn_like(z)

    info_a = reward_head.decompose(
        z,
        rw,
        action_z,
        action_rw,
        action_code_z,
        action_canonical=control_a,
    )
    info_b = reward_head.decompose(
        z,
        rw,
        action_z,
        action_rw,
        action_code_z,
        action_canonical=control_b,
    )

    torch.testing.assert_close(info_a["reward_form_cov"], info_b["reward_form_cov"])


def test_exact_projection_is_enforced_by_construction(reward_head, z, rw, monkeypatch) -> None:
    action_z, action_rw, action_code_z = _random_action_form_inputs(z, reward_head)
    control = torch.randn_like(z)
    raw_form = torch.randn_like(z)

    def constant_raw_form(
        _z_in: torch.Tensor,
        _rw_in: torch.Tensor,
        _action_z_in: torch.Tensor,
        _action_rw_in: torch.Tensor,
        _action_code_z_in: torch.Tensor,
    ) -> torch.Tensor:
        return raw_form

    monkeypatch.setattr(
        reward_head,
        "reward_form",
        constant_raw_form,
    )

    info = reward_head.decompose(
        z,
        rw,
        action_z,
        action_rw,
        action_code_z,
        action_canonical=control,
        exact_covector=raw_form,
    )

    torch.testing.assert_close(info["reward_form_cov_raw"], raw_form)
    torch.testing.assert_close(
        info["reward_form_cov"], torch.zeros_like(raw_form), atol=1e-6, rtol=0
    )
    torch.testing.assert_close(
        info["reward_nonconservative"],
        torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype),
        atol=1e-6,
        rtol=0,
    )


def test_reward_curl_uses_local_exact_covector_field(reward_head, z, rw, monkeypatch) -> None:
    reward_head.eval()
    action_z, action_rw, action_code_z = _random_action_form_inputs(z, reward_head)
    raw_form = torch.zeros_like(z)
    raw_form[:, 0] = 1.0

    def constant_grad_connected_form(
        z_in: torch.Tensor,
        _rw_in: torch.Tensor,
        _action_z_in: torch.Tensor,
        _action_rw_in: torch.Tensor,
        _action_code_z_in: torch.Tensor,
    ) -> torch.Tensor:
        return z_in * 0.0 + raw_form[: z_in.shape[0]]

    monkeypatch.setattr(
        reward_head,
        "reward_form",
        constant_grad_connected_form,
    )

    def exact_covector_fn(z_in: torch.Tensor) -> torch.Tensor:
        cov = torch.zeros_like(z_in)
        cov[:, 0] = torch.cos(z_in[:, 0])
        cov[:, 1] = torch.sin(z_in[:, 0])
        return cov

    exact_covector = exact_covector_fn(z).detach()
    curl_frozen = reward_head.reward_curl(
        z,
        rw,
        action_z,
        action_rw,
        action_code_z,
        exact_covector=exact_covector,
    )
    curl_local = reward_head.reward_curl(
        z,
        rw,
        action_z,
        action_rw,
        action_code_z,
        exact_covector_fn=exact_covector_fn,
    )

    assert torch.allclose(curl_frozen, torch.zeros_like(curl_frozen), atol=1e-6, rtol=0)
    assert torch.linalg.norm(curl_local, dim=(-2, -1)).max() > 1e-4


def test_imagination_reward_curl_batch_limit_preserves_batch_shape(z, rw, monkeypatch) -> None:
    from fragile.rl import train_dreamer

    horizon = 2
    curl_limit = 2
    action_dim = 3
    action_charts = 5

    class DummyWorldModel:
        def momentum_init(self, z_0):
            return torch.zeros_like(z_0)

        def _rollout_transition(self, z_in, p_in, action_canonical, rw_in, track_energy=False):
            del action_canonical, track_energy
            chart_logits = torch.zeros(z_in.shape[0], K, device=z_in.device, dtype=z_in.dtype)
            chart_logits[:, 0] = 1.0
            return {
                "z": z_in + 0.05,
                "p": p_in,
                "rw": rw_in,
                "phi_eff": torch.ones(z_in.shape[0], 1, device=z_in.device, dtype=z_in.dtype),
                "chart_logits": chart_logits,
            }

    class DummyRewardHead:
        def decompose(
            self,
            z_in,
            rw_in,
            action_z,
            action_rw,
            action_code_z,
            action_canonical,
            *,
            exact_covector=None,
            exact_covector_fn=None,
            compute_curl=False,
            curl_batch_limit=None,
        ):
            del (
                rw_in,
                action_z,
                action_rw,
                action_code_z,
                action_canonical,
                exact_covector,
                exact_covector_fn,
                compute_curl,
                curl_batch_limit,
            )
            return {"reward_nonconservative": torch.ones(z_in.shape[0], 1, device=z_in.device)}

        def reward_curl(
            self,
            z_in,
            rw_in,
            action_z,
            action_rw,
            action_code_z,
            *,
            exact_covector=None,
            exact_covector_fn=None,
            max_batch=None,
        ):
            del rw_in, action_z, action_rw, action_code_z, exact_covector, exact_covector_fn
            batch = z_in.shape[0] if max_batch is None else min(z_in.shape[0], max_batch)
            return torch.ones(batch, z_in.shape[-1], z_in.shape[-1], device=z_in.device)

    def fake_symbolize_latent_with_atlas(_model, z_in: torch.Tensor, **_kwargs):
        router = torch.zeros(z_in.shape[0], K, device=z_in.device, dtype=z_in.dtype)
        router[:, 0] = 1.0
        return {
            "z_geo": z_in,
            "z_n": torch.zeros_like(z_in),
            "router_weights": router,
            "chart_idx": torch.zeros(z_in.shape[0], dtype=torch.long, device=z_in.device),
            "code_idx": torch.zeros(z_in.shape[0], dtype=torch.long, device=z_in.device),
            "z_q": torch.zeros_like(z_in),
        }

    def fake_policy_action(_actor, _action_model, obs_info, **_kwargs):
        z_in = obs_info["z_geo"]
        return {
            "action_canonical": torch.ones_like(z_in),
            "action_latent": torch.zeros_like(z_in),
            "action_router_weights": torch.softmax(
                torch.randn(z_in.shape[0], action_charts, device=z_in.device),
                dim=-1,
            ),
            "action_mean": torch.zeros(z_in.shape[0], action_dim, device=z_in.device),
            "action_code_latent": torch.zeros_like(z_in),
        }

    def fake_conservative_reward_from_value(_critic, z_curr, _rw_curr, _z_next, _rw_next, _gamma):
        zeros = torch.zeros(z_curr.shape[0], 1, device=z_curr.device)
        return zeros, zeros, zeros

    def fake_value_covector_from_critic(_critic, z_in, _rw_in, **_kwargs):
        return torch.ones_like(z_in)

    monkeypatch.setattr(
        train_dreamer,
        "symbolize_latent_with_atlas",
        fake_symbolize_latent_with_atlas,
    )
    monkeypatch.setattr(
        train_dreamer,
        "_policy_action",
        fake_policy_action,
    )
    monkeypatch.setattr(
        train_dreamer,
        "_conservative_reward_from_value",
        fake_conservative_reward_from_value,
    )
    monkeypatch.setattr(
        train_dreamer,
        "_value_covector_from_critic",
        fake_value_covector_from_critic,
    )

    out = train_dreamer._imagine(
        object(),
        DummyWorldModel(),
        DummyRewardHead(),
        object(),
        object(),
        object(),
        z,
        rw,
        horizon=horizon,
        gamma=0.99,
        reward_curl_batch_limit=curl_limit,
    )

    assert out["reward_curl_norm"].shape == (z.shape[0], horizon)
    assert out["reward_curl_valid"].shape == (z.shape[0], horizon)
    assert out["reward_curl_valid"][:curl_limit].all()
    assert not out["reward_curl_valid"][curl_limit:].any()
    assert torch.all(out["reward_curl_norm"][curl_limit:] == 0.0)
