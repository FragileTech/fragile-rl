"""Structural tests for the geometry-aware symbolic Markov model."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fragile.losses.markov_model import (
    _flatten_chart_code_probs,
    _masked_mean,
    _normalize_probs,
    _reshape_leading_dims,
    _state_index,
    _state_probs_to_chart_probs,
    _state_probs_to_code_conditionals,
    _validate_macro_geometry,
    compose_absolute_macro_dictionary,
    compute_distribution_alignment_loss,
    compute_markov_shape_loss,
    compute_markov_transition_loss,
    compute_markov_world_model_alignment_loss,
    expected_macro_state,
    MacroTransitionModel,
    soft_macro_state_distribution,
)


B, OBS_C, OBS_K, ACT_C, ACT_K, D = 4, 3, 2, 2, 2, 3
S = OBS_C * OBS_K
A = ACT_C * ACT_K


@pytest.fixture()
def t():
    """Deterministic macro geometry and symbolic beliefs."""
    torch.manual_seed(42)
    obs_chart_centers = torch.randn(OBS_C, D) * 0.2
    obs_codebook = torch.randn(OBS_C, OBS_K, D) * 0.15
    act_chart_centers = torch.randn(ACT_C, D) * 0.2
    act_codebook = torch.randn(ACT_C, ACT_K, D) * 0.15
    z_obs = torch.randn(B, D) * 0.2
    z_act = torch.randn(B, D) * 0.2
    state_probs = F.softmax(torch.randn(B, S), dim=-1)
    action_probs = F.softmax(torch.randn(B, A), dim=-1)
    valid_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])

    obs_macro = soft_macro_state_distribution(z_obs, obs_chart_centers, obs_codebook)
    act_macro = soft_macro_state_distribution(z_act, act_chart_centers, act_codebook)
    model = MacroTransitionModel(
        obs_latent_dim=D,
        act_latent_dim=D,
        num_obs_charts=OBS_C,
        obs_codes_per_chart=OBS_K,
        num_act_charts=ACT_C,
        act_codes_per_chart=ACT_K,
        hidden_dim=16,
        use_residual_transition=False,
    )
    residual_model = MacroTransitionModel(
        obs_latent_dim=D,
        act_latent_dim=D,
        num_obs_charts=OBS_C,
        obs_codes_per_chart=OBS_K,
        num_act_charts=ACT_C,
        act_codes_per_chart=ACT_K,
        hidden_dim=16,
        use_residual_transition=True,
    )

    obs_geometry = {
        "chart_centers": obs_macro["chart_centers"],
        "codebook": obs_macro["codebook"],
        "state_points": obs_macro["state_points"],
        "state_tangent_points": obs_macro["state_tangent_points"],
    }
    act_geometry = {
        "chart_centers": act_macro["chart_centers"],
        "codebook": act_macro["codebook"],
        "state_points": act_macro["state_points"],
        "state_tangent_points": act_macro["state_tangent_points"],
    }

    return SimpleNamespace(
        obs_chart_centers=obs_chart_centers,
        obs_codebook=obs_codebook,
        act_chart_centers=act_chart_centers,
        act_codebook=act_codebook,
        obs_macro=obs_macro,
        act_macro=act_macro,
        state_probs=state_probs,
        action_probs=action_probs,
        valid_mask=valid_mask,
        obs_geometry=obs_geometry,
        act_geometry=act_geometry,
        model=model,
        residual_model=residual_model,
    )


class TestHelpers:
    def test_masked_mean(self):
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
        result = _masked_mean(vals, mask)
        torch.testing.assert_close(result, torch.tensor(7.0 / 3.0), atol=1e-6, rtol=0)

    def test_state_index(self):
        chart_idx = torch.tensor([0, 1, 2])
        code_idx = torch.tensor([0, 1, 0])
        result = _state_index(chart_idx, code_idx, OBS_K)
        assert result.tolist() == [0, 3, 4]

    def test_normalize_probs_zero_safe(self):
        raw = torch.tensor([[0.0, 0.0, 0.0]])
        normed = _normalize_probs(raw)
        torch.testing.assert_close(normed, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6, rtol=0)

    def test_factorization_round_trip(self):
        probs = F.softmax(torch.randn(B, S), dim=-1)
        chart_probs = _state_probs_to_chart_probs(probs, OBS_C, OBS_K)
        code_probs = _state_probs_to_code_conditionals(probs, OBS_C, OBS_K)
        rebuilt = _flatten_chart_code_probs(chart_probs, code_probs)
        torch.testing.assert_close(rebuilt, probs, atol=1e-6, rtol=1e-6)

    def test_reshape_leading_dims_round_trip(self):
        x = torch.randn(2, 3, 5)
        flat = x.reshape(6, 5)
        restored = _reshape_leading_dims(flat, torch.Size([2, 3]))
        torch.testing.assert_close(restored, x, atol=0, rtol=0)

    def test_validate_macro_geometry_valid(self, t):
        _validate_macro_geometry(t.obs_chart_centers, t.obs_codebook)


class TestComposeAbsoluteMacroDictionary:
    def test_shapes(self, t):
        d = compose_absolute_macro_dictionary(t.obs_chart_centers, t.obs_codebook)
        assert d["state_points"].shape == (S, D)
        assert d["state_tangent_points"].shape == (S, D)
        assert d["chart_idx"].shape == (S,)
        assert d["code_idx"].shape == (S,)

    def test_points_inside_ball(self, t):
        d = compose_absolute_macro_dictionary(t.obs_chart_centers, t.obs_codebook)
        assert (d["state_points"].norm(dim=-1) < 1.0).all()


class TestExpectedMacroState:
    def test_shape(self, t):
        d = compose_absolute_macro_dictionary(t.obs_chart_centers, t.obs_codebook)
        result = expected_macro_state(t.state_probs, d["state_points"])
        assert result.shape == (B, D)

    def test_one_hot_selects_point(self, t):
        d = compose_absolute_macro_dictionary(t.obs_chart_centers, t.obs_codebook)
        one_hot = torch.zeros(1, S)
        one_hot[0, 2] = 1.0
        result = expected_macro_state(one_hot, d["state_points"])
        torch.testing.assert_close(result[0], d["state_points"][2], atol=1e-5, rtol=0)


class TestSoftMacroStateDistribution:
    def test_output_keys(self, t):
        sd = t.obs_macro
        for key in [
            "state_probs",
            "chart_probs",
            "code_probs",
            "macro_state_mean",
            "hard_state_point",
            "state_points",
            "state_tangent_points",
            "chart_centers",
            "codebook",
        ]:
            assert key in sd

    def test_state_probs_sum_to_one(self, t):
        torch.testing.assert_close(
            t.obs_macro["state_probs"].sum(dim=-1),
            torch.ones(B),
            atol=1e-5,
            rtol=0,
        )

    def test_low_temperature_sharpens(self, t):
        sd_warm = soft_macro_state_distribution(
            t.obs_macro["z_latent"],
            t.obs_chart_centers,
            t.obs_codebook,
            chart_tau=1.0,
            code_tau=1.0,
        )
        sd_cold = soft_macro_state_distribution(
            t.obs_macro["z_latent"],
            t.obs_chart_centers,
            t.obs_codebook,
            chart_tau=0.1,
            code_tau=0.1,
        )
        assert (
            sd_cold["state_probs"].max(dim=-1).values.mean()
            > sd_warm["state_probs"].max(dim=-1).values.mean()
        )


class TestMacroTransitionModel:
    def test_init_shapes(self, t):
        assert t.model.num_states == S
        assert t.model.num_actions == A
        assert t.model.reward_table.shape == (S, A)
        assert t.model.continuation_logits.shape == (S, A)

    def test_forward_shapes(self, t):
        out = t.model(
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        assert out["next_state_probs"].shape == (B, S)
        assert out["next_chart_probs"].shape == (B, OBS_C)
        assert out["next_code_probs"].shape == (B, OBS_C, OBS_K)
        assert out["next_query_point"].shape == (B, D)

    def test_forward_probabilities_are_normalized(self, t):
        out = t.model(
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        torch.testing.assert_close(
            out["next_state_probs"].sum(dim=-1), torch.ones(B), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            out["next_chart_probs"].sum(dim=-1), torch.ones(B), atol=1e-5, rtol=0
        )
        torch.testing.assert_close(
            out["next_code_probs"].sum(dim=-1), torch.ones(B, OBS_C), atol=1e-5, rtol=0
        )

    def test_without_residual_is_exactly_factorized(self, t):
        out = t.model(
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        rebuilt = _flatten_chart_code_probs(out["next_chart_probs"], out["next_code_probs"])
        torch.testing.assert_close(rebuilt, out["next_state_probs"], atol=1e-6, rtol=1e-6)

    def test_with_residual_keeps_valid_distribution(self, t):
        out = t.residual_model(
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        torch.testing.assert_close(
            out["next_state_probs"].sum(dim=-1), torch.ones(B), atol=1e-5, rtol=0
        )
        assert "residual_transition_logits" in out

    def test_reward_and_continuation_from_probs(self, t):
        reward = t.model.reward_from_probs(t.state_probs, t.action_probs)
        cont = t.model.continuation_from_probs(t.state_probs, t.action_probs)
        torch.testing.assert_close(reward, torch.zeros(B), atol=1e-6, rtol=0)
        torch.testing.assert_close(cont, torch.full((B,), 0.99), atol=1e-4, rtol=0)

    def test_conditional_from_indices(self, t):
        s_idx = torch.tensor([0, 1, 2, 3])
        a_idx = torch.tensor([0, 1, 2, 3])
        out = t.model.conditional_from_indices(
            s_idx,
            a_idx,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        assert out["next_state_probs"].shape == (4, S)

    def test_rollout_shapes(self, t):
        H = 3
        action_seq = F.softmax(torch.randn(B, H, A), dim=-1)
        ro = t.model.rollout(
            t.state_probs,
            action_seq,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
        )
        assert ro["state_probs"].shape == (B, H + 1, S)
        assert ro["next_state_probs"].shape == (B, H, S)


class TestMarkovTransitionLoss:
    def test_soft_targets(self, t):
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        loss, metrics, pred = compute_markov_transition_loss(
            t.model,
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
            target_next_state_probs=target_next,
        )
        assert loss.shape == ()
        assert pred["next_state_probs"].shape == (B, S)
        for key in [
            "markov/L_transition",
            "markov/transition_ce",
            "markov/state_ce",
            "markov/chart_ce",
            "markov/code_ce",
            "markov/chart_acc",
            "markov/code_acc",
            "markov/next_chart_entropy",
            "markov/next_code_entropy",
        ]:
            assert key in metrics

    def test_hard_targets(self, t):
        target_chart = torch.randint(0, OBS_C, (B,))
        target_code = torch.randint(0, OBS_K, (B,))
        loss, metrics, _ = compute_markov_transition_loss(
            t.model,
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
            target_next_chart_idx=target_chart,
            target_next_code_idx=target_code,
            codes_per_chart=OBS_K,
        )
        assert loss.shape == ()
        assert "markov/chart_acc" in metrics
        assert "markov/code_acc" in metrics

    def test_valid_mask_accepted(self, t):
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        loss, _, _ = compute_markov_transition_loss(
            t.model,
            t.state_probs,
            t.action_probs,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
            target_next_state_probs=target_next,
            valid_mask=t.valid_mask,
        )
        assert loss.shape == ()

    def test_gradient_flows_to_inputs(self, t):
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        sp = t.state_probs.clone().requires_grad_(True)
        ap = t.action_probs.clone().requires_grad_(True)
        loss, _, _ = compute_markov_transition_loss(
            t.model,
            sp,
            ap,
            obs_geometry=t.obs_geometry,
            act_geometry=t.act_geometry,
            target_next_state_probs=target_next,
        )
        loss.backward()
        assert sp.grad is not None and (sp.grad != 0).any()
        assert ap.grad is not None and (ap.grad != 0).any()


class TestDistributionAlignmentLoss:
    def test_shape_and_keys(self):
        teacher = F.softmax(torch.randn(B, S), dim=-1)
        student = F.softmax(torch.randn(B, S), dim=-1)
        loss, metrics = compute_distribution_alignment_loss(teacher, student)
        assert loss.shape == ()
        for key in [
            "markov/alignment/L_align",
            "markov/alignment/align_ce",
            "markov/alignment/align_kl",
            "markov/alignment/agreement",
            "markov/alignment/teacher_entropy",
            "markov/alignment/student_entropy",
        ]:
            assert key in metrics

    def test_identical_distributions(self):
        p = F.softmax(torch.randn(B, S), dim=-1)
        loss, _ = compute_distribution_alignment_loss(p, p.clone())
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_markov_shape_helpers(self):
        teacher = F.softmax(torch.randn(B, S), dim=-1)
        student = F.softmax(torch.randn(B, S), dim=-1)
        shape_loss, _ = compute_markov_shape_loss(teacher, student)
        wm_loss, _ = compute_markov_world_model_alignment_loss(teacher, student)
        assert shape_loss.shape == ()
        assert wm_loss.shape == ()
