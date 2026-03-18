"""Golden-value regression tests for fragile.losses.markov_model.

Every test uses a single deterministic fixture (seed=42) and asserts against
hardcoded golden values. Any refactoring that changes numerical output will
be caught immediately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from fragile.losses.markov_model import (
    MacroTransitionModel,
    _masked_mean,
    _normalize_probs,
    _reshape_leading_dims,
    _state_index,
    _validate_macro_geometry,
    compose_absolute_macro_dictionary,
    compute_distribution_alignment_loss,
    compute_markov_shape_loss,
    compute_markov_transition_loss,
    compute_markov_world_model_alignment_loss,
    expected_macro_state,
    soft_macro_state_distribution,
)


B, N_C, K, D = 4, 3, 2, 3
S = N_C * K  # 6 total states
A = 4  # num actions


@pytest.fixture()
def t():
    """Master fixture: deterministic tensors from seed 42."""
    torch.manual_seed(42)
    chart_centers = torch.randn(N_C, D) * 0.3
    codebook = torch.randn(N_C, K, D) * 0.2
    z_latent = torch.randn(B, D) * 0.3
    state_probs = F.softmax(torch.randn(B, S), dim=-1)
    action_probs = F.softmax(torch.randn(B, A), dim=-1)
    valid_mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
    model = MacroTransitionModel(S, A)

    return SimpleNamespace(
        chart_centers=chart_centers,
        codebook=codebook,
        z_latent=z_latent,
        state_probs=state_probs,
        action_probs=action_probs,
        valid_mask=valid_mask,
        model=model,
    )


# =========================================================================
# Helpers
# =========================================================================


class TestHelpers:
    def test_masked_mean(self):
        vals = torch.tensor([1.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
        result = _masked_mean(vals, mask)
        torch.testing.assert_close(
            result, torch.tensor(7.0 / 3.0), atol=1e-6, rtol=0
        )

    def test_masked_mean_all_zero_mask(self):
        vals = torch.tensor([10.0, 20.0])
        mask = torch.tensor([0.0, 0.0])
        result = _masked_mean(vals, mask)
        torch.testing.assert_close(result, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_state_index(self):
        chart_idx = torch.tensor([0, 1, 2])
        code_idx = torch.tensor([0, 1, 0])
        result = _state_index(chart_idx, code_idx, K)
        assert result.tolist() == [0, 3, 4]

    def test_normalize_probs(self):
        raw = torch.tensor([[1.0, 2.0, 3.0]])
        normed = _normalize_probs(raw)
        torch.testing.assert_close(
            normed.sum(dim=-1), torch.tensor([1.0]), atol=1e-6, rtol=0
        )

    def test_normalize_probs_zero_safe(self):
        raw = torch.tensor([[0.0, 0.0, 0.0]])
        normed = _normalize_probs(raw)
        torch.testing.assert_close(
            normed, torch.tensor([[0.0, 0.0, 0.0]]), atol=1e-6, rtol=0
        )

    def test_validate_macro_geometry_valid(self, t):
        _validate_macro_geometry(t.chart_centers, t.codebook)

    @pytest.mark.parametrize(
        "cc_shape,cb_shape,msg_fragment",
        [
            ((3,), (3, 2, 3), "chart_centers must have shape"),
            ((3, 3), (3, 3), "codebook must have shape"),
            ((2, 3), (3, 2, 3), "number of charts"),
            ((3, 4), (3, 2, 3), "latent dimension"),
        ],
    )
    def test_validate_macro_geometry_errors(self, cc_shape, cb_shape, msg_fragment):
        cc = torch.randn(*cc_shape)
        cb = torch.randn(*cb_shape)
        with pytest.raises(ValueError, match=msg_fragment):
            _validate_macro_geometry(cc, cb)

    def test_reshape_leading_dims_round_trip(self):
        x = torch.randn(2, 3, 5)
        flat = x.reshape(6, 5)
        restored = _reshape_leading_dims(flat, torch.Size([2, 3]))
        torch.testing.assert_close(restored, x, atol=0, rtol=0)

    def test_reshape_leading_dims_empty(self):
        x = torch.randn(1, 5)
        restored = _reshape_leading_dims(x, torch.Size([]))
        assert restored.shape == (5,)

    def test_reshape_leading_dims_scalar(self):
        x = torch.tensor([3.0])
        restored = _reshape_leading_dims(x, torch.Size([]))
        assert restored.shape == ()


# =========================================================================
# compose_absolute_macro_dictionary
# =========================================================================


class TestComposeAbsoluteMacroDictionary:
    def test_output_keys(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        assert sorted(d.keys()) == [
            "chart_idx",
            "code_idx",
            "state_points",
            "state_tangent_points",
        ]

    def test_shapes(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        assert d["state_points"].shape == (S, D)
        assert d["state_tangent_points"].shape == (S, D)
        assert d["chart_idx"].shape == (S,)
        assert d["code_idx"].shape == (S,)

    def test_golden_values(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        expected = torch.tensor([
            [-0.041410475969314575, 0.25439128279685974, 0.13580092787742615],
            [-0.05178143084049225, 0.04675149917602539, 0.13685134053230286],
            [0.31950709223747253, -0.5163230895996094, 0.1359129101037979],
        ])
        torch.testing.assert_close(
            d["state_points"][:3], expected, atol=1e-6, rtol=0
        )

    def test_chart_code_indices(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        assert d["chart_idx"].tolist() == [0, 0, 1, 1, 2, 2]
        assert d["code_idx"].tolist() == [0, 1, 0, 1, 0, 1]

    def test_points_inside_ball(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        norms = d["state_points"].norm(dim=-1)
        assert (norms < 1.0).all()


# =========================================================================
# expected_macro_state
# =========================================================================


class TestExpectedMacroState:
    def test_shape(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        result = expected_macro_state(t.state_probs, d["state_points"])
        assert result.shape == (B, D)

    def test_golden_value(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        result = expected_macro_state(t.state_probs, d["state_points"])
        expected = torch.tensor(
            [0.11169666796922684, -0.058720674365758896, 0.12175660580396652]
        )
        torch.testing.assert_close(result[0], expected, atol=1e-6, rtol=0)

    def test_one_hot_selects_point(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        one_hot = torch.zeros(1, S)
        one_hot[0, 2] = 1.0
        result = expected_macro_state(one_hot, d["state_points"])
        torch.testing.assert_close(
            result[0], d["state_points"][2], atol=1e-5, rtol=0
        )

    def test_gradient_flows(self, t):
        d = compose_absolute_macro_dictionary(t.chart_centers, t.codebook)
        sp = t.state_probs.clone().requires_grad_(True)
        result = expected_macro_state(sp, d["state_points"])
        result.sum().backward()
        assert sp.grad is not None
        assert (sp.grad != 0).any()


# =========================================================================
# soft_macro_state_distribution
# =========================================================================


class TestSoftMacroStateDistribution:
    def test_output_keys(self, t):
        sd = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook
        )
        for key in [
            "state_probs",
            "chart_probs",
            "code_probs",
            "state_log_probs",
            "chart_idx",
            "code_idx",
            "state_idx",
            "macro_state_mean",
            "hard_state_point",
            "state_entropy",
            "chart_entropy",
            "state_points",
        ]:
            assert key in sd, f"Missing key: {key}"

    def test_shapes(self, t):
        sd = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook
        )
        assert sd["state_probs"].shape == (B, S)
        assert sd["chart_probs"].shape == (B, N_C)
        assert sd["code_probs"].shape == (B, N_C, K)
        # chart_idx/code_idx come from symbol_dict (**symbol_dict overwrites)
        assert sd["chart_idx"].shape == (S,)
        assert sd["code_idx"].shape == (S,)
        assert sd["state_idx"].shape == (B,)
        assert sd["macro_state_mean"].shape == (B, D)

    def test_state_probs_sum_to_one(self, t):
        sd = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook
        )
        torch.testing.assert_close(
            sd["state_probs"].sum(dim=-1),
            torch.ones(B),
            atol=1e-5,
            rtol=0,
        )

    def test_chart_probs_sum_to_one(self, t):
        sd = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook
        )
        torch.testing.assert_close(
            sd["chart_probs"].sum(dim=-1),
            torch.ones(B),
            atol=1e-5,
            rtol=0,
        )

    def test_golden_values(self, t):
        sd = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook
        )
        expected_sp = torch.tensor([
            0.3441481590270996,
            0.24397771060466766,
            0.1265508532524109,
            0.1400601714849472,
            0.10480938851833344,
            0.040453772991895676,
        ])
        torch.testing.assert_close(
            sd["state_probs"][0], expected_sp, atol=1e-6, rtol=0
        )
        expected_cp = torch.tensor(
            [0.5881258249282837, 0.2666110396385193, 0.1452631652355194]
        )
        torch.testing.assert_close(
            sd["chart_probs"][0], expected_cp, atol=1e-6, rtol=0
        )

    def test_gradient_flows(self, t):
        zl = t.z_latent.clone().requires_grad_(True)
        sd = soft_macro_state_distribution(zl, t.chart_centers, t.codebook)
        sd["state_probs"].sum().backward()
        assert zl.grad is not None
        assert (zl.grad != 0).any()

    def test_batch_dims(self, t):
        z_bt = torch.randn(2, 3, D) * 0.3
        sd = soft_macro_state_distribution(z_bt, t.chart_centers, t.codebook)
        assert sd["state_probs"].shape == (2, 3, S)
        assert sd["chart_probs"].shape == (2, 3, N_C)

    def test_low_temperature_sharpens(self, t):
        sd_warm = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook,
            chart_tau=1.0, code_tau=1.0,
        )
        sd_cold = soft_macro_state_distribution(
            t.z_latent, t.chart_centers, t.codebook,
            chart_tau=0.1, code_tau=0.1,
        )
        warm_peak = sd_warm["state_probs"].max(dim=-1).values.mean()
        cold_peak = sd_cold["state_probs"].max(dim=-1).values.mean()
        assert cold_peak > warm_peak

    def test_validation_errors(self, t):
        with pytest.raises(ValueError, match="z_latent must have shape"):
            soft_macro_state_distribution(
                torch.randn(D), t.chart_centers, t.codebook
            )
        with pytest.raises(ValueError, match="latent dimension"):
            soft_macro_state_distribution(
                torch.randn(B, D + 1), t.chart_centers, t.codebook
            )


# =========================================================================
# MacroTransitionModel
# =========================================================================


class TestMacroTransitionModel:
    def test_init_shapes(self, t):
        assert t.model.transition_logits.shape == (S, A, S)
        assert t.model.reward_table.shape == (S, A)
        assert t.model.continuation_logits.shape == (S, A)

    def test_init_no_reward(self):
        m = MacroTransitionModel(S, A, learn_reward=False)
        assert m.reward_table is None

    def test_init_no_continuation(self):
        m = MacroTransitionModel(S, A, learn_continuation=False)
        assert m.continuation_logits is None

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="positive"):
            MacroTransitionModel(0, A)
        with pytest.raises(ValueError, match="positive"):
            MacroTransitionModel(S, 0)

    def test_transition_table_normalized(self, t):
        tt = t.model.transition_table()
        sums = tt.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(S, A), atol=1e-6, rtol=0)

    def test_continuation_table_range(self, t):
        ct = t.model.continuation_table()
        assert (ct > 0).all()
        assert (ct < 1).all()

    def test_forward_shapes(self, t):
        out = t.model(t.state_probs, t.action_probs)
        assert out["next_state_probs"].shape == (B, S)
        assert out["next_state_log_probs"].shape == (B, S)
        assert out["next_state_entropy"].shape == (B,)
        assert out["next_state_top1_prob"].shape == (B,)
        assert out["reward"].shape == (B,)
        assert out["continuation"].shape == (B,)

    def test_forward_golden(self, t):
        out = t.model(t.state_probs, t.action_probs)
        expected = torch.tensor([
            0.1666666716337204,
            0.1666666716337204,
            0.1666666716337204,
            0.1666666716337204,
            0.1666666716337204,
            0.1666666716337204,
        ])
        torch.testing.assert_close(
            out["next_state_probs"][0], expected, atol=1e-6, rtol=0
        )

    def test_forward_includes_reward_continuation(self, t):
        out = t.model(t.state_probs, t.action_probs)
        assert "reward" in out
        assert "continuation" in out

    def test_forward_excludes_reward_continuation(self):
        m = MacroTransitionModel(S, A, learn_reward=False, learn_continuation=False)
        sp = F.softmax(torch.randn(B, S), dim=-1)
        ap = F.softmax(torch.randn(B, A), dim=-1)
        out = m(sp, ap)
        assert "reward" not in out
        assert "continuation" not in out

    def test_forward_validation_errors(self, t):
        with pytest.raises(ValueError, match="shape"):
            t.model(torch.randn(S), t.action_probs)
        with pytest.raises(ValueError, match="leading shape"):
            t.model(torch.randn(3, S), torch.randn(4, A))
        with pytest.raises(ValueError, match="wrong number of states"):
            t.model(torch.randn(B, S + 1), t.action_probs)
        with pytest.raises(ValueError, match="wrong number of actions"):
            t.model(t.state_probs, torch.randn(B, A + 1))

    def test_reward_from_probs(self, t):
        result = t.model.reward_from_probs(t.state_probs, t.action_probs)
        torch.testing.assert_close(
            result, torch.zeros(B), atol=1e-6, rtol=0
        )

    def test_reward_from_probs_no_table(self):
        m = MacroTransitionModel(S, A, learn_reward=False)
        sp = F.softmax(torch.randn(B, S), dim=-1)
        ap = F.softmax(torch.randn(B, A), dim=-1)
        result = m.reward_from_probs(sp, ap)
        torch.testing.assert_close(result, torch.zeros(B), atol=1e-6, rtol=0)

    def test_continuation_from_probs(self, t):
        result = t.model.continuation_from_probs(t.state_probs, t.action_probs)
        expected = torch.full((B,), 0.99)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=0)

    def test_continuation_from_probs_no_table(self):
        m = MacroTransitionModel(S, A, learn_continuation=False)
        sp = F.softmax(torch.randn(B, S), dim=-1)
        ap = F.softmax(torch.randn(B, A), dim=-1)
        result = m.continuation_from_probs(sp, ap)
        torch.testing.assert_close(result, torch.ones(B), atol=1e-6, rtol=0)

    def test_conditional_from_indices(self, t):
        s_idx = torch.tensor([0, 1, 2, 3])
        a_idx = torch.tensor([0, 1, 2, 3])
        out = t.model.conditional_from_indices(s_idx, a_idx)
        assert out["next_state_probs"].shape == (4, S)
        expected = torch.full((S,), 1.0 / S)
        torch.testing.assert_close(
            out["next_state_probs"][0], expected, atol=1e-6, rtol=0
        )

    def test_conditional_from_indices_shape_error(self, t):
        with pytest.raises(ValueError, match="matching shapes"):
            t.model.conditional_from_indices(
                torch.tensor([0, 1]), torch.tensor([0])
            )

    def test_rollout_shapes(self, t):
        H = 3
        torch.manual_seed(42)
        action_seq = F.softmax(torch.randn(B, H, A), dim=-1)
        ro = t.model.rollout(t.state_probs, action_seq)
        assert ro["state_probs"].shape == (B, H + 1, S)
        assert ro["next_state_probs"].shape == (B, H, S)
        assert "reward" in ro
        assert "continuation" in ro

    def test_rollout_golden(self, t):
        H = 3
        torch.manual_seed(42)
        action_seq = F.softmax(torch.randn(B, H, A), dim=-1)
        ro = t.model.rollout(t.state_probs, action_seq)
        expected_sp0 = torch.tensor([
            0.14898760616779327,
            0.5809765458106995,
            0.10738369077444077,
            0.07570149004459381,
            0.027795124799013138,
            0.05915550887584686,
        ])
        torch.testing.assert_close(
            ro["state_probs"][0, 0], expected_sp0, atol=1e-6, rtol=0
        )

    def test_rollout_validation_errors(self, t):
        with pytest.raises(ValueError, match="state_probs_0 must have shape"):
            t.model.rollout(torch.randn(S), torch.randn(B, 3, A))
        with pytest.raises(ValueError, match="action_probs_seq must have shape"):
            t.model.rollout(t.state_probs, torch.randn(B, A))
        with pytest.raises(ValueError, match="batch shape"):
            t.model.rollout(
                torch.randn(3, S), torch.randn(4, 3, A)
            )


# =========================================================================
# compute_markov_transition_loss
# =========================================================================


class TestMarkovTransitionLoss:
    def test_soft_targets(self, t):
        torch.manual_seed(99)
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        loss, metrics, pred = compute_markov_transition_loss(
            t.model, t.state_probs, t.action_probs,
            target_next_state_probs=target_next,
        )
        assert loss.shape == ()
        torch.testing.assert_close(
            loss, torch.tensor(1.7917594909667969), atol=1e-6, rtol=0
        )
        assert "markov/L_transition" in metrics
        assert "markov/transition_ce" in metrics
        assert "markov/transition_acc" in metrics

    def test_hard_targets(self, t):
        torch.manual_seed(99)
        target_chart = torch.randint(0, N_C, (B,))
        target_code = torch.randint(0, K, (B,))
        loss, metrics, _ = compute_markov_transition_loss(
            t.model, t.state_probs, t.action_probs,
            target_next_chart_idx=target_chart,
            target_next_code_idx=target_code,
            codes_per_chart=K,
        )
        assert loss.shape == ()
        torch.testing.assert_close(
            loss, torch.tensor(1.7917594909667969), atol=1e-6, rtol=0
        )
        assert "markov/chart_acc" in metrics
        assert "markov/code_acc" in metrics

    def test_missing_targets_error(self, t):
        with pytest.raises(ValueError, match="Provide either"):
            compute_markov_transition_loss(
                t.model, t.state_probs, t.action_probs,
            )

    def test_valid_mask_accepted(self, t):
        torch.manual_seed(99)
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        loss, _, _ = compute_markov_transition_loss(
            t.model, t.state_probs, t.action_probs,
            target_next_state_probs=target_next,
            valid_mask=t.valid_mask,
        )
        assert loss.shape == ()

    def test_valid_mask_effect(self):
        vals = torch.tensor([10.0, 2.0, 3.0, 4.0])
        mask = torch.tensor([1.0, 1.0, 0.0, 1.0])
        masked = _masked_mean(vals, mask)
        unmasked = _masked_mean(vals, torch.ones(4))
        assert not torch.allclose(masked, unmasked)

    def test_gradient_flows(self, t):
        torch.manual_seed(99)
        target_next = F.softmax(torch.randn(B, S), dim=-1)
        sp = t.state_probs.clone().requires_grad_(True)
        ap = t.action_probs.clone().requires_grad_(True)
        loss, _, _ = compute_markov_transition_loss(
            t.model, sp, ap,
            target_next_state_probs=target_next,
        )
        loss.backward()
        assert sp.grad is not None
        assert ap.grad is not None


# =========================================================================
# compute_distribution_alignment_loss
# =========================================================================


class TestDistributionAlignmentLoss:
    def test_shape_and_keys(self, t):
        torch.manual_seed(99)
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
            assert key in metrics, f"Missing metric: {key}"

    def test_golden_value(self, t):
        torch.manual_seed(99)
        teacher = F.softmax(torch.randn(B, S), dim=-1)
        student = F.softmax(torch.randn(B, S), dim=-1)
        loss, _ = compute_distribution_alignment_loss(teacher, student)
        torch.testing.assert_close(
            loss, torch.tensor(0.5700478553771973), atol=1e-6, rtol=0
        )

    def test_identical_distributions(self):
        p = F.softmax(torch.randn(B, S), dim=-1)
        loss, _ = compute_distribution_alignment_loss(p, p.clone())
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)

    def test_detach_teacher(self):
        torch.manual_seed(99)
        teacher_raw = torch.randn(B, S, requires_grad=True)
        teacher = F.softmax(teacher_raw, dim=-1)
        student_raw = torch.randn(B, S, requires_grad=True)
        student = F.softmax(student_raw, dim=-1)
        loss, _ = compute_distribution_alignment_loss(
            teacher, student, detach_teacher=True
        )
        loss.backward()
        assert student_raw.grad is not None and (student_raw.grad != 0).any()
        assert teacher_raw.grad is None or not (teacher_raw.grad != 0).any()

    def test_no_detach_teacher(self):
        torch.manual_seed(99)
        teacher_raw = torch.randn(B, S, requires_grad=True)
        teacher = F.softmax(teacher_raw, dim=-1)
        student_raw = torch.randn(B, S, requires_grad=True)
        student = F.softmax(student_raw, dim=-1)
        loss, _ = compute_distribution_alignment_loss(
            teacher, student, detach_teacher=False
        )
        loss.backward()
        assert teacher_raw.grad is not None and (teacher_raw.grad != 0).any()

    def test_shape_mismatch_error(self):
        with pytest.raises(ValueError, match="same shape"):
            compute_distribution_alignment_loss(
                torch.randn(B, S), torch.randn(B, S + 1)
            )


# =========================================================================
# Shape and World Model Wrappers
# =========================================================================


class TestShapeAndWorldModelWrappers:
    def test_markov_shape_loss_delegates(self):
        torch.manual_seed(99)
        t1 = F.softmax(torch.randn(B, S), dim=-1)
        t2 = F.softmax(torch.randn(B, S), dim=-1)
        loss_shape, _ = compute_markov_shape_loss(t1, t2)
        torch.manual_seed(99)
        t1b = F.softmax(torch.randn(B, S), dim=-1)
        t2b = F.softmax(torch.randn(B, S), dim=-1)
        loss_align, _ = compute_distribution_alignment_loss(
            t1b, t2b, metric_prefix="markov/shape"
        )
        torch.testing.assert_close(loss_shape, loss_align, atol=1e-7, rtol=0)

    def test_world_model_alignment_delegates(self):
        torch.manual_seed(99)
        t1 = F.softmax(torch.randn(B, S), dim=-1)
        t2 = F.softmax(torch.randn(B, S), dim=-1)
        loss_wm, _ = compute_markov_world_model_alignment_loss(t1, t2)
        torch.manual_seed(99)
        t1b = F.softmax(torch.randn(B, S), dim=-1)
        t2b = F.softmax(torch.randn(B, S), dim=-1)
        loss_align, _ = compute_distribution_alignment_loss(
            t1b, t2b, metric_prefix="markov/wm_align"
        )
        torch.testing.assert_close(loss_wm, loss_align, atol=1e-7, rtol=0)
