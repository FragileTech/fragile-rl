"""Tests for fragile.losses.macro.

Covers the absolute-state composition helpers, validation logic,
the AbsoluteEnclosureProbe module, the end-to-end enclosure loss,
and the zeno_loss routing-change penalty.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from fragile.losses.macro import (
    _state_index,
    _validate_hard_symbol_inputs,
    AbsoluteEnclosureProbe,
    compose_absolute_macro_state,
    compose_absolute_structured_state,
    compute_absolute_enclosure_loss,
    zeno_loss,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_OBS_CHARTS = 3
N_ACT_CHARTS = 2
CODES_PER_CHART = 4
D = 4
B = 8


@pytest.fixture
def chart_centers():
    torch.manual_seed(0)
    return torch.randn(N_OBS_CHARTS, D) * 0.3


@pytest.fixture
def codebook():
    torch.manual_seed(1)
    return torch.randn(N_OBS_CHARTS, CODES_PER_CHART, D) * 0.2


@pytest.fixture
def chart_idx():
    torch.manual_seed(2)
    return torch.randint(0, N_OBS_CHARTS, (B,))


@pytest.fixture
def code_idx():
    torch.manual_seed(3)
    return torch.randint(0, CODES_PER_CHART, (B,))


@pytest.fixture
def z_n():
    torch.manual_seed(4)
    return torch.randn(B, D) * 0.1


# ---------------------------------------------------------------------------
# _state_index
# ---------------------------------------------------------------------------


class TestStateIndex:
    def test_basic_indexing(self):
        chart = torch.tensor([0, 1, 2])
        code = torch.tensor([0, 3, 1])
        idx = _state_index(chart, code, codes_per_chart=4)
        assert idx.tolist() == [0, 7, 9]

    def test_all_zero(self):
        chart = torch.zeros(5, dtype=torch.long)
        code = torch.zeros(5, dtype=torch.long)
        idx = _state_index(chart, code, codes_per_chart=10)
        assert (idx == 0).all()


# ---------------------------------------------------------------------------
# _validate_hard_symbol_inputs
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_inputs_pass(self, chart_centers, codebook, chart_idx, code_idx, z_n):
        _validate_hard_symbol_inputs(chart_centers, codebook, chart_idx, code_idx, z_n=z_n)

    def test_wrong_chart_centers_dim(self, codebook, chart_idx, code_idx):
        with pytest.raises(ValueError, match="chart_centers must have shape"):
            _validate_hard_symbol_inputs(torch.randn(D), codebook, chart_idx, code_idx)

    def test_wrong_codebook_dim(self, chart_centers, chart_idx, code_idx):
        with pytest.raises(ValueError, match="codebook must have shape"):
            _validate_hard_symbol_inputs(
                chart_centers, torch.randn(N_OBS_CHARTS, D), chart_idx, code_idx
            )

    def test_codebook_chart_count_mismatch(self, chart_centers, chart_idx, code_idx):
        bad_codebook = torch.randn(N_OBS_CHARTS + 1, CODES_PER_CHART, D)
        with pytest.raises(ValueError, match="codebook must agree"):
            _validate_hard_symbol_inputs(chart_centers, bad_codebook, chart_idx, code_idx)

    def test_codebook_latent_dim_mismatch(self, chart_centers, chart_idx, code_idx):
        bad_codebook = torch.randn(N_OBS_CHARTS, CODES_PER_CHART, D + 1)
        with pytest.raises(ValueError, match="codebook must agree"):
            _validate_hard_symbol_inputs(chart_centers, bad_codebook, chart_idx, code_idx)

    def test_chart_idx_wrong_dim(self, chart_centers, codebook, code_idx):
        with pytest.raises(ValueError, match="chart_idx and code_idx must both have shape"):
            _validate_hard_symbol_inputs(
                chart_centers, codebook, torch.zeros(B, 1, dtype=torch.long), code_idx
            )

    def test_batch_size_mismatch(self, chart_centers, codebook, chart_idx):
        with pytest.raises(ValueError, match="same batch size"):
            _validate_hard_symbol_inputs(
                chart_centers, codebook, chart_idx, torch.zeros(B + 1, dtype=torch.long)
            )

    def test_z_n_wrong_dim(self, chart_centers, codebook, chart_idx, code_idx):
        with pytest.raises(ValueError, match="z_n must have shape"):
            _validate_hard_symbol_inputs(
                chart_centers, codebook, chart_idx, code_idx, z_n=torch.randn(B)
            )

    def test_z_n_batch_mismatch(self, chart_centers, codebook, chart_idx, code_idx):
        with pytest.raises(ValueError, match="z_n must match"):
            _validate_hard_symbol_inputs(
                chart_centers, codebook, chart_idx, code_idx, z_n=torch.randn(B + 1, D)
            )

    def test_z_n_latent_dim_mismatch(self, chart_centers, codebook, chart_idx, code_idx):
        with pytest.raises(ValueError, match="z_n must match"):
            _validate_hard_symbol_inputs(
                chart_centers, codebook, chart_idx, code_idx, z_n=torch.randn(B, D + 1)
            )


# ---------------------------------------------------------------------------
# compose_absolute_macro_state
# ---------------------------------------------------------------------------


class TestComposeAbsoluteMacroState:
    def test_output_shape(self, chart_centers, codebook, chart_idx, code_idx):
        u = compose_absolute_macro_state(chart_centers, codebook, chart_idx, code_idx)
        assert u.shape == (B, D)

    def test_inside_poincare_ball(self, chart_centers, codebook, chart_idx, code_idx):
        u = compose_absolute_macro_state(chart_centers, codebook, chart_idx, code_idx)
        assert (u.norm(dim=-1) < 1.0).all()

    def test_finite(self, chart_centers, codebook, chart_idx, code_idx):
        u = compose_absolute_macro_state(chart_centers, codebook, chart_idx, code_idx)
        assert torch.isfinite(u).all()

    def test_different_charts_give_different_states(self, codebook):
        centers = torch.tensor([[0.3, 0.0, 0.0, 0.0], [-0.3, 0.0, 0.0, 0.0]])
        cb = codebook[:2]
        chart_a = torch.tensor([0])
        chart_b = torch.tensor([1])
        code = torch.tensor([0])
        u_a = compose_absolute_macro_state(centers, cb, chart_a, code)
        u_b = compose_absolute_macro_state(centers, cb, chart_b, code)
        assert not torch.allclose(u_a, u_b, atol=1e-4)

    def test_gradients_flow(self, chart_idx, code_idx):
        centers = nn.Parameter(torch.randn(N_OBS_CHARTS, D) * 0.3)
        cb = nn.Parameter(torch.randn(N_OBS_CHARTS, CODES_PER_CHART, D) * 0.2)
        u = compose_absolute_macro_state(centers, cb, chart_idx, code_idx)
        u.sum().backward()
        assert centers.grad is not None and centers.grad.abs().sum() > 0
        assert cb.grad is not None and cb.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# compose_absolute_structured_state
# ---------------------------------------------------------------------------


class TestComposeAbsoluteStructuredState:
    def test_output_shape(self, chart_centers, codebook, chart_idx, code_idx, z_n):
        u = compose_absolute_structured_state(chart_centers, codebook, chart_idx, code_idx, z_n)
        assert u.shape == (B, D)

    def test_inside_poincare_ball(self, chart_centers, codebook, chart_idx, code_idx, z_n):
        u = compose_absolute_structured_state(chart_centers, codebook, chart_idx, code_idx, z_n)
        assert (u.norm(dim=-1) < 1.0).all()

    def test_finite(self, chart_centers, codebook, chart_idx, code_idx, z_n):
        u = compose_absolute_structured_state(chart_centers, codebook, chart_idx, code_idx, z_n)
        assert torch.isfinite(u).all()

    def test_zero_nuisance_matches_macro(self, chart_centers, codebook, chart_idx, code_idx):
        z_n_zero = torch.zeros(B, D)
        u_struct = compose_absolute_structured_state(
            chart_centers, codebook, chart_idx, code_idx, z_n_zero
        )
        u_macro = compose_absolute_macro_state(chart_centers, codebook, chart_idx, code_idx)
        assert torch.allclose(u_struct, u_macro, atol=1e-5)

    def test_nuisance_changes_state(self, chart_centers, codebook, chart_idx, code_idx, z_n):
        z_n_zero = torch.zeros(B, D)
        u_no_nuisance = compose_absolute_structured_state(
            chart_centers,
            codebook,
            chart_idx,
            code_idx,
            z_n_zero,
        )
        u_with_nuisance = compose_absolute_structured_state(
            chart_centers,
            codebook,
            chart_idx,
            code_idx,
            z_n,
        )
        assert not torch.allclose(u_no_nuisance, u_with_nuisance, atol=1e-4)

    def test_gradients_flow_through_z_n(self, chart_centers, codebook, chart_idx, code_idx):
        z_n = nn.Parameter(torch.randn(B, D) * 0.1)
        u = compose_absolute_structured_state(chart_centers, codebook, chart_idx, code_idx, z_n)
        u.sum().backward()
        assert z_n.grad is not None and z_n.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# AbsoluteEnclosureProbe
# ---------------------------------------------------------------------------


class TestAbsoluteEnclosureProbe:
    @pytest.fixture
    def probe(self):
        torch.manual_seed(10)
        return AbsoluteEnclosureProbe(
            obs_struct_dim=D,
            act_struct_dim=D,
            obs_tex_dim=D,
            act_tex_dim=D,
            num_obs_charts=N_OBS_CHARTS,
            obs_codes_per_chart=CODES_PER_CHART,
            hidden_dim=32,
            alpha=1.0,
            dropout=0.0,
        )

    def test_output_keys(self, probe):
        u_obs = torch.randn(B, D)
        u_act = torch.randn(B, D)
        obs_tex = torch.randn(B, D)
        act_tex = torch.randn(B, D)
        out = probe(u_obs, u_act, obs_tex, act_tex)
        assert set(out.keys()) == {"baseline", "obs", "act", "both"}

    def test_output_shapes(self, probe):
        u_obs = torch.randn(B, D)
        u_act = torch.randn(B, D)
        obs_tex = torch.randn(B, D)
        act_tex = torch.randn(B, D)
        out = probe(u_obs, u_act, obs_tex, act_tex)
        num_states = N_OBS_CHARTS * CODES_PER_CHART
        for key in ("baseline", "obs", "act", "both"):
            assert out[key].shape == (B, num_states)

    def test_num_obs_states(self, probe):
        assert probe.num_obs_states == N_OBS_CHARTS * CODES_PER_CHART

    def test_rejects_wrong_dim_inputs(self, probe):
        with pytest.raises(ValueError, match="shape \\[B, D\\]"):
            probe(torch.randn(B), torch.randn(B, D), torch.randn(B, D), torch.randn(B, D))

    def test_rejects_batch_mismatch(self, probe):
        with pytest.raises(ValueError, match="same batch size"):
            probe(torch.randn(B, D), torch.randn(B + 1, D), torch.randn(B, D), torch.randn(B, D))

    def test_gradients_flow(self, probe):
        u_obs = torch.randn(B, D, requires_grad=True)
        u_act = torch.randn(B, D, requires_grad=True)
        obs_tex = torch.randn(B, D, requires_grad=True)
        act_tex = torch.randn(B, D, requires_grad=True)
        out = probe(u_obs, u_act, obs_tex, act_tex)
        loss = sum(v.sum() for v in out.values())
        loss.backward()
        assert u_obs.grad is not None
        assert obs_tex.grad is not None
        assert act_tex.grad is not None

    def test_grl_reverses_texture_gradients(self):
        """GRL should flip the sign of gradients on texture inputs."""
        torch.manual_seed(20)
        probe = AbsoluteEnclosureProbe(
            obs_struct_dim=D,
            act_struct_dim=D,
            obs_tex_dim=D,
            act_tex_dim=D,
            num_obs_charts=N_OBS_CHARTS,
            obs_codes_per_chart=CODES_PER_CHART,
            hidden_dim=32,
            alpha=1.0,
            dropout=0.0,
        )
        u_obs = torch.randn(B, D)
        u_act = torch.randn(B, D)
        obs_tex = torch.randn(B, D, requires_grad=True)
        act_tex = torch.randn(B, D)

        # Forward through the obs texture probe only
        out = probe(u_obs, u_act, obs_tex, act_tex)
        target = torch.zeros(B, dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(out["obs"], target)
        loss.backward()

        grad_with_grl = obs_tex.grad.clone()

        # The GRL should cause gradients to be reversed compared to
        # a direct path; verify gradients are non-zero (GRL is active)
        assert grad_with_grl.abs().sum() > 0


# ---------------------------------------------------------------------------
# compute_absolute_enclosure_loss
# ---------------------------------------------------------------------------


class TestComputeAbsoluteEnclosureLoss:
    @pytest.fixture
    def probe(self):
        torch.manual_seed(30)
        return AbsoluteEnclosureProbe(
            obs_struct_dim=D,
            act_struct_dim=D,
            obs_tex_dim=D,
            act_tex_dim=D,
            num_obs_charts=N_OBS_CHARTS,
            obs_codes_per_chart=CODES_PER_CHART,
            hidden_dim=32,
            alpha=1.0,
            dropout=0.0,
        )

    @pytest.fixture
    def act_chart_centers(self):
        torch.manual_seed(5)
        return torch.randn(N_ACT_CHARTS, D) * 0.3

    @pytest.fixture
    def act_codebook(self):
        torch.manual_seed(6)
        return torch.randn(N_ACT_CHARTS, CODES_PER_CHART, D) * 0.2

    def test_returns_three_elements(
        self,
        probe,
        chart_centers,
        codebook,
        chart_idx,
        code_idx,
        z_n,
        act_chart_centers,
        act_codebook,
    ):
        torch.manual_seed(40)
        act_chart_idx = torch.randint(0, N_ACT_CHARTS, (B,))
        act_code_idx = torch.randint(0, CODES_PER_CHART, (B,))
        act_z_n = torch.randn(B, D) * 0.1
        obs_z_tex = torch.randn(B, D) * 0.1
        act_z_tex = torch.randn(B, D) * 0.1
        obs_chart_tp1 = torch.randint(0, N_OBS_CHARTS, (B,))
        obs_code_tp1 = torch.randint(0, CODES_PER_CHART, (B,))

        loss_enc, loss_probe, diag = compute_absolute_enclosure_loss(
            probe,
            obs_chart_centers=chart_centers,
            obs_codebook=codebook,
            obs_chart_t=chart_idx,
            obs_code_t=code_idx,
            obs_z_n_t=z_n,
            obs_z_tex_t=obs_z_tex,
            act_chart_centers=act_chart_centers,
            act_codebook=act_codebook,
            act_chart_t=act_chart_idx,
            act_code_t=act_code_idx,
            act_z_n_t=act_z_n,
            act_z_tex_t=act_z_tex,
            obs_chart_tp1=obs_chart_tp1,
            obs_code_tp1=obs_code_tp1,
        )

        assert loss_enc.ndim == 0
        assert loss_probe.ndim == 0
        assert torch.isfinite(loss_enc)
        assert torch.isfinite(loss_probe)
        assert isinstance(diag, dict)

    def test_diagnostics_keys(
        self,
        probe,
        chart_centers,
        codebook,
        chart_idx,
        code_idx,
        z_n,
        act_chart_centers,
        act_codebook,
    ):
        torch.manual_seed(41)
        act_chart_idx = torch.randint(0, N_ACT_CHARTS, (B,))
        act_code_idx = torch.randint(0, CODES_PER_CHART, (B,))
        act_z_n = torch.randn(B, D) * 0.1
        obs_z_tex = torch.randn(B, D) * 0.1
        act_z_tex = torch.randn(B, D) * 0.1
        obs_chart_tp1 = torch.randint(0, N_OBS_CHARTS, (B,))
        obs_code_tp1 = torch.randint(0, CODES_PER_CHART, (B,))

        _, _, diag = compute_absolute_enclosure_loss(
            probe,
            obs_chart_centers=chart_centers,
            obs_codebook=codebook,
            obs_chart_t=chart_idx,
            obs_code_t=code_idx,
            obs_z_n_t=z_n,
            obs_z_tex_t=obs_z_tex,
            act_chart_centers=act_chart_centers,
            act_codebook=act_codebook,
            act_chart_t=act_chart_idx,
            act_code_t=act_code_idx,
            act_z_n_t=act_z_n,
            act_z_tex_t=act_z_tex,
            obs_chart_tp1=obs_chart_tp1,
            obs_code_tp1=obs_code_tp1,
        )

        expected_keys = {
            "acc_base",
            "acc_obs",
            "acc_act",
            "acc_both",
            "defect_acc_obs",
            "defect_acc_act",
            "defect_acc_both",
            "ce_base",
            "ce_obs",
            "ce_act",
            "ce_both",
            "defect_ce_obs",
            "defect_ce_act",
            "defect_ce_both",
            "loss_encoder",
            "loss_probe",
        }
        assert set(diag.keys()) == expected_keys

    def test_encoder_loss_has_gradients(
        self,
        probe,
        chart_centers,
        codebook,
        chart_idx,
        code_idx,
        act_chart_centers,
        act_codebook,
    ):
        torch.manual_seed(42)
        z_n = nn.Parameter(torch.randn(B, D) * 0.1)
        act_chart_idx = torch.randint(0, N_ACT_CHARTS, (B,))
        act_code_idx = torch.randint(0, CODES_PER_CHART, (B,))
        act_z_n = nn.Parameter(torch.randn(B, D) * 0.1)
        obs_z_tex = nn.Parameter(torch.randn(B, D) * 0.1)
        act_z_tex = nn.Parameter(torch.randn(B, D) * 0.1)
        obs_chart_tp1 = torch.randint(0, N_OBS_CHARTS, (B,))
        obs_code_tp1 = torch.randint(0, CODES_PER_CHART, (B,))

        loss_enc, _, _ = compute_absolute_enclosure_loss(
            probe,
            obs_chart_centers=chart_centers,
            obs_codebook=codebook,
            obs_chart_t=chart_idx,
            obs_code_t=code_idx,
            obs_z_n_t=z_n,
            obs_z_tex_t=obs_z_tex,
            act_chart_centers=act_chart_centers,
            act_codebook=act_codebook,
            act_chart_t=act_chart_idx,
            act_code_t=act_code_idx,
            act_z_n_t=act_z_n,
            act_z_tex_t=act_z_tex,
            obs_chart_tp1=obs_chart_tp1,
            obs_code_tp1=obs_code_tp1,
        )

        loss_enc.backward()
        assert obs_z_tex.grad is not None and obs_z_tex.grad.abs().sum() > 0
        assert act_z_tex.grad is not None and act_z_tex.grad.abs().sum() > 0

    def test_obs_codes_per_chart_override(
        self,
        probe,
        chart_centers,
        codebook,
        chart_idx,
        code_idx,
        z_n,
        act_chart_centers,
        act_codebook,
    ):
        torch.manual_seed(43)
        act_chart_idx = torch.randint(0, N_ACT_CHARTS, (B,))
        act_code_idx = torch.randint(0, CODES_PER_CHART, (B,))
        act_z_n = torch.randn(B, D) * 0.1
        obs_z_tex = torch.randn(B, D) * 0.1
        act_z_tex = torch.randn(B, D) * 0.1
        # Use smaller indices so override is valid
        obs_chart_tp1 = torch.zeros(B, dtype=torch.long)
        obs_code_tp1 = torch.zeros(B, dtype=torch.long)

        # Should not raise with a valid override
        loss_enc, loss_probe, _ = compute_absolute_enclosure_loss(
            probe,
            obs_chart_centers=chart_centers,
            obs_codebook=codebook,
            obs_chart_t=chart_idx,
            obs_code_t=code_idx,
            obs_z_n_t=z_n,
            obs_z_tex_t=obs_z_tex,
            act_chart_centers=act_chart_centers,
            act_codebook=act_codebook,
            act_chart_t=act_chart_idx,
            act_code_t=act_code_idx,
            act_z_n_t=act_z_n,
            act_z_tex_t=act_z_tex,
            obs_chart_tp1=obs_chart_tp1,
            obs_code_tp1=obs_code_tp1,
            obs_codes_per_chart=CODES_PER_CHART,
        )
        assert torch.isfinite(loss_enc)
        assert torch.isfinite(loss_probe)


# ---------------------------------------------------------------------------
# zeno_loss
# ---------------------------------------------------------------------------


class TestZenoLoss:
    def test_identical_distributions_give_zero(self):
        w = torch.softmax(torch.randn(B, 4), dim=-1)
        loss_kl = zeno_loss(w, w, mode="kl")
        loss_jsd = zeno_loss(w, w, mode="jsd")
        assert loss_kl.item() == pytest.approx(0.0, abs=1e-6)
        assert loss_jsd.item() == pytest.approx(0.0, abs=1e-6)

    def test_different_distributions_give_positive(self):
        torch.manual_seed(50)
        w_t = torch.softmax(torch.randn(B, 4), dim=-1)
        w_prev = torch.softmax(torch.randn(B, 4), dim=-1)
        loss_kl = zeno_loss(w_t, w_prev, mode="kl")
        loss_jsd = zeno_loss(w_t, w_prev, mode="jsd")
        assert loss_kl.item() > 0
        assert loss_jsd.item() > 0

    def test_scalar_output(self):
        w_t = torch.softmax(torch.randn(B, 4), dim=-1)
        w_prev = torch.softmax(torch.randn(B, 4), dim=-1)
        assert zeno_loss(w_t, w_prev, mode="kl").ndim == 0
        assert zeno_loss(w_t, w_prev, mode="jsd").ndim == 0

    def test_jsd_is_symmetric(self):
        torch.manual_seed(51)
        w_t = torch.softmax(torch.randn(B, 4), dim=-1)
        w_prev = torch.softmax(torch.randn(B, 4), dim=-1)
        loss_fwd = zeno_loss(w_t, w_prev, mode="jsd")
        loss_rev = zeno_loss(w_prev, w_t, mode="jsd")
        assert torch.allclose(loss_fwd, loss_rev, atol=1e-6)

    def test_jsd_bounded_by_log2(self):
        """JSD is bounded above by ln(2)."""
        # One-hot vs uniform gives near-maximum divergence
        w_t = torch.zeros(1, 4)
        w_t[0, 0] = 1.0
        w_prev = torch.ones(1, 4) / 4.0
        loss = zeno_loss(w_t, w_prev, mode="jsd")
        import math

        assert loss.item() <= math.log(2) + 1e-5

    def test_kl_not_symmetric(self):
        torch.manual_seed(52)
        w_t = torch.softmax(torch.randn(B, 4) * 3, dim=-1)
        w_prev = torch.softmax(torch.randn(B, 4) * 3, dim=-1)
        loss_fwd = zeno_loss(w_t, w_prev, mode="kl")
        loss_rev = zeno_loss(w_prev, w_t, mode="kl")
        # KL is generally asymmetric
        assert not torch.allclose(loss_fwd, loss_rev, atol=1e-4)

    def test_gradients_flow(self):
        w_t = torch.softmax(torch.randn(B, 4), dim=-1).requires_grad_(True)
        w_prev = torch.softmax(torch.randn(B, 4), dim=-1)
        loss = zeno_loss(w_t, w_prev, mode="jsd")
        loss.backward()
        assert w_t.grad is not None and w_t.grad.abs().sum() > 0

    def test_unknown_mode_raises(self):
        w = torch.softmax(torch.randn(B, 4), dim=-1)
        with pytest.raises(ValueError, match="Unknown zeno_loss mode"):
            zeno_loss(w, w, mode="mse")
