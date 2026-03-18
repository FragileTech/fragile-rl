"""Golden-value regression tests for fragile.losses.encoder.

Every test uses a single deterministic fixture (seed=42) and asserts against
hardcoded golden values. Any refactoring that changes numerical output will
be caught immediately.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import FactorizedJumpOperator
from fragile.losses.encoder import (
    _deterministic_st_router_weights,
    combine_quality_targets,
    compute_chart_center_mean_loss,
    compute_chart_center_radius_loss,
    compute_chart_center_separation_loss,
    compute_chart_usage_band_loss,
    compute_code_usage_band_loss,
    compute_codebook_centering_loss,
    compute_codebook_spread_loss,
    compute_confidence_calibration_loss,
    compute_error_quality_targets,
    compute_hard_routing_nll,
    compute_hyperbolic_uniformity_loss,
    compute_jump_consistency_loss,
    compute_phase1_loss,
    compute_radial_calibration_loss,
    compute_rank_quality_targets,
    compute_router_information_metrics,
    compute_router_margin_loss,
    compute_router_score_metrics,
    compute_router_sharpness_metrics,
    compute_routing_confidence,
    compute_routing_entropy,
    compute_sinkhorn_balanced_chart_loss,
    compute_v_tangent_barrier_loss,
    compute_window_loss,
    get_jump_weight_schedule,
    mix_quality_targets,
    orthogonality_loss,
)
from fragile.vla.config import VLAConfig


B, K, C, D = 8, 4, 6, 3


@pytest.fixture()
def t():
    """Master fixture: deterministic tensors from seed 42."""
    torch.manual_seed(42)
    router_scores = torch.randn(B, K)
    router_weights = F.softmax(router_scores, dim=-1)
    z_geo = torch.randn(B, D) * 0.3
    v_raw = torch.randn(B, D) * 1.5
    codebook = torch.randn(K, C, D) * 0.2
    chart_centers = torch.randn(K, D) * 0.3
    v_local = torch.randn(B, D) * 0.3
    hard_code_indices = torch.randint(0, C, (B, K))
    per_sample_error = torch.rand(B) * 2.0
    quality_target = torch.rand(B)
    quality_secondary = torch.rand(B)
    z_n_all_charts = torch.randn(B, K, D) * 0.3
    zn = torch.randn(B, D)
    ztex = torch.randn(B, D)
    center_points = torch.randn(B, D) * 0.2

    return SimpleNamespace(
        router_scores=router_scores,
        router_weights=router_weights,
        z_geo=z_geo,
        v_raw=v_raw,
        codebook=codebook,
        chart_centers=chart_centers,
        v_local=v_local,
        hard_code_indices=hard_code_indices,
        per_sample_error=per_sample_error,
        quality_target=quality_target,
        quality_secondary=quality_secondary,
        z_n_all_charts=z_n_all_charts,
        zn=zn,
        ztex=ztex,
        center_points=center_points,
    )


# =========================================================================
# Routing Losses
# =========================================================================


class TestRoutingLosses:
    def test_routing_entropy(self, t):
        val = compute_routing_entropy(t.router_weights)
        assert val.shape == ()
        torch.testing.assert_close(val, torch.tensor(1.0981807708740234), atol=1e-6, rtol=0)

    def test_router_margin_loss(self, t):
        val = compute_router_margin_loss(t.router_scores, margin=0.05)
        assert val.shape == ()
        torch.testing.assert_close(val, torch.tensor(0.0014606327749788761), atol=1e-6, rtol=0)

    def test_hard_routing_nll(self, t):
        val = compute_hard_routing_nll(t.router_scores)
        assert val.shape == ()
        torch.testing.assert_close(val, torch.tensor(0.6463645100593567), atol=1e-6, rtol=0)

    def test_routing_confidence(self, t):
        val = compute_routing_confidence(t.router_weights, K)
        assert val.shape == (B,)
        expected = torch.tensor([
            0.2347339391708374,
            0.24239236116409302,
            0.47214311361312866,
            0.18329721689224243,
            0.25382161140441895,
            0.16288399696350098,
            0.02507007122039795,
            0.0882977843284607,
        ])
        torch.testing.assert_close(val, expected, atol=1e-6, rtol=0)


# =========================================================================
# Routing Metrics
# =========================================================================


class TestRoutingMetrics:
    def test_router_information_metrics(self, t):
        d = compute_router_information_metrics(t.router_weights)
        torch.testing.assert_close(d["H_K"], torch.tensor(1.3603382110595703), atol=1e-6, rtol=0)
        torch.testing.assert_close(
            d["H_K_given_X"],
            torch.tensor(1.0981807708740234),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(d["I_XK"], torch.tensor(0.2621574401855469), atol=1e-6, rtol=0)

    def test_router_sharpness_metrics(self, t):
        d = compute_router_sharpness_metrics(t.router_weights)
        torch.testing.assert_close(
            d["top1_prob_mean"],
            torch.tensor(0.5418049097061157),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["top1_prob_p10"],
            torch.tensor(0.4207893908023834),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["top1_prob_p90"],
            torch.tensor(0.6788233518600464),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["top2_prob_mean"],
            torch.tensor(0.22593805193901062),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["top1_gap_mean"],
            torch.tensor(0.3158667981624603),
            atol=1e-6,
            rtol=0,
        )

    def test_router_score_metrics(self, t):
        d = compute_router_score_metrics(t.router_scores)
        torch.testing.assert_close(
            d["score_gap_mean"],
            torch.tensor(0.8982048630714417),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["score_gap_p50"],
            torch.tensor(0.7697924375534058),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["score_gap_p90"],
            torch.tensor(1.5376731157302856),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["score_gap_p99"],
            torch.tensor(1.9908490180969238),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["score_std"],
            torch.tensor(1.0654828548431396),
            atol=1e-6,
            rtol=0,
        )
        torch.testing.assert_close(
            d["score_mean_abs"],
            torch.tensor(0.9362192153930664),
            atol=1e-6,
            rtol=0,
        )


# =========================================================================
# Chart Balancing
# =========================================================================


class TestChartBalancing:
    def test_chart_usage_band_loss(self, t):
        loss, metrics = compute_chart_usage_band_loss(t.router_weights, K)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-5, rtol=0)
        assert abs(metrics["H_usage"] - 1.3603382110595703) < 1e-5
        assert metrics["usage_active"] == 4

    def test_sinkhorn_balanced_chart_loss(self, t):
        loss, metrics = compute_sinkhorn_balanced_chart_loss(t.router_scores)
        torch.testing.assert_close(loss, torch.tensor(0.853971004486084), atol=1e-4, rtol=0)
        assert abs(metrics["ot_target_top1_mean"] - 0.9430776834487915) < 1e-4

    def test_code_usage_band_loss(self, t):
        loss, metrics = compute_code_usage_band_loss(
            t.v_local,
            t.codebook,
            t.router_weights,
            hard_code_indices=t.hard_code_indices,
        )
        torch.testing.assert_close(loss, torch.tensor(0.2044019103050232), atol=1e-5, rtol=0)
        assert abs(metrics["H_code_usage"] - 1.1067204475402832) < 1e-5
        assert metrics["active_code_charts"] == 4

    def test_window_loss(self, t):
        loss, metrics = compute_window_loss(t.router_weights)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=0)
        assert abs(metrics["I_XK"] - 0.2621574401855469) < 1e-6


# =========================================================================
# Geometry Losses
# =========================================================================


class TestGeometryLosses:
    def test_hyperbolic_uniformity_loss(self, t):
        val = compute_hyperbolic_uniformity_loss(t.z_geo)
        torch.testing.assert_close(val, torch.tensor(-0.9990041255950928), atol=1e-5, rtol=0)

    def test_radial_calibration_euclidean(self, t):
        val = compute_radial_calibration_loss(
            t.z_geo,
            t.router_weights,
            K,
            use_hyperbolic_radius=False,
        )
        torch.testing.assert_close(val, torch.tensor(0.09300332516431808), atol=1e-5, rtol=0)

    def test_radial_calibration_hyperbolic_centers(self, t):
        val = compute_radial_calibration_loss(
            t.z_geo,
            t.router_weights,
            K,
            center_points=t.center_points,
            use_hyperbolic_radius=True,
            rho_max=4.0,
            rho_band_width=0.75,
        )
        torch.testing.assert_close(val, torch.tensor(0.00452558696269989), atol=1e-5, rtol=0)

    def test_v_tangent_barrier_loss(self, t):
        val = compute_v_tangent_barrier_loss(t.v_raw, target_radius=0.9)
        torch.testing.assert_close(val, torch.tensor(1.2863603830337524), atol=1e-6, rtol=0)

    def test_confidence_calibration_loss(self, t):
        val = compute_confidence_calibration_loss(t.router_weights, t.quality_target, K)
        torch.testing.assert_close(val, torch.tensor(0.13471870124340057), atol=1e-6, rtol=0)


# =========================================================================
# Codebook Losses
# =========================================================================


class TestCodebookLosses:
    def test_codebook_spread_loss(self, t):
        val = compute_codebook_spread_loss(t.codebook, margin=1.0)
        torch.testing.assert_close(val, torch.tensor(0.1928616315126419), atol=1e-5, rtol=0)

    def test_codebook_centering_loss(self, t):
        val = compute_codebook_centering_loss(t.codebook)
        torch.testing.assert_close(val, torch.tensor(0.02306784689426422), atol=1e-5, rtol=0)


# =========================================================================
# Chart Center Losses
# =========================================================================


class TestChartCenterLosses:
    def test_chart_center_mean_loss(self, t):
        val = compute_chart_center_mean_loss(t.chart_centers)
        torch.testing.assert_close(val, torch.tensor(0.10007878392934799), atol=1e-5, rtol=0)

    def test_chart_center_radius_loss(self, t):
        val = compute_chart_center_radius_loss(t.chart_centers, radius_max=2.0)
        torch.testing.assert_close(val, torch.tensor(0.0), atol=1e-5, rtol=0)

    def test_chart_center_separation_loss(self, t):
        val = compute_chart_center_separation_loss(t.chart_centers, margin=1.0)
        torch.testing.assert_close(val, torch.tensor(0.04025062173604965), atol=1e-5, rtol=0)


# =========================================================================
# Quality Targets
# =========================================================================


class TestQualityTargets:
    def test_error_quality_targets(self, t):
        val = compute_error_quality_targets(t.per_sample_error, alpha=2.0)
        expected = torch.tensor([
            0.3617447316646576,
            0.0724329873919487,
            0.14970622956752777,
            0.1457151174545288,
            0.052932921797037125,
            0.10610686242580414,
            0.15460175275802612,
            0.2267364263534546,
        ])
        torch.testing.assert_close(val, expected, atol=1e-6, rtol=0)

    def test_rank_quality_targets(self, t):
        val = compute_rank_quality_targets(t.per_sample_error)
        expected = torch.tensor([
            1.0,
            0.1428571343421936,
            0.5714285373687744,
            0.4285714030265808,
            0.0,
            0.2857142686843872,
            0.7142857313156128,
            0.8571428656578064,
        ])
        torch.testing.assert_close(val, expected, atol=1e-6, rtol=0)

    def test_mix_quality_targets(self, t):
        abs_q = compute_error_quality_targets(t.per_sample_error, alpha=2.0)
        rank_q = compute_rank_quality_targets(t.per_sample_error)
        val = mix_quality_targets(abs_q, rank_q, rank_mix=0.75)
        expected = torch.tensor([
            0.8404361605644226,
            0.12525109946727753,
            0.46599796414375305,
            0.3578573167324066,
            0.013233230449259281,
            0.24081242084503174,
            0.5743647217750549,
            0.6995412111282349,
        ])
        torch.testing.assert_close(val, expected, atol=1e-6, rtol=0)

    def test_combine_quality_targets(self, t):
        val = combine_quality_targets(
            t.quality_target,
            t.quality_secondary,
            primary_weight=0.7,
        )
        expected = torch.tensor([
            0.0851704552769661,
            0.8530129194259644,
            0.5023209452629089,
            0.6322053670883179,
            0.47912561893463135,
            0.6763246059417725,
            0.8033208250999451,
            0.6923861503601074,
        ])
        torch.testing.assert_close(val, expected, atol=1e-6, rtol=0)


# =========================================================================
# Misc
# =========================================================================


class TestMisc:
    def test_orthogonality_loss_same_dim(self, t):
        val = orthogonality_loss(t.zn, t.ztex)
        torch.testing.assert_close(val, torch.tensor(0.21001170575618744), atol=1e-6, rtol=0)

    def test_orthogonality_loss_diff_dim(self, t):
        val = orthogonality_loss(t.zn[:, :2], t.ztex)
        torch.testing.assert_close(val, torch.tensor(0.1572081595659256), atol=1e-6, rtol=0)

    def test_deterministic_st_router_weights(self, t):
        val = _deterministic_st_router_weights(t.router_scores)
        assert val.shape == (B, K)
        # Forward pass uses one-hot; check argmax positions
        assert (val.argmax(dim=-1) == t.router_scores.argmax(dim=-1)).all()
        # Check values are close to one-hot (ST forward)
        torch.testing.assert_close(val.sum(dim=-1), torch.ones(B), atol=1e-6, rtol=0)

    def test_get_jump_weight_schedule(self):
        assert get_jump_weight_schedule(0, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.0
        assert get_jump_weight_schedule(25, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.0
        assert get_jump_weight_schedule(49, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.0
        assert get_jump_weight_schedule(50, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.01
        assert (
            abs(
                get_jump_weight_schedule(75, warmup_end=50, ramp_end=100, final_weight=0.1) - 0.055
            )
            < 1e-12
        )
        assert get_jump_weight_schedule(100, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.1
        assert get_jump_weight_schedule(150, warmup_end=50, ramp_end=100, final_weight=0.1) == 0.1


# =========================================================================
# Jump Consistency
# =========================================================================


class TestJumpConsistency:
    def test_jump_consistency_loss(self):
        torch.manual_seed(42)
        jump_op = FactorizedJumpOperator(num_charts=4, latent_dim=3)
        torch.manual_seed(99)
        jc_z = torch.randn(B, K, D) * 0.3
        jc_rw = F.softmax(torch.randn(B, K), dim=-1)
        val = compute_jump_consistency_loss(jump_op, jc_z, jc_rw)
        torch.testing.assert_close(val, torch.tensor(7.228351593017578), atol=1e-4, rtol=0)


# =========================================================================
# Phase 1 Loss
# =========================================================================


class _MockAtlasEncoder(nn.Module):
    """Minimal encoder with codebook + chart_centers for compute_phase1_loss."""

    def __init__(self, codebook, chart_centers, **cached):
        super().__init__()
        self.codebook = nn.Parameter(codebook)
        self.chart_centers = nn.Parameter(chart_centers)
        for attr, val in cached.items():
            setattr(self, attr, val)


class _MockWrapper(nn.Module):
    def __init__(self, enc):
        super().__init__()
        self.encoder = enc


class TestPhase1Loss:
    @pytest.fixture()
    def phase1_setup(self, t):
        """Build mock encoder + wrapper + synthetic x/x_recon for Phase 1."""
        # Use a fresh seed for x/x_recon so they're deterministic but separate
        # from the main fixture (which already consumed seed 42)
        torch.manual_seed(42)
        # Re-generate all fixture tensors to get same state
        rs = torch.randn(B, K)
        _rw = F.softmax(rs, dim=-1)
        _zg = torch.randn(B, D) * 0.3
        _vr = torch.randn(B, D) * 1.5
        _cb = torch.randn(K, C, D) * 0.2
        _cc = torch.randn(K, D) * 0.3
        _vl = torch.randn(B, D) * 0.3
        _hi = torch.randint(0, C, (B, K))
        _pe = torch.rand(B) * 2.0
        _qt = torch.rand(B)
        _qs = torch.rand(B)
        _zn = torch.randn(B, K, D) * 0.3
        _z1 = torch.randn(B, D)
        _z2 = torch.randn(B, D)
        _cp = torch.randn(B, D) * 0.2
        # Now we're at the same RNG state after the fixture
        x = torch.randn(B, D)
        x_recon = x + torch.randn(B, D) * 0.1
        vq_loss = torch.tensor(0.5)

        enc = _MockAtlasEncoder(
            t.codebook.clone(),
            t.chart_centers.clone(),
            _last_v_raw=t.v_raw,
        )
        wrapper = _MockWrapper(enc)
        return SimpleNamespace(
            x=x,
            x_recon=x_recon,
            vq_loss=vq_loss,
            wrapper=wrapper,
            t=t,
        )

    def test_all_terms(self, phase1_setup):
        s = phase1_setup
        t = s.t
        cfg = VLAConfig(
            input_dim=D,
            hidden_dim=16,
            latent_dim=D,
            num_charts=K,
            codes_per_chart=C,
            w_feature_recon=1.0,
            w_vq=1.0,
            w_entropy=0.3,
            w_diversity=1.0,
            w_chart_ot=1.0,
            w_uniformity=0.05,
            w_radial_calibration=0.1,
            w_confidence_calibration=0.05,
            w_hard_routing_nll=0.5,
            w_router_margin=2.0,
            w_v_tangent_barrier=0.01,
            w_codebook_spread=0.05,
            w_codebook_center=0.02,
            w_chart_center_mean=0.02,
            w_chart_center_radius=0.05,
            w_chart_center_sep=0.02,
            w_code_collapse=0.5,
            w_window=0.0,
            w_consistency=0.0,
            w_perp=0.01,
            radial_quality_mix=1.0,
            radial_quality_base_weight=0.0,
        )
        base, zn_reg, metrics = compute_phase1_loss(
            s.x,
            s.x_recon,
            s.vq_loss,
            t.router_weights,
            t.router_weights,
            t.z_geo,
            s.wrapper,
            cfg,
            router_reg_weights=t.router_weights,
            usage_router_weights=t.router_weights,
            c_bar=t.center_points,
            v_local=t.v_local,
            indices_stack=t.hard_code_indices,
            router_scores=t.router_scores,
        )
        torch.testing.assert_close(base, torch.tensor(2.1497552394866943), atol=1e-4, rtol=0)
        torch.testing.assert_close(zn_reg, torch.tensor(-0.03226804733276367), atol=1e-4, rtol=0)
        # Spot-check key metrics
        assert abs(metrics["recon"] - 0.007349178660660982) < 1e-5
        assert abs(metrics["entropy"] - 1.0981807708740234) < 1e-5
        assert abs(metrics["uniformity"] - (-0.9990041255950928)) < 1e-4
        assert abs(metrics["chart_ot"] - 0.853971004486084) < 1e-4
        assert abs(metrics["codebook_spread"] - 0.1928616315126419) < 1e-4

    def test_recon_only(self, phase1_setup):
        s = phase1_setup
        t = s.t
        cfg = VLAConfig(
            input_dim=D,
            hidden_dim=16,
            latent_dim=D,
            num_charts=K,
            codes_per_chart=C,
            w_feature_recon=1.0,
            w_vq=0.0,
            w_entropy=0.0,
            w_diversity=0.0,
            w_chart_ot=0.0,
            w_uniformity=0.0,
            w_radial_calibration=0.0,
            w_confidence_calibration=0.0,
            w_hard_routing_nll=0.0,
            w_router_margin=0.0,
            w_v_tangent_barrier=0.0,
            w_codebook_spread=0.0,
            w_codebook_center=0.0,
            w_chart_center_mean=0.0,
            w_chart_center_radius=0.0,
            w_chart_center_sep=0.0,
            w_code_collapse=0.0,
            w_window=0.0,
            w_consistency=0.0,
            w_perp=0.0,
        )
        base, zn_reg, metrics = compute_phase1_loss(
            s.x,
            s.x_recon,
            s.vq_loss,
            t.router_weights,
            t.router_weights,
            t.z_geo,
            s.wrapper,
            cfg,
            router_reg_weights=t.router_weights,
            usage_router_weights=t.router_weights,
            c_bar=t.center_points,
            v_local=t.v_local,
            indices_stack=t.hard_code_indices,
            router_scores=t.router_scores,
        )
        torch.testing.assert_close(base, torch.tensor(0.007349178660660982), atol=1e-5, rtol=0)
        torch.testing.assert_close(zn_reg, torch.tensor(0.0), atol=1e-7, rtol=0)
        assert abs(metrics["recon"] - 0.007349178660660982) < 1e-5
        assert abs(metrics["total"] - 0.007349178660660982) < 1e-5
