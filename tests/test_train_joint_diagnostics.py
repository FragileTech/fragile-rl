"""Smoke tests for Phase 1 diagnostics surfaced by ``train_joint``."""

from __future__ import annotations

import importlib
import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from fragile.layers.jump_operator import FactorizedJumpOperator
from fragile.layers.topoencoder import TopoEncoder
from fragile.losses.encoder import compute_phase1_loss
from fragile.vla.config import VLAConfig
from fragile.vla.train_joint import _compute_encoder_losses, _eval_pass, _get_hard_routing_tau


train_joint_module = importlib.import_module("fragile.vla.train_joint")


class _FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"feature": self.features[idx]}


def test_eval_pass_reports_router_and_geometry_diagnostics() -> None:
    """Eval diagnostics should expose raw-geometry and router-score health."""
    torch.manual_seed(5)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=5,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    loader = DataLoader(_FeatureDataset(torch.randn(12, 3)), batch_size=4, shuffle=False)

    usage, hard_perp, hard_active, soft_usage, soft_perp, soft_active, mean_r, extra = _eval_pass(
        model,
        loader,
        4,
        torch.device("cpu"),
    )

    assert usage.shape == (4,)
    assert soft_usage.shape == (4,)
    assert hard_perp >= 1.0
    assert soft_perp >= 1.0
    assert 0 <= hard_active <= 4
    assert 0 <= soft_active <= 4
    assert 0.0 <= mean_r <= 0.99 + 1e-6

    for key in (
        "score_gap_p50",
        "score_gap_p90",
        "score_gap_p99",
        "score_std",
        "score_mean_abs",
        "soft_equiv_log_ratio",
        "v_boundary_frac",
        "v_local_clip_frac",
        "z_geo_clip_frac",
        "v_raw_r_p99",
        "v_local_raw_r_p99",
        "z_geo_raw_r_p99",
        "vq_dist_p90",
        "vq_dist_p99",
        "cb_raw_r_p99",
        "cc_raw_r_p99",
        "cb_clip_frac",
        "cc_clip_frac",
    ):
        assert key in extra
        assert math.isfinite(extra[key])

    assert extra["score_gap_p99"] >= extra["score_gap_p90"] >= extra["score_gap_p50"] >= 0.0
    assert extra["vq_dist_p99"] >= extra["vq_dist_p90"] >= 0.0
    assert 0.0 <= extra["v_boundary_frac"] <= 1.0
    assert 0.0 <= extra["v_local_clip_frac"] <= 1.0
    assert 0.0 <= extra["z_geo_clip_frac"] <= 1.0
    assert 0.0 <= extra["cb_clip_frac"] <= 1.0
    assert 0.0 <= extra["cc_clip_frac"] <= 1.0


def test_eval_pass_uses_deterministic_hard_routing_for_phase1_diagnostics(
    monkeypatch,
) -> None:
    """Phase-1 eval should log deterministic hard-routing preferences, not Gumbel noise."""
    torch.manual_seed(11)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=5,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    loader = DataLoader(_FeatureDataset(torch.randn(8, 3)), batch_size=4, shuffle=False)
    calls: list[float] = []
    original_forward = model.encoder.forward

    def _wrapped_forward(*args, **kwargs):
        calls.append(float(kwargs.get("routing_tau", 0.0)))
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model.encoder, "forward", _wrapped_forward)

    _eval_pass(
        model,
        loader,
        4,
        torch.device("cpu"),
        hard_routing=True,
        hard_routing_tau=0.5,
    )

    assert calls
    assert all(tau == -1.0 for tau in calls)


def test_negative_hard_routing_tau_stays_deterministic() -> None:
    """Deterministic hard routing should not silently anneal toward positive tau."""
    args = SimpleNamespace(
        hard_routing_tau=-1.0,
        hard_routing_tau_end=0.3,
        hard_routing_tau_anneal_epochs=200,
    )
    assert _get_hard_routing_tau(args, epoch=0, total_epochs=1000) == -1.0
    assert _get_hard_routing_tau(args, epoch=150, total_epochs=1000) == -1.0


def test_phase1_loss_chart_usage_follows_deterministic_usage_weights() -> None:
    """Utilization loss should penalize deterministic preference collapse, not sampled balance."""
    torch.manual_seed(13)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=3,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    config = VLAConfig(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=3,
        w_feature_recon=1.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=1.0,
        w_chart_ot=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_confidence_calibration=0.0,
        w_v_tangent_barrier=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_center_mean=0.0,
        w_chart_center_radius=0.0,
        w_chart_center_sep=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
        radial_vq_alpha=0.0,
    )
    batch = 8
    x = torch.randn(batch, 3)
    x_recon = x.clone()
    enc_router_weights = torch.full((batch, 4), 0.25)
    dec_router_weights = enc_router_weights.clone()
    z_geo = torch.zeros(batch, 2)
    balanced_usage = F.one_hot(torch.arange(batch) % 4, num_classes=4).float()
    collapsed_usage = F.one_hot(torch.zeros(batch, dtype=torch.long), num_classes=4).float()

    _, _, balanced_metrics = compute_phase1_loss(
        x,
        x_recon,
        torch.tensor(0.0),
        enc_router_weights,
        dec_router_weights,
        z_geo,
        model,
        config,
        router_reg_weights=enc_router_weights,
        usage_router_weights=balanced_usage,
    )
    _, _, collapsed_metrics = compute_phase1_loss(
        x,
        x_recon,
        torch.tensor(0.0),
        enc_router_weights,
        dec_router_weights,
        z_geo,
        model,
        config,
        router_reg_weights=enc_router_weights,
        usage_router_weights=collapsed_usage,
    )

    assert collapsed_metrics["chart_usage"] > balanced_metrics["chart_usage"]


def test_covariant_router_temperature_scales_with_latent_geometry() -> None:
    """Hyperbolic router temperature should use latent, not hidden, dimension."""
    torch.manual_seed(19)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=3,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    z = torch.tensor([[0.1, 0.2], [0.3, 0.0]], dtype=torch.float32)
    tau = model.encoder.cov_router._temperature(z)
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=model.encoder.cov_router.tau_denom_min)
    expected = (math.sqrt(model.encoder.latent_dim) * denom / 2.0).clamp(
        min=model.encoder.cov_router.tau_min,
    )
    torch.testing.assert_close(tau, expected)


def test_phase1_loss_skips_jump_consistency_when_weight_is_zero(monkeypatch) -> None:
    """Zero jump weight should bypass the expensive overlap-consistency path."""
    torch.manual_seed(7)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=3,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    jump_op = FactorizedJumpOperator(num_charts=4, latent_dim=2)
    x = torch.randn(8, 3)
    phase1_config = VLAConfig(
        num_charts=4,
        codes_per_chart=3,
        w_feature_recon=1.0,
        w_vq=1.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_chart_ot=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_v_tangent_barrier=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_center_mean=0.0,
        w_chart_center_radius=0.0,
        w_chart_center_sep=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
    )
    args = SimpleNamespace(
        w_jump=0.0,
        w_jump_warmup=20,
        w_jump_ramp_end=50,
        w_perp=0.01,
    )

    def _should_not_run(*_args, **_kwargs):
        msg = "jump consistency should be skipped when its weight is zero"
        raise AssertionError(msg)

    monkeypatch.setattr(train_joint_module, "compute_jump_consistency_loss", _should_not_run)

    base_loss, zn_reg_loss, metrics, *_ = _compute_encoder_losses(
        x,
        model,
        jump_op,
        args,
        epoch=0,
        phase1_config=phase1_config,
    )

    assert math.isfinite(base_loss.item())
    assert math.isfinite(zn_reg_loss.item())
    assert metrics["jump"] == 0.0
