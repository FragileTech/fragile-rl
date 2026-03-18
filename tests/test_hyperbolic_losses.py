"""Tests for router and code-utilization losses used by VLA training."""

import math

import torch
from torch import nn

from fragile.layers import FactorizedJumpOperator, TopoEncoder
from fragile.layers.gauge import smooth_tangent_to_ball
from fragile.losses.encoder import (
    combine_quality_targets,
    compute_chart_center_mean_loss,
    compute_chart_center_radius_loss,
    compute_chart_center_separation_loss,
    compute_chart_usage_band_loss,
    compute_code_usage_band_loss,
    compute_confidence_calibration_loss,
    compute_error_quality_targets,
    compute_hard_routing_nll,
    compute_phase1_loss,
    compute_radial_calibration_loss,
    compute_rank_quality_targets,
    compute_router_margin_loss,
    compute_router_score_metrics,
    get_jump_weight_schedule,
    mix_quality_targets,
)
from fragile.vla.config import VLAConfig
from fragile.vla.optim import build_encoder_param_groups


def test_chart_usage_band_loss_prefers_balanced_hard_assignments() -> None:
    """Balanced hard chart occupancy should beat collapsed occupancy."""
    balanced = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    collapsed = torch.tensor(
        [[1.0, 0.0]] * 4,
        dtype=torch.float32,
        requires_grad=True,
    )

    loss_balanced, _ = compute_chart_usage_band_loss(
        balanced,
        num_charts=2,
        h_low=0.5,
    )
    loss_collapsed, _ = compute_chart_usage_band_loss(
        collapsed,
        num_charts=2,
        h_low=0.5,
    )

    assert loss_balanced.item() < loss_collapsed.item()

    loss_collapsed.backward()
    assert collapsed.grad is not None
    assert collapsed.grad.abs().sum().item() > 0.0


def test_router_margin_loss_penalizes_flat_hard_partitions() -> None:
    """A hard partition with zero score margin should incur positive loss."""
    flat_scores = torch.zeros(4, 3, requires_grad=True)
    loss = compute_router_margin_loss(flat_scores, margin=0.05)
    assert loss.item() > 0.0
    loss.backward()
    assert flat_scores.grad is not None
    assert flat_scores.grad.abs().sum().item() > 0.0


def test_hard_routing_nll_penalizes_flat_score_partitions() -> None:
    """The deterministic hard partition should have nonzero NLL when scores are flat."""
    flat_scores = torch.zeros(4, 3, requires_grad=True)
    loss = compute_hard_routing_nll(flat_scores)
    assert loss.item() > 0.0
    loss.backward()
    assert flat_scores.grad is not None
    assert flat_scores.grad.abs().sum().item() > 0.0


def test_code_usage_band_loss_prefers_per_chart_code_utilization() -> None:
    """Balanced code usage within each chart should beat collapsed usage."""
    codebook = torch.tensor(
        [
            [[-0.20, 0.00], [0.20, 0.00]],
            [[-0.20, 0.00], [0.20, 0.00]],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    router_weights = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    v_local_balanced = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    v_local_collapsed = torch.tensor(
        [
            [-0.20, 0.00],
            [-0.20, 0.00],
            [-0.20, 0.00],
            [-0.20, 0.00],
        ],
        dtype=torch.float32,
    )

    loss_balanced, _ = compute_code_usage_band_loss(
        v_local_balanced,
        codebook,
        router_weights,
        temperature=0.05,
        h_low=math.log(1.5),
    )
    loss_collapsed, _ = compute_code_usage_band_loss(
        v_local_collapsed,
        codebook,
        router_weights,
        temperature=0.05,
        h_low=math.log(1.5),
    )

    assert loss_balanced.item() < loss_collapsed.item()

    loss_collapsed.backward()
    assert codebook.grad is not None
    assert codebook.grad.abs().sum().item() > 0.0


def test_code_usage_band_loss_respects_explicit_hard_indices() -> None:
    """Forward occupancy should follow the encoder's chosen code indices."""
    codebook = torch.tensor(
        [[[0.0, 0.0], [0.3, 0.0]]],
        dtype=torch.float32,
        requires_grad=True,
    )
    router_weights = torch.tensor([[1.0], [1.0]], dtype=torch.float32)
    v_local = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32)

    loss_default, _ = compute_code_usage_band_loss(
        v_local,
        codebook,
        router_weights,
        temperature=0.05,
        h_low=math.log(1.5),
    )
    explicit_indices = torch.tensor([[0], [1]], dtype=torch.long)
    loss_explicit, _ = compute_code_usage_band_loss(
        v_local,
        codebook,
        router_weights,
        hard_code_indices=explicit_indices,
        temperature=0.05,
        h_low=math.log(1.5),
    )

    assert loss_explicit.item() < loss_default.item()


def test_encoder_exposes_live_soft_router_weights() -> None:
    """Encoder should expose soft probabilities for loss terms."""
    torch.manual_seed(13)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=5,
    )
    x = torch.randn(8, 3)
    _, _, _enc_weights, _, _, _, _, _, _ = model(x)

    soft_live = model.encoder._last_soft_router_weights_live

    assert soft_live.shape == (8, 4)
    assert soft_live.requires_grad
    assert torch.allclose(soft_live.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert ((soft_live > 0.0) & (soft_live < 1.0)).any()


def test_chart_center_geometry_losses_respect_mean_radius_and_separation() -> None:
    """Chart-center priors should anchor, confine, and separate the atlas."""
    centered = torch.tensor(
        [[-0.20, 0.00], [0.20, 0.00]],
        dtype=torch.float32,
    )
    shifted = torch.tensor(
        [[0.20, 0.00], [0.30, 0.00]],
        dtype=torch.float32,
        requires_grad=True,
    )
    safe = torch.tensor(
        [[-0.15, 0.00], [0.15, 0.00]],
        dtype=torch.float32,
    )
    boundary = torch.tensor(
        [[-0.75, 0.00], [0.75, 0.00]],
        dtype=torch.float32,
        requires_grad=True,
    )
    separated = torch.tensor(
        [[-0.40, 0.00], [0.40, 0.00]],
        dtype=torch.float32,
    )
    collapsed = torch.tensor(
        [[0.10, 0.00], [0.12, 0.00]],
        dtype=torch.float32,
        requires_grad=True,
    )

    mean_centered = compute_chart_center_mean_loss(centered)
    mean_shifted = compute_chart_center_mean_loss(shifted)
    radius_safe = compute_chart_center_radius_loss(safe, radius_max=1.0)
    radius_boundary = compute_chart_center_radius_loss(boundary, radius_max=1.0)
    sep_far = compute_chart_center_separation_loss(separated, margin=1.0)
    sep_collapsed = compute_chart_center_separation_loss(collapsed, margin=1.0)

    assert mean_centered.item() < 1e-6
    assert mean_shifted.item() > mean_centered.item()
    assert radius_safe.item() < 1e-6
    assert radius_boundary.item() > 0.0
    assert sep_far.item() < sep_collapsed.item()

    total = mean_shifted + radius_boundary + sep_collapsed
    total.backward()

    assert shifted.grad is not None
    assert shifted.grad.abs().sum().item() > 0.0
    assert boundary.grad is not None
    assert boundary.grad.abs().sum().item() > 0.0
    assert collapsed.grad is not None
    assert collapsed.grad.abs().sum().item() > 0.0


def test_encoder_optimizer_groups_split_chart_centers_and_codebooks() -> None:
    """Atlas anchors and codebooks should use their slower LR groups."""
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=5,
    )
    jump_op = FactorizedJumpOperator(num_charts=4, latent_dim=2)

    groups = build_encoder_param_groups(
        model,
        jump_op,
        base_lr=1e-3,
        lr_chart_centers_scale=0.1,
        lr_codebook_scale=0.5,
    )
    ids_by_lr = {group["lr"]: {id(param) for param in group["params"]} for group in groups}

    assert len(groups) == 3
    assert id(model.encoder.chart_centers) in ids_by_lr[1e-4]
    assert id(model.encoder.codebook) in ids_by_lr[5e-4]
    assert id(next(iter(jump_op.parameters()))) in ids_by_lr[1e-3]

    grouped_ids = set().union(*ids_by_lr.values())
    expected_ids = {id(param) for param in model.parameters() if param.requires_grad}
    expected_ids.update(id(param) for param in jump_op.parameters() if param.requires_grad)
    assert grouped_ids == expected_ids


def test_primitive_encoder_caches_v_local_for_phase1_losses() -> None:
    """Phase 1 helpers should be able to reuse the exact chart-local latent."""
    torch.manual_seed(7)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=4,
        codes_per_chart=5,
    )
    x = torch.randn(6, 3)
    model(x)

    assert hasattr(model.encoder, "_last_v_local")
    assert model.encoder._last_v_local.shape == (6, 2)
    assert hasattr(model.encoder, "_last_v_raw")
    assert hasattr(model.encoder, "_last_v_projected")
    assert hasattr(model.encoder, "_last_v_local_raw")
    assert hasattr(model.encoder, "_last_z_geo_raw")
    assert hasattr(model.encoder, "_last_c_bar")
    assert model.encoder._last_v_raw.shape == (6, 2)
    assert model.encoder._last_v_projected.shape == (6, 2)
    assert model.encoder._last_v_local_raw.shape == (6, 2)
    assert model.encoder._last_z_geo_raw.shape == (6, 2)
    assert model.encoder._last_c_bar.shape == (6, 2)
    assert torch.all(model.encoder._last_v_projected.norm(dim=-1) <= 0.99 + 1e-6)
    assert torch.all(model.encoder._last_v_local.norm(dim=-1) <= 0.99 + 1e-6)


class _DummyAtlasEncoder(nn.Module):
    def __init__(
        self,
        codebook: torch.Tensor,
        *,
        router_reg_weights: torch.Tensor | None = None,
        c_bar: torch.Tensor | None = None,
        v_local: torch.Tensor | None = None,
        v_raw: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.codebook = nn.Parameter(codebook)
        if router_reg_weights is not None:
            self._last_soft_router_weights_live = router_reg_weights
        if c_bar is not None:
            self._last_c_bar = c_bar
        if v_local is not None:
            self._last_v_local = v_local
        if v_raw is not None:
            self._last_v_raw = v_raw


class _DummyWrapper(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder


def test_smooth_tangent_to_ball_is_bounded_and_monotone() -> None:
    """Smooth squashing should stay inside the ball without hard clipping."""
    raw = torch.tensor(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [2.0, 0.0],
            [20.0, 0.0],
        ],
        dtype=torch.float32,
    )

    squashed = smooth_tangent_to_ball(raw, max_norm=0.99)
    radii = squashed.norm(dim=-1)

    assert torch.all(radii < 0.99)
    assert torch.all(radii[1:] >= radii[:-1])
    assert radii[-1] > 0.98


def test_vla_config_enables_radial_calibration_by_default() -> None:
    """Phase 1 radial calibration should be on unless explicitly disabled."""
    assert VLAConfig().w_radial_calibration == 0.1


def test_error_quality_targets_decay_with_reconstruction_error() -> None:
    """Lower reconstruction error should map to higher quality targets."""
    errors = torch.tensor([0.1, 0.2, 0.8], dtype=torch.float32)
    quality = compute_error_quality_targets(errors, alpha=2.0)

    assert quality.shape == (3,)
    assert torch.all((quality >= 0.0) & (quality <= 1.0))
    assert quality[0] > quality[1] > quality[2]


def test_rank_quality_targets_decay_with_reconstruction_error() -> None:
    """Rank-based targets should preserve ordering while using relative quality."""
    errors = torch.tensor([0.1, 0.2, 0.8], dtype=torch.float32)
    quality = compute_rank_quality_targets(errors)

    assert quality.shape == (3,)
    assert torch.all((quality >= 0.0) & (quality <= 1.0))
    assert quality[0] > quality[1] > quality[2]
    assert torch.isclose(quality.max(), torch.tensor(1.0))
    assert torch.isclose(quality.min(), torch.tensor(0.0))


def test_mixed_rank_quality_is_less_punitive_than_absolute_product() -> None:
    """Rank-mixed averaging should keep usable quality mass for calibration."""
    recon_abs = torch.tensor([0.18, 0.18], dtype=torch.float32)
    recon_rank = torch.tensor([1.0, 0.0], dtype=torch.float32)
    vq_abs = torch.tensor([0.52, 0.52], dtype=torch.float32)
    vq_rank = torch.tensor([1.0, 0.0], dtype=torch.float32)

    recon_quality = mix_quality_targets(recon_abs, recon_rank, rank_mix=0.75)
    vq_quality = mix_quality_targets(vq_abs, vq_rank, rank_mix=0.75)
    combined = combine_quality_targets(
        recon_quality,
        vq_quality,
        primary_weight=0.7,
    )

    old_product = (recon_abs * vq_abs).mean()

    assert combined.mean().item() > old_product.item()
    assert torch.all((combined >= 0.0) & (combined <= 1.0))


def test_quality_gated_radial_calibration_penalizes_outer_bad_reconstructions() -> None:
    """High-confidence outer-shell samples should be pulled inward when quality is low."""
    z_geo = torch.tensor([[0.95, 0.0], [0.95, 0.0]], dtype=torch.float32)
    router = torch.tensor([[0.999, 0.001], [0.999, 0.001]], dtype=torch.float32)
    quality = torch.zeros(2, dtype=torch.float32)

    loss_conf_only = compute_radial_calibration_loss(
        z_geo,
        router,
        num_charts=2,
        use_hyperbolic_radius=True,
        rho_max=4.0,
    )
    loss_quality_gated = compute_radial_calibration_loss(
        z_geo,
        router,
        num_charts=2,
        quality_target=quality,
        quality_mix=1.0,
        use_hyperbolic_radius=True,
        rho_max=4.0,
    )
    loss_conf_cal = compute_confidence_calibration_loss(router, quality, num_charts=2)

    assert loss_quality_gated.item() > loss_conf_only.item()
    assert loss_conf_cal.item() > 0.0


def test_radial_calibration_uses_local_radius_not_global_radius() -> None:
    """A globally outer point should still be penalized if it is locally near its chart center."""
    z_geo = torch.tensor([[0.95, 0.0], [0.95, 0.0]], dtype=torch.float32)
    router = torch.tensor([[0.999, 0.001], [0.999, 0.001]], dtype=torch.float32)
    outer_centers = torch.tensor([[0.90, 0.0], [0.90, 0.0]], dtype=torch.float32)

    loss_global = compute_radial_calibration_loss(
        z_geo,
        router,
        num_charts=2,
        use_hyperbolic_radius=True,
        rho_max=4.0,
        rho_band_width=0.75,
    )
    loss_local = compute_radial_calibration_loss(
        z_geo,
        router,
        num_charts=2,
        center_points=outer_centers,
        use_hyperbolic_radius=True,
        rho_max=4.0,
        rho_band_width=0.75,
    )

    assert loss_local.item() > loss_global.item()


def test_phase1_loss_uses_explicit_router_and_v_local_over_stale_cache() -> None:
    """Phase 1 loss must use tensors from the matching forward pass."""
    codebook = torch.tensor(
        [
            [[-0.20, 0.00], [0.20, 0.00]],
            [[-0.20, 0.00], [0.20, 0.00]],
        ],
        dtype=torch.float32,
    )
    stale_router = torch.full((4, 2), 0.5, dtype=torch.float32)
    stale_v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [-0.20, 0.00],
            [-0.20, 0.00],
            [-0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    fresh_router = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    fresh_v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )

    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            codebook,
            router_reg_weights=stale_router,
            v_local=stale_v_local,
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=1.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=1.0,
        w_window=0.0,
        w_consistency=0.0,
    )
    x = torch.zeros(4, 2)
    z_geo = fresh_v_local.clone()
    enc_router_weights = fresh_router
    dec_router_weights = fresh_router
    vq_loss = torch.tensor(0.0)

    _, _, stale_metrics = compute_phase1_loss(
        x,
        x,
        vq_loss,
        enc_router_weights,
        dec_router_weights,
        z_geo,
        encoder,
        config,
    )
    _, _, fresh_metrics = compute_phase1_loss(
        x,
        x,
        vq_loss,
        enc_router_weights,
        dec_router_weights,
        z_geo,
        encoder,
        config,
        router_reg_weights=fresh_router,
        v_local=fresh_v_local,
    )

    assert stale_metrics["entropy"] > 0.5
    assert fresh_metrics["entropy"] < 1e-4
    assert abs(fresh_metrics["chart_usage"] - stale_metrics["chart_usage"]) < 1e-6
    assert fresh_metrics["code_usage"] < stale_metrics["code_usage"]


def test_phase1_code_usage_requires_chart_local_latent() -> None:
    """Code-usage regularization should fail fast without the chart-local latent."""
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            torch.tensor(
                [
                    [[-0.20, 0.00], [0.20, 0.00]],
                    [[-0.20, 0.00], [0.20, 0.00]],
                ],
                dtype=torch.float32,
            ),
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=1.0,
        w_window=0.0,
        w_consistency=0.0,
    )

    try:
        compute_phase1_loss(
            torch.zeros(4, 2),
            torch.zeros(4, 2),
            torch.tensor(0.0),
            torch.tensor([[1.0, 0.0]] * 4),
            torch.tensor([[1.0, 0.0]] * 4),
            torch.zeros(4, 2),
            encoder,
            config,
        )
    except RuntimeError as exc:
        assert "chart-local latent" in str(exc)
    else:
        msg = "compute_phase1_loss should require v_local for code usage"
        raise AssertionError(msg)


def test_phase1_loss_reports_information_metrics_even_without_window_loss() -> None:
    """Router MI and sharpness diagnostics should not depend on w_window > 0."""
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            torch.tensor(
                [
                    [[-0.20, 0.00], [0.20, 0.00]],
                    [[-0.20, 0.00], [0.20, 0.00]],
                ],
                dtype=torch.float32,
            ),
            router_reg_weights=torch.tensor(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                ],
                dtype=torch.float32,
            ),
            v_local=torch.tensor(
                [
                    [-0.20, 0.00],
                    [0.20, 0.00],
                ],
                dtype=torch.float32,
            ),
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
    )

    _, _, metrics = compute_phase1_loss(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(2, 2),
        encoder,
        config,
    )

    assert metrics["window"] == 0.0
    assert metrics["H_K"] > 0.6
    assert metrics["H_K_given_X"] < metrics["H_K"]
    assert metrics["I_XK"] > 0.0
    assert abs(metrics["top1_prob_mean"] - 0.95) < 1e-4
    assert abs(metrics["top1_gap_mean"] - 0.90) < 1e-4


def test_phase1_loss_can_penalize_pre_squash_v_saturation() -> None:
    """A large cached v_raw should trigger the tangent barrier term."""
    v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            torch.tensor(
                [
                    [[-0.20, 0.00], [0.20, 0.00]],
                    [[-0.20, 0.00], [0.20, 0.00]],
                ],
                dtype=torch.float32,
            ),
            router_reg_weights=torch.tensor(
                [
                    [0.95, 0.05],
                    [0.05, 0.95],
                ],
                dtype=torch.float32,
            ),
            v_local=v_local,
            v_raw=torch.tensor(
                [
                    [3.0, 0.0],
                    [3.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=0.0,
        w_v_tangent_barrier=1.0,
        v_tangent_barrier_radius=0.5,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
    )

    _, _, metrics = compute_phase1_loss(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.zeros(2, 2),
        encoder,
        config,
        v_local=v_local,
    )

    assert metrics["v_tangent_barrier"] > 0.0


def test_phase1_loss_uses_reconstruction_quality_to_gate_radial_targets() -> None:
    """Quality-aware calibration should pull confident bad reconstructions inward."""
    v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    confident_router = torch.tensor(
        [
            [0.999, 0.001],
            [0.001, 0.999],
        ],
        dtype=torch.float32,
    )
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            torch.tensor(
                [
                    [[-0.20, 0.00], [0.20, 0.00]],
                    [[-0.20, 0.00], [0.20, 0.00]],
                ],
                dtype=torch.float32,
            ),
            router_reg_weights=confident_router,
            v_local=v_local,
        ),
    )
    x = torch.zeros(2, 2)
    x_recon = torch.ones(2, 2)
    z_geo = torch.tensor([[0.95, 0.0], [0.95, 0.0]], dtype=torch.float32)

    old_config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=1.0,
        w_confidence_calibration=0.0,
        radial_quality_mix=0.0,
        radial_calibration_rho_max=4.0,
        w_v_tangent_barrier=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
    )
    new_config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=1.0,
        w_confidence_calibration=1.0,
        radial_quality_alpha=2.0,
        radial_quality_mix=1.0,
        radial_calibration_rho_max=4.0,
        w_v_tangent_barrier=0.0,
        w_codebook_spread=0.0,
        w_codebook_center=0.0,
        w_chart_collapse=0.0,
        w_code_collapse=0.0,
        w_window=0.0,
        w_consistency=0.0,
    )

    _, _, old_metrics = compute_phase1_loss(
        x,
        x_recon,
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        z_geo,
        encoder,
        old_config,
        router_reg_weights=confident_router,
        v_local=v_local,
    )
    _, _, new_metrics = compute_phase1_loss(
        x,
        x_recon,
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        z_geo,
        encoder,
        new_config,
        router_reg_weights=confident_router,
        v_local=v_local,
    )

    assert new_metrics["radial_cal"] < old_metrics["radial_cal"]
    assert new_metrics["confidence_calibration"] > 0.0
    assert new_metrics["recon_quality_mean"] < 0.5


def test_phase1_loss_quality_base_shell_avoids_zero_target_when_confidence_is_zero() -> None:
    """Quality should still induce a basal shell before the router becomes confident."""
    v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    uniform_router = torch.tensor(
        [
            [0.5, 0.5],
            [0.5, 0.5],
        ],
        dtype=torch.float32,
    )
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            torch.tensor(
                [
                    [[-0.20, 0.00], [0.20, 0.00]],
                    [[-0.20, 0.00], [0.20, 0.00]],
                ],
                dtype=torch.float32,
            ),
            router_reg_weights=uniform_router,
            v_local=v_local,
            c_bar=torch.zeros_like(v_local),
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=1.0,
        w_confidence_calibration=0.0,
        radial_quality_alpha=2.0,
        radial_quality_mix=1.0,
        radial_quality_base_weight=0.25,
        radial_calibration_rho_max=4.0,
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

    _, _, metrics = compute_phase1_loss(
        torch.zeros(2, 2),
        torch.zeros(2, 2),
        torch.tensor(0.0),
        uniform_router,
        uniform_router,
        torch.tensor([[0.95, 0.0], [0.95, 0.0]], dtype=torch.float32),
        encoder,
        config,
        router_reg_weights=uniform_router,
        c_bar=torch.zeros_like(v_local),
        v_local=v_local,
    )

    assert metrics["routing_confidence_mean"] < 1e-4
    assert metrics["combined_quality_mean"] > 0.0
    assert metrics["radial_target_mean"] > 0.0


def test_phase1_loss_uses_vq_quality_to_reduce_radial_targets() -> None:
    """Poor VQ matches should lower the quality-gated radial target even with perfect recon."""
    v_local = torch.tensor(
        [
            [-0.20, 0.00],
            [0.20, 0.00],
        ],
        dtype=torch.float32,
    )
    confident_router = torch.tensor(
        [
            [0.999, 0.001],
            [0.001, 0.999],
        ],
        dtype=torch.float32,
    )
    codebook = torch.tensor(
        [
            [[-0.20, 0.00], [0.20, 0.00]],
            [[-0.20, 0.00], [0.20, 0.00]],
        ],
        dtype=torch.float32,
    )
    encoder = _DummyWrapper(
        _DummyAtlasEncoder(
            codebook,
            router_reg_weights=confident_router,
            v_local=v_local,
            c_bar=torch.zeros_like(v_local),
        ),
    )
    config = VLAConfig(
        input_dim=2,
        feature_dim=2,
        num_charts=2,
        codes_per_chart=2,
        w_feature_recon=0.0,
        w_vq=0.0,
        w_entropy=0.0,
        w_diversity=0.0,
        w_uniformity=0.0,
        w_radial_calibration=1.0,
        radial_quality_mix=1.0,
        radial_quality_alpha=2.0,
        radial_vq_alpha=2.0,
        radial_calibration_rho_max=4.0,
        radial_calibration_band_width=0.75,
        w_confidence_calibration=0.0,
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

    x = torch.zeros(2, 2)
    z_geo = torch.tensor([[0.95, 0.0], [0.95, 0.0]], dtype=torch.float32)
    good_indices = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    bad_indices = torch.tensor([[1, 1], [0, 0]], dtype=torch.long)

    _, _, good_metrics = compute_phase1_loss(
        x,
        x,
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        z_geo,
        encoder,
        config,
        router_reg_weights=confident_router,
        c_bar=torch.zeros_like(v_local),
        v_local=v_local,
        indices_stack=good_indices,
    )
    _, _, bad_metrics = compute_phase1_loss(
        x,
        x,
        torch.tensor(0.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        z_geo,
        encoder,
        config,
        router_reg_weights=confident_router,
        c_bar=torch.zeros_like(v_local),
        v_local=v_local,
        indices_stack=bad_indices,
    )

    assert good_metrics["vq_quality_mean"] > bad_metrics["vq_quality_mean"]
    assert good_metrics["combined_quality_mean"] > bad_metrics["combined_quality_mean"]
    assert good_metrics["radial_target_mean"] > bad_metrics["radial_target_mean"]


def test_router_score_metrics_capture_logit_gap_quantiles() -> None:
    """Raw router-score diagnostics should expose separation before softmax."""
    scores = torch.tensor(
        [
            [3.0, 1.0, -1.0],
            [2.0, 1.5, -0.5],
            [0.5, 0.25, 0.0],
        ],
        dtype=torch.float32,
    )

    metrics = compute_router_score_metrics(scores)

    gaps = torch.tensor([2.0, 0.5, 0.25], dtype=torch.float32)
    assert abs(metrics["score_gap_mean"].item() - gaps.mean().item()) < 1e-6
    assert abs(metrics["score_gap_p50"].item() - torch.quantile(gaps, 0.50).item()) < 1e-6
    assert abs(metrics["score_gap_p90"].item() - torch.quantile(gaps, 0.90).item()) < 1e-6
    assert abs(metrics["score_gap_p99"].item() - torch.quantile(gaps, 0.99).item()) < 1e-6
    assert metrics["score_std"].item() > 0.0
    assert metrics["score_mean_abs"].item() > 0.0


def test_jump_weight_schedule_stays_zero_when_final_weight_disabled() -> None:
    """Disabling jump loss should keep its scheduled weight at zero."""
    for epoch in (0, 20, 40, 100):
        assert get_jump_weight_schedule(epoch, warmup_end=20, ramp_end=50, final_weight=0.0) == 0.0
