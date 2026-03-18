"""Parity and smoke tests for the standalone Phase 1 trainer."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

from click.testing import CliRunner
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from fragile.__main__ import run
from fragile.layers import FactorizedJumpOperator
from fragile.layers.topoencoder import TopoEncoder
from fragile.vla.train_joint import (
    _compute_encoder_losses as joint_compute_encoder_losses,
    _eval_pass as joint_eval_pass,
    _phase1_config_from_args as joint_phase1_config_from_args,
)
from fragile.vla.train_phase_1 import (
    _compute_encoder_losses as phase1_compute_encoder_losses,
    _eval_pass as phase1_eval_pass,
    _phase1_config_from_args as phase1_config_from_args,
    train_phase_1,
)


train_phase_1_module = importlib.import_module("fragile.vla.train_phase_1")


class _FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor) -> None:
        self.features = features

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"feature": self.features[idx]}


def _make_phase1_args(**overrides) -> SimpleNamespace:
    args = {
        "feature_cache_dir": "outputs/vla/features-smoke",
        "output_dir": "outputs/vla/phase1-test",
        "epochs": 2,
        "batch_size": 4,
        "lr": 1e-3,
        "lr_chart_centers_scale": 0.1,
        "lr_codebook_scale": 0.5,
        "grad_clip": 1.0,
        "use_scheduler": False,
        "phase1_cosine_lr": False,
        "phase1_eta_min": 1e-6,
        "latent_dim": 2,
        "hidden_dim": 16,
        "num_charts": 4,
        "codes_per_chart": 5,
        "log_every": 1,
        "save_every": 1,
        "resume": "",
        "device": "cpu",
        "hard_routing": True,
        "hard_routing_warmup_epochs": 0,
        "hard_routing_tau": -1.0,
        "hard_routing_tau_end": 0.3,
        "hard_routing_tau_anneal_epochs": 10,
        "w_recon": 1.0,
        "w_vq": 1.0,
        "w_entropy": 0.3,
        "w_consistency": 0.0,
        "w_diversity": 1.0,
        "chart_usage_h_low": None,
        "chart_usage_h_high": None,
        "w_chart_ot": 1.0,
        "chart_ot_epsilon": 0.05,
        "chart_ot_iters": 20,
        "w_uniformity": 0.05,
        "w_radial_cal": 0.1,
        "w_confidence_calibration": 0.05,
        "w_hard_routing_nll": 0.5,
        "w_router_margin": 2.0,
        "router_margin_target": 0.05,
        "radial_quality_alpha": 2.0,
        "radial_vq_alpha": 1.0,
        "radial_quality_rank_mix": 0.75,
        "radial_recon_quality_weight": 0.7,
        "radial_quality_mix": 1.0,
        "radial_quality_base_weight": 0.0,
        "radial_calibration_rho_max": 4.0,
        "radial_calibration_band_width": 0.75,
        "w_v_tangent_barrier": 0.01,
        "v_tangent_barrier_radius": 0.9,
        "w_codebook_spread": 0.05,
        "w_codebook_center": 0.02,
        "w_chart_center_mean": 0.02,
        "w_chart_center_radius": 0.05,
        "chart_center_radius_max": 2.0,
        "w_chart_center_sep": 0.02,
        "chart_center_sep_margin": 1.0,
        "w_chart_collapse": 0.0,
        "w_code_collapse": 0.5,
        "code_usage_h_low": None,
        "code_usage_h_high": None,
        "code_usage_temperature": 1.0,
        "w_window": 0.0,
        "w_window_eps_ground": 0.1,
        "w_jump": 0.2,
        "w_jump_warmup": 0,
        "w_jump_ramp_end": 1,
        "w_perp": 0.01,
        "phase1_adaptive_multipliers": True,
        "phase1_multiplier_max": 8.0,
        "phase1_multiplier_decay": 0.05,
        "conf_target_top1": 0.55,
        "conf_multiplier_lr": 1.5,
        "chart_multiplier_lr": 1.0,
        "chart_ot_i_target": 0.35,
        "chart_ot_multiplier_lr": 1.0,
        "code_usage_gate_h": 1.25,
        "code_usage_ramp_epochs": 50,
        "code_multiplier_lr": 0.5,
    }
    args.update(overrides)
    return SimpleNamespace(**args)


def _build_topoencoder_pair(
    *,
    input_dim: int = 3,
    hidden_dim: int = 16,
    latent_dim: int = 2,
    num_charts: int = 4,
    codes_per_chart: int = 5,
) -> tuple[TopoEncoder, TopoEncoder]:
    model_a = TopoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_charts=num_charts,
        codes_per_chart=codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    model_b = TopoEncoder(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_charts=num_charts,
        codes_per_chart=codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
    )
    model_b.load_state_dict(model_a.state_dict())
    return model_a, model_b


def _write_feature_cache(cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode_ids": [0, 1],
        "num_episodes": 2,
        "held_out_test_episodes": 0,
        "split_strategy": "last_n_episodes",
        "train_episode_ids": [0, 1],
        "test_episode_ids": [],
        "num_train_episodes": 2,
        "num_test_episodes": 0,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta))
    torch.manual_seed(23)
    for ep_id in meta["episode_ids"]:
        ep_dir = cache_dir / f"episode_{ep_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        torch.save(torch.randn(4, 3), ep_dir / "features.pt")
        torch.save(torch.randn(4, 2), ep_dir / "actions.pt")


def test_phase1_config_matches_train_joint() -> None:
    args = _make_phase1_args()

    assert phase1_config_from_args(args) == joint_phase1_config_from_args(args)


def test_compute_encoder_losses_matches_train_joint_phase1() -> None:
    torch.manual_seed(7)
    args = _make_phase1_args()
    x = torch.randn(6, 3)
    model_a, model_b = _build_topoencoder_pair()
    jump_a = FactorizedJumpOperator(num_charts=args.num_charts, latent_dim=args.latent_dim)
    jump_b = FactorizedJumpOperator(num_charts=args.num_charts, latent_dim=args.latent_dim)
    jump_b.load_state_dict(jump_a.state_dict())
    config_a = joint_phase1_config_from_args(args)
    config_b = phase1_config_from_args(args)

    joint = joint_compute_encoder_losses(
        x,
        model_a,
        jump_a,
        args,
        epoch=1,
        routing_tau=-1.0,
        phase1_config=config_a,
    )
    phase1 = phase1_compute_encoder_losses(
        x,
        model_b,
        jump_b,
        args,
        epoch=1,
        routing_tau=-1.0,
        phase1_config=config_b,
    )

    torch.testing.assert_close(joint[0], phase1[0])
    torch.testing.assert_close(joint[1], phase1[1])
    torch.testing.assert_close(joint[3], phase1[3])
    torch.testing.assert_close(joint[4], phase1[4])
    torch.testing.assert_close(joint[5].float(), phase1[5].float())

    joint_metrics = joint[2]
    phase1_metrics = phase1[2]
    for key in (
        "recon",
        "vq",
        "entropy",
        "chart_usage",
        "chart_ot",
        "uniformity",
        "radial_cal",
        "confidence_calibration",
        "hard_routing_nll",
        "router_margin",
        "jump",
        "jump_weight",
        "ortho",
        "total",
        "top1_prob_mean",
        "score_gap_p90",
        "soft_equiv_log_ratio",
        "v_boundary_frac",
    ):
        assert math.isclose(joint_metrics[key], phase1_metrics[key], rel_tol=1e-6, abs_tol=1e-6)


def test_eval_pass_matches_train_joint_phase1() -> None:
    torch.manual_seed(11)
    features = torch.randn(10, 3)
    loader = DataLoader(_FeatureDataset(features), batch_size=5, shuffle=False)
    model_a, model_b = _build_topoencoder_pair()

    joint = joint_eval_pass(
        model_a,
        loader,
        4,
        torch.device("cpu"),
        hard_routing=True,
        hard_routing_tau=0.5,
    )
    phase1 = phase1_eval_pass(
        model_b,
        loader,
        4,
        torch.device("cpu"),
        hard_routing=True,
        hard_routing_tau=0.5,
    )

    np.testing.assert_allclose(joint[0], phase1[0])
    assert math.isclose(joint[1], phase1[1], rel_tol=1e-6, abs_tol=1e-6)
    assert joint[2] == phase1[2]
    np.testing.assert_allclose(joint[3], phase1[3])
    assert math.isclose(joint[4], phase1[4], rel_tol=1e-6, abs_tol=1e-6)
    assert joint[5] == phase1[5]
    assert math.isclose(joint[6], phase1[6], rel_tol=1e-6, abs_tol=1e-6)

    joint_extra = joint[7]
    phase1_extra = phase1[7]
    for key in (
        "hard_entropy",
        "soft_I_XK",
        "soft_H_K",
        "soft_H_K_given_X",
        "soft_top1_prob_mean",
        "soft_top1_gap_mean",
        "soft_equiv_log_ratio",
        "v_boundary_frac",
        "vq_dist_p99",
        "cb_raw_r_p99",
        "cc_raw_r_p99",
        "score_gap_p99",
    ):
        assert math.isclose(
            float(joint_extra[key]),
            float(phase1_extra[key]),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    assert joint_extra["codes_per_chart"] == phase1_extra["codes_per_chart"]
    assert joint_extra["codes_per_chart_total"] == phase1_extra["codes_per_chart_total"]


def test_train_phase1_smoke_writes_checkpoints(tmp_path, monkeypatch) -> None:
    cache_dir = tmp_path / "features"
    output_dir = tmp_path / "phase1"
    _write_feature_cache(cache_dir)
    monkeypatch.setattr(train_phase_1_module, "_run_diagnostics", lambda *_args, **_kwargs: None)

    args = _make_phase1_args(
        feature_cache_dir=str(cache_dir),
        output_dir=str(output_dir),
        epochs=1,
        batch_size=2,
        save_every=1,
    )
    train_phase_1(args)

    assert (output_dir / "p1_epoch_00000.pt").exists()
    assert (output_dir / "checkpoint_final.pt").exists()


def test_vla_phase1_cli_exposes_phase1_only_flags() -> None:
    runner = CliRunner()
    result = runner.invoke(run, ["vla-phase1", "--", "--help"])

    assert result.exit_code == 0
    assert "--epochs" in result.output
    assert "--cosine-lr" in result.output
    assert "--adaptive-multipliers" in result.output
    assert "--phase2-epochs" not in result.output
    assert "--phase3-epochs" not in result.output
