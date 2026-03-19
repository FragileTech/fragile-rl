"""Smoke, resume, and profiling tests for the geometry-training command."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
import time

from click.testing import CliRunner
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader

from fragile.__main__ import run
from fragile.checkpoints import load_checkpoint
from fragile.losses.macro import compute_absolute_enclosure_loss
from fragile.losses.markov_model import compute_markov_transition_loss
from fragile.vla.extract_features import VLAFeatureDataset


train_geometry_module = importlib.import_module("fragile.commands.train_geometry")

CONFIG_PATH = train_geometry_module.CONFIG_PATH


def _write_sequence_feature_cache(cache_dir: Path) -> None:
    """Write a tiny cached VLA dataset with train/test sequence splits."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "episode_ids": [0, 1, 2],
        "num_episodes": 3,
        "held_out_test_episodes": 1,
        "split_strategy": "last_n_episodes",
        "train_episode_ids": [0, 1],
        "test_episode_ids": [2],
        "num_train_episodes": 2,
        "num_test_episodes": 1,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta))

    torch.manual_seed(23)
    for ep_id in meta["episode_ids"]:
        ep_dir = cache_dir / f"episode_{ep_id}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        # Give each episode a slightly different offset so train/test are distinct.
        features = torch.randn(6, 6) + 0.25 * ep_id
        actions = torch.randn(6, 4) + 0.10 * ep_id
        torch.save(features, ep_dir / "features.pt")
        torch.save(actions, ep_dir / "actions.pt")


def _make_runner(
    cache_dir: Path,
    output_dir: Path,
    **overrides,
) -> train_geometry_module.GeometryTrainingRunner:
    """Build a GeometryTrainingRunner with small test dimensions."""
    cfg = OmegaConf.load(CONFIG_PATH)
    test_overrides = OmegaConf.create({
        "feature_cache_dir": str(cache_dir),
        "output_dir": str(output_dir),
        "epochs": 1,
        "batch_size": 2,
        "sequence_length": 2,
        "device": "cpu",
        "log_every": 1,
        "save_every": 1,
        "agent": {
            "enclosure_hidden_dim": 32,
            "obs_encoder": {
                "hidden_dim": 24,
                "latent_dim": 4,
                "num_charts": 4,
                "codes_per_chart": 4,
                "chart_ot_iters": 4,
                "w_jump": 0.1,
                "w_jump_warmup": 0,
                "w_jump_ramp_end": 1,
            },
            "act_encoder": {
                "hidden_dim": 24,
                "latent_dim": 4,
                "num_charts": 2,
                "codes_per_chart": 3,
                "chart_ot_iters": 4,
                "w_jump": 0.1,
                "w_jump_warmup": 0,
                "w_jump_ramp_end": 1,
            },
        },
    })
    merged = OmegaConf.merge(cfg, test_overrides, OmegaConf.create(overrides))
    return instantiate(merged)


def _build_runtime(
    cache_dir: Path,
    output_dir: Path,
    **overrides,
):
    """Construct the dataset/loaders/trainer stack used by the command."""
    runner = _make_runner(cache_dir, output_dir, **overrides)
    window_stride = train_geometry_module._effective_window_stride(
        runner.sequence_length, runner.window_stride
    )
    train_dataset = VLAFeatureDataset(
        cache_dir,
        sequence_length=runner.sequence_length,
        window_stride=window_stride,
        split="train",
    )
    eval_dataset = VLAFeatureDataset(
        cache_dir,
        sequence_length=runner.sequence_length,
        window_stride=window_stride,
        split="test",
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=runner.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=runner.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    sample = train_dataset[0]
    obs_dim = int(sample["features"].shape[-1])
    act_dim = int(sample["actions"].shape[-1])

    runner.agent.obs_encoder.input_dim = obs_dim
    runner.agent.obs_encoder.feature_dim = obs_dim
    runner.agent.act_encoder.input_dim = act_dim
    runner.agent.act_encoder.feature_dim = act_dim

    from fragile.agent import FragileAgent, FragileAgentTrainer

    torch.manual_seed(7)
    agent = FragileAgent(runner.agent)
    trainer = FragileAgentTrainer(agent, runner.trainer)
    trainer.agent.to(torch.device("cpu"))
    batch = train_geometry_module._trainer_batch(next(iter(train_loader)))
    return runner, train_loader, eval_loader, trainer, batch


def _time_call_ms(fn, *, repeats: int = 1, warmup: int = 1) -> float:
    """Measure average wall-clock time for a callable in milliseconds."""
    for _ in range(max(warmup, 0)):
        fn()
    start = time.perf_counter()
    for _ in range(max(repeats, 1)):
        fn()
    elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / max(repeats, 1)


def _enclosure_loss_call(trainer, transitions: dict[str, torch.Tensor]) -> None:
    """Run the enclosure loss on a prepared transition batch."""
    compute_absolute_enclosure_loss(
        trainer.agent.enclosure_probe,
        obs_chart_centers=trainer.agent.obs_encoder.encoder.chart_centers,
        obs_codebook=trainer.agent.obs_encoder.encoder.codebook,
        obs_chart_t=transitions["obs_chart_t_valid"],
        obs_code_t=transitions["obs_code_t_valid"],
        obs_z_n_t=transitions["obs_z_n_t_valid"],
        obs_z_tex_t=transitions["obs_z_tex_t_valid"],
        act_chart_centers=trainer.agent.act_encoder.encoder.chart_centers,
        act_codebook=trainer.agent.act_encoder.encoder.codebook,
        act_chart_t=transitions["act_chart_t_valid"],
        act_code_t=transitions["act_code_t_valid"],
        act_z_n_t=transitions["act_z_n_t_valid"],
        act_z_tex_t=transitions["act_z_tex_t_valid"],
        obs_chart_tp1=transitions["obs_chart_tp1_valid"],
        obs_code_tp1=transitions["obs_code_tp1_valid"],
        obs_codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
    )


def _markov_loss_call(
    trainer,
    transitions: dict[str, torch.Tensor],
    macro: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Run the coarse Markov transition loss on a prepared transition batch."""
    compute_markov_transition_loss(
        trainer.agent.macro_model,
        transitions["obs_state_probs_t_valid"],
        transitions["act_state_probs_t_valid"],
        obs_geometry={
            "chart_centers": macro["obs"]["chart_centers"],
            "codebook": macro["obs"]["codebook"],
            "state_points": macro["obs"]["state_points"],
            "state_tangent_points": macro["obs"]["state_tangent_points"],
        },
        act_geometry={
            "chart_centers": macro["act"]["chart_centers"],
            "codebook": macro["act"]["codebook"],
            "state_points": macro["act"]["state_points"],
            "state_tangent_points": macro["act"]["state_tangent_points"],
        },
        target_next_state_probs=transitions["obs_state_probs_tp1_valid"].detach(),
        target_next_chart_idx=transitions["obs_chart_tp1_valid"],
        target_next_code_idx=transitions["obs_code_tp1_valid"],
        codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
        metric_prefix="markov",
    )


def test_train_geometry_smoke_writes_checkpoints_and_logs_metrics(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "features"
    output_dir = tmp_path / "geometry"
    _write_sequence_feature_cache(cache_dir)

    def fail_on_rescan(*_args, **_kwargs):
        msg = "train_geometry should not rescan the loader for code activity."
        raise AssertionError(msg)

    monkeypatch.setattr(train_geometry_module, "_collect_code_activity", fail_on_rescan)

    runner = _make_runner(cache_dir, output_dir, epochs=1, save_every=1)
    runner.run()

    output = capsys.readouterr().out
    assert (output_dir / "geometry_best.pt").exists()
    assert (output_dir / "geometry_epoch_00000.pt").exists()
    assert (output_dir / "geometry_final.pt").exists()
    assert "Train metrics:" in output
    assert "train obs codes/chart:" in output
    assert "train act codes/chart:" in output
    assert "Eval metrics:" in output
    assert "eval obs codes/chart:" in output
    assert "eval act codes/chart:" in output
    assert "/" in output and "[" in output and "]" in output
    assert "split=test" in output
    assert "loss:" in output
    assert "markov:" in output
    assert "enclosure:" in output


def test_train_geometry_resume_restores_progress(tmp_path) -> None:
    cache_dir = tmp_path / "features"
    output_dir = tmp_path / "geometry"
    _write_sequence_feature_cache(cache_dir)

    runner = _make_runner(cache_dir, output_dir, epochs=1, save_every=1)
    runner.run()

    epoch_ckpt = output_dir / "geometry_epoch_00000.pt"
    first_ckpt = load_checkpoint(str(epoch_ckpt))
    first_global_step = int(first_ckpt["global_step"])

    runner2 = _make_runner(
        cache_dir,
        output_dir,
        epochs=2,
        save_every=1,
        resume=str(epoch_ckpt),
    )
    runner2.run()

    final_ckpt = load_checkpoint(str(output_dir / "geometry_final.pt"))
    assert int(final_ckpt["epoch"]) == 1
    assert int(final_ckpt["global_step"]) > first_global_step


def test_train_geometry_early_stops_on_stale_eval_metric(
    tmp_path,
    monkeypatch,
) -> None:
    cache_dir = tmp_path / "features"
    output_dir = tmp_path / "geometry"
    _write_sequence_feature_cache(cache_dir)

    eval_scores = iter([0.9, 0.8, 0.7])

    def fake_train_epoch(*_args, **_kwargs) -> dict[str, float]:
        return {"loss/main": 1.0, "obs/I_XK": 0.0}, {"obs": [1, 1, 1, 1], "act": [1, 1]}

    def fake_eval_epoch(*_args, **_kwargs):
        score = next(eval_scores)
        return {"loss/main": 1.0, "obs/I_XK": score}, {"obs": [1, 1, 1, 1], "act": [1, 1]}

    monkeypatch.setattr(train_geometry_module, "_run_train_epoch", fake_train_epoch)
    monkeypatch.setattr(train_geometry_module, "_run_eval_epoch", fake_eval_epoch)

    runner = _make_runner(
        cache_dir,
        output_dir,
        epochs=10,
        eval_every=1,
        save_every=10,
        early_stop_patience=2,
        early_stop_min_epochs=0,
    )
    runner.run()

    final_ckpt = load_checkpoint(str(output_dir / "geometry_final.pt"))
    best_ckpt = load_checkpoint(str(output_dir / "geometry_best.pt"))
    assert int(final_ckpt["epoch"]) == 2
    assert int(best_ckpt["best_eval_epoch"]) == 0
    assert float(best_ckpt["best_eval_metric_value"]) == 0.9


def test_train_geometry_eval_every_decouples_eval_from_logging(tmp_path, capsys) -> None:
    cache_dir = tmp_path / "features"
    output_dir = tmp_path / "geometry"
    _write_sequence_feature_cache(cache_dir)

    runner = _make_runner(
        cache_dir,
        output_dir,
        epochs=3,
        log_every=1,
        eval_every=10,
        save_every=3,
    )
    runner.run()

    output = capsys.readouterr().out
    assert output.count("Geometry E") == 3
    assert output.count("Train metrics:") == 3
    assert output.count("train obs codes/chart:") == 3
    assert output.count("train act codes/chart:") == 3
    assert output.count("eval obs codes/chart:") == 2
    assert output.count("eval act codes/chart:") == 2
    assert "eval=skipped" in output
    assert "Eval metrics: skipped (runs every 10 epochs)" in output


def test_vla_geometry_config_loads_and_instantiates() -> None:
    """Verify that the YAML config loads and produces a valid runner."""
    cfg = OmegaConf.load(CONFIG_PATH)
    assert "markov_hidden_dim" in cfg.agent
    assert "markov_feature_scale" in cfg.agent
    assert "markov_use_residual_transition" in cfg.agent
    assert "chart_usage_entropy_low" in cfg.agent.obs_encoder
    assert "chart_usage_entropy_high" in cfg.agent.obs_encoder
    assert "code_usage_entropy_low" in cfg.agent.obs_encoder
    assert "code_usage_entropy_high" in cfg.agent.obs_encoder
    assert "soft_equiv_bundle_size" in cfg.agent.obs_encoder
    assert "input_affine_min_scale" in cfg.agent.act_encoder
    runner = instantiate(cfg)
    assert isinstance(runner, train_geometry_module.GeometryTrainingRunner)
    assert runner.epochs == 1000
    assert runner.output_dir == "outputs/vla/geometry-obs16x16-act8x8"
    assert runner.eval_every == 10
    assert runner.agent.obs_encoder.num_charts == 16
    assert runner.agent.obs_encoder.codes_per_chart == 16
    assert runner.agent.act_encoder.input_affine_enabled is True
    assert runner.agent.act_encoder.num_charts == 8
    assert runner.agent.act_encoder.codes_per_chart == 8
    assert runner.agent.obs_encoder.w_window == 1.0
    assert runner.agent.act_encoder.w_window == 1.0
    assert runner.agent.markov_hidden_dim == 128
    assert runner.trainer.lr_encoder == 0.001


def test_train_geometry_profile_breakdown(tmp_path, capsys) -> None:
    """Print timing breakdowns so `pytest -s` can double as a simple profiler."""
    cache_dir = tmp_path / "features"
    _write_sequence_feature_cache(cache_dir)

    _runner, _train_loader, _eval_loader, trainer, batch = _build_runtime(
        cache_dir,
        tmp_path / "profile",
    )
    obs = batch["obs"].to(trainer.device)
    act = batch["act"].to(trainer.device)
    routing_tau = trainer.routing_tau_for_step(training=True)

    forward = trainer.agent.forward_batch(
        obs,
        act,
        routing_tau=routing_tau,
        macro_chart_tau=trainer.config.macro_chart_tau,
        macro_code_tau=trainer.config.macro_code_tau,
    )
    transitions = forward["transitions"]
    assert transitions["num_valid"] > 0

    benchmarks = {
        "obs_encode_ms": _time_call_ms(
            lambda: trainer.agent.encode_observations(obs, routing_tau=routing_tau),
            repeats=3,
        ),
        "act_encode_ms": _time_call_ms(
            lambda: trainer.agent.encode_actions(act, routing_tau=routing_tau),
            repeats=3,
        ),
        "forward_batch_ms": _time_call_ms(
            lambda: trainer.agent.forward_batch(
                obs,
                act,
                routing_tau=routing_tau,
                macro_chart_tau=trainer.config.macro_chart_tau,
                macro_code_tau=trainer.config.macro_code_tau,
            ),
            repeats=3,
        ),
        "obs_phase1_stack_ms": _time_call_ms(
            lambda: trainer._compute_phase1_stack(
                forward["obs"],
                trainer.agent.obs_encoder,
                trainer.agent.obs_jump_operator,
                trainer.agent.config.obs_encoder,
                epoch=0,
                prefix="obs",
            ),
            repeats=3,
        ),
        "act_phase1_stack_ms": _time_call_ms(
            lambda: trainer._compute_phase1_stack(
                forward["act"],
                trainer.agent.act_encoder,
                trainer.agent.act_jump_operator,
                trainer.agent.config.act_encoder,
                epoch=0,
                prefix="act",
            ),
            repeats=3,
        ),
        "enclosure_loss_ms": _time_call_ms(
            lambda: _enclosure_loss_call(trainer, transitions),
            repeats=3,
        ),
        "markov_transition_loss_ms": _time_call_ms(
            lambda: _markov_loss_call(trainer, transitions, forward["macro"]),
            repeats=3,
        ),
        "compute_batch_losses_train_ms": _time_call_ms(
            lambda: trainer.compute_batch_losses(batch, epoch=0, global_step=0, training=True),
            repeats=2,
        ),
        "compute_batch_losses_eval_ms": _time_call_ms(
            lambda: trainer.compute_batch_losses(batch, epoch=0, global_step=0, training=False),
            repeats=2,
        ),
    }

    _, _, _, train_step_trainer, train_step_batch = _build_runtime(
        cache_dir,
        tmp_path / "profile-train-step",
    )
    benchmarks["train_step_ms"] = _time_call_ms(
        lambda: train_step_trainer.train_step(train_step_batch, epoch=0),
        repeats=1,
        warmup=0,
    )

    _, _, _, eval_step_trainer, eval_step_batch = _build_runtime(
        cache_dir,
        tmp_path / "profile-eval-step",
    )
    benchmarks["eval_step_ms"] = _time_call_ms(
        lambda: eval_step_trainer.eval_step(eval_step_batch, epoch=0),
        repeats=1,
        warmup=0,
    )

    _, _, code_activity_loader, code_activity_trainer, _ = _build_runtime(
        cache_dir,
        tmp_path / "profile-code-activity",
    )
    benchmarks["code_activity_ms"] = _time_call_ms(
        lambda: train_geometry_module._collect_code_activity(
            code_activity_trainer,
            code_activity_loader,
        ),
        repeats=1,
        warmup=0,
    )

    _, epoch_train_loader, epoch_eval_loader, train_epoch_trainer, _ = _build_runtime(
        cache_dir,
        tmp_path / "profile-epochs",
    )
    benchmarks["train_epoch_ms"] = _time_call_ms(
        lambda: train_geometry_module._run_train_epoch(
            train_epoch_trainer,
            epoch_train_loader,
            epoch=0,
        ),
        repeats=1,
        warmup=0,
    )
    benchmarks["eval_epoch_ms"] = _time_call_ms(
        lambda: train_geometry_module._run_eval_epoch(
            train_epoch_trainer,
            epoch_eval_loader,
            epoch=0,
        ),
        repeats=1,
        warmup=0,
    )

    with capsys.disabled():
        print("\n[train_geometry profile]")
        print(f"  batch_shape_obs={tuple(obs.shape)} batch_shape_act={tuple(act.shape)}")
        for key in sorted(benchmarks):
            print(f"  {key}={benchmarks[key]:.3f} ms")

    assert all(value > 0.0 for value in benchmarks.values())
