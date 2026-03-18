"""CLI for sequence-based geometry training with ``FragileAgent``.

This command replaces the old hand-written Phase-1 loop with a simpler stack:

- observation and action sequences come from cached VLA windows,
- ``FragileAgent`` owns both topoencoders, the enclosure probe, and the
  symbolic Markov model,
- ``FragileAgentTrainer`` computes all losses and metrics,
- the CLI only handles config assembly, dataloading, logging, and checkpoints.
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from fragile.agent import (
    FragileAgent,
    FragileAgentConfig,
    FragileAgentTrainer,
    FragileAgentTrainerConfig,
)
from fragile.checkpoints import count_parameters, load_checkpoint, load_optimizer_state
from fragile.vla.config import VLAConfig
from fragile.vla.extract_features import VLAFeatureDataset


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested training device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _value_or(default: int | float, override: int | float | None) -> int | float:
    """Use an override value when provided, otherwise keep the default."""
    return default if override is None else override


def _average_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Average scalar metric dictionaries over one epoch."""
    if not metric_list:
        return {}
    averaged: dict[str, float] = {}
    keys = set().union(*(metrics.keys() for metrics in metric_list))
    for key in keys:
        values = [metrics[key] for metrics in metric_list if key in metrics]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged


def _format_metric_value(value: float) -> str:
    """Format metric values compactly for CLI output."""
    value = float(value)
    if value == 0.0:
        return "0"
    abs_value = abs(value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.3e}"
    return f"{value:.4f}"


def _print_metric_groups(title: str, metrics: dict[str, float]) -> None:
    """Print every metric grouped by its prefix."""
    print(f"{title}:")
    grouped: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for key, value in sorted(metrics.items()):
        prefix, sep, rest = key.partition("/")
        label = rest if sep else key
        grouped[prefix].append((label, value))

    for prefix in sorted(grouped):
        parts = " ".join(
            f"{name}={_format_metric_value(value)}" for name, value in grouped[prefix]
        )
        print(f"  {prefix}: {parts}")


def _trainer_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapt sequence-mode dataset batches to ``FragileAgentTrainer`` inputs."""
    return {
        "obs": batch["features"],
        "act": batch["actions"],
    }


def _collect_code_activity(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
) -> dict[str, list[int]]:
    """Collect per-chart active code counts for observation and action encoders."""
    obs_num_charts = trainer.agent.config.obs_encoder.num_charts
    act_num_charts = trainer.agent.config.act_encoder.num_charts
    obs_codes = [set() for _ in range(obs_num_charts)]
    act_codes = [set() for _ in range(act_num_charts)]

    was_training = trainer.agent.training
    trainer.agent.eval()
    with torch.no_grad():
        routing_tau = trainer.routing_tau_for_step(training=False)
        for batch in loader:
            adapted = _trainer_batch(batch)
            obs = adapted["obs"].to(trainer.device)
            act = adapted["act"].to(trainer.device)
            forward = trainer.agent.forward_batch(
                obs,
                act,
                routing_tau=routing_tau,
                macro_chart_tau=trainer.config.macro_chart_tau,
                macro_code_tau=trainer.config.macro_code_tau,
            )

            obs_chart = forward["obs"]["chart_idx_valid"].reshape(-1).detach().cpu()
            obs_code = forward["obs"]["code_idx_valid"].reshape(-1).detach().cpu()
            act_chart = forward["act"]["chart_idx_valid"].reshape(-1).detach().cpu()
            act_code = forward["act"]["code_idx_valid"].reshape(-1).detach().cpu()

            for chart in range(obs_num_charts):
                mask = obs_chart == chart
                if mask.any():
                    obs_codes[chart].update(int(code) for code in obs_code[mask].tolist())
            for chart in range(act_num_charts):
                mask = act_chart == chart
                if mask.any():
                    act_codes[chart].update(int(code) for code in act_code[mask].tolist())

    if was_training:
        trainer.agent.train()

    return {
        "obs": [len(codes) for codes in obs_codes],
        "act": [len(codes) for codes in act_codes],
    }


def _make_vla_config(
    args: argparse.Namespace,
    *,
    input_dim: int,
    hidden_dim: int,
    latent_dim: int,
    num_charts: int,
    codes_per_chart: int,
) -> VLAConfig:
    """Build one ``VLAConfig`` for either the observation or action manifold."""
    return VLAConfig(
        input_dim=input_dim,
        feature_dim=input_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_charts=num_charts,
        codes_per_chart=codes_per_chart,
        soft_equiv_metric=True,
        commitment_beta=args.commitment_beta,
        codebook_loss_weight=args.codebook_loss_weight,
        w_feature_recon=args.w_recon,
        w_vq=args.w_vq,
        w_entropy=args.w_entropy,
        w_diversity=args.w_diversity,
        chart_usage_entropy_low=args.chart_usage_h_low,
        chart_usage_entropy_high=args.chart_usage_h_high,
        w_chart_ot=args.w_chart_ot,
        chart_ot_epsilon=args.chart_ot_epsilon,
        chart_ot_iters=args.chart_ot_iters,
        w_uniformity=args.w_uniformity,
        w_radial_calibration=args.w_radial_cal,
        w_confidence_calibration=args.w_confidence_calibration,
        w_hard_routing_nll=args.w_hard_routing_nll,
        w_router_margin=args.w_router_margin,
        router_margin_target=args.router_margin_target,
        radial_quality_alpha=args.radial_quality_alpha,
        radial_vq_alpha=args.radial_vq_alpha,
        radial_quality_rank_mix=args.radial_quality_rank_mix,
        radial_recon_quality_weight=args.radial_recon_quality_weight,
        radial_quality_mix=args.radial_quality_mix,
        radial_quality_base_weight=args.radial_quality_base_weight,
        radial_calibration_rho_max=args.radial_calibration_rho_max,
        radial_calibration_band_width=args.radial_calibration_band_width,
        w_v_tangent_barrier=args.w_v_tangent_barrier,
        v_tangent_barrier_radius=args.v_tangent_barrier_radius,
        w_codebook_spread=args.w_codebook_spread,
        w_codebook_center=args.w_codebook_center,
        w_chart_center_mean=args.w_chart_center_mean,
        w_chart_center_radius=args.w_chart_center_radius,
        chart_center_radius_max=args.chart_center_radius_max,
        w_chart_center_sep=args.w_chart_center_sep,
        chart_center_sep_margin=args.chart_center_sep_margin,
        w_chart_collapse=args.w_chart_collapse,
        w_code_collapse=args.w_code_collapse,
        code_usage_entropy_low=args.code_usage_h_low,
        code_usage_entropy_high=args.code_usage_h_high,
        w_code_collapse_temperature=args.code_usage_temperature,
        w_window=args.w_window,
        w_window_eps_ground=args.w_window_eps_ground,
        w_consistency=args.w_consistency,
        w_jump=args.w_jump,
        w_jump_warmup=args.w_jump_warmup,
        w_jump_ramp_end=args.w_jump_ramp_end,
        w_perp=args.w_perp,
        lr_chart_centers_scale=args.lr_chart_centers_scale,
        lr_codebook_scale=args.lr_codebook_scale,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=str(_resolve_device(args.device)),
    )


def _build_agent_and_trainer(
    args: argparse.Namespace,
    *,
    obs_dim: int,
    act_dim: int,
    num_train_batches: int,
) -> tuple[FragileAgent, FragileAgentTrainer]:
    """Construct the new geometry model/trainer pair from CLI args."""
    act_hidden_dim = int(_value_or(args.hidden_dim, args.act_hidden_dim))
    act_latent_dim = int(_value_or(args.latent_dim, args.act_latent_dim))
    act_num_charts = int(_value_or(args.num_charts, args.act_num_charts))
    act_codes_per_chart = int(_value_or(args.codes_per_chart, args.act_codes_per_chart))

    obs_config = _make_vla_config(
        args,
        input_dim=obs_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
    )
    act_config = _make_vla_config(
        args,
        input_dim=act_dim,
        hidden_dim=act_hidden_dim,
        latent_dim=act_latent_dim,
        num_charts=act_num_charts,
        codes_per_chart=act_codes_per_chart,
    )

    agent = FragileAgent(
        FragileAgentConfig(
            obs_encoder=obs_config,
            act_encoder=act_config,
            enclosure_hidden_dim=args.enclosure_hidden_dim,
            enclosure_dropout=args.enclosure_dropout,
            enclosure_alpha=args.enclosure_alpha_max,
        ),
    )
    trainer = FragileAgentTrainer(
        agent,
        FragileAgentTrainerConfig(
            lr_encoder=args.lr,
            lr_probe=args.lr_probe,
            lr_markov=args.lr_markov,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            lr_chart_centers_scale=args.lr_chart_centers_scale,
            lr_codebook_scale=args.lr_codebook_scale,
            use_cosine_lr=args.use_cosine_lr,
            cosine_t_max=args.epochs,
            cosine_eta_min=args.cosine_eta_min,
            routing_tau=args.routing_tau,
            routing_tau_end=args.routing_tau_end,
            routing_tau_anneal_steps=max(args.routing_tau_anneal_epochs, 0) * max(
                num_train_batches,
                1,
            ),
            eval_routing_tau=args.eval_routing_tau,
            macro_chart_tau=args.macro_chart_tau,
            macro_code_tau=args.macro_code_tau,
            weight_obs_phase1=args.w_obs_phase1,
            weight_act_phase1=args.w_act_phase1,
            weight_enclosure_encoder=args.w_enclosure_encoder,
            weight_enclosure_probe=args.w_enclosure_probe,
            weight_markov_transition=args.w_markov_transition,
            weight_markov_shape=args.w_markov_shape,
            enclosure_alpha_max=args.enclosure_alpha_max,
            enclosure_alpha_warmup_steps=args.enclosure_alpha_warmup_steps,
        ),
    )
    return agent, trainer


def _checkpoint_payload(
    trainer: FragileAgentTrainer,
    args: argparse.Namespace,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
) -> dict[str, Any]:
    """Build the checkpoint payload for periodic and final saves."""
    return {
        "epoch": epoch,
        "global_step": trainer.global_step,
        "agent_state": trainer.agent.state_dict(),
        "encoder_optimizer": trainer.encoder_optimizer.state_dict(),
        "probe_optimizer": trainer.probe_optimizer.state_dict(),
        "markov_optimizer": trainer.markov_optimizer.state_dict(),
        "encoder_scheduler": (
            trainer.encoder_scheduler.state_dict() if trainer.encoder_scheduler is not None else None
        ),
        "args": vars(args),
        "agent_config": copy.deepcopy(trainer.agent.config),
        "trainer_config": copy.deepcopy(trainer.config),
        "train_metrics": dict(train_metrics),
        "eval_metrics": dict(eval_metrics),
    }


def _save_checkpoint(
    path: Path,
    trainer: FragileAgentTrainer,
    args: argparse.Namespace,
    *,
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
) -> None:
    """Save a geometry-training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        _checkpoint_payload(
            trainer,
            args,
            epoch=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
        ),
        path,
    )
    print(f"  Saved checkpoint: {path}")


def _load_resume_checkpoint(
    trainer: FragileAgentTrainer,
    resume_path: str,
    *,
    device: torch.device,
) -> int:
    """Load trainer/model state and return the first epoch to run next."""
    ckpt = load_checkpoint(resume_path)
    trainer.agent.load_state_dict(ckpt["agent_state"])
    load_optimizer_state(trainer.encoder_optimizer, ckpt.get("encoder_optimizer"), device)
    load_optimizer_state(trainer.probe_optimizer, ckpt.get("probe_optimizer"), device)
    load_optimizer_state(trainer.markov_optimizer, ckpt.get("markov_optimizer"), device)
    if trainer.encoder_scheduler is not None and ckpt.get("encoder_scheduler") is not None:
        trainer.encoder_scheduler.load_state_dict(ckpt["encoder_scheduler"])
    trainer.global_step = int(ckpt.get("global_step", 0))
    start_epoch = max(int(ckpt.get("epoch", -1)) + 1, 0)
    print(
        f"Resumed from {resume_path} "
        f"(epoch {ckpt.get('epoch', '?')}, global_step {trainer.global_step})",
    )
    return start_epoch


def _run_train_epoch(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
    *,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch and average the per-batch metrics."""
    batch_metrics = []
    for batch in loader:
        batch_metrics.append(trainer.train_step(_trainer_batch(batch), epoch=epoch))
    if trainer.encoder_scheduler is not None:
        trainer.encoder_scheduler.step()
    return _average_metrics(batch_metrics)


def _run_eval_epoch(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
    *,
    epoch: int,
) -> tuple[dict[str, float], dict[str, list[int]]]:
    """Run one evaluation epoch and return averaged metrics plus code activity."""
    metrics = _average_metrics(
        [trainer.eval_step(_trainer_batch(batch), epoch=epoch) for batch in loader],
    )
    code_activity = _collect_code_activity(trainer, loader)
    return metrics, code_activity


def _print_startup_summary(
    agent: FragileAgent,
    *,
    train_dataset: VLAFeatureDataset,
    eval_dataset: VLAFeatureDataset,
    eval_split: str,
    obs_dim: int,
    act_dim: int,
    sequence_length: int,
) -> None:
    """Print dataset and parameter-count summary before training starts."""
    obs_stack = count_parameters(agent.obs_encoder) + count_parameters(agent.obs_jump_operator)
    act_stack = count_parameters(agent.act_encoder) + count_parameters(agent.act_jump_operator)
    probe_params = count_parameters(agent.enclosure_probe)
    markov_params = count_parameters(agent.macro_model)
    total_params = obs_stack + act_stack + probe_params + markov_params

    print(f"Train windows: {len(train_dataset)} across {len(train_dataset.episode_ids)} episodes")
    print(
        f"Eval windows:  {len(eval_dataset)} across {len(eval_dataset.episode_ids)} episodes "
        f"(split={eval_split})",
    )
    print(f"Sequence length: {sequence_length}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {act_dim}")
    print(f"  Obs stack:  {obs_stack:>10,} params")
    print(f"  Act stack:  {act_stack:>10,} params")
    print(f"  Enclosure:  {probe_params:>10,} params")
    print(f"  Markov:     {markov_params:>10,} params")
    print(f"  TOTAL:      {total_params:>10,} params")


def train_geometry(args: argparse.Namespace) -> None:
    """Train the new geometry stack on cached VLA feature/action windows."""
    if args.epochs <= 0:
        msg = "--epochs must be positive."
        raise ValueError(msg)
    if args.batch_size <= 0:
        msg = "--batch-size must be positive."
        raise ValueError(msg)
    if args.log_every <= 0:
        msg = "--log-every must be positive."
        raise ValueError(msg)
    if args.eval_every <= 0:
        msg = "--eval-every must be positive."
        raise ValueError(msg)
    if args.sequence_length < 2:
        msg = "train_geometry requires --sequence-length >= 2 so transition losses are active."
        raise ValueError(msg)

    device = _resolve_device(args.device)
    print(f"Device: {device}")

    train_dataset = VLAFeatureDataset(
        args.feature_cache_dir,
        sequence_length=args.sequence_length,
        split="train",
    )
    if len(train_dataset) == 0:
        msg = (
            "The train split has no valid windows. "
            "Check that the feature cache exists and that sequence_length fits the episodes."
        )
        raise RuntimeError(msg)

    test_dataset = VLAFeatureDataset(
        args.feature_cache_dir,
        sequence_length=args.sequence_length,
        split="test",
    )
    if len(test_dataset) > 0:
        eval_dataset = test_dataset
        eval_split = "test"
    else:
        eval_dataset = train_dataset
        eval_split = "train"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    sample = train_dataset[0]
    obs_dim = int(sample["features"].shape[-1])
    act_dim = int(sample["actions"].shape[-1])

    agent, trainer = _build_agent_and_trainer(
        args,
        obs_dim=obs_dim,
        act_dim=act_dim,
        num_train_batches=len(train_loader),
    )
    trainer.agent.to(device)

    _print_startup_summary(
        trainer.agent,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        eval_split=eval_split,
        obs_dim=obs_dim,
        act_dim=act_dim,
        sequence_length=args.sequence_length,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = _load_resume_checkpoint(trainer, args.resume, device=device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    last_train_metrics: dict[str, float] = {}
    last_eval_metrics: dict[str, float] = {}
    for epoch in range(start_epoch, args.epochs):
        train_metrics = _run_train_epoch(trainer, train_loader, epoch=epoch)
        should_eval = (epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
        if should_eval:
            eval_metrics, code_activity = _run_eval_epoch(trainer, eval_loader, epoch=epoch)
            last_eval_metrics = eval_metrics
        else:
            eval_metrics = last_eval_metrics
            code_activity = None

        should_log = (epoch % args.log_every == 0) or (epoch == args.epochs - 1)
        if should_log:
            eval_display = (
                _format_metric_value(eval_metrics.get("loss/main", 0.0))
                if should_eval
                else "skipped"
            )
            print(
                f"Geometry E{epoch:05d} | "
                f"train={_format_metric_value(train_metrics.get('loss/main', 0.0))} | "
                f"eval={eval_display} | "
                f"step={trainer.global_step}",
            )
            _print_metric_groups("Train metrics", train_metrics)
            if should_eval:
                _print_metric_groups("Eval metrics", eval_metrics)
                print(
                    "  obs active codes/chart: "
                    f"{code_activity['obs']} / {trainer.agent.config.obs_encoder.codes_per_chart}",
                )
                print(
                    "  act active codes/chart: "
                    f"{code_activity['act']} / {trainer.agent.config.act_encoder.codes_per_chart}",
                )
            else:
                print(f"Eval metrics: skipped (runs every {args.eval_every} epochs)")
            print("-" * 80)
        last_train_metrics = train_metrics

        should_save = (
            args.save_every > 0
            and (((epoch + 1) % args.save_every == 0) or (epoch == args.epochs - 1))
        )
        if should_save:
            _save_checkpoint(
                output_dir / f"geometry_epoch_{epoch:05d}.pt",
                trainer,
                args,
                epoch=epoch,
                train_metrics=train_metrics,
                eval_metrics=eval_metrics,
            )

    final_path = output_dir / "geometry_final.pt"
    _save_checkpoint(
        final_path,
        trainer,
        args,
        epoch=args.epochs - 1,
        train_metrics=last_train_metrics,
        eval_metrics=last_eval_metrics,
    )
    print(f"Final checkpoint saved to {final_path}")


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the geometry trainer."""
    parser = argparse.ArgumentParser(description="Sequence-based geometry training with FragileAgent")

    parser.add_argument("--feature-cache-dir", default="outputs/vla/features")
    parser.add_argument("--output-dir", default="outputs/vla/geometry")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sequence-length", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3, help="Encoder LR")
    parser.add_argument("--lr-probe", type=float, default=3e-3)
    parser.add_argument("--lr-markov", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--lr-chart-centers-scale",
        type=float,
        default=0.1,
        help="LR scale for chart centers relative to the encoder LR",
    )
    parser.add_argument(
        "--lr-codebook-scale",
        type=float,
        default=0.5,
        help="LR scale for codebook parameters relative to the encoder LR",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--cosine-lr",
        "--use-scheduler",
        dest="use_cosine_lr",
        action="store_true",
        help="Cosine anneal the encoder LR over training",
    )
    parser.add_argument(
        "--eta-min",
        "--cosine-eta-min",
        dest="cosine_eta_min",
        type=float,
        default=1e-6,
        help="Minimum encoder LR for cosine scheduling",
    )

    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-charts", type=int, default=8)
    parser.add_argument("--codes-per-chart", type=int, default=32)
    parser.add_argument("--act-hidden-dim", type=int, default=None)
    parser.add_argument("--act-latent-dim", type=int, default=None)
    parser.add_argument("--act-num-charts", type=int, default=None)
    parser.add_argument("--act-codes-per-chart", type=int, default=None)

    parser.add_argument("--commitment-beta", type=float, default=0.25)
    parser.add_argument("--codebook-loss-weight", type=float, default=1.0)

    parser.add_argument("--routing-tau", "--hard-routing-tau", type=float, default=1.0)
    parser.add_argument("--routing-tau-end", "--hard-routing-tau-end", type=float, default=1.0)
    parser.add_argument(
        "--routing-tau-anneal-epochs",
        "--hard-routing-tau-anneal-epochs",
        type=int,
        default=0,
    )
    parser.add_argument("--eval-routing-tau", type=float, default=1.0)

    parser.add_argument("--w-recon", type=float, default=1.0)
    parser.add_argument("--w-vq", type=float, default=1.0)
    parser.add_argument("--w-entropy", type=float, default=0.3)
    parser.add_argument("--w-consistency", type=float, default=0.0)
    parser.add_argument("--w-diversity", type=float, default=1.0)
    parser.add_argument("--chart-usage-h-low", type=float, default=None)
    parser.add_argument("--chart-usage-h-high", type=float, default=None)
    parser.add_argument("--w-chart-ot", type=float, default=1.0)
    parser.add_argument("--chart-ot-epsilon", type=float, default=0.05)
    parser.add_argument("--chart-ot-iters", type=int, default=20)
    parser.add_argument("--w-uniformity", type=float, default=0.05)
    parser.add_argument("--w-radial-cal", type=float, default=0.1)
    parser.add_argument("--w-confidence-calibration", type=float, default=0.05)
    parser.add_argument("--w-hard-routing-nll", type=float, default=0.5)
    parser.add_argument("--w-router-margin", type=float, default=2.0)
    parser.add_argument("--router-margin-target", type=float, default=0.05)
    parser.add_argument("--radial-quality-alpha", type=float, default=2.0)
    parser.add_argument("--radial-vq-alpha", type=float, default=1.0)
    parser.add_argument("--radial-quality-rank-mix", type=float, default=0.75)
    parser.add_argument("--radial-recon-quality-weight", type=float, default=0.7)
    parser.add_argument("--radial-quality-mix", type=float, default=1.0)
    parser.add_argument("--radial-quality-base-weight", type=float, default=0.0)
    parser.add_argument("--radial-calibration-rho-max", type=float, default=4.0)
    parser.add_argument("--radial-calibration-band-width", type=float, default=0.75)
    parser.add_argument("--w-v-tangent-barrier", type=float, default=0.01)
    parser.add_argument("--v-tangent-barrier-radius", type=float, default=0.9)
    parser.add_argument("--w-codebook-spread", type=float, default=0.05)
    parser.add_argument("--w-codebook-center", type=float, default=0.02)
    parser.add_argument("--w-chart-center-mean", type=float, default=0.02)
    parser.add_argument("--w-chart-center-radius", type=float, default=0.05)
    parser.add_argument("--chart-center-radius-max", type=float, default=2.0)
    parser.add_argument("--w-chart-center-sep", type=float, default=0.02)
    parser.add_argument("--chart-center-sep-margin", type=float, default=1.0)
    parser.add_argument("--w-chart-collapse", type=float, default=0.0)
    parser.add_argument("--w-code-collapse", type=float, default=0.5)
    parser.add_argument("--code-usage-h-low", type=float, default=None)
    parser.add_argument("--code-usage-h-high", type=float, default=None)
    parser.add_argument("--code-usage-temperature", type=float, default=1.0)
    parser.add_argument("--w-window", type=float, default=0.0)
    parser.add_argument("--w-window-eps-ground", type=float, default=0.1)
    parser.add_argument("--w-jump", type=float, default=0.0)
    parser.add_argument("--w-jump-warmup", type=int, default=20)
    parser.add_argument("--w-jump-ramp-end", type=int, default=50)
    parser.add_argument("--w-perp", type=float, default=0.01)

    parser.add_argument("--w-obs-phase1", type=float, default=1.0)
    parser.add_argument("--w-act-phase1", type=float, default=1.0)
    parser.add_argument("--w-enclosure-encoder", type=float, default=1.0)
    parser.add_argument("--w-enclosure-probe", type=float, default=1.0)
    parser.add_argument("--w-markov-transition", type=float, default=1.0)
    parser.add_argument("--w-markov-shape", type=float, default=1.0)
    parser.add_argument("--macro-chart-tau", type=float, default=1.0)
    parser.add_argument("--macro-code-tau", type=float, default=1.0)
    parser.add_argument("--enclosure-hidden-dim", type=int, default=128)
    parser.add_argument("--enclosure-dropout", type=float, default=0.1)
    parser.add_argument("--enclosure-alpha-max", type=float, default=1.0)
    parser.add_argument("--enclosure-alpha-warmup-steps", type=int, default=5000)

    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--resume", default="", help="Checkpoint path to resume from")
    parser.add_argument("--device", default="auto")

    return parser


def main() -> None:
    """CLI entrypoint for geometry training."""
    parser = _build_parser()
    args = parser.parse_args()
    train_geometry(args)


if __name__ == "__main__":
    main()
