"""CLI for sequence-based geometry training with ``FragileAgent``.

This command replaces the old hand-written Phase-1 loop with a simpler stack:

- observation and action sequences come from cached VLA windows,
- ``FragileAgent`` owns both topoencoders, the enclosure probe, and the
  symbolic Markov model,
- ``FragileAgentTrainer`` computes all losses and metrics,
- the CLI only handles config assembly, dataloading, logging, and checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import MISSING
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fragile.agent import (
    FragileAgent,
    FragileAgentConfig,
    FragileAgentTrainer,
    FragileAgentTrainerConfig,
)
from fragile.checkpoints import (
    count_parameters,
    load_geometry_resume_checkpoint,
    save_geometry_checkpoint,
)
from fragile.metrics import average_metrics, format_metric_value, print_metric_groups
from fragile.vla.extract_features import VLAFeatureDataset


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested training device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _trainer_batch(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Adapt sequence-mode dataset batches to ``FragileAgentTrainer`` inputs."""
    return {
        "obs": batch["features"],
        "act": batch["actions"],
    }


def _effective_window_stride(sequence_length: int, window_stride: int) -> int:
    """Resolve the sequence stride, defaulting to non-overlapping chunks."""
    if sequence_length <= 1:
        return 1
    if window_stride in {None, 0}:
        return int(sequence_length)
    return int(window_stride)


def _resolve_phase1_frame_mode(
    phase1_frame_mode: str,
    *,
    sequence_length: int,
    window_stride: int,
) -> str:
    """Choose how frame-local Phase-1 losses are applied inside a sequence batch."""
    if phase1_frame_mode != "auto":
        return str(phase1_frame_mode)
    if sequence_length > 1 and window_stride < sequence_length:
        return "anchor"
    return "all"


def _dataset_stats(
    dataset: VLAFeatureDataset,
    *,
    key: str,
    min_std: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute dataset-level mean/std over preloaded episode tensors."""
    if key == "features":
        tensors = dataset._features
    elif key == "actions":
        tensors = dataset._actions
    else:
        msg = "key must be one of {'features', 'actions'}."
        raise ValueError(msg)
    if not tensors:
        msg = f"No tensors available for {key} stats."
        raise RuntimeError(msg)
    flat = torch.cat([tensor.reshape(-1, tensor.shape[-1]).float() for tensor in tensors], dim=0)
    mean = flat.mean(dim=0)
    std = flat.std(dim=0, unbiased=False).clamp_min(float(min_std))
    return mean, std


def _print_code_activity(
    prefix: str,
    code_activity: dict[str, list[Any]],
    trainer: FragileAgentTrainer,
) -> None:
    """Print per-chart code usage in `num_active/num_total [p0, ...]` format."""

    def _format_chart_distribution(chart_counts: Any, total_codes: int) -> str:
        if isinstance(chart_counts, (int, float)):
            active = int(chart_counts)
            distribution = ", ".join(
                (f"{round(100.0 / float(active)):02d}" if idx < active and active > 0 else "00")
                for idx in range(int(total_codes))
            )
            return f"{active}/{int(total_codes)} [{distribution}]"
        if torch.is_tensor(chart_counts):
            counts = chart_counts.detach().cpu().to(dtype=torch.float32)
        else:
            counts = torch.as_tensor(chart_counts, dtype=torch.float32)
        active = int((counts > 0).sum().item())
        probs = counts / counts.sum().clamp_min(1.0)
        distribution = ", ".join(f"{round(100.0 * float(value)):02d}" for value in probs.tolist())
        return f"{active}/{int(total_codes)} [{distribution}]"

    obs_total = trainer.agent.config.obs_encoder.codes_per_chart
    act_total = trainer.agent.config.act_encoder.codes_per_chart
    print(f"  {prefix} obs codes/chart:")
    for chart_idx, chart_counts in enumerate(code_activity["obs"]):
        print(f"    c{chart_idx:02d} {_format_chart_distribution(chart_counts, obs_total)}")
    print(f"  {prefix} act codes/chart:")
    for chart_idx, chart_counts in enumerate(code_activity["act"]):
        print(f"    c{chart_idx:02d} {_format_chart_distribution(chart_counts, act_total)}")


def _collect_code_activity(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
) -> dict[str, list[Any]]:
    """Compatibility helper for tests and profiling.

    The main geometry loop no longer rescans loaders for code usage, but the
    profiling tests still time this helper explicitly. Keep it as a thin,
    opt-in wrapper around `eval_step` so those tests can continue to benchmark
    the old extra-pass path without affecting the normal runtime.
    """
    code_activity_acc = trainer.init_code_activity_accumulator()
    for batch in loader:
        trainer.eval_step(_trainer_batch(batch), code_activity_accumulator=code_activity_acc)
    return trainer.finalize_code_activity(code_activity_acc)


def _metric_improved(
    value: float,
    best_value: float | None,
    *,
    mode: str,
    min_delta: float,
) -> bool:
    """Check whether a new metric value improves over the current best."""
    if best_value is None:
        return True
    if mode == "max":
        return value > (best_value + min_delta)
    if mode == "min":
        return value < (best_value - min_delta)
    msg = "mode must be 'max' or 'min'."
    raise ValueError(msg)


def _run_train_epoch(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
    *,
    epoch: int,
) -> tuple[dict[str, float], dict[str, list[Any]]]:
    """Run one training epoch and average the per-batch metrics plus code activity."""
    batch_metrics = []
    code_activity_acc = trainer.init_code_activity_accumulator()
    for batch in loader:
        batch_metrics.append(
            trainer.train_step(
                _trainer_batch(batch),
                epoch=epoch,
                code_activity_accumulator=code_activity_acc,
            ),
        )
    if trainer.encoder_scheduler is not None:
        trainer.encoder_scheduler.step()
    return average_metrics(batch_metrics), trainer.finalize_code_activity(code_activity_acc)


def _run_eval_epoch(
    trainer: FragileAgentTrainer,
    loader: DataLoader,
    *,
    epoch: int,
) -> tuple[dict[str, float], dict[str, list[Any]]]:
    """Run one evaluation epoch and return averaged metrics plus code activity."""
    code_activity_acc = trainer.init_code_activity_accumulator()
    metrics = average_metrics(
        [
            trainer.eval_step(
                _trainer_batch(batch),
                epoch=epoch,
                code_activity_accumulator=code_activity_acc,
            )
            for batch in loader
        ],
    )
    return metrics, trainer.finalize_code_activity(code_activity_acc)


def _print_startup_summary(
    agent: FragileAgent,
    *,
    train_dataset: VLAFeatureDataset,
    eval_dataset: VLAFeatureDataset,
    eval_split: str,
    obs_dim: int,
    act_dim: int,
    sequence_length: int,
    window_stride: int,
    phase1_frame_mode: str,
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
    print(f"Window stride:   {window_stride}")
    print(f"Frame loss mode: {phase1_frame_mode}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim:      {act_dim}")
    print(f"  Obs stack:  {obs_stack:>10,} params")
    print(f"  Act stack:  {act_stack:>10,} params")
    print(f"  Enclosure:  {probe_params:>10,} params")
    print(f"  Markov:     {markov_params:>10,} params")
    print(f"  TOTAL:      {total_params:>10,} params")


@dataclass
class GeometryTrainingRunner:
    """Declarative geometry training runner instantiated from Hydra config.

    All nested configs (``agent``, ``trainer``) are ``_target_``-instantiated by
    Hydra before being passed here.  Runtime-computed values (``input_dim``,
    ``device``, ``routing_tau_anneal_steps``, ``phase1_frame_mode``) are patched
    inside ``run()`` after inspecting the dataset. The YAML config is the source
    of truth for command defaults; these fields are marked missing here on
    purpose so new defaults are added in one place only.
    """

    # Paths
    feature_cache_dir: str = MISSING
    output_dir: str = MISSING
    # Training loop
    epochs: int = MISSING
    batch_size: int = MISSING
    sequence_length: int = MISSING
    window_stride: int = MISSING
    phase1_frame_mode: str = MISSING
    device: str = MISSING
    # Logging / checkpoints
    log_every: int = MISSING
    eval_every: int = MISSING
    save_every: int = MISSING
    resume: str = MISSING
    # Early stopping
    best_eval_metric: str = MISSING
    best_eval_mode: str = MISSING
    early_stop_patience: int = MISSING
    early_stop_min_epochs: int = MISSING
    early_stop_min_delta: float = MISSING
    # Epoch → step conversion
    routing_tau_anneal_epochs: int = MISSING
    # Nested configs (instantiated by Hydra from _target_)
    agent: FragileAgentConfig = MISSING
    trainer: FragileAgentTrainerConfig = MISSING

    def _config_dict(self) -> dict[str, Any]:
        """Serializable snapshot of the runner config for checkpoints."""
        from dataclasses import asdict

        return asdict(self)

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Execute the full geometry training loop."""
        # --- Validate ---
        if self.epochs <= 0:
            msg = "epochs must be positive."
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if self.log_every <= 0:
            msg = "log_every must be positive."
            raise ValueError(msg)
        if self.eval_every <= 0:
            msg = "eval_every must be positive."
            raise ValueError(msg)
        if self.window_stride is not None and int(self.window_stride) < 0:
            msg = "window_stride must be non-negative."
            raise ValueError(msg)
        if self.early_stop_patience < 0:
            msg = "early_stop_patience must be non-negative."
            raise ValueError(msg)
        if self.early_stop_min_epochs < 0:
            msg = "early_stop_min_epochs must be non-negative."
            raise ValueError(msg)
        if self.sequence_length < 2:
            msg = "sequence_length must be >= 2 so transition losses are active."
            raise ValueError(msg)

        device = _resolve_device(self.device)
        print(f"Device: {device}")
        window_stride = _effective_window_stride(self.sequence_length, self.window_stride)
        phase1_frame_mode = _resolve_phase1_frame_mode(
            self.phase1_frame_mode,
            sequence_length=self.sequence_length,
            window_stride=window_stride,
        )

        # --- Datasets ---
        train_dataset = VLAFeatureDataset(
            self.feature_cache_dir,
            sequence_length=self.sequence_length,
            window_stride=window_stride,
            split="train",
        )
        if len(train_dataset) == 0:
            msg = (
                "The train split has no valid windows. "
                "Check that the feature cache exists and that sequence_length fits the episodes."
            )
            raise RuntimeError(msg)

        test_dataset = VLAFeatureDataset(
            self.feature_cache_dir,
            sequence_length=self.sequence_length,
            window_stride=window_stride,
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
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )

        # --- Patch runtime-computed config values ---
        sample = train_dataset[0]
        obs_dim = int(sample["features"].shape[-1])
        act_dim = int(sample["actions"].shape[-1])

        self.agent.obs_encoder.input_dim = obs_dim
        self.agent.obs_encoder.feature_dim = obs_dim
        self.agent.obs_encoder.device = str(device)
        self.agent.obs_encoder.batch_size = self.batch_size
        self.agent.obs_encoder.sequence_length = self.sequence_length

        self.agent.act_encoder.input_dim = act_dim
        self.agent.act_encoder.feature_dim = act_dim
        self.agent.act_encoder.device = str(device)
        self.agent.act_encoder.batch_size = self.batch_size
        self.agent.act_encoder.sequence_length = self.sequence_length

        num_train_batches = len(train_loader)
        self.trainer.routing_tau_anneal_steps = max(self.routing_tau_anneal_epochs, 0) * max(
            num_train_batches, 1
        )
        self.trainer.cosine_t_max = self.epochs
        self.trainer.phase1_frame_mode = phase1_frame_mode

        # --- Build agent and trainer ---
        agent = FragileAgent(self.agent)
        trainer = FragileAgentTrainer(agent, self.trainer)

        act_mean, act_std = _dataset_stats(
            train_dataset,
            key="actions",
            min_std=trainer.agent.config.act_encoder.input_affine_min_scale,
        )
        trainer.agent.act_encoder.set_io_affine_stats(act_mean, act_std, learnable=False)
        trainer.agent.to(device)

        print(
            "Action affine stats: "
            f"mean_abs={act_mean.abs().mean().item():.4f} "
            f"std_min={act_std.min().item():.4f} "
            f"std_mean={act_std.mean().item():.4f} "
            f"std_max={act_std.max().item():.4f}",
        )

        _print_startup_summary(
            trainer.agent,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_split=eval_split,
            obs_dim=obs_dim,
            act_dim=act_dim,
            sequence_length=self.sequence_length,
            window_stride=window_stride,
            phase1_frame_mode=phase1_frame_mode,
        )

        # --- Resume ---
        config_dict = self._config_dict()
        start_epoch = 0
        best_eval_metric_name = self.best_eval_metric
        best_eval_metric_value: float | None = None
        best_eval_epoch: int | None = None
        evals_since_improvement = 0
        if self.resume:
            (
                start_epoch,
                resumed_best_name,
                resumed_best_value,
                resumed_best_epoch,
            ) = load_geometry_resume_checkpoint(trainer, self.resume, device=device)
            if resumed_best_name is not None:
                best_eval_metric_name = resumed_best_name
            if resumed_best_value is not None:
                best_eval_metric_value = float(resumed_best_value)
            if resumed_best_epoch is not None:
                best_eval_epoch = int(resumed_best_epoch)

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # --- Epoch loop ---
        last_train_metrics: dict[str, float] = {}
        last_eval_metrics: dict[str, float] = {}
        last_eval_code_activity: dict[str, list[Any]] = {"obs": [], "act": []}
        last_epoch = start_epoch - 1
        epoch_iter = tqdm(
            range(start_epoch, self.epochs),
            desc="Geometry",
            unit="epoch",
            initial=start_epoch,
            total=self.epochs,
        )
        for epoch in epoch_iter:
            last_epoch = epoch
            train_metrics, train_code_activity = _run_train_epoch(
                trainer, train_loader, epoch=epoch
            )
            # Update tqdm postfix with key metrics
            postfix = {"loss": format_metric_value(train_metrics.get("loss/main", 0.0))}
            should_eval = (epoch % self.eval_every == 0) or (epoch == self.epochs - 1)
            if should_eval:
                eval_metrics, code_activity = _run_eval_epoch(trainer, eval_loader, epoch=epoch)
                last_eval_metrics = eval_metrics
                last_eval_code_activity = code_activity
                eval_score = eval_metrics.get(best_eval_metric_name)
                if eval_score is None:
                    print(
                        f"Best-eval metric '{best_eval_metric_name}' missing from eval metrics; "
                        "skipping best-checkpoint/early-stop update.",
                    )
                else:
                    eval_score = float(eval_score)
                    if _metric_improved(
                        eval_score,
                        best_eval_metric_value,
                        mode=self.best_eval_mode,
                        min_delta=self.early_stop_min_delta,
                    ):
                        best_eval_metric_value = eval_score
                        best_eval_epoch = epoch
                        evals_since_improvement = 0
                        save_geometry_checkpoint(
                            output_dir / "geometry_best.pt",
                            trainer,
                            config_dict,
                            epoch=epoch,
                            train_metrics=train_metrics,
                            eval_metrics=eval_metrics,
                            best_eval_metric_name=best_eval_metric_name,
                            best_eval_metric_value=best_eval_metric_value,
                            best_eval_epoch=best_eval_epoch,
                        )
                        print(
                            "  New best eval checkpoint: "
                            f"{best_eval_metric_name}={format_metric_value(eval_score)} "
                            f"at epoch {epoch}",
                        )
                    else:
                        evals_since_improvement += 1
            else:
                eval_metrics = last_eval_metrics
                code_activity = last_eval_code_activity

            if should_eval:
                postfix["eval"] = format_metric_value(eval_metrics.get("loss/main", 0.0))
            if best_eval_metric_value is not None:
                postfix["best"] = format_metric_value(best_eval_metric_value)
            epoch_iter.set_postfix(postfix)

            should_log = (epoch % self.log_every == 0) or (epoch == self.epochs - 1)
            if should_log:
                eval_display = (
                    format_metric_value(eval_metrics.get("loss/main", 0.0))
                    if should_eval
                    else "skipped"
                )
                print(
                    f"Geometry E{epoch:05d} | "
                    f"train={format_metric_value(train_metrics.get('loss/main', 0.0))} | "
                    f"eval={eval_display} | "
                    f"step={trainer.global_step}",
                )
                print_metric_groups("Train metrics", train_metrics)
                _print_code_activity("train", train_code_activity, trainer)
                if should_eval:
                    print_metric_groups("Eval metrics", eval_metrics)
                    _print_code_activity("eval", code_activity, trainer)
                else:
                    print(f"Eval metrics: skipped (runs every {self.eval_every} epochs)")
                print("-" * 80)
            last_train_metrics = train_metrics

            should_save = self.save_every > 0 and (
                ((epoch + 1) % self.save_every == 0) or (epoch == self.epochs - 1)
            )
            if should_save:
                save_geometry_checkpoint(
                    output_dir / f"geometry_epoch_{epoch:05d}.pt",
                    trainer,
                    config_dict,
                    epoch=epoch,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                    best_eval_metric_name=best_eval_metric_name,
                    best_eval_metric_value=best_eval_metric_value,
                    best_eval_epoch=best_eval_epoch,
                )
            if (
                should_eval
                and self.early_stop_patience > 0
                and (epoch + 1) >= self.early_stop_min_epochs
                and evals_since_improvement >= self.early_stop_patience
            ):
                print(
                    "Early stopping: "
                    f"no improvement in {best_eval_metric_name} for "
                    f"{evals_since_improvement} evals "
                    f"(best epoch {best_eval_epoch}, "
                    f"value={format_metric_value(best_eval_metric_value or 0.0)}).",
                )
                break

        final_path = output_dir / "geometry_final.pt"
        save_geometry_checkpoint(
            final_path,
            trainer,
            config_dict,
            epoch=last_epoch,
            train_metrics=last_train_metrics,
            eval_metrics=last_eval_metrics,
            best_eval_metric_name=best_eval_metric_name,
            best_eval_metric_value=best_eval_metric_value,
            best_eval_epoch=best_eval_epoch,
        )
        print(f"Final checkpoint saved to {final_path}")


# ---------------------------------------------------------------------------
# Config path (importable for tests)
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "train_geometry.yml"


def main() -> None:
    """CLI entrypoint for geometry training."""
    import sys

    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(CONFIG_PATH)
    if len(sys.argv) > 1:
        cli = OmegaConf.from_cli(sys.argv[1:])
        cfg = OmegaConf.merge(cfg, cli)
    runner: GeometryTrainingRunner = instantiate(cfg)
    runner.run()


if __name__ == "__main__":
    main()
