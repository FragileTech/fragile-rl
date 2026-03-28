"""Standalone off-policy macro RL with joint topoencoder training.

This runner trains the current `FragileAgent` stack from scratch on dm_control:

- observation and action topoencoders are optimized online from replay,
- the coarse Markov model is trained in parallel on the same replay windows,
- a simple Q-learning head controls discrete macro actions,
- macro actions are executed through learned continuous action prototypes.

There is intentionally no planning yet. The goal of this loop is to behave
like a standard off-policy RL algorithm while keeping the learned symbolic
geometry and coarse dynamics model on the critical path from day one.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys
from typing import Any

from omegaconf import MISSING, OmegaConf
import torch
from torch import nn
from tqdm import tqdm

from fragile.agent import (
    FragileAgent,
    FragileAgentConfig,
    FragileAgentTrainer,
    FragileAgentTrainerConfig,
)
from fragile.checkpoints import (
    compute_grad_norm,
    compute_param_norm,
    count_parameters,
    load_macro_rl_resume_checkpoint,
)
from fragile.losses.markov_model import compute_macro_auxiliary_loss
from fragile.metrics import (
    average_metrics,
    format_metric_value,
    init_symbol_usage_accumulator,
    log_epoch,
    prototype_metrics,
    summarize_collection,
    update_symbol_usage_from_episode_info,
    update_symbol_usage_from_forward,
)
from fragile.rl.env_helpers import (
    _flatten_obs,
    _infer_action_dim,
    _make_env,
    ObservationNormalizer,
)
from fragile.rl.macro_collect import collect_macro_episodes_batched
from fragile.rl.macro_control import (
    compute_q_learning_loss,
    hard_update_target,
    MacroQNetwork,
    soft_update_target,
)
from fragile.rl.macro_data import (
    action_stats_from_episodes,
    ActionPrototypeTable,
    prepare_macro_transition_batch_from_forward,
    replay_buffer_state,
    update_action_symbol_prototypes,
    update_action_symbol_prototypes_from_rollouts,
)
from fragile.rl.replay_buffer import SequenceReplayBuffer


REPO_ROOT = Path(__file__).resolve().parents[3]


def _resolve_device(device_arg: str) -> torch.device:
    """Resolve the requested device string into a concrete torch device."""
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _linear_schedule(start: float, end: float, step: int, duration: int) -> float:
    """Simple linear interpolation with clamped endpoints."""
    if duration <= 0:
        return float(end)
    mix = min(max(float(step) / float(duration), 0.0), 1.0)
    return float(start + mix * (end - start))


def _utc_now_iso() -> str:
    """Return a compact UTC timestamp string for manifests."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _jsonable(value: Any) -> Any:
    """Convert nested runtime values into JSON-safe structures."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    return str(value)


def _git_capture(*args: str) -> str | None:
    """Run one git command against the repo root and return trimmed stdout."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    output = result.stdout.strip()
    return output or None


def _git_metadata() -> dict[str, Any]:
    """Collect lightweight git metadata for reproducibility."""
    status = _git_capture("status", "--short")
    return {
        "repo_root": str(REPO_ROOT),
        "commit": _git_capture("rev-parse", "HEAD"),
        "commit_short": _git_capture("rev-parse", "--short", "HEAD"),
        "branch": _git_capture("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "status_short": [] if status is None else status.splitlines()[:200],
    }


def _system_metadata(device: torch.device | None) -> dict[str, Any]:
    """Collect host/runtime metadata that is useful when diagnosing runs."""
    cuda_available = torch.cuda.is_available()
    metadata = {
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "torch": str(torch.__version__),
        "cuda_available": cuda_available,
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "requested_device": None if device is None else str(device),
        "cwd": os.getcwd(),
    }
    if device is not None:
        metadata["resolved_device"] = str(device)
        if device.type == "cuda" and cuda_available:
            metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    return metadata


def _write_macro_rl_run_artifacts(
    *,
    output_dir: Path,
    resolved_config: dict[str, Any],
    cli_overrides: list[str],
    launch_argv: list[str],
    status: str,
    started_at_utc: str,
    config_path: Path,
    device: torch.device | None = None,
    obs_dim: int | None = None,
    act_dim: int | None = None,
    model_summary: dict[str, int] | None = None,
    env_steps: int | None = None,
    update_steps: int | None = None,
    last_epoch: int | None = None,
    final_checkpoint: Path | None = None,
    train_metrics: dict[str, float] | None = None,
    eval_metrics: dict[str, float] | None = None,
    error: BaseException | None = None,
) -> None:
    """Write a Hydra-like metadata bundle plus a JSON manifest into the run dir."""
    hydra_dir = output_dir / ".hydra"
    hydra_dir.mkdir(parents=True, exist_ok=True)

    command = shlex.join(launch_argv) if launch_argv else None
    metadata = {
        "schema_version": 1,
        "status": status,
        "started_at_utc": started_at_utc,
        "updated_at_utc": _utc_now_iso(),
        "config_path": str(config_path),
        "output_dir": str(output_dir.resolve()),
        "launch": {
            "argv": launch_argv,
            "command": command,
            "overrides": cli_overrides,
        },
        "system": _system_metadata(device),
        "git": _git_metadata(),
        "resolved_config": resolved_config,
        "dimensions": {
            "obs_dim": obs_dim,
            "act_dim": act_dim,
        },
        "model_summary": model_summary,
        "progress": {
            "last_epoch": last_epoch,
            "env_steps": env_steps,
            "update_steps": update_steps,
        },
        "artifacts": {
            "final_checkpoint": None if final_checkpoint is None else str(final_checkpoint),
        },
        "metrics": {
            "train": None if train_metrics is None else dict(train_metrics),
            "eval": None if eval_metrics is None else dict(eval_metrics),
        },
        "error": (
            None
            if error is None
            else {
                "type": type(error).__name__,
                "message": str(error),
            }
        ),
    }

    (output_dir / "run_metadata.json").write_text(
        json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n",
    )
    OmegaConf.save(config=OmegaConf.create(resolved_config), f=hydra_dir / "config.yaml")
    OmegaConf.save(config=OmegaConf.create(cli_overrides), f=hydra_dir / "overrides.yaml")
    OmegaConf.save(
        config=OmegaConf.create({
            "runtime": {
                "status": status,
                "started_at_utc": started_at_utc,
                "updated_at_utc": metadata["updated_at_utc"],
                "config_path": str(config_path),
                "output_dir": str(output_dir.resolve()),
                "cwd": os.getcwd(),
            },
            "launch": metadata["launch"],
            "system": metadata["system"],
            "git": metadata["git"],
            "dimensions": metadata["dimensions"],
            "model_summary": model_summary,
            "progress": metadata["progress"],
            "artifacts": metadata["artifacts"],
        }),
        f=hydra_dir / "hydra.yaml",
    )


def _trainer_batch_from_replay(
    replay_batch: dict[str, torch.Tensor],
    obs_normalizer: ObservationNormalizer | None,
) -> dict[str, torch.Tensor]:
    """Adapt a replay sample to the `FragileAgentTrainer` batch format."""
    obs = replay_batch["obs"]
    if obs_normalizer is not None:
        obs = obs_normalizer.normalize_tensor(obs)
    return {
        "obs": obs,
        "act": replay_batch["actions"],
    }


def _save_macro_rl_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist one macro RL checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    print(f"  Saved checkpoint: {path}")


def _build_checkpoint_payload(
    runner: MacroRLRunner,
    trainer: FragileAgentTrainer,
    q_network: MacroQNetwork,
    target_q_network: MacroQNetwork,
    q_optimizer: torch.optim.Optimizer,
    replay: SequenceReplayBuffer,
    obs_normalizer: ObservationNormalizer | None,
    action_prototypes: ActionPrototypeTable | None,
    *,
    epoch: int,
    env_steps: int,
    update_steps: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
) -> dict[str, Any]:
    """Bundle the full standalone macro RL state into one checkpoint payload."""
    return {
        "epoch": epoch,
        "env_steps": env_steps,
        "update_steps": update_steps,
        "trainer_global_step": trainer.global_step,
        "agent_state": trainer.agent.state_dict(),
        "encoder_optimizer": trainer.encoder_optimizer.state_dict(),
        "probe_optimizer": trainer.probe_optimizer.state_dict(),
        "markov_optimizer": trainer.markov_optimizer.state_dict(),
        "encoder_scheduler": (
            trainer.encoder_scheduler.state_dict()
            if trainer.encoder_scheduler is not None
            else None
        ),
        "q_state": q_network.state_dict(),
        "target_q_state": target_q_network.state_dict(),
        "q_optimizer": q_optimizer.state_dict(),
        "replay_buffer": replay_buffer_state(replay),
        "obs_normalizer": None if obs_normalizer is None else obs_normalizer.state_dict(),
        "action_prototypes": (
            None if action_prototypes is None else action_prototypes.state_dict()
        ),
        "args": runner._config_dict(),
        "agent_config": copy.deepcopy(trainer.agent.config),
        "trainer_config": copy.deepcopy(trainer.config),
        "train_metrics": dict(train_metrics),
        "eval_metrics": dict(eval_metrics),
    }


@dataclass
class MacroRLRunner:
    """Hydra-instantiated standalone macro RL runner."""

    domain: str = MISSING
    task: str = MISSING
    output_dir: str = MISSING
    resume: str = MISSING
    device: str = MISSING

    epochs: int = MISSING
    seed_episodes: int = MISSING
    collect_episodes_per_epoch: int = MISSING
    eval_episodes: int = MISSING
    num_collect_envs: int = MISSING
    num_eval_envs: int = MISSING
    updates_per_epoch: int = MISSING
    batch_size: int = MISSING
    replay_seq_len: int = MISSING
    replay_capacity: int = MISSING
    max_episode_steps: int = MISSING
    action_repeat: int = MISSING
    obs_min_std: float = MISSING
    action_min_std: float = MISSING

    log_every: int = MISSING
    eval_every: int = MISSING
    save_every: int = MISSING
    routing_tau_anneal_epochs: int = MISSING

    lr_q: float = MISSING
    q_weight_decay: float = MISSING
    q_grad_clip: float = MISSING
    gamma: float = MISSING
    q_target_update_every: int = MISSING
    q_target_tau: float = MISSING
    q_loss_type: str = MISSING
    epsilon_start: float = MISSING
    epsilon_end: float = MISSING
    epsilon_decay_epochs: int = MISSING

    prototype_refresh_every: int = MISSING
    prototype_min_count: int = MISSING
    prototype_ema: float = MISSING
    sigma_motor: float = MISSING

    weight_reward: float = MISSING
    weight_continuation: float = MISSING

    agent: FragileAgentConfig = MISSING
    trainer: FragileAgentTrainerConfig = MISSING

    def _config_dict(self) -> dict[str, Any]:
        """Return a checkpoint-serializable copy of the runner config."""
        return asdict(self)

    def _validate_config(self) -> None:
        """Check that all runner hyperparameters are consistent."""
        if self.epochs <= 0:
            msg = "epochs must be positive."
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = "batch_size must be positive."
            raise ValueError(msg)
        if self.replay_seq_len <= 0:
            msg = "replay_seq_len must be positive."
            raise ValueError(msg)
        if self.replay_capacity <= 0:
            msg = "replay_capacity must be positive."
            raise ValueError(msg)
        if self.num_collect_envs <= 0 or self.num_eval_envs <= 0:
            msg = "num_collect_envs and num_eval_envs must both be positive."
            raise ValueError(msg)
        if self.seed_episodes <= 0 and not self.resume:
            msg = "seed_episodes must be positive when starting from scratch."
            raise ValueError(msg)
        if self.updates_per_epoch <= 0:
            msg = "updates_per_epoch must be positive."
            raise ValueError(msg)
        if self.log_every <= 0 or self.eval_every <= 0:
            msg = "log_every and eval_every must both be positive."
            raise ValueError(msg)

    def _setup(
        self,
    ) -> tuple[
        list[Any],  # train_envs
        list[Any],  # eval_envs
        FragileAgentTrainer,  # trainer
        MacroQNetwork,  # q_network
        MacroQNetwork,  # target_q_network
        torch.optim.Adam,  # q_optimizer
        SequenceReplayBuffer,  # replay
        torch.device,  # device
        int,  # obs_dim
        int,  # act_dim
    ]:
        """Create environments, agent, trainer, Q-networks, and replay buffer."""
        device = _resolve_device(self.device)
        print(f"Device: {device}")

        train_envs = [_make_env(self.domain, self.task) for _ in range(self.num_collect_envs)]
        eval_envs = [_make_env(self.domain, self.task) for _ in range(self.num_eval_envs)]
        obs_dim = int(_flatten_obs(train_envs[0].reset()).shape[0])
        act_dim = int(_infer_action_dim(train_envs[0]))

        self.agent.obs_encoder.input_dim = obs_dim
        self.agent.obs_encoder.feature_dim = obs_dim
        self.agent.obs_encoder.device = str(device)
        self.agent.obs_encoder.batch_size = self.batch_size
        self.agent.obs_encoder.sequence_length = self.replay_seq_len + 1

        self.agent.act_encoder.input_dim = act_dim
        self.agent.act_encoder.feature_dim = act_dim
        self.agent.act_encoder.device = str(device)
        self.agent.act_encoder.batch_size = self.batch_size
        self.agent.act_encoder.sequence_length = self.replay_seq_len + 1

        self.trainer.routing_tau_anneal_steps = max(self.routing_tau_anneal_epochs, 0) * max(
            self.updates_per_epoch,
            1,
        )
        self.trainer.cosine_t_max = self.epochs

        agent = FragileAgent(self.agent)
        trainer = FragileAgentTrainer(agent, self.trainer)
        trainer.agent.to(device)

        q_network = MacroQNetwork(trainer.agent.num_obs_states, trainer.agent.num_act_states).to(
            device,
        )
        target_q_network = MacroQNetwork(
            trainer.agent.num_obs_states,
            trainer.agent.num_act_states,
        ).to(device)
        hard_update_target(target_q_network, q_network)
        q_optimizer = torch.optim.Adam(
            q_network.parameters(),
            lr=self.lr_q,
            weight_decay=self.q_weight_decay,
        )

        replay = SequenceReplayBuffer(capacity=self.replay_capacity, seq_len=self.replay_seq_len)

        return (
            train_envs,
            eval_envs,
            trainer,
            q_network,
            target_q_network,
            q_optimizer,
            replay,
            device,
            obs_dim,
            act_dim,
        )

    def _print_model_summary(
        self,
        trainer: FragileAgentTrainer,
        q_network: MacroQNetwork,
        *,
        obs_dim: int,
        act_dim: int,
    ) -> dict[str, int]:
        """Print environment info and return parameter counts for each module."""
        obs_stack = count_parameters(trainer.agent.obs_encoder) + count_parameters(
            trainer.agent.obs_jump_operator,
        )
        act_stack = count_parameters(trainer.agent.act_encoder) + count_parameters(
            trainer.agent.act_jump_operator,
        )
        probe_params = count_parameters(trainer.agent.enclosure_probe)
        markov_params = count_parameters(trainer.agent.macro_model)
        q_params = count_parameters(q_network)
        print(f"Environment: {self.domain}/{self.task}")
        print(f"Observation dim: {obs_dim}")
        print(f"Action dim:      {act_dim}")
        print(f"Replay seq len:  {self.replay_seq_len}")
        print(f"  Obs stack:  {obs_stack:>10,} params")
        print(f"  Act stack:  {act_stack:>10,} params")
        print(f"  Enclosure:  {probe_params:>10,} params")
        print(f"  Markov:     {markov_params:>10,} params")
        print(f"  Q head:     {q_params:>10,} params")
        return {
            "obs_stack_params": obs_stack,
            "act_stack_params": act_stack,
            "enclosure_params": probe_params,
            "markov_params": markov_params,
            "q_head_params": q_params,
        }

    def _evaluate(
        self,
        eval_envs: list[Any],
        trainer: FragileAgentTrainer,
        q_network: MacroQNetwork,
        action_prototypes: ActionPrototypeTable | None,
        obs_normalizer: ObservationNormalizer | None,
        device: torch.device,
    ) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
        """Run evaluation episodes and return metrics and symbol usage."""
        trainer.agent.eval()
        eval_symbol_usage = init_symbol_usage_accumulator(
            obs_num_charts=trainer.agent.config.obs_encoder.num_charts,
            obs_codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
            act_num_charts=trainer.agent.config.act_encoder.num_charts,
            act_codes_per_chart=trainer.agent.config.act_encoder.codes_per_chart,
        )
        eval_episodes, eval_infos = collect_macro_episodes_batched(
            eval_envs,
            trainer.agent,
            q_network,
            action_prototypes,
            num_episodes=max(int(self.eval_episodes), 1),
            device=device,
            obs_normalizer=obs_normalizer,
            epsilon=0.0,
            action_repeat=self.action_repeat,
            max_steps=self.max_episode_steps,
            routing_tau=trainer.routing_tau_for_step(training=False),
            macro_chart_tau=trainer.config.macro_chart_tau,
            macro_code_tau=trainer.config.macro_code_tau,
            sigma_motor=0.0,
        )
        del eval_episodes
        for info in eval_infos:
            update_symbol_usage_from_episode_info(
                eval_symbol_usage,
                info,
                obs_codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
                act_codes_per_chart=trainer.agent.config.act_encoder.codes_per_chart,
            )
        eval_metrics = summarize_collection(eval_infos, prefix="eval")
        return eval_metrics, eval_symbol_usage

    def _seed_replay(
        self,
        train_envs: list[Any],
        trainer: FragileAgentTrainer,
        replay: SequenceReplayBuffer,
        device: torch.device,
    ) -> tuple[SequenceReplayBuffer, ObservationNormalizer, ActionPrototypeTable | None, int]:
        """Collect seed episodes and initialise normalizers and prototypes.

        Returns the (possibly replaced) replay buffer, the fitted observation
        normalizer, initial action prototypes, and accumulated env steps.
        """
        seed_episodes, seed_infos = collect_macro_episodes_batched(
            train_envs,
            agent=None,
            q_network=None,
            action_prototypes=None,
            num_episodes=self.seed_episodes,
            device=device,
            obs_normalizer=None,
            epsilon=1.0,
            action_repeat=self.action_repeat,
            max_steps=self.max_episode_steps,
            routing_tau=trainer.routing_tau_for_step(training=False),
            macro_chart_tau=trainer.config.macro_chart_tau,
            macro_code_tau=trainer.config.macro_code_tau,
            sigma_motor=self.sigma_motor,
        )
        env_steps = 0
        for episode, info in zip(seed_episodes, seed_infos, strict=False):
            replay.add_episode(episode)
            env_steps += int(info["length"])

        obs_normalizer = ObservationNormalizer.from_episodes(
            seed_episodes,
            device,
            min_std=self.obs_min_std,
        )
        act_mean, act_std = action_stats_from_episodes(
            seed_episodes,
            min_std=self.action_min_std,
        )
        trainer.agent.act_encoder.set_io_affine_stats(
            act_mean.to(device=device),
            act_std.to(device=device),
            learnable=trainer.agent.config.act_encoder.input_affine_learnable,
        )
        action_prototypes = update_action_symbol_prototypes(
            trainer.agent,
            replay._episodes,  # noqa: SLF001 - replay episodes are the fitting source.
            None,
            device=device,
            routing_tau=trainer.routing_tau_for_step(training=False),
            macro_chart_tau=trainer.config.macro_chart_tau,
            macro_code_tau=trainer.config.macro_code_tau,
            min_count=self.prototype_min_count,
            ema=self.prototype_ema,
        )
        print("Seed replay:", summarize_collection(seed_infos, prefix="seed"))
        return replay, obs_normalizer, action_prototypes, env_steps

    def _collect_epoch(
        self,
        *,
        epoch: int,
        epsilon: float,
        train_envs: list[Any],
        trainer: FragileAgentTrainer,
        q_network: MacroQNetwork,
        action_prototypes: ActionPrototypeTable | None,
        replay: SequenceReplayBuffer,
        obs_normalizer: ObservationNormalizer | None,
        device: torch.device,
        env_steps: int,
    ) -> tuple[int, ActionPrototypeTable | None, dict[str, torch.Tensor], list[dict[str, Any]]]:
        """Collect episodes, insert into replay, and update action prototypes."""
        train_symbol_usage = init_symbol_usage_accumulator(
            obs_num_charts=trainer.agent.config.obs_encoder.num_charts,
            obs_codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
            act_num_charts=trainer.agent.config.act_encoder.num_charts,
            act_codes_per_chart=trainer.agent.config.act_encoder.codes_per_chart,
        )

        trainer.agent.eval()
        collected_episodes, collect_infos = collect_macro_episodes_batched(
            train_envs,
            trainer.agent,
            q_network,
            action_prototypes,
            num_episodes=self.collect_episodes_per_epoch,
            device=device,
            obs_normalizer=obs_normalizer,
            epsilon=epsilon,
            action_repeat=self.action_repeat,
            max_steps=self.max_episode_steps,
            routing_tau=trainer.routing_tau_for_step(training=False),
            macro_chart_tau=trainer.config.macro_chart_tau,
            macro_code_tau=trainer.config.macro_code_tau,
            sigma_motor=self.sigma_motor,
        )
        for episode, info in zip(collected_episodes, collect_infos, strict=False):
            replay.add_episode(episode)
            env_steps += int(info["length"])
            update_symbol_usage_from_episode_info(
                train_symbol_usage,
                info,
                obs_codes_per_chart=trainer.agent.config.obs_encoder.codes_per_chart,
                act_codes_per_chart=trainer.agent.config.act_encoder.codes_per_chart,
            )

        action_prototypes = update_action_symbol_prototypes_from_rollouts(
            collected_episodes,
            collect_infos,
            action_prototypes,
            num_actions=trainer.agent.num_act_states,
            action_dim=trainer.agent.config.act_encoder.input_dim,
            min_count=self.prototype_min_count,
            ema=self.prototype_ema,
        )

        if self.prototype_refresh_every > 0 and ((epoch + 1) % self.prototype_refresh_every == 0):
            action_prototypes = update_action_symbol_prototypes(
                trainer.agent,
                replay._episodes,  # noqa: SLF001 - replay episodes are the fitting source.
                action_prototypes,
                device=device,
                routing_tau=trainer.routing_tau_for_step(training=False),
                macro_chart_tau=trainer.config.macro_chart_tau,
                macro_code_tau=trainer.config.macro_code_tau,
                min_count=self.prototype_min_count,
                ema=self.prototype_ema,
            )

        return env_steps, action_prototypes, train_symbol_usage, collect_infos

    def _train_epoch(
        self,
        *,
        epoch: int,
        epsilon: float,
        trainer: FragileAgentTrainer,
        q_network: MacroQNetwork,
        target_q_network: MacroQNetwork,
        q_optimizer: torch.optim.Optimizer,
        replay: SequenceReplayBuffer,
        obs_normalizer: ObservationNormalizer | None,
        action_prototypes: ActionPrototypeTable | None,
        device: torch.device,
        env_steps: int,
        update_steps: int,
        train_symbol_usage: dict[str, torch.Tensor],
        collect_infos: list[dict[str, Any]],
    ) -> tuple[int, dict[str, float]]:
        """Run N gradient updates and return updated counters and averaged metrics."""
        update_metrics: list[dict[str, float]] = []
        for _ in range(self.updates_per_epoch):
            replay_batch = replay.sample(self.batch_size, device=device)
            trainer.agent.train()
            trainer_batch = _trainer_batch_from_replay(replay_batch, obs_normalizer)
            outputs = trainer.compute_batch_losses(
                trainer_batch,
                epoch=epoch,
                global_step=trainer.global_step,
                training=True,
            )
            macro_batch = prepare_macro_transition_batch_from_forward(
                outputs["forward"],
                replay_batch,
            )
            update_symbol_usage_from_forward(
                train_symbol_usage,
                outputs["forward"],
            )

            q_loss, q_metrics = compute_q_learning_loss(
                q_network,
                target_q_network,
                macro_batch["obs_state_probs_t"].detach(),
                macro_batch["act_state_idx_t"].detach(),
                macro_batch["reward_t"].detach(),
                macro_batch["continuation_t"].detach(),
                macro_batch["obs_state_probs_tp1"].detach(),
                gamma=self.gamma,
                loss_type=self.q_loss_type,
            )

            aux_loss, aux_metrics = compute_macro_auxiliary_loss(
                trainer.agent.macro_model,
                macro_batch["obs_state_probs_t"].detach(),
                macro_batch["act_state_probs_t"].detach(),
                macro_batch["reward_t"].detach(),
                macro_batch["continuation_t"].detach(),
                weight_reward=self.weight_reward,
                weight_continuation=self.weight_continuation,
            )
            trainer.probe_optimizer.zero_grad()
            trainer.encoder_optimizer.zero_grad()
            trainer.markov_optimizer.zero_grad()
            q_optimizer.zero_grad()

            joint_loss = outputs["main_loss"] + q_loss + aux_loss
            joint_loss.backward()

            encoder_params = [
                param
                for group in trainer.encoder_optimizer.param_groups
                for param in group["params"]
                if param.requires_grad
            ]
            markov_params_live = [
                param for param in trainer.agent.macro_model.parameters() if param.requires_grad
            ]
            q_params_live = [param for param in q_network.parameters() if param.requires_grad]

            encoder_grad_norm = compute_grad_norm(encoder_params)
            encoder_param_norm = compute_param_norm(encoder_params)
            markov_grad_norm = compute_grad_norm(markov_params_live)
            markov_param_norm = compute_param_norm(markov_params_live)
            q_grad_norm = compute_grad_norm(q_params_live)
            q_param_norm = compute_param_norm(q_params_live)

            if trainer.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(encoder_params, trainer.config.grad_clip)
                nn.utils.clip_grad_norm_(markov_params_live, trainer.config.grad_clip)
            if self.q_grad_clip > 0:
                nn.utils.clip_grad_norm_(q_params_live, self.q_grad_clip)

            trainer.encoder_optimizer.step()
            trainer.markov_optimizer.step()
            q_optimizer.step()

            probe_grad_norm = 0.0
            probe_param_norm = compute_param_norm(
                [
                    param
                    for param in trainer.agent.enclosure_probe.parameters()
                    if param.requires_grad
                ],
            )
            if outputs["probe_loss"].requires_grad and outputs["probe_loss"].detach().item() > 0:
                trainer.probe_optimizer.zero_grad()
                outputs["probe_loss"].backward()
                probe_params_live = [
                    param
                    for param in trainer.agent.enclosure_probe.parameters()
                    if param.requires_grad
                ]
                probe_grad_norm = compute_grad_norm(probe_params_live)
                if trainer.config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(probe_params_live, trainer.config.grad_clip)
                trainer.probe_optimizer.step()

            update_steps += 1
            trainer.global_step += 1
            if self.q_target_update_every > 0 and update_steps % self.q_target_update_every == 0:
                if self.q_target_tau >= 1.0:
                    hard_update_target(target_q_network, q_network)
                else:
                    soft_update_target(target_q_network, q_network, self.q_target_tau)

            metrics = dict(outputs["metrics"])
            metrics.update(q_metrics)
            metrics.update(aux_metrics)
            metrics.update({
                "q/epsilon": float(epsilon),
                "q/grad_norm": float(q_grad_norm),
                "q/param_norm": float(q_param_norm),
                "model/aux_grad_norm": float(markov_grad_norm),
                "grad/encoder_norm": float(encoder_grad_norm),
                "param/encoder_norm": float(encoder_param_norm),
                "grad/markov_norm": float(markov_grad_norm),
                "param/markov_norm": float(markov_param_norm),
                "grad/probe_norm": float(probe_grad_norm),
                "param/probe_norm": float(probe_param_norm),
                "replay/episodes": float(replay.num_episodes),
                "replay/steps": float(replay.total_steps),
                "replay/env_steps": float(env_steps),
                "replay/update_steps": float(update_steps),
            })
            update_metrics.append(metrics)

        train_metrics = average_metrics(update_metrics)
        train_metrics.update(summarize_collection(collect_infos, prefix="collect"))
        train_metrics.update(prototype_metrics(action_prototypes))
        return update_steps, train_metrics

    def _save_checkpoint_if_needed(
        self,
        *,
        epoch: int,
        output_dir: Path,
        trainer: FragileAgentTrainer,
        q_network: MacroQNetwork,
        target_q_network: MacroQNetwork,
        q_optimizer: torch.optim.Optimizer,
        replay: SequenceReplayBuffer,
        obs_normalizer: ObservationNormalizer | None,
        action_prototypes: ActionPrototypeTable | None,
        env_steps: int,
        update_steps: int,
        train_metrics: dict[str, float],
        eval_metrics: dict[str, float],
    ) -> None:
        """Save a periodic checkpoint if the current epoch requires one."""
        should_save = self.save_every > 0 and (
            ((epoch + 1) % self.save_every == 0) or (epoch == self.epochs - 1)
        )
        if should_save:
            _save_macro_rl_checkpoint(
                output_dir / f"macro_rl_epoch_{epoch:05d}.pt",
                _build_checkpoint_payload(
                    self,
                    trainer,
                    q_network,
                    target_q_network,
                    q_optimizer,
                    replay,
                    obs_normalizer,
                    action_prototypes,
                    epoch=epoch,
                    env_steps=env_steps,
                    update_steps=update_steps,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                ),
            )

    def run(self) -> None:
        """Execute the standalone off-policy macro RL loop."""
        self._validate_config()
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        resolved_config = copy.deepcopy(getattr(self, "_resolved_config", self._config_dict()))
        cli_overrides = list(getattr(self, "_cli_overrides", []))
        launch_argv = list(getattr(self, "_launch_argv", []))
        config_path = Path(getattr(self, "_config_path", CONFIG_PATH))
        started_at_utc = getattr(self, "_run_started_at_utc", _utc_now_iso())

        _write_macro_rl_run_artifacts(
            output_dir=output_dir,
            resolved_config=resolved_config,
            cli_overrides=cli_overrides,
            launch_argv=launch_argv,
            status="starting",
            started_at_utc=started_at_utc,
            config_path=config_path,
        )

        last_train_metrics: dict[str, float] = {}
        last_eval_metrics: dict[str, float] = {}
        last_epoch = -1
        env_steps = 0
        update_steps = 0
        obs_dim = 0
        act_dim = 0
        device: torch.device | None = None
        model_summary: dict[str, int] | None = None

        try:
            (
                train_envs,
                eval_envs,
                trainer,
                q_network,
                target_q_network,
                q_optimizer,
                replay,
                device,
                obs_dim,
                act_dim,
            ) = self._setup()

            obs_normalizer: ObservationNormalizer | None = None
            action_prototypes: ActionPrototypeTable | None = None
            start_epoch = 0

            if self.resume:
                resumed = load_macro_rl_resume_checkpoint(
                    self.resume,
                    trainer,
                    q_network,
                    target_q_network,
                    q_optimizer,
                    replay,
                    device=device,
                )
                replay = resumed.replay
                obs_normalizer = resumed.obs_normalizer
                action_prototypes = resumed.action_prototypes
                env_steps = resumed.env_steps
                update_steps = resumed.update_steps
                start_epoch = resumed.start_epoch
            else:
                replay, obs_normalizer, action_prototypes, env_steps = self._seed_replay(
                    train_envs,
                    trainer,
                    replay,
                    device,
                )

            model_summary = self._print_model_summary(
                trainer,
                q_network,
                obs_dim=obs_dim,
                act_dim=act_dim,
            )
            _write_macro_rl_run_artifacts(
                output_dir=output_dir,
                resolved_config=resolved_config,
                cli_overrides=cli_overrides,
                launch_argv=launch_argv,
                status="running",
                started_at_utc=started_at_utc,
                config_path=config_path,
                device=device,
                obs_dim=obs_dim,
                act_dim=act_dim,
                model_summary=model_summary,
                env_steps=env_steps,
                update_steps=update_steps,
                last_epoch=start_epoch - 1,
            )

            last_epoch = start_epoch - 1

            epoch_iter = tqdm(
                range(start_epoch, self.epochs),
                desc="MacroRL",
                unit="epoch",
                initial=start_epoch,
                total=self.epochs,
            )
            for epoch in epoch_iter:
                last_epoch = epoch
                epsilon = _linear_schedule(
                    self.epsilon_start,
                    self.epsilon_end,
                    epoch,
                    self.epsilon_decay_epochs,
                )

                env_steps, action_prototypes, train_symbol_usage, collect_infos = self._collect_epoch(
                    epoch=epoch,
                    epsilon=epsilon,
                    train_envs=train_envs,
                    trainer=trainer,
                    q_network=q_network,
                    action_prototypes=action_prototypes,
                    replay=replay,
                    obs_normalizer=obs_normalizer,
                    device=device,
                    env_steps=env_steps,
                )

                update_steps, train_metrics = self._train_epoch(
                    epoch=epoch,
                    epsilon=epsilon,
                    trainer=trainer,
                    q_network=q_network,
                    target_q_network=target_q_network,
                    q_optimizer=q_optimizer,
                    replay=replay,
                    obs_normalizer=obs_normalizer,
                    action_prototypes=action_prototypes,
                    device=device,
                    env_steps=env_steps,
                    update_steps=update_steps,
                    train_symbol_usage=train_symbol_usage,
                    collect_infos=collect_infos,
                )

                should_eval = (epoch % self.eval_every == 0) or (epoch == self.epochs - 1)
                if should_eval:
                    eval_metrics, eval_symbol_usage = self._evaluate(
                        eval_envs,
                        trainer,
                        q_network,
                        action_prototypes,
                        obs_normalizer,
                        device,
                    )
                    last_eval_metrics = eval_metrics
                else:
                    eval_metrics = last_eval_metrics
                    eval_symbol_usage = None

                postfix = {
                    "return": format_metric_value(train_metrics.get("collect/return_mean", 0.0)),
                    "q": format_metric_value(train_metrics.get("q/loss", 0.0)),
                }
                if should_eval:
                    postfix["eval"] = format_metric_value(eval_metrics.get("eval/return_mean", 0.0))
                epoch_iter.set_postfix(postfix)

                should_log = (epoch % self.log_every == 0) or (epoch == self.epochs - 1)
                if should_log:
                    log_epoch(
                        header="MacroRL",
                        epoch=epoch,
                        train_metrics=train_metrics,
                        eval_metrics=eval_metrics,
                        train_symbol_usage=train_symbol_usage,
                        eval_symbol_usage=eval_symbol_usage,
                        env_steps=env_steps,
                        update_steps=update_steps,
                        should_eval=should_eval,
                        eval_every=self.eval_every,
                    )

                self._save_checkpoint_if_needed(
                    epoch=epoch,
                    output_dir=output_dir,
                    trainer=trainer,
                    q_network=q_network,
                    target_q_network=target_q_network,
                    q_optimizer=q_optimizer,
                    replay=replay,
                    obs_normalizer=obs_normalizer,
                    action_prototypes=action_prototypes,
                    env_steps=env_steps,
                    update_steps=update_steps,
                    train_metrics=train_metrics,
                    eval_metrics=eval_metrics,
                )

                last_train_metrics = train_metrics

            final_path = output_dir / "macro_rl_final.pt"
            _save_macro_rl_checkpoint(
                final_path,
                _build_checkpoint_payload(
                    self,
                    trainer,
                    q_network,
                    target_q_network,
                    q_optimizer,
                    replay,
                    obs_normalizer,
                    action_prototypes,
                    epoch=last_epoch,
                    env_steps=env_steps,
                    update_steps=update_steps,
                    train_metrics=last_train_metrics,
                    eval_metrics=last_eval_metrics,
                ),
            )
            _write_macro_rl_run_artifacts(
                output_dir=output_dir,
                resolved_config=resolved_config,
                cli_overrides=cli_overrides,
                launch_argv=launch_argv,
                status="completed",
                started_at_utc=started_at_utc,
                config_path=config_path,
                device=device,
                obs_dim=obs_dim,
                act_dim=act_dim,
                model_summary=model_summary,
                env_steps=env_steps,
                update_steps=update_steps,
                last_epoch=last_epoch,
                final_checkpoint=final_path,
                train_metrics=last_train_metrics,
                eval_metrics=last_eval_metrics,
            )
            print(f"Final checkpoint saved to {final_path}")
        except Exception as exc:
            _write_macro_rl_run_artifacts(
                output_dir=output_dir,
                resolved_config=resolved_config,
                cli_overrides=cli_overrides,
                launch_argv=launch_argv,
                status="failed",
                started_at_utc=started_at_utc,
                config_path=config_path,
                device=device,
                obs_dim=obs_dim if obs_dim > 0 else None,
                act_dim=act_dim if act_dim > 0 else None,
                model_summary=model_summary,
                env_steps=env_steps,
                update_steps=update_steps,
                last_epoch=last_epoch,
                train_metrics=last_train_metrics or None,
                eval_metrics=last_eval_metrics or None,
                error=exc,
            )
            raise


CONFIG_PATH = Path(__file__).resolve().parents[3] / "config" / "train_macro_rl.yml"


def main() -> None:
    """CLI entrypoint for standalone macro RL."""
    from hydra.utils import instantiate

    cfg = OmegaConf.load(CONFIG_PATH)
    if len(sys.argv) > 1:
        cli = OmegaConf.from_cli(sys.argv[1:])
        cfg = OmegaConf.merge(cfg, cli)
    runner: MacroRLRunner = instantiate(cfg)
    runner._resolved_config = OmegaConf.to_container(cfg, resolve=True)
    runner._cli_overrides = list(sys.argv[1:])
    runner._launch_argv = list(sys.argv)
    runner._config_path = str(CONFIG_PATH)
    runner._run_started_at_utc = _utc_now_iso()
    runner.run()


if __name__ == "__main__":
    main()
