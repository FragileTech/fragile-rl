"""Checkpoint utilities for VLA and RL training."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import os
import pathlib
import tempfile
from typing import Any, TYPE_CHECKING

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import torch
from torch import nn, optim


if TYPE_CHECKING:
    from fragile.agent import FragileAgentTrainer
    from fragile.rl.env_helpers import ObservationNormalizer
    from fragile.rl.macro_data import ActionPrototypeTable
    from fragile.rl.replay_buffer import SequenceReplayBuffer


def _atomic_save(obj: object, path: str) -> None:
    """Save a PyTorch object atomically to prevent 0-byte files."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        size = pathlib.Path(tmp_path).stat().st_size
        if size == 0:
            raise RuntimeError(
                f"torch.save produced 0-byte file for {path}. "
                "Check that all objects in the checkpoint are picklable."
            )
        pathlib.Path(tmp_path).replace(path)
    except BaseException:
        if os.path.exists(tmp_path):
            pathlib.Path(tmp_path).unlink()
        raise


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def compute_perplexity(assignments: torch.Tensor, num_charts: int) -> float:
    """Compute chart usage perplexity from chart assignments."""
    if assignments.numel() == 0:
        return 0.0
    counts = torch.bincount(assignments, minlength=num_charts).float()
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -(probs * torch.log(probs)).sum()
    return math.exp(entropy.item())


def _optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer] | None,
) -> dict | None:
    if optimizer is None:
        return None
    if isinstance(optimizer, dict):
        return {name: opt.state_dict() for name, opt in optimizer.items()}
    return optimizer.state_dict()


def compute_param_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        total += p.detach().pow(2).sum().item()
    return math.sqrt(total)


def compute_grad_norm(params: list[torch.Tensor]) -> float:
    total = 0.0
    for p in params:
        if p.grad is None:
            continue
        total += p.grad.detach().pow(2).sum().item()
    return math.sqrt(total)


def _state_dict_cpu(module: nn.Module | None) -> dict[str, torch.Tensor] | None:
    if module is None:
        return None
    return {k: v.detach().cpu() for k, v in module.state_dict().items()}


def load_checkpoint(path: str) -> dict:
    """Load checkpoint with unsafe deserialization allowed for trusted outputs."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _move_optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer],
    device: torch.device,
) -> None:
    if isinstance(optimizer, dict):
        for opt in optimizer.values():
            _move_optimizer_state(opt, device)
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_optimizer_state(
    optimizer: optim.Optimizer | dict[str, optim.Optimizer] | None,
    state: dict | None,
    device: torch.device,
) -> None:
    if optimizer is None or state is None:
        return
    if isinstance(optimizer, dict):
        if isinstance(state, dict) and "state" in state and "param_groups" in state:
            if len(optimizer) == 1:
                opt = next(iter(optimizer.values()))
                opt.load_state_dict(state)
                _move_optimizer_state(opt, device)
            else:
                print(
                    "  Optimizer state mismatch: single state for multiple optimizers; skipping."
                )
            return
        if isinstance(state, dict):
            for name, opt in optimizer.items():
                opt_state = state.get(name)
                if opt_state is not None:
                    opt.load_state_dict(opt_state)
                    _move_optimizer_state(opt, device)
        elif len(optimizer) == 1:
            opt = next(iter(optimizer.values()))
            opt.load_state_dict(state)
            _move_optimizer_state(opt, device)
        else:
            print("  Optimizer state mismatch: single state for multiple optimizers; skipping.")
        return
    if isinstance(state, dict):
        state = state.get("all")
        if state is None:
            print("  Optimizer state mismatch: multi-state for single optimizer; skipping.")
            return
    optimizer.load_state_dict(state)
    _move_optimizer_state(optimizer, device)


def compute_ami(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Compute Adjusted Mutual Information score."""
    return float(adjusted_mutual_info_score(labels_true, labels_pred))


def geometry_checkpoint_payload(
    trainer: FragileAgentTrainer,
    config: dict[str, Any],
    *,
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    best_eval_metric_name: str | None = None,
    best_eval_metric_value: float | None = None,
    best_eval_epoch: int | None = None,
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
            trainer.encoder_scheduler.state_dict()
            if trainer.encoder_scheduler is not None
            else None
        ),
        "args": config,
        "agent_config": copy.deepcopy(trainer.agent.config),
        "trainer_config": copy.deepcopy(trainer.config),
        "train_metrics": dict(train_metrics),
        "eval_metrics": dict(eval_metrics),
        "best_eval_metric_name": best_eval_metric_name,
        "best_eval_metric_value": best_eval_metric_value,
        "best_eval_epoch": best_eval_epoch,
    }


def save_geometry_checkpoint(
    path: pathlib.Path,
    trainer: FragileAgentTrainer,
    config: dict[str, Any],
    *,
    epoch: int,
    train_metrics: dict[str, float],
    eval_metrics: dict[str, float],
    best_eval_metric_name: str | None = None,
    best_eval_metric_value: float | None = None,
    best_eval_epoch: int | None = None,
) -> None:
    """Save a geometry-training checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        geometry_checkpoint_payload(
            trainer,
            config,
            epoch=epoch,
            train_metrics=train_metrics,
            eval_metrics=eval_metrics,
            best_eval_metric_name=best_eval_metric_name,
            best_eval_metric_value=best_eval_metric_value,
            best_eval_epoch=best_eval_epoch,
        ),
        path,
    )
    print(f"  Saved checkpoint: {path}")


def load_geometry_resume_checkpoint(
    trainer: FragileAgentTrainer,
    resume_path: str,
    *,
    device: torch.device,
) -> tuple[int, str | None, float | None, int | None]:
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
    return (
        start_epoch,
        ckpt.get("best_eval_metric_name"),
        ckpt.get("best_eval_metric_value"),
        ckpt.get("best_eval_epoch"),
    )


@dataclass
class MacroRLResumeState:
    """Return value of :func:`load_macro_rl_resume_checkpoint`."""

    replay: SequenceReplayBuffer
    obs_normalizer: ObservationNormalizer | None
    action_prototypes: ActionPrototypeTable | None
    env_steps: int
    update_steps: int
    start_epoch: int


def load_macro_rl_resume_checkpoint(
    resume_path: str,
    trainer: FragileAgentTrainer,
    q_network: nn.Module,
    target_q_network: nn.Module,
    q_optimizer: optim.Optimizer,
    replay: SequenceReplayBuffer,
    *,
    device: torch.device,
) -> MacroRLResumeState:
    """Restore full macro-RL state from a checkpoint.

    Mutates *trainer*, *q_network*, *target_q_network*, and *q_optimizer*
    in place, then returns the remaining mutable state that the caller
    must rebind.
    """
    from fragile.rl.env_helpers import ObservationNormalizer
    from fragile.rl.macro_data import ActionPrototypeTable, replay_buffer_from_state

    ckpt = load_checkpoint(resume_path)

    # --- trainer / encoder state ---
    trainer.agent.load_state_dict(ckpt["agent_state"])
    load_optimizer_state(trainer.encoder_optimizer, ckpt.get("encoder_optimizer"), device)
    load_optimizer_state(trainer.probe_optimizer, ckpt.get("probe_optimizer"), device)
    load_optimizer_state(trainer.markov_optimizer, ckpt.get("markov_optimizer"), device)
    if trainer.encoder_scheduler is not None and ckpt.get("encoder_scheduler") is not None:
        trainer.encoder_scheduler.load_state_dict(ckpt["encoder_scheduler"])

    # --- Q networks ---
    q_network.load_state_dict(ckpt["q_state"])
    target_q_network.load_state_dict(ckpt.get("target_q_state", ckpt["q_state"]))
    load_optimizer_state(q_optimizer, ckpt.get("q_optimizer"), device)

    # --- replay buffer ---
    replay_state = ckpt.get("replay_buffer")
    if replay_state is not None:
        replay = replay_buffer_from_state(replay_state)

    # --- observation normalizer ---
    obs_normalizer: ObservationNormalizer | None = None
    obs_state = ckpt.get("obs_normalizer")
    if obs_state is not None:
        obs_normalizer = ObservationNormalizer.from_state_dict(obs_state, device)

    # --- action prototypes ---
    action_prototypes: ActionPrototypeTable | None = None
    prototype_state = ckpt.get("action_prototypes")
    if prototype_state is not None:
        action_prototypes = ActionPrototypeTable.from_state_dict(prototype_state)

    # --- scalar counters ---
    trainer.global_step = int(
        ckpt.get("trainer_global_step", ckpt.get("global_step", 0)),
    )
    env_steps = int(ckpt.get("env_steps", 0))
    update_steps = int(ckpt.get("update_steps", trainer.global_step))
    start_epoch = max(int(ckpt.get("epoch", -1)) + 1, 0)

    print(
        f"Resumed from {resume_path} "
        f"(epoch {ckpt.get('epoch', '?')}, updates {update_steps}, env_steps {env_steps})",
    )
    return MacroRLResumeState(
        replay=replay,
        obs_normalizer=obs_normalizer,
        action_prototypes=action_prototypes,
        env_steps=env_steps,
        update_steps=update_steps,
        start_epoch=start_epoch,
    )
