"""Checkpoint utilities for VLA and RL training."""

import math
import os
import tempfile

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score
import torch
from torch import nn, optim


def _atomic_save(obj: object, path: str) -> None:
    """Save a PyTorch object atomically to prevent 0-byte files."""
    dir_name = os.path.dirname(path) or "."
    os.makedirs(dir_name, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".pt.tmp")
    try:
        os.close(fd)
        torch.save(obj, tmp_path)
        size = os.path.getsize(tmp_path)
        if size == 0:
            raise RuntimeError(
                f"torch.save produced 0-byte file for {path}. "
                "Check that all objects in the checkpoint are picklable."
            )
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
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
