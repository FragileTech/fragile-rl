"""Shared helper utilities for loss modules."""

from __future__ import annotations

from torch import Tensor

from fragile.layers.gauge import log_map_zero


def _project_to_ball(z: Tensor, max_norm: float = 0.99, eps: float = 1e-6) -> Tensor:
    """Project points to interior of the Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def _as_tangent(z: Tensor, assume_tangent: bool) -> Tensor:
    """Return tangent vectors; map from ball if needed."""
    if assume_tangent:
        return z
    return log_map_zero(_project_to_ball(z))
