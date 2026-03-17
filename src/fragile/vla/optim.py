"""Optimizer helpers for multi-timescale VLA training."""

from __future__ import annotations

from typing import Any

from torch import nn


def _atlas_encoder(module: nn.Module) -> nn.Module:
    """Return the atlas encoder module whether wrapped or bare."""
    return getattr(module, "encoder", module)


def get_codebook_like_params(module: nn.Module) -> list[nn.Parameter]:
    """Return codebook parameters that should use the slower codebook LR."""
    atlas_encoder = _atlas_encoder(module)
    params: list[nn.Parameter] = []
    for name in ("codebook", "codebook_dyn"):
        param = getattr(atlas_encoder, name, None)
        if isinstance(param, nn.Parameter) and param.requires_grad:
            params.append(param)
    return params


def build_encoder_param_groups(
    encoder: nn.Module,
    jump_op: nn.Module | None,
    *,
    base_lr: float,
    lr_chart_centers_scale: float = 1.0,
    lr_codebook_scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Build encoder optimizer groups with slower atlas-anchor/codebook updates."""
    atlas_encoder = _atlas_encoder(encoder)
    special_ids: set[int] = set()

    chart_center_params: list[nn.Parameter] = []
    chart_centers = getattr(atlas_encoder, "chart_centers", None)
    if isinstance(chart_centers, nn.Parameter) and chart_centers.requires_grad:
        chart_center_params.append(chart_centers)
        special_ids.add(id(chart_centers))

    codebook_params = get_codebook_like_params(encoder)
    special_ids.update(id(param) for param in codebook_params)

    base_params = [
        param
        for param in encoder.parameters()
        if param.requires_grad and id(param) not in special_ids
    ]
    if jump_op is not None:
        base_params.extend(param for param in jump_op.parameters() if param.requires_grad)

    param_groups: list[dict[str, Any]] = []
    if base_params:
        param_groups.append({"params": base_params, "lr": base_lr})
    if chart_center_params:
        param_groups.append(
            {
                "params": chart_center_params,
                "lr": base_lr * lr_chart_centers_scale,
            }
        )
    if codebook_params:
        param_groups.append(
            {
                "params": codebook_params,
                "lr": base_lr * lr_codebook_scale,
            }
        )
    return param_groups


__all__ = ["build_encoder_param_groups", "get_codebook_like_params"]
