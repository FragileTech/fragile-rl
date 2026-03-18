"""Action-manifold helpers for structured latent state composition."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import (
    exp_map_zero,
    log_map_zero,
    mobius_add,
    poincare_hyperbolic_score,
    poincare_temperature,
    poincare_weighted_mean,
    poincare_weighted_mean_per_chart,
    project_to_ball,
)
from fragile.layers.primitives import SpectralLinear
from fragile.layers.router import routing_weights


class LatentTokenizer(nn.Module):
    """Embed a single latent point as one token."""

    def __init__(self, latent_dim: int, d_model: int) -> None:
        super().__init__()
        self.embed = SpectralLinear(latent_dim, d_model)

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.embed(z).unsqueeze(1), z.unsqueeze(1)


def _state_index(
    chart_idx: torch.Tensor, code_idx: torch.Tensor, codes_per_chart: int
) -> torch.Tensor:
    """Flatten `(chart, code)` into one discrete symbolic state index."""
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _straight_through_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Return straight-through hard one-hot probabilities."""
    soft = F.softmax(torch.zeros(indices.shape[0], num_classes, device=indices.device), dim=-1)
    hard = F.one_hot(indices, num_classes=num_classes).to(dtype=soft.dtype)
    return hard + soft - soft.detach()


def _extract_codebook_tensor(atlas_model: nn.Module) -> torch.Tensor:
    """Return the projected codebook tensor for an atlas-like module."""
    atlas = getattr(atlas_model, "encoder", atlas_model)
    return project_to_ball(atlas.codebook)


def compose_structured_state_with_atlas(
    atlas_model: nn.Module,
    z_n: torch.Tensor,
    *,
    chart_weights: torch.Tensor | None = None,
    chart_idx: torch.Tensor | None = None,
    code_probs: torch.Tensor | None = None,
    code_idx: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Compose `z_geo` from discrete symbol state and structured nuisance `z_n`.

    The returned `z_geo` is an internal geometric realization of `(K, z_n)`.
    It is not a separate policy state.
    """
    atlas = getattr(atlas_model, "encoder", atlas_model)
    dtype = z_n.dtype
    device = z_n.device

    if chart_weights is None:
        if chart_idx is None:
            msg = "Either chart_weights or chart_idx must be provided."
            raise ValueError(msg)
        chart_weights = F.one_hot(chart_idx.long(), num_classes=atlas.num_charts).to(dtype=dtype)
    chart_idx = chart_weights.argmax(dim=-1)

    chart_centers = project_to_ball(atlas.chart_centers).to(device=device, dtype=dtype)
    codebook = project_to_ball(atlas.codebook).to(device=device, dtype=dtype)

    if code_probs is None:
        if code_idx is None:
            msg = "Either code_probs or code_idx must be provided."
            raise ValueError(msg)
        selected_code = codebook[chart_idx, code_idx.long()]
        selected_idx = code_idx.long()
        # Keep a per-chart view for downstream diagnostics and compatibility.
        code_probs = (
            F
            .one_hot(
                selected_idx,
                num_classes=codebook.shape[1],
            )
            .to(dtype=dtype)
            .unsqueeze(1)
            .expand(-1, codebook.shape[0], -1)
        )
        z_q_all = poincare_weighted_mean_per_chart(codebook, code_probs)
        z_q = selected_code
    else:
        z_q_all = poincare_weighted_mean_per_chart(codebook, code_probs)
        z_q = poincare_weighted_mean(z_q_all, chart_weights)
        per_chart_code_idx = code_probs.argmax(dim=-1)
        selected_idx = per_chart_code_idx.gather(1, chart_idx.unsqueeze(1)).squeeze(1)

    c_bar = poincare_weighted_mean(chart_centers, chart_weights)
    z_local = mobius_add(z_q, exp_map_zero(z_n))
    z_geo = project_to_ball(mobius_add(c_bar, z_local))
    return {
        "z_geo": z_geo,
        "router_weights": chart_weights,
        "chart_idx": chart_idx,
        "code_idx": selected_idx,
        "state_idx": _state_index(chart_idx, selected_idx, codebook.shape[1]),
        "c_bar": c_bar,
        "z_q": z_q,
        "z_q_all": z_q_all,
        "z_n": z_n,
        "code_probs": code_probs,
    }


def symbolize_latent_with_atlas(
    atlas_model: nn.Module,
    z_latent: torch.Tensor,
    *,
    router_weights_override: torch.Tensor | None = None,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Attach symbolic `(K_chart, K_code, z_n)` structure to a bulk latent `z_geo`."""
    atlas = getattr(atlas_model, "encoder", atlas_model)
    z_latent = project_to_ball(z_latent)
    chart_centers = project_to_ball(atlas.chart_centers)

    if router_weights_override is not None:
        router_weights = router_weights_override.to(device=z_latent.device, dtype=z_latent.dtype)
        router_weights = router_weights / router_weights.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        if hard_routing:
            chart_idx = router_weights.argmax(dim=-1)
            router_weights = F.one_hot(chart_idx, num_classes=atlas.num_charts).to(
                dtype=z_latent.dtype
            )
        else:
            chart_idx = router_weights.argmax(dim=-1)
    else:
        cov_router = getattr(atlas, "cov_router", None)
        if cov_router is not None:
            router_weights, chart_idx = cov_router(
                z_latent,
                chart_tokens=chart_centers,
                routing_tau=hard_routing_tau,
            )
        else:
            scores = poincare_hyperbolic_score(
                z_latent,
                chart_centers,
                key_dim=atlas.latent_dim,
                tau_min=atlas.router_tau_min,
                tau_denom_min=atlas.router_tau_denom_min,
                eps=atlas.router_transport_eps,
            )
            latent_router = getattr(atlas, "latent_router", None)
            if latent_router is not None:
                tau = poincare_temperature(
                    z_latent,
                    key_dim=atlas.latent_dim,
                    tau_min=atlas.router_tau_min,
                    tau_denom_min=atlas.router_tau_denom_min,
                )
                scores = scores + 0.1 * latent_router(z_latent) / tau.unsqueeze(1)
            router_weights = routing_weights(scores, hard_routing_tau)
            chart_idx = router_weights.argmax(dim=-1)

    c_bar = poincare_weighted_mean(chart_centers, router_weights)
    v_local = project_to_ball(mobius_add(-c_bar, z_latent))
    z_q, code_idx, indices_stack, _vq, z_q_all = atlas._hyperbolic_vq(
        v_local,
        atlas.codebook,
        router_weights,
        0.0,
        0.0,
        use_soft_equiv=True,
    )
    v_bc = v_local.unsqueeze(1)
    delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
    z_n_all = atlas.structure_filter(delta.reshape(-1, atlas.latent_dim))
    z_n_all_charts_tan = z_n_all.view(z_latent.shape[0], atlas.num_charts, atlas.latent_dim)
    z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)
    return {
        "z_geo": z_latent,
        "z_latent": z_latent,
        "router_weights": router_weights,
        "chart_idx": chart_idx,
        "code_idx": code_idx,
        "state_idx": _state_index(chart_idx, code_idx, atlas.codebook.shape[1]),
        "c_bar": c_bar,
        "z_q": z_q,
        "v_local": v_local,
        "z_n": z_n_tan,
        "indices_stack": indices_stack,
    }
