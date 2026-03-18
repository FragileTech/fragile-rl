"""Structured actor on the action manifold.

The actor consumes the observation-side symbolic state `(K, z_n)` and predicts
the action-side symbolic state `(K^act, z_{n,act})`. A geometric latent `z_geo`
is reconstructed internally from that structured action state for the action
decoder and world-model interfaces.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import project_to_ball
from fragile.layers.primitives import SpectralLinear
from fragile.rl.action_manifold import _state_index, compose_structured_state_with_atlas


class GeometricActor(nn.Module):
    """Policy over action symbols and structured nuisance."""

    def __init__(
        self,
        latent_dim: int,
        num_obs_charts: int,
        obs_codes_per_chart: int,
        num_action_charts: int,
        action_codes_per_chart: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.num_obs_charts = num_obs_charts
        self.obs_codes_per_chart = obs_codes_per_chart
        self.num_action_charts = num_action_charts
        self.action_codes_per_chart = action_codes_per_chart

        self.obs_state_embed = nn.Embedding(num_obs_charts * obs_codes_per_chart, d_model)
        self.obs_zn_embed = SpectralLinear(latent_dim, d_model)
        self.backbone = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.GELU(),
        )
        self.action_chart_head = nn.Linear(d_model, num_action_charts)
        self.action_code_head = nn.Linear(
            d_model,
            num_action_charts * action_codes_per_chart,
        )
        self.action_zn_head = SpectralLinear(d_model, latent_dim)

        self.register_buffer("action_chart_centers", torch.zeros(num_action_charts, latent_dim))
        self.register_buffer(
            "action_codebook",
            torch.zeros(num_action_charts, action_codes_per_chart, latent_dim),
        )

    def bind_action_atlas(self, chart_centers: torch.Tensor, codebook: torch.Tensor) -> None:
        """Bind the action atlas used to reconstruct `z_geo` from `(K, z_n)`."""
        self.action_chart_centers.copy_(project_to_ball(chart_centers.detach()))
        self.action_codebook.copy_(project_to_ball(codebook.detach()))

    def _action_code_probs(
        self,
        action_code_logits: torch.Tensor,
        *,
        hard_routing: bool,
        hard_routing_tau: float,
    ) -> torch.Tensor:
        tau = 1.0 if hard_routing_tau <= 0 else hard_routing_tau
        code_soft = F.softmax(action_code_logits / tau, dim=-1)
        if not hard_routing:
            return code_soft
        code_hard = F.one_hot(
            code_soft.argmax(dim=-1),
            num_classes=self.action_codes_per_chart,
        ).to(dtype=code_soft.dtype)
        return code_hard + code_soft - code_soft.detach()

    def _action_chart_probs(
        self,
        action_chart_logits: torch.Tensor,
        *,
        hard_routing: bool,
        hard_routing_tau: float,
    ) -> torch.Tensor:
        tau = 1.0 if hard_routing_tau <= 0 else hard_routing_tau
        chart_soft = F.softmax(action_chart_logits / tau, dim=-1)
        if not hard_routing:
            return chart_soft
        chart_hard = F.one_hot(
            chart_soft.argmax(dim=-1),
            num_classes=self.num_action_charts,
        ).to(dtype=chart_soft.dtype)
        return chart_hard + chart_soft - chart_soft.detach()

    def forward(
        self,
        obs_chart_idx: torch.Tensor,
        obs_code_idx: torch.Tensor,
        obs_z_n: torch.Tensor,
        *,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Predict structured action state from structured observation state."""
        obs_state_idx = _state_index(obs_chart_idx, obs_code_idx, self.obs_codes_per_chart)
        feat = torch.cat(
            [self.obs_state_embed(obs_state_idx), self.obs_zn_embed(obs_z_n)],
            dim=-1,
        )
        feat = self.backbone(feat)

        action_chart_logits = self.action_chart_head(feat)
        action_chart_probs = self._action_chart_probs(
            action_chart_logits,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        action_chart_idx = action_chart_probs.argmax(dim=-1)

        action_code_logits = self.action_code_head(feat).view(
            -1,
            self.num_action_charts,
            self.action_codes_per_chart,
        )
        action_code_probs = self._action_code_probs(
            action_code_logits,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        per_chart_code_idx = action_code_probs.argmax(dim=-1)
        action_code_idx = per_chart_code_idx.gather(1, action_chart_idx.unsqueeze(1)).squeeze(1)

        action_z_n = self.action_zn_head(feat)

        dummy_atlas = type("AtlasView", (), {})()
        dummy_atlas.num_charts = self.num_action_charts
        dummy_atlas.chart_centers = self.action_chart_centers
        dummy_atlas.codebook = self.action_codebook
        composed = compose_structured_state_with_atlas(
            dummy_atlas,
            action_z_n,
            chart_weights=action_chart_probs,
            code_probs=action_code_probs,
        )
        return {
            "action_chart_logits": action_chart_logits,
            "action_chart_probs": action_chart_probs,
            "action_chart_idx": action_chart_idx,
            "action_code_logits": action_code_logits,
            "action_code_probs": action_code_probs,
            "action_code_idx": action_code_idx,
            "action_z_n": action_z_n,
            "action_state_idx": composed["state_idx"],
            "action_z_q": composed["z_q"],
            "action_z_geo": composed["z_geo"],
            "action_router_weights": composed["router_weights"],
        }

    def sample_latent(
        self,
        obs_chart_idx: torch.Tensor,
        obs_code_idx: torch.Tensor,
        obs_z_n: torch.Tensor,
        *,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        """Compatibility alias for the deterministic structured actor."""
        return self.forward(
            obs_chart_idx,
            obs_code_idx,
            obs_z_n,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

    def mode_latent(
        self,
        obs_chart_idx: torch.Tensor,
        obs_code_idx: torch.Tensor,
        obs_z_n: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the deterministic structured action state."""
        return self.forward(obs_chart_idx, obs_code_idx, obs_z_n)
