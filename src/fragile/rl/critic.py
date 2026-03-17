"""Chart-conditioned critic on the Poincare ball."""

from __future__ import annotations

import torch
from torch import nn

from fragile.layers.attention import CovariantAttention, GeodesicConfig
from fragile.layers.primitives import SpectralLinear
from fragile.vla.covariant_world_model import ChartTokenizer


class GeometricCritic(nn.Module):
    """Estimate scalar state value from latent state and chart routing."""

    def __init__(
        self,
        latent_dim: int,
        num_charts: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.value_head = SpectralLinear(d_model, 1)

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Predict scalar value for each latent state."""
        chart_x, chart_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, chart_z, x_q, chart_x, chart_x)
        return self.value_head(feat)

    def task_value(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Canonical theory-facing value-field interface."""
        return self.forward(z, rw)
