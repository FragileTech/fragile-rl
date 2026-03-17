from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LorentzianConfig:
    """Configuration for Lorentzian memory attention."""

    d_model: int = 256
    d_latent: int = 64
    n_heads: int = 4
    c_info: float = 1.0
    T_c: float = 0.1
    gamma_friction: float = 1.0
    dt: float = 0.01


class LorentzianMetric(nn.Module):
    """Lorentzian metric utilities on a Poincare disk."""

    def __init__(self, config: LorentzianConfig, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.config = config
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda(z).

        Args:
            z: [B, d] positions

        Returns:
            lambda_z: [B, 1] conformal factor
        """
        norm_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
        return 2.0 / (1.0 - norm_sq + self.epsilon)  # [B, 1]

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance on the Poincare disk.

        Args:
            z1: [B, d] query positions
            z2: [B, N, d] memory positions

        Returns:
            d_g: [B, N] geodesic distances
        """
        diff = z1.unsqueeze(1) - z2  # [B, N, d]
        diff_sq = (diff**2).sum(dim=-1)  # [B, N]
        norm1 = (z1**2).sum(dim=-1, keepdim=True)  # [B, 1]
        norm2 = (z2**2).sum(dim=-1)  # [B, N]
        denom = (1.0 - norm1) * (1.0 - norm2) + self.epsilon  # [B, N]
        arg = 1.0 + 2.0 * diff_sq / denom  # [B, N]
        arg = torch.clamp(arg, min=1.0 + self.epsilon)  # [B, N]
        return torch.acosh(arg)  # [B, N]

    def spacetime_interval(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        z_mem: torch.Tensor,
        t_mem: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Lorentzian spacetime interval.

        Args:
            z: [B, d] query position
            t: [B, 1] query time
            z_mem: [B, N, d] memory positions
            t_mem: [B, N, 1] memory times

        Returns:
            interval: [B, N] spacetime intervals
        """
        d_g = self.geodesic_distance(z, z_mem)  # [B, N]
        dt = (t.unsqueeze(1) - t_mem).squeeze(-1)  # [B, N]
        return -((self.config.c_info * dt) ** 2) + d_g**2  # [B, N]

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """Compute metric temperature.

        Args:
            z: [B, d] positions
            d_k: key dimension

        Returns:
            tau: [B, 1] temperature
        """
        lambda_z = self.conformal_factor(z)  # [B, 1]
        return (d_k**0.5) / lambda_z  # [B, 1]


class CausalMask(nn.Module):
    """Causal light-cone mask for Lorentzian memory."""

    def __init__(self, config: LorentzianConfig) -> None:
        super().__init__()
        self.metric = LorentzianMetric(config)
        self.config = config

    def forward(
        self,
        z: torch.Tensor,
        t: torch.Tensor,
        z_mem: torch.Tensor,
        t_mem: torch.Tensor,
    ) -> torch.Tensor:
        """Compute causal mask.

        Args:
            z: [B, d] query position
            t: [B, 1] query time
            z_mem: [B, N, d] memory positions
            t_mem: [B, N, 1] memory times

        Returns:
            mask: [B, N] causal mask (1 = causal)
        """
        d_g = self.metric.geodesic_distance(z, z_mem)  # [B, N]
        dt = (t.unsqueeze(1) - t_mem).squeeze(-1)  # [B, N]
        time_ok = dt > 0.0  # [B, N]
        cone_ok = d_g <= self.config.c_info * dt  # [B, N]
        return (time_ok & cone_ok).float()  # [B, N]


class TemporalChristoffelQuery(nn.Module):
    """Geodesic query with temporal Christoffel terms."""

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        super().__init__()
        self.w_x = nn.Linear(d_in, d_out)
        self.w_z = nn.Linear(d_latent, d_out)
        self.w_t = nn.Linear(1, d_out)
        self.w_v = nn.Linear(d_latent, d_out)

        self.w_zz = nn.Parameter(torch.randn(d_out, d_latent, d_latent) * 0.01)
        self.w_tt = nn.Parameter(torch.randn(d_out) * 0.01)
        self.w_zt = nn.Parameter(torch.randn(d_out, d_latent) * 0.01)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        v_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute temporal geodesic query.

        Args:
            x: [B, d_in] feature vector
            z: [B, d_latent] position
            t: [B, 1] time
            v_feat: [B, d_latent] optional velocity features

        Returns:
            q: [B, d_out] query vector
        """
        q_x = self.w_x(x)  # [B, d_out]
        q_z = self.w_z(z)  # [B, d_out]
        q_t = self.w_t(t)  # [B, d_out]
        q_v = self.w_v(v_feat) if v_feat is not None else torch.zeros_like(q_x)  # [B, d_out]

        # Quadratic and mixed terms approximate temporal Christoffel-symbol corrections.
        q_zz = torch.einsum("bi,oij,bj->bo", z, self.w_zz, z)  # [B, d_out]
        q_tt = (t.squeeze(-1) ** 2).unsqueeze(-1) * self.w_tt  # [B, d_out]
        q_zt = t * torch.matmul(z, self.w_zt.t())  # [B, d_out]

        return q_x + q_z + q_t + q_v + q_zz + q_tt + q_zt  # [B, d_out]


class LorentzianMemoryAttention(nn.Module):
    """Lorentzian memory attention with causal masking."""

    def __init__(self, config: LorentzianConfig) -> None:
        super().__init__()
        self.config = config
        self.metric = LorentzianMetric(config)
        self.causal_mask = CausalMask(config)

        self.query = TemporalChristoffelQuery(config.d_model, config.d_model, config.d_latent)
        self.key_proj = nn.Linear(config.d_model, config.d_model)
        self.value_proj = nn.Linear(config.d_model, config.d_model)
        self.wilson_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        x_mem: torch.Tensor,
        z_mem: torch.Tensor,
        t_mem: torch.Tensor,
        v_feat: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Attend to memory under Lorentzian causality.

        Args:
            x: [B, d_model] current features
            z: [B, d_latent] current position
            t: [B, 1] current time
            x_mem: [B, N, d_model] memory features
            z_mem: [B, N, d_latent] memory positions
            t_mem: [B, N, 1] memory times
            v_feat: [B, d_latent] optional velocity features

        Returns:
            output: [B, d_model] attended features
            weights: [B, N] attention weights
        """
        q = self.query(x, z, t, v_feat=v_feat)  # [B, d_model]
        k = self.key_proj(x_mem)  # [B, N, d_model]
        v = self.value_proj(x_mem)  # [B, N, d_model]

        # Geodesic distance sets Wilson-style attenuation in memory.
        d_g = self.metric.geodesic_distance(z, z_mem)  # [B, N]
        wilson = torch.exp(-self.wilson_scale * d_g)  # [B, N]

        # Metric temperature rescales logits in curved space.
        tau = self.metric.temperature(z, k.shape[-1])  # [B, 1]
        scores = (q.unsqueeze(1) * k).sum(dim=-1)  # [B, N]
        scores = scores * wilson / (tau + 1e-6)  # [B, N]

        # Light-cone mask enforces Lorentzian causality.
        mask = self.causal_mask(z, t, z_mem, t_mem)  # [B, N]
        scores = scores.masked_fill(mask == 0.0, -1e9)  # [B, N]

        weights = F.softmax(scores, dim=-1)  # [B, N]
        weights *= mask  # [B, N]
        weights /= weights.sum(dim=-1, keepdim=True) + 1e-08  # [B, N]

        # Aggregate values from causal, geodesically-weighted memory.
        output = (weights.unsqueeze(-1) * v).sum(dim=1)  # [B, d_model]
        return output, weights
