from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class LorentzianConfig:
    """Configuration for Lorentzian memory attention.

    Attributes:
        d_model: Dimension of the model feature space.
        d_latent: Dimension of the latent Poincare disk embedding.
        n_heads: Number of attention heads.
        c_info: Speed of information propagation in the light-cone.
        T_c: Critical temperature for phase transitions.
        gamma_friction: Friction coefficient for geodesic dynamics.
        dt: Integration time step for temporal evolution.
    """

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
        """Initialize Lorentzian metric utilities.

        Args:
            config: Lorentzian configuration specifying model hyperparameters.
            epsilon: Small constant for numerical stability in denominators
                and clamping operations.
        """
        super().__init__()
        self.config = config
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda(z) on the Poincare disk.

        Args:
            z: Batch of positions on the Poincare disk with shape ``[B, d]``.

        Returns:
            torch.Tensor: Conformal factor ``2 / (1 - ||z||^2)`` with shape
                ``[B, 1]``.
        """
        norm_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
        return 2.0 / (1.0 - norm_sq + self.epsilon)  # [B, 1]

    def geodesic_distance(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Compute geodesic distance on the Poincare disk.

        Uses the Mobius addition formula: ``d(z1, z2) = acosh(1 + 2 * ||z1 - z2||^2
        / ((1 - ||z1||^2)(1 - ||z2||^2)))``.

        Args:
            z1: Query positions on the Poincare disk with shape ``[B, d]``.
            z2: Memory positions on the Poincare disk with shape ``[B, N, d]``.

        Returns:
            torch.Tensor: Pairwise geodesic distances with shape ``[B, N]``.
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

        Calculates ``s^2 = -(c * dt)^2 + d_g^2`` where ``d_g`` is the geodesic
        distance on the Poincare disk. Negative intervals indicate timelike
        separation (inside the light-cone).

        Args:
            z: Query position on the Poincare disk with shape ``[B, d]``.
            t: Query time with shape ``[B, 1]``.
            z_mem: Memory positions on the Poincare disk with shape
                ``[B, N, d]``.
            t_mem: Memory times with shape ``[B, N, 1]``.

        Returns:
            torch.Tensor: Spacetime intervals with shape ``[B, N]``. Negative
                values indicate timelike (causal) separation.
        """
        d_g = self.geodesic_distance(z, z_mem)  # [B, N]
        dt = (t.unsqueeze(1) - t_mem).squeeze(-1)  # [B, N]
        return -((self.config.c_info * dt) ** 2) + d_g**2  # [B, N]

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """Compute metric-aware temperature for attention scaling.

        The temperature is inversely proportional to the conformal factor,
        so that attention becomes sharper near the boundary of the Poincare
        disk where curvature is higher.

        Args:
            z: Positions on the Poincare disk with shape ``[B, d]``.
            d_k: Dimension of the key vectors, used for ``sqrt(d_k)`` scaling.

        Returns:
            torch.Tensor: Temperature values with shape ``[B, 1]``.
        """
        lambda_z = self.conformal_factor(z)  # [B, 1]
        return (d_k**0.5) / lambda_z  # [B, 1]


class CausalMask(nn.Module):
    """Causal light-cone mask for Lorentzian memory."""

    def __init__(self, config: LorentzianConfig) -> None:
        """Initialize causal light-cone mask.

        Args:
            config: Lorentzian configuration. The ``c_info`` field controls
                the speed of information propagation that defines the
                light-cone boundary.
        """
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
        """Compute causal mask from the forward light-cone.

        A memory slot is considered causal if it lies in the past
        (``dt > 0``) and its geodesic distance satisfies
        ``d_g <= c_info * dt`` (inside the light-cone).

        Args:
            z: Query position on the Poincare disk with shape ``[B, d]``.
            t: Query time with shape ``[B, 1]``.
            z_mem: Memory positions on the Poincare disk with shape
                ``[B, N, d]``.
            t_mem: Memory times with shape ``[B, N, 1]``.

        Returns:
            torch.Tensor: Binary causal mask with shape ``[B, N]`` where
                ``1.0`` indicates a causally accessible memory slot and
                ``0.0`` indicates a slot outside the light-cone.
        """
        d_g = self.metric.geodesic_distance(z, z_mem)  # [B, N]
        dt = (t.unsqueeze(1) - t_mem).squeeze(-1)  # [B, N]
        time_ok = dt > 0.0  # [B, N]
        cone_ok = d_g <= self.config.c_info * dt  # [B, N]
        return (time_ok & cone_ok).float()  # [B, N]


class TemporalChristoffelQuery(nn.Module):
    """Geodesic query with temporal Christoffel terms."""

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        """Initialize geodesic query with temporal Christoffel correction terms.

        Creates linear projections for input features, latent position, time,
        and velocity, plus learnable parameters for quadratic (``z-z``,
        ``t-t``) and mixed (``z-t``) Christoffel-symbol correction terms.

        Args:
            d_in: Dimension of the input feature vector.
            d_out: Dimension of the output query vector.
            d_latent: Dimension of the latent Poincare disk embedding.
        """
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
        """Compute temporal geodesic query with Christoffel corrections.

        Combines linear projections of input features, position, time, and
        velocity with quadratic and mixed Christoffel-symbol correction terms
        that approximate parallel transport on the curved manifold.

        Args:
            x: Input feature vector with shape ``[B, d_in]``.
            z: Latent position on the Poincare disk with shape
                ``[B, d_latent]``.
            t: Scalar time for each sample with shape ``[B, 1]``.
            v_feat: Optional velocity features with shape ``[B, d_latent]``.
                When ``None``, the velocity contribution is set to zero.

        Returns:
            torch.Tensor: Query vector with shape ``[B, d_out]``, combining
                linear and Christoffel correction terms.
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
        """Initialize Lorentzian memory attention.

        Sets up the Lorentzian metric, causal light-cone mask, geodesic
        query generator, key/value projections, and the learnable Wilson
        loop attenuation scale.

        Args:
            config: Lorentzian configuration specifying model dimensions,
                latent dimensions, information speed, and other
                hyperparameters.
        """
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

        Computes attention scores using geodesic queries and keys, applies
        Wilson-loop attenuation based on geodesic distance, rescales by
        the metric-aware temperature, and masks out memory slots outside
        the causal light-cone.

        Args:
            x: Current input features with shape ``[B, d_model]``.
            z: Current latent position on the Poincare disk with shape
                ``[B, d_latent]``.
            t: Current time with shape ``[B, 1]``.
            x_mem: Memory feature vectors with shape ``[B, N, d_model]``.
            z_mem: Memory positions on the Poincare disk with shape
                ``[B, N, d_latent]``.
            t_mem: Memory times with shape ``[B, N, 1]``.
            v_feat: Optional velocity features with shape ``[B, d_latent]``.
                When ``None``, velocity contribution to the query is zero.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of:
                - **output**: Attended feature vector with shape
                  ``[B, d_model]``, aggregated from causally accessible
                  memory slots weighted by geodesic proximity.
                - **weights**: Normalized attention weights with shape
                  ``[B, N]``, zero for acausal memory slots.
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
