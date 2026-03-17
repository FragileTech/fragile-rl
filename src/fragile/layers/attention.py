from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import ConformalMetric


def _as_query_tokens(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    """Normalize a query tensor to shape ``[B, Q, ...]``."""
    if tensor.dim() == 2:
        return tensor.unsqueeze(1), True
    if tensor.dim() == 3:
        return tensor, False
    msg = f"{name} must have shape [B, D] or [B, Q, D]."
    raise ValueError(msg)


def _as_optional_query_tokens(
    tensor: torch.Tensor | None,
    name: str,
    expected_batch: int,
    expected_queries: int,
) -> torch.Tensor | None:
    """Normalize optional query-conditioned inputs to ``[B, Q, ...]``."""
    if tensor is None:
        return None
    tokens, _ = _as_query_tokens(tensor, name)
    if tokens.shape[0] != expected_batch or tokens.shape[1] != expected_queries:
        msg = f"{name} must match the query batch shape [B, Q, D]."
        raise ValueError(msg)
    return tokens


@dataclass
class GeodesicConfig:
    """Configuration for hyperbolic covariant attention on the Poincare ball."""

    d_model: int = 256
    d_latent: int = 64
    n_heads: int = 1
    T_c: float = 0.1
    gamma_friction: float = 1.0
    dt: float = 0.01
    g_s: float = 1.0
    g_2: float = 0.5
    g_1: float = 0.3
    use_learned_thermostat: bool = False
    thermostat_residual_scale: float = 0.1


class HyperbolicTransport(nn.Module):
    """Hyperbolic transport scale on the Poincare ball.

    This module is intentionally hyperbolic-only. It uses the conformal-factor
    ratio between source and destination points, which is the closed-form
    transport scale for the Poincare ball along radial geodesics.
    """

    def __init__(self, config: GeodesicConfig, d_k: int, curvature: float = 1.0) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_latent = config.d_latent
        self.curvature = curvature
        self.metric = ConformalMetric()

    def _scale_factors(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        z_query_tokens, _ = _as_query_tokens(z_query, "z_query")
        if z_key.dim() != 3:
            msg = "z_key must have shape [B, N, D]."
            raise ValueError(msg)
        if (
            z_query_tokens.shape[0] != z_key.shape[0]
            or z_query_tokens.shape[-1] != z_key.shape[-1]
        ):
            msg = "z_query and z_key must agree on batch size and latent dimension."
            raise ValueError(msg)

        batch_size, n_queries, d_latent = z_query_tokens.shape
        n_keys = z_key.shape[1]

        lambda_query = self.metric.conformal_factor(
            z_query_tokens.reshape(batch_size * n_queries, d_latent),
        ).reshape(batch_size, n_queries, 1, 1)
        lambda_key = self.metric.conformal_factor(
            z_key.reshape(batch_size * n_keys, d_latent),
        ).reshape(batch_size, 1, n_keys, 1)
        return lambda_key / (lambda_query + 1e-6)

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Return hyperbolic transport scales.

        Returns:
            ``[B, N, 1]`` for a single query point or ``[B, Q, N, 1]`` for a
            bank of query points.
        """
        _, squeeze_query = _as_query_tokens(z_query, "z_query")
        scale = self._scale_factors(z_query, z_key)
        return scale.squeeze(1) if squeeze_query else scale


class WilsonLineApprox(nn.Module):
    """Hyperbolic Wilson-line-style transport on the Poincare ball.

    The exact hyperbolic transport is the conformal scale factor; this module
    adds a first-order skew correction in the local displacement to mimic the
    Wilson-line term used in the theory notes while remaining specific to the
    Poincare-ball geometry.
    """

    def __init__(self, config: GeodesicConfig, d_k: int, d_conn: int = 8) -> None:
        super().__init__()
        self.d_k = d_k
        self.d_conn = min(d_conn, config.d_latent)
        self.delta_proj = nn.Linear(config.d_latent, self.d_conn, bias=False)
        self.basis_binding = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))
        self.basis_error = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))
        self.basis_opportunity = nn.Parameter(0.01 * torch.randn(self.d_conn, d_k, d_k))
        self.g_s = config.g_s
        self.g_2 = config.g_2
        self.g_1 = config.g_1
        self.transport = HyperbolicTransport(config, d_k)

    @staticmethod
    def _skew(basis: torch.Tensor) -> torch.Tensor:
        return basis - basis.transpose(-1, -2)

    def _transport_matrices(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        z_query_tokens, _ = _as_query_tokens(z_query, "z_query")
        if z_key.dim() != 3:
            msg = "z_key must have shape [B, N, D]."
            raise ValueError(msg)

        delta_z = z_query_tokens.unsqueeze(2) - z_key.unsqueeze(1)
        coeff = self.delta_proj(delta_z)

        h = (
            self.g_s * torch.einsum("bqnr,rij->bqnij", coeff, self._skew(self.basis_binding))
            + self.g_2 * torch.einsum("bqnr,rij->bqnij", coeff, self._skew(self.basis_error))
            + self.g_1 * torch.einsum("bqnr,rij->bqnij", coeff, self._skew(self.basis_opportunity))
        )

        scale = self.transport._scale_factors(z_query_tokens, z_key).unsqueeze(-1)
        identity = torch.eye(self.d_k, device=z_key.device, dtype=z_key.dtype).view(
            1,
            1,
            1,
            self.d_k,
            self.d_k,
        )
        return scale * (identity + h)

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Return hyperbolic transport matrices.

        Returns:
            ``[B, N, d_k, d_k]`` for a single query point or
            ``[B, Q, N, d_k, d_k]`` for a bank of query points.
        """
        _, squeeze_query = _as_query_tokens(z_query, "z_query")
        matrices = self._transport_matrices(z_query, z_key)
        return matrices.squeeze(1) if squeeze_query else matrices


class ChristoffelQuery(nn.Module):
    """Geometric query projection encoding Poincare-ball Christoffel terms."""

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_Qz = nn.Linear(d_latent, d_out, bias=False)
        self.W_Qv = nn.Linear(d_in, d_out, bias=False)

        self.W_Q_gamma = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))
        self._init_christoffel(d_latent)
        self.W_Qzv = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))

    def _init_christoffel(self, d_latent: int) -> None:
        with torch.no_grad():
            for k in range(min(d_latent, self.W_Q_gamma.shape[0])):
                for i in range(d_latent):
                    for j in range(d_latent):
                        if k in {i, j}:
                            self.W_Q_gamma[k, i, j] = 0.01
                        if i == j:
                            self.W_Q_gamma[k, i, j] -= 0.01

    def forward(
        self,
        x: torch.Tensor,
        z_geom: torch.Tensor,
        v_feat: torch.Tensor | None = None,
        v_geom: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the Christoffel-aware query vector."""
        q = self.W_Q(x) + self.W_Qz(z_geom)
        if v_feat is not None:
            q = q + self.W_Qv(v_feat)

        d_latent = min(z_geom.shape[-1], self.W_Q_gamma.shape[-1])
        z_trunc = z_geom[..., :d_latent]
        q_gamma = torch.einsum(
            "aij,bi,bj->ba",
            self.W_Q_gamma[:, :d_latent, :d_latent],
            z_trunc,
            z_trunc,
        )
        q = q + q_gamma

        if v_geom is not None:
            v_trunc = v_geom[..., :d_latent]
            q_zv = torch.einsum(
                "aij,bi,bj->ba",
                self.W_Qzv[:, :d_latent, :d_latent],
                z_trunc,
                v_trunc,
            )
            q = q + q_zv

        return q


class ChiralProjector(nn.Module):
    """SU(2)-style chiral projector driven by the value gradient."""

    def __init__(self, d_latent: int) -> None:
        super().__init__()
        self.grad_proj = nn.Linear(d_latent, 3, bias=False)

        self.register_buffer("identity", torch.eye(2))
        self.register_buffer("sigma_1", torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_2", torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_3", torch.tensor([[1.0, 0.0], [0.0, -1.0]]))

    def forward(self, psi_doublet: torch.Tensor, grad_V: torch.Tensor) -> torch.Tensor:
        """Project a doublet-valued representation onto the committed channel."""
        n_vec = self.grad_proj(grad_V)
        n_hat = n_vec / (torch.norm(n_vec, dim=-1, keepdim=True) + 1e-8)
        n_x, n_y, n_z = n_hat.unbind(dim=-1)

        proj = 0.5 * (
            self.identity
            + n_x[:, None, None] * self.sigma_1
            + n_y[:, None, None] * self.sigma_2
            + n_z[:, None, None] * self.sigma_3
        )

        psi_proj = torch.einsum("bij,bjd->bid", proj, psi_doublet)
        commit_strength = (psi_doublet * psi_proj).sum(dim=1, keepdim=True)
        psi_proj = psi_proj * commit_strength
        return psi_proj.reshape(psi_proj.shape[0], -1)


class AreaLawScreening(nn.Module):
    """Area-law screening for hyperbolic attention weights."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(config.g_s**2)))

    @property
    def sigma(self) -> torch.Tensor:
        return torch.exp(self.log_sigma)

    def string_area(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the hyperbolic string-area proxy.

        Supports single-query inputs ``[B, D]`` and query banks ``[B, Q, D]``.
        """
        if z_query.dim() == 2:
            delta = z_query.unsqueeze(1) - z_key
            dist_sq = (delta**2).sum(dim=-1)
            return 0.5 * (lambda_z**2) * dist_sq
        if z_query.dim() == 3:
            delta = z_query.unsqueeze(2) - z_key.unsqueeze(1)
            dist_sq = (delta**2).sum(dim=-1)
            return 0.5 * (lambda_z**2) * dist_sq
        msg = "z_query must have shape [B, D] or [B, Q, D]."
        raise ValueError(msg)

    def forward(
        self,
        attention: torch.Tensor,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
        level: int = 0,
        l_max: float = 10.0,
    ) -> torch.Tensor:
        """Apply area-law screening to attention weights."""
        area = self.string_area(z_query, z_key, lambda_z)
        sigma_eff = self.sigma * math.exp(-level / l_max)
        screening = torch.exp(-sigma_eff * area)

        if attention.dim() == screening.dim():
            return attention * screening
        if attention.dim() == screening.dim() + 1:
            return attention * screening.unsqueeze(-2)
        msg = "attention shape is incompatible with hyperbolic area-law screening."
        raise ValueError(msg)


class CovariantCrossAttention(nn.Module):
    """Hyperbolic covariant cross-attention on the Poincare ball.

    This implementation is intentionally specific to hyperbolic geometry. It
    combines:
    - Christoffel-aware query projections,
    - hyperbolic Wilson-line transport between query/key positions,
    - position-dependent temperature from the Poincare conformal factor,
    - optional area-law screening,
    - optional chiral projection of the output.
    """

    def __init__(
        self,
        config: GeodesicConfig,
        use_chirality: bool = False,
        use_screening: bool = False,
        head_type: str = "generic",
    ) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            msg = "d_model must be divisible by n_heads."
            raise ValueError(msg)

        self.config = config
        self.use_chirality = use_chirality
        self.use_screening = use_screening
        self.head_type = head_type
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        self.d_model = config.d_model

        self.query = ChristoffelQuery(config.d_model, config.d_model, config.d_latent)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)

        self.wilson = WilsonLineApprox(config, self.d_k)
        self.metric = ConformalMetric()

        if use_chirality:
            self.chiral = ChiralProjector(config.d_latent)
        if use_screening:
            self.screening = AreaLawScreening(config)

    def _prepare_qkv(
        self,
        z_query: torch.Tensor,
        x_query: torch.Tensor,
        v_query: torch.Tensor | None,
        v_query_geom: torch.Tensor | None,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        z_query_tokens, squeeze_query = _as_query_tokens(z_query, "z_query")
        x_query_tokens, _ = _as_query_tokens(x_query, "x_query")
        if z_query_tokens.shape[:2] != x_query_tokens.shape[:2]:
            msg = "z_query and x_query must describe the same query bank."
            raise ValueError(msg)

        batch_size, n_queries, _ = z_query_tokens.shape
        v_query_tokens = _as_optional_query_tokens(v_query, "v_query", batch_size, n_queries)
        v_query_geom_tokens = _as_optional_query_tokens(
            v_query_geom,
            "v_query_geom",
            batch_size,
            n_queries,
        )

        if x_key.dim() != 3 or x_value.dim() != 3:
            msg = "x_key and x_value must have shape [B, N, d_model]."
            raise ValueError(msg)
        if x_key.shape != x_value.shape:
            msg = "x_key and x_value must have identical shapes."
            raise ValueError(msg)
        if x_key.shape[0] != batch_size:
            msg = "Key/value batch size must match the query batch size."
            raise ValueError(msg)

        q = self.query(
            x_query_tokens.reshape(batch_size * n_queries, -1),
            z_query_tokens.reshape(batch_size * n_queries, -1),
            None if v_query_tokens is None else v_query_tokens.reshape(batch_size * n_queries, -1),
            None
            if v_query_geom_tokens is None
            else v_query_geom_tokens.reshape(batch_size * n_queries, -1),
        ).reshape(batch_size, n_queries, self.n_heads, self.d_k)

        n_keys = x_key.shape[1]
        k = self.key(x_key).reshape(batch_size, n_keys, self.n_heads, self.d_k)
        v = self.value(x_value).reshape(batch_size, n_keys, self.n_heads, self.d_k)
        return z_query_tokens, q, k, v, squeeze_query

    def forward(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        x_query: torch.Tensor,
        x_key: torch.Tensor,
        x_value: torch.Tensor,
        v_query: torch.Tensor | None = None,
        v_query_geom: torch.Tensor | None = None,
        grad_V: torch.Tensor | None = None,
        level: int = 0,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute hyperbolic covariant cross-attention.

        The query inputs may be a single latent ``[B, D]`` or a bank of query
        latents ``[B, Q, D]``. Keys and values always use ``[B, N, D]``.
        """
        if z_key.dim() != 3:
            msg = "z_key must have shape [B, N, d_latent]."
            raise ValueError(msg)

        z_query_tokens, q, k, v, squeeze_query = self._prepare_qkv(
            z_query,
            x_query,
            v_query,
            v_query_geom,
            x_key,
            x_value,
        )
        batch_size, n_queries, _, _ = q.shape
        n_keys = z_key.shape[1]
        if z_key.shape[0] != batch_size:
            msg = "z_query and z_key must have matching batch size."
            raise ValueError(msg)

        transport = self.wilson(z_query_tokens, z_key)
        if transport.dim() == 4:
            transport = transport.unsqueeze(1)
        k_transported = torch.einsum("bqnde,bnhe->bqnhd", transport, k)
        scores = torch.einsum("bqhd,bqnhd->bqhn", q, k_transported)

        tau = self.metric.temperature(
            z_query_tokens.reshape(batch_size * n_queries, -1),
            self.d_k,
        ).reshape(batch_size, n_queries, 1, 1)
        scores = scores / (tau + 1e-8)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            if mask.shape != (batch_size, n_queries, n_keys):
                msg = "mask must have shape [B, N] or [B, Q, N]."
                raise ValueError(msg)
            scores = scores.masked_fill(~mask.unsqueeze(-2).bool(), torch.finfo(scores.dtype).min)

        attention = F.softmax(scores, dim=-1)

        if self.use_screening:
            lambda_z = self.metric.conformal_factor(
                z_query_tokens.reshape(batch_size * n_queries, -1),
            ).reshape(batch_size, n_queries, 1)
            attention = self.screening(attention, z_query_tokens, z_key, lambda_z, level)
            attention = attention / (attention.sum(dim=-1, keepdim=True) + 1e-8)

        output = torch.einsum("bqhn,bnhd->bqhd", attention, v).reshape(
            batch_size,
            n_queries,
            self.d_model,
        )

        grad_tokens = _as_optional_query_tokens(grad_V, "grad_V", batch_size, n_queries)
        if self.use_chirality and grad_tokens is not None:
            if output.shape[-1] % 2 != 0:
                msg = "Chiral projection requires an even model dimension."
                raise ValueError(msg)
            output = self.chiral(
                output.reshape(batch_size * n_queries, 2, -1),
                grad_tokens.reshape(batch_size * n_queries, -1),
            ).reshape(batch_size, n_queries, -1)

        output = self.output(output)
        attention_out = attention.mean(dim=-2)

        if squeeze_query:
            return output.squeeze(1), attention_out.squeeze(1)
        return output, attention_out


class CovariantAttention(CovariantCrossAttention):
    """Backward-compatible alias for single-query hyperbolic covariant attention."""


class GeodesicCrossAttention(nn.Module):
    """Hyperbolic BAOAB integrator driven by covariant cross-attention heads."""

    def __init__(self, config: GeodesicConfig) -> None:
        super().__init__()
        self.config = config
        self.dt = config.dt
        self.gamma = config.gamma_friction
        self.T_c = config.T_c
        self.c1 = math.exp(-self.gamma * self.dt)
        self.c2 = math.sqrt((1.0 - self.c1**2) * self.T_c) if self.T_c > 0 else 0.0

        self.metric = ConformalMetric()

        self.head_B1 = CovariantAttention(config, head_type="B")
        self.head_A1 = CovariantAttention(config, head_type="A")
        self.use_learned_thermostat = config.use_learned_thermostat
        self.thermostat_residual_scale = config.thermostat_residual_scale
        if self.use_learned_thermostat:
            self.head_O = CovariantAttention(config, head_type="O")
        else:
            self.head_O = None
        self.head_A2 = CovariantAttention(config, head_type="A")
        self.head_B2 = CovariantAttention(config, head_type="B")

        self.pos_encoder = nn.Linear(config.d_latent, config.d_model)
        self.grad_encoder = nn.Linear(config.d_latent, config.d_model)
        self.velocity_encoder = nn.Linear(config.d_latent, config.d_model)
        self.state_proj = nn.Linear(config.d_model, config.d_latent)

        if self.use_learned_thermostat:
            self.noise_proj = nn.Linear(config.d_latent, config.d_model)
        else:
            self.noise_proj = None

    def forward(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        context_z: torch.Tensor,
        context_x: torch.Tensor,
        context_force: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One BAOAB step on the Poincare ball."""
        h = self.dt
        force_features = self.grad_encoder(context_force)

        delta_p1, _ = self.head_B1(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z),
            x_key=force_features,
            x_value=force_features,
        )
        p = p - h / 2.0 * self.state_proj(delta_p1)

        g_inv = self.metric.metric_inv(z)
        v = torch.einsum("bij,bj->bi", g_inv, p)

        delta_z1, _ = self.head_A1(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z) + self.velocity_encoder(v),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
            v_query_geom=v,
        )
        z = z + h / 2.0 * (v + self.state_proj(delta_z1))
        z = self._project_to_disk(z)

        g_sqrt = self.metric.conformal_factor(z)
        xi = torch.randn_like(p)
        p = self.c1 * p + self.c2 * g_sqrt * xi

        if self.use_learned_thermostat:
            noise_bank = torch.randn_like(context_z)
            noise_features = self.noise_proj(noise_bank)
            delta_p_noise, _ = self.head_O(
                z_query=z,
                z_key=context_z,
                x_query=self.velocity_encoder(p),
                x_key=noise_features,
                x_value=noise_features,
            )
            p = p + self.thermostat_residual_scale * self.state_proj(delta_p_noise)

        g_inv = self.metric.metric_inv(z)
        v = torch.einsum("bij,bj->bi", g_inv, p)

        delta_z2, _ = self.head_A2(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z) + self.velocity_encoder(v),
            x_key=context_x,
            x_value=context_x,
            v_query=self.velocity_encoder(v),
            v_query_geom=v,
        )
        z = z + h / 2.0 * (v + self.state_proj(delta_z2))
        z = self._project_to_disk(z)

        delta_p2, _ = self.head_B2(
            z_query=z,
            z_key=context_z,
            x_query=self.pos_encoder(z),
            x_key=force_features,
            x_value=force_features,
        )
        p = p - h / 2.0 * self.state_proj(delta_p2)
        return z, p

    def _project_to_disk(self, z: torch.Tensor, max_norm: float = 0.999) -> torch.Tensor:
        """Project positions to the interior of the Poincare ball."""
        norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.where(norm > max_norm, z * max_norm / norm, z)


class GeodesicBAOAB(GeodesicCrossAttention):
    """Backward-compatible alias for the documented GeodesicCrossAttention."""


__all__ = [
    "AreaLawScreening",
    "ChiralProjector",
    "ChristoffelQuery",
    "CovariantAttention",
    "CovariantCrossAttention",
    "GeodesicBAOAB",
    "GeodesicConfig",
    "GeodesicCrossAttention",
    "HyperbolicTransport",
    "WilsonLineApprox",
]
