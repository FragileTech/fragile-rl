from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import ConformalMetric


def _as_query_tokens(tensor: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    """Normalize single-query and multi-query inputs to a common token layout.

    Args:
        tensor: Tensor shaped either ``[batch, dim]`` for one query per sample
            or ``[batch, n_queries, dim]`` for an explicit query bank.
        name: Name used in validation errors.

    Returns:
        A pair ``(tokens, squeezed)`` where ``tokens`` always has shape
        ``[batch, n_queries, dim]`` and ``squeezed`` records whether a singleton
        query axis was introduced.
    """
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
    """Normalize an optional query-aligned tensor and validate its leading shape.

    Args:
        tensor: Optional tensor shaped like a single query bank input.
        name: Name used in validation errors.
        expected_batch: Batch size that must match the normalized query bank.
        expected_queries: Query count that must match the normalized query bank.

    Returns:
        ``None`` when ``tensor`` is ``None``; otherwise a rank-3 tensor with
        leading shape ``[expected_batch, expected_queries, ...]``.
    """
    if tensor is None:
        return None
    tokens, _ = _as_query_tokens(tensor, name)
    if tokens.shape[0] != expected_batch or tokens.shape[1] != expected_queries:
        msg = f"{name} must match the query batch shape [B, Q, D]."
        raise ValueError(msg)
    return tokens


@dataclass
class GeodesicConfig:
    """Shared hyperparameters for hyperbolic attention and BAOAB updates.

    The fields are consumed by several modules in this file:
    - ``d_model``, ``d_latent``, and ``n_heads`` control feature and latent sizes.
    - ``g_s``, ``g_2``, and ``g_1`` weight the three skew bases used by
      :class:`WilsonLineApprox`.
    - ``dt``, ``gamma_friction``, and ``T_c`` parameterize the BAOAB-style
      update in :class:`GeodesicCrossAttention`.
    - ``use_learned_thermostat`` and ``thermostat_residual_scale`` enable and
      scale the optional learned thermostat correction.
    """

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
        """Store latent dimensions and metric helpers for transport scaling.

        Args:
            config: Global geometry configuration. Only ``d_latent`` is used
                directly by this module.
            d_k: Per-head feature width associated with transported vectors.
            curvature: Stored for API compatibility; the current implementation
                always uses :class:`ConformalMetric` on the unit Poincare ball.
        """
        super().__init__()
        self.d_k = d_k
        self.d_latent = config.d_latent
        self.curvature = curvature
        self.metric = ConformalMetric()

    def _scale_factors(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Compute conformal-factor ratios between query and key positions.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.

        Returns:
            A tensor of shape ``[batch, n_queries, n_keys, 1]`` containing
            ``lambda(z_key) / lambda(z_query)`` for each query-key pair.
        """
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
        """Return pairwise transport scales in a layout matching the query input.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.

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
        """Initialize the displacement projection and skew transport bases.

        Args:
            config: Hyperparameters providing latent size and basis weights.
            d_k: Per-head feature width of the transported key vectors.
            d_conn: Width of the learned displacement feature used to combine the
                skew bases. It is clipped to ``config.d_latent``.
        """
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
        """Return the antisymmetric part of a basis tensor.

        Args:
            basis: A square matrix or batch of square matrices of shape
                ``[..., d, d]``.

        Returns:
            The antisymmetric component ``(basis - basis^T) / 1``, with the
            same shape as the input.
        """
        return basis - basis.transpose(-1, -2)

    def _transport_matrices(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        """Build Wilson-line-style transport matrices for each query-key pair.

        The matrices combine the scalar hyperbolic transport factor from
        :class:`HyperbolicTransport` with a first-order skew correction derived
        from the projected displacement ``z_query - z_key``.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.

        Returns:
            A tensor of shape ``[batch, n_queries, n_keys, d_k, d_k]``.
        """
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
        """Return transport matrices in a layout matching the query input.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.

        Returns:
            ``[B, N, d_k, d_k]`` for a single query point or
            ``[B, Q, N, d_k, d_k]`` for a bank of query points.
        """
        _, squeeze_query = _as_query_tokens(z_query, "z_query")
        matrices = self._transport_matrices(z_query, z_key)
        return matrices.squeeze(1) if squeeze_query else matrices


class ChristoffelQuery(nn.Module):
    """Query projection augmented with latent and geometry-dependent corrections.

    The output starts from linear projections of the feature input ``x`` and the
    latent geometry ``z_geom``. Optional velocity features add another linear
    term, while two learned bilinear tensors contribute quadratic ``z-z`` and
    mixed ``z-v`` corrections. ``W_Q_gamma`` is initialized with a small
    Christoffel-inspired pattern and ``W_Qzv`` starts at zero.
    """

    def __init__(self, d_in: int, d_out: int, d_latent: int) -> None:
        """Create the linear and bilinear terms used to build query vectors.

        Args:
            d_in: Width of the incoming feature representation.
            d_out: Width of the resulting query representation.
            d_latent: Width of the latent geometric state.
        """
        super().__init__()
        self.W_Q = nn.Linear(d_in, d_out, bias=False)
        self.W_Qz = nn.Linear(d_latent, d_out, bias=False)
        self.W_Qv = nn.Linear(d_in, d_out, bias=False)

        self.W_Q_gamma = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))
        self._init_christoffel(d_latent)
        self.W_Qzv = nn.Parameter(torch.zeros(d_out, d_latent, d_latent))

    def _init_christoffel(self, d_latent: int) -> None:
        """Seed ``W_Q_gamma`` with a small structured Christoffel-like pattern.

        The initialization sets entries to +0.01 where the output index matches
        either spatial index, and subtracts 0.01 on the diagonal (i == j),
        mimicking the symmetry structure of Christoffel symbols.

        Args:
            d_latent: Latent space dimensionality used to bound the loop ranges
                over the spatial indices of ``W_Q_gamma``.

        Returns:
            None. The method modifies ``self.W_Q_gamma`` in-place.
        """
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
        """Project feature and geometric inputs into the query space.

        Args:
            x: Query features of shape ``[batch, d_in]``.
            z_geom: Query positions or geometric descriptors of shape
                ``[batch, d_latent]``.
            v_feat: Optional velocity-like features in the same feature space as
                ``x``.
            v_geom: Optional geometry-space velocity term used by the mixed
                bilinear correction.

        Returns:
            A tensor of shape ``[batch, d_out]``.
        """
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
    """Project 2-channel features onto a gradient-chosen SU(2)-like direction."""

    def __init__(self, d_latent: int) -> None:
        """Learn the gradient-to-direction map and register Pauli-like bases.

        Args:
            d_latent: Width of the gradient signal used to choose the projection
                direction.
        """
        super().__init__()
        self.grad_proj = nn.Linear(d_latent, 3, bias=False)

        self.register_buffer("identity", torch.eye(2))
        self.register_buffer("sigma_1", torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_2", torch.tensor([[0.0, -1.0], [1.0, 0.0]]))
        self.register_buffer("sigma_3", torch.tensor([[1.0, 0.0], [0.0, -1.0]]))

    def forward(self, psi_doublet: torch.Tensor, grad_V: torch.Tensor) -> torch.Tensor:
        """Project a doublet-valued representation onto the committed channel.

        Args:
            psi_doublet: Tensor of shape ``[batch, 2, width]`` representing a
                two-channel feature doublet.
            grad_V: Tensor of shape ``[batch, d_latent]`` whose projection picks
                the chiral direction.

        Returns:
            A flattened tensor of shape ``[batch, 2 * width]``.
        """
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
    """Exponentially damp attention using a hyperbolic string-area proxy."""

    def __init__(self, config: GeodesicConfig) -> None:
        """Initialize the learnable screening strength from ``config.g_s``.

        Args:
            config: Geometry configuration whose ``g_s`` field sets the initial
                screening coefficient via ``log_sigma = log(g_s ** 2)``.
        """
        super().__init__()
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(config.g_s**2)))

    @property
    def sigma(self) -> torch.Tensor:
        """Return the positive screening coefficient.

        Returns:
            A scalar tensor containing ``exp(log_sigma)``, guaranteed to be
            positive.
        """
        return torch.exp(self.log_sigma)

    def string_area(
        self,
        z_query: torch.Tensor,
        z_key: torch.Tensor,
        lambda_z: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the query-key area proxy used for screening.

        Supports single-query inputs ``[B, D]`` and query banks ``[B, Q, D]`` by
        forming squared Euclidean displacements and scaling them by the squared
        conformal factor at the query location.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.
            lambda_z: Query conformal factor shaped ``[batch, 1]`` or
                ``[batch, n_queries, 1]``.

        Returns:
            A tensor of pairwise area proxies with shape ``[batch, n_keys]`` or
            ``[batch, n_queries, n_keys]``.
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
        """Apply level-dependent screening to an attention tensor.

        Args:
            attention: Attention weights shaped either ``[batch, n_queries,
                n_keys]`` or ``[batch, n_queries, n_heads, n_keys]``.
            z_query: Query positions.
            z_key: Key positions.
            lambda_z: Conformal factor at the query positions.
            level: Hierarchical level used to decay the effective screening
                coefficient.
            l_max: Characteristic scale for the exponential level decay.

        Returns:
            ``attention`` multiplied by the screening factor, preserving the
            input shape.
        """
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
        """Construct a hyperbolic cross-attention head stack.

        Args:
            config: Shared model and geometry hyperparameters.
            use_chirality: Whether to post-process the attention output with
                :class:`ChiralProjector` when ``grad_V`` is provided.
            use_screening: Whether to damp attention weights with
                :class:`AreaLawScreening`.
            head_type: Label retained on the module for higher-level callers; it
                does not change the computation in this class.
        """
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
        """Normalize query inputs and build multi-head query/key/value tensors.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            x_query: Query features matching the query bank layout.
            v_query: Optional feature-space velocity term aligned with the query
                bank.
            v_query_geom: Optional latent-space velocity term aligned with the
                query bank.
            x_key: Key features of shape ``[batch, n_keys, d_model]``.
            x_value: Value features of shape ``[batch, n_keys, d_model]``.

        Returns:
            ``(z_query_tokens, q, k, v, squeeze_query)`` where ``q`` has shape
            ``[batch, n_queries, n_heads, d_k]``, ``k`` and ``v`` have shape
            ``[batch, n_keys, n_heads, d_k]``, and ``squeeze_query`` records
            whether the original query input had no explicit query axis.
        """
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
        """Compute hyperbolic cross-attention with geometric transport.

        The query inputs may be a single latent ``[B, D]`` or a bank of query
        latents ``[B, Q, D]``. Keys and values always use ``[B, N, D]``.

        The computation proceeds as follows:
        1. Build Christoffel-aware multi-head queries and plain projected keys/values.
        2. Transport keys from each key position to each query position with
           :class:`WilsonLineApprox`.
        3. Score transported keys against queries, divide by a hyperbolic
           temperature, and apply an optional mask.
        4. Softmax the scores, optionally screen them, and aggregate values.
        5. Optionally apply chiral projection and the final output projection.

        Args:
            z_query: Query positions shaped ``[batch, d_latent]`` or
                ``[batch, n_queries, d_latent]``.
            z_key: Key positions shaped ``[batch, n_keys, d_latent]``.
            x_query: Query features aligned with ``z_query``.
            x_key: Key features of shape ``[batch, n_keys, d_model]``.
            x_value: Value features of shape ``[batch, n_keys, d_model]``.
            v_query: Optional query-aligned feature-space velocity term.
            v_query_geom: Optional query-aligned latent-space velocity term.
            grad_V: Optional query-aligned gradient used only when chirality is
                enabled.
            level: Hierarchy level passed to the screening module.
            mask: Optional boolean or boolean-like mask shaped ``[batch, n_keys]``
                or ``[batch, n_queries, n_keys]``.

        Returns:
            A pair ``(output, attention)``. ``output`` has shape
            ``[batch, d_model]`` or ``[batch, n_queries, d_model]`` depending on
            the query layout. ``attention`` is the head-averaged attention over
            keys with the corresponding query layout.
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
    """Backward-compatible alias for :class:`CovariantCrossAttention`."""


class GeodesicCrossAttention(nn.Module):
    """BAOAB-style latent integrator assembled from covariant attention heads.

    Despite the name, this class is not a plain cross-attention block. It uses
    four attention heads to generate the ``B-A-O-A-B`` updates for momentum and
    position, plus an optional learned thermostat correction during the ``O``
    step. Latent positions are kept inside the Poincare ball after each drift.
    """

    def __init__(self, config: GeodesicConfig) -> None:
        """Build the BAOAB sub-heads and feature encoders from ``config``.

        Creates five covariant attention heads (B1, A1, O, A2, B2) and the
        linear encoders that map latent positions, velocities, gradients, and
        noise into the feature space consumed by those heads.

        Args:
            config: Shared hyperparameters providing model dimensions
                (``d_model``, ``d_latent``), integrator constants (``dt``,
                ``gamma_friction``, ``T_c``), and the learned-thermostat flag
                (``use_learned_thermostat``, ``thermostat_residual_scale``).
        """
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
        """Advance position and momentum by one BAOAB-style update.

        Args:
            z: Current latent positions of shape ``[batch, d_latent]``.
            p: Current latent momenta of shape ``[batch, d_latent]``.
            context_z: Context positions used as keys for the attention heads.
            context_x: Context features used during the ``A`` drift substeps.
            context_force: Context force-like features used during the ``B``
                momentum substeps.

        Returns:
            The updated ``(z, p)`` pair after ``B-A-O-A-B`` integration.
        """
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
        """Clamp latent positions to the interior of the Poincare ball.

        Points whose norm exceeds ``max_norm`` are rescaled to lie on the
        boundary; all other points are left unchanged.

        Args:
            z: Latent positions of shape ``[batch, d_latent]``.
            max_norm: Maximum allowed Euclidean norm. Defaults to 0.999 to
                keep points strictly inside the open unit ball.

        Returns:
            A tensor of the same shape as ``z`` with all norms at most
            ``max_norm``.
        """
        norm = torch.norm(z, dim=-1, keepdim=True).clamp(min=1e-8)
        return torch.where(norm > max_norm, z * max_norm / norm, z)


class GeodesicBAOAB(GeodesicCrossAttention):
    """Backward-compatible alias for :class:`GeodesicCrossAttention`."""


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
