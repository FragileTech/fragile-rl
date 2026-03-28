"""Routing-style geometry heads for the coarse symbolic Markov model.

These modules keep the atlas symbols as the actual macro state and only learn
how to summarize soft symbolic beliefs and score next-chart / next-code
probabilities against the existing atlas geometry.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import (
    exp_map_zero,
    hyperbolic_distance,
    log_map_zero,
    mobius_add,
    project_to_ball,
)
from fragile.layers.primitives import SpectralLinear


def _normalize_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a non-negative tensor along the last axis.

    Args:
        probs: Non-negative tensor of arbitrary shape to be normalized along
            its last dimension.
        eps: Small constant to clamp the denominator and avoid division by
            zero.

    Returns:
        torch.Tensor: Tensor of the same shape as ``probs`` whose last
            dimension sums to 1.
    """
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)


def _routing_temperature(
    z: torch.Tensor,
    latent_dim: int,
    *,
    tau_min: float = 1e-2,
    tau_denom_min: float = 1e-3,
) -> torch.Tensor:
    """Compute a geometry-derived temperature schedule for the chart router.

    The temperature decreases as points approach the boundary of the Poincare
    ball, sharpening routing distributions in high-curvature regions.

    Args:
        z: Points in the Poincare ball of shape ``[..., latent_dim]``.
        latent_dim: Dimensionality of the hyperbolic space, used for scaling.
        tau_min: Lower clamp on the returned temperature values.
        tau_denom_min: Lower clamp on ``(1 - ||z||^2)`` to avoid division
            instabilities near the ball boundary.

    Returns:
        torch.Tensor: Per-point temperatures of shape ``[...]`` (the last
            dimension is reduced).
    """
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=tau_denom_min)
    tau = math.sqrt(float(latent_dim)) * denom / 2.0
    return tau.clamp(min=tau_min)


class BeliefGeometryEncoder(nn.Module):
    """Summarize a soft symbolic belief using geometry-derived token features.

    Combines probability-weighted token projections with a mean tangent
    projection to produce a fixed-size summary of a belief distribution over
    atlas symbols.
    """

    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        """Initialize the BeliefGeometryEncoder.

        Args:
            latent_dim: Dimensionality of the tangent-space vectors for each
                atlas symbol.
            hidden_dim: Dimensionality of the internal hidden representations
                and the output summary vector.
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_proj = nn.Sequential(
            SpectralLinear(self.latent_dim, self.hidden_dim),
            nn.GELU(),
            SpectralLinear(self.hidden_dim, self.hidden_dim),
        )
        self.mean_proj = SpectralLinear(self.latent_dim, self.hidden_dim)
        self.output_proj = nn.Sequential(
            SpectralLinear(2 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            SpectralLinear(self.hidden_dim, self.hidden_dim),
        )

    def forward(
        self,
        state_probs: torch.Tensor,
        state_tangent_points: torch.Tensor,
        *,
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """Encode a soft symbolic belief against the atlas symbol geometry.

        Args:
            state_probs: Soft probability distribution over symbols of shape
                ``[..., num_symbols]``. Will be L1-normalized internally.
            state_tangent_points: Tangent-space coordinates for each atlas
                symbol of shape ``[num_symbols, latent_dim]``.
            eps: Small constant used when normalizing ``state_probs`` to avoid
                division by zero.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``"summary"``: Fused belief summary of shape
                  ``[..., hidden_dim]``.
                - ``"expected_tangent"``: Probability-weighted mean tangent
                  vector of shape ``[..., latent_dim]``.
                - ``"token_bank"``: Projected token features of shape
                  ``[num_symbols, hidden_dim]``.

        Raises:
            ValueError: If ``state_tangent_points`` is not 2-D, if the
                number of symbols disagrees between inputs, or if the latent
                dimension does not match ``self.latent_dim``.
        """
        if state_tangent_points.dim() != 2:
            msg = "state_tangent_points must have shape [num_symbols, latent_dim]."
            raise ValueError(msg)
        if state_probs.shape[-1] != state_tangent_points.shape[0]:
            msg = "state_probs and state_tangent_points must agree on the number of symbols."
            raise ValueError(msg)
        if state_tangent_points.shape[-1] != self.latent_dim:
            msg = "state_tangent_points has the wrong latent dimension."
            raise ValueError(msg)

        state_probs = _normalize_probs(state_probs, eps=eps)
        token_bank = self.token_proj(state_tangent_points)
        expected_tangent = torch.einsum("...s,sd->...d", state_probs, state_tangent_points)
        token_summary = torch.einsum("...s,sh->...h", state_probs, token_bank)
        mean_summary = self.mean_proj(expected_tangent)
        summary = self.output_proj(torch.cat([token_summary, mean_summary], dim=-1))
        return {
            "summary": summary,
            "expected_tangent": expected_tangent,
            "token_bank": token_bank,
        }


class NextStateQueryPredictor(nn.Module):
    """Fuse observation and action belief summaries into a next-state query.

    Takes the observation and action belief summaries produced by
    :class:`BeliefGeometryEncoder` and predicts a query point on the
    observation Poincare manifold that represents the expected next state.
    """

    def __init__(
        self,
        hidden_dim: int,
        obs_latent_dim: int,
    ) -> None:
        """Initialize the NextStateQueryPredictor.

        Args:
            hidden_dim: Dimensionality of the belief summaries produced by
                :class:`BeliefGeometryEncoder` and of the internal context
                representation.
            obs_latent_dim: Dimensionality of the observation manifold where
                the output query point lives.
        """
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.obs_latent_dim = int(obs_latent_dim)
        self.context_mlp = nn.Sequential(
            SpectralLinear(4 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            SpectralLinear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.query_head = SpectralLinear(self.hidden_dim, self.obs_latent_dim)

    def forward(
        self,
        obs_summary: torch.Tensor,
        act_summary: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict a next observation query point in the observation manifold.

        The observation and action summaries are fused via concatenation of
        their sum, difference, and element-wise product, then projected to a
        tangent vector which is mapped onto the Poincare ball.

        Args:
            obs_summary: Observation belief summary of shape
                ``[..., hidden_dim]``.
            act_summary: Action belief summary of shape
                ``[..., hidden_dim]``. Must have the same shape as
                ``obs_summary``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``"context"``: Fused context vector of shape
                  ``[..., hidden_dim]``, used downstream by the routers.
                - ``"query_tangent"``: Predicted tangent vector of shape
                  ``[..., obs_latent_dim]``.
                - ``"query_point"``: Predicted query point projected onto the
                  Poincare ball of shape ``[..., obs_latent_dim]``.

        Raises:
            ValueError: If ``obs_summary`` and ``act_summary`` do not have the
                same shape.
        """
        if obs_summary.shape != act_summary.shape:
            msg = "obs_summary and act_summary must have the same shape."
            raise ValueError(msg)
        fused = torch.cat(
            [
                obs_summary,
                act_summary,
                obs_summary - act_summary,
                obs_summary * act_summary,
            ],
            dim=-1,
        )
        context = self.context_mlp(fused)
        query_tangent = self.query_head(context)
        query_point = project_to_ball(exp_map_zero(query_tangent))
        return {
            "context": context,
            "query_tangent": query_tangent,
            "query_point": query_point,
        }


class ChartTransitionRouter(nn.Module):
    """Score next-chart probabilities against the real observation chart centers.

    Combines hyperbolic distance-based logits with learned context-key feature
    logits to produce a soft distribution over observation charts.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        feature_scale: float = 0.1,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
    ) -> None:
        """Initialize the ChartTransitionRouter.

        Args:
            latent_dim: Dimensionality of the Poincare ball where chart centers
                and query points live.
            context_dim: Dimensionality of the context vector produced by
                :class:`NextStateQueryPredictor`.
            feature_scale: Multiplicative weight for the learned feature-based
                logit term added to the geometric distance logits.
            tau_min: Lower clamp on the geometry-derived softmax temperature.
            tau_denom_min: Lower clamp on ``(1 - ||z||^2)`` inside the
                temperature computation to prevent instabilities near the ball
                boundary.
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.feature_scale = float(feature_scale)
        self.tau_min = float(tau_min)
        self.tau_denom_min = float(tau_denom_min)
        self.context_proj = SpectralLinear(self.context_dim, self.context_dim, bias=False)
        self.chart_key_proj = SpectralLinear(self.latent_dim, self.context_dim, bias=False)

    def forward(
        self,
        query_point: torch.Tensor,
        context: torch.Tensor,
        chart_centers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute soft next-chart probabilities from a geometry-aware query.

        Logits are the sum of a temperature-scaled negative hyperbolic distance
        term and a learned bilinear feature term, followed by a softmax over
        charts.

        Args:
            query_point: Predicted next-state query in the Poincare ball of
                shape ``[..., latent_dim]``.
            context: Context vector from :class:`NextStateQueryPredictor` of
                shape ``[..., context_dim]``. Must share the same leading
                dimensions as ``query_point``.
            chart_centers: Observation chart center embeddings of shape
                ``[num_charts, latent_dim]``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``"chart_logits"``: Raw logits of shape
                  ``[..., num_charts]``.
                - ``"chart_log_probs"``: Log-softmax probabilities of shape
                  ``[..., num_charts]``.
                - ``"chart_probs"``: Softmax probabilities of shape
                  ``[..., num_charts]``.
                - ``"chart_tau"``: Per-query temperature values of shape
                  ``[...]``.

        Raises:
            ValueError: If leading dimensions of ``query_point`` and
                ``context`` disagree, or if ``chart_centers`` is not 2-D.
        """
        if query_point.shape[:-1] != context.shape[:-1]:
            msg = "query_point and context must share the same leading shape."
            raise ValueError(msg)
        if chart_centers.dim() != 2:
            msg = "chart_centers must have shape [num_charts, latent_dim]."
            raise ValueError(msg)

        leading_shape = query_point.shape[:-1]
        flat_query = project_to_ball(query_point).reshape(-1, query_point.shape[-1])
        flat_context = context.reshape(-1, context.shape[-1])
        chart_centers = project_to_ball(chart_centers).to(
            device=flat_query.device, dtype=flat_query.dtype
        )

        tau = _routing_temperature(
            flat_query,
            self.latent_dim,
            tau_min=self.tau_min,
            tau_denom_min=self.tau_denom_min,
        )
        base_logits = -hyperbolic_distance(flat_query.unsqueeze(1), chart_centers.unsqueeze(0))
        base_logits = base_logits / tau.unsqueeze(-1)

        context_features = self.context_proj(flat_context)
        chart_keys = self.chart_key_proj(log_map_zero(chart_centers))
        feature_logits = torch.einsum("bh,ch->bc", context_features, chart_keys)
        logits = base_logits + self.feature_scale * feature_logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        return {
            "chart_logits": logits.reshape(*leading_shape, logits.shape[-1]),
            "chart_log_probs": log_probs.reshape(*leading_shape, log_probs.shape[-1]),
            "chart_probs": probs.reshape(*leading_shape, probs.shape[-1]),
            "chart_tau": tau.reshape(*leading_shape),
        }


class ConditionalCodeRouter(nn.Module):
    """Score chart-local codes given a predicted next observation query.

    For each chart, the query point is translated into the chart-local frame
    via Mobius addition, and logits are computed from both hyperbolic distance
    and learned feature similarity against the per-chart codebook entries.
    """

    def __init__(
        self,
        latent_dim: int,
        context_dim: int,
        *,
        feature_scale: float = 0.1,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
    ) -> None:
        """Initialize the ConditionalCodeRouter.

        Args:
            latent_dim: Dimensionality of the Poincare ball where chart centers
                and codebook entries live.
            context_dim: Dimensionality of the context vector produced by
                :class:`NextStateQueryPredictor`.
            feature_scale: Multiplicative weight for the learned feature-based
                logit term added to the geometric distance logits.
            tau_min: Lower clamp on the geometry-derived softmax temperature.
            tau_denom_min: Lower clamp on ``(1 - ||z||^2)`` inside the
                temperature computation to prevent instabilities near the ball
                boundary.
        """
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_dim = int(context_dim)
        self.feature_scale = float(feature_scale)
        self.tau_min = float(tau_min)
        self.tau_denom_min = float(tau_denom_min)
        self.context_proj = SpectralLinear(self.context_dim, self.context_dim, bias=False)
        self.code_key_proj = SpectralLinear(self.latent_dim, self.context_dim, bias=False)

    def forward(
        self,
        query_point: torch.Tensor,
        context: torch.Tensor,
        chart_centers: torch.Tensor,
        codebook: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute chart-conditional next-code probabilities from atlas geometry.

        For every chart, the query is transported into the local frame via
        Mobius addition of the negated chart center and scored against that
        chart's codebook entries using temperature-scaled hyperbolic distance
        plus a learned feature term.

        Args:
            query_point: Predicted next-state query in the Poincare ball of
                shape ``[..., latent_dim]``.
            context: Context vector from :class:`NextStateQueryPredictor` of
                shape ``[..., context_dim]``. Must share the same leading
                dimensions as ``query_point``.
            chart_centers: Observation chart center embeddings of shape
                ``[num_charts, latent_dim]``.
            codebook: Per-chart codebook entries of shape
                ``[num_charts, codes_per_chart, latent_dim]``.

        Returns:
            dict[str, torch.Tensor]: Dictionary with keys:
                - ``"code_logits"``: Raw logits of shape
                  ``[..., num_charts, codes_per_chart]``.
                - ``"code_log_probs"``: Log-softmax probabilities of shape
                  ``[..., num_charts, codes_per_chart]``.
                - ``"code_probs"``: Softmax probabilities of shape
                  ``[..., num_charts, codes_per_chart]``.
                - ``"local_query"``: Chart-local query points of shape
                  ``[..., num_charts, latent_dim]``.
                - ``"code_tau"``: Per-query-per-chart temperature values of
                  shape ``[..., num_charts]``.

        Raises:
            ValueError: If leading dimensions of ``query_point`` and
                ``context`` disagree, if ``chart_centers`` is not 2-D, if
                ``codebook`` is not 3-D, or if the number of charts disagrees
                between ``chart_centers`` and ``codebook``.
        """
        if query_point.shape[:-1] != context.shape[:-1]:
            msg = "query_point and context must share the same leading shape."
            raise ValueError(msg)
        if chart_centers.dim() != 2:
            msg = "chart_centers must have shape [num_charts, latent_dim]."
            raise ValueError(msg)
        if codebook.dim() != 3:
            msg = "codebook must have shape [num_charts, codes_per_chart, latent_dim]."
            raise ValueError(msg)
        if codebook.shape[0] != chart_centers.shape[0]:
            msg = "chart_centers and codebook must agree on the number of charts."
            raise ValueError(msg)

        leading_shape = query_point.shape[:-1]
        flat_query = project_to_ball(query_point).reshape(-1, query_point.shape[-1])
        flat_context = context.reshape(-1, context.shape[-1])
        chart_centers = project_to_ball(chart_centers).to(
            device=flat_query.device, dtype=flat_query.dtype
        )
        codebook = project_to_ball(codebook).to(device=flat_query.device, dtype=flat_query.dtype)
        num_charts, codes_per_chart, _ = codebook.shape

        local_query = project_to_ball(
            mobius_add(-chart_centers.unsqueeze(0), flat_query.unsqueeze(1))
        )
        tau = _routing_temperature(
            local_query.reshape(-1, local_query.shape[-1]),
            self.latent_dim,
            tau_min=self.tau_min,
            tau_denom_min=self.tau_denom_min,
        ).reshape(flat_query.shape[0], num_charts)

        base_logits = -hyperbolic_distance(local_query.unsqueeze(2), codebook.unsqueeze(0))
        base_logits = base_logits / tau.unsqueeze(-1)

        context_features = self.context_proj(flat_context)
        code_keys = self.code_key_proj(
            log_map_zero(codebook.reshape(num_charts * codes_per_chart, -1))
        )
        code_keys = code_keys.reshape(num_charts, codes_per_chart, self.context_dim)
        feature_logits = torch.einsum("bh,nkh->bnk", context_features, code_keys)
        logits = base_logits + self.feature_scale * feature_logits
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        return {
            "code_logits": logits.reshape(*leading_shape, num_charts, codes_per_chart),
            "code_log_probs": log_probs.reshape(*leading_shape, num_charts, codes_per_chart),
            "code_probs": probs.reshape(*leading_shape, num_charts, codes_per_chart),
            "local_query": local_query.reshape(*leading_shape, num_charts, self.latent_dim),
            "code_tau": tau.reshape(*leading_shape, num_charts),
        }


__all__ = [
    "BeliefGeometryEncoder",
    "ChartTransitionRouter",
    "ConditionalCodeRouter",
    "NextStateQueryPredictor",
]
