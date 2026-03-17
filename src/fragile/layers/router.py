import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import SpectralLinear


def routing_weights(scores: torch.Tensor, routing_tau: float) -> torch.Tensor:
    if routing_tau < 0:
        # Negative tau → deterministic straight-through argmax (no Gumbel noise).
        # Forward: one-hot from argmax.  Backward: gradients through softmax.
        # This lets routing losses see the router's true preference, unlike
        # Gumbel-softmax which masks collapse with random noise.
        soft = F.softmax(scores, dim=-1)
        one_hot = F.one_hot(scores.argmax(-1), scores.shape[-1]).float()
        return one_hot + soft - soft.detach()
    tau = max(float(routing_tau), 1e-6)
    return F.gumbel_softmax(scores, tau=tau, hard=True)


class CovariantChartRouter(nn.Module):
    """Gauge-covariant chart router with hyperbolic transport and metric-aware temperature.

    Uses O(n) Poincaré ball parallel transport instead of O(n³) Cayley transform.
    """

    def __init__(
        self,
        latent_dim: int,
        key_dim: int,
        num_charts: int,
        feature_dim: int | None = None,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
        transport_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.num_charts = num_charts
        self.tau_min = tau_min
        self.tau_denom_min = tau_denom_min
        self.transport_eps = transport_eps

        if feature_dim is not None:
            self.q_feat_proj = SpectralLinear(feature_dim, key_dim, bias=True)
        else:
            self.q_feat_proj = None
        self.q_z_proj = SpectralLinear(latent_dim, key_dim, bias=True)

        self.q_gamma = nn.Parameter(torch.randn(key_dim, latent_dim, latent_dim) * 0.02)
        self.q_gamma_out = None
        self.q_gamma_u = None
        self.q_gamma_v = None

        self.chart_queries = nn.Parameter(torch.randn(num_charts, key_dim) * 0.02)
        self.chart_key_proj = SpectralLinear(latent_dim, key_dim, bias=False)

        self._last_soft_router_weights = None
        self._last_soft_router_weights_live = None
        self._last_router_scores = None
        self._last_router_scores_live = None

    def _gamma_term(self, z: torch.Tensor) -> torch.Tensor:
        # Quadratic term captures Christoffel-symbol curvature corrections.
        z_outer = z.unsqueeze(2) * z.unsqueeze(1)  # [B, D, D]
        return torch.einsum("bij,kij->bk", z_outer, self.q_gamma)

    def _conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute Poincaré ball conformal factor λ(z) = 2 / (1 - |z|²)."""
        r2 = (z**2).sum(dim=-1, keepdim=True)
        r2 = torch.clamp(r2, max=1.0 - self.transport_eps)
        return 2.0 / (1.0 - r2 + self.transport_eps)

    def _transport_queries(
        self, z: torch.Tensor, chart_tokens: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transport chart queries using O(n) hyperbolic parallel transport.

        In the Poincaré ball, parallel transport from the origin scales vectors
        by the conformal factor ratio. This replaces the O(n³) Cayley transform.
        """
        batch_size = z.shape[0]
        if chart_tokens is None:
            base_queries = self.chart_queries
        else:
            if chart_tokens.ndim != 2 or chart_tokens.shape[0] != self.num_charts:
                msg = "chart_tokens must have shape [N_c, D] or [N_c, K]."
                raise ValueError(msg)
            if chart_tokens.shape[1] == self.key_dim:
                base_queries = chart_tokens
            elif chart_tokens.shape[1] == self.latent_dim:
                base_queries = self.chart_key_proj(chart_tokens)
            else:
                msg = "chart_tokens must have shape [N_c, D] or [N_c, K]."
                raise ValueError(msg)

        # O(n) hyperbolic parallel transport using conformal factors
        # Transport from origin (where chart_queries live) to z
        # P_{0→z}(v) = v / λ(z) (scales by inverse conformal factor)
        lambda_z = self._conformal_factor(z)  # [B, 1]

        # Expand base_queries: [N_c, K] -> [B, N_c, K]
        queries_expanded = base_queries.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply transport scaling: divide by conformal factor at destination
        # This preserves the hyperbolic norm of the queries
        return queries_expanded / lambda_z.unsqueeze(1)

    def _temperature(self, z: torch.Tensor) -> torch.Tensor:
        # Router energies are hyperbolic distances in the latent manifold, so
        # their Gibbs temperature should scale with the latent geometry, not
        # with the hidden/key projection width used for auxiliary feature terms.
        r2 = (z**2).sum(dim=-1)
        denom = (1.0 - r2).clamp(min=self.tau_denom_min)
        tau = math.sqrt(self.latent_dim) * denom / 2.0
        return tau.clamp(min=self.tau_min)

    def _hyperbolic_score(self, z: torch.Tensor, chart_centers: torch.Tensor) -> torch.Tensor:
        """Compute logits based on negative hyperbolic distance. O(N*D).

        Uses the Poincaré ball distance formula for efficient chart scoring
        without requiring matrix operations.

        Args:
            z: [B, D] latent positions
            chart_centers: [N_c, D] chart center positions

        Returns:
            scores: [B, N_c] negative distances (higher = closer)
        """
        # z: [B, D], chart_centers: [N_c, D]
        z_exp = z.unsqueeze(1)  # [B, 1, D]
        c_exp = chart_centers.unsqueeze(0)  # [1, N_c, D]

        # Squared Euclidean norm of difference
        diff = z_exp - c_exp
        dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]

        # Boundary terms (1 - |z|²) and (1 - |c|²)
        z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
        c_sq = (chart_centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
        denom = (1 - z_sq) * (1 - c_sq)  # [B, N_c]

        # Poincaré distance formula: d(z, c) = acosh(1 + 2 * |z-c|² / ((1-|z|²)(1-|c|²)))
        arg = 1 + 2 * dist_sq / (denom + self.transport_eps)
        dist = torch.acosh(arg.clamp(min=1.0 + self.transport_eps))  # [B, N_c]

        # Temperature scaling
        tau = self._temperature(z)  # [B]
        return -dist / tau.unsqueeze(1)

    def forward(
        self,
        z: torch.Tensor,
        features: torch.Tensor | None = None,
        chart_tokens: torch.Tensor | None = None,
        routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Route to charts using hyperbolic distance scoring.

        Args:
            z: [B, D] latent positions
            features: [B, F] optional feature vectors
            chart_tokens: [N_c, D] optional chart centers (defaults to self.chart_centers)

        Returns:
            router_weights: [B, N_c] routing weights
            K_chart: [B] argmax chart assignments
        """
        # Get chart centers for scoring
        if chart_tokens is not None:
            if chart_tokens.ndim != 2 or chart_tokens.shape[0] != self.num_charts:
                msg = "chart_tokens must have shape [N_c, D]."
                raise ValueError(msg)
            # Project to latent dim if needed
            if chart_tokens.shape[1] != self.latent_dim:
                # Use key_proj if chart_tokens are in key space
                centers = chart_tokens
            else:
                centers = chart_tokens
        else:
            # Use learned chart queries projected to latent space
            # Note: chart_queries are in key_dim, we need latent_dim for distance
            # Fall back to using q_z_proj inverse or just use chart_queries directly
            centers = self.chart_queries[:, : self.latent_dim]  # Truncate to latent_dim

        # O(n) hyperbolic distance-based scoring
        scores = self._hyperbolic_score(z, centers)

        # Optional: add feature-based corrections via gamma term
        if self.q_feat_proj is not None and features is not None:
            q = self.q_z_proj(z) + self.q_feat_proj(features) + self._gamma_term(z)
            # Add small correction from feature projection
            keys = self._transport_queries(z, chart_tokens=chart_tokens)
            feature_scores = (keys * q.unsqueeze(1)).sum(dim=-1)
            tau = self._temperature(z)
            scores = scores + 0.1 * feature_scores / tau.unsqueeze(1)

        # Cache both detached and live soft weights. The detached copy is safe for
        # diagnostics, while the live copy lets training losses act on the router's
        # actual confidence even when the forward pass uses hard routing.
        soft_router_weights = F.softmax(scores, dim=-1)
        self._last_soft_router_weights = soft_router_weights.detach()
        self._last_soft_router_weights_live = soft_router_weights
        self._last_router_scores = scores.detach()
        self._last_router_scores_live = scores
        router_weights = routing_weights(scores, routing_tau)
        k_chart = torch.argmax(router_weights, dim=1)
        return router_weights, k_chart
