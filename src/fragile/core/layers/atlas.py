from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

from .gauge import exp_map_zero, hyperbolic_distance, log_map_zero, mobius_add
from .primitives import IsotropicBlock, NormGatedGELU, SpectralLinear
from .topology import FactorizedJumpOperator, InvariantChartClassifier
from .ugn import SoftEquivariantLayer


# ---------------------------------------------------------------------------
# Quasi-uniform initialization helpers
# ---------------------------------------------------------------------------


def _fibonacci_sphere(n: int) -> torch.Tensor:
    """Generate *n* quasi-uniformly spaced points on S² (Fibonacci lattice).

    Returns a [n, 3] tensor on the unit sphere. For D != 3, falls back to
    ``_spread_directions``.
    """
    golden = (1 + math.sqrt(5)) / 2
    indices = torch.arange(n, dtype=torch.float32)
    theta = 2 * math.pi * indices / golden          # azimuth
    phi = torch.acos(1 - 2 * (indices + 0.5) / n)   # polar
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack([x, y, z], dim=-1)


def _spread_directions(n: int, dim: int) -> torch.Tensor:
    """Generate *n* spread-out unit directions in R^dim.

    Uses the Fibonacci sphere for dim == 3. Otherwise generates random
    directions and iteratively repels them (simple Lloyd-like relaxation).
    """
    if dim == 3:
        return _fibonacci_sphere(n)

    # Random init + greedy repulsion (5 iterations suffice for init quality)
    pts = torch.randn(n, dim)
    pts = torch.nn.functional.normalize(pts, dim=-1)
    for _ in range(20):
        # Compute pairwise cosine similarity
        sim = pts @ pts.t()                  # [n, n]
        sim.fill_diagonal_(-1e9)             # ignore self
        # Push each point away from its nearest neighbor
        nearest = sim.argmax(dim=1)          # [n]
        neighbors = pts[nearest]             # [n, dim]
        pts = pts - 0.3 * neighbors          # repel
        pts = torch.nn.functional.normalize(pts, dim=-1)
    return pts


def _spread_codebook(num_charts: int, codes_per_chart: int, dim: int,
                     radius: float = 0.3) -> torch.Tensor:
    """Initialize codebook entries spread around the local origin.

    Each chart gets ``codes_per_chart`` codes arranged as quasi-uniform
    directions scaled to ``radius`` in the Poincaré ball.  This avoids the
    usual failure mode where all codes start near zero and instantly collapse
    to a single nearest-neighbor.

    Returns [num_charts, codes_per_chart, dim].
    """
    cb = torch.zeros(num_charts, codes_per_chart, dim)
    for c in range(num_charts):
        dirs = _spread_directions(codes_per_chart, dim)
        # Uniform radii in [radius/2, radius] so codes aren't on a thin shell
        r = torch.rand(codes_per_chart, 1) * (radius / 2) + (radius / 2)
        cb[c] = dirs * r
    return cb


def _poincare_temperature(
    z: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
) -> torch.Tensor:
    """Compute position-dependent temperature for Poincare ball."""
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=tau_denom_min)
    tau = math.sqrt(key_dim) * denom / 2.0
    return tau.clamp(min=tau_min)


def _poincare_hyperbolic_score(
    z: torch.Tensor,
    centers: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
    eps: float,
) -> torch.Tensor:
    """Compute hyperbolic distance-based scores with metric temperature."""
    z_exp = z.unsqueeze(1)  # [B, 1, D]
    c_exp = centers.unsqueeze(0)  # [1, N_c, D]
    diff = z_exp - c_exp
    dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]
    z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
    c_sq = (centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
    denom = (1.0 - z_sq) * (1.0 - c_sq)
    arg = 1.0 + 2.0 * dist_sq / (denom + eps)
    dist = torch.acosh(arg.clamp(min=1.0 + eps))  # [B, N_c]
    tau = _poincare_temperature(z, key_dim, tau_min, tau_denom_min)
    return -dist / tau.unsqueeze(1)


def _project_to_ball(
    z: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project points to interior of the Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def _smooth_tangent_to_ball(
    v: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map unconstrained tangent vectors smoothly into the Poincare ball."""
    tangent_cap = math.atanh(max_norm)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    tangent_norm = tangent_cap * torch.tanh(v_norm / tangent_cap)
    tangent = tangent_norm * (v / v_norm)
    return exp_map_zero(tangent, eps=eps)


def _poincare_weighted_mean(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Approximate hyperbolic barycenter using log/exp maps at the origin."""
    if points.dim() == 2:
        points = points.unsqueeze(0).expand(weights.shape[0], -1, -1)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)
    tangent = log_map_zero(points)
    mean_tan = (w * tangent).sum(dim=1) / w_sum.squeeze(1)
    return exp_map_zero(mean_tan)


def _poincare_weighted_mean_per_chart(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-chart hyperbolic barycenter for codebook soft assignment."""
    points_exp = points.unsqueeze(0).expand(weights.shape[0], -1, -1, -1)
    tangent = log_map_zero(points_exp)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=2, keepdim=True).clamp(min=eps)
    mean_tan = (w * tangent).sum(dim=2) / w_sum.squeeze(2)
    return exp_map_zero(mean_tan)


class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out = diff_out * scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        x: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through the attentive atlas.

        Returns:
            K_chart: [B] chart assignment
            K_code: [B] code index within chart
            z_n: [B, D] nuisance latent
            z_tex: [B, D] texture residual
            router_weights: [B, N_c] routing weights
            z_geo: [B, D] geometric latent
            vq_loss: [] VQ loss
            indices_stack: [B, N_c] code indices per chart
            z_n_all_charts: [B, N_c, D] per-chart nuisance
            c_bar: [B, D] chart center mixture
            v_local: [B, D] local residual (for code collapse penalty)
        """
        # Extract features via fully-connected front-end.
        features = self._encode_features(x)  # [B, H]

        # Map features into chart-space coordinates.
        v = self.val_proj(features)  # [B, D]
        scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
        # Chart routing distributes mass across atlas charts.
        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # Chart-center mixture defines the macro coordinate (hyperbolic barycenter).
        c_bar = _poincare_weighted_mean(self.chart_centers, router_weights)  # [B, D]
        v_local = _project_to_ball(mobius_add(-c_bar, v))  # [B, D]

        # Per-chart codebook lookup using hyperbolic geometry.
        codebook = _project_to_ball(self.codebook)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)
        dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = _poincare_weighted_mean_per_chart(codebook, weights)
            # Straight-through soft assignment so gradients reach the metric network.
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ loss using tangent-space distances in the Poincaré ball.
        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * w).mean(dim=(0, 2)).sum()  # []
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss = (delta_codebook**2 * w).mean(dim=(0, 2)).sum()  # []
        vq_loss = self._codebook_loss_weight * codebook_loss + self._commitment_beta * commitment

        # Blend chart codes to form macro latent (hyperbolic barycenter).
        z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        # Structure filter extracts nuisance in tangent space.
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = _project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        # Texture residual is what's left after nuisance subtraction.
        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_geo = _project_to_ball(mobius_add(c_bar, mobius_add(z_q_st, exp_map_zero(z_n_tan))))  # [B, D]

        return (
            K_chart,
            K_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
            z_q_blended,
        )


class TopologicalDecoder(nn.Module):
    """Inverse atlas decoder with optional autonomous routing."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        weight = torch.empty(num_charts, hidden_dim, latent_dim)
        nn.init.uniform_(weight, -1.0 / math.sqrt(latent_dim), 1.0 / math.sqrt(latent_dim))
        self.chart_weight = nn.Parameter(weight)
        self.chart_bias = nn.Parameter(torch.zeros(num_charts, hidden_dim))

        self.latent_router = nn.Linear(latent_dim, num_charts)
        self.tex_residual = nn.Linear(latent_dim, output_dim)
        self.tex_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.renderer = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.render_skip = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor | None = None,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode from latent geometry.

        Args:
            z_geo: [B, D] geometric latent
            z_tex: [B, D] optional texture residual (added to output)
            chart_index: [B] optional chart ids

        Returns:
            x_hat: [B, D_out] reconstruction
            router_weights: [B, N_c] routing weights
        """
        # Clamp geometry inside atlas chart range.
        z_geo = torch.tanh(z_geo)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        else:
            # Autonomous routing predicts chart membership from geometry.
            logits = self.latent_router(z_geo)  # [B, N_c]
            router_weights = _routing_weights(logits, hard_routing, hard_routing_tau)  # [B, N_c]

        # Chart-specific linear maps reconstruct local observations.
        h_stack = torch.einsum("bl,chl->bch", z_geo, self.chart_weight) + self.chart_bias.unsqueeze(0)  # [B, N_c, H]
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        x_hat = self.renderer(h_global) + self.render_skip(h_global)  # [B, D_out]
        # z_tex residual injection disabled: at inference the WM produces
        # z_geo with no z_tex available, so using it here creates a
        # train/test mismatch.  Layers kept for checkpoint compatibility.
        return x_hat, router_weights


class TopoEncoder(nn.Module):
    """Attentive Atlas encoder + topological decoder."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
        )
        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Full forward pass.

        Args:
            x: [B, D_in] input tensor
            use_hard_routing: whether to use hard routing in decoder

        Returns:
            x_recon: [B, D_in] reconstruction
            vq_loss: [] VQ loss
            enc_router_weights: [B, N_c] encoder routing
            dec_router_weights: [B, N_c] decoder routing
            K_chart: [B] chart assignments
            z_geo: [B, D] geometric latent (macro + gauge residual)
            z_n: [B, D] nuisance latent (continuous gauge vector)
            c_bar: [B, D] chart center mixture
        """
        (
            K_chart,
            _K_code,
            z_n,
            z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
            _z_q_blended,
        ) = self.encoder(
            x,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        router_override = enc_router_weights if use_hard_routing else None
        x_recon, dec_router_weights = self.decoder(
            z_geo,
            z_tex,
            chart_index=None,
            router_weights=router_override,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart, z_geo, z_n, c_bar

    def compute_consistency_loss(
        self, enc_weights: torch.Tensor, dec_weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """KL divergence between encoder and decoder routing.

        Args:
            enc_weights: [B, N_c] encoder routing weights
            dec_weights: [B, N_c] decoder routing weights

        Returns:
            loss: [] consistency loss
        """
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(
            dim=-1
        )  # [B]
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        """Chart usage perplexity.

        Args:
            K_chart: [B] chart assignments

        Returns:
            perplexity: scalar perplexity
        """
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()  # [N_c]
        probs = counts / counts.sum()  # [N_c]
        probs = probs[probs > 0]  # [N_c_nonzero]
        entropy = -(probs * torch.log(probs)).sum()  # []
        return math.exp(entropy.item())


def _resolve_bundle_params(
    hidden_dim: int,
    latent_dim: int,
    bundle_size: int | None,
) -> tuple[int, int]:
    if bundle_size is None:
        if latent_dim > 0 and hidden_dim % latent_dim == 0:
            bundle_size = latent_dim
        else:
            bundle_size = 1
    if bundle_size <= 0:
        msg = "bundle_size must be positive."
        raise ValueError(msg)
    if hidden_dim % bundle_size != 0:
        msg = "hidden_dim must be divisible by bundle_size."
        raise ValueError(msg)
    return bundle_size, hidden_dim // bundle_size


def _init_soft_equiv_layers(layers: nn.ModuleList) -> None:
    """Initialize soft-equivariant layers to be purely equivariant (no mixing)."""
    with torch.no_grad():
        for layer in layers:
            if isinstance(layer.mixing_weights, torch.Tensor):
                layer.mixing_weights.zero_()
            else:
                for row in layer.mixing_weights:
                    for weight in row:
                        weight.zero_()


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
        tensorization: str = "full",
        rank: int = 8,
        tau_min: float = 1e-2,
        tau_denom_min: float = 1e-3,
        use_transport: bool = True,
        transport_eps: float = 1e-3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.key_dim = key_dim
        self.num_charts = num_charts
        self.tensorization = tensorization
        self.tau_min = tau_min
        self.tau_denom_min = tau_denom_min
        self.use_transport = use_transport
        self.transport_eps = transport_eps

        if feature_dim is not None:
            self.q_feat_proj = SpectralLinear(feature_dim, key_dim, bias=True)
        else:
            self.q_feat_proj = None
        self.q_z_proj = SpectralLinear(latent_dim, key_dim, bias=True)

        if tensorization == "full":
            self.q_gamma = nn.Parameter(torch.randn(key_dim, latent_dim, latent_dim) * 0.02)
            self.q_gamma_out = None
            self.q_gamma_u = None
            self.q_gamma_v = None
        elif tensorization == "sum":
            self.q_gamma_out = nn.Parameter(torch.randn(rank, key_dim) * 0.02)
            self.q_gamma_u = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma_v = nn.Parameter(torch.randn(rank, latent_dim) * 0.02)
            self.q_gamma = None
        else:
            msg = "tensorization must be 'full' or 'sum'."
            raise ValueError(msg)

        self.chart_queries = nn.Parameter(torch.randn(num_charts, key_dim) * 0.02)
        self.chart_key_proj = SpectralLinear(latent_dim, key_dim, bias=False)

        # Note: transport_proj removed - using O(n) hyperbolic transport instead

    def _gamma_term(self, z: torch.Tensor) -> torch.Tensor:
        if self.tensorization == "full":
            # Quadratic term captures Christoffel-symbol curvature corrections.
            z_outer = z.unsqueeze(2) * z.unsqueeze(1)  # [B, D, D]
            return torch.einsum("bij,kij->bk", z_outer, self.q_gamma)
        # Low-rank quadratic term for efficiency.
        z_u = z @ self.q_gamma_u.t()  # [B, R]
        z_v = z @ self.q_gamma_v.t()  # [B, R]
        return (z_u * z_v) @ self.q_gamma_out  # [B, K]

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

        if not self.use_transport:
            return base_queries.unsqueeze(0).expand(batch_size, -1, -1)

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
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
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
        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)
        K_chart = torch.argmax(router_weights, dim=1)
        return router_weights, K_chart


class PrimitiveAttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder using gauge-covariant primitives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        dyn_codes_per_chart: int = 0,
        dyn_commitment_beta: float = 0.25,
        dyn_codebook_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        # Dynamics codebook (disabled when dyn_codes_per_chart == 0)
        self.dyn_codes_per_chart = dyn_codes_per_chart
        self._dyn_commitment_beta = dyn_commitment_beta
        self._dyn_codebook_loss_weight = dyn_codebook_loss_weight
        if dyn_codes_per_chart > 0:
            self.codebook_dyn = nn.Parameter(
                _spread_codebook(num_charts, dyn_codes_per_chart, latent_dim, radius=0.3)
            )
        else:
            self.codebook_dyn = None

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        if conv_backbone:
            from .vision import ConvFeatureExtractor

            self.feature_extractor = ConvFeatureExtractor(
                img_channels, hidden_dim, img_size, conv_channels,
            )
        else:
            self.feature_extractor = nn.Sequential(
                SpectralLinear(input_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
                SpectralLinear(hidden_dim, hidden_dim, bias=True),
                NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            )
        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=hidden_dim,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.key_proj = None
            self.chart_queries = None
            self.scale = None
        else:
            self.key_proj = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
            self.scale = math.sqrt(hidden_dim)

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        self.val_proj_scale = nn.Parameter(torch.tensor(2.0))  # learnable pre-squash scale

        # Quasi-uniform chart centers: Fibonacci sphere (3-D) or repulsion
        # init so every chart starts with a distinct, well-separated catchment
        # region — prevents softmax winner-take-all collapse at epoch 0.
        self.chart_centers = nn.Parameter(
            _spread_directions(num_charts, latent_dim) * 0.5
        )

        # Spread codebook codes around the local origin of each chart so that
        # VQ does not instantly collapse to a single nearest-neighbor.
        self.codebook = nn.Parameter(
            _spread_codebook(num_charts, codes_per_chart, latent_dim, radius=0.3)
        )

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )
        self._last_v_raw: torch.Tensor | None = None
        self._last_v_projected: torch.Tensor | None = None
        self._last_v_local_raw: torch.Tensor | None = None
        self._last_z_geo_raw: torch.Tensor | None = None
        self._last_c_bar: torch.Tensor | None = None
        self._last_v_local: torch.Tensor | None = None
        self._last_indices_stack: torch.Tensor | None = None

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out = diff_out * scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def _hyperbolic_vq(
        self,
        v_local: torch.Tensor,
        codebook_param: torch.Tensor,
        router_weights: torch.Tensor,
        commitment_beta: float,
        codebook_loss_weight: float,
        use_soft_equiv: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared VQ against any codebook.

        Returns:
            z_q_blended: [B, D] router-weighted code blend.
            K_code: [B] winning code index for the winning chart.
            indices: [B, N_c] nearest code per chart.
            vq_loss: scalar VQ loss.
            z_q_all: [B, N_c, D] nearest code per chart (full tensor).
        """
        codebook = _project_to_ball(codebook_param)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)

        if use_soft_equiv:
            dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        else:
            dist = (diff_tan ** 2).sum(dim=-1)  # [B, N_c, K]

        indices = torch.argmin(dist, dim=-1)  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v_local.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]

        if use_soft_equiv and self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = _poincare_weighted_mean_per_chart(codebook, weights)
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing.
        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit ** 2 * w).mean(dim=(0, 2)).sum()
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss_val = (delta_codebook ** 2 * w).mean(dim=(0, 2)).sum()
        vq_loss = codebook_loss_weight * codebook_loss_val + commitment_beta * commitment

        K_chart = torch.argmax(router_weights, dim=1)  # [B]
        z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        K_code = indices.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        return z_q_blended, K_code, indices, vq_loss, z_q_all

    def dynamics_vq(
        self,
        v_local: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """VQ v_local against the dynamics codebook.

        Returns:
            z_q_dyn_blended: [B, D] router-weighted dynamics code blend.
            K_code_dyn: [B] winning dynamics code index.
            indices_dyn: [B, N_c] nearest dynamics code per chart.
            vq_loss_dyn: scalar VQ loss for dynamics codebook.
        """
        assert self.codebook_dyn is not None, "dynamics codebook not initialized (dyn_codes_per_chart=0)"
        return self._hyperbolic_vq(
            v_local, self.codebook_dyn, router_weights,
            self._dyn_commitment_beta, self._dyn_codebook_loss_weight,
            use_soft_equiv=False,
        )[:4]

    @torch.no_grad()
    def warmstart_chart_centers(
        self,
        dataloader,
        device,
        max_batches: int = 10,
        radius_floor: float = 0.0,
    ):
        """Replace chart_centers with k-means centroids of initial v distribution."""
        vs = []
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            if isinstance(batch, dict):
                x = batch.get("feature", batch.get("features")).to(device)
            else:
                x = batch[0].to(device)
            if x.ndim == 3:  # sequence [B, T, D] → flatten
                x = x.reshape(-1, x.shape[-1])
            features = self._encode_features(x)
            v = _smooth_tangent_to_ball(self.val_proj(features) * self.val_proj_scale)
            vs.append(v.cpu())
        vs = torch.cat(vs, dim=0)

        from sklearn.cluster import KMeans

        km = KMeans(n_clusters=self.num_charts, n_init=10, random_state=42)
        km.fit(vs.numpy())
        centroids = torch.from_numpy(km.cluster_centers_).float()
        centroids = _project_to_ball(centroids)
        if radius_floor > 0:
            min_radius = min(float(radius_floor), 0.95)
            norms = centroids.norm(dim=-1, keepdim=True)
            fallback_dirs = _spread_directions(self.num_charts, self.latent_dim)
            dirs = torch.where(
                norms > 1e-6,
                centroids / norms.clamp_min(1e-6),
                fallback_dirs,
            )
            target_norms = norms.clamp(min=min_radius)
            centroids = _project_to_ball(dirs * target_norms)
        self.chart_centers.copy_(centroids)

        counts = torch.bincount(
            torch.from_numpy(km.labels_), minlength=self.num_charts
        )
        print(f"  K-means warm-start: {len(vs)} points → {self.num_charts} charts")
        print(f"  Cluster sizes: {counts.tolist()}")
        if radius_floor > 0:
            print(f"  Applied chart-center radius floor: {min(float(radius_floor), 0.95):.3f}")

    def forward(
        self,
        x: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through the attentive atlas."""
        # Extract features and map into chart coordinates (Poincare ball).
        features = self._encode_features(x)  # [B, H]
        v_raw = self.val_proj(features) * self.val_proj_scale
        v = _smooth_tangent_to_ball(v_raw)  # [B, D]
        chart_centers = _project_to_ball(self.chart_centers)  # [N_c, D]

        if self.covariant_attn:
            router_weights, K_chart = self.cov_router(
                v,
                features=features,
                chart_tokens=chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
            self._last_soft_router_weights = self.cov_router._last_soft_router_weights
            self._last_soft_router_weights_live = self.cov_router._last_soft_router_weights_live
            self._last_router_scores = self.cov_router._last_router_scores
            self._last_router_scores_live = self.cov_router._last_router_scores_live
        else:
            scores = _poincare_hyperbolic_score(
                v,
                chart_centers,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
                eps=self.router_transport_eps,
            )
            soft_router_weights = F.softmax(scores, dim=-1)
            self._last_soft_router_weights = soft_router_weights.detach()
            self._last_soft_router_weights_live = soft_router_weights
            self._last_router_scores = scores.detach()
            self._last_router_scores_live = scores
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]
            K_chart = torch.argmax(router_weights, dim=1)  # [B]

        c_bar = _poincare_weighted_mean(chart_centers, router_weights)  # [B, D]
        v_local_raw = mobius_add(-c_bar, v)
        v_local = _project_to_ball(v_local_raw)  # [B, D]

        # Per-chart codebook lookup via shared VQ helper.
        z_q_blended, K_code, indices_stack, vq_loss, z_q_all = self._hyperbolic_vq(
            v_local, self.codebook, router_weights,
            self._commitment_beta, self._codebook_loss_weight,
            use_soft_equiv=True,
        )

        # Structure filter extracts nuisance; remainder is texture.
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = _project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_local = mobius_add(z_q_st, exp_map_zero(z_n_tan))
        z_geo_raw = mobius_add(c_bar, z_local)
        z_geo = _project_to_ball(z_geo_raw)  # [B, D]

        # Cache the chart-local latent so auxiliary losses can read the exact
        # codebook input from this forward pass instead of reconstructing it.
        self._last_v_raw = v_raw
        self._last_v_projected = v
        self._last_v_local_raw = v_local_raw
        self._last_z_geo_raw = z_geo_raw
        self._last_c_bar = c_bar
        self._last_v_local = v_local
        self._last_indices_stack = indices_stack

        return (
            K_chart,
            K_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
            z_q_blended,
        )


class _ChartFiLM1d(nn.Module):
    """Per-chart FiLM conditioning for 1-D feature vectors [B, H]."""

    def __init__(self, num_charts: int, dim: int) -> None:
        super().__init__()
        self.gammas = nn.Parameter(torch.zeros(num_charts, dim))
        self.betas = nn.Parameter(torch.zeros(num_charts, dim))

    def forward(self, h: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
        gamma = router_weights @ self.gammas  # [B, H]
        beta = router_weights @ self.betas
        return h * (1.0 + gamma) + beta


class PrimitiveTopologicalDecoder(nn.Module):
    """Topological decoder using gauge-covariant primitives."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
        film_conditioning: bool = False,
        conformal_freq_gating: bool = False,
        texture_flow: bool = False,
        texture_flow_layers: int = 4,
        texture_flow_hidden: int = 64,
        texture_flow_clamp: float = 5.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.covariant_attn = covariant_attn
        self.output_dim = output_dim
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps

        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projectors = nn.ModuleList([
            SpectralLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_charts)
        ])
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=None,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.latent_router = None
        else:
            self.latent_router = SpectralLinear(latent_dim, num_charts, bias=True)
        self.tex_residual_scale = nn.Parameter(torch.tensor(0.1))

        # Conformal frequency gating (conv mode only)
        self.conformal_freq_gating = conformal_freq_gating and conv_backbone
        self._img_channels = img_channels
        self._img_size = img_size

        if conv_backbone:
            from .vision import ConvImageDecoder

            film_num_charts = num_charts if film_conditioning else 0
            self.renderer = ConvImageDecoder(
                hidden_dim, img_channels, img_size, conv_channels,
                film_num_charts=film_num_charts,
            )
            self.render_skip = None
            # Texture residual maps to hidden_dim (added before conv decoder)
            self.tex_residual = SpectralLinear(latent_dim, hidden_dim, bias=True)
        else:
            self.render_fc1 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.render_act1 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
            self.render_fc2 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.render_act2 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
            self.render_out = SpectralLinear(hidden_dim, output_dim, bias=True)
            self.renderer = None  # signals FC-explicit path in forward
            self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)
            self.tex_residual = SpectralLinear(latent_dim, output_dim, bias=True)
            if film_conditioning:
                self.film1 = _ChartFiLM1d(num_charts, hidden_dim)
                self.film2 = _ChartFiLM1d(num_charts, hidden_dim)
            else:
                self.film1 = None
                self.film2 = None

        # Conditional texture flow
        if texture_flow:
            from .vision import ConditionalTextureFlow

            self.texture_flow: ConditionalTextureFlow | None = ConditionalTextureFlow(
                tex_dim=latent_dim,
                geo_dim=latent_dim,
                hidden_dim=texture_flow_hidden,
                n_layers=texture_flow_layers,
                clamp=texture_flow_clamp,
            )
        else:
            self.texture_flow = None

    def forward(
        self,
        z_geo: torch.Tensor,
        z_tex: torch.Tensor | None = None,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Decode from latent geometry.

        Returns:
            x_hat: [B, D_out] reconstruction
            router_weights: [B, N_c] routing weights
            aux_losses: dict of auxiliary losses (e.g. flow_loss)
        """
        aux_losses: dict[str, torch.Tensor] = {}

        # Clamp geometry to chart range (Poincare ball).
        z_geo = _project_to_ball(z_geo)
        chart_centers = _project_to_ball(self.chart_centers)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        elif self.covariant_attn:
            # Covariant router predicts chart membership from geometry.
            router_weights, _ = self.cov_router(
                z_geo,
                chart_tokens=chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        else:
            scores = _poincare_hyperbolic_score(
                z_geo,
                chart_centers,
                key_dim=self.latent_dim,
                tau_min=self.router_tau_min,
                tau_denom_min=self.router_tau_denom_min,
                eps=self.router_transport_eps,
            )
            if self.latent_router is not None:
                tau = _poincare_temperature(
                    z_geo,
                    key_dim=self.latent_dim,
                    tau_min=self.router_tau_min,
                    tau_denom_min=self.router_tau_denom_min,
                )
                scores = scores + 0.1 * self.latent_router(z_geo) / tau.unsqueeze(1)
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]

        # Chart-specific projections + gauge-covariant gating.
        h_stack = torch.stack(
            [proj(z_geo) for proj in self.chart_projectors], dim=1
        )  # [B, N_c, H]
        h_stack = self.chart_gate(h_stack.view(-1, self.hidden_dim)).view(
            z_geo.shape[0], self.num_charts, self.hidden_dim
        )
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        if self.renderer is not None:
            # Conv mode
            h = h_global
            # z_tex residual injection disabled: at inference the WM produces
            # z_geo with no z_tex available, creating a train/test mismatch.
            # Layers kept for checkpoint compatibility.
            need_spatial = self.conformal_freq_gating
            x_hat = self.renderer(
                h,
                router_weights=router_weights,
                return_spatial=need_spatial,
            )
            if self.conformal_freq_gating:
                from .vision import conformal_frequency_gate

                x_hat = conformal_frequency_gate(x_hat, z_geo, self.latent_dim)
                x_hat = x_hat.reshape(x_hat.shape[0], -1)
        else:
            # FC mode: layer-by-layer with optional FiLM conditioning
            h = self.render_fc1(h_global)
            if self.film1 is not None:
                h = self.film1(h, router_weights)
            h = self.render_act1(h)
            h = self.render_fc2(h)
            if self.film2 is not None:
                h = self.film2(h, router_weights)
            h = self.render_act2(h)
            x_hat = self.render_out(h) + self.render_skip(h_global)
            # z_tex residual injection disabled (see conv mode comment above).

        if self.texture_flow is not None and z_tex is not None:
            _, log_det = self.texture_flow.forward(z_tex, z_geo)
            aux_losses["flow_loss"] = -log_det.mean()

        return x_hat, router_weights, aux_losses


class TopoEncoderPrimitives(nn.Module):
    """Attentive Atlas encoder + topological decoder built from primitives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        conv_backbone: bool = False,
        img_channels: int = 1,
        img_size: int = 28,
        conv_channels: int = 0,
        film_conditioning: bool = False,
        conformal_freq_gating: bool = False,
        texture_flow: bool = False,
        texture_flow_layers: int = 4,
        texture_flow_hidden: int = 64,
        texture_flow_clamp: float = 5.0,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        dyn_codes_per_chart: int = 0,
        dyn_commitment_beta: float = 0.25,
        dyn_codebook_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.encoder = PrimitiveAttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            bundle_size=bundle_size,
            covariant_attn=covariant_attn,
            covariant_attn_tensorization=covariant_attn_tensorization,
            covariant_attn_rank=covariant_attn_rank,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_use_transport=covariant_attn_use_transport,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
            conv_backbone=conv_backbone,
            img_channels=img_channels,
            img_size=img_size,
            conv_channels=conv_channels,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
            dyn_codes_per_chart=dyn_codes_per_chart,
            dyn_commitment_beta=dyn_commitment_beta,
            dyn_codebook_loss_weight=dyn_codebook_loss_weight,
        )
        self.decoder = PrimitiveTopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
            bundle_size=bundle_size,
            covariant_attn=covariant_attn,
            covariant_attn_tensorization=covariant_attn_tensorization,
            covariant_attn_rank=covariant_attn_rank,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_use_transport=covariant_attn_use_transport,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            conv_backbone=conv_backbone,
            img_channels=img_channels,
            img_size=img_size,
            conv_channels=conv_channels,
            film_conditioning=film_conditioning,
            conformal_freq_gating=conformal_freq_gating,
            texture_flow=texture_flow,
            texture_flow_layers=texture_flow_layers,
            texture_flow_hidden=texture_flow_hidden,
            texture_flow_clamp=texture_flow_clamp,
        )

    def forward(
        self,
        x: torch.Tensor,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        (
            K_chart,
            _K_code,
            z_n,
            z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
            _z_q_blended,
        ) = self.encoder(
            x,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        router_override = enc_router_weights if use_hard_routing else None
        x_recon, dec_router_weights, aux_losses = self.decoder(
            z_geo,
            None,  # z_tex not used by decoder (train/test mismatch)
            chart_index=None,
            router_weights=router_override,
            hard_routing=use_hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        return x_recon, vq_loss, enc_router_weights, dec_router_weights, K_chart, z_geo, z_n, c_bar, aux_losses

    def compute_consistency_loss(
        self, enc_weights: torch.Tensor, dec_weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())

    def warmstart_chart_centers(self, dataloader, device, max_batches=10, radius_floor: float = 0.0):
        """Delegate to inner encoder's warmstart_chart_centers."""
        self.encoder.warmstart_chart_centers(
            dataloader,
            device,
            max_batches=max_batches,
            radius_floor=radius_floor,
        )


def _expand_list(value: int | list[int], n_levels: int, name: str) -> list[int]:
    if isinstance(value, list):
        if len(value) != n_levels:
            raise ValueError(f"{name} must have length {n_levels}.")
        return [int(v) for v in value]
    return [int(value) for _ in range(n_levels)]


def _select_chart_latent(z_by_chart: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
    if z_by_chart.ndim != 3:
        msg = "z_by_chart must have shape [B, N_c, D]."
        raise ValueError(msg)
    idx = chart_idx.view(-1, 1, 1).expand(-1, 1, z_by_chart.shape[-1])
    return z_by_chart.gather(1, idx).squeeze(1)


def _routing_weights(
    scores: torch.Tensor, hard_routing: bool, hard_routing_tau: float
) -> torch.Tensor:
    if not hard_routing:
        return F.softmax(scores, dim=-1)
    if hard_routing_tau < 0:
        # Negative tau → deterministic straight-through argmax (no Gumbel noise).
        # Forward: one-hot from argmax.  Backward: gradients through softmax.
        # This lets routing losses see the router's true preference, unlike
        # Gumbel-softmax which masks collapse with random noise.
        soft = F.softmax(scores, dim=-1)
        one_hot = F.one_hot(scores.argmax(-1), scores.shape[-1]).float()
        return one_hot + soft - soft.detach()
    tau = max(float(hard_routing_tau), 1e-6)
    return F.gumbel_softmax(scores, tau=tau, hard=True)


class _SharedFeatureExtractor(nn.Module):
    """Shared feature extractor for hierarchical atlas stacks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        bundle_size: int | None,
    ) -> None:
        super().__init__()
        bundle_size, n_bundles = _resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)


class _AtlasEncoderLevel(nn.Module):
    """Encoder head that consumes shared features and produces charted latents."""

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int,
        num_charts: int,
        codes_per_chart: int,
        covariant_attn: bool,
        covariant_attn_tensorization: str,
        covariant_attn_rank: int,
        covariant_attn_tau_min: float,
        covariant_attn_denom_min: float,
        covariant_attn_use_transport: bool,
        covariant_attn_transport_eps: float,
        soft_equiv_metric: bool,
        soft_equiv_bundle_size: int | None,
        soft_equiv_hidden_dim: int,
        soft_equiv_use_spectral_norm: bool,
        soft_equiv_zero_self_mixing: bool,
        soft_equiv_soft_assign: bool,
        soft_equiv_temperature: float,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        if covariant_attn:
            self.cov_router = CovariantChartRouter(
                latent_dim=latent_dim,
                key_dim=hidden_dim,
                num_charts=num_charts,
                feature_dim=hidden_dim,
                tensorization=covariant_attn_tensorization,
                rank=covariant_attn_rank,
                tau_min=covariant_attn_tau_min,
                tau_denom_min=covariant_attn_denom_min,
                use_transport=covariant_attn_use_transport,
                transport_eps=covariant_attn_transport_eps,
            )
            self.key_proj = None
            self.chart_queries = None
            self.scale = None
        else:
            self.key_proj = SpectralLinear(hidden_dim, hidden_dim, bias=True)
            self.chart_queries = nn.Parameter(torch.randn(num_charts, hidden_dim) * 0.02)
            self.scale = math.sqrt(hidden_dim)

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )
        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out = diff_out * scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        features: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Map shared features into chart coordinates.
        v = self.val_proj(features)  # [B, D]

        if self.covariant_attn:
            # Covariant router assigns charts with gauge-aware transport.
            router_weights, K_chart = self.cov_router(
                v,
                features=features,
                chart_tokens=self.chart_centers,
                hard_routing=hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
        else:
            scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)
            router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)
            K_chart = torch.argmax(router_weights, dim=1)

        # Chart-center mixture defines the macro coordinate (hyperbolic barycenter).
        c_bar = _poincare_weighted_mean(self.chart_centers, router_weights)
        v_local = _project_to_ball(mobius_add(-c_bar, v))

        # Per-chart codebook match using hyperbolic geometry.
        codebook = _project_to_ball(self.codebook)
        v_exp = v_local.unsqueeze(1).unsqueeze(2)
        codebook_exp = codebook.unsqueeze(0)
        diff = mobius_add(-codebook_exp, v_exp)
        diff_tan = log_map_zero(diff)
        dist = self._apply_soft_equiv_metric(diff_tan)
        indices = torch.argmin(dist, dim=-1)
        indices_stack = indices

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = _poincare_weighted_mean_per_chart(codebook, weights)
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing (tangent-space distances).
        w = router_weights.unsqueeze(-1).detach()
        v_bc = v_local.unsqueeze(1)
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * w).mean(dim=(0, 2)).sum()
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss = (delta_codebook**2 * w).mean(dim=(0, 2)).sum()
        vq_loss = self._codebook_loss_weight * codebook_loss + self._commitment_beta * commitment

        # Blend chart codes to form macro latent (hyperbolic barycenter).
        z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)

        # Structure filter extracts nuisance in tangent space.
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = _project_to_ball(exp_map_zero(z_n_all_charts_tan))

        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))
        z_tex = delta_blended - z_n_tan

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_geo = _project_to_ball(mobius_add(c_bar, mobius_add(z_q_st, exp_map_zero(z_n_tan))))

        return (
            K_chart,
            K_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
            z_q_blended,
        )


class HierarchicalAtlasStack(nn.Module):
    """Multi-scale TopoEncoder stack with optional shared feature extractor."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int | list[int] = 2,
        num_charts: int | list[int] = 3,
        codes_per_chart: int | list[int] = 21,
        n_levels: int = 3,
        level_update_freqs: list[int] | None = None,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tensorization: str = "full",
        covariant_attn_rank: int = 8,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_use_transport: bool = True,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        share_feature_extractor: bool = True,
        enable_cross_level_jump: bool = False,
        jump_global_rank: int | None = None,
    ) -> None:
        super().__init__()
        if n_levels <= 0:
            msg = "n_levels must be positive."
            raise ValueError(msg)

        self.n_levels = n_levels
        self.share_feature_extractor = share_feature_extractor

        level_latent_dims = _expand_list(latent_dim, n_levels, "latent_dim")
        level_num_charts = _expand_list(num_charts, n_levels, "num_charts")
        level_codes = _expand_list(codes_per_chart, n_levels, "codes_per_chart")
        if level_update_freqs is None:
            level_update_freqs = [1 for _ in range(n_levels)]
        if len(level_update_freqs) != n_levels:
            msg = "level_update_freqs must match n_levels."
            raise ValueError(msg)
        self.level_update_freqs = [int(v) for v in level_update_freqs]

        if share_feature_extractor:
            self.feature_extractor = _SharedFeatureExtractor(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                latent_dim=min(level_latent_dims),
                bundle_size=bundle_size,
            )
            self.feature_extractors = None
        else:
            self.feature_extractor = None
            self.feature_extractors = nn.ModuleList([
                _SharedFeatureExtractor(
                    input_dim=input_dim,
                    hidden_dim=hidden_dim,
                    latent_dim=level_latent_dims[idx],
                    bundle_size=bundle_size,
                )
                for idx in range(n_levels)
            ])

        self.encoder_levels = nn.ModuleList()
        self.decoder_levels = nn.ModuleList()
        for idx in range(n_levels):
            self.encoder_levels.append(
                _AtlasEncoderLevel(
                    hidden_dim=hidden_dim,
                    latent_dim=level_latent_dims[idx],
                    num_charts=level_num_charts[idx],
                    codes_per_chart=level_codes[idx],
                    covariant_attn=covariant_attn,
                    covariant_attn_tensorization=covariant_attn_tensorization,
                    covariant_attn_rank=covariant_attn_rank,
                    covariant_attn_tau_min=covariant_attn_tau_min,
                    covariant_attn_denom_min=covariant_attn_denom_min,
                    covariant_attn_use_transport=covariant_attn_use_transport,
                    covariant_attn_transport_eps=covariant_attn_transport_eps,
                    soft_equiv_metric=soft_equiv_metric,
                    soft_equiv_bundle_size=soft_equiv_bundle_size,
                    soft_equiv_hidden_dim=soft_equiv_hidden_dim,
                    soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
                    soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
                    soft_equiv_soft_assign=soft_equiv_soft_assign,
                    soft_equiv_temperature=soft_equiv_temperature,
                )
            )
            self.decoder_levels.append(
                PrimitiveTopologicalDecoder(
                    latent_dim=level_latent_dims[idx],
                    hidden_dim=hidden_dim,
                    num_charts=level_num_charts[idx],
                    output_dim=input_dim,
                    bundle_size=bundle_size,
                    covariant_attn=covariant_attn,
                    covariant_attn_tensorization=covariant_attn_tensorization,
                    covariant_attn_rank=covariant_attn_rank,
                    covariant_attn_tau_min=covariant_attn_tau_min,
                    covariant_attn_denom_min=covariant_attn_denom_min,
                    covariant_attn_use_transport=covariant_attn_use_transport,
                    covariant_attn_transport_eps=covariant_attn_transport_eps,
                )
            )

        self.jump_operators = None
        if enable_cross_level_jump:
            if n_levels < 2:
                msg = "enable_cross_level_jump requires at least 2 levels."
                raise ValueError(msg)
            if len(set(level_latent_dims)) != 1 or len(set(level_num_charts)) != 1:
                msg = (
                    "enable_cross_level_jump requires identical latent_dim and num_charts "
                    "across levels."
                )
                raise ValueError(msg)
            if jump_global_rank is not None and jump_global_rank < 0:
                msg = "jump_global_rank must be non-negative when provided."
                raise ValueError(msg)
            rank = None if jump_global_rank in {None, 0} else int(jump_global_rank)
            self.jump_operators = nn.ModuleList([
                FactorizedJumpOperator(
                    num_charts=level_num_charts[0],
                    latent_dim=level_latent_dims[0],
                    global_rank=rank,
                )
                for _ in range(n_levels - 1)
            ])

    def forward(
        self,
        x: torch.Tensor,
        step: int | torch.Tensor | None = None,
        prev_state: list[dict[str, torch.Tensor]] | None = None,
        use_hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
    ) -> list[dict[str, torch.Tensor]]:
        if prev_state is not None and len(prev_state) != self.n_levels:
            msg = "prev_state must have one entry per level."
            raise ValueError(msg)

        if step is not None:
            if isinstance(step, torch.Tensor):
                step_value = int(step.item())
            else:
                step_value = int(step)
        else:
            step_value = None

        # Shared feature extractor yields a common view for all levels.
        if self.share_feature_extractor:
            features = self.feature_extractor(x)
        else:
            features = None

        outputs: list[dict[str, torch.Tensor]] = []
        for idx in range(self.n_levels):
            if (
                step_value is not None
                and prev_state is not None
                and self.level_update_freqs[idx] > 1
                and step_value % self.level_update_freqs[idx] != 0
            ):
                outputs.append(prev_state[idx])
                continue

            # Optionally reuse cached features; otherwise compute per-level.
            if self.share_feature_extractor:
                level_features = features
            else:
                level_features = self.feature_extractors[idx](x)

            (
                K_chart,
                K_code,
                z_n,
                z_tex,
                enc_router_weights,
                z_geo,
                vq_loss,
                indices_stack,
                z_n_all_charts,
                c_bar,
                _v_local,
                _z_q_blended,
            ) = self.encoder_levels[idx](
                level_features,
                hard_routing=use_hard_routing,
                hard_routing_tau=hard_routing_tau,
            )

            router_override = enc_router_weights if use_hard_routing else None
            x_recon, dec_router_weights, _aux = self.decoder_levels[idx](
                z_geo,
                None,  # z_tex not used by decoder
                chart_index=None,
                router_weights=router_override,
                hard_routing=use_hard_routing,
                hard_routing_tau=hard_routing_tau,
            )
            z_n_local = _select_chart_latent(z_n_all_charts, K_chart)

            outputs.append({
                "x_recon": x_recon,
                "vq_loss": vq_loss,
                "enc_router_weights": enc_router_weights,
                "dec_router_weights": dec_router_weights,
                "K_chart": K_chart,
                "K_code": K_code,
                "z_geo": z_geo,
                "z_n": z_n,
                "z_n_local": z_n_local,
                "z_tex": z_tex,
                "indices_stack": indices_stack,
                "z_n_all_charts": z_n_all_charts,
                "c_bar": c_bar,
            })

        # Optional cross-level jump operators align nuisance coordinates across scales.
        if self.jump_operators is not None:
            for idx, jump_op in enumerate(self.jump_operators):
                src = outputs[idx]
                tgt = outputs[idx + 1]
                if "z_n_local" not in src:
                    src["z_n_local"] = _select_chart_latent(src["z_n_all_charts"], src["K_chart"])
                if "z_n_local" not in tgt:
                    tgt["z_n_local"] = _select_chart_latent(tgt["z_n_all_charts"], tgt["K_chart"])
                z_jump = jump_op(src["z_n_local"], src["K_chart"], tgt["K_chart"])
                src["z_n_jump_to_next"] = z_jump
                tgt["z_n_jump_from_prev"] = z_jump

        return outputs


class TopoEncoderAttachments(nn.Module):
    """Optional modules commonly attached to TopoEncoder training."""

    def __init__(
        self,
        num_charts: int,
        latent_dim: int,
        num_classes: int | None = None,
        enable_jump: bool = True,
        enable_classifier: bool = False,
        jump_global_rank: int | None = None,
        classifier_bundle_size: int | None = None,
    ) -> None:
        super().__init__()
        if num_charts <= 0:
            msg = "num_charts must be positive."
            raise ValueError(msg)
        if latent_dim <= 0:
            msg = "latent_dim must be positive."
            raise ValueError(msg)

        self.jump_operator = None
        if enable_jump:
            self.jump_operator = FactorizedJumpOperator(
                num_charts=num_charts,
                latent_dim=latent_dim,
                global_rank=jump_global_rank,
            )

        self.classifier_head = None
        if enable_classifier:
            if num_classes is None or num_classes <= 0:
                msg = "num_classes must be positive when classifier is enabled."
                raise ValueError(msg)
            self.classifier_head = InvariantChartClassifier(
                num_charts=num_charts,
                num_classes=num_classes,
                latent_dim=latent_dim,
                bundle_size=classifier_bundle_size,
            )

    def forward(
        self,
        router_weights: torch.Tensor | None = None,
        z_geo: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {}
        if self.classifier_head is not None:
            if router_weights is None or z_geo is None:
                msg = "router_weights and z_geo are required for classifier_head."
                raise ValueError(msg)
            outputs["classifier_logits"] = self.classifier_head(router_weights, z_geo)
        return outputs
