import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import IsotropicBlock, NormGatedGELU, SoftEquivariantLayer, SpectralLinear
from fragile.layers.gauge import (
    exp_map_zero,
    log_map_zero,
    mobius_add,
    poincare_weighted_mean,
    poincare_weighted_mean_per_chart,
    project_to_ball,
    smooth_tangent_to_ball,
)
from fragile.layers.initialization import (
    init_soft_equiv_layers,
    resolve_bundle_params,
    spread_codebook,
    spread_directions,
)
from fragile.layers.router import CovariantChartRouter


class GlobalAffineMap(nn.Module):
    """Global per-dimension affine map shared by encoder input and decoder output.

    The map is deterministic and invertible as long as every scale is positive.
    It can be initialized from dataset statistics and optionally left frozen, so
    dreamed latents can still be decoded back to the original raw coordinate
    system without needing an accompanying input sample.
    """

    def __init__(
        self,
        dim: int,
        *,
        enabled: bool = False,
        learnable: bool = False,
        min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.min_scale = float(min_scale)
        self.register_buffer(
            "_enabled",
            torch.tensor(bool(enabled), dtype=torch.bool),
            persistent=True,
        )
        self.offset = nn.Parameter(torch.zeros(dim), requires_grad=learnable)
        self.log_scale = nn.Parameter(torch.zeros(dim), requires_grad=learnable)

    @property
    def enabled(self) -> bool:
        """Whether the affine normalization is active."""
        return bool(self._enabled.item())

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the affine map."""
        self._enabled.fill_(bool(enabled))

    def set_learnable(self, learnable: bool) -> None:
        """Toggle gradient updates for the affine parameters."""
        self.offset.requires_grad_(learnable)
        self.log_scale.requires_grad_(learnable)

    def scale(self) -> torch.Tensor:
        """Return the positive per-dimension scale."""
        return self.log_scale.exp().clamp_min(self.min_scale)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw inputs into the normalized model space."""
        if not self.enabled:
            return x
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        scale = self.scale().to(device=x.device, dtype=x.dtype)
        return (x - offset) / scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized model outputs back to the raw data space."""
        if not self.enabled:
            return x
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        scale = self.scale().to(device=x.device, dtype=x.dtype)
        return x * scale + offset

    @torch.no_grad()
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Initialize the affine map from dataset mean/std statistics."""
        mean_t = torch.as_tensor(mean, device=self.offset.device, dtype=self.offset.dtype)
        std_t = torch.as_tensor(std, device=self.offset.device, dtype=self.offset.dtype)
        if mean_t.shape != self.offset.shape:
            msg = "mean must have shape [input_dim]."
            raise ValueError(msg)
        if std_t.shape != self.offset.shape:
            msg = "std must have shape [input_dim]."
            raise ValueError(msg)
        self.offset.copy_(mean_t)
        self.log_scale.copy_(std_t.clamp_min(self.min_scale).log())
        self.set_enabled(True)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, enabled={self.enabled}, "
            f"learnable={self.offset.requires_grad}, min_scale={self.min_scale}"
        )


class AttentiveAtlasEncoder(nn.Module):
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
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
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
        self.covariant_attn = covariant_attn
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
        )

        self.cov_router = CovariantChartRouter(
            latent_dim=latent_dim,
            key_dim=hidden_dim,
            num_charts=num_charts,
            feature_dim=hidden_dim,
            tau_min=covariant_attn_tau_min,
            tau_denom_min=covariant_attn_denom_min,
            transport_eps=covariant_attn_transport_eps,
        )
        self.key_proj = None
        self.chart_queries = None
        self.scale = None

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        self.val_proj_scale = nn.Parameter(torch.tensor(2.0))  # learnable pre-squash scale

        # Quasi-uniform chart centers: Fibonacci sphere (3-D) or repulsion
        # init so every chart starts with a distinct, well-separated catchment
        # region — prevents softmax winner-take-all collapse at epoch 0.
        self.chart_centers = nn.Parameter(spread_directions(num_charts, latent_dim) * 0.5)

        # Spread codebook codes around the local origin of each chart so that
        # VQ does not instantly collapse to a single nearest-neighbor.
        self.codebook = nn.Parameter(
            spread_codebook(num_charts, codes_per_chart, latent_dim, radius=0.3)
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
            init_soft_equiv_layers(self.soft_equiv_layers)

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
        self._last_v_raw: torch.Tensor | None = None
        self._last_soft_router_weights: torch.Tensor | None = None
        self._last_soft_router_weights_live: torch.Tensor | None = None
        self._last_router_scores: torch.Tensor | None = None
        self._last_router_scores_live: torch.Tensor | None = None

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
            k_code: [B] winning code index for the winning chart.
            indices: [B, N_c] nearest code per chart.
            vq_loss: scalar VQ loss.
            z_q_all: [B, N_c, D] nearest code per chart (full tensor).
        """
        codebook = project_to_ball(codebook_param)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)

        if use_soft_equiv:
            dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        else:
            dist = (diff_tan**2).sum(dim=-1)  # [B, N_c, K]

        indices = torch.argmin(dist, dim=-1)  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v_local.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]

        if use_soft_equiv and self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = poincare_weighted_mean_per_chart(codebook, weights)
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing.
        weights = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * weights).mean(dim=(0, 2)).sum()
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss_val = (delta_codebook**2 * weights).mean(dim=(0, 2)).sum()
        vq_loss = codebook_loss_weight * codebook_loss_val + commitment_beta * commitment

        k_chart = torch.argmax(router_weights, dim=1)  # [B]
        z_q_blended = poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        k_code = indices.gather(1, k_chart.unsqueeze(1)).squeeze(1)  # [B]

        return z_q_blended, k_code, indices, vq_loss, z_q_all

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
        assert self.codebook_dyn is not None, (
            "dynamics codebook not initialized (dyn_codes_per_chart=0)"
        )
        return self._hyperbolic_vq(
            v_local,
            self.codebook_dyn,
            router_weights,
            self._dyn_commitment_beta,
            self._dyn_codebook_loss_weight,
            use_soft_equiv=False,
        )[:4]

    def forward(
        self,
        x: torch.Tensor,
        routing_tau: float = 1.0,
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
        v = smooth_tangent_to_ball(v_raw)  # [B, D]
        chart_centers = project_to_ball(self.chart_centers)  # [N_c, D]
        router_weights, K_chart = self.cov_router(
            v,
            features=features,
            chart_tokens=chart_centers,
            routing_tau=routing_tau,
        )
        self._last_soft_router_weights = self.cov_router._last_soft_router_weights
        self._last_soft_router_weights_live = self.cov_router._last_soft_router_weights_live
        self._last_router_scores = self.cov_router._last_router_scores
        self._last_router_scores_live = self.cov_router._last_router_scores_live

        c_bar = poincare_weighted_mean(chart_centers, router_weights)  # [B, D]
        v_local_raw = mobius_add(-c_bar, v)
        v_local = project_to_ball(v_local_raw)  # [B, D]

        # Per-chart codebook lookup via shared VQ helper.
        z_q_blended, K_code, indices_stack, vq_loss, z_q_all = self._hyperbolic_vq(
            v_local,
            self.codebook,
            router_weights,
            self._commitment_beta,
            self._codebook_loss_weight,
            use_soft_equiv=True,
        )

        # Structure filter extracts nuisance; remainder is texture.
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_local = mobius_add(z_q_st, exp_map_zero(z_n_tan))
        z_geo_raw = mobius_add(c_bar, z_local)
        z_geo = project_to_ball(z_geo_raw)  # [B, D]

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


class TopologicalDecoder(nn.Module):
    """Topological decoder using gauge-covariant primitives."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
        bundle_size: int | None = None,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
        film_conditioning: bool = False,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps

        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projectors = nn.ModuleList([
            SpectralLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_charts)
        ])
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )
        self.cov_router = CovariantChartRouter(
            latent_dim=latent_dim,
            key_dim=hidden_dim,
            num_charts=num_charts,
            feature_dim=None,
            tau_min=covariant_attn_tau_min,
            tau_denom_min=covariant_attn_denom_min,
            transport_eps=covariant_attn_transport_eps,
        )

        self.render_fc1 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act1 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_fc2 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act2 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_out = SpectralLinear(hidden_dim, output_dim, bias=True)
        self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)
        if film_conditioning:
            self.film1 = _ChartFiLM1d(num_charts, hidden_dim)
            self.film2 = _ChartFiLM1d(num_charts, hidden_dim)
        else:
            self.film1 = None
            self.film2 = None

    def forward(
        self,
        z_geo: torch.Tensor,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Decode from latent geometry.

        Returns:
            x_hat: [B, D_out] reconstruction
            router_weights: [B, N_c] routing weights
            aux_losses: dict of auxiliary losses
        """
        aux_losses: dict[str, torch.Tensor] = {}

        # Clamp geometry to chart range (Poincare ball).
        z_geo = project_to_ball(z_geo)
        chart_centers = project_to_ball(self.chart_centers)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        else:
            # Covariant router predicts chart membership from geometry.
            router_weights, _ = self.cov_router(
                z_geo,
                chart_tokens=chart_centers,
                routing_tau=routing_tau,
            )

        # Chart-specific projections + gauge-covariant gating.
        h_stack = torch.stack(
            [proj(z_geo) for proj in self.chart_projectors], dim=1
        )  # [B, N_c, H]
        h_stack = self.chart_gate(h_stack.view(-1, self.hidden_dim)).view(
            z_geo.shape[0], self.num_charts, self.hidden_dim
        )
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

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

        return x_hat, router_weights, aux_losses


class TopoEncoder(nn.Module):
    """Attentive Atlas encoder + topological decoder."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        film_conditioning: bool = False,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        input_affine_enabled: bool = False,
        input_affine_learnable: bool = False,
        input_affine_min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.io_affine = GlobalAffineMap(
            input_dim,
            enabled=input_affine_enabled,
            learnable=input_affine_learnable,
            min_scale=input_affine_min_scale,
        )

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            bundle_size=bundle_size,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
        )
        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
            bundle_size=bundle_size,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            film_conditioning=film_conditioning,
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw inputs into the model's normalized coordinate space."""
        return self.io_affine.normalize(x)

    def denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Project normalized decoder outputs back into raw data coordinates."""
        return self.io_affine.denormalize(x)

    def loss_space_pair(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction pair in the normalized training space."""
        return self.normalize_input(x), self.normalize_input(x_recon)

    @torch.no_grad()
    def set_io_affine_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        learnable: bool | None = None,
    ) -> None:
        """Initialize the optional affine map from dataset-level statistics."""
        self.io_affine.set_stats(mean, std)
        if learnable is not None:
            self.io_affine.set_learnable(learnable)

    def decode(
        self,
        z_geo: torch.Tensor,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        routing_tau: float = 1.0,
        *,
        return_model_space: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Decode a latent state and optionally expose the normalized output."""
        x_model, dec_router_weights, aux_losses = self.decoder(
            z_geo,
            chart_index=chart_index,
            router_weights=router_weights,
            routing_tau=routing_tau,
        )
        x_raw = self.denormalize_output(x_model)
        if return_model_space:
            aux_losses = dict(aux_losses)
            aux_losses["x_model"] = x_model
        return x_raw, dec_router_weights, aux_losses

    def forward(
        self,
        x: torch.Tensor,
        routing_tau: float = 1.0,
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
            _z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
            _z_q_blended,
        ) = self.encoder(
            self.normalize_input(x),
            routing_tau=routing_tau,
        )

        router_override = enc_router_weights
        x_recon, dec_router_weights, aux_losses = self.decode(
            z_geo,
            chart_index=None,
            router_weights=router_override,
            routing_tau=routing_tau,
        )

        return (
            x_recon,
            vq_loss,
            enc_router_weights,
            dec_router_weights,
            K_chart,
            z_geo,
            z_n,
            c_bar,
            aux_losses,
        )

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
