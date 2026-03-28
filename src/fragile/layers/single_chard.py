from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import (
    exp_map_zero,
    log_map_zero,
    mobius_add,
    poincare_weighted_mean,
    project_to_ball,
    smooth_tangent_to_ball,
)
from fragile.layers.initialization import resolve_bundle_params, spread_codebook
from fragile.layers.primitives import IsotropicBlock, NormGatedGELU, SoftEquivariantLayer, SpectralLinear
from fragile.layers.topoencoder import GlobalAffineMap


class SingleChardEncoder(nn.Module):
    """Single-chart hyperbolic encoder with symbol, nuisance, and texture splits.

    This module keeps the same local geometry decomposition as ``TopoEncoder``
    but removes atlas routing entirely. The latent lives in a single implicit
    chart centered at the origin, with a hyperbolic codebook, a nuisance
    ``z_n`` branch, and a residual texture ``z_tex`` branch.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
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
        self.num_charts = 1
        self.latent_dim = int(latent_dim)
        self.codes_per_chart = int(codes_per_chart)
        self._commitment_beta = float(commitment_beta)
        self._codebook_loss_weight = float(codebook_loss_weight)
        self.soft_equiv_soft_assign = bool(soft_equiv_soft_assign)
        self.soft_equiv_temperature = float(soft_equiv_temperature)

        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)
        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
        )
        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        self.val_proj_scale = nn.Parameter(torch.tensor(2.0))

        # Single implicit chart centered at the origin.
        self.register_buffer("chart_centers", torch.zeros(1, latent_dim))
        self.codebook = nn.Parameter(
            spread_codebook(1, codes_per_chart, latent_dim, radius=0.3),
        )

        self.soft_equiv_layer: SoftEquivariantLayer | None = None
        if soft_equiv_metric:
            equiv_bundle = soft_equiv_bundle_size or latent_dim
            if equiv_bundle <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % equiv_bundle != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            self.soft_equiv_layer = SoftEquivariantLayer(
                n_bundles=latent_dim // equiv_bundle,
                bundle_dim=equiv_bundle,
                hidden_dim=soft_equiv_hidden_dim,
                use_spectral_norm=soft_equiv_use_spectral_norm,
                zero_self_mixing=soft_equiv_zero_self_mixing,
            )

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff_tan: torch.Tensor) -> torch.Tensor:
        if self.soft_equiv_layer is None:
            return (diff_tan**2).sum(dim=-1)
        batch_size, num_codes, latent_dim = diff_tan.shape
        diff_flat = diff_tan.reshape(-1, latent_dim)
        diff_out = self.soft_equiv_layer(diff_flat).view(batch_size, num_codes, latent_dim)
        return (diff_out**2).sum(dim=-1)

    def _hyperbolic_vq(
        self,
        v_local: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        codebook = project_to_ball(self.codebook[0])  # [K, D]
        diff = mobius_add(-codebook.unsqueeze(0), v_local.unsqueeze(1))  # [B, K, D]
        diff_tan = log_map_zero(diff)
        dist = self._apply_soft_equiv_metric(diff_tan)

        code_idx = torch.argmin(dist, dim=-1)  # [B]
        z_q = codebook[code_idx]  # [B, D]

        if self.soft_equiv_layer is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = poincare_weighted_mean(codebook, weights)
            z_q = z_q + z_q_soft - z_q_soft.detach()

        delta_commit = log_map_zero(mobius_add(-z_q.detach(), v_local))
        commitment = (delta_commit**2).sum(dim=-1).mean()
        delta_codebook = log_map_zero(mobius_add(-v_local.detach(), z_q))
        codebook_loss = (delta_codebook**2).sum(dim=-1).mean()
        vq_loss = self._codebook_loss_weight * codebook_loss + self._commitment_beta * commitment
        return z_q, code_idx, code_idx.unsqueeze(1), vq_loss

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
        del routing_tau
        features = self._encode_features(x)
        v_raw = self.val_proj(features) * self.val_proj_scale
        v_local = smooth_tangent_to_ball(v_raw)

        z_q, k_code, indices_stack, vq_loss = self._hyperbolic_vq(v_local)

        delta = log_map_zero(mobius_add(-z_q.detach(), v_local))
        z_n_tan = self.structure_filter(delta)
        z_n = project_to_ball(exp_map_zero(z_n_tan)).unsqueeze(1)
        z_tex = delta - z_n_tan

        delta_to_code = log_map_zero(mobius_add(-v_local, z_q))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_geo = project_to_ball(mobius_add(z_q_st, exp_map_zero(z_n_tan)))

        batch_size = x.shape[0]
        k_chart = torch.zeros(batch_size, dtype=torch.long, device=x.device)
        router_weights = torch.ones(batch_size, 1, device=x.device, dtype=x.dtype)
        c_bar = torch.zeros(batch_size, self.latent_dim, device=x.device, dtype=x.dtype)

        return (
            k_chart,
            k_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n,
            c_bar,
            v_local,
            z_q,
        )


class SingleChardDecoder(nn.Module):
    """Single-chart hyperbolic decoder without routing overhead."""

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        output_dim: int = 2,
        bundle_size: int | None = None,
    ) -> None:
        super().__init__()
        self.num_charts = 1
        self.hidden_dim = int(hidden_dim)
        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projector = SpectralLinear(latent_dim, hidden_dim, bias=False)
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_fc1 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act1 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_fc2 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act2 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_out = SpectralLinear(hidden_dim, output_dim, bias=True)
        self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)

    def forward(
        self,
        z_geo: torch.Tensor,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        del chart_index, routing_tau
        z_geo = project_to_ball(z_geo)
        batch_size = z_geo.shape[0]
        if router_weights is None:
            router_weights = torch.ones(batch_size, 1, device=z_geo.device, dtype=z_geo.dtype)
        elif router_weights.ndim != 2 or router_weights.shape[1] != 1:
            msg = "router_weights must have shape [B, 1] for SingleChardDecoder."
            raise ValueError(msg)

        h = self.chart_projector(z_geo)
        h = self.chart_gate(h)
        h_skip = self.render_skip(h)
        h = self.render_fc1(h)
        h = self.render_act1(h)
        h = self.render_fc2(h)
        h = self.render_act2(h)
        x_hat = self.render_out(h) + h_skip
        return x_hat, router_weights, {}


class SingleChard(nn.Module):
    """Single-chart autoencoder with the same high-level API as ``TopoEncoder``."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        input_affine_enabled: bool = False,
        input_affine_learnable: bool = False,
        input_affine_min_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        self.num_charts = 1
        self.codes_per_chart = int(codes_per_chart)
        self.io_affine = GlobalAffineMap(
            input_dim,
            enabled=input_affine_enabled,
            learnable=input_affine_learnable,
            min_scale=input_affine_min_scale,
        )
        self.encoder = SingleChardEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            codes_per_chart=codes_per_chart,
            bundle_size=bundle_size,
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
        self.decoder = SingleChardDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            bundle_size=bundle_size,
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return self.io_affine.normalize(x)

    def denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        return self.io_affine.denormalize(x)

    def loss_space_pair(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.normalize_input(x), self.normalize_input(x_recon)

    @torch.no_grad()
    def set_io_affine_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        learnable: bool | None = None,
    ) -> None:
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
            k_chart,
            _k_code,
            z_n,
            _z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices_stack,
            _z_n_all,
            c_bar,
            _v_local,
            _z_q,
        ) = self.encoder(
            self.normalize_input(x),
            routing_tau=routing_tau,
        )
        x_recon, dec_router_weights, aux_losses = self.decode(
            z_geo,
            chart_index=None,
            router_weights=enc_router_weights,
            routing_tau=routing_tau,
        )
        return (
            x_recon,
            vq_loss,
            enc_router_weights,
            dec_router_weights,
            k_chart,
            z_geo,
            z_n,
            c_bar,
            aux_losses,
        )

    def compute_consistency_loss(
        self,
        enc_weights: torch.Tensor,
        dec_weights: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)
        return kl.mean()

    def compute_perplexity(self, k_chart: torch.Tensor) -> float:
        counts = torch.bincount(k_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())


SingleChartEncoder = SingleChardEncoder
SingleChartDecoder = SingleChardDecoder
SingleChart = SingleChard
