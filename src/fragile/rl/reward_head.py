"""Reward decomposition into residual connection and Poisson source density."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from fragile.layers.attention import CovariantAttention, GeodesicConfig
from fragile.layers.primitives import SpectralLinear
from fragile.rl.action_manifold import LatentTokenizer
from fragile.vla.covariant_world_model import (
    ChartTokenizer,
    CovariantPotentialNet,
)


class RewardHead(nn.Module):
    """Model the non-exact reward sector and the conservative source density."""

    def __init__(
        self,
        potential_net: CovariantPotentialNet,
        num_action_charts: int,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        latent_dim = potential_net.latent_dim

        self.chart_tok = potential_net.chart_tok
        self.z_embed = potential_net.z_embed
        self.rho_attn = CovariantAttention(
            GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1),
        )
        self.reward_density_out = SpectralLinear(d_model, 1)
        self.action_chart_tok = ChartTokenizer(num_action_charts, d_model, latent_dim)
        self.action_lat_tok = LatentTokenizer(latent_dim, d_model)
        self.action_code_tok = LatentTokenizer(latent_dim, d_model)
        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.reward_attn = CovariantAttention(geo_cfg)
        self.reward_form_out = SpectralLinear(d_model, latent_dim)

    def reward_density(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """State-only conservative source density for the screened Poisson equation."""
        chart_x, chart_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        feat, _ = self.rho_attn(z, chart_z, x_q, chart_x, chart_x)
        return self.reward_density_out(feat)

    def _reward_features(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
    ) -> torch.Tensor:
        """Build the shared context used by the residual reward one-form."""
        chart_x, chart_z = self.chart_tok(rw, z)
        action_chart_x, action_chart_z = self.action_chart_tok(action_rw, action_z)
        action_lat_x, action_lat_z = self.action_lat_tok(action_z)
        action_code_x, action_code_z_tok = self.action_code_tok(action_code_z)
        ctx_x = torch.cat(
            [action_lat_x, action_code_x, action_chart_x, chart_x],
            dim=1,
        )
        ctx_z = torch.cat(
            [action_lat_z, action_code_z_tok, action_chart_z, chart_z],
            dim=1,
        )

        x_q = self.z_embed(z)
        feat, _ = self.reward_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return feat

    def _project_exact_component(
        self,
        reward_form_cov: torch.Tensor,
        exact_covector: torch.Tensor | None,
        *,
        detach_exact_covector: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the learned covector off the exact critic direction."""
        if exact_covector is None:
            return reward_form_cov, torch.zeros_like(reward_form_cov)

        exact_covector = exact_covector.to(
            device=reward_form_cov.device, dtype=reward_form_cov.dtype
        )
        if detach_exact_covector:
            exact_covector = exact_covector.detach()
        # The current latent geometry is conformal: G^{ij}(z) = lambda(z)^{-2} delta^{ij}.
        # For a projection onto a single covector direction, that scalar factor cancels
        # between numerator and denominator, so the pointwise coefficient matches the
        # Euclidean expression used here.
        exact_norm_sq = exact_covector.pow(2).sum(dim=-1, keepdim=True)
        safe_exact_norm_sq = exact_norm_sq.clamp_min(1e-8)
        exact_coeff = (reward_form_cov * exact_covector).sum(dim=-1, keepdim=True)
        exact_coeff = torch.where(
            exact_norm_sq > 1e-8,
            exact_coeff / safe_exact_norm_sq,
            torch.zeros_like(exact_coeff),
        )
        exact_component = exact_coeff * exact_covector
        return reward_form_cov - exact_component, exact_component

    def reward_form(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
    ) -> torch.Tensor:
        """Return the learned residual reward one-form ``A(z, a)``."""
        feat = self._reward_features(
            z,
            rw,
            action_z,
            action_rw,
            action_code_z,
        )
        return self.reward_form_out(feat)

    def reward_curl(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
        *,
        exact_covector: torch.Tensor | None = None,
        exact_covector_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        max_batch: int | None = None,
    ) -> torch.Tensor:
        """Return the actual exterior derivative ``dA`` of the projected one-form."""
        if max_batch is not None and max_batch > 0 and z.shape[0] > max_batch:
            z = z[:max_batch]
            rw = rw[:max_batch]
            action_z = action_z[:max_batch]
            action_rw = action_rw[:max_batch]
            action_code_z = action_code_z[:max_batch]
            if exact_covector is not None:
                exact_covector = exact_covector[:max_batch]

        z_req = z.detach().requires_grad_(True)
        exact_covector_local = (
            exact_covector_fn(z_req) if exact_covector_fn is not None else exact_covector
        )
        reward_form_cov_raw = self.reward_form(
            z_req,
            rw.detach(),
            action_z.detach(),
            action_rw.detach(),
            action_code_z.detach(),
        )
        reward_form_cov, _reward_form_exact_component = self._project_exact_component(
            reward_form_cov_raw,
            exact_covector_local,
            detach_exact_covector=exact_covector_fn is None,
        )
        jacobian_rows: list[torch.Tensor] = []
        latent_dim = reward_form_cov.shape[-1]
        for coord in range(latent_dim):
            grad_coord = torch.autograd.grad(
                reward_form_cov[:, coord].sum(),
                z_req,
                retain_graph=coord + 1 < latent_dim,
                create_graph=False,
            )[0]
            jacobian_rows.append(grad_coord.detach())
        jacobian = torch.stack(jacobian_rows, dim=1)
        return 0.5 * (jacobian - jacobian.transpose(1, 2))

    def decompose(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
        action_canonical: torch.Tensor,
        *,
        exact_covector: torch.Tensor | None = None,
        exact_covector_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        compute_curl: bool = False,
        curl_batch_limit: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return the residual reward connection and conservative source density.

        The deployed RL path contracts the learned residual one-form with the
        canonical motor-side action latent, without introducing a second motor
        variable downstream of the policy.
        """
        rho_r = self.reward_density(z, rw)
        reward_form_cov_raw = self.reward_form(
            z,
            rw,
            action_z,
            action_rw,
            action_code_z,
        )
        reward_form_cov, reward_form_exact_component = self._project_exact_component(
            reward_form_cov_raw,
            exact_covector,
        )
        reward_nonconservative = (reward_form_cov * action_canonical).sum(dim=-1, keepdim=True)
        reward_curl = (
            self.reward_curl(
                z,
                rw,
                action_z,
                action_rw,
                action_code_z,
                exact_covector=exact_covector,
                exact_covector_fn=exact_covector_fn,
                max_batch=curl_batch_limit,
            )
            if compute_curl
            else z.new_zeros((0, z.shape[-1], z.shape[-1]))
        )
        return {
            "reward_nonconservative": reward_nonconservative,
            "reward_density": rho_r,
            "reward_form_cov": reward_form_cov,
            "reward_form_cov_raw": reward_form_cov_raw,
            "reward_form_exact_component": reward_form_exact_component,
            "reward_curl": reward_curl,
        }

    def forward(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
        action_z: torch.Tensor,
        action_rw: torch.Tensor,
        action_code_z: torch.Tensor,
        action_canonical: torch.Tensor,
        *,
        exact_covector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict the non-conservative scalar reward contribution.

        Args:
            z: [B, D] latent position.
            rw: [B, K] chart weights.
            action_z: [B, D] action-manifold latent.
            action_rw: [B, K_a] action-manifold routing weights.
            action_code_z: [B, D] action-manifold code latent.
            action_canonical: [B, D] canonical motor-side action latent.

        Returns:
            r_noncons: [B, 1] non-conservative reward contribution.
        """
        return self.decompose(
            z,
            rw,
            action_z,
            action_rw,
            action_code_z,
            action_canonical,
            exact_covector=exact_covector,
        )["reward_nonconservative"]
