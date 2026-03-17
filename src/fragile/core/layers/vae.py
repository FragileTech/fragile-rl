from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class DisentangledConfig:
    """Configuration for the split-latent VQ-VAE agent."""

    obs_dim: int = 64 * 64 * 3
    hidden_dim: int = 256
    macro_embed_dim: int = 32
    codebook_size: int = 512
    nuisance_dim: int = 32
    tex_dim: int = 96
    action_dim: int = 4
    rnn_hidden_dim: int = 256
    lambda_closure: float = 1.0
    lambda_slowness: float = 0.1
    lambda_nuis_kl: float = 0.01
    lambda_tex_kl: float = 0.05
    lambda_vq: float = 1.0
    lambda_recon: float = 1.0
    tex_dropout_prob: float = 0.5


class Encoder(nn.Module):
    """Convolutional encoder backbone for observations."""

    def __init__(self, obs_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
        )
        self.flatten = nn.Flatten()
        self.proj = nn.Linear(256 * 4 * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations into a feature vector.

        Args:
            x: [B, C, H, W] input observations

        Returns:
            h: [B, hidden_dim] encoded features
        """
        h_conv = self.conv(x)  # [B, 256, 4, 4]
        h_flat = self.flatten(h_conv)  # [B, 4096]
        return self.proj(h_flat)  # [B, hidden_dim]


class VectorQuantizer(nn.Module):
    """Vector quantizer with straight-through estimator."""

    def __init__(self, codebook_size: int, embed_dim: int, beta: float = 0.25) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.beta = beta

        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def forward(self, z_e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize encoder outputs.

        Args:
            z_e: [B, D] continuous encoder outputs

        Returns:
            z_q: [B, D] quantized outputs (straight-through)
            indices: [B] code indices
            vq_loss: [] codebook + commitment loss
        """
        embed = self.embedding.weight  # [K, D]
        # Nearest-code lookup in macro latent space.
        z_sq = (z_e**2).sum(dim=1, keepdim=True)  # [B, 1]
        e_sq = (embed**2).sum(dim=1).unsqueeze(0)  # [1, K]
        dot = torch.matmul(z_e, embed.t())  # [B, K]
        dist = z_sq + e_sq - 2.0 * dot  # [B, K]

        indices = torch.argmin(dist, dim=1)  # [B]
        z_q = embed[indices]  # [B, D]

        commitment = F.mse_loss(z_e, z_q.detach())  # []
        codebook = F.mse_loss(z_q, z_e.detach())  # []
        vq_loss = codebook + self.beta * commitment  # []

        # Straight-through estimator preserves encoder gradients.
        z_q_st = z_e + (z_q - z_e).detach()  # [B, D]
        return z_q_st, indices, vq_loss


class Decoder(nn.Module):
    """Transposed-convolution decoder for reconstruction."""

    def __init__(
        self,
        macro_dim: int,
        nuisance_dim: int,
        tex_dim: int,
        obs_channels: int = 3,
    ) -> None:
        super().__init__()
        latent_dim = macro_dim + nuisance_dim + tex_dim

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.GELU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(32, obs_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent components into reconstructions.

        Args:
            z_macro: [B, macro_dim] macro latent
            z_nuis: [B, nuisance_dim] nuisance latent
            z_tex: [B, tex_dim] texture latent

        Returns:
            x_recon: [B, C, H, W] reconstructed observations
        """
        z = torch.cat([z_macro, z_nuis, z_tex], dim=-1)  # [B, D_total]
        h = self.fc(z)  # [B, 4096]
        h = h.view(z.shape[0], 256, 4, 4)  # [B, 256, 4, 4]
        return self.deconv(h)  # [B, C, 64, 64]


class MacroDynamicsModel(nn.Module):
    """Micro-blind dynamics model over macro codes."""

    def __init__(
        self,
        macro_embed_dim: int,
        action_dim: int,
        hidden_dim: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.gru = nn.GRUCell(macro_embed_dim + action_dim, hidden_dim)
        self.logits_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, codebook_size),
        )
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, macro_embed_dim),
        )

    def forward(
        self,
        z_macro: torch.Tensor,
        action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next macro embedding and logits.

        Args:
            z_macro: [B, Dm] macro embedding
            action: [B, Da] action vector
            hidden: [B, Dh] GRU hidden state

        Returns:
            logits: [B, K] logits over next macro code
            hidden_next: [B, Dh] updated hidden state
            z_pred: [B, Dm] predicted next embedding
        """
        # Macro dynamics ignore micro texture; action drives macro transitions.
        x = torch.cat([z_macro, action], dim=-1)  # [B, Dm+Da]
        hidden_next = self.gru(x, hidden)  # [B, Dh]
        logits = self.logits_head(hidden_next)  # [B, K]
        z_pred = self.pred_head(hidden_next)  # [B, Dm]
        return logits, hidden_next, z_pred


class DisentangledAgent(nn.Module):
    """Full split-latent VQ-VAE agent."""

    def __init__(self, config: DisentangledConfig) -> None:
        super().__init__()
        self.config = config

        self.encoder = Encoder(config.obs_dim, config.hidden_dim)
        self.vq = VectorQuantizer(config.codebook_size, config.macro_embed_dim)

        self.head_macro = nn.Linear(config.hidden_dim, config.macro_embed_dim)
        self.head_nuis_mu = nn.Linear(config.hidden_dim, config.nuisance_dim)
        self.head_nuis_logvar = nn.Linear(config.hidden_dim, config.nuisance_dim)
        self.head_tex_mu = nn.Linear(config.hidden_dim, config.tex_dim)
        self.head_tex_logvar = nn.Linear(config.hidden_dim, config.tex_dim)

        self.decoder = Decoder(
            config.macro_embed_dim, config.nuisance_dim, config.tex_dim, obs_channels=3
        )
        self.dynamics = MacroDynamicsModel(
            config.macro_embed_dim,
            config.action_dim,
            config.rnn_hidden_dim,
            config.codebook_size,
        )

    def _encode_stats(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode observations and return latent statistics.

        Args:
            obs: [B, C, H, W] observations

        Returns:
            stats: dict of latent tensors with shapes annotated in comments
        """
        h = self.encoder(obs)  # [B, H]
        # Macro code represents discrete world-state ("chart") summary.
        z_macro_e = self.head_macro(h)  # [B, Dm]
        z_macro_q, indices, vq_loss = self.vq(z_macro_e)  # [B, Dm], [B], []

        # Nuisance/texture are modeled as Gaussian latents.
        nuis_mu = self.head_nuis_mu(h)  # [B, Dn]
        nuis_logvar = self.head_nuis_logvar(h)  # [B, Dn]
        tex_mu = self.head_tex_mu(h)  # [B, Dt]
        tex_logvar = self.head_tex_logvar(h)  # [B, Dt]

        return {
            "z_macro_q": z_macro_q,
            "indices": indices,
            "vq_loss": vq_loss,
            "nuis_mu": nuis_mu,
            "nuis_logvar": nuis_logvar,
            "tex_mu": tex_mu,
            "tex_logvar": tex_logvar,
        }

    def encode(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observations into macro, nuisance, and texture latents.

        Args:
            obs: [B, C, H, W] observations

        Returns:
            z_macro: [B, Dm] quantized macro latent
            z_nuis: [B, Dn] nuisance latent
            z_tex: [B, Dt] texture latent
            indices: [B] macro code indices
            vq_loss: [] vector-quantizer loss
        """
        stats = self._encode_stats(obs)
        nuis_mu = stats["nuis_mu"]  # [B, Dn]
        nuis_logvar = stats["nuis_logvar"]  # [B, Dn]
        tex_mu = stats["tex_mu"]  # [B, Dt]
        tex_logvar = stats["tex_logvar"]  # [B, Dt]

        # Reparameterize continuous latents (nuisance, texture).
        eps_n = torch.randn_like(nuis_mu)  # [B, Dn]
        eps_t = torch.randn_like(tex_mu)  # [B, Dt]
        z_nuis = nuis_mu + eps_n * torch.exp(0.5 * nuis_logvar)  # [B, Dn]
        z_tex = tex_mu + eps_t * torch.exp(0.5 * tex_logvar)  # [B, Dt]

        if self.training and self.config.tex_dropout_prob > 0.0:
            # Texture dropout encourages robustness to micro-variations.
            keep = (
                torch.rand(z_tex.shape[0], 1, device=z_tex.device) > self.config.tex_dropout_prob
            ).float()  # [B, 1]
            z_tex *= keep  # [B, Dt]

        return stats["z_macro_q"], z_nuis, z_tex, stats["indices"], stats["vq_loss"]

    def decode(
        self, z_macro: torch.Tensor, z_nuis: torch.Tensor, z_tex: torch.Tensor
    ) -> torch.Tensor:
        """Decode latent triplet into reconstructed observations.

        Args:
            z_macro: [B, Dm] macro latent
            z_nuis: [B, Dn] nuisance latent
            z_tex: [B, Dt] texture latent

        Returns:
            x_recon: [B, C, H, W] reconstructed observations
        """
        return self.decoder(z_macro, z_nuis, z_tex)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor, hidden: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Full forward pass with reconstruction and dynamics prediction.

        Args:
            obs: [B, C, H, W] observations
            action: [B, Da] action vector
            hidden: [B, Dh] dynamics hidden state

        Returns:
            outputs: dict with latent tensors, reconstruction, dynamics, and losses
        """
        stats = self._encode_stats(obs)
        nuis_mu = stats["nuis_mu"]  # [B, Dn]
        nuis_logvar = stats["nuis_logvar"]  # [B, Dn]
        tex_mu = stats["tex_mu"]  # [B, Dt]
        tex_logvar = stats["tex_logvar"]  # [B, Dt]

        # Reparameterize continuous latents (nuisance, texture).
        eps_n = torch.randn_like(nuis_mu)  # [B, Dn]
        eps_t = torch.randn_like(tex_mu)  # [B, Dt]
        z_nuis = nuis_mu + eps_n * torch.exp(0.5 * nuis_logvar)  # [B, Dn]
        z_tex = tex_mu + eps_t * torch.exp(0.5 * tex_logvar)  # [B, Dt]

        if self.training and self.config.tex_dropout_prob > 0.0:
            # Texture dropout encourages robustness to micro-variations.
            keep = (
                torch.rand(z_tex.shape[0], 1, device=z_tex.device) > self.config.tex_dropout_prob
            ).float()  # [B, 1]
            z_tex *= keep  # [B, Dt]

        recon = self.decoder(stats["z_macro_q"], z_nuis, z_tex)  # [B, C, H, W]

        logits, hidden_next, z_pred = self.dynamics(
            stats["z_macro_q"], action, hidden
        )  # [B, K], [B, Dh], [B, Dm]

        recon_loss = F.mse_loss(recon, obs)  # []
        # KL terms regularize nuisance/texture to Gaussian priors.
        nuis_kl = (
            -0.5 * (1.0 + nuis_logvar - nuis_mu.pow(2) - nuis_logvar.exp()).sum(dim=1).mean()
        )  # []
        tex_kl = (
            -0.5 * (1.0 + tex_logvar - tex_mu.pow(2) - tex_logvar.exp()).sum(dim=1).mean()
        )  # []

        closure_loss = torch.tensor(0.0, device=obs.device)  # []
        slowness_loss = torch.tensor(0.0, device=obs.device)  # []

        loss_total = (  # []
            self.config.lambda_recon * recon_loss
            + self.config.lambda_vq * stats["vq_loss"]
            + self.config.lambda_nuis_kl * nuis_kl
            + self.config.lambda_tex_kl * tex_kl
            + self.config.lambda_closure * closure_loss
            + self.config.lambda_slowness * slowness_loss
        )

        return {
            "z_macro": stats["z_macro_q"],
            "z_nuis": z_nuis,
            "z_tex": z_tex,
            "indices": stats["indices"],
            "recon": recon,
            "next_logits": logits,
            "hidden_next": hidden_next,
            "z_pred": z_pred,
            "losses": {
                "loss_total": loss_total,
                "loss_recon": recon_loss,
                "loss_vq": stats["vq_loss"],
                "loss_nuis_kl": nuis_kl,
                "loss_tex_kl": tex_kl,
                "loss_closure": closure_loss,
                "loss_slowness": slowness_loss,
            },
        }


class HierarchicalDisentangled(nn.Module):
    """Multi-scale split-latent model with vectorized per-level quantization."""

    def __init__(
        self,
        config: DisentangledConfig,
        n_levels: int = 3,
        level_dims: list[int] | None = None,
        level_codebook_sizes: list[int] | None = None,
        level_update_freqs: list[int] | None = None,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.config = config
        self.n_levels = n_levels
        self.beta = beta

        level_dims = level_dims or [8, 16, 32]
        level_codebook_sizes = level_codebook_sizes or [64, 128, 256]
        level_update_freqs = level_update_freqs or [8, 4, 1]

        if not (
            len(level_dims) == len(level_codebook_sizes) == len(level_update_freqs) == n_levels
        ):
            msg = "Hierarchy lists must match n_levels."
            raise ValueError(msg)

        max_dim = int(max(level_dims))
        max_codes = int(max(level_codebook_sizes))

        self.encoder = Encoder(config.obs_dim, config.hidden_dim)

        self.macro_proj_weight = nn.Parameter(
            torch.randn(n_levels, config.hidden_dim, max_dim) * 0.02
        )
        self.macro_proj_bias = nn.Parameter(torch.zeros(n_levels, max_dim))

        self.codebook = nn.Parameter(torch.randn(n_levels, max_codes, max_dim) * 0.02)

        dim_range = torch.arange(max_dim).unsqueeze(0)  # [1, Dmax]
        code_range = torch.arange(max_codes).unsqueeze(0)  # [1, Kmax]
        dims = torch.tensor(level_dims).unsqueeze(1)  # [L, 1]
        codes = torch.tensor(level_codebook_sizes).unsqueeze(1)  # [L, 1]

        self.register_buffer("level_dims", torch.tensor(level_dims))
        self.register_buffer("level_codebook_sizes", torch.tensor(level_codebook_sizes))
        self.register_buffer("level_update_freqs", torch.tensor(level_update_freqs))
        self.register_buffer("level_dim_mask", (dim_range < dims).float())
        self.register_buffer("level_code_mask", (code_range < codes).float())

    def forward(
        self,
        obs: torch.Tensor,
        step: int | torch.Tensor | None = None,
        prev_z_macro: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Encode observations into hierarchical macro latents.

        Args:
            obs: [B, C, H, W] observations
            step: optional time step for update gating
            prev_z_macro: [B, L, Dmax] previous macro states (optional)

        Returns:
            outputs: dict with hierarchical latents and VQ loss
        """
        h = self.encoder(obs)  # [B, H]
        z_macro_e = torch.einsum("bh,lhd->bld", h, self.macro_proj_weight)  # [B, L, Dmax]
        z_macro_e += self.macro_proj_bias.unsqueeze(0)  # [B, L, Dmax]

        dim_mask = self.level_dim_mask.unsqueeze(0)  # [1, L, Dmax]
        z_macro_e *= dim_mask  # [B, L, Dmax]

        codebook = self.codebook.unsqueeze(0)  # [1, L, Kmax, Dmax]
        z_exp = z_macro_e.unsqueeze(2)  # [B, L, 1, Dmax]
        diff = z_exp - codebook  # [B, L, Kmax, Dmax]
        dist = (diff**2).sum(dim=-1)  # [B, L, Kmax]

        code_mask = self.level_code_mask.unsqueeze(0)  # [1, L, Kmax]
        dist += (1.0 - code_mask) * 1000000000.0  # [B, L, Kmax]

        indices = torch.argmin(dist, dim=-1)  # [B, L]
        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, z_macro_e.shape[-1])  # [B, L, 1, Dmax]
        z_q = torch.gather(codebook.expand(z_macro_e.shape[0], -1, -1, -1), 2, indices_exp)
        z_q = z_q.squeeze(2)  # [B, L, Dmax]

        commitment = ((z_macro_e - z_q.detach()) ** 2 * dim_mask).mean()  # []
        codebook_loss = ((z_q - z_macro_e.detach()) ** 2 * dim_mask).mean()  # []
        vq_loss = codebook_loss + self.beta * commitment  # []

        z_q_st = z_macro_e + (z_q - z_macro_e).detach()  # [B, L, Dmax]

        if step is not None and prev_z_macro is not None:
            step_t = torch.as_tensor(step, device=obs.device)  # []
            update_mask = (step_t % self.level_update_freqs == 0).float()  # [L]
            update_mask = update_mask.view(1, -1, 1)  # [1, L, 1]
            z_macro = update_mask * z_q_st + (1.0 - update_mask) * prev_z_macro  # [B, L, Dmax]
        else:
            z_macro = z_q_st  # [B, L, Dmax]

        return {
            "z_macro": z_macro,
            "indices": indices,
            "vq_loss": vq_loss,
            "z_macro_e": z_macro_e,
        }
