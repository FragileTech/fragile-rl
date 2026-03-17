from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .gauge import CovariantAttention, GeodesicConfig
from .primitives import SpectralLinear


class FactoredTensorLayer(nn.Module):
    """Low-rank factorization of a tensor-product interaction."""

    def __init__(self, d_C: int, d_L: int, d_Y: int, rank: int, d_out: int) -> None:
        super().__init__()
        if min(d_C, d_L, d_Y, rank, d_out) <= 0:
            msg = "All dimensions must be positive."
            raise ValueError(msg)
        self.rank = rank

        self.U_C = nn.Linear(d_C, rank, bias=False)
        self.U_L = nn.Linear(d_L, rank, bias=False)
        self.U_Y = nn.Linear(d_Y, rank, bias=False)
        self.U_out = nn.Linear(rank, d_out, bias=False)

    def forward(self, z_C: torch.Tensor, z_L: torch.Tensor, z_Y: torch.Tensor) -> torch.Tensor:
        if z_C.dim() != 2 or z_L.dim() != 2 or z_Y.dim() != 2:
            msg = "Inputs must have shape [B, d]."
            raise ValueError(msg)
        if z_C.shape[0] != z_L.shape[0] or z_C.shape[0] != z_Y.shape[0]:
            msg = "Input batch sizes must match."
            raise ValueError(msg)
        # Low-rank tensor interaction via elementwise product in factor space.
        interaction = self.U_C(z_C) * self.U_L(z_L) * self.U_Y(z_Y)
        return self.U_out(interaction)


class NormInteractionLayer(nn.Module):
    """Level 1: Norms-only cross-bundle interaction."""

    def __init__(self, n_bundles: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if n_bundles <= 0:
            msg = "n_bundles must be positive."
            raise ValueError(msg)
        if hidden_dim <= 0:
            msg = "hidden_dim must be positive."
            raise ValueError(msg)
        self.n_bundles = n_bundles

        self.norm_mlp = nn.Sequential(
            nn.Linear(n_bundles, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bundles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3 or z.shape[1] != self.n_bundles:
            msg = "Expected input shape [B, n_bundles, bundle_dim]."
            raise ValueError(msg)
        # Use bundle norms (gauge-invariant energy) to scale each bundle.
        norms = torch.norm(z, dim=-1)
        scales = F.softplus(self.norm_mlp(norms))
        return z * scales.unsqueeze(-1)


class GramInteractionLayer(nn.Module):
    """Level 2: Gram matrix cross-bundle interaction."""

    def __init__(self, n_bundles: int, hidden_dim: int = 64) -> None:
        super().__init__()
        if n_bundles <= 0:
            msg = "n_bundles must be positive."
            raise ValueError(msg)
        if hidden_dim <= 0:
            msg = "hidden_dim must be positive."
            raise ValueError(msg)
        self.n_bundles = n_bundles
        gram_dim = n_bundles * n_bundles

        self.mlp = nn.Sequential(
            nn.Linear(gram_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_bundles),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() != 3 or z.shape[1] != self.n_bundles:
            msg = "Expected input shape [B, n_bundles, bundle_dim]."
            raise ValueError(msg)
        # Gram matrix captures inter-bundle angles for texture-level mixing.
        gram = torch.bmm(z, z.transpose(-1, -2))
        gram_flat = gram.view(z.shape[0], -1)
        scales = F.softplus(self.mlp(gram_flat))
        return z * scales.unsqueeze(-1)


class L1Scheduler:
    """Adaptive L1 regularization schedule."""

    def __init__(
        self,
        lambda_init: float = 0.01,
        target_violation: float = 0.1,
        adaptation_rate: float = 0.01,
    ) -> None:
        self.lambda_L1 = float(lambda_init)
        self.target = float(target_violation)
        self.alpha = float(adaptation_rate)

    def step(self, current_violation: float) -> float:
        error = current_violation - self.target
        self.lambda_L1 *= 1 + self.alpha * error
        self.lambda_L1 = max(1e-6, min(1.0, self.lambda_L1))
        return self.lambda_L1


class AdaptiveL1Scheduler:
    """Adaptive L1 regularization schedule with history."""

    def __init__(
        self,
        initial_lambda: float = 0.01,
        target_violation: float = 0.22,
        learning_rate: float = 0.05,
        min_lambda: float = 1e-4,
        max_lambda: float = 1.0,
    ) -> None:
        self.lambda_l1 = float(initial_lambda)
        self.target_violation = float(target_violation)
        self.alpha = float(learning_rate)
        self.min_lambda = float(min_lambda)
        self.max_lambda = float(max_lambda)

        self.history = {
            "lambda_l1": [self.lambda_l1],
            "violation": [],
        }

    def step(self, current_violation: float) -> float:
        error = current_violation - self.target_violation
        self.lambda_l1 *= 1 + self.alpha * error
        self.lambda_l1 = max(self.min_lambda, min(self.max_lambda, self.lambda_l1))

        self.history["lambda_l1"].append(self.lambda_l1)
        self.history["violation"].append(current_violation)
        return self.lambda_l1


@dataclass
class BundleConfig:
    """Configuration for a single gauge bundle."""

    name: str
    dim: int
    semantic_role: str = ""


@dataclass
class UGNConfig:
    """Configuration for Universal Geometric Network."""

    input_dim: int
    output_dim: int
    bundles: list[BundleConfig] | None = None
    n_bundles: int | None = None
    bundle_dim: int | None = None
    n_latent_layers: int = 4
    encoder_hidden_dim: int | None = None
    decoder_hidden_dim: int | None = None
    lambda_l1: float = 0.01
    lambda_equiv: float = 0.0
    use_spectral_norm: bool = True

    def __post_init__(self) -> None:
        if self.bundles is None:
            if self.n_bundles is None or self.bundle_dim is None:
                msg = "Provide bundles or (n_bundles, bundle_dim)."
                raise ValueError(msg)
            if self.n_bundles <= 0 or self.bundle_dim <= 0:
                msg = "n_bundles and bundle_dim must be positive."
                raise ValueError(msg)
            self.bundles = [
                BundleConfig(name=f"bundle_{i}", dim=self.bundle_dim)
                for i in range(self.n_bundles)
            ]
        else:
            if len(self.bundles) == 0:
                msg = "bundles must be non-empty."
                raise ValueError(msg)
            for bundle in self.bundles:
                if bundle.dim <= 0:
                    msg = "bundle dimensions must be positive."
                    raise ValueError(msg)
            if self.n_bundles is not None and self.n_bundles != len(self.bundles):
                msg = "n_bundles does not match bundles length."
                raise ValueError(msg)
            if self.bundle_dim is not None:
                if any(bundle.dim != self.bundle_dim for bundle in self.bundles):
                    msg = "bundle_dim provided but bundles are heterogeneous."
                    raise ValueError(msg)
            self.n_bundles = len(self.bundles)
            if self.bundle_dim is None:
                dims = {bundle.dim for bundle in self.bundles}
                if len(dims) == 1:
                    self.bundle_dim = next(iter(dims))

        if self.encoder_hidden_dim is None:
            self.encoder_hidden_dim = self.total_latent_dim
        if self.decoder_hidden_dim is None:
            self.decoder_hidden_dim = self.total_latent_dim

    @property
    def total_latent_dim(self) -> int:
        if self.bundles is None:
            return 0
        return sum(bundle.dim for bundle in self.bundles)

    @property
    def bundle_dims(self) -> list[int]:
        if self.bundles is None:
            return []
        return [bundle.dim for bundle in self.bundles]


class SoftEquivariantLayer(nn.Module):
    """Soft equivariant latent dynamics layer."""

    def __init__(
        self,
        n_bundles: int | None = None,
        bundle_dim: int | None = None,
        bundle_dims: Sequence[int] | None = None,
        hidden_dim: int = 64,
        use_spectral_norm: bool = True,
        zero_self_mixing: bool = False,
    ) -> None:
        super().__init__()
        if bundle_dims is None:
            if n_bundles is None or bundle_dim is None:
                msg = "Provide bundle_dims or (n_bundles, bundle_dim)."
                raise ValueError(msg)
            bundle_dims = [bundle_dim] * n_bundles
        else:
            bundle_dims = list(bundle_dims)
            if n_bundles is not None and n_bundles != len(bundle_dims):
                msg = "n_bundles does not match bundle_dims length."
                raise ValueError(msg)
            if bundle_dim is not None and any(dim != bundle_dim for dim in bundle_dims):
                msg = "bundle_dim provided but bundle_dims are heterogeneous."
                raise ValueError(msg)

        if len(bundle_dims) == 0 or any(dim <= 0 for dim in bundle_dims):
            msg = "bundle_dims must contain positive dimensions."
            raise ValueError(msg)
        if hidden_dim <= 0:
            msg = "hidden_dim must be positive."
            raise ValueError(msg)

        self.bundle_dims = list(bundle_dims)
        self.n_bundles = len(self.bundle_dims)
        self.total_dim = sum(self.bundle_dims)
        self.hidden_dim = hidden_dim
        self.zero_self_mixing = zero_self_mixing
        dims = set(self.bundle_dims)
        self.bundle_dim = self.bundle_dims[0] if len(dims) == 1 else None

        LinearLayer = SpectralLinear if use_spectral_norm else nn.Linear
        self.norm_mlp = nn.Sequential(
            LinearLayer(self.n_bundles, hidden_dim, bias=True),
            nn.GELU(),
            LinearLayer(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
            LinearLayer(hidden_dim, self.n_bundles, bias=False),
        )

        if self.bundle_dim is not None:
            self.mixing_weights = nn.Parameter(
                torch.randn(self.n_bundles, self.n_bundles, self.bundle_dim, self.bundle_dim)
                * 0.01
            )
        else:
            self.mixing_weights = nn.ParameterList([
                nn.ParameterList([
                    nn.Parameter(torch.randn(self.bundle_dims[i], self.bundle_dims[j]) * 0.01)
                    for j in range(self.n_bundles)
                ])
                for i in range(self.n_bundles)
            ])
        if self.zero_self_mixing and self.bundle_dim is not None:
            mask = torch.ones(self.n_bundles, self.n_bundles)
            mask.fill_diagonal_(0.0)
            self.register_buffer("_mixing_mask", mask)
        else:
            self.register_buffer("_mixing_mask", None)

        self.gate_bias = nn.Parameter(torch.zeros(self.n_bundles))

    def _split_bundles(self, z: torch.Tensor) -> tuple[list[torch.Tensor], bool]:
        if z.dim() == 3:
            if z.shape[1] != self.n_bundles:
                msg = "Expected input shape [B, n_bundles, bundle_dim]."
                raise ValueError(msg)
            if any(dim != z.shape[2] for dim in self.bundle_dims):
                msg = "Bundle dimensions are heterogeneous; expected flattened input."
                raise ValueError(msg)
            return [z[:, i, :] for i in range(self.n_bundles)], True
        if z.dim() == 2:
            if z.shape[1] != self.total_dim:
                msg = "Expected input shape [B, sum(bundle_dims)]."
                raise ValueError(msg)
            bundles = []
            offset = 0
            for dim in self.bundle_dims:
                bundles.append(z[:, offset : offset + dim])
                offset += dim
            return bundles, False
        msg = "Expected input with shape [B, D] or [B, n_bundles, d_b]."
        raise ValueError(msg)

    def _bundle_view(self, z: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if self.bundle_dim is None:
            msg = "Bundle dimensions are heterogeneous; expected list-based access."
            raise ValueError(msg)
        if z.dim() == 3:
            if z.shape[1] != self.n_bundles or z.shape[2] != self.bundle_dim:
                msg = "Expected input shape [B, n_bundles, bundle_dim]."
                raise ValueError(msg)
            return z, True
        if z.dim() == 2:
            if z.shape[1] != self.total_dim:
                msg = "Expected input shape [B, sum(bundle_dims)]."
                raise ValueError(msg)
            return z.view(z.shape[0], self.n_bundles, self.bundle_dim), False
        msg = "Expected input with shape [B, D] or [B, n_bundles, d_b]."
        raise ValueError(msg)

    def split_bundles(self, z: torch.Tensor) -> list[torch.Tensor]:
        bundles, _ = self._split_bundles(z)
        return bundles

    def cat_bundles(self, bundles: list[torch.Tensor]) -> torch.Tensor:
        return torch.cat(bundles, dim=-1)

    def _cat_bundles(self, bundles: list[torch.Tensor], stacked: bool) -> torch.Tensor:
        if stacked:
            return torch.stack(bundles, dim=1)
        return torch.cat(bundles, dim=-1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if self.bundle_dim is not None:
            bundled, was_stacked = self._bundle_view(z)
            # Norm-based scaling is SO(d_b)-equivariant within each bundle.
            norms = torch.norm(bundled, dim=-1) + 1e-8
            scales = F.softplus(self.norm_mlp(norms))
            equivariant = bundled * scales.unsqueeze(-1)

            # Mixing injects cross-bundle texture interactions.
            weights = self.mixing_weights
            if self.zero_self_mixing:
                weights = weights * self._mixing_mask[:, :, None, None]
            mixing = torch.einsum("bjd,ijkd->bik", bundled, weights)
            gates = torch.sigmoid(self.gate_bias).view(1, -1, 1)
            combined = equivariant + gates * mixing
            z_out = bundled + combined  # Residual keeps dynamics near identity.
            if was_stacked:
                return z_out
            return z_out.reshape(z.shape[0], -1)

        bundles, stacked = self._split_bundles(z)

        # Per-bundle norms drive equivariant scaling for heterogeneous bundles.
        norms = torch.stack([torch.norm(v, dim=-1) + 1e-8 for v in bundles], dim=-1)
        scales = F.softplus(self.norm_mlp(norms))
        equivariant_outputs = [bundles[i] * scales[:, i : i + 1] for i in range(self.n_bundles)]

        # Cross-bundle mixing models texture coupling across gauge fibers.
        mixing_outputs = []
        for i in range(self.n_bundles):
            mixed = None
            for j in range(self.n_bundles):
                if self.zero_self_mixing and i == j:
                    continue
                term = F.linear(bundles[j], self.mixing_weights[i][j])
                mixed = term if mixed is None else mixed + term
            if mixed is None:
                mixed = torch.zeros_like(bundles[i])
            mixing_outputs.append(mixed)

        gates = torch.sigmoid(self.gate_bias)
        combined = [  # Gate controls how much mixing leaks into each bundle.
            equivariant_outputs[i] + gates[i] * mixing_outputs[i] for i in range(self.n_bundles)
        ]
        z_out = self._cat_bundles(combined, stacked)
        return z + z_out

    def l1_loss(self) -> torch.Tensor:
        if isinstance(self.mixing_weights, torch.Tensor):
            return torch.sum(torch.abs(self.mixing_weights))
        return sum(
            torch.sum(torch.abs(self.mixing_weights[i][j]))
            for i in range(self.n_bundles)
            for j in range(self.n_bundles)
        )

    def mixing_strength(self) -> float:
        if isinstance(self.mixing_weights, torch.Tensor):
            total_norm_sq = torch.sum(self.mixing_weights**2)
            return torch.sqrt(total_norm_sq).item()
        total_norm_sq = sum(
            torch.sum(self.mixing_weights[i][j] ** 2)
            for i in range(self.n_bundles)
            for j in range(self.n_bundles)
        )
        return torch.sqrt(total_norm_sq).item()


class _IdentityWilson(nn.Module):
    def __init__(self, d_k: int) -> None:
        super().__init__()
        self.d_k = d_k

    def forward(self, z_query: torch.Tensor, z_key: torch.Tensor) -> torch.Tensor:
        batch, n, _ = z_key.shape
        identity = torch.eye(self.d_k, device=z_key.device, dtype=z_key.dtype)
        return identity.expand(batch, n, self.d_k, self.d_k)


class CovariantAttentionLayer(nn.Module):
    """Covariant attention wrapper for bundle-wise world modeling."""

    def __init__(
        self,
        bundle_dims: Sequence[int],
        n_heads: int = 1,
        use_wilson_lines: bool = True,
    ) -> None:
        super().__init__()
        if len(bundle_dims) == 0 or any(dim <= 0 for dim in bundle_dims):
            msg = "bundle_dims must contain positive dimensions."
            raise ValueError(msg)
        if n_heads <= 0:
            msg = "n_heads must be positive."
            raise ValueError(msg)
        self.bundle_dims = list(bundle_dims)
        self.n_heads = n_heads
        self.use_wilson_lines = use_wilson_lines

        self.attention_heads = nn.ModuleList([])
        for dim in self.bundle_dims:
            if dim % n_heads != 0:
                msg = "bundle_dim must be divisible by n_heads."
                raise ValueError(msg)
            config = GeodesicConfig(d_model=dim, d_latent=dim, n_heads=n_heads)
            head = CovariantAttention(config)
            if not use_wilson_lines:
                head.wilson = _IdentityWilson(head.d_k)
            self.attention_heads.append(head)

    def _split_bundles(self, z: torch.Tensor) -> list[torch.Tensor]:
        if z.dim() != 2:
            msg = "Expected latent input shape [B, sum(bundle_dims)]."
            raise ValueError(msg)
        expected = sum(self.bundle_dims)
        if z.shape[1] != expected:
            msg = "Expected latent input shape [B, sum(bundle_dims)]."
            raise ValueError(msg)
        bundles = []
        offset = 0
        for dim in self.bundle_dims:
            bundles.append(z[:, offset : offset + dim])
            offset += dim
        return bundles

    def _split_context(self, context: torch.Tensor) -> list[torch.Tensor]:
        if context.dim() == 2:
            context = context.unsqueeze(1)
        if context.dim() != 3:
            msg = "Expected context shape [B, T, sum(bundle_dims)]."
            raise ValueError(msg)
        expected = sum(self.bundle_dims)
        if context.shape[2] != expected:
            msg = "Expected context shape [B, T, sum(bundle_dims)]."
            raise ValueError(msg)
        bundles = []
        offset = 0
        for dim in self.bundle_dims:
            bundles.append(context[:, :, offset : offset + dim])
            offset += dim
        return bundles

    def forward(self, z: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        bundles = self._split_bundles(z)
        if context is None:
            context_bundles = [v.unsqueeze(1) for v in bundles]
        else:
            context_bundles = self._split_context(context)

        attended = []
        for v, ctx, attn in zip(bundles, context_bundles, self.attention_heads):
            # Bundle-wise covariant attention preserves gauge structure per fiber.
            v_out, _ = attn(z_query=v, z_key=ctx, x_query=v, x_key=ctx, x_value=ctx)
            attended.append(v_out)

        return torch.cat(attended, dim=-1)


class UniversalGeometricNetwork(nn.Module):
    """Universal Geometric Network (UGN)."""

    def __init__(self, config: UGNConfig) -> None:
        super().__init__()
        self.config = config
        LinearLayer = SpectralLinear if config.use_spectral_norm else nn.Linear

        self.encoder = nn.Sequential(
            LinearLayer(config.input_dim, config.encoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.encoder_hidden_dim, config.total_latent_dim),
        )

        self.latent_layers = nn.ModuleList([
            SoftEquivariantLayer(
                bundle_dims=config.bundle_dims,
                hidden_dim=64,
                use_spectral_norm=config.use_spectral_norm,
            )
            for _ in range(config.n_latent_layers)
        ])

        self.decoder = nn.Sequential(
            LinearLayer(config.total_latent_dim, config.decoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.decoder_hidden_dim, config.decoder_hidden_dim),
            nn.GELU(),
            LinearLayer(config.decoder_hidden_dim, config.output_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def dynamics(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.latent_layers:
            z = layer(z)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode -> equivariant latent dynamics -> decode back to outputs.
        z = self.encode(x)
        z = self.dynamics(z)
        return self.decode(z)

    def regularization_loss(self) -> torch.Tensor:
        return sum(layer.l1_loss() for layer in self.latent_layers)

    def _split_latent(self, z: torch.Tensor) -> list[torch.Tensor]:
        bundles = []
        offset = 0
        for dim in self.config.bundle_dims:
            bundles.append(z[:, offset : offset + dim])
            offset += dim
        return bundles

    def equivariance_violation(
        self, z: torch.Tensor | None = None, n_samples: int = 16
    ) -> torch.Tensor:
        if z is None:
            z = torch.randn(1, self.config.total_latent_dim, device=next(self.parameters()).device)

        violations = []
        for _ in range(n_samples):
            rotations = []
            for d_b in self.config.bundle_dims:
                A = torch.randn(d_b, d_b, device=z.device)
                Q, _ = torch.linalg.qr(A)
                if torch.det(Q) < 0:
                    Q[:, 0] = -Q[:, 0]
                rotations.append(Q)

            bundles = self._split_latent(z)
            rotated = [b @ R.T for b, R in zip(bundles, rotations)]
            z_rot = torch.cat(rotated, dim=-1)

            z_out_1 = self.dynamics(z_rot)
            z_out_2 = self.dynamics(z)
            z_out_2_bundles = self._split_latent(z_out_2)
            z_out_2_rot = torch.cat([b @ R.T for b, R in zip(z_out_2_bundles, rotations)], dim=-1)

            violations.append(F.mse_loss(z_out_1, z_out_2_rot))

        return torch.stack(violations).mean()

    def total_loss(
        self,
        x: torch.Tensor,
        y_target: torch.Tensor,
        task_loss_fn: nn.Module = nn.MSELoss(),
    ) -> dict:
        y_pred = self.forward(x)
        loss_task = task_loss_fn(y_pred, y_target)
        loss_l1 = self.regularization_loss()
        total = loss_task + self.config.lambda_l1 * loss_l1

        loss_dict = {
            "task": loss_task,
            "l1": loss_l1,
            "total": total,
        }

        if self.config.lambda_equiv > 0:
            loss_equiv = self.equivariance_violation()
            loss_dict["equiv"] = loss_equiv
            loss_dict["total"] = total + self.config.lambda_equiv * loss_equiv

        return loss_dict

    def get_diagnostics(self) -> dict:
        mixing_strengths = [layer.mixing_strength() for layer in self.latent_layers]
        return {
            "mixing_strength": float(sum(mixing_strengths)),
            "mixing_strengths": mixing_strengths,
            "gate_values": [
                torch.sigmoid(layer.gate_bias).detach().cpu().numpy()
                for layer in self.latent_layers
            ],
        }


def log_sparsity_diagnostics(
    model: UniversalGeometricNetwork,
    step: int,
    logger: Callable[[dict], None] | None = None,
) -> dict:
    """Compute and optionally log sparsity statistics for mixing weights."""
    if logger is None:
        try:
            import wandb

            if wandb.run is not None:

                def logger(data):
                    return wandb.log(data, step=step)
        except ModuleNotFoundError:
            pass

    logs: dict = {}
    for i, layer in enumerate(model.latent_layers):
        epsilon = 1e-3
        total = 0
        near_zero = 0
        magnitudes = []

        if isinstance(layer.mixing_weights, torch.Tensor):
            weights = layer.mixing_weights
            total = weights.numel()
            near_zero = (weights.abs() < epsilon).sum().item()
            magnitudes.append(weights.abs().flatten())

            block_norms = torch.norm(weights, p="fro", dim=(2, 3))
            texture_zeros = []
            for bi in range(layer.n_bundles):
                for bj in range(layer.n_bundles):
                    if bi == bj:
                        continue
                    if block_norms[bi, bj].item() < epsilon:
                        texture_zeros.append((bi, bj))
        else:
            for bi in range(layer.n_bundles):
                for bj in range(layer.n_bundles):
                    block = layer.mixing_weights[bi][bj]
                    total += block.numel()
                    near_zero += (block.abs() < epsilon).sum().item()
                    magnitudes.append(block.abs().flatten())

            texture_zeros = []
            for bi in range(layer.n_bundles):
                for bj in range(layer.n_bundles):
                    if bi == bj:
                        continue
                    block_norm = torch.norm(layer.mixing_weights[bi][bj], p="fro").item()
                    if block_norm < epsilon:
                        texture_zeros.append((bi, bj))

        sparsity = near_zero / total if total > 0 else 0.0

        if magnitudes:
            all_magnitudes = torch.cat(magnitudes)
            top_k = min(10, all_magnitudes.numel())
            top_10 = torch.topk(all_magnitudes, k=top_k).values.tolist()
        else:
            top_10 = []

        logs.update({
            f"layer_{i}/sparsity": sparsity,
            f"layer_{i}/texture_zeros": len(texture_zeros),
            f"layer_{i}/top_10_magnitudes": top_10,
        })

    if logger is not None:
        logger(logs)

    return logs


def train_ugn(
    model: UniversalGeometricNetwork,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    n_epochs: int = 100,
    lr: float = 1e-3,
    task_loss_fn: Callable = nn.MSELoss(),
    use_adaptive_l1: bool = True,
    device: str = "cuda",
    equivariance_eval_batches: int = 1,
    equivariance_eval_samples: int = 4,
) -> dict:
    """Train a Universal Geometric Network with optional adaptive L1 scheduling."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    l1_scheduler = AdaptiveL1Scheduler() if use_adaptive_l1 else None

    history = {
        "train_loss": [],
        "val_loss": [],
        "l1_loss": [],
        "equiv_violation": [],
        "lambda_l1": [],
        "mixing_strength": [],
    }

    for epoch in range(n_epochs):
        model.train()
        epoch_losses = {"total": 0.0, "task": 0.0, "l1": 0.0, "equiv": 0.0}

        for x, y_target in train_loader:
            x = x.to(device)
            y_target = y_target.to(device)

            optimizer.zero_grad()
            losses = model.total_loss(x, y_target, task_loss_fn)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            for key in epoch_losses:
                if key in losses:
                    value = losses[key]
                    epoch_losses[key] += value if isinstance(value, float) else value.item()

        n_batches = max(len(train_loader), 1)
        for key in epoch_losses:
            epoch_losses[key] /= n_batches

        model.eval()
        val_loss = 0.0
        val_equiv_violation = 0.0
        equiv_batches = 0
        with torch.no_grad():
            for batch_idx, (x, y_target) in enumerate(val_loader):
                x = x.to(device)
                y_target = y_target.to(device)

                y_pred = model(x)
                val_loss += task_loss_fn(y_pred, y_target).item()

                if batch_idx < equivariance_eval_batches and equivariance_eval_samples > 0:
                    z = model.encode(x)
                    val_equiv_violation += model.equivariance_violation(
                        z, n_samples=equivariance_eval_samples
                    ).item()
                    equiv_batches += 1

        val_batches = max(len(val_loader), 1)
        val_loss /= val_batches
        if equiv_batches > 0:
            val_equiv_violation /= equiv_batches

        scheduler.step()

        if l1_scheduler is not None and equiv_batches > 0:
            model.config.lambda_l1 = l1_scheduler.step(val_equiv_violation)

        diagnostics = model.get_diagnostics()

        history["train_loss"].append(epoch_losses["task"])
        history["val_loss"].append(val_loss)
        history["l1_loss"].append(epoch_losses["l1"])
        history["equiv_violation"].append(val_equiv_violation)
        history["lambda_l1"].append(model.config.lambda_l1)
        history["mixing_strength"].append(diagnostics["mixing_strength"])

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}")
            print(f"  Train Loss: {epoch_losses['task']:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  L1 Loss: {epoch_losses['l1']:.4f}")
            print(f"  Equiv Violation: {val_equiv_violation:.4f}")
            print(f"  lambda_L1: {model.config.lambda_l1:.4f}")
            print(f"  Mixing Strength: {diagnostics['mixing_strength']:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
            print()

    return history


def compute_diagnostics(model: UniversalGeometricNetwork, x_sample: torch.Tensor) -> dict:
    """Compute diagnostics for spectral norms, gauge invariance, and mixing stats."""
    model.eval()
    diagnostics: dict = {}

    with torch.no_grad():
        spectral_norms: list[float] = []
        for module in model.modules():
            if isinstance(module, SpectralLinear):
                weight = module._spectral_normalized_weight(update_u=False)
                spectral_norms.append(torch.linalg.matrix_norm(weight, ord=2).item())
            elif isinstance(module, nn.Linear):
                spectral_norms.append(torch.linalg.matrix_norm(module.weight, ord=2).item())

        if spectral_norms:
            diagnostics["node_62_spectral_norm_max"] = max(spectral_norms)
            diagnostics["node_62_spectral_norm_mean"] = sum(spectral_norms) / len(spectral_norms)

        z = model.encode(x_sample)
        y_original = model.decode(z)

        bundles = model.latent_layers[0].split_bundles(z)
        rotated_bundles = []
        for i, v in enumerate(bundles):
            d_b = model.config.bundle_dims[i]
            Q, _ = torch.linalg.qr(torch.randn(d_b, d_b, device=v.device))
            rotated_bundles.append(v @ Q.T)
        z_rotated = model.latent_layers[0].cat_bundles(rotated_bundles)
        y_rotated = model.decode(z_rotated)

        diagnostics["node_67_gauge_invariance"] = torch.norm(y_rotated - y_original).item()
        diagnostics["l1_mixing_strength"] = model.regularization_loss().item()
        diagnostics["equivariance_violation"] = model.equivariance_violation(z).item()

        gate_values = []
        for layer in model.latent_layers:
            gate_values.append(torch.sigmoid(layer.gate_bias).cpu().numpy())
        diagnostics["mixing_gates"] = gate_values

        threshold = 1e-3
        n_zeros = 0
        n_total = 0
        for layer in model.latent_layers:
            if isinstance(layer.mixing_weights, torch.Tensor):
                weights = layer.mixing_weights
                n_total += weights.numel()
                n_zeros += (weights.abs() < threshold).sum().item()
            else:
                for i in range(layer.n_bundles):
                    for j in range(layer.n_bundles):
                        weights = layer.mixing_weights[i][j]
                        n_total += weights.numel()
                        n_zeros += (weights.abs() < threshold).sum().item()
        diagnostics["texture_zeros_count"] = n_zeros
        diagnostics["texture_zeros_fraction"] = (n_zeros / n_total) if n_total > 0 else 0.0

    return diagnostics
