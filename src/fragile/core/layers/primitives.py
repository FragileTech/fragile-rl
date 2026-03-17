from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
from torch import nn
import torch.nn.functional as F


class SpectralLinear(nn.Module):
    """Linear layer with spectral normalization (non-expansive).

    The weight is scaled by max(sigma_max, 1) to avoid expanding operators
    while keeping smaller operators unchanged.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        n_power_iterations: int = 3,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if n_power_iterations < 1:
            msg = "n_power_iterations must be >= 1."
            raise ValueError(msg)
        self.in_features = in_features
        self.out_features = out_features
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.register_buffer("_u", F.normalize(torch.randn(out_features), dim=0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def _spectral_normalized_weight(self, update_u: bool = True) -> torch.Tensor:
        weight = self.weight
        u = self._u
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.mv(weight, v), dim=0, eps=self.eps)
            if update_u:
                self._u.copy_(u)
        sigma = torch.dot(u, torch.mv(weight, v)).abs()
        return weight / sigma.clamp(min=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Enforce non-expansive linear map (Lipschitz <= 1) for stability/causality.
        weight = self._spectral_normalized_weight(update_u=self.training)
        return F.linear(x, weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, n_power_iterations={self.n_power_iterations}"
        )


class NormGate(nn.Module):
    """Radial (norm-gated) activation per bundle."""

    def __init__(
        self,
        bundle_size: int,
        n_bundles: int,
        gate_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        smooth_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if bundle_size <= 0:
            msg = "bundle_size must be positive."
            raise ValueError(msg)
        if n_bundles <= 0:
            msg = "n_bundles must be positive."
            raise ValueError(msg)
        if smooth_norm_eps < 0.0:
            msg = "smooth_norm_eps must be >= 0."
            raise ValueError(msg)

        self.bundle_size = bundle_size
        self.n_bundles = n_bundles
        self.smooth_norm_eps = smooth_norm_eps
        self.gate_fn = gate_fn or F.gelu

        self.norm_bias = nn.Parameter(torch.zeros(1, n_bundles, 1))

    def _bundle_view(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.dim() == 2:
            batch, dim = x.shape
            expected = self.n_bundles * self.bundle_size
            if dim != expected:
                raise ValueError(f"Expected input dim {expected}, got {dim}.")
            return x.reshape(batch, self.n_bundles, self.bundle_size), True
        if x.dim() == 3:
            if x.shape[1] != self.n_bundles or x.shape[2] != self.bundle_size:
                raise ValueError(
                    "Expected input shape [B, n_bundles, bundle_size] = "
                    f"[B, {self.n_bundles}, {self.bundle_size}], got {tuple(x.shape)}."
                )
            return x, False
        msg = "NormGate expects input with shape [B, D] or [B, n_bundles, d_b]."
        raise ValueError(msg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bundled, flatten = self._bundle_view(x)
        # Bundle energy acts as a gauge-invariant radial coordinate.
        if self.smooth_norm_eps > 0.0:
            energy = torch.sqrt((bundled**2).sum(dim=-1, keepdim=True) + self.smooth_norm_eps**2)
        else:
            energy = torch.norm(bundled, dim=-1, keepdim=True)
        # Gate depends on energy, not direction, preserving bundle isotropy.
        gate = self.gate_fn(energy + self.norm_bias)
        out = bundled * gate
        if flatten:
            return out.reshape(x.shape[0], self.n_bundles * self.bundle_size)
        return out


class NormGatedGELU(NormGate):
    """NormGate with GELU gating (default in the docs)."""

    def __init__(
        self,
        bundle_size: int,
        n_bundles: int,
        smooth_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__(
            bundle_size=bundle_size,
            n_bundles=n_bundles,
            gate_fn=F.gelu,
            smooth_norm_eps=smooth_norm_eps,
        )


class IsotropicBlock(nn.Module):
    """Gauge-covariant primitive: SpectralLinear -> NormGate."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bundle_size: int = 16,
        exact: bool = False,
        n_power_iterations: int = 3,
        eps: float = 1e-12,
        smooth_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if out_dim % bundle_size != 0:
            msg = "out_dim must be divisible by bundle_size."
            raise ValueError(msg)
        if exact and in_dim != out_dim:
            msg = "Exact mode requires in_dim == out_dim."
            raise ValueError(msg)
        if exact and in_dim % bundle_size != 0:
            msg = "Exact mode requires in_dim divisible by bundle_size."
            raise ValueError(msg)

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bundle_size = bundle_size
        self.n_bundles = out_dim // bundle_size
        self.exact = exact
        self.n_power_iterations = n_power_iterations
        self.eps = eps

        if exact:
            self.bundle_scales = nn.Parameter(torch.ones(self.n_bundles))
            self.input_proj = None
        else:
            self.block_weights = nn.Parameter(
                torch.randn(self.n_bundles, bundle_size, bundle_size) / math.sqrt(bundle_size)
            )
            self.input_proj = None
            if in_dim != out_dim:
                self.input_proj = SpectralLinear(
                    in_dim,
                    out_dim,
                    bias=False,
                    n_power_iterations=n_power_iterations,
                    eps=eps,
                )

            self.register_buffer(
                "_block_u",
                F.normalize(torch.randn(self.n_bundles, bundle_size), dim=-1),
            )

        self.norm_gate = NormGate(
            bundle_size=bundle_size,
            n_bundles=self.n_bundles,
            gate_fn=F.gelu,
            smooth_norm_eps=smooth_norm_eps,
        )

    def _spectral_normalize_block(self, weight: torch.Tensor, idx: int) -> torch.Tensor:
        u = self._block_u[idx]
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight.t(), u), dim=0, eps=self.eps)
                u = F.normalize(torch.mv(weight, v), dim=0, eps=self.eps)
            if self.training:
                self._block_u[idx].copy_(u)
        sigma = torch.dot(u, torch.mv(weight, v)).abs()
        return weight / sigma.clamp(min=1.0)

    def _spectral_normalize_block_bank(self, weight: torch.Tensor) -> torch.Tensor:
        u = self._block_u
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = torch.bmm(weight.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1)
                v = F.normalize(v, dim=-1, eps=self.eps)
                u = torch.bmm(weight, v.unsqueeze(-1)).squeeze(-1)
                u = F.normalize(u, dim=-1, eps=self.eps)
            if self.training:
                self._block_u.copy_(u)
        sigma = torch.einsum("bi,bij,bj->b", u, weight, v).abs()
        sigma = sigma.clamp(min=1.0).view(-1, 1, 1)
        return weight / sigma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        if self.exact:
            # Exact isotropy: per-bundle radial scaling without cross-bundle mixing.
            scales = self.bundle_scales.clamp(-1.0, 1.0)
            bundled = x.reshape(batch, self.n_bundles, self.bundle_size)
            bundled *= scales.view(1, -1, 1)
        else:
            if self.input_proj is not None:
                # Non-expansive projection to match bundle geometry.
                x = self.input_proj(x)
            bundled = x.reshape(batch, self.n_bundles, self.bundle_size)
            # Block-diagonal mixing within each bundle, spectrally normalized.
            weight = self._spectral_normalize_block_bank(self.block_weights)
            bundled = torch.einsum("bnd,ndk->bnk", bundled, weight)

        # Norm-gated activation keeps the nonlinearity gauge-covariant.
        gated = self.norm_gate(bundled)
        return gated.reshape(batch, self.n_bundles * self.bundle_size)

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, bundle_size={self.bundle_size}, "
            f"exact={self.exact}"
        )


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