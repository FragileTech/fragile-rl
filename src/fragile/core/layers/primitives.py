from __future__ import annotations

import math
from typing import Callable

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
