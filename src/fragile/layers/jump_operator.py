from __future__ import annotations

import torch
from torch import nn

from .gauge import ConformalMetric, mobius_add
from .primitives import SpectralLinear


class FactorizedJumpOperator(nn.Module):
    """Möbius-based jump operator between charts using O(n) hyperbolic geometry.

    Implements chart transitions via: z_target = c_target ⊕ R((-c_source) ⊕ z_source)
    where ⊕ is Möbius addition and R is an optional gauge rotation.
    """

    def __init__(
        self,
        num_charts: int,
        latent_dim: int,
        curvature: float = 1.0,
        # Legacy args (ignored, kept for API compat)
        global_rank: int | None = None,
        use_spectral: bool = True,
        use_mobius: bool = True,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.curvature = curvature

        # Chart centers in the Poincaré ball
        self.chart_centers = nn.Parameter(torch.randn(num_charts, latent_dim) * 0.1)

        # Learnable rotation matrices for gauge transformations (init as identity)
        self.rotations = nn.Parameter(
            torch.eye(latent_dim).unsqueeze(0).expand(num_charts, -1, -1).clone()
        )

    def _project_to_ball(self, z: torch.Tensor, max_norm: float = 0.99) -> torch.Tensor:
        """Project points to interior of the Poincaré ball."""
        norm = z.norm(dim=-1, keepdim=True)
        return torch.where(norm > max_norm, z * max_norm / norm, z)

    def lift_to_global(self, z_n: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
        """Lift local coordinates to global frame via Möbius subtraction."""
        c_source = self._project_to_ball(self.chart_centers[chart_idx])
        return mobius_add(-c_source, z_n, c=self.curvature)

    def project_from_global(self, h: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
        """Project global coordinates to local chart via Möbius addition."""
        c_target = self._project_to_ball(self.chart_centers[chart_idx])
        return mobius_add(c_target, h, c=self.curvature)

    def forward(
        self,
        z_n: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply chart transition using Möbius transformations.

        Implements: z_target = c_target ⊕ R((-c_source) ⊕ z_source)

        Args:
            z_n: [B, D] source nuisance coordinates
            source_idx: [B] source chart indices
            target_idx: [B] target chart indices

        Returns:
            z_out: [B, D] target nuisance coordinates
        """
        source_idx = source_idx.to(device=z_n.device, dtype=torch.long)
        target_idx = target_idx.to(device=z_n.device, dtype=torch.long)

        # Ensure input is inside ball
        z_n = self._project_to_ball(z_n)

        # 1. Move from source chart to origin (Möbius subtraction)
        c_source = self._project_to_ball(self.chart_centers[source_idx])
        z_global = mobius_add(-c_source, z_n, c=self.curvature)

        # 2. Apply gauge rotation at origin
        R_source = self.rotations[source_idx]
        R_target = self.rotations[target_idx]
        z_rotated = torch.einsum("bij,bj->bi", R_target, z_global)
        z_rotated = torch.einsum("bij,bj->bi", R_source.transpose(-1, -2), z_rotated)

        # 3. Move from origin to target chart (Möbius addition)
        c_target = self._project_to_ball(self.chart_centers[target_idx])
        z_out = mobius_add(c_target, z_rotated, c=self.curvature)

        return self._project_to_ball(z_out)

    def get_transition_matrix(self, source: int, target: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return affine map (M, b) for chart transition.

        Returns:
            M: [D, D] linear map
            b: [D] bias
        """
        if isinstance(self.encoders[source], SpectralLinear):
            b_src = self.encoders[source]._spectral_normalized_weight(update_u=False)
        else:
            b_src = self.encoders[source].weight

        if isinstance(self.decoders[target], SpectralLinear):
            a_tgt = self.decoders[target]._spectral_normalized_weight(update_u=False)
        else:
            a_tgt = self.decoders[target].weight

        M = a_tgt @ b_src  # [D, D]
        b = a_tgt @ self.c[source] + self.d[target]  # [D]
        return M, b


def compute_jump_consistency_loss(
    z_n_by_chart: torch.Tensor,
    router_weights: torch.Tensor,
    jump_operator: FactorizedJumpOperator,
    overlap_threshold: float = 0.1,
    max_pairs_per_batch: int = 1024,
    metric: ConformalMetric | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Overlap consistency loss for jump operators."""
    device = z_n_by_chart.device

    in_chart = router_weights > overlap_threshold
    overlap_mask = in_chart.sum(dim=1) >= 2

    if not overlap_mask.any():
        return torch.tensor(0.0, device=device), {"num_overlaps": 0}

    overlap_indices = overlap_mask.nonzero(as_tuple=True)[0]
    losses = []
    total_pairs = 0

    for b_idx in overlap_indices[:max_pairs_per_batch]:
        active = in_chart[b_idx].nonzero(as_tuple=True)[0]
        if active.numel() < 2:
            continue
        for i_idx, chart_i in enumerate(active[:-1]):
            for chart_j in active[i_idx + 1 :]:
                i = chart_i.item()
                j = chart_j.item()
                z_i = z_n_by_chart[b_idx, i]
                z_j = z_n_by_chart[b_idx, j]

                z_pred = jump_operator(
                    z_i.unsqueeze(0),
                    torch.tensor([i], device=device),
                    torch.tensor([j], device=device),
                ).squeeze(0)

                # Penalize mismatch between predicted and observed overlap coordinates.
                delta = z_j - z_pred
                if metric is not None:
                    lambda_i = metric.conformal_factor(z_i.unsqueeze(0)).squeeze()
                    lambda_j = metric.conformal_factor(z_j.unsqueeze(0)).squeeze()
                    weight = 0.5 * (lambda_i + lambda_j)
                    loss_ij = weight * (delta**2).sum()
                else:
                    loss_ij = (delta**2).mean()

                losses.append(loss_ij)
                total_pairs += 1

                if total_pairs >= max_pairs_per_batch:
                    break
            if total_pairs >= max_pairs_per_batch:
                break
        if total_pairs >= max_pairs_per_batch:
            break

    if not losses:
        return torch.tensor(0.0, device=device), {"num_overlaps": 0}

    loss = torch.stack(losses).mean()
    return loss, {
        "num_overlaps": float(total_pairs),
        "mean_error": loss.item(),
        "points_in_overlap": float(overlap_mask.sum().item()),
    }
