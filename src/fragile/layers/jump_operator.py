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
        """Initialize the factorized jump operator.

        Sets up chart centers in the Poincare ball and learnable rotation
        matrices (initialized as identity) for gauge transformations.

        Args:
            num_charts: Number of charts in the atlas.
            latent_dim: Dimensionality of the latent (nuisance) space.
            curvature: Curvature parameter of the Poincare ball model.
            global_rank: Legacy argument, ignored. Kept for API compatibility.
            use_spectral: Legacy argument, ignored. Kept for API compatibility.
            use_mobius: Legacy argument, ignored. Kept for API compatibility.
        """
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
        """Project points to the interior of the Poincare ball.

        Clamps the norm of each point so that it does not exceed ``max_norm``,
        keeping all representations strictly inside the ball boundary.

        Args:
            z: Tensor of shape ``[..., D]`` containing points in the latent space.
            max_norm: Maximum allowed norm for the projected points. Points with
                a larger norm are rescaled to this value.

        Returns:
            torch.Tensor: Tensor of the same shape as ``z`` with all points
                having norm at most ``max_norm``.
        """
        norm = z.norm(dim=-1, keepdim=True)
        return torch.where(norm > max_norm, z * max_norm / norm, z)

    def lift_to_global(self, z_n: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
        """Lift local chart coordinates to the global frame via Mobius subtraction.

        Computes ``(-c_source) oplus z_n`` to translate points from a local
        chart centred at ``c_source`` back to the origin of the Poincare ball.

        Args:
            z_n: Tensor of shape ``[B, D]`` containing local nuisance coordinates.
            chart_idx: Tensor of shape ``[B]`` with integer indices selecting the
                source chart for each sample.

        Returns:
            torch.Tensor: Tensor of shape ``[B, D]`` with coordinates expressed
                in the global (origin-centred) frame.
        """
        c_source = self._project_to_ball(self.chart_centers[chart_idx])
        return mobius_add(-c_source, z_n, c=self.curvature)

    def project_from_global(self, h: torch.Tensor, chart_idx: torch.Tensor) -> torch.Tensor:
        """Project global coordinates into a local chart via Mobius addition.

        Computes ``c_target oplus h`` to translate points from the global
        (origin-centred) frame into the local chart centred at ``c_target``.

        Args:
            h: Tensor of shape ``[B, D]`` containing coordinates in the global frame.
            chart_idx: Tensor of shape ``[B]`` with integer indices selecting the
                target chart for each sample.

        Returns:
            torch.Tensor: Tensor of shape ``[B, D]`` with coordinates expressed
                in the selected local chart.
        """
        c_target = self._project_to_ball(self.chart_centers[chart_idx])
        return mobius_add(c_target, h, c=self.curvature)

    def forward(
        self,
        z_n: torch.Tensor,
        source_idx: torch.Tensor,
        target_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a chart transition using Mobius transformations.

        Implements the transition map:
        ``z_target = c_target oplus R_target R_source^T ((-c_source) oplus z_source)``

        The procedure is:
        1. Mobius-subtract the source centre to move to the origin.
        2. Apply a gauge rotation (source -> target) at the origin.
        3. Mobius-add the target centre to land in the target chart.

        Args:
            z_n: Tensor of shape ``[B, D]`` with source nuisance coordinates
                inside the Poincare ball.
            source_idx: Tensor of shape ``[B]`` with integer indices of the
                source charts.
            target_idx: Tensor of shape ``[B]`` with integer indices of the
                target charts.

        Returns:
            torch.Tensor: Tensor of shape ``[B, D]`` with the transformed
                nuisance coordinates in the target chart, projected to lie
                inside the Poincare ball.
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
        """Return the affine map ``(M, b)`` approximating a chart transition.

        Computes the linear map ``M = A_target @ B_source`` and the bias
        ``b = A_target @ c_source + d_target`` that together define the
        first-order affine approximation of the transition from ``source`` to
        ``target``.

        Args:
            source: Integer index of the source chart.
            target: Integer index of the target chart.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A 2-tuple where the first
                element is ``M`` of shape ``[D, D]`` (the linear map) and the
                second element is ``b`` of shape ``[D]`` (the bias vector).
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
    """Compute overlap consistency loss for jump operators.

    For every sample that belongs to at least two charts (determined by
    ``router_weights > overlap_threshold``), the function enumerates pairs of
    overlapping charts and penalises the mismatch between the observed
    coordinates in chart *j* and the coordinates predicted by applying the
    jump operator from chart *i* to chart *j*.

    When a ``metric`` is provided the squared error is weighted by the
    average conformal factor of the two chart representations, giving a
    geometry-aware loss.

    Args:
        z_n_by_chart: Tensor of shape ``[B, K, D]`` containing per-chart
            nuisance coordinates for each sample, where *K* is the number
            of charts and *D* is the latent dimension.
        router_weights: Tensor of shape ``[B, K]`` with soft routing
            weights indicating each sample's membership in each chart.
        jump_operator: The :class:`FactorizedJumpOperator` used to predict
            the transition between charts.
        overlap_threshold: Minimum router weight for a sample to be
            considered inside a chart.
        max_pairs_per_batch: Upper bound on the number of chart pairs
            evaluated per batch to limit computational cost.
        metric: Optional :class:`ConformalMetric` used to weight the
            consistency error by the local conformal factor. When ``None``
            a plain MSE is used.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A 2-tuple where the first
            element is the scalar mean consistency loss and the second is a
            diagnostics dictionary with keys ``"num_overlaps"``,
            ``"mean_error"``, and ``"points_in_overlap"``.
    """
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
