from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .gauge import ConformalMetric, mobius_add
from .primitives import IsotropicBlock, SpectralLinear


def class_modulated_jump_rate(
    lambda_base: torch.Tensor,
    chart_to_class: torch.Tensor,
    gamma_sep: float = 5.0,
) -> torch.Tensor:
    """Compute class-consistent jump rates.

    Args:
        lambda_base: [N_c, N_c] base jump rates
        chart_to_class: [N_c, C] chart-to-class logits
        gamma_sep: separation strength

    Returns:
        lambda_sup: [N_c, N_c] modulated rates
    """
    dominant = torch.argmax(chart_to_class, dim=1)  # [N_c]
    diff = dominant.unsqueeze(1) != dominant.unsqueeze(0)  # [N_c, N_c]
    return lambda_base * torch.exp(-gamma_sep * diff.float())  # [N_c, N_c]


class InvariantChartClassifier(nn.Module):
    """Invariant chart-aware classifier using router weights and radial features."""

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        latent_dim: int,
        bundle_size: int | None = None,
        use_router_logits: bool = True,
        use_radial_logits: bool = True,
        smooth_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not use_router_logits and not use_radial_logits:
            msg = "At least one of use_router_logits or use_radial_logits must be True."
            raise ValueError(msg)
        if latent_dim <= 0:
            msg = "latent_dim must be positive."
            raise ValueError(msg)
        if bundle_size is not None and bundle_size <= 0:
            msg = "bundle_size must be positive."
            raise ValueError(msg)
        if bundle_size is not None and latent_dim % bundle_size != 0:
            msg = "latent_dim must be divisible by bundle_size."
            raise ValueError(msg)
        if smooth_norm_eps < 0.0:
            msg = "smooth_norm_eps must be >= 0."
            raise ValueError(msg)

        self.num_charts = num_charts
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.bundle_size = bundle_size
        self.smooth_norm_eps = smooth_norm_eps
        self.use_router_logits = use_router_logits
        self.use_radial_logits = use_radial_logits

        if use_router_logits:
            self.chart_logits = nn.Parameter(torch.zeros(num_charts, num_classes))
        else:
            self.register_parameter("chart_logits", None)

        self.n_radial = 1
        if use_radial_logits:
            if bundle_size is not None:
                self.n_radial = latent_dim // bundle_size
            self.radial_weight = nn.Parameter(
                torch.randn(num_charts, self.n_radial, num_classes) * 0.01
            )
            self.radial_bias = nn.Parameter(torch.zeros(num_charts, num_classes))
        else:
            self.register_parameter("radial_weight", None)
            self.register_parameter("radial_bias", None)

    def _radial_features(self, z_geo: torch.Tensor) -> torch.Tensor:
        batch = z_geo.shape[0]
        if self.bundle_size is None:
            energy = (z_geo**2).sum(dim=-1, keepdim=True)
            return torch.sqrt(energy + self.smooth_norm_eps**2)

        bundled = z_geo.reshape(batch, self.n_radial, self.bundle_size)
        energy = (bundled**2).sum(dim=-1)
        return torch.sqrt(energy + self.smooth_norm_eps**2)

    def forward(self, router_weights: torch.Tensor, z_geo: torch.Tensor) -> torch.Tensor:
        logits = 0.0
        if self.use_router_logits:
            # Chart routing provides a topological prior over classes.
            logits = router_weights @ self.chart_logits
        if self.use_radial_logits:
            # Radial energy features are gauge-invariant signals per chart.
            radial = self._radial_features(z_geo)
            radial_logits = torch.einsum(
                "bk,bm,kmc->bc", router_weights, radial, self.radial_weight
            )
            radial_logits += router_weights @ self.radial_bias
            logits += radial_logits
        return logits

    def extra_repr(self) -> str:
        return (
            f"num_charts={self.num_charts}, num_classes={self.num_classes}, "
            f"latent_dim={self.latent_dim}, bundle_size={self.bundle_size}, "
            f"radial_features={self.n_radial}"
        )


class SupervisedTopologyLoss(nn.Module):
    """Supervised topology loss enforcing purity, balance, and metric separation."""

    def __init__(
        self,
        num_charts: int,
        num_classes: int,
        lambda_purity: float = 0.1,
        lambda_balance: float = 0.01,
        lambda_metric: float = 0.01,
        margin: float = 1.0,
        temperature: float = 1.0,
        metric: ConformalMetric | None = None,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts
        self.num_classes = num_classes
        self.lambda_purity = lambda_purity
        self.lambda_balance = lambda_balance
        self.lambda_metric = lambda_metric
        self.margin = margin
        self.temperature = temperature
        self.metric = metric

        self.chart_to_class = nn.Parameter(torch.randn(num_charts, num_classes) * 0.01)

    @property
    def p_y_given_k(self) -> torch.Tensor:
        """Chart-to-class probabilities.

        Returns:
            p_y_given_k: [N_c, C] class probabilities per chart
        """
        return F.softmax(self.chart_to_class / self.temperature, dim=1)

    def forward(
        self,
        chart_assignments: torch.Tensor,
        class_labels: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute supervised topology losses.

        Args:
            chart_assignments: [B, N_c] soft routing weights
            class_labels: [B] ground-truth classes
            embeddings: [B, D] latent embeddings

        Returns:
            total_loss: [] total loss
            loss_dict: dict of individual losses
        """
        p_y_k = self.p_y_given_k  # [N_c, C]
        p_k = chart_assignments.mean(dim=0)  # [N_c]

        p_y_x = torch.matmul(chart_assignments, p_y_k)  # [B, C]
        # Routing loss encourages charts to specialize by class.
        loss_route = F.nll_loss(torch.log(p_y_x + 1e-8), class_labels)  # []

        # Purity loss penalizes high per-chart label entropy.
        entropy_per_chart = -(p_y_k * torch.log(p_y_k + 1e-8)).sum(dim=1)  # [N_c]
        loss_purity = (p_k * entropy_per_chart).sum()  # []

        # Balance loss encourages uniform chart usage.
        uniform = torch.ones_like(p_k) / self.num_charts  # [N_c]
        loss_balance = (p_k * (torch.log(p_k + 1e-8) - torch.log(uniform))).sum()  # []

        if self.lambda_metric > 0 and embeddings.shape[0] > 1:
            # Metric separation enforces class margins in the latent geometry.
            dists = torch.cdist(embeddings, embeddings)  # [B, B]
            if self.metric is not None:
                lambda_vec = self.metric.conformal_factor(embeddings).squeeze(-1)  # [B]
                lambda_ij = 0.5 * (lambda_vec.unsqueeze(0) + lambda_vec.unsqueeze(1))  # [B, B]
                dists *= lambda_ij
            match = (class_labels.unsqueeze(1) == class_labels.unsqueeze(0)).float()  # [B, B]
            diff = 1.0 - match  # [B, B]
            pos_loss = (match * dists).sum() / (match.sum() + 1e-8)  # []
            neg_loss = (diff * F.relu(self.margin - dists)).sum() / (diff.sum() + 1e-8)  # []
            loss_metric = pos_loss + neg_loss  # []
        else:
            loss_metric = torch.tensor(0.0, device=chart_assignments.device)  # []

        total_loss = (  # []
            loss_route
            + self.lambda_purity * loss_purity
            + self.lambda_balance * loss_balance
            + self.lambda_metric * loss_metric
        )

        return total_loss, {
            "loss_total": total_loss,
            "loss_route": loss_route,
            "loss_purity": loss_purity,
            "loss_balance": loss_balance,
            "loss_metric": loss_metric,
        }


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


def compute_topology_loss(
    weights: torch.Tensor,
    num_charts: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Topology loss: sharp routing + balanced chart usage."""
    entropy = -(weights * torch.log(weights + eps)).sum(dim=1)
    loss_entropy = entropy.mean()

    mean_usage = weights.mean(dim=0)
    target_usage = torch.full_like(mean_usage, 1.0 / num_charts)
    loss_balance = torch.norm(mean_usage - target_usage) ** 2

    return loss_entropy, loss_balance


def compute_separation_loss(
    chart_outputs: Iterable[torch.Tensor],
    weights: torch.Tensor,
    margin: float = 4.0,
    metric: ConformalMetric | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Separation loss: enforce margins between chart centers."""
    centers = []
    for idx, z_i in enumerate(chart_outputs):
        w_i = weights[:, idx : idx + 1]
        if w_i.sum() <= eps:
            continue
        center = (z_i * w_i).sum(dim=0) / (w_i.sum() + eps)
        centers.append(center)

    if len(centers) < 2:
        return torch.tensor(0.0, device=weights.device)

    loss_sep = torch.tensor(0.0, device=weights.device)
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = torch.norm(centers[i] - centers[j])
            if metric is not None:
                lambda_i = metric.conformal_factor(centers[i].unsqueeze(0)).squeeze()
                lambda_j = metric.conformal_factor(centers[j].unsqueeze(0)).squeeze()
                dist = dist * 0.5 * (lambda_i + lambda_j)
            loss_sep += torch.relu(margin - dist)

    return loss_sep


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


def compute_orthogonality_loss(
    modules: Iterable[nn.Module],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Compute orthogonality defect for SpectralLinear/IsotropicBlock modules."""
    device = None
    dtype = None
    for module in modules:
        for param in module.parameters(recurse=True):
            device = param.device
            dtype = param.dtype
            break
        if device is not None:
            break

    loss = torch.tensor(0.0, device=device, dtype=dtype)
    count = 0

    def orth_defect(weight: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(weight.shape[1], device=weight.device, dtype=weight.dtype)
        prod = weight.t() @ weight
        return ((prod - eye) ** 2).sum()

    for module in modules:
        for sub in module.modules():
            if isinstance(sub, SpectralLinear):
                # Measure orthogonality after spectral normalization.
                w = sub._spectral_normalized_weight(update_u=False)
                loss += orth_defect(w)
                count += 1
            elif isinstance(sub, IsotropicBlock) and not sub.exact:
                if sub.input_proj is not None:
                    w = sub.input_proj._spectral_normalized_weight(update_u=False)
                    loss += orth_defect(w)
                    count += 1
                if isinstance(sub.block_weights, torch.Tensor):
                    weights = sub._spectral_normalize_block_bank(sub.block_weights)
                    eye = torch.eye(
                        weights.shape[1],
                        device=weights.device,
                        dtype=weights.dtype,
                    )
                    prod = weights.transpose(1, 2) @ weights
                    loss += ((prod - eye) ** 2).sum()
                    count += weights.shape[0]
                else:
                    for idx, block in enumerate(sub.block_weights):
                        w = sub._spectral_normalize_block(block, idx)
                        loss += orth_defect(w)
                        count += 1

    if count == 0:
        return torch.tensor(0.0)
    return loss / (count + eps)
