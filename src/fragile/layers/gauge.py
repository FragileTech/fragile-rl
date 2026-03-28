from __future__ import annotations

import math

import torch
from torch import nn, Tensor


# =============================================================================
# Hyperbolic Geometry Primitives (Poincaré Ball Model)
# =============================================================================


def mobius_add(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Möbius addition in the Poincaré ball: x ⊕ y.

    This is the fundamental operation in hyperbolic geometry, replacing
    Euclidean addition. It's O(n) in the embedding dimension.

    Args:
        x: [..., D] first point in the Poincaré ball
        y: [..., D] second point in the Poincaré ball
        c: curvature (c=1 for unit ball)
        eps: numerical stability epsilon

    Returns:
        result: [..., D] Möbius sum x ⊕ y
    """
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True)
    xy = (x * y).sum(dim=-1, keepdim=True)

    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c**2 * x2 * y2
    return num / (denom + eps)


def mobius_scalar_mul(
    r: torch.Tensor, x: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Möbius scalar multiplication: r ⊗ x.

    Args:
        r: [...] or [..., 1] scalar multiplier
        x: [..., D] point in the Poincaré ball
        c: curvature
        eps: numerical stability epsilon

    Returns:
        result: [..., D] scaled point
    """
    if r.dim() < x.dim():
        r = r.unsqueeze(-1)
    sqrt_c = math.sqrt(c)
    x_norm = x.norm(dim=-1, keepdim=True).clamp(min=eps)
    return (1 / sqrt_c) * torch.tanh(r * torch.atanh(sqrt_c * x_norm + eps)) * (x / x_norm)


def hyperbolic_distance(
    x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Hyperbolic distance in the Poincaré ball.

    Args:
        x: [..., D] first point
        y: [..., D] second point
        c: curvature
        eps: numerical stability epsilon

    Returns:
        dist: [...] hyperbolic distance
    """
    sqrt_c = math.sqrt(c)
    diff = mobius_add(-x, y, c=c, eps=eps)
    diff_norm = diff.norm(dim=-1).clamp(min=eps, max=1 - eps)
    return (2 / sqrt_c) * torch.atanh(sqrt_c * diff_norm)


def exp_map_zero(v: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Exponential map from origin: exp_0(v).

    Maps a tangent vector at the origin to a point in the Poincaré ball.

    Args:
        v: [..., D] tangent vector at origin
        c: curvature
        eps: numerical stability epsilon

    Returns:
        point: [..., D] point in the Poincaré ball
    """
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    return torch.tanh(sqrt_c * v_norm) * (v / v_norm) / sqrt_c


def log_map_zero(y: torch.Tensor, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    """Logarithmic map to origin: log_0(y).

    Maps a point in the Poincaré ball to a tangent vector at the origin.

    Args:
        y: [..., D] point in the Poincaré ball
        c: curvature
        eps: numerical stability epsilon

    Returns:
        v: [..., D] tangent vector at origin
    """
    sqrt_c = math.sqrt(c)
    y_norm = y.norm(dim=-1, keepdim=True).clamp(min=eps, max=1 - eps)
    return torch.atanh(sqrt_c * y_norm) * (y / y_norm) / sqrt_c


def parallel_transport_zero(
    v: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Parallel transport from origin to y: P_{0→y}(v).

    In the Poincaré ball, parallel transport from the origin has the
    closed-form solution: P_{0→y}(v) = λ_y · v, where λ_y is the
    conformal factor at y.

    Args:
        v: [..., D] tangent vector at origin
        y: [..., D] destination point
        c: curvature
        eps: numerical stability epsilon

    Returns:
        v_transported: [..., D] tangent vector at y
    """
    y2 = (y**2).sum(dim=-1, keepdim=True)
    lambda_y = 2.0 / (1.0 - c * y2 + eps)
    return v / lambda_y


def parallel_transport(
    v: torch.Tensor, x: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Parallel transport from x to y: P_{x→y}(v).

    This uses the gyration-based formula for general parallel transport.

    Args:
        v: [..., D] tangent vector at x
        x: [..., D] source point
        y: [..., D] destination point
        c: curvature
        eps: numerical stability epsilon

    Returns:
        v_transported: [..., D] tangent vector at y
    """
    # Conformal factors
    x2 = (x**2).sum(dim=-1, keepdim=True)
    y2 = (y**2).sum(dim=-1, keepdim=True)
    lambda_x = 2.0 / (1.0 - c * x2 + eps)
    lambda_y = 2.0 / (1.0 - c * y2 + eps)

    # Scale by conformal factor ratio (preserves norm in tangent space)
    return v * (lambda_x / lambda_y)


def poincare_exp_map(
    z: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Exponential map at arbitrary base point z: exp_z(v).

    Maps a tangent vector v at z to a point in the Poincaré ball.
    Generalises ``exp_map_zero`` which only works from the origin.

    Args:
        z: [..., D] base point in the Poincaré ball
        v: [..., D] tangent vector at z
        c: curvature (c=1 for unit ball)
        eps: numerical stability epsilon

    Returns:
        point: [..., D] resulting point in the Poincaré ball
    """
    sqrt_c = math.sqrt(c)
    z_sq = (z**2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - c * z_sq).clamp(min=eps)

    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    # Hyperbolic norm of tangent vector at z
    v_hyp_norm = lambda_z * v_norm
    direction = v / v_norm
    w = torch.tanh(sqrt_c * v_hyp_norm / 2.0) * direction / sqrt_c
    return mobius_add(z, w, c=c, eps=eps)


def poincare_log_map(
    z: torch.Tensor, y: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Logarithmic map at arbitrary base point z: log_z(y).

    Maps a point y to the tangent vector at z. Inverse of poincare_exp_map.
    Formula: log_z(y) = (2 / (sqrt_c · λ_z)) · atanh(sqrt_c · ||(-z) ⊕ y||) · dir

    Args:
        z: [..., D] base point in the Poincaré ball
        y: [..., D] target point in the Poincaré ball
        c: curvature (c=1 for unit ball)
        eps: numerical stability epsilon

    Returns:
        v: [..., D] tangent vector at z such that exp_z(v) ≈ y
    """
    diff = mobius_add(-z, y, c=c, eps=eps)
    v0 = log_map_zero(diff, c=c, eps=eps)
    z_sq = (z**2).sum(dim=-1, keepdim=True)
    lambda_z = 2.0 / (1.0 - c * z_sq).clamp(min=eps)
    return 2.0 * v0 / lambda_z


def christoffel_contraction(
    z: torch.Tensor, v: torch.Tensor, c: float = 1.0, eps: float = 1e-5
) -> torch.Tensor:
    """Christoffel symbol contraction Γ^k_{ij} v^i v^j for the Poincaré ball.

    For the conformal metric g_{ij} = λ(z)^2 δ_{ij}, the Christoffel symbols
    give the geodesic acceleration correction:

        Γ^k v^i v^j = 4c(z·v)v / (1 - c|z|^2) - 2c|v|^2 z / (1 - c|z|^2)

    This is O(D) — no matrix inversions needed.

    Args:
        z: [..., D] position in the Poincaré ball
        v: [..., D] velocity (contravariant)
        c: curvature
        eps: numerical stability epsilon

    Returns:
        correction: [..., D] geodesic correction vector
    """
    z_sq = (z**2).sum(dim=-1, keepdim=True)
    zv = (z * v).sum(dim=-1, keepdim=True)
    v_sq = (v**2).sum(dim=-1, keepdim=True)
    denom = (1.0 - c * z_sq).clamp(min=eps)
    return 4.0 * c * zv * v / denom - 2.0 * c * v_sq * z / denom


def project_to_ball(
    z: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Project points to the interior of the Poincare ball.

    Clamps the norm of each point so that it does not exceed ``max_norm``,
    keeping all points strictly inside the ball for numerical stability.

    Args:
        z: [..., D] points that may lie outside or on the boundary of the ball.
        max_norm: Maximum allowed norm after projection.
        eps: Small constant to avoid division by zero when computing norms.

    Returns:
        Projected points with shape [..., D] whose norms are at most ``max_norm``.
    """
    norm = z.norm(dim=-1, keepdim=True).clamp(min=eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def smooth_tangent_to_ball(
    v: torch.Tensor,
    max_norm: float = 0.99,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Map unconstrained tangent vectors smoothly into the Poincare ball.

    Applies a soft saturation (tanh) to the tangent vector norm before
    mapping through the exponential map at the origin, ensuring the
    resulting point stays well inside the ball boundary.

    Args:
        v: [..., D] unconstrained tangent vectors in Euclidean space.
        max_norm: Maximum norm of the image inside the ball.
        eps: Small constant to avoid division by zero when computing norms.

    Returns:
        Points with shape [..., D] inside the Poincare ball with norm < ``max_norm``.
    """
    tangent_cap = math.atanh(max_norm)
    v_norm = v.norm(dim=-1, keepdim=True).clamp(min=eps)
    tangent_norm = tangent_cap * torch.tanh(v_norm / tangent_cap)
    tangent = tangent_norm * (v / v_norm)
    return exp_map_zero(tangent, eps=eps)


def poincare_weighted_mean(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Approximate hyperbolic barycenter using log/exp maps at the origin.

    Computes a weighted Frechet mean approximation by lifting points to the
    tangent space at the origin, computing the weighted average there, and
    mapping back to the ball.

    Args:
        points: [N, D] or [B, N, D] points in the Poincare ball. When 2-D the
            points are broadcast across the batch dimension of ``weights``.
        weights: [B, N] non-negative weights for each point.
        eps: Small constant to avoid division by zero during weight normalisation.

    Returns:
        Weighted barycenter with shape [B, D] in the Poincare ball.
    """
    if points.dim() == 2:
        points = points.unsqueeze(0).expand(weights.shape[0], -1, -1)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=1, keepdim=True).clamp(min=eps)
    tangent = log_map_zero(points)
    mean_tan = (w * tangent).sum(dim=1) / w_sum.squeeze(1)
    return exp_map_zero(mean_tan)


def poincare_weighted_mean_per_chart(
    points: torch.Tensor,
    weights: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-chart hyperbolic barycenter for codebook soft assignment.

    Similar to :func:`poincare_weighted_mean` but operates over an extra
    chart (codebook) dimension, computing one barycenter per chart entry.

    Args:
        points: [N, C, D] codebook points in the Poincare ball, where N is the
            number of source points and C is the number of charts.
        weights: [B, N, C] soft-assignment weights for each (point, chart) pair.
        eps: Small constant to avoid division by zero during weight normalisation.

    Returns:
        Per-chart barycenters with shape [B, C, D] in the Poincare ball.
    """
    points_exp = points.unsqueeze(0).expand(weights.shape[0], -1, -1, -1)
    tangent = log_map_zero(points_exp)
    w = weights.unsqueeze(-1)
    w_sum = w.sum(dim=2, keepdim=True).clamp(min=eps)
    mean_tan = (w * tangent).sum(dim=2) / w_sum.squeeze(2)
    return exp_map_zero(mean_tan)


def poincare_temperature(
    z: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
) -> torch.Tensor:
    """Compute position-dependent temperature for the Poincare ball.

    The temperature decreases as points approach the boundary of the ball,
    making attention scores sharper near the edge where the metric blows up.

    Args:
        z: [..., D] positions in the Poincare ball.
        key_dim: Dimensionality of the key vectors used for scaling.
        tau_min: Minimum allowed temperature to prevent division by zero.
        tau_denom_min: Minimum clamp for the denominator ``(1 - |z|^2)`` to
            ensure numerical stability near the boundary.

    Returns:
        Temperature values with shape [...], clamped to be at least ``tau_min``.
    """
    r2 = (z**2).sum(dim=-1)
    denom = (1.0 - r2).clamp(min=tau_denom_min)
    tau = math.sqrt(key_dim) * denom / 2.0
    return tau.clamp(min=tau_min)


def poincare_hyperbolic_score(
    z: torch.Tensor,
    centers: torch.Tensor,
    key_dim: int,
    tau_min: float,
    tau_denom_min: float,
    eps: float,
) -> torch.Tensor:
    """Compute hyperbolic distance-based scores with metric temperature.

    For each input point, computes the negative hyperbolic distance to every
    center, scaled by a position-dependent temperature. Higher scores indicate
    closer proximity in hyperbolic space.

    Args:
        z: [B, D] query points in the Poincare ball.
        centers: [N_c, D] center (codebook) points in the Poincare ball.
        key_dim: Key dimensionality used for temperature scaling.
        tau_min: Minimum temperature clamp passed to :func:`poincare_temperature`.
        tau_denom_min: Denominator clamp passed to :func:`poincare_temperature`.
        eps: Small constant for numerical stability in distance computation.

    Returns:
        Scores with shape [B, N_c] where higher values mean closer distance.
    """
    z_exp = z.unsqueeze(1)  # [B, 1, D]
    c_exp = centers.unsqueeze(0)  # [1, N_c, D]
    diff = z_exp - c_exp
    dist_sq = (diff**2).sum(dim=-1)  # [B, N_c]
    z_sq = (z**2).sum(dim=-1, keepdim=True)  # [B, 1]
    c_sq = (centers**2).sum(dim=-1).unsqueeze(0)  # [1, N_c]
    denom = (1.0 - z_sq) * (1.0 - c_sq)
    arg = 1.0 + 2.0 * dist_sq / (denom + eps)
    dist = torch.acosh(arg.clamp(min=1.0 + eps))  # [B, N_c]
    tau = poincare_temperature(z, key_dim, tau_min, tau_denom_min)
    return -dist / tau.unsqueeze(1)


def as_tangent(z: Tensor, assume_tangent: bool) -> Tensor:
    """Return tangent vectors, mapping from the ball if needed.

    When ``assume_tangent`` is True the input is returned as-is. Otherwise
    the input is treated as a point in the Poincare ball: it is first
    projected inside the ball and then mapped to the tangent space at the
    origin via the logarithmic map.

    Args:
        z: [..., D] either tangent vectors or points in the Poincare ball.
        assume_tangent: If True, ``z`` is already in tangent space and is
            returned unchanged. If False, ``z`` is mapped from the ball.

    Returns:
        Tangent vectors with shape [..., D] at the origin.
    """
    if assume_tangent:
        return z
    return log_map_zero(project_to_ball(z))


class ConformalMetric(nn.Module):
    """Poincare ball/disk conformal metric utilities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        """Initialise the conformal metric module.

        Args:
            epsilon: Small constant for numerical stability when computing
                the conformal factor near the ball boundary.
        """
        super().__init__()
        self.epsilon = epsilon

    def conformal_factor(self, z: torch.Tensor) -> torch.Tensor:
        """Compute conformal factor lambda(z) = 2 / (1 - |z|^2).

        Args:
            z: [B, d] positions inside the Poincare ball.

        Returns:
            Conformal factors with shape [B, 1].
        """
        r_sq = (z**2).sum(dim=-1, keepdim=True)
        r_sq = torch.clamp(r_sq, max=1.0 - self.epsilon)
        return 2.0 / (1.0 - r_sq + self.epsilon)

    def metric(self, z: torch.Tensor) -> torch.Tensor:
        """Compute metric tensor G_ij(z) = lambda(z)^2 * I.

        Args:
            z: [B, d] positions inside the Poincare ball.

        Returns:
            Metric tensors with shape [B, d, d].
        """
        _, d = z.shape
        lambda_sq = self.conformal_factor(z) ** 2
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq.unsqueeze(-1) * eye

    def metric_inv(self, z: torch.Tensor) -> torch.Tensor:
        """Compute inverse metric tensor G^{ij}(z) = I / lambda(z)^2.

        Args:
            z: [B, d] positions inside the Poincare ball.

        Returns:
            Inverse metric tensors with shape [B, d, d].
        """
        _, d = z.shape
        lambda_sq_inv = 1.0 / (self.conformal_factor(z) ** 2 + self.epsilon)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq_inv.unsqueeze(-1) * eye

    def temperature(self, z: torch.Tensor, d_k: int) -> torch.Tensor:
        """Compute position-dependent attention temperature.

        Temperature is ``sqrt(d_k) / lambda(z)``, decreasing near the ball
        boundary where the conformal factor diverges.

        Args:
            z: [B, d] positions inside the Poincare ball.
            d_k: Key dimensionality used for scaling.

        Returns:
            Temperature values with shape [B, 1].
        """
        lambda_z = self.conformal_factor(z)
        return math.sqrt(d_k) / lambda_z


class RiskAdaptiveConformalMetric(ConformalMetric):
    """Conformal metric adapted by risk tensor: lambda(z,T) = lambda_0(z) * (1 + alpha * ||T||_F)."""

    def __init__(self, risk_coupling_alpha: float = 0.1, epsilon: float = 1e-6) -> None:
        """Initialise the risk-adaptive conformal metric.

        Args:
            risk_coupling_alpha: Coupling strength that controls how much the
                Frobenius norm of the risk tensor inflates the conformal factor.
            epsilon: Small constant for numerical stability when computing
                the conformal factor near the ball boundary.
        """
        super().__init__(epsilon=epsilon)
        self.risk_coupling_alpha = risk_coupling_alpha

    def _risk_scale(self, risk_tensor: torch.Tensor | None) -> torch.Tensor:
        """Compute multiplicative risk scaling factor.

        Args:
            risk_tensor: [B, D, D] symmetric risk tensor, or None.

        Returns:
            scale: [B, 1] risk scaling factor >= 1.
        """
        if risk_tensor is None:
            return 1.0
        # Frobenius norm of risk tensor
        t_norm = torch.linalg.norm(risk_tensor, ord="fro", dim=(-2, -1), keepdim=False)  # [B]
        return (1.0 + self.risk_coupling_alpha * t_norm).unsqueeze(-1)  # [B, 1]

    def conformal_factor(
        self, z: torch.Tensor, risk_tensor: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute risk-adapted conformal factor lambda(z) * (1 + alpha * ||T||_F).

        Multiplies the base Poincare conformal factor by a risk-dependent
        scaling term so that regions of higher risk have a larger effective
        curvature.

        Args:
            z: [B, d] positions inside the Poincare ball.
            risk_tensor: [B, D, D] symmetric risk tensor, or None. When None
                the base conformal factor is returned unchanged.

        Returns:
            Risk-scaled conformal factors with shape [B, 1].
        """
        lam = super().conformal_factor(z)  # [B, 1]
        return lam * self._risk_scale(risk_tensor)

    def metric(self, z: torch.Tensor, risk_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """Compute risk-adapted metric tensor G_ij(z, T) = lambda(z, T)^2 * I.

        Args:
            z: [B, d] positions inside the Poincare ball.
            risk_tensor: [B, D, D] symmetric risk tensor, or None. When None
                the metric reduces to the base conformal metric.

        Returns:
            Metric tensors with shape [B, d, d].
        """
        _, d = z.shape
        lambda_sq = self.conformal_factor(z, risk_tensor) ** 2
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq.unsqueeze(-1) * eye

    def metric_inv(self, z: torch.Tensor, risk_tensor: torch.Tensor | None = None) -> torch.Tensor:
        """Compute inverse of the risk-adapted metric tensor G^{ij}(z, T).

        Args:
            z: [B, d] positions inside the Poincare ball.
            risk_tensor: [B, D, D] symmetric risk tensor, or None. When None
                the inverse metric reduces to the base conformal inverse.

        Returns:
            Inverse metric tensors with shape [B, d, d].
        """
        _, d = z.shape
        lambda_sq_inv = 1.0 / (self.conformal_factor(z, risk_tensor) ** 2 + self.epsilon)
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        return lambda_sq_inv.unsqueeze(-1) * eye

    def temperature(
        self, z: torch.Tensor, d_k: int, risk_tensor: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute risk-adapted position-dependent attention temperature.

        Temperature is inversely proportional to the risk-adapted conformal
        factor, so regions of higher risk produce sharper attention.

        Args:
            z: [B, d] positions inside the Poincare ball.
            d_k: Key dimensionality used for scaling.
            risk_tensor: [B, D, D] symmetric risk tensor, or None. When None
                the temperature is identical to the base class result.

        Returns:
            Temperature values with shape [B, 1].
        """
        lambda_z = self.conformal_factor(z, risk_tensor)
        return math.sqrt(d_k) / lambda_z
