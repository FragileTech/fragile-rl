"""World model / dynamics losses on the Poincare ball."""

from __future__ import annotations

from typing import Callable, TYPE_CHECKING

import torch
import torch.nn.functional as F

from fragile.layers.gauge import (
    hyperbolic_distance,
    poincare_exp_map,
    poincare_log_map,
)


if TYPE_CHECKING:
    from fragile.vla.config import VLAConfig


# ---------------------------------------------------------------------------
# Dynamics losses (from vla/losses.py)
# ---------------------------------------------------------------------------


def compute_dynamics_geodesic_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
) -> torch.Tensor:
    """Mean geodesic distance between predicted and target latent trajectories.

    Args:
        z_pred: [B, H, D] predicted positions.
        z_target: [B, H, D] target positions.

    Returns:
        Scalar loss (mean hyperbolic distance across batch and horizon).
    """
    B, H, D = z_pred.shape
    pred_flat = z_pred.reshape(B * H, D)
    tgt_flat = z_target.reshape(B * H, D)
    return hyperbolic_distance(pred_flat, tgt_flat).mean()


def compute_dynamics_chart_loss(
    chart_logits: torch.Tensor,
    target_charts: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for chart transition prediction.

    Args:
        chart_logits: [B, H, K] predicted chart logits.
        target_charts: [B, H] ground-truth chart indices.

    Returns:
        Scalar cross-entropy loss.
    """
    B, H, K = chart_logits.shape
    logits_flat = chart_logits.reshape(B * H, K)
    targets_flat = target_charts.reshape(B * H)
    return F.cross_entropy(logits_flat, targets_flat)


def compute_momentum_regularization(
    momenta: torch.Tensor,
    z_trajectory: torch.Tensor,
) -> torch.Tensor:
    """Metric-aware momentum penalty: mean kinetic energy 1/2 p^T G^{-1} p.

    Args:
        momenta: [B, H, D] momentum vectors.
        z_trajectory: [B, H, D] latent positions.

    Returns:
        Scalar regularization loss.
    """
    r_sq = (z_trajectory**2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta**2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    kinetic = 0.5 * g_inv_factor * p_sq
    return kinetic.mean()


def compute_energy_conservation_loss(
    phi_eff: torch.Tensor,
    momenta: torch.Tensor,
    z_trajectory: torch.Tensor,
) -> torch.Tensor:
    """Penalise Hamiltonian drift across horizon steps.

    Computes H = Φ_eff + ½ p^T G^{-1} p at each step and penalises the
    variance of H across the horizon (a perfectly symplectic integrator
    would keep H constant).

    Args:
        phi_eff: [B, H, 1] effective potential values.
        momenta: [B, H, D] momentum vectors.
        z_trajectory: [B, H, D] latent positions.

    Returns:
        Scalar loss (variance of H across horizon).
    """
    # Kinetic energy: ½ |p|^2 * ((1 - |z|^2) / 2)^2  (diagonal inverse metric)
    r_sq = (z_trajectory**2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta**2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    kinetic = 0.5 * g_inv_factor * p_sq  # [B, H, 1]

    H_total = phi_eff + kinetic  # [B, H, 1]
    # Variance of Hamiltonian across horizon (per batch element, then mean)
    return H_total.squeeze(-1).var(dim=-1).mean()


def compute_hodge_consistency_loss(
    hodge_harmonic_forces: torch.Tensor,
) -> torch.Tensor:
    """Penalize the harmonic residual in the Hodge decomposition.

    A well-structured model should explain all forces through either
    the conservative potential or the solenoidal curl field. The harmonic
    residual should be small.

    Args:
        hodge_harmonic_forces: [B, H, D] harmonic force residuals.

    Returns:
        Scalar L2 loss on harmonic forces.
    """
    return (hodge_harmonic_forces**2).mean()


# ---------------------------------------------------------------------------
# Screened Poisson critic (PDE residual loss)
# ---------------------------------------------------------------------------


def hyperbolic_laplacian(
    V_func: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
    n_probes: int = 3,
    eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Compute Laplace-Beltrami operator Delta_G V on the Poincare ball.

    Uses **batched finite-difference Hutchinson trace estimation** with a
    single forward pass through V_func for all perturbation points.
    The Poincare correction term ``z . grad V`` is also computed via
    finite differences (no autograd passes required).

    .. math::
        \Delta_G f = \frac{1}{\lambda^2}
            \bigl[\Delta_E f + (D-2)\,\lambda\,(z \cdot \nabla f)\bigr]

    where :math:`\lambda(z) = 2/(1-|z|^2)`.

    **Hutchinson estimator** (Rademacher probes v_i):

    .. math::
        \mathrm{Tr}(H) \approx \frac{1}{k}\sum_{i=1}^{k}
            \frac{V(z+\varepsilon v_i) - 2\,V(z) + V(z-\varepsilon v_i)}
                 {\varepsilon^2}

    **Directional derivative** (finite-difference):

    .. math::
        z \cdot \nabla V \approx \|z\| \,
            \frac{V(z+\varepsilon\hat{z}) - V(z-\varepsilon\hat{z})}
                 {2\varepsilon}

    All (2k+3)*N perturbation points are evaluated in a single batched
    V_func call. Gradients flow through the network parameters via the
    forward pass (no ``create_graph=True`` needed).

    Args:
        V_func: Callable mapping z [B, D] -> V [B, 1] (differentiable).
        z: [B, D] positions inside the Poincare ball.
        n_probes: Number of Rademacher probe vectors (default 3).
        eps: Finite-difference step size (default 1e-3).

    Returns:
        lap_G: [B, 1] Laplace-Beltrami of V at each position.
        V_center: [B, 1] V evaluated at z (reusable by caller).
    """
    N, D = z.shape
    device = z.device
    dtype = z.dtype
    z_det = z.detach()

    # --- Probe directions ---
    probes = torch.randint(0, 2, (n_probes, N, D), device=device, dtype=dtype) * 2 - 1

    # Radial unit direction: z_hat = z / ||z||  (clamped for origin safety)
    z_norm = z_det.norm(dim=-1, keepdim=True)  # [N, 1]
    z_hat = z_det / z_norm.clamp(min=1e-8)  # [N, D]

    # --- Build all perturbation points in one stack ---
    # Layout: [center, +v1, -v1, +v2, -v2, ..., +z_hat, -z_hat]
    points = [z_det]
    for i in range(n_probes):
        points.append(z_det + eps * probes[i])
        points.append(z_det - eps * probes[i])
    points.append(z_det + eps * z_hat)
    points.append(z_det - eps * z_hat)

    z_all = torch.cat(points, dim=0)  # [(2k+3)*N, D]

    # --- Single batched forward pass ---
    V_all = V_func(z_all)  # [(2k+3)*N, 1]

    # --- Split results ---
    n_groups = 2 * n_probes + 3
    V_split = V_all.reshape(n_groups, N, 1)

    V_center = V_split[0]  # [N, 1]

    # --- Hutchinson trace estimate (vectorised over probes) ---
    V_plus = V_split[1 : 1 + 2 * n_probes : 2]  # [k, N, 1]
    V_minus = V_split[2 : 2 + 2 * n_probes : 2]  # [k, N, 1]
    trace_terms = V_plus - 2 * V_center.unsqueeze(0) + V_minus  # [k, N, 1]
    laplacian_E = trace_terms.sum(dim=0) / (n_probes * eps**2)  # [N, 1]

    # --- Directional derivative: z · ∇V ≈ ||z|| * (V(z+εẑ) - V(z-εẑ)) / (2ε) ---
    V_zhat_plus = V_split[-2]  # [N, 1]
    V_zhat_minus = V_split[-1]  # [N, 1]
    z_dot_grad = z_norm * (V_zhat_plus - V_zhat_minus) / (2 * eps)  # [N, 1]

    # --- Poincare ball correction ---
    r_sq = (z_det**2).sum(dim=-1, keepdim=True)  # [N, 1]
    one_minus_r_sq = (1.0 - r_sq).clamp(min=1e-6)
    lambda_z = 2.0 / one_minus_r_sq  # [N, 1]
    inv_lambda_sq = (one_minus_r_sq / 2.0) ** 2  # [N, 1]

    lap_G = inv_lambda_sq * (laplacian_E + (D - 2) * lambda_z * z_dot_grad)

    return lap_G, V_center  # [N, 1], [N, 1]


def compute_screened_poisson_loss(
    value_net: torch.nn.Module,
    z_trajectory: torch.Tensor,
    z_targets: torch.Tensor | None,
    router_weights: torch.Tensor,
    reward_density: torch.Tensor | None = None,
    kappa: float = 1.0,
    max_samples: int = 64,
) -> torch.Tensor:
    """PDE residual loss: ||(-Delta_G + kappa^2) V - rho_r||^2.

    Enforces that the critic V approximately solves the screened Poisson
    equation on the Poincare ball, with reward density rho_r approximated
    by the geodesic miss-distance to target.

    Args:
        value_net: Scalar value field exposing ``task_value(z, rw)``.
        z_trajectory: [B, H, D] predicted positions.
        z_targets: [B, H, D] target positions. Used only when
            ``reward_density`` is not provided.
        router_weights: [B, K] chart routing weights.
        reward_density: [B, H] or [B, H, 1] scalar source term. When given,
            it is used directly as ``rho_r`` instead of geodesic miss-distance.
        kappa: Screening mass (controls decay length).
        max_samples: Max z samples to evaluate (limits second-order grad cost).

    Returns:
        Scalar PDE residual loss.
    """
    B, H, D = z_trajectory.shape

    # Flatten and subsample for efficiency
    z_flat = z_trajectory.reshape(B * H, D)
    n = z_flat.shape[0]
    z_tgt_flat = None
    rho_r = None
    if reward_density is not None:
        rho_r = reward_density.reshape(B * H, -1)
        if rho_r.shape[1] != 1:
            msg = "reward_density must have shape [B, H] or [B, H, 1]."
            raise ValueError(msg)
    elif z_targets is not None:
        z_tgt_flat = z_targets.reshape(B * H, D)
    else:
        msg = "compute_screened_poisson_loss requires either z_targets or reward_density."
        raise ValueError(msg)

    if router_weights.ndim == 3:
        rw_flat = router_weights.reshape(B * H, router_weights.shape[-1])
    elif router_weights.ndim == 2:
        if router_weights.shape[0] == B * H:
            rw_flat = router_weights
        elif router_weights.shape[0] == B:
            rw_flat = router_weights.repeat_interleave(H, dim=0)
        elif router_weights.shape[0] == 1:
            rw_flat = router_weights.expand(B * H, -1)
        else:
            msg = "router_weights has incompatible leading dimension for z_trajectory."
            raise ValueError(msg)
    else:
        msg = "router_weights must have shape [B, K] or [B, H, K]."
        raise ValueError(msg)

    if n > max_samples:
        idx = torch.randperm(n, device=z_flat.device)[:max_samples]
        z_flat = z_flat[idx]
        rw_flat = rw_flat[idx]
        if z_tgt_flat is not None:
            z_tgt_flat = z_tgt_flat[idx]
        if rho_r is not None:
            rho_r = rho_r[idx]

    if rho_r is None:
        assert z_tgt_flat is not None
        rho_r = hyperbolic_distance(z_flat.detach(), z_tgt_flat.detach()).unsqueeze(-1)  # [N, 1]
    else:
        rho_r = rho_r.detach()

    # Define V as a function of z using the local chart mixture for each sample.
    def V_func(z_in: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar value field at given positions.

        Wraps ``value_net.task_value`` and broadcasts the local router
        weights to match the (possibly expanded) collocation batch produced
        by the Hutchinson estimator inside ``hyperbolic_laplacian``.

        Args:
            z_in: [N, D] positions inside the Poincare ball, where N is
                either equal to or a multiple of the flattened sample count.

        Returns:
            [N, 1] scalar value predictions at each position.
        """
        n_in = z_in.shape[0]
        if n_in == rw_flat.shape[0]:
            rw_in = rw_flat
        elif n_in % rw_flat.shape[0] == 0:
            rw_in = rw_flat.repeat(n_in // rw_flat.shape[0], 1)
        else:
            msg = "Local router weights do not align with collocation batch size."
            raise ValueError(msg)
        return value_net.task_value(z_in, rw_in)

    # Single batched call inside hyperbolic_laplacian (returns V_center too)
    lap_V, V_center = hyperbolic_laplacian(V_func, z_flat)  # [N, 1], [N, 1]

    # PDE residual: (-Delta_G + kappa^2) V - rho_r
    residual = -lap_V + kappa**2 * V_center - rho_r

    return (residual**2).mean()


# ---------------------------------------------------------------------------
# Phase 2 loss assembler
# ---------------------------------------------------------------------------


def compute_phase2_loss(
    wm_output: dict[str, torch.Tensor],
    z_targets: torch.Tensor,
    chart_targets: torch.Tensor,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Assemble Phase 2 (world model) loss.

    Args:
        wm_output: Dict from ``GeometricWorldModel.forward``.
        z_targets: [B, H, D] ground-truth latent positions.
        chart_targets: [B, H] ground-truth chart indices.
        config: VLA configuration.

    Returns:
        A tuple of (total_loss, metrics) where:
            total_loss: Scalar weighted sum of all phase-2 loss components.
            metrics: Dict mapping loss component names (e.g. "geodesic",
                "chart_transition", "momentum_reg", "energy_conservation",
                "hodge", "screened_poisson", "total") to their scalar float
                values.
    """
    metrics: dict[str, float] = {}

    loss_geo = compute_dynamics_geodesic_loss(wm_output["z_trajectory"], z_targets)
    total = config.w_geodesic * loss_geo
    metrics["geodesic"] = loss_geo.item()

    loss_chart = compute_dynamics_chart_loss(wm_output["chart_logits"], chart_targets)
    total = total + config.w_chart_transition * loss_chart
    metrics["chart_transition"] = loss_chart.item()

    loss_mom = compute_momentum_regularization(wm_output["momenta"], wm_output["z_trajectory"])
    total = total + config.w_momentum_reg * loss_mom
    metrics["momentum_reg"] = loss_mom.item()

    # Energy conservation loss
    if config.w_energy_conservation > 0:
        if "energy_var" in wm_output:
            # Pre-computed variance across BAOAB sub-steps (more accurate)
            loss_energy = wm_output["energy_var"]
        elif "phi_eff" in wm_output:
            # Fallback: variance across horizon steps
            loss_energy = compute_energy_conservation_loss(
                wm_output["phi_eff"],
                wm_output["momenta"],
                wm_output["z_trajectory"],
            )
        else:
            loss_energy = z_targets.new_tensor(0.0)
        total = total + config.w_energy_conservation * loss_energy
        metrics["energy_conservation"] = loss_energy.item()

    # Hodge consistency loss
    if getattr(config, "w_hodge", 0.0) > 0 and "hodge_harmonic_forces" in wm_output:
        loss_hodge = compute_hodge_consistency_loss(wm_output["hodge_harmonic_forces"])
        total = total + config.w_hodge * loss_hodge
        metrics["hodge"] = loss_hodge.item()

    # Screened Poisson critic loss
    if getattr(config, "w_screened_poisson", 0.0) > 0 and "potential_net" in wm_output:
        rw = wm_output.get(
            "router_weights_final",
            torch.softmax(wm_output["chart_logits"][:, -1, :], dim=-1),
        )
        loss_sp = compute_screened_poisson_loss(
            wm_output["potential_net"],
            wm_output["z_trajectory"],
            wm_output.get("z_targets", z_targets),
            rw,
            kappa=getattr(config, "wm_screening_kappa", 1.0),
        )
        total = total + config.w_screened_poisson * loss_sp
        metrics["screened_poisson"] = loss_sp.item()

    metrics["total"] = total.item()
    return total, metrics


# ---------------------------------------------------------------------------
# Geodesic diffusion losses (from vla/geodesic_losses.py)
# ---------------------------------------------------------------------------


def geodesic_interpolation(
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    N: int,
) -> torch.Tensor:
    """Create N+1 waypoints along the Poincare geodesic from z_start to z_end.

    Uses log/exp maps: v = log_{z_start}(z_end), z_k = exp_{z_start}(k/N * v).

    Args:
        z_start: [B, D] start position in the Poincare ball.
        z_end: [B, D] end position in the Poincare ball.
        N: Number of intermediate steps (returns N+1 waypoints total).

    Returns:
        waypoints: [B, N+1, D] geodesic waypoints from z_start to z_end.
    """
    v = poincare_log_map(z_start, z_end)  # [B, D] tangent vector at z_start
    waypoints = []
    for k in range(N + 1):
        t = k / max(N, 1)
        z_k = poincare_exp_map(z_start, t * v)
        waypoints.append(z_k)
    return torch.stack(waypoints, dim=1)  # [B, N+1, D]


def compute_momentum_targets(
    z_waypoints: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Finite-difference momentum targets from geodesic waypoints.

    p_k = log_{z_k}(z_{k+1}) / dt  (cotangent vectors).

    Args:
        z_waypoints: [B, N+1, D] geodesic waypoints.
        dt: Integration time step.

    Returns:
        p_targets: [B, N, D] momentum targets.
    """
    N = z_waypoints.shape[1] - 1
    p_list = []
    for k in range(N):
        v_k = poincare_log_map(z_waypoints[:, k], z_waypoints[:, k + 1])
        p_list.append(v_k / dt)
    return torch.stack(p_list, dim=1)  # [B, N, D]


def position_loss(
    z_pred_traj: torch.Tensor,
    z_target_traj: torch.Tensor,
) -> torch.Tensor:
    """Mean hyperbolic distance between predicted and target waypoints.

    L_pos = (1/N) sum_k d_H(z_pred_k, z_target_k)

    Args:
        z_pred_traj: [B, N+1, D] predicted trajectory.
        z_target_traj: [B, N+1, D] target trajectory.

    Returns:
        Scalar loss.
    """
    B, Np1, D = z_pred_traj.shape
    pred_flat = z_pred_traj.reshape(B * Np1, D)
    tgt_flat = z_target_traj.reshape(B * Np1, D)
    return hyperbolic_distance(pred_flat, tgt_flat).mean()


def endpoint_loss(
    z_pred_N: torch.Tensor,
    z_target_end: torch.Tensor,
) -> torch.Tensor:
    """Hyperbolic distance at the final step.

    L_end = d_H(z_pred_N, z_{t+1})

    Args:
        z_pred_N: [B, D] predicted final position.
        z_target_end: [B, D] target final position.

    Returns:
        Scalar loss.
    """
    return hyperbolic_distance(z_pred_N, z_target_end).mean()


def momentum_loss(
    p_pred: torch.Tensor,
    p_target: torch.Tensor,
    z_traj: torch.Tensor,
) -> torch.Tensor:
    """Metric-aware momentum error.

    L_mom = mean_k [ ((1-|z_k|^2)/2)^2 * |p_pred_k - p_target_k|^2 ]

    Since momenta are cotangent vectors, the inverse metric g^{-1} must be
    used to compute their norm.

    Args:
        p_pred: [B, N, D] predicted momenta.
        p_target: [B, N, D] target momenta.
        z_traj: [B, N, D] positions at which momenta are evaluated.

    Returns:
        Scalar loss.
    """
    r_sq = (z_traj**2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, N, 1]
    diff_sq = ((p_pred - p_target) ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
    return (g_inv_factor * diff_sq).mean()


def compute_supervised_wm_loss(
    wm: torch.nn.Module,
    z_start: torch.Tensor,
    z_end: torch.Tensor,
    action: torch.Tensor,
    rw: torch.Tensor,
    N: int,
    dt: float,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervised geodesic diffusion loss for one consecutive pair.

    1. Creates geodesic waypoints between z_start and z_end.
    2. Computes momentum targets from waypoints.
    3. Runs supervised integration through the world model.
    4. Sums weighted position, endpoint, momentum, and auxiliary losses.

    Args:
        wm: GeometricWorldModel instance.
        z_start: [B, D] start position.
        z_end: [B, D] target end position.
        action: [B, A] action for this transition.
        rw: [B, K] router weights.
        N: Number of integration sub-steps.
        dt: Integration time step.
        config: VLAConfig with loss weights.

    Returns:
        A tuple of (total_loss, metrics) where:
            total_loss: Scalar weighted sum of position, endpoint, momentum,
                Hodge consistency, and energy conservation losses.
            metrics: Dict mapping loss component names (e.g. "position",
                "endpoint", "momentum_target", "hodge_perp",
                "energy_conservation", "mean_momentum", "mean_phi_eff",
                "hodge_cons", "hodge_sol", "hodge_harm", "geo_miss",
                "total") to their scalar float values. Diagnostic keys
                (mean_momentum, geo_miss, etc.) are computed under
                torch.no_grad.
    """
    metrics: dict[str, float] = {}

    # 1. Geodesic waypoints (targets)
    z_targets = geodesic_interpolation(z_start, z_end, N)  # [B, N+1, D]

    # 2. Momentum targets
    p_targets = compute_momentum_targets(z_targets, dt)  # [B, N, D]

    # 3. Initial momentum from first target momentum
    p_init = p_targets[:, 0]  # [B, D]

    # 4. Supervised integration
    integ = wm.supervised_integration(
        z_start,
        p_init,
        action,
        rw,
        n_steps=N,
        deterministic=True,
    )
    z_pred = integ["z_traj"]  # [B, N+1, D]
    p_pred = integ["p_traj"]  # [B, N+1, D]

    # 5. Position loss (all waypoints)
    L_pos = position_loss(z_pred, z_targets)
    metrics["position"] = L_pos.item()

    # 6. Endpoint loss (final waypoint)
    L_end = endpoint_loss(z_pred[:, -1], z_end)
    metrics["endpoint"] = L_end.item()

    # 7. Momentum loss (predicted vs target, excluding initial)
    p_pred_steps = p_pred[:, 1:, :]  # [B, N, D] momenta after each step
    z_traj_steps = z_pred[:, 1:, :]  # [B, N, D] positions after each step
    L_mom = momentum_loss(p_pred_steps, p_targets, z_traj_steps)
    metrics["momentum_target"] = L_mom.item()

    total = (
        config.w_position * L_pos + config.w_endpoint * L_end + config.w_momentum_target * L_mom
    )

    # 8. Hodge consistency (harmonic force penalty)
    hodge_info = integ.get("hodge_info", {})
    if config.w_hodge_perp > 0 and "harmonic" in hodge_info:
        L_hodge = (hodge_info["harmonic"] ** 2).mean()
        total = total + config.w_hodge_perp * L_hodge
        metrics["hodge_perp"] = L_hodge.item()

    # 9. Energy conservation (variance of phi_eff across sub-steps)
    w_energy = getattr(config, "w_energy_conservation", 0.0)
    if w_energy > 0 and "phi_eff" in integ:
        phi = integ["phi_eff"]  # [B, N, 1]
        r_sq = (z_pred[:, 1:] ** 2).sum(dim=-1, keepdim=True)
        g_inv = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2
        p_sq = (p_pred[:, 1:] ** 2).sum(dim=-1, keepdim=True)
        H_vals = phi + 0.5 * g_inv * p_sq  # [B, N, 1]
        energy_var = H_vals.squeeze(-1).var(dim=-1).mean()
        total = total + w_energy * energy_var
        metrics["energy_conservation"] = energy_var.item()

    # 10. WM diagnostics (non-loss, for monitoring)
    with torch.no_grad():
        metrics["mean_momentum"] = p_pred[:, 1:].norm(dim=-1).mean().item()
        if "phi_eff" in integ:
            metrics["mean_phi_eff"] = integ["phi_eff"].mean().item()
        hodge_info = integ.get("hodge_info", {})
        if "conservative_ratio" in hodge_info:
            metrics["hodge_cons"] = hodge_info["conservative_ratio"].mean().item()
            metrics["hodge_sol"] = hodge_info["solenoidal_ratio"].mean().item()
            metrics["hodge_harm"] = hodge_info["harmonic_ratio"].mean().item()
        # Geodesic miss distance (how far predicted endpoint is from target)
        metrics["geo_miss"] = (
            hyperbolic_distance(
                z_pred[:, -1],
                z_end,
            )
            .mean()
            .item()
        )

    metrics["total"] = total.item()
    return total, metrics


def compute_phase2_geodesic_diffusion_loss(
    wm: torch.nn.Module,
    z_all: torch.Tensor,
    rw_all: torch.Tensor,
    K_all: torch.Tensor,
    actions: torch.Tensor,
    config: VLAConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Geodesic diffusion loss over consecutive pairs in a sequence.

    For same-chart pairs: supervised integration + geodesic waypoint matching.
    For cross-chart pairs: chart transition CE only (jump operator training).

    Args:
        wm: GeometricWorldModel instance.
        z_all: [B, H, D] encoded latent positions for all frames.
        rw_all: [B, H, K] router weights for all frames.
        K_all: [B, H] chart indices for all frames.
        actions: [B, H, A] action sequence.
        config: VLAConfig with loss weights and hyperparameters.

    Returns:
        A tuple of (total_loss, metrics) where:
            total_loss: Scalar loss averaged over consecutive same-chart
                pairs, plus chart transition cross-entropy.
            metrics: Dict mapping aggregated loss component names to their
                scalar float values. Includes per-pair supervised loss
                components (e.g. "position", "endpoint", "momentum_target"),
                "chart_transition" CE, "total", "n_same_chart_pairs",
                "same_chart_frac", and "chart_accuracy".
    """
    B, H, _ = z_all.shape
    N = getattr(config, "wm_diffusion_substeps", 8)
    dt = getattr(config, "wm_dt", 0.01)

    total_loss = z_all.new_tensor(0.0)
    chart_ce_total = z_all.new_tensor(0.0)
    chart_correct = 0
    chart_total_samples = 0
    same_chart_samples = 0
    total_samples = 0
    pair_count = 0
    chart_pair_count = 0

    agg_metrics: dict[str, float] = {}
    metric_accum: dict[str, float] = {}

    for t in range(H - 1):
        z_t = z_all[:, t]  # [B, D]
        z_tp1 = z_all[:, t + 1]  # [B, D]
        rw_t = rw_all[:, t]  # [B, K]
        action_t = actions[:, t]  # [B, A]
        K_t = K_all[:, t]  # [B]
        K_tp1 = K_all[:, t + 1]  # [B]

        # Chart transition CE (always computed)
        chart_logits = wm.chart_predictor(z_t, action_t, rw_t)  # [B, K]
        chart_ce = F.cross_entropy(chart_logits, K_tp1.long())
        chart_ce_total = chart_ce_total + chart_ce
        chart_pair_count += 1

        # Chart prediction accuracy
        with torch.no_grad():
            chart_correct += (chart_logits.argmax(dim=-1) == K_tp1.long()).sum().item()
            chart_total_samples += B

        # Same-chart mask: only do supervised integration for same-chart pairs
        same_chart = K_t == K_tp1  # [B]
        same_chart_samples += same_chart.sum().item()
        total_samples += B
        if same_chart.any():
            # Select same-chart samples
            idx = same_chart.nonzero(as_tuple=True)[0]
            z_s = z_t[idx]
            z_e = z_tp1[idx]
            a_s = action_t[idx]
            rw_s = rw_t[idx]

            pair_loss, pair_metrics = compute_supervised_wm_loss(
                wm,
                z_s,
                z_e,
                a_s,
                rw_s,
                N,
                dt,
                config,
            )
            total_loss = total_loss + pair_loss
            pair_count += 1

            for k, v in pair_metrics.items():
                metric_accum[k] = metric_accum.get(k, 0.0) + v

    # Average losses
    if pair_count > 0:
        total_loss = total_loss / pair_count
        for k in metric_accum:
            metric_accum[k] /= pair_count

    if chart_pair_count > 0:
        chart_ce_avg = chart_ce_total / chart_pair_count
        w_chart = getattr(config, "w_chart_transition", 0.5)
        total_loss = total_loss + w_chart * chart_ce_avg
        agg_metrics["chart_transition"] = chart_ce_avg.item()

    agg_metrics.update(metric_accum)
    agg_metrics["total"] = total_loss.item()
    agg_metrics["n_same_chart_pairs"] = pair_count
    agg_metrics["same_chart_frac"] = same_chart_samples / max(total_samples, 1)
    agg_metrics["chart_accuracy"] = chart_correct / max(chart_total_samples, 1)

    return total_loss, agg_metrics


# ---------------------------------------------------------------------------
# World model closure losses (from train_dreamer.py)
# ---------------------------------------------------------------------------


def _hodge_conservative_preference_losses(
    config,
    *,
    hodge_conservative_ratio: torch.Tensor,
    hodge_solenoidal_ratio: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Bias dynamics toward conservative explanation once harmonic residue is small.

    Computes a margin loss that penalises the conservative Hodge ratio when
    it falls below a configurable target, plus an L2 penalty on the solenoidal
    ratio to encourage the model to explain forces conservatively.

    Args:
        config: Configuration object with a ``hodge_conservative_target``
            attribute (float) specifying the desired minimum conservative
            ratio.
        hodge_conservative_ratio: [B, ...] tensor of per-sample conservative
            force ratios from the Hodge decomposition.
        hodge_solenoidal_ratio: [B, ...] tensor of per-sample solenoidal
            force ratios from the Hodge decomposition.

    Returns:
        A tuple of (L_hodge_conservative_margin, L_hodge_solenoidal, metrics)
        where:
            L_hodge_conservative_margin: Scalar mean-squared deficit of the
                conservative ratio below the target.
            L_hodge_solenoidal: Scalar mean-squared solenoidal ratio penalty.
            metrics: Dict with keys "wm/L_hodge_conservative_margin",
                "wm/L_hodge_solenoidal", "geometric/hodge_conservative_deficit",
                and "geometric/hodge_conservative_target" mapped to their
                float values.
    """
    conservative_target = float(config.hodge_conservative_target)
    conservative_deficit = (conservative_target - hodge_conservative_ratio).clamp(min=0.0)
    L_hodge_conservative_margin = conservative_deficit.pow(2).mean()
    L_hodge_solenoidal = hodge_solenoidal_ratio.pow(2).mean()
    metrics = {
        "wm/L_hodge_conservative_margin": float(L_hodge_conservative_margin),
        "wm/L_hodge_solenoidal": float(L_hodge_solenoidal),
        "geometric/hodge_conservative_deficit": float(conservative_deficit.mean()),
        "geometric/hodge_conservative_target": conservative_target,
    }
    return L_hodge_conservative_margin, L_hodge_solenoidal, metrics


def _world_model_closure_losses(
    config,
    world_model,
    enclosure_probe,
    z_0: torch.Tensor,
    rw_0: torch.Tensor,
    chart_embed_t: torch.Tensor,
    z_tex_t: torch.Tensor,
    action_canonicals: torch.Tensor,
    code_t: torch.Tensor,
    target_charts: torch.Tensor,
    target_codes: torch.Tensor,
    update_idx: int,
    *,
    zeno_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Measure closure and router smoothness from the observation Markov model.

    Runs the world model forward on the given initial state and action
    sequence, then computes chart-transition cross-entropy, Zeno
    regularisation (router smoothness), and enclosure losses via a
    gradient-reversal probe. Returns early with zeroed losses when the
    input sequences are empty.

    Args:
        config: Configuration object with attributes
            ``enclosure_grl_warmup_updates`` (int),
            ``enclosure_grl_alpha_max`` (float), and
            ``codes_per_chart`` (int).
        world_model: GeometricWorldModel instance whose ``forward`` method
            produces chart logits.
        enclosure_probe: Enclosure probe module with a gradient-reversal
            layer (``grl``) used for the enclosure adversarial loss.
        z_0: [B, D] initial latent position.
        rw_0: [B, K] initial router weights.
        chart_embed_t: [B, T, E] chart embedding features for the sequence.
        z_tex_t: [B, T, D_tex] texture latent features for the sequence.
        action_canonicals: [B, T, A] canonical action vectors.
        code_t: [B, T] integer code indices at each time step.
        target_charts: [B, T] ground-truth chart indices for supervision.
        target_codes: [B, T] ground-truth code indices for the enclosure
            probe.
        update_idx: Current training update index, used for GRL alpha
            scheduling.
        zeno_mode: Mode string forwarded to ``zeno_loss`` (e.g. "kl",
            "cosine").

    Returns:
        A tuple of (L_closure_obs, L_obs_zeno, L_enclosure,
        L_enclosure_probe, metrics) where:
            L_closure_obs: Scalar chart-transition cross-entropy loss.
            L_obs_zeno: Scalar Zeno regularisation loss for router
                smoothness.
            L_enclosure: Scalar enclosure loss (main model side).
            L_enclosure_probe: Scalar enclosure probe loss (adversarial
                side).
            metrics: Dict with keys "closure/obs_state_acc",
                "closure/obs_symbol_acc", "closure/chart_entropy",
                "closure/enclosure_acc_full", "closure/enclosure_acc_base",
                "closure/enclosure_defect_acc",
                "closure/enclosure_defect_ce", and "closure/grl_alpha"
                mapped to their float values.
    """
    from fragile.losses.macro import zeno_loss
    from fragile.losses.old_macro import (
        compute_enclosure_loss,
        grl_alpha_schedule,
    )

    zero = z_0.new_zeros(())
    if action_canonicals.shape[1] == 0 or target_charts.shape[1] == 0:
        return (
            zero,
            zero,
            zero,
            zero,
            {
                "closure/obs_state_acc": 0.0,
                "closure/obs_symbol_acc": 0.0,
                "closure/chart_entropy": 0.0,
                "closure/enclosure_acc_full": 0.0,
                "closure/enclosure_acc_base": 0.0,
                "closure/enclosure_defect_acc": 0.0,
                "closure/enclosure_defect_ce": 0.0,
                "closure/grl_alpha": 0.0,
            },
        )

    wm_out = world_model(z_0, action_canonicals, rw_0)
    chart_logits_all = wm_out["chart_logits"]
    T_sync = min(chart_logits_all.shape[1], target_charts.shape[1])
    if T_sync == 0:
        return (
            zero,
            zero,
            zero,
            zero,
            {
                "closure/obs_state_acc": 0.0,
                "closure/obs_symbol_acc": 0.0,
                "closure/chart_entropy": 0.0,
                "closure/enclosure_acc_full": 0.0,
                "closure/enclosure_acc_base": 0.0,
                "closure/enclosure_defect_acc": 0.0,
                "closure/enclosure_defect_ce": 0.0,
                "closure/grl_alpha": 0.0,
            },
        )

    chart_logits = chart_logits_all[:, :T_sync]
    target = target_charts[:, :T_sync].detach()
    chart_probs = F.softmax(chart_logits, dim=-1)
    prev_rw = torch.cat([rw_0.unsqueeze(1), chart_probs[:, :-1]], dim=1)
    L_closure_obs = compute_dynamics_chart_loss(chart_logits, target)
    L_obs_zeno = zeno_loss(
        chart_probs.reshape(-1, chart_probs.shape[-1]),
        prev_rw.reshape(-1, prev_rw.shape[-1]),
        mode=zeno_mode,
    )
    grl_alpha = grl_alpha_schedule(
        update_idx,
        warmup_steps=config.enclosure_grl_warmup_updates,
        max_alpha=config.enclosure_grl_alpha_max,
    )
    enclosure_probe.grl.alpha.copy_(enclosure_probe.grl.alpha.new_tensor(grl_alpha))
    L_enclosure, L_enclosure_probe, enclosure_metrics = compute_enclosure_loss(
        enclosure_probe,
        chart_embed_t[:, :T_sync].reshape(-1, chart_embed_t.shape[-1]),
        action_canonicals[:, :T_sync].reshape(-1, action_canonicals.shape[-1]),
        z_tex_t[:, :T_sync].reshape(-1, z_tex_t.shape[-1]),
        target_charts[:, :T_sync].reshape(-1),
        K_code_t=code_t[:, :T_sync].reshape(-1),
        K_code_tp1=target_codes[:, :T_sync].reshape(-1),
        codes_per_chart=config.codes_per_chart,
    )
    metrics = {
        "closure/obs_state_acc": float((chart_logits.argmax(dim=-1) == target).float().mean()),
        "closure/obs_symbol_acc": enclosure_metrics["acc_full"],
        "closure/chart_entropy": float(
            -(chart_probs * chart_probs.clamp(min=1e-8).log()).sum(dim=-1).mean(),
        ),
        "closure/enclosure_acc_full": enclosure_metrics["acc_full"],
        "closure/enclosure_acc_base": enclosure_metrics["acc_base"],
        "closure/enclosure_defect_acc": enclosure_metrics["defect_acc"],
        "closure/enclosure_defect_ce": enclosure_metrics["defect_ce"],
        "closure/grl_alpha": grl_alpha,
    }
    return L_closure_obs, L_obs_zeno, L_enclosure, L_enclosure_probe, metrics
