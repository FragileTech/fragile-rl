"""VLA-specific loss functions and phase loss assemblers."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F

from fragile.core.layers.gauge import (
    hyperbolic_distance,
    poincare_exp_map,
    poincare_log_map,
)
from fragile.hyperbolic_losses import (
    combine_quality_targets,
    compute_chart_center_mean_loss,
    compute_chart_center_radius_loss,
    compute_chart_center_separation_loss,
    compute_chart_usage_band_loss,
    compute_code_usage_band_loss,
    compute_codebook_centering_loss,
    compute_codebook_spread_loss,
    compute_confidence_calibration_loss,
    compute_error_quality_targets,
    compute_hard_routing_nll,
    compute_hyperbolic_uniformity_loss,
    compute_radial_calibration_loss,
    compute_rank_quality_targets,
    compute_router_information_metrics,
    compute_router_margin_loss,
    compute_router_sharpness_metrics,
    compute_routing_confidence,
    compute_routing_entropy,
    compute_sinkhorn_balanced_chart_loss,
    compute_v_tangent_barrier_loss,
    compute_window_loss,
    mix_quality_targets,
)

from .config import VLAConfig


def _project_to_ball(z: torch.Tensor, max_norm: float = 0.99, eps: float = 1e-6) -> torch.Tensor:
    """Project points to the open Poincare ball."""
    norm = z.norm(dim=-1, keepdim=True).clamp_min(eps)
    scale = (max_norm / norm).clamp(max=1.0)
    return z * scale


def _deterministic_st_router_weights(router_scores: torch.Tensor) -> torch.Tensor:
    """Build deterministic straight-through one-hot router weights from scores."""
    soft = F.softmax(router_scores, dim=-1)
    one_hot = F.one_hot(router_scores.argmax(dim=-1), router_scores.shape[-1]).to(soft.dtype)
    return one_hot + soft - soft.detach()


# ---------------------------------------------------------------------------
# New dynamics losses
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
    r_sq = (z_trajectory ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
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
    r_sq = (z_trajectory ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
    g_inv_factor = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2  # [B, H, 1]
    p_sq = (momenta ** 2).sum(dim=-1, keepdim=True)  # [B, H, 1]
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
    return (hodge_harmonic_forces ** 2).mean()


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
    V_plus = V_split[1:1 + 2 * n_probes:2]   # [k, N, 1]
    V_minus = V_split[2:2 + 2 * n_probes:2]  # [k, N, 1]
    trace_terms = V_plus - 2 * V_center.unsqueeze(0) + V_minus  # [k, N, 1]
    laplacian_E = trace_terms.sum(dim=0) / (n_probes * eps ** 2)  # [N, 1]

    # --- Directional derivative: z · ∇V ≈ ||z|| * (V(z+εẑ) - V(z-εẑ)) / (2ε) ---
    V_zhat_plus = V_split[-2]   # [N, 1]
    V_zhat_minus = V_split[-1]  # [N, 1]
    z_dot_grad = z_norm * (V_zhat_plus - V_zhat_minus) / (2 * eps)  # [N, 1]

    # --- Poincare ball correction ---
    r_sq = (z_det ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
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
    residual = -lap_V + kappa ** 2 * V_center - rho_r

    return (residual ** 2).mean()


# ---------------------------------------------------------------------------
# Orthogonality & enclosure losses
# ---------------------------------------------------------------------------


def orthogonality_loss(zn: torch.Tensor, ztex: torch.Tensor) -> torch.Tensor:
    """Penalize correlation between z_n and z_tex.

    Uses squared cosine similarity (normalized dot product) so the loss is
    scale-invariant.  This prevents explosions when tangent-vector norms grow
    as points migrate toward the Poincaré boundary (expected behaviour from
    radial calibration).

    If dimensions match: squared cosine similarity, meaned over batch.
    If dimensions differ: Frobenius norm of cross-correlation matrix
    (columns pre-normalized).

    Args:
        zn: [B, D1] navigational latent (tangent space).
        ztex: [B, D2] texture latent (tangent space).

    Returns:
        Scalar loss in [0, 1].
    """
    if zn.shape[-1] == ztex.shape[-1]:
        # Squared cosine similarity per sample, mean over batch
        zn_n = F.normalize(zn, dim=-1)
        ztex_n = F.normalize(ztex, dim=-1)
        return ((zn_n * ztex_n).sum(dim=-1) ** 2).mean()

    zn_c = zn - zn.mean(dim=0, keepdim=True)
    ztex_c = ztex - ztex.mean(dim=0, keepdim=True)
    # Normalize columns so the cross-correlation is scale-invariant
    zn_c = F.normalize(zn_c, dim=0)
    ztex_c = F.normalize(ztex_c, dim=0)
    C = zn_c.T @ ztex_c  # [D1, D2], entries in [-1, 1]
    return (C ** 2).mean()


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, negates gradients backward with alpha scaling."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as an nn.Module."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


class EnclosureProbe(nn.Module):
    """Adversarial probe enforcing that z_tex carries no dynamics information.

    z_tex is the high-frequency texture residual used only by the decoder.
    It must not leak (chart, symbol) transition information — all dynamics
    should live in z_q (codes) and z_n (continuous refinement for the world
    model).

    Two probes share the same architecture:
      - full_probe:     chart_embed + code_embed + action + GRL(z_tex) -> logits [B, S]
      - baseline_probe: chart_embed + code_embed + action              -> logits [B, S]

    where S = num_charts * codes_per_chart is the flat (chart, symbol) state
    count.  The GRL reverses gradients into the encoder so that the structure
    filter learns to keep dynamics out of z_tex.

    Args:
        chart_dim: Dimension of chart embedding (c_bar).
        action_dim: Dimension of action vector.
        ztex_dim: Dimension of z_tex.
        num_charts: Number of chart classes.
        codes_per_chart: Number of VQ codes per chart.
        hidden_dim: Hidden layer width.
        alpha: Initial GRL alpha.
    """

    def __init__(
        self,
        chart_dim: int = 16,
        action_dim: int = 6,
        ztex_dim: int = 16,
        num_charts: int = 8,
        codes_per_chart: int = 32,
        hidden_dim: int = 128,
        alpha: float = 1.0,
    ):
        super().__init__()
        self.grl = GradientReversalLayer(alpha=alpha)
        self.num_states = num_charts * codes_per_chart
        self.codes_per_chart = codes_per_chart

        self.code_embed = nn.Embedding(codes_per_chart, chart_dim)

        full_in = chart_dim + chart_dim + action_dim + ztex_dim
        self.full_probe = nn.Sequential(
            nn.Linear(full_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_states),
        )

        base_in = chart_dim + chart_dim + action_dim
        self.baseline_probe = nn.Sequential(
            nn.Linear(base_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_states),
        )

    def forward(
        self,
        chart_embed: torch.Tensor,
        action: torch.Tensor,
        z_tex: torch.Tensor,
        code_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            chart_embed: [B, chart_dim] e.g. c_bar.
            action: [B, action_dim].
            z_tex: [B, ztex_dim] texture residual.
            code_idx: [B] long tensor of current VQ code indices.

        Returns:
            (logits_full, logits_baseline) each [B, num_states].
        """
        code_e = self.code_embed(code_idx)
        ztex_rev = self.grl(z_tex)
        full_input = torch.cat([chart_embed, code_e, action, ztex_rev], dim=-1)
        base_input = torch.cat([chart_embed, code_e, action], dim=-1)
        return self.full_probe(full_input), self.baseline_probe(base_input)


class DynamicsTransitionModel(nn.Module):
    """Coarse Markov model: P(c_{t+1}, k_{t+1} | c_bar_t, k_t, a_t).

    Same architecture as EnclosureProbe but without GRL. The transition
    operates over the code symbols used by the encoder, which in the
    shared-codebook setting are the same symbols used for reconstruction.
    """

    def __init__(
        self,
        chart_dim: int,
        action_dim: int,
        num_charts: int,
        codes_per_chart: int | None = None,
        dyn_codes_per_chart: int | None = None,
        hidden_dim: int = 128,
    ):
        super().__init__()
        if codes_per_chart is None:
            if dyn_codes_per_chart is None:
                msg = "DynamicsTransitionModel requires codes_per_chart."
                raise ValueError(msg)
            codes_per_chart = dyn_codes_per_chart
        elif dyn_codes_per_chart is not None and dyn_codes_per_chart != codes_per_chart:
            msg = "codes_per_chart and dyn_codes_per_chart must match when both are set."
            raise ValueError(msg)

        self.num_states = num_charts * codes_per_chart
        self.codes_per_chart = codes_per_chart
        # Backward-compatible alias for older call sites and checkpoints.
        self.dyn_codes_per_chart = codes_per_chart
        self.code_embed = nn.Embedding(codes_per_chart, chart_dim)
        self.mlp = nn.Sequential(
            nn.Linear(chart_dim + chart_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_states),
        )

    def forward(
        self,
        chart_embed: torch.Tensor,
        action: torch.Tensor,
        code_idx: torch.Tensor | None = None,
        code_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Returns logits [B, num_states]."""
        if code_features is None:
            if code_idx is None:
                msg = "Either code_idx or code_features must be provided."
                raise ValueError(msg)
            code_e = self.code_embed(code_idx)
        else:
            code_e = code_features
        inp = torch.cat([chart_embed, code_e, action], dim=-1)
        return self.mlp(inp)


def compute_dyn_transition_loss(
    model: DynamicsTransitionModel,
    chart_embed_t: torch.Tensor,
    action_t: torch.Tensor,
    K_code_dyn_t: torch.Tensor,
    K_chart_tp1: torch.Tensor,
    K_code_dyn_tp1: torch.Tensor,
    code_features_t: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """CE loss + accuracy metric for dynamics transition prediction."""
    target = K_chart_tp1.long() * model.codes_per_chart + K_code_dyn_tp1.long()
    logits = model(
        chart_embed_t,
        action_t,
        K_code_dyn_t,
        code_features=code_features_t,
    )
    loss = F.cross_entropy(logits, target)
    with torch.no_grad():
        acc = (logits.argmax(dim=-1) == target).float().mean().item()
    return loss, {"dyn_trans_ce": loss.item(), "dyn_trans_acc": acc}


def compute_enclosure_loss(
    probe: EnclosureProbe,
    chart_embed_t: torch.Tensor,
    action_t: torch.Tensor,
    ztex_t: torch.Tensor,
    K_chart_tp1: torch.Tensor,
    K_code_t: torch.Tensor | None = None,
    K_code_tp1: torch.Tensor | None = None,
    codes_per_chart: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute enclosure probe losses and diagnostics.

    The probe checks whether z_tex leaks dynamics (chart, code) transition
    information.  Gradient reversal pushes the structure filter to keep
    dynamics out of z_tex — all dynamics should live in z_q and z_n.

    Args:
        probe: The EnclosureProbe module.
        chart_embed_t: [B, D] chart embedding at time t (e.g. c_bar).
        action_t: [B, action_dim] action at time t.
        ztex_t: [B, ztex_dim] texture residual at time t.
        K_chart_tp1: [B] ground-truth chart index at t+1.
        K_code_t: [B] current VQ code index (defaults to zeros).
        K_code_tp1: [B] next VQ code index (defaults to zeros).
        codes_per_chart: Number of VQ codes per chart.

    Returns:
        loss_encoder: CE on full probe (GRL reverses gradient into encoder).
        loss_probe: CE on detached inputs for both probes (trains probe only).
        diagnostics: dict with acc_full, acc_base, defect_acc, defect_ce,
                     ce_full, ce_base.
    """
    B = K_chart_tp1.shape[0]
    device = K_chart_tp1.device

    if codes_per_chart is None:
        codes_per_chart = probe.codes_per_chart

    if K_code_t is None:
        K_code_t = torch.zeros(B, dtype=torch.long, device=device)
    if K_code_tp1 is None:
        K_code_tp1 = torch.zeros(B, dtype=torch.long, device=device)

    # Flat (chart, code) target
    target = K_chart_tp1.long() * codes_per_chart + K_code_tp1.long()

    # -- Encoder loss: gradients flow through GRL into structure filter --
    logits_full, _ = probe(chart_embed_t, action_t, ztex_t, K_code_t)
    ce_full = F.cross_entropy(logits_full, target)
    loss_encoder = ce_full

    # -- Probe loss: train probe on detached inputs --
    logits_full_det, logits_base_det = probe(
        chart_embed_t.detach(), action_t.detach(), ztex_t.detach(),
        K_code_t.detach(),
    )
    ce_full_det = F.cross_entropy(logits_full_det, target)
    ce_base_det = F.cross_entropy(logits_base_det, target)
    loss_probe = ce_full_det + ce_base_det

    # -- Diagnostics --
    with torch.no_grad():
        acc_full = (logits_full_det.argmax(dim=-1) == target).float().mean().item()
        acc_base = (logits_base_det.argmax(dim=-1) == target).float().mean().item()
        defect_acc = acc_full - acc_base
        defect_ce = ce_base_det.item() - ce_full_det.item()

    diagnostics = {
        "acc_full": acc_full,
        "acc_base": acc_base,
        "defect_acc": defect_acc,
        "defect_ce": defect_ce,
        "ce_full": ce_full_det.item(),
        "ce_base": ce_base_det.item(),
    }

    return loss_encoder, loss_probe, diagnostics


def grl_alpha_schedule(
    step: int,
    warmup_steps: int = 5000,
    max_alpha: float = 1.0,
) -> float:
    """Linear warmup schedule for GRL alpha.

    Args:
        step: Current training step.
        warmup_steps: Number of steps to linearly ramp alpha.
        max_alpha: Maximum alpha value after warmup.

    Returns:
        Alpha value for the current step.
    """
    if step >= warmup_steps:
        return max_alpha
    return max_alpha * step / warmup_steps


# ---------------------------------------------------------------------------
# Zeno loss (routing distribution smoothness)
# ---------------------------------------------------------------------------


def zeno_loss(
    w_t: torch.Tensor,
    w_t_prev: torch.Tensor,
    mode: str = "jsd",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize rapid changes in the soft routing distribution.

    Args:
        w_t: [B, N_c] current routing weights (softmax, has grad).
        w_t_prev: [B, N_c] previous routing weights (softmax, has grad).
        mode: "kl" for D_KL(w_t || w_{t-1}), "jsd" for Jensen-Shannon.
        eps: Floor to prevent log(0).

    Returns:
        Scalar loss, mean over batch.
    """
    w_t_safe = w_t.clamp(min=eps)
    w_prev_safe = w_t_prev.clamp(min=eps)

    if mode == "kl":
        kl = (w_t_safe * (w_t_safe.log() - w_prev_safe.log())).sum(dim=-1)
        return kl.mean()
    if mode == "jsd":
        m = 0.5 * (w_t_safe + w_prev_safe)
        kl_t = (w_t_safe * (w_t_safe.log() - m.log())).sum(dim=-1)
        kl_prev = (w_prev_safe * (w_prev_safe.log() - m.log())).sum(dim=-1)
        return (0.5 * kl_t + 0.5 * kl_prev).mean()
    raise ValueError(f"Unknown zeno_loss mode: {mode}")


def compute_dynamics_markov_loss(
    atlas_encoder: torch.nn.Module,
    dyn_trans_model: DynamicsTransitionModel | None,
    v_local_all: torch.Tensor,
    router_weights_all: torch.Tensor,
    chart_embed_all: torch.Tensor,
    chart_targets_all: torch.Tensor,
    actions: torch.Tensor,
    *,
    transition_weight: float = 0.5,
    zeno_weight: float = 0.0,
    zeno_mode: str = "jsd",
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor | None]:
    """Auxiliary macro-Markov losses for Phase 2/3 dynamics symbols.

    The Phase 1 atlas stays frozen while a separate dynamics codebook learns
    symbols on the same chart-local latent. The simple Markov model operates on
    the frozen macro geometry ``c_bar_t`` plus the trainable dynamics symbol.
    This makes the closure signal trainable without changing the Phase 1 atlas.

    Returns:
        total_loss: VQ + weighted transition CE + optional zeno smoothness.
        metrics: Logged diagnostics for the dynamics-symbol auxiliary.
        K_code_dyn_all: [B, H] hard dynamics-code assignments, or None.
    """
    zero = v_local_all.new_tensor(0.0)
    metrics = {
        "dyn_vq": 0.0,
        "dyn_trans_ce": 0.0,
        "dyn_trans_acc": 0.0,
        "dyn_zeno": 0.0,
        "dyn_state_flip_rate": 0.0,
        "dyn_state_entropy": 0.0,
        "dyn_state_max_prob": 0.0,
        "dyn_code_flip_rate": 0.0,
    }
    if dyn_trans_model is None or v_local_all.shape[1] < 2:
        return zero, metrics, None

    B, H, D = v_local_all.shape
    chart_dim = chart_embed_all.shape[-1]
    action_dim = actions.shape[-1]

    z_q_dyn_flat, K_code_dyn_flat, _, vq_dyn_loss = atlas_encoder.dynamics_vq(
        v_local_all.reshape(B * H, D),
        router_weights_all.reshape(B * H, router_weights_all.shape[-1]),
    )
    z_q_dyn_all = z_q_dyn_flat.reshape(B, H, D)
    K_code_dyn_all = K_code_dyn_flat.reshape(B, H)

    n_transitions = min(H - 1, actions.shape[1])
    if n_transitions < 1:
        metrics["dyn_vq"] = vq_dyn_loss.item()
        return vq_dyn_loss, metrics, K_code_dyn_all

    trans_logits = dyn_trans_model(
        chart_embed_all[:, :n_transitions].reshape(B * n_transitions, chart_dim),
        actions[:, :n_transitions].reshape(B * n_transitions, action_dim),
        K_code_dyn_all[:, :n_transitions].reshape(B * n_transitions),
        code_features=z_q_dyn_all[:, :n_transitions].reshape(B * n_transitions, D),
    )
    trans_target = (
        chart_targets_all[:, 1 : n_transitions + 1].long() * dyn_trans_model.codes_per_chart
        + K_code_dyn_all[:, 1 : n_transitions + 1].long()
    ).reshape(B * n_transitions)
    trans_loss = F.cross_entropy(trans_logits, trans_target)
    total = vq_dyn_loss + transition_weight * trans_loss

    metrics["dyn_vq"] = vq_dyn_loss.item()
    metrics["dyn_trans_ce"] = trans_loss.item()
    metrics["dyn_trans_acc"] = float((trans_logits.argmax(dim=-1) == trans_target).float().mean())

    code_flips = (
        K_code_dyn_all[:, 1 : n_transitions + 1] != K_code_dyn_all[:, :n_transitions]
    ).float().mean()
    metrics["dyn_code_flip_rate"] = code_flips.item()

    if n_transitions > 1:
        probs = F.softmax(trans_logits, dim=-1).reshape(B, n_transitions, -1)
        pred_states = probs.argmax(dim=-1)
        state_entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
        metrics["dyn_state_entropy"] = state_entropy.item()
        metrics["dyn_state_max_prob"] = probs.max(dim=-1).values.mean().item()
        metrics["dyn_state_flip_rate"] = (
            (pred_states[:, 1:] != pred_states[:, :-1]).float().mean().item()
        )
        if zeno_weight > 0:
            dyn_zeno = zeno_loss(
                probs[:, 1:].reshape(-1, probs.shape[-1]),
                probs[:, :-1].reshape(-1, probs.shape[-1]),
                mode=zeno_mode,
            )
            total = total + zeno_weight * dyn_zeno
            metrics["dyn_zeno"] = dyn_zeno.item()

    return total, metrics, K_code_dyn_all


# ---------------------------------------------------------------------------
# Supervised geodesic diffusion losses
# ---------------------------------------------------------------------------


def geodesic_interpolation(
    z_start: torch.Tensor, z_end: torch.Tensor, N: int,
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
    z_waypoints: torch.Tensor, dt: float,
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
    z_pred_traj: torch.Tensor, z_target_traj: torch.Tensor,
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
    z_pred_N: torch.Tensor, z_target_end: torch.Tensor,
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
    p_pred: torch.Tensor, p_target: torch.Tensor, z_traj: torch.Tensor,
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
    r_sq = (z_traj ** 2).sum(dim=-1, keepdim=True)  # [B, N, 1]
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
        total_loss: Scalar loss.
        metrics: Dict of individual loss components.
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
        z_start, p_init, action, rw, n_steps=N, deterministic=True,
    )
    z_pred = integ["z_traj"]   # [B, N+1, D]
    p_pred = integ["p_traj"]   # [B, N+1, D]

    # 5. Position loss (all waypoints)
    L_pos = position_loss(z_pred, z_targets)
    metrics["position"] = L_pos.item()

    # 6. Endpoint loss (final waypoint)
    L_end = endpoint_loss(z_pred[:, -1], z_end)
    metrics["endpoint"] = L_end.item()

    # 7. Momentum loss (predicted vs target, excluding initial)
    p_pred_steps = p_pred[:, 1:, :]   # [B, N, D] momenta after each step
    z_traj_steps = z_pred[:, 1:, :]   # [B, N, D] positions after each step
    L_mom = momentum_loss(p_pred_steps, p_targets, z_traj_steps)
    metrics["momentum_target"] = L_mom.item()

    total = (
        config.w_position * L_pos
        + config.w_endpoint * L_end
        + config.w_momentum_target * L_mom
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
        metrics["geo_miss"] = hyperbolic_distance(
            z_pred[:, -1], z_end,
        ).mean().item()

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
        total_loss: Scalar loss.
        metrics: Dict of aggregated loss components.
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
        z_t = z_all[:, t]          # [B, D]
        z_tp1 = z_all[:, t + 1]    # [B, D]
        rw_t = rw_all[:, t]        # [B, K]
        action_t = actions[:, t]   # [B, A]
        K_t = K_all[:, t]          # [B]
        K_tp1 = K_all[:, t + 1]    # [B]

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
        same_chart = (K_t == K_tp1)  # [B]
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
                wm, z_s, z_e, a_s, rw_s, N, dt, config,
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
# Phase loss assemblers
# ---------------------------------------------------------------------------


def compute_phase1_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    vq_loss: torch.Tensor,
    enc_router_weights: torch.Tensor,
    dec_router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    encoder: torch.nn.Module,
    config: VLAConfig,
    *,
    router_reg_weights: torch.Tensor | None = None,
    usage_router_weights: torch.Tensor | None = None,
    c_bar: torch.Tensor | None = None,
    v_local: torch.Tensor | None = None,
    indices_stack: torch.Tensor | None = None,
    router_scores: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Assemble Phase 1 (encoder-only) loss from all active terms.

    Returns:
        base_loss: Scalar loss for terms that do NOT flow through z_n.
        zn_reg_loss: Scalar loss for z_n regularization (uniformity, radial_cal).
        metrics: Dict of individual loss components for logging.
    """
    metrics: dict[str, float] = {}
    atlas_encoder = getattr(encoder, "encoder", encoder)
    if router_reg_weights is None:
        router_reg_weights = getattr(
            atlas_encoder, "_last_soft_router_weights_live", enc_router_weights,
        )
    if usage_router_weights is None:
        usage_router_weights = enc_router_weights
    if c_bar is None:
        c_bar = getattr(atlas_encoder, "_last_c_bar", None)
    if v_local is None:
        v_local = getattr(atlas_encoder, "_last_v_local", None)
    v_raw = getattr(atlas_encoder, "_last_v_raw", None)
    if indices_stack is None:
        indices_stack = getattr(atlas_encoder, "_last_indices_stack", None)
    if router_scores is None:
        router_scores = getattr(atlas_encoder, "_last_router_scores_live", None)

    # Feature reconstruction (MSE)
    loss_recon = F.mse_loss(x_recon, x)
    base_loss = config.w_feature_recon * loss_recon
    metrics["recon"] = loss_recon.item()
    per_sample_recon_error = F.mse_loss(
        x_recon, x, reduction="none",
    ).reshape(x.shape[0], -1).mean(dim=1)
    recon_quality_abs = compute_error_quality_targets(
        per_sample_recon_error,
        alpha=config.radial_quality_alpha,
    )
    recon_quality_rank = compute_rank_quality_targets(per_sample_recon_error)
    recon_quality = mix_quality_targets(
        recon_quality_abs,
        recon_quality_rank,
        rank_mix=config.radial_quality_rank_mix,
    )
    vq_quality = torch.ones_like(recon_quality)
    has_vq_quality = False
    if (
        hasattr(atlas_encoder, "codebook")
        and v_local is not None
        and indices_stack is not None
        and getattr(config, "radial_vq_alpha", 0.0) > 0
    ):
        selected_chart = torch.argmax(enc_router_weights.detach(), dim=1)
        selected_code = indices_stack.gather(1, selected_chart.unsqueeze(1)).squeeze(1)
        codebook_proj = _project_to_ball(atlas_encoder.codebook)
        selected_codes = codebook_proj[selected_chart, selected_code]
        per_sample_vq_error = hyperbolic_distance(v_local, selected_codes).pow(2)
        vq_quality_abs = compute_error_quality_targets(
            per_sample_vq_error,
            alpha=config.radial_vq_alpha,
        )
        vq_quality_rank = compute_rank_quality_targets(per_sample_vq_error)
        vq_quality = mix_quality_targets(
            vq_quality_abs,
            vq_quality_rank,
            rank_mix=config.radial_quality_rank_mix,
        )
        has_vq_quality = True
    if has_vq_quality:
        quality_target = combine_quality_targets(
            recon_quality,
            vq_quality,
            primary_weight=config.radial_recon_quality_weight,
        )
    else:
        quality_target = recon_quality
    quality_mix = min(max(float(config.radial_quality_mix), 0.0), 1.0)
    quality_base_weight = min(max(float(getattr(config, "radial_quality_base_weight", 0.0)), 0.0), 1.0)
    routing_confidence = compute_routing_confidence(
        router_reg_weights.detach(),
        config.num_charts,
    )
    gated_radial_target = routing_confidence * ((1.0 - quality_mix) + quality_mix * quality_target)
    radial_target = (
        (1.0 - quality_base_weight) * gated_radial_target
        + quality_base_weight * quality_target
    )
    metrics["recon_quality_mean"] = recon_quality.mean().item()
    metrics["vq_quality_mean"] = vq_quality.mean().item()
    metrics["combined_quality_mean"] = quality_target.mean().item()
    metrics["routing_confidence_mean"] = routing_confidence.mean().item()
    metrics["radial_target_mean"] = radial_target.mean().item()

    # VQ loss (from encoder)
    base_loss = base_loss + config.w_vq * vq_loss
    metrics["vq"] = vq_loss.item()

    # Local routing entropy: minimize H(K|X) for high-confidence assignments.
    loss_entropy = compute_routing_entropy(router_reg_weights)
    base_loss = base_loss + config.w_entropy * loss_entropy
    metrics["entropy"] = loss_entropy.item()
    if config.w_router_margin > 0 and router_scores is not None:
        loss_margin = compute_router_margin_loss(
            router_scores,
            margin=config.router_margin_target,
        )
        base_loss = base_loss + config.w_router_margin * loss_margin
        metrics["router_margin"] = loss_margin.item()
    else:
        metrics["router_margin"] = 0.0
    if config.w_hard_routing_nll > 0 and router_scores is not None:
        loss_hard_nll = compute_hard_routing_nll(router_scores)
        base_loss = base_loss + config.w_hard_routing_nll * loss_hard_nll
        metrics["hard_routing_nll"] = loss_hard_nll.item()
    else:
        metrics["hard_routing_nll"] = 0.0
    info_metrics = compute_router_information_metrics(router_reg_weights)
    metrics.update({k: v.item() for k, v in info_metrics.items()})
    sharpness_metrics = compute_router_sharpness_metrics(router_reg_weights)
    metrics.update({k: v.item() for k, v in sharpness_metrics.items()})

    if config.w_chart_ot > 0:
        if router_scores is None:
            router_scores = torch.log(router_reg_weights.clamp(min=1e-8))
        loss_chart_ot, chart_ot_metrics = compute_sinkhorn_balanced_chart_loss(
            router_scores,
            epsilon=config.chart_ot_epsilon,
            num_iters=config.chart_ot_iters,
        )
        base_loss = base_loss + config.w_chart_ot * loss_chart_ot
        metrics["chart_ot"] = loss_chart_ot.item()
        metrics.update(chart_ot_metrics)
    else:
        metrics["chart_ot"] = 0.0

    # Chart usage should reflect the actual forward assignment, so it is
    # measured on the hard/ST router tensor rather than the live soft router.
    loss_chart_usage, chart_usage_metrics = compute_chart_usage_band_loss(
        usage_router_weights,
        config.num_charts,
        h_low=config.chart_usage_entropy_low,
        h_high=config.chart_usage_entropy_high,
    )
    base_loss = base_loss + config.w_diversity * loss_chart_usage
    metrics["chart_usage"] = loss_chart_usage.item()
    metrics.update(chart_usage_metrics)

    # z_n regularization terms (flow through z_geo -> z_n)
    zn_reg_loss = torch.zeros((), device=x.device)

    # Hyperbolic uniformity
    if config.w_uniformity > 0:
        loss_unif = compute_hyperbolic_uniformity_loss(z_geo)
        zn_reg_loss = zn_reg_loss + config.w_uniformity * loss_unif
        metrics["uniformity"] = loss_unif.item()

    # Radial calibration
    if config.w_radial_calibration > 0:
        radial_latent = z_geo
        radial_center = c_bar
        if radial_center is None and v_local is not None:
            radial_latent = v_local
        if radial_center is not None:
            metrics["local_radius_mean"] = hyperbolic_distance(
                _project_to_ball(z_geo),
                _project_to_ball(radial_center),
            ).mean().item()
        elif v_local is not None:
            origin = torch.zeros_like(v_local)
            metrics["local_radius_mean"] = hyperbolic_distance(
                _project_to_ball(v_local), origin,
            ).mean().item()
        else:
            metrics["local_radius_mean"] = 0.0
        loss_radcal = compute_radial_calibration_loss(
            radial_latent,
            router_reg_weights.detach(),
            config.num_charts,
            center_points=radial_center.detach() if radial_center is not None else None,
            quality_target=quality_target.detach(),
            quality_mix=quality_mix,
            quality_base_weight=quality_base_weight,
            rho_max=config.radial_calibration_rho_max,
            rho_band_width=config.radial_calibration_band_width,
            use_hyperbolic_radius=True,
        )
        zn_reg_loss = zn_reg_loss + config.w_radial_calibration * loss_radcal
        metrics["radial_cal"] = loss_radcal.item()
    else:
        metrics["local_radius_mean"] = 0.0

    if config.w_confidence_calibration > 0:
        loss_confcal = compute_confidence_calibration_loss(
            router_reg_weights,
            quality_target.detach(),
            config.num_charts,
        )
        base_loss = base_loss + config.w_confidence_calibration * loss_confcal
        metrics["confidence_calibration"] = loss_confcal.item()

    if config.w_v_tangent_barrier > 0 and v_raw is not None:
        loss_v_tangent = compute_v_tangent_barrier_loss(
            v_raw,
            target_radius=config.v_tangent_barrier_radius,
        )
        base_loss = base_loss + config.w_v_tangent_barrier * loss_v_tangent
        metrics["v_tangent_barrier"] = loss_v_tangent.item()

    # Codebook spread
    if config.w_codebook_spread > 0 and hasattr(atlas_encoder, "codebook"):
        codebook = atlas_encoder.codebook
        loss_spread = compute_codebook_spread_loss(
            codebook, margin=config.w_codebook_spread_margin,
        )
        base_loss = base_loss + config.w_codebook_spread * loss_spread
        metrics["codebook_spread"] = loss_spread.item()

    if config.w_codebook_center > 0 and hasattr(atlas_encoder, "codebook"):
        codebook = atlas_encoder.codebook
        loss_center = compute_codebook_centering_loss(codebook)
        base_loss = base_loss + config.w_codebook_center * loss_center
        metrics["codebook_center"] = loss_center.item()

    if hasattr(atlas_encoder, "chart_centers"):
        chart_centers = atlas_encoder.chart_centers
        if config.w_chart_center_mean > 0:
            loss_chart_center_mean = compute_chart_center_mean_loss(chart_centers)
            base_loss = base_loss + config.w_chart_center_mean * loss_chart_center_mean
            metrics["chart_center_mean"] = loss_chart_center_mean.item()
        if config.w_chart_center_radius > 0:
            loss_chart_center_radius = compute_chart_center_radius_loss(
                chart_centers,
                radius_max=config.chart_center_radius_max,
            )
            base_loss = base_loss + config.w_chart_center_radius * loss_chart_center_radius
            metrics["chart_center_radius"] = loss_chart_center_radius.item()
        if config.w_chart_center_sep > 0:
            loss_chart_center_sep = compute_chart_center_separation_loss(
                chart_centers,
                margin=config.chart_center_sep_margin,
            )
            base_loss = base_loss + config.w_chart_center_sep * loss_chart_center_sep
            metrics["chart_center_sep"] = loss_chart_center_sep.item()

    # Code usage is measured with a straight-through code assignment so the
    # forward value reflects actual selected codes while gradients still flow.
    if config.w_code_collapse > 0 and hasattr(atlas_encoder, "codebook"):
        if v_local is None:
            msg = "Phase 1 code-usage loss requires the encoder's chart-local latent."
            raise RuntimeError(msg)
        codebook = atlas_encoder.codebook
        loss_code_usage, code_usage_metrics = compute_code_usage_band_loss(
            v_local, codebook, usage_router_weights,
            hard_code_indices=indices_stack,
            h_low=config.code_usage_entropy_low,
            h_high=config.code_usage_entropy_high,
            temperature=config.w_code_collapse_temperature,
        )
        base_loss = base_loss + config.w_code_collapse * loss_code_usage
        metrics["code_usage"] = loss_code_usage.item()
        metrics.update(code_usage_metrics)

    # Window loss
    if config.w_window > 0:
        loss_window, _ = compute_window_loss(
            router_reg_weights, config.num_charts,
            eps_ground=config.w_window_eps_ground,
        )
        base_loss = base_loss + config.w_window * loss_window
        metrics["window"] = loss_window.item()
    else:
        metrics["window"] = 0.0

    # Encoder-decoder routing consistency
    if config.w_consistency > 0:
        eps = 1e-6
        kl = (enc_router_weights * torch.log(
            (enc_router_weights + eps) / (dec_router_weights + eps)
        )).sum(dim=-1).mean()
        base_loss = base_loss + config.w_consistency * kl
        metrics["consistency"] = kl.item()
    else:
        metrics["consistency"] = 0.0

    metrics.setdefault("uniformity", 0.0)
    metrics.setdefault("radial_cal", 0.0)
    metrics.setdefault("confidence_calibration", 0.0)
    metrics.setdefault("router_margin", 0.0)
    metrics.setdefault("hard_routing_nll", 0.0)
    metrics.setdefault("v_tangent_barrier", 0.0)
    metrics.setdefault("codebook_spread", 0.0)
    metrics.setdefault("codebook_center", 0.0)
    metrics.setdefault("chart_center_mean", 0.0)
    metrics.setdefault("chart_center_radius", 0.0)
    metrics.setdefault("chart_center_sep", 0.0)
    metrics.setdefault("chart_ot", 0.0)
    metrics.setdefault("ot_target_top1_mean", 0.0)
    metrics.setdefault("ot_plan_col_l1", 0.0)
    metrics.setdefault("ot_plan_row_l1", 0.0)
    metrics.setdefault("code_usage", 0.0)
    metrics.setdefault("H_code_usage", 0.0)
    metrics.setdefault("code_usage_perplexity", 1.0)
    metrics.setdefault("active_code_charts", 0.0)
    metrics.setdefault("recon_quality_mean", 0.0)
    metrics.setdefault("vq_quality_mean", 0.0)
    metrics.setdefault("combined_quality_mean", 0.0)
    metrics.setdefault("routing_confidence_mean", 0.0)
    metrics.setdefault("radial_target_mean", 0.0)
    metrics.setdefault("local_radius_mean", 0.0)

    total = base_loss + zn_reg_loss
    metrics["total"] = total.item()
    return base_loss, zn_reg_loss, metrics


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
        total_loss, metrics dict.
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
                wm_output["phi_eff"], wm_output["momenta"], wm_output["z_trajectory"],
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


def compute_phase3_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    vq_loss: torch.Tensor,
    enc_router_weights: torch.Tensor,
    dec_router_weights: torch.Tensor,
    z_geo: torch.Tensor,
    encoder: torch.nn.Module,
    wm_output: dict[str, torch.Tensor],
    z_targets: torch.Tensor,
    chart_targets: torch.Tensor,
    config: VLAConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Assemble Phase 3 (joint fine-tuning) loss.

    Combines scaled encoder loss and dynamics loss, returning split
    components for gradient surgery.

    Returns:
        base_enc_loss: Encoder base loss (no z_n gradient path).
        zn_reg_loss: z_n regularization loss (uniformity, radial_cal).
        dyn_loss: Dynamics loss from world model.
        metrics: Dict of individual loss components for logging.
    """
    base_enc, zn_reg, enc_metrics = compute_phase1_loss(
        x, x_recon, vq_loss, enc_router_weights, dec_router_weights,
        z_geo, encoder, config,
    )
    loss_dyn, dyn_metrics = compute_phase2_loss(
        wm_output, z_targets, chart_targets, config,
    )

    total = (
        config.phase3_encoder_scale * base_enc
        + config.phase3_zn_reg_scale * zn_reg
        + config.phase3_dynamics_scale * loss_dyn
    )

    metrics: dict[str, float] = {}
    for k, v in enc_metrics.items():
        metrics[f"enc/{k}"] = v
    for k, v in dyn_metrics.items():
        metrics[f"dyn/{k}"] = v
    metrics["total"] = total.item()

    return base_enc, zn_reg, loss_dyn, metrics
