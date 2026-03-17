"""Covariant geometric world model with geodesic Boris-BAOAB integration on the Poincare ball.

Replaces the MLP-based sub-modules of the original GeometricWorldModel with
hyperbolic covariant attention layers, ensuring all force computations
respect the Poincare geometry.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.attention import CovariantAttention, GeodesicConfig
from fragile.layers.gauge import (
    christoffel_contraction,
    ConformalMetric,
    hyperbolic_distance,
    poincare_exp_map,
    project_to_ball,
    RiskAdaptiveConformalMetric,
)
from fragile.layers.primitives import SpectralLinear


def compute_risk_tensor(
    force: torch.Tensor,
    curl_tensor: torch.Tensor | None = None,
    lambda_inv_sq: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute risk tensor T = T_grad + T_maxwell.

    Args:
        force: [B, D] conservative force vector.
        curl_tensor: [B, D, D] antisymmetric field strength, or None.
        lambda_inv_sq: [B, 1] scalar inverse-squared conformal factor lambda^{-2}.

    Returns:
        T: [B, D, D] risk tensor (symmetric).
    """
    # Gradient stress: T_grad = f (x) f
    T = torch.einsum("bi,bj->bij", force, force)  # [B, D, D]

    if curl_tensor is not None and lambda_inv_sq is not None:
        # Maxwell stress: T_maxwell = F_ik F^k_j - 1/4 delta_ij F_kl F^kl
        # F^k_j = G^{km} F_{mj} = lambda^{-2} * F_{mj} (conformal metric is diagonal)
        F_up = lambda_inv_sq.unsqueeze(-1) * curl_tensor  # [B, D, D] scalar broadcast
        # F_ik F^k_j
        FF = torch.bmm(curl_tensor, F_up)  # [B, D, D]
        # Trace: F_kl F^kl
        trace_FF = (
            torch.diagonal(FF, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
        )  # [B, 1, 1]
        D = force.shape[1]
        eye = torch.eye(D, device=force.device, dtype=force.dtype).unsqueeze(0)
        T_maxwell = FF - 0.25 * trace_FF * eye
        T = T + T_maxwell

    return T


# ---------------------------------------------------------------------------
# Tokenizers
# ---------------------------------------------------------------------------


class ActionTokenizer(nn.Module):
    """Lifts a flat action [B, A] into N context tokens for cross-attention."""

    def __init__(self, action_dim: int, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.weight = nn.Parameter(torch.randn(action_dim, d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(action_dim, d_model) * 0.02)

    def forward(
        self,
        action: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize action into context tokens.

        Args:
            action: [B, A] action vector.
            z: [B, D] latent position.

        Returns:
            tokens_x: [B, A, d_model] feature tokens.
            tokens_z: [B, A, D] position tokens (all at z).
        """
        tokens_x = action.unsqueeze(-1) * self.weight.unsqueeze(0) + self.pos_embed.unsqueeze(
            0
        )  # [B, A, d_model]
        tokens_z = z.unsqueeze(1).expand(-1, self.action_dim, -1).contiguous()  # [B, A, D]
        return tokens_x, tokens_z


class ChartTokenizer(nn.Module):
    """Builds chart context tokens from router weights."""

    def __init__(self, num_charts: int, d_model: int, latent_dim: int) -> None:
        super().__init__()
        self.chart_embeddings = nn.Parameter(
            torch.randn(num_charts, d_model) * 0.02,
        )
        centers = F.normalize(torch.randn(num_charts, latent_dim), dim=-1) * 0.3
        self.chart_centers = nn.Parameter(centers)

    def forward(
        self,
        rw: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize chart weights into context tokens.

        Args:
            rw: [B, K] router / chart weights.
            z: [B, D] latent position (unused, kept for API consistency).

        Returns:
            tokens_x: [B, K, d_model] feature tokens.
            tokens_z: [B, K, D] chart center positions.
        """
        B = rw.shape[0]
        tokens_x = rw.unsqueeze(-1) * self.chart_embeddings.unsqueeze(0)  # [B, K, d_model]
        # Project chart centers to stay inside the Poincare ball
        safe_centers = project_to_ball(self.chart_centers)  # [K, D]
        tokens_z = safe_centers.unsqueeze(0).expand(B, -1, -1).contiguous()  # [B, K, D]
        return tokens_x, tokens_z


# ---------------------------------------------------------------------------
# Covariant sub-modules
# ---------------------------------------------------------------------------


class CovariantPotentialNet(nn.Module):
    r"""Direct force prediction + scalar potential for diagnostics.

    Produces a cotangent force vector for the BAOAB B-step **without**
    ``torch.autograd.grad`` (no second-order graph), plus the scalar
    effective potential :math:`\Phi_{\text{eff}}` for the energy
    conservation loss.

    Force decomposition (cotangent vector):

    .. math::
        F = \alpha\,\partial U/\partial z
          + (1-\alpha)\, f_{\text{critic}}(z, K)
          + \gamma_{\text{risk}}\, f_{\text{risk}}(z, K)

    where :math:`\partial U/\partial z = -2z\,/\,(|z|\,(1-|z|^2))` is
    the *analytical* gradient of the hyperbolic drive (no learnable
    params, no autograd).  The learned force components are predicted
    directly by CovariantAttention heads.

    Scalar potential (for energy conservation loss only):

    .. math::
        \Phi_{\text{eff}} = \alpha\, U(z)
                          + (1-\alpha)\, V_{\text{critic}}(z, K)
                          + \gamma_{\text{risk}}\, \Psi_{\text{risk}}(z, K)
    """

    def __init__(
        self,
        latent_dim: int,
        num_charts: int,
        d_model: int,
        alpha: float = 0.5,
        gamma_risk: float = 0.01,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma_risk = gamma_risk
        self.latent_dim = latent_dim

        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)

        # Critic head: shared attention → force vector + scalar V
        self.v_critic_attn = CovariantAttention(geo_cfg)
        self.v_force_out = SpectralLinear(d_model, latent_dim)
        self.v_out = SpectralLinear(d_model, 1)

        # Risk head: shared attention → force vector + scalar Ψ
        self.psi_risk_attn = CovariantAttention(geo_cfg)
        self.psi_force_out = SpectralLinear(d_model, latent_dim)
        self.psi_out = SpectralLinear(d_model, 1)

    def _analytic_U_and_grad(
        self,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""Analytic hyperbolic drive and its exact gradient.

        .. math::
            U(z) = -2\operatorname{artanh}(|z|)

            \frac{\partial U}{\partial z_i}
                = \frac{-2\, z_i}{|z|\,(1 - |z|^2)}

        Returns:
            U: [B, 1] scalar potential.
            dU_dz: [B, D] Euclidean gradient (cotangent vector).
        """
        r = z.norm(dim=-1, keepdim=True).clamp(min=1e-6, max=1.0 - 1e-6)
        U = -2.0 * torch.atanh(r)  # [B, 1]
        dU_dz = -2.0 * z / (r * (1.0 - r**2))  # [B, D]
        return U, dU_dz

    def _value_features(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return shared attention features for task and risk scalars."""
        ctx_x, ctx_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        v_feat, _ = self.v_critic_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        psi_feat, _ = self.psi_risk_attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return v_feat, psi_feat

    def task_value(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Task-value scalar used by policy improvement and jump selection."""
        v_feat, _ = self._value_features(z, rw)
        return self.v_out(v_feat)

    def force_terms(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the direct-force decomposition used by BAOAB."""
        U, dU_dz = self._analytic_U_and_grad(z)
        v_feat, psi_feat = self._value_features(z, rw)
        task_force_raw = self.v_force_out(v_feat)
        task_value = self.v_out(v_feat)
        risk_force_raw = self.psi_force_out(psi_feat)
        risk_value = self.psi_out(psi_feat)
        analytic_force = self.alpha * dU_dz
        task_force = (1.0 - self.alpha) * task_force_raw
        risk_force = self.gamma_risk * risk_force_raw
        return {
            "analytic_potential": U,
            "analytic_force": analytic_force,
            "task_value": task_value,
            "task_force_raw": task_force_raw,
            "task_force": task_force,
            "risk_value": risk_value,
            "risk_force_raw": risk_force_raw,
            "risk_force": risk_force,
            "force": analytic_force + task_force + risk_force,
            "phi": self.alpha * U + (1.0 - self.alpha) * task_value + self.gamma_risk * risk_value,
        }

    def exact_force_terms(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return the exact scalar-gradient conservative field for certification."""
        z_req = z.detach().requires_grad_(True)
        rw_detached = rw.detach()
        U, dU_dz = self._analytic_U_and_grad(z_req)
        v_feat, psi_feat = self._value_features(z_req, rw_detached)
        task_value = self.v_out(v_feat)
        risk_value = self.psi_out(psi_feat)
        task_force_raw = torch.autograd.grad(
            task_value.sum(),
            z_req,
            retain_graph=True,
            create_graph=False,
        )[0]
        risk_force_raw = torch.autograd.grad(
            risk_value.sum(),
            z_req,
            retain_graph=False,
            create_graph=False,
        )[0]
        analytic_force = self.alpha * dU_dz
        task_force = (1.0 - self.alpha) * task_force_raw
        risk_force = self.gamma_risk * risk_force_raw
        return {
            "analytic_potential": U.detach(),
            "analytic_force": analytic_force.detach(),
            "task_value": task_value.detach(),
            "task_force_raw": task_force_raw.detach(),
            "task_force": task_force.detach(),
            "risk_value": risk_value.detach(),
            "risk_force_raw": risk_force_raw.detach(),
            "risk_force": risk_force.detach(),
            "force": (analytic_force + task_force + risk_force).detach(),
            "phi": (
                self.alpha * U + (1.0 - self.alpha) * task_value + self.gamma_risk * risk_value
            ).detach(),
        }

    def effective_potential(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Effective structural potential used for energy diagnostics.

        Args:
            z: [B, D] position in the Poincare ball.
            rw: [B, K] router / chart weights.

        Returns:
            phi_eff: [B, 1] effective structural potential.
        """
        U, _ = self._analytic_U_and_grad(z)
        v_feat, psi_feat = self._value_features(z, rw)
        V = self.v_out(v_feat)
        Psi = self.psi_out(psi_feat)

        return self.alpha * U + (1.0 - self.alpha) * V + self.gamma_risk * Psi

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Return task value for value-field uses."""
        return self.task_value(z, rw)

    def force_and_potential(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Direct force prediction + scalar potential (no autograd).

        The force is a cotangent vector for the Hamiltonian momentum kick
        (dp/dt = -force + u_pi).  The scalar potential is for the energy
        conservation diagnostic.

        Args:
            z: [B, D] position.
            rw: [B, K] router weights.

        Returns:
            force: [B, D] conservative force (cotangent vector).
            phi: [B, 1] scalar effective potential.
        """
        force_info = self.force_terms(z, rw)
        return force_info["force"], force_info["phi"]


class CovariantControlField(nn.Module):
    """Action-conditioned control force in tangent space using CovariantAttention."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        """Compute control force u_pi.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.
            rw: [B, K] chart weights.

        Returns:
            u: [B, D] control force in tangent space.
        """
        act_x, act_z = self.action_tok(action, z)
        chart_x, chart_z = self.chart_tok(rw, z)

        # Concatenate action + chart context tokens
        ctx_x = torch.cat([act_x, chart_x], dim=1)  # [B, A+K, d_model]
        ctx_z = torch.cat([act_z, chart_z], dim=1)  # [B, A+K, D]

        x_q = self.z_embed(z)
        output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return self.out(output)


class CovariantValueCurl(nn.Module):
    r"""Predicts antisymmetric field strength :math:`\mathcal{F}_{ij}` for Boris rotation.

    Uses CovariantAttention over action tokens instead of an MLP.
    """

    def __init__(self, latent_dim: int, action_dim: int, d_model: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_upper = latent_dim * (latent_dim - 1) // 2

        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, self.n_upper)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        self.register_buffer(
            "_tri_rows",
            torch.triu_indices(latent_dim, latent_dim, offset=1)[0],
        )
        self.register_buffer(
            "_tri_cols",
            torch.triu_indices(latent_dim, latent_dim, offset=1)[1],
        )

    def forward(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute antisymmetric field strength tensor.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.

        Returns:
            F_mat: [B, D, D] antisymmetric tensor.
        """
        B = z.shape[0]
        D = self.latent_dim

        act_x, act_z = self.action_tok(action, z)
        x_q = self.z_embed(z)
        output, _ = self.attn(z, act_z, x_q, act_x, act_x)
        upper = self.out(output)  # [B, n_upper]

        F_mat = z.new_zeros(B, D, D)
        F_mat[:, self._tri_rows, self._tri_cols] = upper
        F_mat[:, self._tri_cols, self._tri_rows] = -upper
        return F_mat


class CovariantChartTarget(nn.Module):
    """Predicts target chart logits for the jump process using CovariantAttention.

    Uses ChartTokenizer + ActionTokenizer → CovariantAttention → output projection,
    matching the pattern of CovariantControlField.  This allows the module to
    condition on the current chart weights ``rw`` via the attention mechanism.
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.num_charts = num_charts

        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, num_charts)
        self.z_embed = SpectralLinear(latent_dim, d_model)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        """Predict chart transition logits.

        Args:
            z: [B, D] position.
            action: [B, A] action vector.
            rw: [B, K] current chart weights.

        Returns:
            logits: [B, K] chart logits.
        """
        act_x, act_z = self.action_tok(action, z)
        chart_x, chart_z = self.chart_tok(rw, z)

        # Concatenate action + chart context tokens
        ctx_x = torch.cat([act_x, chart_x], dim=1)  # [B, A+K, d_model]
        ctx_z = torch.cat([act_z, chart_z], dim=1)  # [B, A+K, D]

        x_q = self.z_embed(z)
        output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return self.out(output)  # [B, K]


class CovariantJumpRate(nn.Module):
    """Predicts Poisson jump rate lambda(z, K) >= 0 using CovariantAttention."""

    def __init__(
        self,
        latent_dim: int,
        num_charts: int,
        d_model: int,
    ) -> None:
        super().__init__()
        self.chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.out = SpectralLinear(d_model, 1)
        self.z_embed = SpectralLinear(latent_dim, d_model)

    def forward(self, z: torch.Tensor, rw: torch.Tensor) -> torch.Tensor:
        """Compute jump rate.

        Args:
            z: [B, D] position.
            rw: [B, K] chart weights.

        Returns:
            rate: [B, 1] non-negative jump rate.
        """
        ctx_x, ctx_z = self.chart_tok(rw, z)
        x_q = self.z_embed(z)
        output, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return F.softplus(self.out(output))


class CovariantMomentumInit(nn.Module):
    """Initializes momentum from position with conformal metric scaling."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.net = SpectralLinear(latent_dim, latent_dim)
        self.metric = ConformalMetric()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize momentum from position.

        Args:
            z: [B, D] position in the Poincare ball.

        Returns:
            p: [B, D] initial momentum scaled by conformal factor.
        """
        lam_sq = self.metric.conformal_factor(z) ** 2  # [B, 1]
        return lam_sq * self.net(z)


class HodgeDecomposer(nn.Module):
    """Decomposes total force into conservative, solenoidal, and harmonic parts.

    Hodge decomposition: f_total = f_conservative + f_solenoidal + f_harmonic

    - f_conservative = -nabla V (gradient of critic potential, from potential_net)
    - f_solenoidal = beta * F * G^{-1} * p (from Boris/curl field strength)
    - f_harmonic = f_total - f_conservative - f_solenoidal (residual)

    The harmonic component should be small if the model is well-structured.
    """

    def __init__(self, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

    def forward(
        self,
        force_total: torch.Tensor,
        force_conservative: torch.Tensor,
        force_solenoidal: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute Hodge decomposition and diagnostics.

        Args:
            force_total: [B, D] total non-control force.
            force_conservative: [B, D] conservative part (-nabla Phi from potential_net).
            force_solenoidal: [B, D] solenoidal part (Boris curl force).

        Returns:
            Dictionary with:
                harmonic: [B, D] harmonic residual
                conservative_ratio: [B] ||f_cons|| / ||f_total||
                solenoidal_ratio: [B] ||f_sol|| / ||f_total||
                harmonic_ratio: [B] ||f_harm|| / ||f_total||
        """
        f_harmonic = force_total - force_conservative - force_solenoidal

        eps = 1e-8
        total_norm = force_total.norm(dim=-1).clamp(min=eps)  # [B]
        cons_norm = force_conservative.norm(dim=-1)  # [B]
        sol_norm = force_solenoidal.norm(dim=-1)  # [B]
        harm_norm = f_harmonic.norm(dim=-1)  # [B]

        return {
            "harmonic": f_harmonic,
            "conservative_ratio": cons_norm / total_norm,
            "solenoidal_ratio": sol_norm / total_norm,
            "harmonic_ratio": harm_norm / total_norm,
        }


# ---------------------------------------------------------------------------
# Main world model
# ---------------------------------------------------------------------------


class GeometricWorldModel(nn.Module):
    r"""Action-conditioned dynamics model using geodesic Boris-BAOAB integration.

    Drop-in replacement for the MLP-based GeometricWorldModel, using
    CovariantAttention-based sub-modules that respect Poincare geometry.

    When ``min_length > 0``, three CFL-derived bounds are enforced using
    smooth squashing maps (same :math:`\psi` from the Euclidean Gas theory):

    .. math::
        V_{\text{alg}} = \frac{\ell_{\min}}{\Delta t},\qquad
        F_{\max} = \frac{2\gamma\,\ell_{\min}}{\Delta t},\qquad
        \lambda_{\max} = \frac{2\,\ell_{\min}}{c_2\,\sqrt{D}\,\Delta t}

    Args:
        latent_dim: Dimension of the Poincare ball latent space.
        action_dim: Dimension of the action vector.
        num_charts: Number of atlas charts.
        d_model: Feature dimension for CovariantAttention modules.
        hidden_dim: Kept for API compatibility (unused).
        dt: Integration time step.
        gamma_friction: Langevin friction coefficient.
        T_c: Thermostat temperature.
        alpha_potential: Balance between analytic drive U and learned critic V.
        beta_curl: Value-curl coupling strength for Lorentz/Boris forces.
        gamma_risk: Risk-stress penalty weight in the effective potential.
        use_boris: Whether to enable Boris rotation for curl forces.
        use_jump: Whether to enable the conditional chart jump (WFR Fisher-Rao).
        n_refine_steps: Number of BAOAB sub-steps per horizon step (WFR W2).
        jump_beta: Inverse temperature for Boltzmann chart selection.
        min_length: Minimum resolvable geodesic length scale.
            When > 0, derives F_max, V_alg and cf_max from CFL conditions.
            When 0, all squashing is disabled (backward compat).
    """

    def __init__(
        self,
        latent_dim: int = 16,
        action_dim: int = 14,
        control_dim: int | None = None,
        num_charts: int = 8,
        d_model: int = 128,
        hidden_dim: int = 256,
        dt: float = 0.01,
        gamma_friction: float = 1.0,
        T_c: float = 0.1,
        alpha_potential: float = 0.5,
        beta_curl: float = 0.1,
        gamma_risk: float = 0.01,
        use_boris: bool = True,
        use_jump: bool = True,
        n_refine_steps: int = 3,
        jump_beta: float = 1.0,
        min_length: float = 0.0,
        risk_metric_alpha: float = 0.0,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.control_dim = control_dim if control_dim is not None else action_dim
        self.num_charts = num_charts
        self.dt = dt
        self.gamma = gamma_friction
        self.T_c = T_c
        self.beta_curl = beta_curl
        self.use_boris = use_boris
        self.use_jump = use_jump
        self.n_refine_steps = n_refine_steps
        self.jump_beta = jump_beta
        self.min_length = min_length
        self.risk_metric_alpha = risk_metric_alpha

        # Ornstein-Uhlenbeck coefficients
        self.c1 = math.exp(-gamma_friction * dt)
        self.c2 = math.sqrt(max(0.0, (1.0 - self.c1**2) * T_c))

        # CFL-derived bounds from minimum length scale
        if min_length > 0:
            self.V_alg = min_length / dt
            self.F_max = 2.0 * gamma_friction * min_length / dt
            # cf_max = 2 * ℓ_min / (c2 * √D * dt); floor at 2.0 (origin value)
            denom = max(self.c2, 1e-12) * math.sqrt(latent_dim) * dt
            self.cf_max = max(2.0 * min_length / denom, 2.0)
        else:
            self.V_alg = 0.0
            self.F_max = 0.0
            self.cf_max = 0.0

        # Metric
        if risk_metric_alpha > 0:
            self.metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=risk_metric_alpha)
        else:
            self.metric = ConformalMetric()

        # Covariant sub-modules
        self.potential_net = CovariantPotentialNet(
            latent_dim,
            num_charts,
            d_model,
            alpha=alpha_potential,
            gamma_risk=gamma_risk,
        )
        if self.control_dim == latent_dim:
            self.control_lift: nn.Module = nn.Identity()
        else:
            self.control_lift = SpectralLinear(self.control_dim, latent_dim)
        if use_boris:
            self.curl_net = CovariantValueCurl(latent_dim, self.control_dim, d_model)
        else:
            self.curl_net = None

        self.chart_predictor = CovariantChartTarget(
            latent_dim,
            self.control_dim,
            num_charts,
            d_model,
        )

        # Initial momentum from position
        self.momentum_init = CovariantMomentumInit(latent_dim)

        # Hodge decomposition diagnostic (no learnable parameters)
        self.hodge_decomposer = HodgeDecomposer(latent_dim)

    def _chart_tokenizers(self) -> list[ChartTokenizer]:
        """Return every chart tokenizer that should share the Phase-1 atlas."""
        return [
            self.potential_net.chart_tok,
            self.chart_predictor.chart_tok,
        ]

    @torch.no_grad()
    def bind_chart_centers(
        self,
        chart_centers: torch.Tensor,
        freeze: bool = True,
    ) -> None:
        """Copy external atlas centers into every chart-conditioned submodule.

        Phase 2 keeps the encoder/router atlas fixed and learns dynamics on top
        of that geometry. The world model should therefore consume the same
        chart centers instead of inventing its own atlas.
        """
        safe_centers = project_to_ball(chart_centers.detach())
        for tok in self._chart_tokenizers():
            tok.chart_centers.copy_(
                safe_centers.to(device=tok.chart_centers.device, dtype=tok.chart_centers.dtype)
            )
            tok.chart_centers.requires_grad_(not freeze)

    # -----------------------------------------------------------------
    # Boris rotation
    # -----------------------------------------------------------------

    def _boris_rotation(
        self,
        p_minus: torch.Tensor,
        z: torch.Tensor,
        action: torch.Tensor,
        curl_F: torch.Tensor | None = None,
        lambda_inv_sq: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Norm-preserving Boris rotation for the Lorentz/curl force.

        For D > 3 the cross product generalises to the antisymmetric
        matrix-vector product.  The Boris algorithm ensures
        ``|p_plus| = |p_minus|`` because F is antisymmetric.

        Args:
            p_minus: [B, D] momentum after potential half-kick.
            z: [B, D] current position.
            action: [B, A] action vector.
            curl_F: [B, D, D] pre-computed curl tensor, or None to compute.
            lambda_inv_sq: [B, 1] scalar inverse-squared conformal factor, or None to compute.

        Returns:
            p_plus: [B, D] rotated momentum (same norm).
            curl_F: [B, D, D] the curl tensor used (for reuse), or None.
        """
        if self.curl_net is None:
            return p_minus, None

        h = self.dt
        F = curl_F if curl_F is not None else self.curl_net(z, action)  # [B, D, D] antisymmetric
        if lambda_inv_sq is None:
            cf = self.metric.conformal_factor(z)  # [B, 1]
            lambda_inv_sq = 1.0 / (cf**2 + self.metric.epsilon)  # [B, 1]
        T = (h / 2.0) * self.beta_curl * lambda_inv_sq.unsqueeze(-1) * F  # [B, D, D]

        # Boris half-rotation
        t_vec = torch.bmm(T, p_minus.unsqueeze(-1)).squeeze(-1)  # [B, D]
        p_prime = p_minus + t_vec

        t_sq = (T**2).sum(dim=(-2, -1), keepdim=False)  # [B] Frobenius norm^2
        s_factor = 2.0 / (1.0 + t_sq).unsqueeze(-1)  # [B, 1]
        s_vec = s_factor * torch.bmm(T, p_prime.unsqueeze(-1)).squeeze(-1)  # [B, D]

        return p_minus + s_vec, F

    # -----------------------------------------------------------------
    # BAOAB integration step
    # -----------------------------------------------------------------

    def _baoab_step(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action_canonical: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """One geodesic Boris-BAOAB integration step.

        Fixes five geometry bugs in the original skeleton:
        1. Uses ``poincare_exp_map`` at z (not origin-based).
        2. Includes Christoffel geodesic correction.
        3. Boris rotation for Lorentz/curl forces.
        4. Structured effective potential with Riemannian gradient.
        5. Spectral-normalised layers throughout.

        Args:
            z: [B, D] current position.
            p: [B, D] current momentum (cotangent vector).
            action_canonical: [B, C] canonical motor-side action latent.
            rw: [B, K] chart weights.

        Returns:
            z_next: [B, D] updated position.
            p_next: [B, D] updated momentum.
            phi_eff: [B, 1] effective potential at the step.
            hodge_info: Hodge decomposition diagnostics dict (may be empty).
        """
        h = self.dt
        use_risk = self.risk_metric_alpha > 0

        # --- B step (first half): momentum kick ---
        # Direct force prediction (cotangent vector, no autograd).
        # The canonical action latent is lifted into the same cotangent space.
        force, _ = self.potential_net.force_and_potential(z, rw)
        u_pi = self.control_lift(action_canonical)  # [B, D] lifted motor drive
        kick = force - u_pi

        # Pre-compute curl tensor once for both Boris rotation and risk tensor
        curl_F = None
        if self.curl_net is not None:
            curl_F = self.curl_net(z, action_canonical)

        # Compute risk tensor for metric adaptation
        risk_T = None
        lis_base = None  # lambda^{-2} at current z (base metric, no risk)
        if use_risk:
            cf_base = self.metric.conformal_factor(z)  # [B, 1] -- base ConformalMetric
            lis_base = 1.0 / (cf_base**2 + self.metric.epsilon)  # [B, 1]
            risk_T = compute_risk_tensor(force, curl_F, lis_base)

        # ψ_F: smooth force squashing (1-Lipschitz, C∞, direction-preserving)
        if self.F_max > 0:
            kick = self.F_max * kick / (self.F_max + kick.norm(dim=-1, keepdim=True))

        p_minus = p - (h / 2.0) * kick
        # pass pre-computed lambda_inv_sq (None in non-risk case, boris computes its own)
        p_plus, _ = self._boris_rotation(
            p_minus,
            z,
            action_canonical,
            curl_F=curl_F,
            lambda_inv_sq=lis_base,
        )

        # Hodge decomposition: solenoidal force = Boris impulse / dt
        hodge_info: dict[str, torch.Tensor] = {}
        boris_impulse = p_plus - p_minus  # [B, D]
        f_solenoidal = boris_impulse / max(h, 1e-8)
        f_total_no_ctrl = force + f_solenoidal
        hodge_info = self.hodge_decomposer(f_total_no_ctrl, force, f_solenoidal)

        p = p_plus - (h / 2.0) * kick

        # --- A step (first half): geodesic drift ---
        if use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        lambda_inv_sq = 1.0 / (cf**2 + self.metric.epsilon)  # [B, 1]
        v = lambda_inv_sq * p  # [B, D] contravariant velocity
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        # ψ_v: smooth velocity squashing (V_alg = ℓ_min / Δt)
        if self.V_alg > 0:
            v_corr = self.V_alg * v_corr / (self.V_alg + v_corr.norm(dim=-1, keepdim=True))
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = project_to_ball(z)

        # --- O step: Ornstein-Uhlenbeck thermostat ---
        if use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        if self.cf_max > 0:
            # Smooth conformal factor cap via tanh: preserves interior values,
            # smoothly saturates near boundary
            cf = self.cf_max * torch.tanh(cf / self.cf_max)
        xi = torch.randn_like(p)
        p = self.c1 * p + self.c2 * cf * xi

        # --- A step (second half): geodesic drift ---
        if use_risk:
            cf = self.metric.conformal_factor(z, risk_tensor=risk_T)  # [B, 1]
        else:
            cf = self.metric.conformal_factor(z)  # [B, 1]
        lambda_inv_sq = 1.0 / (cf**2 + self.metric.epsilon)  # [B, 1]
        v = lambda_inv_sq * p  # [B, D] contravariant velocity
        geo_corr = christoffel_contraction(z, v)
        v_corr = v - (h / 4.0) * geo_corr
        if self.V_alg > 0:
            v_corr = self.V_alg * v_corr / (self.V_alg + v_corr.norm(dim=-1, keepdim=True))
        z = poincare_exp_map(z, (h / 2.0) * v_corr)
        z = project_to_ball(z)

        # --- B step (second half): momentum kick ---
        force2, phi_eff = self.potential_net.force_and_potential(z, rw)
        u_pi2 = self.control_lift(action_canonical)
        kick2 = force2 - u_pi2

        if self.F_max > 0:
            kick2 = self.F_max * kick2 / (self.F_max + kick2.norm(dim=-1, keepdim=True))

        p = p - (h / 2.0) * kick2

        return z, p, phi_eff, hodge_info

    # -----------------------------------------------------------------
    # WFR conditional jump (Fisher-Rao component)
    # -----------------------------------------------------------------

    @torch.no_grad()
    def _boltzmann_chart_logits(
        self,
        z: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        """Value-driven Boltzmann logits for chart selection.

        logit_k = β · (V(c_k) - V(z) - d_geo(z, c_k))

        Higher logits for charts with better value AND reachable by short jumps.

        Runs under torch.no_grad() because Boltzmann logits are used only for
        jump *decisions* (argmax), not for supervised loss.  The supervised
        chart signal comes from ``chart_logits`` via CovariantChartTarget.
        This avoids retaining the B*K computation graph through
        CovariantPotentialNet's attention modules (the OOM root cause in
        Phase 3 with retain_graph=True).

        Args:
            z: [B, D] current position.
            rw: [B, K] current chart weights.

        Returns:
            logits: [B, K] chart selection logits (detached).
        """
        centers = self.chart_predictor.chart_tok.chart_centers  # [K, D]
        K = centers.shape[0]
        B = z.shape[0]

        # Potential at current position — only K forward passes worth of state
        V_z = self.potential_net.task_value(z, rw)  # [B, 1]

        # Potential at each chart center — batch size K (not B*K).
        # V(c_k) depends only on the center position and one-hot routing,
        # so it's identical across all B samples.  Evaluate once, broadcast.
        rw_onehot = torch.eye(K, device=z.device)  # [K, K]
        V_ck = self.potential_net.task_value(centers, rw_onehot)  # [K, 1]
        V_ck = V_ck.squeeze(-1).unsqueeze(0).expand(B, -1)  # [B, K]

        # Geodesic distance from z to each center
        c_exp = centers.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]
        d_geo = hyperbolic_distance(
            z.unsqueeze(1).expand(-1, K, -1),
            c_exp,
        )  # [B, K]

        # Boltzmann logits: value gain minus transport cost
        return self.jump_beta * (V_ck - V_z - d_geo)  # [B, K]

    def _conditional_jump(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action_canonical: torch.Tensor,
        rw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conditional chart jump (Fisher-Rao component of WFR).

        If predicted target chart differs from current chart:
          - Teleport to target chart center
          - Reset momentum via momentum_init at new position
        If same chart: position and momentum continue unchanged.

        Args:
            z: [B, D] current position.
            p: [B, D] current momentum.
            action_canonical: [B, C] canonical motor-side action latent.
            rw: [B, K] current chart weights.

        Returns:
            z: [B, D] (possibly jumped) position.
            p: [B, D] (possibly reset) momentum.
            chart_logits: [B, K] chart logits (for supervised CE loss).
            rw: [B, K] updated chart weights.
            jumped: [B] boolean mask of which samples jumped.
        """
        # Supervised chart logits (from learned CovariantChartTarget)
        chart_logits = self.chart_predictor(z, action_canonical, rw)

        # Boltzmann logits for actual jump decision
        boltz_logits = self._boltzmann_chart_logits(z, rw)

        current_chart = rw.argmax(dim=-1)  # [B]
        target_chart = boltz_logits.argmax(dim=-1)  # [B]
        jumped = current_chart != target_chart  # [B]

        if jumped.any():
            centers = self.chart_predictor.chart_tok.chart_centers  # [K, D]
            target_centers = centers[target_chart]  # [B, D]

            # Teleport to chart center
            z_jumped = project_to_ball(target_centers)
            z = torch.where(jumped.unsqueeze(-1), z_jumped, z)

            # Reset momentum at new position
            p_new = self.momentum_init(z_jumped)
            p = torch.where(jumped.unsqueeze(-1), p_new, p)

        # Update router weights from supervised chart logits (gradient-carrying).
        # Boltzmann logits are detached (used only for jump decisions above).
        rw = F.softmax(chart_logits, dim=-1)

        return z, p, chart_logits, rw, jumped

    # -----------------------------------------------------------------
    # Supervised integration (geodesic diffusion training)
    # -----------------------------------------------------------------

    def supervised_integration(
        self,
        z_start: torch.Tensor,
        p_start: torch.Tensor,
        action_canonical: torch.Tensor,
        rw: torch.Tensor,
        n_steps: int,
        deterministic: bool = True,
    ) -> dict[str, torch.Tensor]:
        """N-step BAOAB from z_start, returning intermediate z/p trajectory.

        When ``deterministic=True``, the O-step noise is zeroed so the
        integrator output is a deterministic function of (z_start, p_start,
        action_canonical) — required for supervised training against geodesic targets.

        Args:
            z_start: [B, D] starting position.
            p_start: [B, D] starting momentum.
            action_canonical: [B, C] canonical action latent (same input for all sub-steps).
            rw: [B, K] chart routing weights.
            n_steps: Number of BAOAB sub-steps.
            deterministic: If True, zero O-step noise (c2=0).

        Returns:
            Dict with:
                z_traj: [B, N+1, D]  (z_start, z_1, ..., z_N)
                p_traj: [B, N+1, D]  (p_start, p_1, ..., p_N)
                phi_eff: [B, N, 1]   potential at each step
                hodge_info: dict      from last step
        """
        # Save and optionally zero noise coefficient
        orig_c2 = self.c2
        if deterministic:
            self.c2 = 0.0

        z_traj = [z_start]
        p_traj = [p_start]
        phi_eff_list = []
        last_hodge_info: dict[str, torch.Tensor] = {}

        z = z_start
        p = p_start

        for _ in range(n_steps):
            z, p, phi_eff, hodge_info = self._baoab_step(z, p, action_canonical, rw)
            z_traj.append(z)
            p_traj.append(p)
            phi_eff_list.append(phi_eff)
            if hodge_info:
                last_hodge_info = hodge_info

        # Restore noise coefficient
        self.c2 = orig_c2

        return {
            "z_traj": torch.stack(z_traj, dim=1),  # [B, N+1, D]
            "p_traj": torch.stack(p_traj, dim=1),  # [B, N+1, D]
            "phi_eff": torch.stack(phi_eff_list, dim=1),  # [B, N, 1]
            "hodge_info": last_hodge_info,
        }

    def _rollout_transition(
        self,
        z: torch.Tensor,
        p: torch.Tensor,
        action_canonical: torch.Tensor,
        rw: torch.Tensor,
        *,
        track_energy: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor] | None]:
        """Advance one horizon step under the canonical action latent."""
        B = z.shape[0]
        N = self.n_refine_steps

        if self.use_jump:
            z, p, chart_logits, rw, jumped = self._conditional_jump(z, p, action_canonical, rw)
        else:
            chart_logits = self.chart_predictor(z, action_canonical, rw)
            jumped = z.new_zeros(B, dtype=torch.bool)

        last_hodge_info: dict[str, torch.Tensor] | None = None
        energy_substeps = None
        if track_energy:
            energy_substeps = z.new_zeros((B, N))

        phi_eff = self.potential_net.effective_potential(z, rw)
        for s in range(N):
            z, p, phi_eff, hodge_info = self._baoab_step(z, p, action_canonical, rw)

            if energy_substeps is not None:
                r_sq = (z**2).sum(dim=-1, keepdim=True)
                g_inv = ((1.0 - r_sq).clamp(min=1e-6) / 2.0) ** 2
                p_sq = (p**2).sum(dim=-1, keepdim=True)
                H_s = phi_eff + 0.5 * g_inv * p_sq
                energy_substeps[:, s] = H_s.squeeze(-1)

            if hodge_info:
                last_hodge_info = hodge_info

        return {
            "z": z,
            "p": p,
            "rw": rw,
            "chart_logits": chart_logits,
            "jumped": jumped,
            "phi_eff": phi_eff,
            "hodge_info": last_hodge_info,
            "energy_substeps": energy_substeps,
        }

    # -----------------------------------------------------------------
    # Forward (multi-step rollout)
    # -----------------------------------------------------------------

    def forward(
        self,
        z_0: torch.Tensor,
        action_canonicals: torch.Tensor,
        router_weights_0: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Multi-step rollout through the world model.

        Each horizon step consists of:
        1. Conditional chart jump (Fisher-Rao): teleport if chart changes
        2. N BAOAB refinement sub-steps (W2 transport): Hamiltonian flow

        Args:
            z_0: [B, D] initial latent position.
            action_canonicals: [B, H, C] canonical action-latent sequence.
            router_weights_0: [B, K] initial chart routing weights.

        Returns:
            Dictionary with:
                z_trajectory: [B, H, D] predicted latent positions.
                chart_logits: [B, H, K] predicted chart logits per step.
                momenta: [B, H, D] momenta at each step.
                jumped: [B, H] boolean jump events.
                phi_eff: [B, H, 1] effective potential (diagnostic).
                energy_var: scalar, Hamiltonian variance across sub-steps.
                hodge_conservative_ratio: [B, H] ||f_cons|| / ||f_total||.
                hodge_solenoidal_ratio: [B, H] ||f_sol|| / ||f_total||.
                hodge_harmonic_ratio: [B, H] ||f_harm|| / ||f_total||.
                hodge_harmonic_forces: [B, H, D] harmonic residual forces.
        """
        B, H, _A = action_canonicals.shape
        device = z_0.device
        D = self.latent_dim
        K = self.num_charts
        N = self.n_refine_steps

        # Initialize momentum from position
        p = self.momentum_init(z_0)  # [B, D]

        z = z_0
        rw = router_weights_0

        # Pre-allocate output tensors
        z_traj = torch.zeros(B, H, D, device=device)
        chart_logits_out = torch.zeros(B, H, K, device=device)
        momenta = torch.zeros(B, H, D, device=device)
        jumped_out = torch.zeros(B, H, dtype=torch.bool, device=device)
        phi_eff_out = torch.zeros(B, H, 1, device=device)
        hodge_conservative = torch.zeros(B, H, device=device)
        hodge_solenoidal = torch.zeros(B, H, device=device)
        hodge_harmonic = torch.zeros(B, H, device=device)
        hodge_harmonic_forces = torch.zeros(B, H, D, device=device)
        energy_substeps = torch.zeros(B, H, N, device=device)

        for t in range(H):
            step_out = self._rollout_transition(
                z,
                p,
                action_canonicals[:, t, :],
                rw,
                track_energy=True,
            )
            z = step_out["z"]
            p = step_out["p"]
            rw = step_out["rw"]
            cl = step_out["chart_logits"]
            jumped = step_out["jumped"]
            phi_eff = step_out["phi_eff"]
            last_hodge_info = step_out["hodge_info"]
            if step_out["energy_substeps"] is not None:
                energy_substeps[:, t, :] = step_out["energy_substeps"]

            # Store outputs (from final sub-step)
            z_traj[:, t, :] = z
            chart_logits_out[:, t, :] = cl
            momenta[:, t, :] = p
            jumped_out[:, t] = jumped
            phi_eff_out[:, t, :] = phi_eff

            if last_hodge_info:
                hodge_conservative[:, t] = last_hodge_info["conservative_ratio"]
                hodge_solenoidal[:, t] = last_hodge_info["solenoidal_ratio"]
                hodge_harmonic[:, t] = last_hodge_info["harmonic_ratio"]
                hodge_harmonic_forces[:, t] = last_hodge_info["harmonic"]

        # Energy variance across sub-steps (for conservation loss)
        energy_var = energy_substeps.var(dim=-1).mean()

        return {
            "z_trajectory": z_traj,
            "chart_logits": chart_logits_out,
            "momenta": momenta,
            "jumped": jumped_out,
            "phi_eff": phi_eff_out,
            "potential_net": self.potential_net,
            "router_weights_final": rw,
            "energy_var": energy_var,
            "hodge_conservative_ratio": hodge_conservative,
            "hodge_solenoidal_ratio": hodge_solenoidal,
            "hodge_harmonic_ratio": hodge_harmonic,
            "hodge_harmonic_forces": hodge_harmonic_forces,
        }
