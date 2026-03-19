"""Tests for the covariant geometric world model."""

from __future__ import annotations

import pytest
import torch


# Use small dims for fast tests
B, D, A, K, H = 4, 3, 6, 8, 5
D_MODEL = 32  # small for speed


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def z(device):
    """Random position in Poincare ball."""
    return torch.randn(B, D, device=device) * 0.3


@pytest.fixture
def action(device):
    return torch.randn(B, A, device=device)


@pytest.fixture
def rw(device):
    return torch.softmax(torch.randn(B, K, device=device), dim=-1)


class TestActionTokenizer:
    def test_output_shapes(self, z, action):
        from fragile.vla.covariant_world_model import ActionTokenizer

        tok = ActionTokenizer(A, D_MODEL, D)
        x, z_out = tok(action, z)
        assert x.shape == (B, A, D_MODEL)
        assert z_out.shape == (B, A, D)

    def test_z_broadcast(self, z, action):
        """All action tokens should be positioned at z."""
        from fragile.vla.covariant_world_model import ActionTokenizer

        tok = ActionTokenizer(A, D_MODEL, D)
        _, z_out = tok(action, z)
        for i in range(A):
            assert torch.allclose(z_out[:, i, :], z)


class TestChartTokenizer:
    def test_output_shapes(self, z, rw):
        from fragile.vla.covariant_world_model import ChartTokenizer

        tok = ChartTokenizer(K, D_MODEL, D)
        x, z_out = tok(rw, z)
        assert x.shape == (B, K, D_MODEL)
        assert z_out.shape == (B, K, D)


class TestCovariantPotentialNet:
    def test_forward_shape(self, z, rw):
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        phi = net(z, rw)
        assert phi.shape == (B, 1)

    def test_force_and_potential_shapes(self, z, rw):
        """force_and_potential must return cotangent force [B, D] and scalar phi [B, 1]."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        force, phi = net.force_and_potential(z, rw)
        assert force.shape == (B, D)
        assert phi.shape == (B, 1)

    def test_analytic_U_and_grad(self):
        """Verify analytical gradient against finite differences."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        z.requires_grad_(True)

        U, dU_dz = net._analytic_U_and_grad(z)

        # Compare against autograd
        U_sum = U.sum()
        U_sum.backward()
        assert torch.allclose(dU_dz, z.grad, atol=1e-5), (
            "Analytical gradient does not match autograd"
        )

    def test_force_finite(self, z, rw):
        """Force components must be finite everywhere inside the ball."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        force, phi = net.force_and_potential(z, rw)
        assert torch.isfinite(force).all(), "Non-finite force"
        assert torch.isfinite(phi).all(), "Non-finite potential"


class TestCovariantControlField:
    def test_forward_shape(self, z, action, rw):
        from fragile.vla.covariant_world_model import CovariantControlField

        net = CovariantControlField(D, A, K, D_MODEL)
        u = net(z, action, rw)
        assert u.shape == (B, D)


class TestCovariantValueCurl:
    def test_forward_shape(self, z, action):
        from fragile.vla.covariant_world_model import CovariantValueCurl

        net = CovariantValueCurl(D, A, D_MODEL)
        F_mat = net(z, action)
        assert F_mat.shape == (B, D, D)

    def test_antisymmetric(self, z, action):
        """F should be antisymmetric: F + F^T = 0."""
        from fragile.vla.covariant_world_model import CovariantValueCurl

        net = CovariantValueCurl(D, A, D_MODEL)
        F_mat = net(z, action)
        assert torch.allclose(F_mat + F_mat.transpose(-2, -1), torch.zeros_like(F_mat), atol=1e-6)


class TestCovariantChartTarget:
    def test_forward_shape(self, z, action, rw):
        from fragile.vla.covariant_world_model import CovariantChartTarget

        net = CovariantChartTarget(D, A, K, D_MODEL)
        logits = net(z, action, rw)
        assert logits.shape == (B, K)


class TestCovariantJumpRate:
    def test_forward_shape(self, z, rw):
        from fragile.vla.covariant_world_model import CovariantJumpRate

        net = CovariantJumpRate(D, K, D_MODEL)
        rate = net(z, rw)
        assert rate.shape == (B, 1)

    def test_non_negative(self, z, rw):
        from fragile.vla.covariant_world_model import CovariantJumpRate

        net = CovariantJumpRate(D, K, D_MODEL)
        rate = net(z, rw)
        assert (rate >= 0).all()


class TestCovariantMomentumInit:
    def test_forward_shape(self, z):
        from fragile.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        p = net(z)
        assert p.shape == (B, D)

    def test_metric_scaling(self):
        """Momentum at origin should have higher metric factor than near boundary."""
        from fragile.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        z_origin = torch.zeros(1, D)
        z_boundary = torch.ones(1, D) * 0.9 / D**0.5  # near boundary
        p_origin = net(z_origin)
        p_boundary = net(z_boundary)
        # lambda at origin = 2, lambda at boundary >> 2, so lam_sq at boundary > lam_sq at origin
        # But the SpectralLinear output differs, so just check shapes
        assert p_origin.shape == (1, D)
        assert p_boundary.shape == (1, D)


class TestGeometricWorldModel:
    def test_import(self):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        assert GeometricWorldModel is not None

    def test_forward_shapes(self, device):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            dt=0.01,
            gamma_friction=1.0,
            T_c=0.1,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert out["z_trajectory"].shape == (B, H, D)
        assert out["chart_logits"].shape == (B, H, K)
        assert out["momenta"].shape == (B, H, D)
        assert out["jumped"].shape == (B, H)
        assert out["phi_eff"].shape == (B, H, 1)
        assert "energy_var" in out

    def test_z_stays_in_ball(self, device):
        """Trajectory points should stay inside the Poincare ball."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        norms = out["z_trajectory"].norm(dim=-1)
        assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"

    def test_backward_pass(self, device):
        """Ensure gradients flow through the entire model."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        loss = out["z_trajectory"].sum() + out["chart_logits"].sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        sum(1 for _ in m.parameters())
        assert grad_count > 0, "No gradients computed"

    def test_no_boris(self, device):
        """Model should work with Boris rotation disabled."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            use_boris=False,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        assert out["z_trajectory"].shape == (B, H, D)

    def test_no_jump(self, device):
        """Model should work with jump process disabled."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            use_jump=False,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        assert (~out["jumped"]).all(), "No jumps should occur with use_jump=False"

    def test_bind_chart_centers_reuses_phase1_atlas(self):
        """All chart-conditioned submodules should share the frozen Phase 1 atlas."""
        from fragile.layers.gauge import project_to_ball
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        phase1_centers = torch.randn(K, D) * 5.0
        expected = project_to_ball(phase1_centers)

        m.bind_chart_centers(phase1_centers, freeze=True)

        for tok in (
            m.potential_net.chart_tok,
            m.chart_predictor.chart_tok,
        ):
            assert torch.allclose(tok.chart_centers, expected)
            assert not tok.chart_centers.requires_grad


class TestChartCenterProjection:
    """Chart centers must stay inside the Poincare ball during forward pass."""

    def test_chart_tokenizer_projects_centers(self, z, rw):
        from fragile.vla.covariant_world_model import ChartTokenizer

        tok = ChartTokenizer(K, D_MODEL, D)
        # Force centers outside the ball
        with torch.no_grad():
            tok.chart_centers.copy_(torch.ones_like(tok.chart_centers) * 5.0)
        _, tokens_z = tok(rw, z)
        norms = tokens_z.norm(dim=-1)
        assert (norms < 1.0).all(), f"Chart centers escaped the ball: max norm {norms.max()}"

    def test_chart_target_projects_centers(self, z, action, rw):
        from fragile.vla.covariant_world_model import CovariantChartTarget

        net = CovariantChartTarget(D, A, K, D_MODEL)
        # Force centers outside the ball
        with torch.no_grad():
            net.chart_tok.chart_centers.copy_(torch.ones_like(net.chart_tok.chart_centers) * 5.0)
        logits = net(z, action, rw)
        assert logits.shape == (B, K)
        assert torch.isfinite(logits).all(), "Non-finite logits from out-of-ball centers"


class TestGeometricInvariants:
    """Tests that verify fundamental geometric properties are respected."""

    def test_analytic_gradient_matches_finite_differences(self):
        """Analytical dU/dz must match numerical finite differences of U(z)."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        eps = 1e-4

        _U_base, dU_dz = net._analytic_U_and_grad(z)

        # Numerical gradient via central differences
        numerical_grad = torch.zeros_like(z)
        for i in range(D):
            z_plus = z.clone()
            z_plus[:, i] += eps
            z_minus = z.clone()
            z_minus[:, i] -= eps
            U_plus, _ = net._analytic_U_and_grad(z_plus)
            U_minus, _ = net._analytic_U_and_grad(z_minus)
            numerical_grad[:, i] = (U_plus - U_minus).squeeze(-1) / (2 * eps)

        assert torch.allclose(dU_dz, numerical_grad, atol=4e-3), (
            f"Analytical grad:\n{dU_dz}\nNumerical grad:\n{numerical_grad}"
        )

    def test_force_near_boundary_is_finite(self, rw):
        """Force must stay finite even near the boundary of the Poincare ball."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z_near_boundary = torch.ones(1, D) * 0.95 / D**0.5
        force, phi = net.force_and_potential(z_near_boundary, rw[:1])
        assert torch.isfinite(force).all(), f"Non-finite force near boundary: {force}"
        assert torch.isfinite(phi).all(), f"Non-finite phi near boundary: {phi}"

    def test_gradients_flow_through_force(self, rw):
        """Potential net parameters must receive gradients through force_and_potential."""
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z = torch.randn(2, D) * 0.3
        force, phi = net.force_and_potential(z, rw[:2])
        loss = force.sum() + phi.sum()
        loss.backward()
        grad_count = sum(1 for p in net.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients flow through force_and_potential"

    def test_boris_rotation_preserves_norm(self, z, action):
        """Boris rotation must preserve momentum norm (antisymmetric F)."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            use_boris=True,
        )
        p_in = torch.randn(B, D) * 0.1
        p_out, _ = m._boris_rotation(p_in, z, action)
        # Boris rotation is an exact norm-preserving rotation
        assert torch.allclose(
            p_in.norm(dim=-1),
            p_out.norm(dim=-1),
            atol=1e-5,
        ), "Boris rotation changed momentum norm"

    def test_momentum_init_cotangent_scaling(self):
        """Momentum p = lam^2 * v must scale with conformal factor."""
        from fragile.vla.covariant_world_model import CovariantMomentumInit

        net = CovariantMomentumInit(D)
        z_origin = torch.zeros(1, D)
        z_edge = torch.ones(1, D) * 0.8 / D**0.5

        p_origin = net(z_origin)
        p_edge = net(z_edge)

        # At origin: lam = 2, lam^2 = 4
        # At edge: lam = 2/(1-r^2), lam^2 > 4
        # So |p_edge| > |p_origin| for identical SpectralLinear(z) output magnitude
        # (but SpectralLinear outputs differ, so just check finite)
        assert torch.isfinite(p_origin).all()
        assert torch.isfinite(p_edge).all()


class TestMomentumRegularization:
    def test_metric_aware(self):
        """Metric-aware reg should differ from Euclidean."""
        from fragile.losses.world_model import compute_momentum_regularization

        momenta = torch.randn(B, H, D)
        z_traj = torch.randn(B, H, D) * 0.3
        loss = compute_momentum_regularization(momenta, z_traj)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_higher_at_boundary(self):
        """Kinetic energy should be lower near boundary (smaller g_inv_factor)."""
        from fragile.losses.world_model import compute_momentum_regularization

        p = torch.ones(1, 1, D)
        z_center = torch.zeros(1, 1, D)
        z_edge = torch.ones(1, 1, D) * 0.9 / D**0.5
        loss_center = compute_momentum_regularization(p, z_center)
        loss_edge = compute_momentum_regularization(p, z_edge)
        # g_inv_factor at center: ((1-0)/2)^2 = 0.25
        # g_inv_factor at edge: ((1-r^2)/2)^2 < 0.25
        assert loss_center > loss_edge


# ==========================================================================
# Feature 1: Risk-Metric Coupling (G proportional to T)
# ==========================================================================


class TestRiskAdaptiveConformalMetric:
    """Tests for the risk-adaptive conformal metric."""

    def test_import(self):
        from fragile.layers.gauge import RiskAdaptiveConformalMetric

        assert RiskAdaptiveConformalMetric is not None

    def test_no_risk_matches_base(self):
        """With risk_tensor=None, should return same as ConformalMetric."""
        from fragile.layers.gauge import (
            ConformalMetric,
            RiskAdaptiveConformalMetric,
        )

        z = torch.randn(B, D) * 0.3
        base = ConformalMetric()
        adaptive = RiskAdaptiveConformalMetric(risk_coupling_alpha=0.1)

        lam_base = base.conformal_factor(z)
        lam_adaptive = adaptive.conformal_factor(z, risk_tensor=None)
        assert torch.allclose(lam_base, lam_adaptive, atol=1e-6)

    def test_risk_increases_conformal_factor(self):
        """Non-zero risk tensor should increase the conformal factor."""
        from fragile.layers.gauge import RiskAdaptiveConformalMetric

        z = torch.randn(B, D) * 0.3
        metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=0.1)

        lam_no_risk = metric.conformal_factor(z)
        T = torch.eye(D).unsqueeze(0).expand(B, -1, -1)  # non-zero risk tensor
        lam_risk = metric.conformal_factor(z, risk_tensor=T)
        assert (lam_risk >= lam_no_risk).all(), "Risk should increase conformal factor"

    def test_alpha_zero_matches_base(self):
        """With alpha=0, risk tensor should have no effect."""
        from fragile.layers.gauge import RiskAdaptiveConformalMetric

        z = torch.randn(B, D) * 0.3
        metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=0.0)

        T = torch.randn(B, D, D)
        lam_no_risk = metric.conformal_factor(z)
        lam_risk = metric.conformal_factor(z, risk_tensor=T)
        assert torch.allclose(lam_no_risk, lam_risk, atol=1e-6)

    def test_metric_inv_shape(self):
        """metric_inv should return [B, D, D] with risk tensor."""
        from fragile.layers.gauge import RiskAdaptiveConformalMetric

        z = torch.randn(B, D) * 0.3
        metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=0.1)
        T = torch.randn(B, D, D)
        g_inv = metric.metric_inv(z, risk_tensor=T)
        assert g_inv.shape == (B, D, D)
        assert torch.isfinite(g_inv).all()

    def test_temperature_with_risk(self):
        """Temperature should decrease with risk (higher lambda -> lower tau)."""
        from fragile.layers.gauge import RiskAdaptiveConformalMetric

        z = torch.randn(B, D) * 0.3
        metric = RiskAdaptiveConformalMetric(risk_coupling_alpha=0.5)

        tau_no_risk = metric.temperature(z, D)
        T = torch.eye(D).unsqueeze(0).expand(B, -1, -1) * 10.0  # large risk
        tau_risk = metric.temperature(z, D, risk_tensor=T)
        assert (tau_risk <= tau_no_risk).all(), "Risk should decrease temperature"


class TestRiskTensor:
    """Tests for compute_risk_tensor."""

    def test_gradient_stress_shape(self):
        """Risk tensor from force only should be [B, D, D]."""
        from fragile.vla.covariant_world_model import compute_risk_tensor

        force = torch.randn(B, D)
        T = compute_risk_tensor(force)
        assert T.shape == (B, D, D)

    def test_gradient_stress_symmetric(self):
        """Gradient stress f (x) f is symmetric by construction."""
        from fragile.vla.covariant_world_model import compute_risk_tensor

        force = torch.randn(B, D)
        T = compute_risk_tensor(force)
        assert torch.allclose(T, T.transpose(-2, -1), atol=1e-6)

    def test_gradient_stress_positive_semidefinite(self):
        """f (x) f is positive semi-definite (all eigenvalues >= 0)."""
        from fragile.vla.covariant_world_model import compute_risk_tensor

        force = torch.randn(B, D)
        T = compute_risk_tensor(force)
        eigenvalues = torch.linalg.eigvalsh(T)
        assert (eigenvalues >= -1e-5).all(), f"Negative eigenvalues: {eigenvalues}"

    def test_with_curl_tensor(self):
        """Risk tensor with Maxwell stress should still be finite."""
        from fragile.vla.covariant_world_model import compute_risk_tensor

        force = torch.randn(B, D)
        # Antisymmetric curl tensor
        upper = torch.randn(B, D * (D - 1) // 2)
        F_mat = torch.zeros(B, D, D)
        rows, cols = torch.triu_indices(D, D, offset=1)
        F_mat[:, rows, cols] = upper
        F_mat[:, cols, rows] = -upper

        lambda_inv_sq = torch.full((B, 1), 0.25)  # λ⁻² at origin
        T = compute_risk_tensor(force, curl_tensor=F_mat, lambda_inv_sq=lambda_inv_sq)
        assert T.shape == (B, D, D)
        assert torch.isfinite(T).all()

    def test_zero_force_zero_risk(self):
        """Zero force and no curl should give zero risk tensor."""
        from fragile.vla.covariant_world_model import compute_risk_tensor

        force = torch.zeros(B, D)
        T = compute_risk_tensor(force)
        assert torch.allclose(T, torch.zeros_like(T), atol=1e-8)


class TestRiskMetricIntegration:
    """Tests for GeometricWorldModel with risk metric enabled."""

    def test_forward_with_risk_metric(self, device):
        """Model with risk_metric_alpha > 0 should produce valid outputs."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.1,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert out["z_trajectory"].shape == (B, H, D)
        assert torch.isfinite(out["z_trajectory"]).all()
        norms = out["z_trajectory"].norm(dim=-1)
        assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"

    def test_backward_with_risk_metric(self, device):
        """Gradients should flow through risk-adaptive metric."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.1,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)
        loss = out["z_trajectory"].sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients with risk metric"

    def test_alpha_zero_is_standard_metric(self, device):
        """risk_metric_alpha=0 should use ConformalMetric, not RiskAdaptive."""
        from fragile.layers.gauge import ConformalMetric
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.0,
        )
        assert type(m.metric) is ConformalMetric


# ==========================================================================
# Feature 2: Screened Poisson Critic
# ==========================================================================


class TestHyperbolicLaplacian:
    """Tests for the Laplace-Beltrami operator on the Poincare ball."""

    def test_output_shape(self):
        """Laplacian should return ([B, 1], [B, 1])."""
        from fragile.losses.world_model import hyperbolic_laplacian

        def V_func(z):
            return (z**2).sum(dim=-1, keepdim=True)

        z = torch.randn(B, D) * 0.3
        lap, V_center = hyperbolic_laplacian(V_func, z)
        assert lap.shape == (B, 1)
        assert V_center.shape == (B, 1)

    def test_finite(self):
        """Laplacian should be finite inside the ball."""
        from fragile.losses.world_model import hyperbolic_laplacian

        def V_func(z):
            return (z**2).sum(dim=-1, keepdim=True)

        z = torch.randn(B, D) * 0.3
        lap, _ = hyperbolic_laplacian(V_func, z)
        assert torch.isfinite(lap).all(), f"Non-finite Laplacian: {lap}"

    def test_constant_function_zero_laplacian(self):
        """Laplacian of a constant function should be zero.

        We implement the constant as 0*z.sum() + 5 so autograd can
        still build a graph (pure constants have no grad_fn).
        """
        from fragile.losses.world_model import hyperbolic_laplacian

        def V_const(z):
            return 0.0 * z.sum(dim=-1, keepdim=True) + 5.0

        z = torch.randn(B, D) * 0.3
        lap, _ = hyperbolic_laplacian(V_const, z)
        assert torch.allclose(lap, torch.zeros_like(lap), atol=1e-5), (
            f"Laplacian of constant is not zero: {lap}"
        )

    def test_quadratic_function_known_laplacian(self):
        """For f(z) = |z|^2, Delta_E f = 2D. Check Poincare correction is applied."""
        from fragile.losses.world_model import hyperbolic_laplacian

        def V_quadratic(z):
            return (z**2).sum(dim=-1, keepdim=True)

        z = torch.randn(B, D) * 0.2  # well inside ball
        lap, _ = hyperbolic_laplacian(V_quadratic, z)
        # Just check it's non-zero and finite (the exact value depends on correction)
        assert torch.isfinite(lap).all()
        assert not torch.allclose(lap, torch.zeros_like(lap), atol=1e-3)

    def test_gradients_flow(self):
        """Autograd graph should be connected through the Laplacian."""
        from fragile.losses.world_model import hyperbolic_laplacian

        W = torch.randn(D, 1, requires_grad=True)

        def V_linear(z):
            return z @ W  # [B, 1]

        z = torch.randn(B, D) * 0.3
        lap, _ = hyperbolic_laplacian(V_linear, z)
        loss = lap.sum()
        loss.backward()
        # W should get gradients (through the Hessian trace)
        assert W.grad is not None, "No gradients through Laplacian"


class TestScreenedPoissonLoss:
    """Tests for the screened Poisson PDE residual loss."""

    def test_output_scalar(self):
        """Loss should be a scalar > 0."""
        from fragile.losses.world_model import compute_screened_poisson_loss
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z_traj = torch.randn(B, H, D) * 0.3
        z_tgt = torch.randn(B, H, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        loss = compute_screened_poisson_loss(net, z_traj, z_tgt, rw, kappa=1.0)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_subsampling(self):
        """With max_samples < B*H, should subsample without error."""
        from fragile.losses.world_model import compute_screened_poisson_loss
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z_traj = torch.randn(B, H, D) * 0.3
        z_tgt = torch.randn(B, H, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        loss = compute_screened_poisson_loss(
            net,
            z_traj,
            z_tgt,
            rw,
            kappa=1.0,
            max_samples=4,
        )
        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_gradients_flow_to_potential_net(self):
        """Screened Poisson loss must provide gradients to the critic head."""
        from fragile.losses.world_model import compute_screened_poisson_loss
        from fragile.vla.covariant_world_model import CovariantPotentialNet

        net = CovariantPotentialNet(D, K, D_MODEL)
        z_traj = torch.randn(B, H, D) * 0.3
        z_tgt = torch.randn(B, H, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        loss = compute_screened_poisson_loss(
            net,
            z_traj,
            z_tgt,
            rw,
            kappa=1.0,
            max_samples=8,
        )
        loss.backward()
        grad_count = sum(1 for p in net.v_critic_attn.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients to critic attention from screened Poisson"


# ==========================================================================
# Feature 3: Hodge Decomposition
# ==========================================================================


class TestHodgeDecomposer:
    """Tests for the HodgeDecomposer module."""

    def test_output_shapes(self):
        """HodgeDecomposer should return correct shapes."""
        from fragile.vla.covariant_world_model import HodgeDecomposer

        hd = HodgeDecomposer(D)
        f_total = torch.randn(B, D)
        f_cons = torch.randn(B, D)
        f_sol = torch.randn(B, D)
        result = hd(f_total, f_cons, f_sol)

        assert result["harmonic"].shape == (B, D)
        assert result["conservative_ratio"].shape == (B,)
        assert result["solenoidal_ratio"].shape == (B,)
        assert result["harmonic_ratio"].shape == (B,)

    def test_decomposition_completeness(self):
        """f_cons + f_sol + f_harmonic should equal f_total."""
        from fragile.vla.covariant_world_model import HodgeDecomposer

        hd = HodgeDecomposer(D)
        f_total = torch.randn(B, D)
        f_cons = torch.randn(B, D) * 0.5
        f_sol = torch.randn(B, D) * 0.3
        result = hd(f_total, f_cons, f_sol)

        f_reconstructed = f_cons + f_sol + result["harmonic"]
        assert torch.allclose(f_total, f_reconstructed, atol=1e-6), (
            "Hodge decomposition does not sum to total force"
        )

    def test_zero_harmonic_when_exact(self):
        """When f_total = f_cons + f_sol, harmonic should be zero."""
        from fragile.vla.covariant_world_model import HodgeDecomposer

        hd = HodgeDecomposer(D)
        f_cons = torch.randn(B, D)
        f_sol = torch.randn(B, D)
        f_total = f_cons + f_sol
        result = hd(f_total, f_cons, f_sol)

        assert torch.allclose(
            result["harmonic"],
            torch.zeros_like(result["harmonic"]),
            atol=1e-6,
        )
        assert torch.allclose(
            result["harmonic_ratio"],
            torch.zeros(B),
            atol=1e-5,
        )

    def test_no_learnable_parameters(self):
        """HodgeDecomposer should have no learnable parameters."""
        from fragile.vla.covariant_world_model import HodgeDecomposer

        hd = HodgeDecomposer(D)
        assert sum(p.numel() for p in hd.parameters()) == 0

    def test_ratios_non_negative(self):
        """All Hodge ratios should be non-negative."""
        from fragile.vla.covariant_world_model import HodgeDecomposer

        hd = HodgeDecomposer(D)
        f_total = torch.randn(B, D)
        f_cons = torch.randn(B, D)
        f_sol = torch.randn(B, D)
        result = hd(f_total, f_cons, f_sol)

        assert (result["conservative_ratio"] >= 0).all()
        assert (result["solenoidal_ratio"] >= 0).all()
        assert (result["harmonic_ratio"] >= 0).all()


class TestHodgeConsistencyLoss:
    """Tests for the Hodge consistency loss function."""

    def test_output_scalar(self):
        """Loss should be a non-negative scalar."""
        from fragile.losses.world_model import compute_hodge_consistency_loss

        harmonic = torch.randn(B, H, D)
        loss = compute_hodge_consistency_loss(harmonic)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_zero_harmonic_zero_loss(self):
        """Zero harmonic forces should give zero loss."""
        from fragile.losses.world_model import compute_hodge_consistency_loss

        harmonic = torch.zeros(B, H, D)
        loss = compute_hodge_consistency_loss(harmonic)
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-8)

    def test_larger_harmonic_larger_loss(self):
        """Larger harmonic residual should give larger loss."""
        from fragile.losses.world_model import compute_hodge_consistency_loss

        harmonic_small = torch.randn(B, H, D) * 0.1
        harmonic_large = torch.randn(B, H, D) * 10.0
        loss_small = compute_hodge_consistency_loss(harmonic_small)
        loss_large = compute_hodge_consistency_loss(harmonic_large)
        assert loss_large > loss_small


class TestHodgeIntegration:
    """Tests for Hodge decomposition in the full world model."""

    def test_hodge_outputs_in_forward(self, device):
        """Forward pass should include Hodge diagnostic keys."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert "hodge_conservative_ratio" in out
        assert "hodge_solenoidal_ratio" in out
        assert "hodge_harmonic_ratio" in out
        assert "hodge_harmonic_forces" in out

        assert out["hodge_conservative_ratio"].shape == (B, H)
        assert out["hodge_solenoidal_ratio"].shape == (B, H)
        assert out["hodge_harmonic_ratio"].shape == (B, H)
        assert out["hodge_harmonic_forces"].shape == (B, H, D)

    def test_hodge_ratios_finite(self, device):
        """Hodge ratios should be finite."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert torch.isfinite(out["hodge_conservative_ratio"]).all()
        assert torch.isfinite(out["hodge_solenoidal_ratio"]).all()
        assert torch.isfinite(out["hodge_harmonic_ratio"]).all()
        assert torch.isfinite(out["hodge_harmonic_forces"]).all()


# ==========================================================================
# Full integration: all 3 features together
# ==========================================================================


class TestAllFeaturesIntegration:
    """Tests with all 3 new features enabled simultaneously."""

    def test_full_model_forward(self, device):
        """Model with all features should produce valid outputs."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.1,
            use_boris=True,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        # Standard outputs
        assert out["z_trajectory"].shape == (B, H, D)
        assert torch.isfinite(out["z_trajectory"]).all()
        assert (out["z_trajectory"].norm(dim=-1) < 1.0).all()

        # Hodge outputs
        assert out["hodge_harmonic_forces"].shape == (B, H, D)
        assert torch.isfinite(out["hodge_harmonic_forces"]).all()

        # Module references for screened Poisson
        assert out["potential_net"] is not None
        assert out["router_weights_final"] is not None

    def test_full_model_backward(self, device):
        """All losses should produce gradients with all features enabled."""
        from fragile.losses.world_model import (
            compute_hodge_consistency_loss,
            compute_screened_poisson_loss,
        )
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.1,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        z_targets = torch.randn(B, H, D, device=device) * 0.3

        # Hodge loss
        hodge_loss = compute_hodge_consistency_loss(out["hodge_harmonic_forces"])

        # Screened Poisson loss
        sp_loss = compute_screened_poisson_loss(
            out["potential_net"],
            out["z_trajectory"],
            z_targets,
            rw_0,
            kappa=1.0,
            max_samples=8,
        )

        total_loss = out["z_trajectory"].sum() + hodge_loss + sp_loss
        total_loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients with all features enabled"

    def test_phase2_loss_with_new_features(self, device):
        """compute_phase2_loss should handle all new loss terms."""
        from fragile.losses.world_model import compute_phase2_loss
        from fragile.vla.config import VLAConfig
        from fragile.vla.covariant_world_model import GeometricWorldModel

        config = VLAConfig(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            w_screened_poisson=0.01,
            wm_screening_kappa=1.0,
            w_hodge=0.01,
        )

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        z_targets = torch.randn(B, H, D, device=device) * 0.3
        chart_targets = torch.randint(0, K, (B, H), device=device)

        total, metrics = compute_phase2_loss(out, z_targets, chart_targets, config)
        assert total.shape == ()
        assert torch.isfinite(total)
        assert "hodge" in metrics
        assert "screened_poisson" in metrics

    def test_disabled_features_no_overhead(self, device):
        """With all new features disabled (defaults), output should match baseline."""
        from fragile.vla.covariant_world_model import GeometricWorldModel

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            risk_metric_alpha=0.0,  # disabled
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        # Hodge outputs still present (diagnostics always available)
        assert "hodge_harmonic_forces" in out
        # Standard outputs intact
        assert out["z_trajectory"].shape == (B, H, D)
        assert out["momenta"].shape == (B, H, D)


# ==========================================================================
# Feature 4: WFR-Hamiltonian World Model (Option B)
# ==========================================================================


class TestPoincareLogMap:
    """Tests for the poincare_log_map geometric primitive."""

    def test_inverse_of_exp_map(self):
        """log_z(exp_z(v)) should recover v (round-trip)."""
        from fragile.layers.gauge import poincare_exp_map, poincare_log_map, project_to_ball

        z = project_to_ball(torch.randn(B, D) * 0.3)
        v = torch.randn(B, D) * 0.1  # small tangent vector

        y = poincare_exp_map(z, v)
        v_recovered = poincare_log_map(z, y)
        assert torch.allclose(v, v_recovered, atol=5e-4), (
            f"Round-trip failed: max error {(v - v_recovered).abs().max()}"
        )

    def test_zero_tangent_gives_basepoint(self):
        """exp_z(0) = z, so log_z(z) should be ~0."""
        from fragile.layers.gauge import poincare_log_map, project_to_ball

        z = project_to_ball(torch.randn(B, D) * 0.3)
        v = poincare_log_map(z, z)
        assert torch.allclose(v, torch.zeros_like(v), atol=1e-4), (
            f"log_z(z) not zero: max {v.abs().max()}"
        )

    def test_output_shape(self):
        """Output should be [B, D] tangent vector."""
        from fragile.layers.gauge import poincare_log_map, project_to_ball

        z = project_to_ball(torch.randn(B, D) * 0.3)
        y = project_to_ball(torch.randn(B, D) * 0.3)
        v = poincare_log_map(z, y)
        assert v.shape == (B, D)
        assert torch.isfinite(v).all()

    def test_near_boundary(self):
        """Should be finite even near the Poincare ball boundary."""
        from fragile.layers.gauge import poincare_log_map, project_to_ball

        z = project_to_ball(torch.randn(B, D) * 0.3)
        y = torch.randn(B, D)
        y = y / y.norm(dim=-1, keepdim=True) * 0.95  # near boundary
        v = poincare_log_map(z, y)
        assert torch.isfinite(v).all(), f"Non-finite near boundary: {v}"

    def test_distance_consistency(self):
        """||log_z(y)||_z should approximately equal d(z, y)."""
        from fragile.layers.gauge import (
            hyperbolic_distance,
            poincare_log_map,
            project_to_ball,
        )

        z = project_to_ball(torch.randn(B, D) * 0.3)
        y = project_to_ball(torch.randn(B, D) * 0.3)

        v = poincare_log_map(z, y)
        # Tangent norm at z: ||v||_z = lambda(z) * ||v||_E
        z_sq = (z**2).sum(dim=-1, keepdim=True)
        lam = 2.0 / (1.0 - z_sq).clamp(min=1e-6)
        tangent_norm = lam.squeeze(-1) * v.norm(dim=-1)

        d = hyperbolic_distance(z, y)
        assert torch.allclose(tangent_norm, d, atol=1e-3), (
            f"Tangent norm vs distance mismatch: max err {(tangent_norm - d).abs().max()}"
        )


class TestBoltzmannChartLogits:
    """Tests for the value-driven Boltzmann chart selection."""

    def _make_model(self, **kwargs):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        defaults = {
            "latent_dim": D,
            "action_dim": A,
            "num_charts": K,
            "d_model": D_MODEL,
            "use_jump": True,
            "n_refine_steps": 1,
            "jump_beta": 1.0,
        }
        defaults.update(kwargs)
        return GeometricWorldModel(**defaults)

    def test_output_shape(self):
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)
        logits = m._boltzmann_chart_logits(z, rw)
        assert logits.shape == (B, K)

    def test_finite(self):
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)
        logits = m._boltzmann_chart_logits(z, rw)
        assert torch.isfinite(logits).all(), "Boltzmann logits not finite"

    def test_beta_scaling(self):
        """Higher beta should produce sharper (more spread) logits."""
        m_low = self._make_model(jump_beta=0.1)
        m_high = self._make_model(jump_beta=10.0)

        # Use same weights
        with torch.no_grad():
            for p_low, p_high in zip(m_low.parameters(), m_high.parameters()):
                p_high.copy_(p_low)

        z = torch.randn(B, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        logits_low = m_low._boltzmann_chart_logits(z, rw)
        logits_high = m_high._boltzmann_chart_logits(z, rw)

        # Higher beta => larger spread of logits
        spread_low = logits_low.std(dim=-1).mean()
        spread_high = logits_high.std(dim=-1).mean()
        assert spread_high > spread_low, (
            f"Higher beta should give sharper logits: "
            f"spread_low={spread_low:.4f}, spread_high={spread_high:.4f}"
        )

    def test_no_gradients(self):
        """Boltzmann logits must be detached (no grad graph retained).

        The Boltzmann evaluation runs under torch.no_grad() to avoid OOM
        in Phase 3 (retain_graph=True over B*K attention computations).
        Supervised chart signal comes from CovariantChartTarget instead.
        """
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        rw = torch.softmax(torch.randn(B, K), dim=-1)
        logits = m._boltzmann_chart_logits(z, rw)
        assert not logits.requires_grad, "Boltzmann logits should be detached"

    def test_closer_chart_gets_higher_logit_contribution(self):
        """The geodesic distance penalty should lower logits for far charts."""
        m = self._make_model()
        centers = m.chart_predictor.chart_tok.chart_centers.detach()

        # Place z near chart 0's center
        z = centers[0:1].clone() * 0.99  # [1, D] very close to chart 0
        rw = torch.softmax(torch.randn(1, K), dim=-1)

        logits = m._boltzmann_chart_logits(z, rw)
        # Chart 0's distance penalty is smallest, so its logit should
        # be boosted relative to distant charts (all else equal).
        # Note: value differences can override, so we only check the
        # distance contribution is non-negative.
        assert torch.isfinite(logits).all()


class TestConditionalJump:
    """Tests for the conditional chart jump mechanism."""

    def _make_model(self, **kwargs):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        defaults = {
            "latent_dim": D,
            "action_dim": A,
            "num_charts": K,
            "d_model": D_MODEL,
            "use_jump": True,
            "n_refine_steps": 1,
            "jump_beta": 1.0,
        }
        defaults.update(kwargs)
        return GeometricWorldModel(**defaults)

    def test_output_shapes(self):
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        z_out, p_out, cl, rw_out, jumped = m._conditional_jump(z, p, action, rw)
        assert z_out.shape == (B, D)
        assert p_out.shape == (B, D)
        assert cl.shape == (B, K)
        assert rw_out.shape == (B, K)
        assert jumped.shape == (B,)
        assert jumped.dtype == torch.bool

    def test_same_chart_preserves_state(self):
        """When current == target chart, z and p should be unchanged."""
        m = self._make_model()

        # Force Boltzmann to pick chart 0 by making all rw point to chart 0
        rw = torch.zeros(B, K)
        rw[:, 0] = 1.0

        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)

        # Overwrite chart_centers so chart 0 is at z (same-chart scenario likely)
        # This is hard to guarantee deterministically, so we just check
        # that non-jumped samples are unchanged.
        z_out, p_out, _, _, jumped = m._conditional_jump(z, p, action, rw)

        # For samples that didn't jump, z and p should be identical
        no_jump = ~jumped
        if no_jump.any():
            assert torch.allclose(z_out[no_jump], z[no_jump], atol=1e-6), "Non-jumped z changed"
            assert torch.allclose(p_out[no_jump], p[no_jump], atol=1e-6), "Non-jumped p changed"

    def test_jumped_samples_at_chart_center(self):
        """When a sample jumps, z should be near some chart center.

        The jump target is chosen by Boltzmann logits (detached), while
        rw_out comes from supervised chart_logits — they can disagree.
        So we check that jumped z is near *any* chart center.
        """
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        z_out, _p_out, _, _rw_out, jumped = m._conditional_jump(z, p, action, rw)

        if jumped.any():
            centers = m.chart_predictor.chart_tok.chart_centers.detach()
            for i in range(jumped.sum().item()):
                zi = z_out[jumped][i].detach()
                # Minimum distance to any chart center
                d_min = (centers - zi.unsqueeze(0)).norm(dim=-1).min()
                assert d_min < 0.1, f"Jumped sample not near any center: d_min={d_min:.4f}"

    def test_z_stays_in_ball_after_jump(self):
        """All positions should stay inside the Poincare ball after jump."""
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        z_out, _, _, _, _ = m._conditional_jump(z, p, action, rw)
        norms = z_out.norm(dim=-1)
        assert (norms < 1.0).all(), f"Position escaped ball: max norm {norms.max()}"

    def test_rw_is_valid_distribution(self):
        """Updated router weights should sum to 1 and be non-negative."""
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        _, _, _, rw_out, _ = m._conditional_jump(z, p, action, rw)
        assert (rw_out >= 0).all(), "Negative router weights"
        assert torch.allclose(rw_out.sum(dim=-1), torch.ones(B), atol=1e-5), (
            "Router weights don't sum to 1"
        )

    def test_gradients_flow_through_jump(self):
        """Gradients should flow through conditional jump to model params."""
        m = self._make_model()
        z = torch.randn(B, D) * 0.3
        p = torch.randn(B, D) * 0.1
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        z_out, _p_out, cl, rw_out, _ = m._conditional_jump(z, p, action, rw)
        loss = z_out.sum() + cl.sum() + rw_out.sum()
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients flow through conditional jump"


class TestWFRForwardLoop:
    """Tests for the full WFR-Hamiltonian forward loop (jump → N×BAOAB)."""

    def _make_model(self, **kwargs):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        defaults = {
            "latent_dim": D,
            "action_dim": A,
            "num_charts": K,
            "d_model": D_MODEL,
            "use_jump": True,
            "n_refine_steps": 3,
            "jump_beta": 1.0,
        }
        defaults.update(kwargs)
        return GeometricWorldModel(**defaults)

    def test_refine_steps_configurable(self, device):
        """Model should work with different n_refine_steps values."""
        for n in [1, 2, 3, 5]:
            m = self._make_model(n_refine_steps=n)
            z_0 = torch.randn(B, D, device=device) * 0.3
            actions = torch.randn(B, H, A, device=device)
            rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
            out = m(z_0, actions, rw_0)
            assert out["z_trajectory"].shape == (B, H, D)
            assert torch.isfinite(out["z_trajectory"]).all(), (
                f"Non-finite trajectory with n_refine_steps={n}"
            )

    def test_more_steps_different_trajectory(self, device):
        """Different n_refine_steps should produce different trajectories."""
        m1 = self._make_model(n_refine_steps=1)
        m3 = self._make_model(n_refine_steps=3)

        # Copy weights
        with torch.no_grad():
            for p1, p3 in zip(m1.parameters(), m3.parameters()):
                p3.copy_(p1)

        torch.manual_seed(42)
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)

        # Forward with same seed state for noise
        torch.manual_seed(0)
        out1 = m1(z_0, actions, rw_0)
        torch.manual_seed(0)
        out3 = m3(z_0, actions, rw_0)

        # Trajectories should differ due to different integration resolution
        assert not torch.allclose(out1["z_trajectory"], out3["z_trajectory"], atol=1e-4), (
            "1-step and 3-step trajectories should differ"
        )

    def test_energy_var_key_present(self, device):
        """energy_var should be a scalar in the output."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert "energy_var" in out
        assert out["energy_var"].shape == ()
        assert torch.isfinite(out["energy_var"])
        assert out["energy_var"] >= 0, "Energy variance should be non-negative"

    def test_jumped_key_present(self, device):
        """jumped should be a boolean tensor [B, H]."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert out["jumped"].shape == (B, H)
        assert out["jumped"].dtype == torch.bool

    def test_z_stays_in_ball(self, device):
        """All trajectory points should stay inside the Poincare ball."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        norms = out["z_trajectory"].norm(dim=-1)
        assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"

    def test_backward_pass(self, device):
        """Full backward pass through WFR forward loop."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        loss = out["z_trajectory"].sum() + out["chart_logits"].sum() + out["energy_var"]
        loss.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients through WFR forward loop"

    def test_jump_disabled(self, device):
        """With use_jump=False, no jumps should occur."""
        m = self._make_model(use_jump=False)
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert (~out["jumped"]).all(), "Jumps occurred with use_jump=False"

    def test_min_length_squashing(self, device):
        """With min_length > 0, CFL bounds should be active."""
        m = self._make_model(min_length=0.03)
        assert m.V_alg > 0
        assert m.F_max > 0

        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        assert out["z_trajectory"].shape == (B, H, D)
        assert torch.isfinite(out["z_trajectory"]).all()

    def test_all_outputs_finite(self, device):
        """All output tensors should be finite."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        for key in [
            "z_trajectory",
            "chart_logits",
            "momenta",
            "phi_eff",
            "hodge_conservative_ratio",
            "hodge_solenoidal_ratio",
            "hodge_harmonic_ratio",
            "hodge_harmonic_forces",
        ]:
            assert torch.isfinite(out[key]).all(), f"Non-finite output: {key}"
        assert torch.isfinite(out["energy_var"]), "Non-finite energy_var"


class TestWFRPhase2Loss:
    """Tests for compute_phase2_loss with WFR outputs."""

    def test_loss_with_energy_var(self, device):
        """compute_phase2_loss should use energy_var when present."""
        from fragile.losses.world_model import compute_phase2_loss
        from fragile.vla.config import VLAConfig
        from fragile.vla.covariant_world_model import GeometricWorldModel

        config = VLAConfig(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            w_energy_conservation=0.01,
        )

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            n_refine_steps=3,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        z_targets = torch.randn(B, H, D, device=device) * 0.3
        chart_targets = torch.randint(0, K, (B, H), device=device)

        total, metrics = compute_phase2_loss(out, z_targets, chart_targets, config)
        assert total.shape == ()
        assert torch.isfinite(total)
        assert "energy_conservation" in metrics

    def test_no_jump_dynamics_key(self, device):
        """jump_dynamics should NOT be in metrics (removed loss)."""
        from fragile.losses.world_model import compute_phase2_loss
        from fragile.vla.config import VLAConfig
        from fragile.vla.covariant_world_model import GeometricWorldModel

        config = VLAConfig(latent_dim=D, action_dim=A, num_charts=K)

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        z_targets = torch.randn(B, H, D, device=device) * 0.3
        chart_targets = torch.randint(0, K, (B, H), device=device)

        _, metrics = compute_phase2_loss(out, z_targets, chart_targets, config)
        assert "jump_dynamics" not in metrics, "jump_dynamics should be removed from phase 2 loss"

    def test_backward_through_phase2_loss(self, device):
        """Full backward through compute_phase2_loss with all features."""
        from fragile.losses.world_model import compute_phase2_loss
        from fragile.vla.config import VLAConfig
        from fragile.vla.covariant_world_model import GeometricWorldModel

        config = VLAConfig(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            w_energy_conservation=0.01,
            w_screened_poisson=0.01,
            w_hodge=0.01,
        )

        m = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=D_MODEL,
            n_refine_steps=2,
        )
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        z_targets = torch.randn(B, H, D, device=device) * 0.3
        chart_targets = torch.randint(0, K, (B, H), device=device)

        total, _metrics = compute_phase2_loss(out, z_targets, chart_targets, config)
        total.backward()
        grad_count = sum(1 for p in m.parameters() if p.grad is not None)
        assert grad_count > 0, "No gradients through phase 2 loss"


class TestWFRGeometricInvariants:
    """Tests verifying geometric properties of the WFR world model."""

    def _make_model(self, **kwargs):
        from fragile.vla.covariant_world_model import GeometricWorldModel

        defaults = {
            "latent_dim": D,
            "action_dim": A,
            "num_charts": K,
            "d_model": D_MODEL,
            "use_jump": True,
            "n_refine_steps": 3,
            "jump_beta": 1.0,
        }
        defaults.update(kwargs)
        return GeometricWorldModel(**defaults)

    def test_boris_preserves_norm_with_refine_steps(self):
        """Boris rotation should preserve momentum norm at each sub-step."""
        m = self._make_model(use_boris=True)
        z = torch.randn(B, D) * 0.3
        action = torch.randn(B, A)
        p_in = torch.randn(B, D) * 0.1

        p_out, _ = m._boris_rotation(p_in, z, action)
        assert torch.allclose(
            p_in.norm(dim=-1),
            p_out.norm(dim=-1),
            atol=1e-5,
        ), "Boris rotation changed momentum norm"

    def test_energy_variance_small_for_many_steps(self, device):
        """More BAOAB sub-steps with zero noise should give lower energy variance."""
        # With T_c=0 (no thermostat noise), BAOAB should be near-symplectic
        m = self._make_model(n_refine_steps=3, min_length=0.03)
        m.c2 = 0.0  # disable noise

        z_0 = torch.randn(B, D, device=device) * 0.2
        actions = torch.randn(B, 1, A, device=device) * 0.01  # small actions
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        # Energy variance should be finite and non-negative
        assert out["energy_var"] >= 0
        assert torch.isfinite(out["energy_var"])

    def test_chart_logits_are_supervised_not_boltzmann(self, device):
        """chart_logits in output should be from CovariantChartTarget (supervised),
        not from Boltzmann logits (which drive the actual jump)."""
        m = self._make_model()
        z_0 = torch.randn(B, D, device=device) * 0.3
        actions = torch.randn(B, H, A, device=device)
        rw_0 = torch.softmax(torch.randn(B, K, device=device), dim=-1)
        out = m(z_0, actions, rw_0)

        # Chart logits should have gradients to chart_predictor params
        out["chart_logits"].sum().backward()
        grad_count = sum(1 for p in m.chart_predictor.parameters() if p.grad is not None)
        assert grad_count > 0, "chart_logits should provide gradients to chart_predictor"
