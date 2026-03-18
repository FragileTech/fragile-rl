"""Tests for orthogonality, causal enclosure, zeno, and geodesic diffusion losses."""

import pytest
import torch
import torch.nn.functional as F

from fragile.layers.gauge import hyperbolic_distance, poincare_exp_map
from fragile.losses.encoder import orthogonality_loss
from fragile.losses.old_macro import (
    compute_dynamics_markov_loss,
    compute_enclosure_loss,
    DynamicsTransitionModel,
    EnclosureProbe,
    GradientReversalLayer,
    grl_alpha_schedule,
    zeno_loss,
)
from fragile.losses.world_model import (
    compute_momentum_targets,
    compute_supervised_wm_loss,
    endpoint_loss,
    geodesic_interpolation,
    momentum_loss,
    position_loss,
)


class _DummyAtlasEncoder(torch.nn.Module):
    def __init__(self, num_charts: int, dyn_codes_per_chart: int, latent_dim: int) -> None:
        super().__init__()
        self.codebook_dyn = torch.nn.Parameter(
            torch.randn(num_charts, dyn_codes_per_chart, latent_dim) * 0.1
        )

    def dynamics_vq(
        self,
        v_local: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chart_idx = router_weights.argmax(dim=-1)
        codebook = self.codebook_dyn[chart_idx]
        diff = v_local.unsqueeze(1) - codebook
        dist = (diff**2).sum(dim=-1)
        code_idx = dist.argmin(dim=-1)
        z_q = codebook[torch.arange(v_local.shape[0]), code_idx]
        vq_loss = ((v_local - z_q.detach()) ** 2).mean() + ((v_local.detach() - z_q) ** 2).mean()
        return z_q, code_idx, code_idx.unsqueeze(1), vq_loss


class TestOrthogonalityLoss:
    def test_zero_when_orthogonal(self):
        """Orthogonal vectors should give zero loss."""
        B, D = 32, 16
        zn = torch.zeros(B, D)
        zn[:, :8] = torch.randn(B, 8)
        ztex = torch.zeros(B, D)
        ztex[:, 8:] = torch.randn(B, 8)
        loss = orthogonality_loss(zn, ztex)
        assert loss.item() < 1e-10

    def test_positive_when_correlated(self):
        """Correlated vectors should give positive loss."""
        B, D = 32, 16
        zn = torch.randn(B, D)
        ztex = zn + 0.1 * torch.randn(B, D)  # highly correlated
        loss = orthogonality_loss(zn, ztex)
        assert loss.item() > 0.01

    def test_different_dims(self):
        """Cross-correlation matrix version for different dims."""
        B = 64
        zn = torch.randn(B, 8)
        ztex = torch.randn(B, 16)
        loss = orthogonality_loss(zn, ztex)
        assert loss.shape == ()
        assert loss.item() >= 0

    def test_gradient_flows(self):
        """Gradients should flow through the loss."""
        zn = torch.randn(4, 8, requires_grad=True)
        ztex = torch.randn(4, 8, requires_grad=True)
        loss = orthogonality_loss(zn, ztex)
        loss.backward()
        assert zn.grad is not None
        assert ztex.grad is not None


class TestGradientReversal:
    def test_forward_identity(self):
        """Forward pass should be identity."""
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 8)
        y = grl(x)
        assert torch.allclose(x, y)

    def test_backward_negates(self):
        """Backward pass should negate gradients."""
        grl = GradientReversalLayer(alpha=1.0)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        # gradient of sum(x) = 1, but GRL negates it
        assert torch.allclose(x.grad, -torch.ones_like(x))

    def test_backward_scales(self):
        """Backward should scale by alpha."""
        grl = GradientReversalLayer(alpha=0.5)
        x = torch.randn(4, 8, requires_grad=True)
        y = grl(x)
        loss = y.sum()
        loss.backward()
        assert torch.allclose(x.grad, -0.5 * torch.ones_like(x))


class TestEnclosureProbe:
    def test_output_shapes(self):
        """Forward pass should return correct shapes for (chart, code) state space."""
        B, D, A, K, C = 16, 8, 6, 4, 32
        probe = EnclosureProbe(
            chart_dim=D,
            ztex_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=C,
            hidden_dim=64,
        )
        S = K * C  # total state count
        chart_embed = torch.randn(B, D)
        action = torch.randn(B, A)
        z_tex = torch.randn(B, D)
        code_idx = torch.randint(0, C, (B,))
        logits_full, logits_base = probe(chart_embed, action, z_tex, code_idx)
        assert logits_full.shape == (B, S)
        assert logits_base.shape == (B, S)

    def test_parameter_count_small(self):
        """Probe should have reasonable parameter count."""
        probe = EnclosureProbe(
            chart_dim=16,
            ztex_dim=16,
            action_dim=6,
            num_charts=8,
            codes_per_chart=32,
            hidden_dim=128,
        )
        n_params = sum(p.numel() for p in probe.parameters())
        assert n_params < 200000  # larger output but still small


class TestComputeEnclosureLoss:
    def test_returns_correct_types(self):
        """Should return (tensor, tensor, dict)."""
        B, D, A, K, C = 16, 8, 6, 4, 32
        probe = EnclosureProbe(
            chart_dim=D,
            ztex_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=C,
            hidden_dim=64,
        )
        chart_embed = torch.randn(B, D, requires_grad=True)
        action = torch.randn(B, A)
        z_tex = torch.randn(B, D, requires_grad=True)
        K_chart_tp1 = torch.randint(0, K, (B,))
        K_code_t = torch.randint(0, C, (B,))
        K_code_tp1 = torch.randint(0, C, (B,))

        loss_enc, loss_probe, diag = compute_enclosure_loss(
            probe,
            chart_embed,
            action,
            z_tex,
            K_chart_tp1,
            K_code_t=K_code_t,
            K_code_tp1=K_code_tp1,
            codes_per_chart=C,
        )
        assert isinstance(loss_enc, torch.Tensor)
        assert isinstance(loss_probe, torch.Tensor)
        assert isinstance(diag, dict)
        assert loss_enc.shape == ()
        assert loss_probe.shape == ()

    def test_diagnostics_keys(self):
        """Diagnostics should contain expected keys."""
        B, D, A, K, C = 8, 4, 2, 3, 8
        probe = EnclosureProbe(
            chart_dim=D,
            ztex_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=C,
            hidden_dim=32,
        )
        chart_embed = torch.randn(B, D)
        action = torch.randn(B, A)
        z_tex = torch.randn(B, D)
        K_chart_tp1 = torch.randint(0, K, (B,))
        K_code_t = torch.randint(0, C, (B,))
        K_code_tp1 = torch.randint(0, C, (B,))

        _, _, diag = compute_enclosure_loss(
            probe,
            chart_embed,
            action,
            z_tex,
            K_chart_tp1,
            K_code_t=K_code_t,
            K_code_tp1=K_code_tp1,
            codes_per_chart=C,
        )
        expected_keys = {"acc_full", "acc_base", "defect_acc", "defect_ce", "ce_full", "ce_base"}
        assert expected_keys == set(diag.keys())

    def test_gradient_reversal_on_ztex(self):
        """Gradients through z_tex should be reversed."""
        B, D, A, K, C = 8, 4, 2, 3, 8
        probe = EnclosureProbe(
            chart_dim=D,
            ztex_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=C,
            hidden_dim=32,
            alpha=1.0,
        )
        z_tex = torch.randn(B, D, requires_grad=True)
        chart_embed = torch.randn(B, D, requires_grad=True)
        action = torch.randn(B, A)
        K_chart_tp1 = torch.randint(0, K, (B,))
        K_code_t = torch.randint(0, C, (B,))
        K_code_tp1 = torch.randint(0, C, (B,))

        loss_enc, _, _ = compute_enclosure_loss(
            probe,
            chart_embed,
            action,
            z_tex,
            K_chart_tp1,
            K_code_t=K_code_t,
            K_code_tp1=K_code_tp1,
            codes_per_chart=C,
        )
        loss_enc.backward()
        # z_tex should have gradients (reversed direction)
        assert z_tex.grad is not None
        # chart_embed should also have gradients (normal direction)
        assert chart_embed.grad is not None

    def test_backward_compat_without_code_args(self):
        """Should still work when K_code args are omitted (defaults to zeros)."""
        B, D, A, K = 8, 4, 2, 3
        probe = EnclosureProbe(
            chart_dim=D,
            ztex_dim=D,
            action_dim=A,
            num_charts=K,
            codes_per_chart=8,
            hidden_dim=32,
        )
        chart_embed = torch.randn(B, D)
        action = torch.randn(B, A)
        z_tex = torch.randn(B, D)
        K_chart_tp1 = torch.randint(0, K, (B,))

        loss_enc, loss_probe, _diag = compute_enclosure_loss(
            probe,
            chart_embed,
            action,
            z_tex,
            K_chart_tp1,
        )
        assert loss_enc.shape == ()
        assert loss_probe.shape == ()


class TestGRLAlphaSchedule:
    def test_zero_at_start(self):
        assert grl_alpha_schedule(0, warmup_steps=100, max_alpha=1.0) == 0.0

    def test_max_after_warmup(self):
        assert grl_alpha_schedule(100, warmup_steps=100, max_alpha=1.0) == 1.0
        assert grl_alpha_schedule(200, warmup_steps=100, max_alpha=1.0) == 1.0

    def test_linear_ramp(self):
        alpha = grl_alpha_schedule(50, warmup_steps=100, max_alpha=1.0)
        assert abs(alpha - 0.5) < 1e-6

    def test_custom_max_alpha(self):
        alpha = grl_alpha_schedule(100, warmup_steps=100, max_alpha=0.5)
        assert abs(alpha - 0.5) < 1e-6


class TestZenoLoss:
    def test_zero_when_identical(self):
        """Identical distributions should give zero loss."""
        B, K = 32, 8
        w = torch.softmax(torch.randn(B, K), dim=-1)
        loss_jsd = zeno_loss(w, w, mode="jsd")
        loss_kl = zeno_loss(w, w, mode="kl")
        assert loss_jsd.item() < 1e-6
        assert loss_kl.item() < 1e-6

    def test_positive_when_different(self):
        """Different distributions should give positive loss."""
        B, K = 32, 8
        w_t = torch.softmax(torch.randn(B, K), dim=-1)
        w_prev = torch.softmax(torch.randn(B, K), dim=-1)
        loss_jsd = zeno_loss(w_t, w_prev, mode="jsd")
        loss_kl = zeno_loss(w_t, w_prev, mode="kl")
        assert loss_jsd.item() > 0.0
        assert loss_kl.item() > 0.0

    def test_jsd_bounded(self):
        """JSD should be bounded in [0, log 2]."""
        import math

        B, K = 64, 8
        # Maximally different: one-hot vs another one-hot
        w_t = torch.zeros(B, K)
        w_t[:, 0] = 1.0
        w_prev = torch.zeros(B, K)
        w_prev[:, 1] = 1.0
        loss = zeno_loss(w_t, w_prev, mode="jsd")
        assert loss.item() <= math.log(2) + 1e-4

    def test_kl_asymmetric(self):
        """KL should be asymmetric."""
        B, K = 32, 8
        w_a = torch.softmax(torch.randn(B, K), dim=-1)
        w_b = torch.softmax(torch.randn(B, K), dim=-1)
        kl_ab = zeno_loss(w_a, w_b, mode="kl")
        kl_ba = zeno_loss(w_b, w_a, mode="kl")
        # Should generally not be equal (asymmetric)
        assert not torch.allclose(kl_ab, kl_ba, atol=1e-6)

    def test_gradient_flows_both(self):
        """Gradients should flow through both w_t and w_t_prev."""
        B, K = 16, 4
        w_t = torch.softmax(torch.randn(B, K), dim=-1).requires_grad_(True)
        w_prev = torch.softmax(torch.randn(B, K), dim=-1).requires_grad_(True)
        loss = zeno_loss(w_t, w_prev, mode="jsd")
        loss.backward()
        assert w_t.grad is not None
        assert w_prev.grad is not None

    def test_scalar_output(self):
        """Loss should be a scalar."""
        B, K = 8, 4
        w_t = torch.softmax(torch.randn(B, K), dim=-1)
        w_prev = torch.softmax(torch.randn(B, K), dim=-1)
        loss = zeno_loss(w_t, w_prev, mode="jsd")
        assert loss.shape == ()

    def test_invalid_mode_raises(self):
        """Unknown mode should raise ValueError."""
        B, K = 4, 4
        w = torch.softmax(torch.randn(B, K), dim=-1)
        with pytest.raises(ValueError, match="Unknown zeno_loss mode"):
            zeno_loss(w, w, mode="invalid")


class TestDynamicsMarkovLoss:
    def test_updates_dynamics_codebook(self):
        """Transition CE should backprop into the trainable dynamics symbols."""
        B, H, D, A, K, C = 6, 4, 5, 3, 3, 4
        atlas_encoder = _DummyAtlasEncoder(K, C, D)
        dyn_model = DynamicsTransitionModel(
            chart_dim=D,
            action_dim=A,
            num_charts=K,
            dyn_codes_per_chart=C,
            hidden_dim=32,
        )
        v_local_all = torch.randn(B, H, D)
        router_logits = torch.randn(B, H, K)
        router_weights_all = F.softmax(router_logits, dim=-1)
        chart_embed_all = torch.randn(B, H, D)
        chart_targets_all = router_weights_all.argmax(dim=-1)
        actions = torch.randn(B, H, A)

        loss, metrics, K_code_dyn_all = compute_dynamics_markov_loss(
            atlas_encoder,
            dyn_model,
            v_local_all,
            router_weights_all,
            chart_embed_all,
            chart_targets_all,
            actions,
            transition_weight=0.7,
            zeno_weight=0.2,
            zeno_mode="jsd",
        )

        expected_keys = {
            "dyn_vq",
            "dyn_trans_ce",
            "dyn_trans_acc",
            "dyn_zeno",
            "dyn_state_flip_rate",
            "dyn_state_entropy",
            "dyn_state_max_prob",
            "dyn_code_flip_rate",
        }
        assert loss.shape == ()
        assert expected_keys == set(metrics.keys())
        assert K_code_dyn_all.shape == (B, H)

        loss.backward()
        assert atlas_encoder.codebook_dyn.grad is not None
        assert atlas_encoder.codebook_dyn.grad.abs().sum() > 0


# ---------------------------------------------------------------------------
# Geodesic diffusion loss tests
# ---------------------------------------------------------------------------


def _random_poincare(B: int, D: int, max_r: float = 0.8) -> torch.Tensor:
    """Sample random points inside the Poincare ball."""
    z = torch.randn(B, D)
    z = z / z.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    r = torch.rand(B, 1) * max_r
    return z * r


class TestGeodesicInterpolation:
    def test_endpoints_match(self):
        """z_0 and z_N should match inputs."""
        B, D, N = 8, 4, 10
        z_start = _random_poincare(B, D)
        z_end = _random_poincare(B, D)
        waypoints = geodesic_interpolation(z_start, z_end, N)
        assert waypoints.shape == (B, N + 1, D)
        assert torch.allclose(waypoints[:, 0], z_start, atol=1e-5)
        assert torch.allclose(waypoints[:, -1], z_end, atol=1e-4)

    def test_intermediate_on_geodesic(self):
        """All waypoints should be inside the ball and distances monotonically increase."""
        B, D, N = 16, 4, 8
        z_start = _random_poincare(B, D, max_r=0.5)
        z_end = _random_poincare(B, D, max_r=0.5)
        waypoints = geodesic_interpolation(z_start, z_end, N)

        # All inside ball
        norms = waypoints.norm(dim=-1)  # [B, N+1]
        assert (norms < 1.0).all(), "Some waypoints outside the Poincare ball"

        # Distances from start should monotonically increase
        for b in range(min(B, 4)):
            dists = []
            for k in range(N + 1):
                d = hyperbolic_distance(
                    z_start[b : b + 1],
                    waypoints[b : b + 1, k],
                )
                dists.append(d.item())
            for i in range(1, len(dists)):
                assert dists[i] >= dists[i - 1] - 1e-4, (
                    f"Distance not monotonic at step {i}: {dists[i]} < {dists[i - 1]}"
                )

    def test_N_equals_1(self):
        """N=1 should return [z_start, z_end]."""
        B, D = 8, 4
        z_start = _random_poincare(B, D)
        z_end = _random_poincare(B, D)
        waypoints = geodesic_interpolation(z_start, z_end, 1)
        assert waypoints.shape == (B, 2, D)
        assert torch.allclose(waypoints[:, 0], z_start, atol=1e-5)
        assert torch.allclose(waypoints[:, 1], z_end, atol=1e-4)


class TestMomentumTargets:
    def test_shapes(self):
        """Output should be [B, N, D]."""
        B, D, N = 8, 4, 6
        z_start = _random_poincare(B, D)
        z_end = _random_poincare(B, D)
        waypoints = geodesic_interpolation(z_start, z_end, N)
        p_targets = compute_momentum_targets(waypoints, dt=0.01)
        assert p_targets.shape == (B, N, D)

    def test_finite_values(self):
        """No NaN or Inf in momentum targets."""
        B, D, N = 16, 4, 8
        z_start = _random_poincare(B, D, max_r=0.5)
        z_end = _random_poincare(B, D, max_r=0.5)
        waypoints = geodesic_interpolation(z_start, z_end, N)
        p_targets = compute_momentum_targets(waypoints, dt=0.01)
        assert torch.isfinite(p_targets).all(), "Non-finite momentum targets"


class TestPositionLoss:
    def test_zero_when_identical(self):
        """Matching trajectories should give near-zero loss."""
        B, D, N = 8, 4, 6
        z = _random_poincare(B, D).unsqueeze(1).expand(-1, N + 1, -1).contiguous()
        loss = position_loss(z, z)
        assert loss.item() < 1e-3  # hyperbolic_distance has eps floor

    def test_positive_when_different(self):
        """Different trajectories should give positive loss."""
        B, D, N = 8, 4, 6
        z_start = _random_poincare(B, D)
        z_end = _random_poincare(B, D)
        traj1 = geodesic_interpolation(z_start, z_end, N)
        traj2 = geodesic_interpolation(z_end, z_start, N)
        loss = position_loss(traj1, traj2)
        assert loss.item() > 0.01

    def test_gradient_flows(self):
        """Gradients should flow through predicted trajectory."""
        B, D, N = 4, 4, 4
        z_pred = _random_poincare(B, D).unsqueeze(1).expand(-1, N + 1, -1).contiguous()
        z_pred = z_pred.clone().requires_grad_(True)
        z_target = _random_poincare(B, D).unsqueeze(1).expand(-1, N + 1, -1).contiguous()
        loss = position_loss(z_pred, z_target)
        loss.backward()
        assert z_pred.grad is not None


class TestEndpointLoss:
    def test_zero_when_identical(self):
        """Same points should give near-zero loss."""
        B, D = 8, 4
        z = _random_poincare(B, D)
        loss = endpoint_loss(z, z)
        assert loss.item() < 1e-3  # hyperbolic_distance has eps floor

    def test_positive_when_different(self):
        """Different points should give positive loss."""
        B, D = 8, 4
        z1 = _random_poincare(B, D)
        z2 = _random_poincare(B, D)
        loss = endpoint_loss(z1, z2)
        assert loss.item() > 0.01


class TestSupervisedWMLoss:
    @pytest.fixture
    def wm_and_config(self):
        """Create a small world model and config for testing."""
        from types import SimpleNamespace

        from fragile.vla.covariant_world_model import GeometricWorldModel

        D, A, K = 4, 2, 4
        wm = GeometricWorldModel(
            latent_dim=D,
            action_dim=A,
            num_charts=K,
            d_model=32,
            hidden_dim=32,
            dt=0.01,
            gamma_friction=1.0,
            T_c=0.1,
            use_boris=False,
            use_jump=False,
            n_refine_steps=2,
        )
        config = SimpleNamespace(
            w_position=1.0,
            w_endpoint=2.0,
            w_momentum_target=0.1,
            w_hodge_perp=0.01,
            w_energy_conservation=0.01,
            wm_dt=0.01,
        )
        return wm, config, D, A, K

    def test_returns_correct_types(self, wm_and_config):
        """Loss should be scalar, metrics should be dict."""
        wm, config, D, A, K = wm_and_config
        B, N = 4, 4
        z_start = _random_poincare(B, D, max_r=0.5)
        z_end = _random_poincare(B, D, max_r=0.5)
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        loss, metrics = compute_supervised_wm_loss(
            wm,
            z_start,
            z_end,
            action,
            rw,
            N,
            config.wm_dt,
            config,
        )
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert isinstance(metrics, dict)
        assert "position" in metrics
        assert "endpoint" in metrics
        assert "momentum_target" in metrics

    def test_gradient_flows_to_wm(self, wm_and_config):
        """WM params should have gradients after backward."""
        wm, config, D, A, K = wm_and_config
        B, N = 4, 4
        z_start = _random_poincare(B, D, max_r=0.5)
        z_end = _random_poincare(B, D, max_r=0.5)
        action = torch.randn(B, A)
        rw = torch.softmax(torch.randn(B, K), dim=-1)

        loss, _ = compute_supervised_wm_loss(
            wm,
            z_start,
            z_end,
            action,
            rw,
            N,
            config.wm_dt,
            config,
        )
        loss.backward()

        has_grad = False
        for p in wm.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowed to world model parameters"
