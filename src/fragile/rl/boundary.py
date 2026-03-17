"""Auxiliary boundary-control modules for explicit control-field experiments.

The canonical-action Dreamer path in `train_dreamer.py` uses the action
manifold directly as its motor interface. This module collects utilities for
setups that instead work with explicit control fields and boundary maps.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.attention import CovariantAttention, GeodesicConfig
from fragile.layers.gauge import ConformalMetric
from fragile.layers.primitives import SpectralLinear
from fragile.vla.covariant_world_model import ActionTokenizer, ChartTokenizer


class ControlTokenizer(ActionTokenizer):
    """Alias of ``ActionTokenizer`` used for latent controls."""


def _resolve_action_chart_count(
    num_visual_charts: int,
    num_action_charts: int,
    num_action_macros: int,
) -> int:
    """Resolve the size of the action atlas independently from the visual atlas."""
    if num_action_charts > 0:
        return num_action_charts
    if num_action_macros > 0:
        return num_action_macros
    return num_visual_charts


def _lambda_sq(metric: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Return the scalar conformal metric factor squared."""
    epsilon = float(getattr(metric, "epsilon", 1e-6))
    return metric.conformal_factor(z).pow(2).clamp(min=epsilon)


def raise_control(
    z: torch.Tensor,
    control_cov: torch.Tensor,
    *,
    metric: nn.Module | None = None,
) -> torch.Tensor:
    """Raise a control covector to the tangent control field ``G^{-1} u_cov``."""
    metric_mod = metric if metric is not None else ConformalMetric()
    return control_cov / _lambda_sq(metric_mod, z)


def lower_control(
    z: torch.Tensor,
    control_tan: torch.Tensor,
    *,
    metric: nn.Module | None = None,
) -> torch.Tensor:
    """Lower a tangent control field to the cotangent control ``G u_tan``."""
    metric_mod = metric if metric is not None else ConformalMetric()
    return control_tan * _lambda_sq(metric_mod, z)


def critic_value(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
) -> torch.Tensor:
    """Evaluate the canonical task-value field."""
    return critic.task_value(z, rw)


def _compliance_matrix_from_raw(
    raw: torch.Tensor,
    action_dim: int,
    *,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Build a positive-semidefinite compliance matrix from an unconstrained tensor."""
    batch = raw.shape[0]
    chol = raw.view(batch, action_dim, action_dim)
    eye = torch.eye(action_dim, device=raw.device, dtype=raw.dtype).unsqueeze(0)
    return torch.bmm(chol, chol.transpose(1, 2)) + eps * eye


class GeometricActionEncoder(nn.Module):
    """Infer latent tangent controls from replay actions at the boundary."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        num_action_charts: int = 0,
        num_action_macros: int = 0,
        d_model: int = 128,
    ) -> None:
        super().__init__()
        self.num_action_charts = _resolve_action_chart_count(
            num_charts,
            num_action_charts,
            num_action_macros,
        )
        self.num_action_macros = num_action_macros or self.num_action_charts
        self.action_dim = action_dim
        self.action_tok = ActionTokenizer(action_dim, d_model, latent_dim)
        self.visual_chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.chart_tok = self.visual_chart_tok
        self.action_chart_tok = ChartTokenizer(self.num_action_macros, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.control_out = SpectralLinear(d_model, latent_dim)
        self.macro_logits_out = SpectralLinear(d_model, self.num_action_macros)
        self.motor_nuisance_out = SpectralLinear(d_model, latent_dim)
        self.motor_compliance_out = SpectralLinear(latent_dim, action_dim * action_dim)

    def forward(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        rw: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Encode replay action into boundary macro/nuisance/control variables."""
        act_x, act_z = self.action_tok(action, z)
        vis_chart_x, vis_chart_z = self.visual_chart_tok(rw, z)
        ctx_x = torch.cat([act_x, vis_chart_x], dim=1)
        ctx_z = torch.cat([act_z, vis_chart_z], dim=1)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)

        control_tan = self.control_out(feat)
        macro_logits = self.macro_logits_out(feat)
        macro_probs = F.softmax(macro_logits, dim=-1)
        macro_idx = macro_probs.argmax(dim=-1)
        action_chart_x, action_chart_z = self.action_chart_tok(macro_probs, z)
        feat_action, _ = self.attn(
            z,
            torch.cat([ctx_z, action_chart_z], dim=1),
            feat,
            torch.cat([ctx_x, action_chart_x], dim=1),
            torch.cat([ctx_x, action_chart_x], dim=1),
        )
        motor_nuisance = self.motor_nuisance_out(feat_action)
        motor_compliance = _compliance_matrix_from_raw(
            self.motor_compliance_out(motor_nuisance),
            self.action_dim,
        )
        return {
            "control_tan": control_tan,
            "macro_logits": macro_logits,
            "macro_probs": macro_probs,
            "macro_idx": macro_idx,
            "motor_nuisance": motor_nuisance,
            "motor_compliance": motor_compliance,
        }


class GeometricActionBoundaryDecoder(nn.Module):
    """Decode latent control into motor macro, nuisance/compliance, and texture."""

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        num_charts: int,
        num_action_charts: int = 0,
        num_action_macros: int = 0,
        d_model: int = 128,
        sigma_motor: float = 0.1,
        metric: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.num_action_charts = _resolve_action_chart_count(
            num_charts,
            num_action_charts,
            num_action_macros,
        )
        self.num_action_macros = num_action_macros or self.num_action_charts
        self.metric = metric if metric is not None else ConformalMetric()
        self.register_buffer("sigma_motor", torch.tensor(float(sigma_motor)))
        self.control_tok = ControlTokenizer(latent_dim, d_model, latent_dim)
        self.visual_chart_tok = ChartTokenizer(num_charts, d_model, latent_dim)
        self.chart_tok = self.visual_chart_tok
        self.action_chart_tok = ChartTokenizer(self.num_action_macros, d_model, latent_dim)
        self.z_embed = SpectralLinear(latent_dim, d_model)

        geo_cfg = GeodesicConfig(d_model=d_model, d_latent=latent_dim, n_heads=1)
        self.attn = CovariantAttention(geo_cfg)
        self.macro_logits_head = SpectralLinear(d_model, self.num_action_macros)
        self.macro_embed = nn.Embedding(self.num_action_macros, d_model)
        self.motor_nuisance_head = SpectralLinear(d_model, latent_dim)
        self.motor_compliance_head = SpectralLinear(latent_dim, action_dim * action_dim)
        self.action_head = SpectralLinear(d_model, action_dim)
        self.nuisance_action_head = SpectralLinear(latent_dim, action_dim)

    def motor_texture_std(self, z: torch.Tensor) -> torch.Tensor:
        """Return geometry-scaled motor-texture std with conformal suppression."""
        g_inv_sqrt = torch.rsqrt(_lambda_sq(self.metric, z))
        return self.sigma_motor.to(dtype=z.dtype) * g_inv_sqrt

    def sample_motor_texture(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample execution-only motor texture from the inverse metric law."""
        texture_std = self.motor_texture_std(z)
        texture = torch.randn(z.shape[0], self.action_dim, device=z.device, dtype=z.dtype)
        texture = texture * texture_std.expand(-1, self.action_dim)
        return texture, texture_std.expand(-1, self.action_dim)

    def _features(
        self,
        z: torch.Tensor,
        control: torch.Tensor,
        rw: torch.Tensor,
    ) -> torch.Tensor:
        ctrl_x, ctrl_z = self.control_tok(control, z)
        vis_chart_x, vis_chart_z = self.visual_chart_tok(rw, z)
        ctx_x = torch.cat([ctrl_x, vis_chart_x], dim=1)
        ctx_z = torch.cat([ctrl_z, vis_chart_z], dim=1)
        x_q = self.z_embed(z)
        feat, _ = self.attn(z, ctx_z, x_q, ctx_x, ctx_x)
        return feat

    def forward(
        self,
        z: torch.Tensor,
        control: torch.Tensor,
        rw: torch.Tensor,
        *,
        macro_probs: torch.Tensor | None = None,
        motor_nuisance: torch.Tensor | None = None,
        motor_compliance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict deterministic motor boundary state and geometry-scaled texture stats."""
        feat = self._features(z, control, rw)
        macro_logits = self.macro_logits_head(feat)
        macro_soft = F.softmax(macro_logits, dim=-1)
        if macro_probs is None:
            macro_idx = macro_soft.argmax(dim=-1)
            macro_onehot = F.one_hot(macro_idx, num_classes=self.num_action_macros).to(feat.dtype)
            macro_probs = macro_soft + (macro_onehot - macro_soft).detach()
        else:
            macro_probs = macro_probs.to(device=feat.device, dtype=feat.dtype)
            macro_idx = macro_probs.argmax(dim=-1)
        macro_ctx = macro_probs @ self.macro_embed.weight
        action_chart_x, action_chart_z = self.action_chart_tok(macro_probs, z)
        decoder_feat, _ = self.attn(
            z,
            action_chart_z,
            feat + macro_ctx,
            action_chart_x,
            action_chart_x,
        )
        if motor_nuisance is None:
            motor_nuisance = self.motor_nuisance_head(decoder_feat)
        if motor_compliance is None:
            motor_compliance = _compliance_matrix_from_raw(
                self.motor_compliance_head(motor_nuisance),
                self.action_dim,
            )

        mean_base = self.action_head(decoder_feat)
        nuisance_drive = self.nuisance_action_head(motor_nuisance)
        action_raw = mean_base + torch.bmm(
            motor_compliance,
            nuisance_drive.unsqueeze(-1),
        ).squeeze(-1)
        action_mean = torch.tanh(action_raw)
        texture_std = self.motor_texture_std(z).expand(-1, self.action_dim)
        log_std = torch.log(texture_std.clamp(min=1e-8))
        return {
            "macro_logits": macro_logits,
            "macro_probs": macro_probs,
            "macro_idx": macro_idx,
            "motor_nuisance": motor_nuisance,
            "motor_compliance": motor_compliance,
            "action_raw": action_raw,
            "action_mean": action_mean,
            "texture_std": texture_std,
            "log_std": log_std,
        }

    def decode(
        self,
        z: torch.Tensor,
        control: torch.Tensor,
        rw: torch.Tensor,
        *,
        macro_probs: torch.Tensor | None = None,
        motor_nuisance: torch.Tensor | None = None,
        motor_compliance: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Deterministic action used for dreaming and boundary reconstruction."""
        return self.forward(
            z,
            control,
            rw,
            macro_probs=macro_probs,
            motor_nuisance=motor_nuisance,
            motor_compliance=motor_compliance,
        )["action_mean"]

    def sample_execution_action(
        self,
        z: torch.Tensor,
        control: torch.Tensor,
        rw: torch.Tensor,
        *,
        macro_probs: torch.Tensor | None = None,
        motor_nuisance: torch.Tensor | None = None,
        motor_compliance: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Apply execution-only motor texture in raw action space."""
        out = self.forward(
            z,
            control,
            rw,
            macro_probs=macro_probs,
            motor_nuisance=motor_nuisance,
            motor_compliance=motor_compliance,
        )
        texture, texture_std = self.sample_motor_texture(z)
        action = torch.tanh(out["action_raw"] + texture)
        out["action"] = action
        out["motor_texture"] = texture
        out["texture_std"] = texture_std
        out["log_std"] = torch.log(texture_std.clamp(min=1e-8))
        return out


def critic_control_field(
    critic: nn.Module,
    z: torch.Tensor,
    rw: torch.Tensor,
    *,
    metric: nn.Module | None = None,
    create_graph: bool = False,
    detach_input: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the critic-induced control in covector and tangent form."""
    metric_mod = metric if metric is not None else ConformalMetric()
    if detach_input:
        z_in = z.detach().requires_grad_(True)
    else:
        z_in = z.requires_grad_(True)
    value = critic_value(critic, z_in, rw)
    (control_cov,) = torch.autograd.grad(
        value.sum(),
        z_in,
        create_graph=create_graph,
    )
    control_tan = raise_control(z_in, control_cov, metric=metric_mod)
    return control_cov, control_tan, value
