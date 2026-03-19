"""Standalone topoencoder agent and trainer without a micro world model.

This module packages the pieces discussed for the new symbolic-first setup:

- one observation ``TopoEncoder``,
- one action ``TopoEncoder``,
- one jump operator per manifold,
- the absolute enclosure probe from ``fragile.losses.macro``,
- the differentiable coarse Markov model from ``fragile.losses.markov_model``.

The trainer keeps the Phase-1 encoder stack active on both manifolds while
adding enclosure and coarse symbolic transition losses. There is deliberately
no micro world model here yet; the goal is to learn a cleaner symbolic atlas
and a fast stochastic planner first.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from fragile.checkpoints import compute_grad_norm, compute_param_norm
from fragile.layers import FactorizedJumpOperator, TopoEncoder
from fragile.layers.jump_operator import compute_jump_consistency_loss
from fragile.losses.encoder import (
    _deterministic_st_router_weights,
    compute_phase1_loss,
    get_jump_weight_schedule,
    orthogonality_loss,
)
from fragile.losses.macro import AbsoluteEnclosureProbe, compute_absolute_enclosure_loss
from fragile.losses.markov_model import (
    compute_markov_shape_loss,
    compute_markov_transition_loss,
    MacroTransitionModel,
    soft_macro_state_distribution,
)
from fragile.losses.old_macro import grl_alpha_schedule
from fragile.vla.config import VLAConfig
from fragile.vla.optim import build_encoder_param_groups


def _default_obs_vla_config() -> VLAConfig:
    """Return a fresh observation encoder/loss config."""
    return VLAConfig()


def _default_act_vla_config() -> VLAConfig:
    """Return a fresh action encoder/loss config with action-sized defaults."""
    return VLAConfig(
        input_dim=6,
        feature_dim=6,
        hidden_dim=128,
        latent_dim=16,
        num_charts=8,
        codes_per_chart=32,
        batch_size=256,
        input_affine_enabled=True,
    )


def _encoder_kwargs(config: VLAConfig) -> dict[str, Any]:
    """Translate a ``VLAConfig`` into ``TopoEncoder`` constructor kwargs."""
    return {
        "input_dim": config.input_dim,
        "hidden_dim": config.hidden_dim,
        "latent_dim": config.latent_dim,
        "num_charts": config.num_charts,
        "codes_per_chart": config.codes_per_chart,
        "covariant_attn_tau_min": config.covariant_attn_tau_min,
        "covariant_attn_denom_min": config.covariant_attn_denom_min,
        "covariant_attn_transport_eps": config.covariant_attn_transport_eps,
        "soft_equiv_metric": config.soft_equiv_metric,
        "soft_equiv_bundle_size": (
            config.soft_equiv_bundle_size if config.soft_equiv_bundle_size > 0 else None
        ),
        "soft_equiv_hidden_dim": config.soft_equiv_hidden_dim,
        "soft_equiv_use_spectral_norm": config.soft_equiv_use_spectral_norm,
        "soft_equiv_zero_self_mixing": config.soft_equiv_zero_self_mixing,
        "soft_equiv_soft_assign": config.soft_equiv_soft_assign,
        "soft_equiv_temperature": config.soft_equiv_temperature,
        "input_affine_enabled": config.input_affine_enabled,
        "input_affine_learnable": config.input_affine_learnable,
        "input_affine_min_scale": config.input_affine_min_scale,
        "film_conditioning": True,
        "commitment_beta": config.commitment_beta,
        "codebook_loss_weight": config.codebook_loss_weight,
    }


def _prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Attach a namespace prefix to metric keys."""
    return {f"{prefix}/{key}": float(value) for key, value in metrics.items()}


def _scatter_valid(
    values: torch.Tensor,
    valid_idx: torch.Tensor,
    leading_shape: tuple[int, ...],
    *,
    fill_value: float | int = 0,
) -> torch.Tensor:
    """Scatter valid rows back to the full batch/time grid."""
    flat_size = 1
    for dim in leading_shape:
        flat_size *= int(dim)
    full = values.new_full((flat_size, *values.shape[1:]), fill_value)
    full[valid_idx] = values
    return full.reshape(*leading_shape, *values.shape[1:])


def _flatten_selected(values: torch.Tensor, valid_idx: torch.Tensor) -> torch.Tensor:
    """Flatten leading batch/time axes and keep only valid entries."""
    return values.reshape(-1, *values.shape[2:])[valid_idx]


def _mean_metrics(metric_list: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of scalar metric dictionaries."""
    if not metric_list:
        return {}
    keys = set().union(*(metrics.keys() for metrics in metric_list))
    averaged: dict[str, float] = {}
    for key in keys:
        values = [metrics[key] for metrics in metric_list if key in metrics]
        if values:
            averaged[key] = float(sum(values) / len(values))
    return averaged


@dataclass
class FragileAgentConfig:
    """Module-composition config for ``FragileAgent``.

    The observation and action manifolds each carry their own ``VLAConfig`` so
    the Phase-1 encoder stack can be reused directly without pulling in the old
    script-level training code.
    """

    obs_encoder: VLAConfig = field(default_factory=_default_obs_vla_config)
    act_encoder: VLAConfig = field(default_factory=_default_act_vla_config)
    obs_jump_curvature: float = 1.0
    act_jump_curvature: float = 1.0
    enclosure_hidden_dim: int = 128
    enclosure_dropout: float = 0.1
    enclosure_alpha: float = 1.0
    markov_hidden_dim: int = 128
    markov_feature_scale: float = 0.1
    markov_use_residual_transition: bool = True
    markov_residual_scale: float = 1.0
    markov_learn_reward: bool = True
    markov_learn_continuation: bool = True
    markov_initial_continuation: float = 0.99


@dataclass
class FragileAgentTrainerConfig:
    """Optimization and loss config for ``FragileAgentTrainer``."""

    lr_encoder: float = 1e-3
    lr_probe: float = 3e-3
    lr_markov: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    lr_chart_centers_scale: float | None = None
    lr_codebook_scale: float | None = None
    use_cosine_lr: bool = False
    cosine_t_max: int = 100
    cosine_eta_min: float = 1e-6
    routing_tau: float = -1.0
    routing_tau_end: float | None = None
    routing_tau_anneal_steps: int = 0
    eval_routing_tau: float = -1.0
    macro_chart_tau: float = 1.0
    macro_code_tau: float = 1.0
    weight_obs_phase1: float = 1.0
    weight_act_phase1: float = 1.0
    weight_enclosure_encoder: float = 1.0
    weight_enclosure_probe: float = 1.0
    weight_markov_transition: float = 1.0
    weight_markov_shape: float = 1.0
    enclosure_alpha_max: float = 1.0
    enclosure_alpha_warmup_steps: int = 5000
    phase1_frame_mode: str = "all"


class FragileAgent(nn.Module):
    """Compose the standalone symbolic-first agent modules.

    The class exposes a tensor-first API:

    - ``encode_observations`` and ``encode_actions`` run the corresponding
      topoencoder on sequences of frames,
    - ``forward_batch`` encodes both streams, constructs aligned transitions,
      and attaches differentiable symbolic state distributions for the coarse
      Markov path.
    """

    def __init__(self, config: FragileAgentConfig | None = None) -> None:
        super().__init__()
        self.config = copy.deepcopy(config or FragileAgentConfig())

        self.obs_encoder = TopoEncoder(**_encoder_kwargs(self.config.obs_encoder))
        self.act_encoder = TopoEncoder(**_encoder_kwargs(self.config.act_encoder))

        self.obs_jump_operator = FactorizedJumpOperator(
            num_charts=self.config.obs_encoder.num_charts,
            latent_dim=self.config.obs_encoder.latent_dim,
            curvature=self.config.obs_jump_curvature,
        )
        self.act_jump_operator = FactorizedJumpOperator(
            num_charts=self.config.act_encoder.num_charts,
            latent_dim=self.config.act_encoder.latent_dim,
            curvature=self.config.act_jump_curvature,
        )

        self.enclosure_probe = AbsoluteEnclosureProbe(
            obs_struct_dim=self.config.obs_encoder.latent_dim,
            act_struct_dim=self.config.act_encoder.latent_dim,
            obs_tex_dim=self.config.obs_encoder.latent_dim,
            act_tex_dim=self.config.act_encoder.latent_dim,
            num_obs_charts=self.config.obs_encoder.num_charts,
            obs_codes_per_chart=self.config.obs_encoder.codes_per_chart,
            hidden_dim=self.config.enclosure_hidden_dim,
            alpha=self.config.enclosure_alpha,
            dropout=self.config.enclosure_dropout,
        )

        self.macro_model = MacroTransitionModel(
            obs_latent_dim=self.config.obs_encoder.latent_dim,
            act_latent_dim=self.config.act_encoder.latent_dim,
            num_obs_charts=self.config.obs_encoder.num_charts,
            obs_codes_per_chart=self.config.obs_encoder.codes_per_chart,
            num_act_charts=self.config.act_encoder.num_charts,
            act_codes_per_chart=self.config.act_encoder.codes_per_chart,
            hidden_dim=self.config.markov_hidden_dim,
            feature_scale=self.config.markov_feature_scale,
            use_residual_transition=self.config.markov_use_residual_transition,
            residual_scale=self.config.markov_residual_scale,
            learn_reward=self.config.markov_learn_reward,
            learn_continuation=self.config.markov_learn_continuation,
            initial_continuation=self.config.markov_initial_continuation,
        )

    @property
    def num_obs_states(self) -> int:
        """Number of flattened observation macro symbols."""
        return self.config.obs_encoder.num_charts * self.config.obs_encoder.codes_per_chart

    @property
    def num_act_states(self) -> int:
        """Number of flattened action macro symbols."""
        return self.config.act_encoder.num_charts * self.config.act_encoder.codes_per_chart

    @classmethod
    def from_dims(
        cls,
        obs_dim: int,
        act_dim: int,
        *,
        config: FragileAgentConfig | None = None,
    ) -> FragileAgent:
        """Build an agent while overriding the observation/action input dims."""
        cfg = copy.deepcopy(config or FragileAgentConfig())
        cfg.obs_encoder.input_dim = obs_dim
        cfg.obs_encoder.feature_dim = obs_dim
        cfg.act_encoder.input_dim = act_dim
        cfg.act_encoder.feature_dim = act_dim
        return cls(cfg)

    def _normalize_sequence_input(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Validate sequence input and return a boolean mask."""
        if x.dim() != 3:
            msg = "Sequence inputs must have shape [B, T, D]."
            raise ValueError(msg)
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        else:
            if mask.shape != x.shape[:2]:
                msg = "mask must have shape [B, T] matching the leading obs/action axes."
                raise ValueError(msg)
            mask = mask.to(device=x.device, dtype=torch.bool)
        return x, mask

    def _encode_sequence(
        self,
        model: TopoEncoder,
        x: torch.Tensor,
        mask: torch.Tensor | None,
        *,
        routing_tau: float,
    ) -> dict[str, torch.Tensor]:
        """Encode one observation/action sequence with the full Phase-1 outputs."""
        x, mask = self._normalize_sequence_input(x, mask)
        batch_size, horizon, feat_dim = x.shape
        flat_x = x.reshape(batch_size * horizon, feat_dim)
        flat_mask = mask.reshape(batch_size * horizon)
        valid_idx = flat_mask.nonzero(as_tuple=True)[0]
        if valid_idx.numel() == 0:
            msg = "Received a fully masked batch; at least one valid frame is required."
            raise ValueError(msg)

        time_index = torch.arange(horizon, device=x.device, dtype=torch.long).expand(
            batch_size, -1
        )
        last_valid_t = (
            torch.where(mask, time_index, -torch.ones_like(time_index)).max(dim=1).values
        )
        anchor_batch = (last_valid_t >= 0).nonzero(as_tuple=True)[0]
        anchor_flat_idx = anchor_batch * horizon + last_valid_t[anchor_batch]
        valid_lookup = torch.full(
            (batch_size * horizon,),
            -1,
            device=x.device,
            dtype=torch.long,
        )
        valid_lookup[valid_idx] = torch.arange(valid_idx.numel(), device=x.device)
        anchor_valid_pos = valid_lookup[anchor_flat_idx]

        x_valid = flat_x[valid_idx]
        x_model_valid = model.normalize_input(x_valid)
        (
            chart_idx_valid,
            code_idx_valid,
            z_n_valid,
            z_tex_valid,
            enc_router_weights_valid,
            z_geo_valid,
            vq_loss,
            indices_stack_valid,
            z_n_all_valid,
            c_bar_valid,
            v_local_valid,
            z_q_blended_valid,
        ) = model.encoder(x_model_valid, routing_tau=routing_tau)

        x_recon_valid, dec_router_weights_valid, _ = model.decode(
            z_geo_valid,
            chart_index=None,
            router_weights=enc_router_weights_valid,
            routing_tau=routing_tau,
        )

        router_scores_valid = getattr(model.encoder, "_last_router_scores_live", None)
        soft_router_weights_valid = getattr(model.encoder, "_last_soft_router_weights_live", None)
        if soft_router_weights_valid is None:
            soft_router_weights_valid = enc_router_weights_valid
        usage_router_weights_valid = enc_router_weights_valid
        if router_scores_valid is not None:
            usage_router_weights_valid = _deterministic_st_router_weights(router_scores_valid)

        full_shape = (batch_size, horizon)
        out = {
            "input": x,
            "mask": mask,
            "valid_idx": valid_idx,
            "anchor_valid_pos": anchor_valid_pos,
            "num_valid": torch.tensor(valid_idx.numel(), device=x.device),
            "x_valid": x_valid,
            "x_recon_valid": x_recon_valid,
            "x_recon": _scatter_valid(x_recon_valid, valid_idx, full_shape),
            "vq_loss": vq_loss,
            "enc_router_weights_valid": enc_router_weights_valid,
            "enc_router_weights": _scatter_valid(enc_router_weights_valid, valid_idx, full_shape),
            "soft_router_weights_valid": soft_router_weights_valid,
            "soft_router_weights": _scatter_valid(
                soft_router_weights_valid,
                valid_idx,
                full_shape,
            ),
            "usage_router_weights_valid": usage_router_weights_valid,
            "usage_router_weights": _scatter_valid(
                usage_router_weights_valid,
                valid_idx,
                full_shape,
            ),
            "dec_router_weights_valid": dec_router_weights_valid,
            "dec_router_weights": _scatter_valid(dec_router_weights_valid, valid_idx, full_shape),
            "chart_idx_valid": chart_idx_valid,
            "chart_idx": _scatter_valid(chart_idx_valid, valid_idx, full_shape),
            "code_idx_valid": code_idx_valid,
            "code_idx": _scatter_valid(code_idx_valid, valid_idx, full_shape),
            "z_geo_valid": z_geo_valid,
            "z_geo": _scatter_valid(z_geo_valid, valid_idx, full_shape),
            "z_n_valid": z_n_valid,
            "z_n": _scatter_valid(z_n_valid, valid_idx, full_shape),
            "z_tex_valid": z_tex_valid,
            "z_tex": _scatter_valid(z_tex_valid, valid_idx, full_shape),
            "z_n_all_valid": z_n_all_valid,
            "z_n_all": _scatter_valid(z_n_all_valid, valid_idx, full_shape),
            "c_bar_valid": c_bar_valid,
            "c_bar": _scatter_valid(c_bar_valid, valid_idx, full_shape),
            "v_local_valid": v_local_valid,
            "v_local": _scatter_valid(v_local_valid, valid_idx, full_shape),
            "z_q_blended_valid": z_q_blended_valid,
            "z_q_blended": _scatter_valid(z_q_blended_valid, valid_idx, full_shape),
            "indices_stack_valid": indices_stack_valid,
            "indices_stack": _scatter_valid(indices_stack_valid, valid_idx, full_shape),
        }

        if router_scores_valid is not None:
            out["router_scores_valid"] = router_scores_valid
            out["router_scores"] = _scatter_valid(router_scores_valid, valid_idx, full_shape)
        else:
            out["router_scores_valid"] = None
            out["router_scores"] = None

        return out

    def _symbolize_sequence(
        self,
        encoded: dict[str, torch.Tensor],
        chart_centers: torch.Tensor,
        codebook: torch.Tensor,
        *,
        chart_tau: float,
        code_tau: float,
    ) -> dict[str, torch.Tensor]:
        """Attach a differentiable macro-symbol distribution to encoded latents."""
        symbol_valid = soft_macro_state_distribution(
            encoded["z_geo_valid"],
            chart_centers,
            codebook,
            chart_tau=chart_tau,
            code_tau=code_tau,
        )
        codes_per_chart = int(codebook.shape[1])
        state_idx_valid = symbol_valid["state_idx"]
        chart_idx_valid = torch.div(state_idx_valid, codes_per_chart, rounding_mode="floor")
        code_idx_valid = state_idx_valid.remainder(codes_per_chart)

        full_shape = tuple(int(dim) for dim in encoded["mask"].shape)
        valid_idx = encoded["valid_idx"]
        sample_keys = [
            "z_latent",
            "router_weights",
            "chart_logits",
            "chart_probs",
            "c_bar",
            "v_local",
            "code_logits",
            "code_probs",
            "state_log_probs",
            "state_probs",
            "state_entropy",
            "state_value_entropy",
            "chart_entropy",
            "macro_state_mean",
            "hard_state_point",
        ]
        out = {
            "chart_centers": symbol_valid["chart_centers"],
            "chart_tangent_points": symbol_valid["chart_tangent_points"],
            "codebook": symbol_valid["codebook"],
            "code_tangent_points": symbol_valid["code_tangent_points"],
            "state_points": symbol_valid["state_points"],
            "state_tangent_points": symbol_valid["state_tangent_points"],
            "dictionary_chart_idx": symbol_valid["chart_idx"],
            "dictionary_code_idx": symbol_valid["code_idx"],
            "state_idx_valid": state_idx_valid,
            "state_idx": _scatter_valid(state_idx_valid, valid_idx, full_shape),
            "chart_idx_valid": chart_idx_valid,
            "chart_idx": _scatter_valid(chart_idx_valid, valid_idx, full_shape),
            "code_idx_valid": code_idx_valid,
            "code_idx": _scatter_valid(code_idx_valid, valid_idx, full_shape),
        }
        for key in sample_keys:
            value = symbol_valid[key]
            out[f"{key}_valid"] = value
            out[key] = _scatter_valid(value, valid_idx, full_shape)
        return out

    def encode_observations(
        self,
        obs: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        routing_tau: float = -1.0,
    ) -> dict[str, torch.Tensor]:
        """Encode an observation sequence with the observation topoencoder."""
        return self._encode_sequence(self.obs_encoder, obs, mask, routing_tau=routing_tau)

    def encode_actions(
        self,
        act: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        routing_tau: float = -1.0,
    ) -> dict[str, torch.Tensor]:
        """Encode an action sequence with the action topoencoder."""
        return self._encode_sequence(self.act_encoder, act, mask, routing_tau=routing_tau)

    def forward_batch(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
        routing_tau: float = -1.0,
        macro_chart_tau: float = 1.0,
        macro_code_tau: float = 1.0,
        compute_macro: bool = True,
    ) -> dict[str, Any]:
        """Encode a batch of trajectories and build aligned transition views."""
        obs, mask = self._normalize_sequence_input(obs, mask)
        act, act_mask = self._normalize_sequence_input(act, mask)
        if obs.shape[:2] != act.shape[:2]:
            msg = "obs and act must share the same [B, T] leading dimensions."
            raise ValueError(msg)
        if act.shape[-1] != self.config.act_encoder.input_dim:
            msg = "Action feature dimension does not match act_encoder.input_dim."
            raise ValueError(msg)
        if obs.shape[-1] != self.config.obs_encoder.input_dim:
            msg = "Observation feature dimension does not match obs_encoder.input_dim."
            raise ValueError(msg)

        obs_out = self._encode_sequence(self.obs_encoder, obs, mask, routing_tau=routing_tau)
        act_out = self._encode_sequence(self.act_encoder, act, act_mask, routing_tau=routing_tau)

        obs_macro = None
        act_macro = None
        if compute_macro:
            obs_macro = self._symbolize_sequence(
                obs_out,
                self.obs_encoder.encoder.chart_centers,
                self.obs_encoder.encoder.codebook,
                chart_tau=macro_chart_tau,
                code_tau=macro_code_tau,
            )
            act_macro = self._symbolize_sequence(
                act_out,
                self.act_encoder.encoder.chart_centers,
                self.act_encoder.encoder.codebook,
                chart_tau=macro_chart_tau,
                code_tau=macro_code_tau,
            )

        transition_mask = mask[:, :-1] & mask[:, 1:]
        transition_valid_idx = transition_mask.reshape(-1).nonzero(as_tuple=True)[0]

        transitions: dict[str, Any] = {
            "mask": transition_mask,
            "valid_idx": transition_valid_idx,
            "num_valid": int(transition_valid_idx.numel()),
            "obs_chart_t": obs_out["chart_idx"][:, :-1],
            "obs_code_t": obs_out["code_idx"][:, :-1],
            "act_chart_t": act_out["chart_idx"][:, :-1],
            "act_code_t": act_out["code_idx"][:, :-1],
            "obs_chart_tp1": obs_out["chart_idx"][:, 1:],
            "obs_code_tp1": obs_out["code_idx"][:, 1:],
            "obs_z_n_t": obs_out["z_n"][:, :-1],
            "obs_z_tex_t": obs_out["z_tex"][:, :-1],
            "act_z_n_t": act_out["z_n"][:, :-1],
            "act_z_tex_t": act_out["z_tex"][:, :-1],
            "obs_z_geo_t": obs_out["z_geo"][:, :-1],
            "obs_z_geo_tp1": obs_out["z_geo"][:, 1:],
            "act_z_geo_t": act_out["z_geo"][:, :-1],
            "obs_state_probs_t": None if obs_macro is None else obs_macro["state_probs"][:, :-1],
            "act_state_probs_t": None if act_macro is None else act_macro["state_probs"][:, :-1],
            "obs_state_probs_tp1": None if obs_macro is None else obs_macro["state_probs"][:, 1:],
            "obs_state_idx_t": None if obs_macro is None else obs_macro["state_idx"][:, :-1],
            "act_state_idx_t": None if act_macro is None else act_macro["state_idx"][:, :-1],
            "obs_state_idx_tp1": None if obs_macro is None else obs_macro["state_idx"][:, 1:],
        }
        if transition_valid_idx.numel() > 0:
            transitions.update({
                "obs_chart_t_valid": _flatten_selected(
                    transitions["obs_chart_t"], transition_valid_idx
                ),
                "obs_code_t_valid": _flatten_selected(
                    transitions["obs_code_t"], transition_valid_idx
                ),
                "act_chart_t_valid": _flatten_selected(
                    transitions["act_chart_t"], transition_valid_idx
                ),
                "act_code_t_valid": _flatten_selected(
                    transitions["act_code_t"], transition_valid_idx
                ),
                "obs_chart_tp1_valid": _flatten_selected(
                    transitions["obs_chart_tp1"],
                    transition_valid_idx,
                ),
                "obs_code_tp1_valid": _flatten_selected(
                    transitions["obs_code_tp1"],
                    transition_valid_idx,
                ),
                "obs_z_n_t_valid": _flatten_selected(
                    transitions["obs_z_n_t"], transition_valid_idx
                ),
                "obs_z_tex_t_valid": _flatten_selected(
                    transitions["obs_z_tex_t"],
                    transition_valid_idx,
                ),
                "act_z_n_t_valid": _flatten_selected(
                    transitions["act_z_n_t"], transition_valid_idx
                ),
                "act_z_tex_t_valid": _flatten_selected(
                    transitions["act_z_tex_t"],
                    transition_valid_idx,
                ),
                "obs_z_geo_t_valid": _flatten_selected(
                    transitions["obs_z_geo_t"],
                    transition_valid_idx,
                ),
                "obs_z_geo_tp1_valid": _flatten_selected(
                    transitions["obs_z_geo_tp1"],
                    transition_valid_idx,
                ),
                "act_z_geo_t_valid": _flatten_selected(
                    transitions["act_z_geo_t"],
                    transition_valid_idx,
                ),
            })
            if obs_macro is not None and act_macro is not None:
                transitions.update({
                    "obs_state_probs_t_valid": _flatten_selected(
                        transitions["obs_state_probs_t"],
                        transition_valid_idx,
                    ),
                    "act_state_probs_t_valid": _flatten_selected(
                        transitions["act_state_probs_t"],
                        transition_valid_idx,
                    ),
                    "obs_state_probs_tp1_valid": _flatten_selected(
                        transitions["obs_state_probs_tp1"],
                        transition_valid_idx,
                    ),
                    "obs_state_idx_t_valid": _flatten_selected(
                        transitions["obs_state_idx_t"],
                        transition_valid_idx,
                    ),
                    "act_state_idx_t_valid": _flatten_selected(
                        transitions["act_state_idx_t"],
                        transition_valid_idx,
                    ),
                    "obs_state_idx_tp1_valid": _flatten_selected(
                        transitions["obs_state_idx_tp1"],
                        transition_valid_idx,
                    ),
                })
            else:
                transitions.update({
                    "obs_state_probs_t_valid": None,
                    "act_state_probs_t_valid": None,
                    "obs_state_probs_tp1_valid": None,
                    "obs_state_idx_t_valid": None,
                    "act_state_idx_t_valid": None,
                    "obs_state_idx_tp1_valid": None,
                })
        else:
            obs_latent_dim = self.config.obs_encoder.latent_dim
            act_latent_dim = self.config.act_encoder.latent_dim
            transitions.update({
                "obs_chart_t_valid": obs_out["chart_idx_valid"].new_empty((0,)),
                "obs_code_t_valid": obs_out["code_idx_valid"].new_empty((0,)),
                "act_chart_t_valid": act_out["chart_idx_valid"].new_empty((0,)),
                "act_code_t_valid": act_out["code_idx_valid"].new_empty((0,)),
                "obs_chart_tp1_valid": obs_out["chart_idx_valid"].new_empty((0,)),
                "obs_code_tp1_valid": obs_out["code_idx_valid"].new_empty((0,)),
                "obs_z_n_t_valid": obs_out["z_n_valid"].new_empty((0, obs_latent_dim)),
                "obs_z_tex_t_valid": obs_out["z_tex_valid"].new_empty((0, obs_latent_dim)),
                "act_z_n_t_valid": act_out["z_n_valid"].new_empty((0, act_latent_dim)),
                "act_z_tex_t_valid": act_out["z_tex_valid"].new_empty((0, act_latent_dim)),
                "obs_z_geo_t_valid": obs_out["z_geo_valid"].new_empty((0, obs_latent_dim)),
                "obs_z_geo_tp1_valid": obs_out["z_geo_valid"].new_empty((0, obs_latent_dim)),
                "act_z_geo_t_valid": act_out["z_geo_valid"].new_empty((0, act_latent_dim)),
            })
            if obs_macro is not None and act_macro is not None:
                transitions.update({
                    "obs_state_probs_t_valid": obs_macro["state_probs_valid"].new_empty(
                        (0, self.num_obs_states),
                    ),
                    "act_state_probs_t_valid": act_macro["state_probs_valid"].new_empty(
                        (0, self.num_act_states),
                    ),
                    "obs_state_probs_tp1_valid": obs_macro["state_probs_valid"].new_empty(
                        (0, self.num_obs_states),
                    ),
                    "obs_state_idx_t_valid": obs_macro["state_idx_valid"].new_empty((0,)),
                    "act_state_idx_t_valid": act_macro["state_idx_valid"].new_empty((0,)),
                    "obs_state_idx_tp1_valid": obs_macro["state_idx_valid"].new_empty((0,)),
                })
            else:
                transitions.update({
                    "obs_state_probs_t_valid": None,
                    "act_state_probs_t_valid": None,
                    "obs_state_probs_tp1_valid": None,
                    "obs_state_idx_t_valid": None,
                    "act_state_idx_t_valid": None,
                    "obs_state_idx_tp1_valid": None,
                })

        return {
            "obs": obs_out,
            "act": act_out,
            "macro": {
                "obs": obs_macro,
                "act": act_macro,
            },
            "transitions": transitions,
            "mask": mask,
            "routing_tau": routing_tau,
            "macro_chart_tau": macro_chart_tau,
            "macro_code_tau": macro_code_tau,
        }


class FragileAgentTrainer:
    """Train ``FragileAgent`` without a micro world model.

    The trainer keeps three optimizers:

    - one optimizer for both topoencoders and both jump operators,
    - one optimizer for the enclosure probe,
    - one optimizer for the coarse Markov model.

    The Markov transition loss is allowed to reshape the symbolic geometry via
    the live current observation/action encoders, while the next-state target is
    detached. The Markov shape loss then uses the coarse predictor as a teacher
    for the live next-observation symbolizer.
    """

    def __init__(
        self,
        agent: FragileAgent,
        config: FragileAgentTrainerConfig | None = None,
    ) -> None:
        self.agent = agent
        self.config = copy.deepcopy(config or FragileAgentTrainerConfig())
        self.global_step = 0

        obs_lr_chart_scale = (
            self.config.lr_chart_centers_scale
            if self.config.lr_chart_centers_scale is not None
            else self.agent.config.obs_encoder.lr_chart_centers_scale
        )
        obs_lr_code_scale = (
            self.config.lr_codebook_scale
            if self.config.lr_codebook_scale is not None
            else self.agent.config.obs_encoder.lr_codebook_scale
        )
        act_lr_chart_scale = (
            self.config.lr_chart_centers_scale
            if self.config.lr_chart_centers_scale is not None
            else self.agent.config.act_encoder.lr_chart_centers_scale
        )
        act_lr_code_scale = (
            self.config.lr_codebook_scale
            if self.config.lr_codebook_scale is not None
            else self.agent.config.act_encoder.lr_codebook_scale
        )

        encoder_param_groups = build_encoder_param_groups(
            self.agent.obs_encoder,
            self.agent.obs_jump_operator,
            base_lr=self.config.lr_encoder,
            lr_chart_centers_scale=obs_lr_chart_scale,
            lr_codebook_scale=obs_lr_code_scale,
        )
        encoder_param_groups.extend(
            build_encoder_param_groups(
                self.agent.act_encoder,
                self.agent.act_jump_operator,
                base_lr=self.config.lr_encoder,
                lr_chart_centers_scale=act_lr_chart_scale,
                lr_codebook_scale=act_lr_code_scale,
            )
        )

        self.encoder_optimizer = Adam(
            encoder_param_groups,
            weight_decay=self.config.weight_decay,
        )
        self.probe_optimizer = Adam(
            self.agent.enclosure_probe.parameters(),
            lr=self.config.lr_probe,
            weight_decay=self.config.weight_decay,
        )
        self.markov_optimizer = Adam(
            self.agent.macro_model.parameters(),
            lr=self.config.lr_markov,
            weight_decay=self.config.weight_decay,
        )

        self.encoder_scheduler = None
        if self.config.use_cosine_lr:
            self.encoder_scheduler = CosineAnnealingLR(
                self.encoder_optimizer,
                T_max=self.config.cosine_t_max,
                eta_min=self.config.cosine_eta_min,
            )

    @property
    def device(self) -> torch.device:
        """Device inferred from the agent parameters."""
        return next(self.agent.parameters()).device

    def routing_tau_for_step(self, global_step: int | None = None, *, training: bool) -> float:
        """Return the routing temperature for the current step."""
        if not training:
            return self.config.eval_routing_tau
        if self.config.routing_tau_end is None or self.config.routing_tau_anneal_steps <= 0:
            return self.config.routing_tau
        step = self.global_step if global_step is None else global_step
        mix = min(max(step / self.config.routing_tau_anneal_steps, 0.0), 1.0)
        return self.config.routing_tau + mix * (
            self.config.routing_tau_end - self.config.routing_tau
        )

    def enclosure_alpha_for_step(self, global_step: int | None = None) -> float:
        """Return the GRL alpha for the enclosure probe."""
        step = self.global_step if global_step is None else global_step
        return grl_alpha_schedule(
            step,
            warmup_steps=self.config.enclosure_alpha_warmup_steps,
            max_alpha=self.config.enclosure_alpha_max,
        )

    def _move_batch_to_device(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Support dict or tuple batches and move tensors to the agent device."""
        if isinstance(batch, dict):
            obs = batch.get("obs", batch.get("observation"))
            act = batch.get("act", batch.get("action"))
            mask = batch.get("mask")
        elif isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                obs, act = batch
                mask = None
            elif len(batch) == 3:
                obs, act, mask = batch
            else:
                msg = "Tuple batches must be (obs, act) or (obs, act, mask)."
                raise ValueError(msg)
        else:
            msg = "batch must be a mapping or tuple/list of tensors."
            raise TypeError(msg)

        if obs is None or act is None:
            msg = "batch must provide observation and action tensors."
            raise ValueError(msg)
        obs = obs.to(self.device)
        act = act.to(self.device)
        if mask is not None:
            mask = mask.to(self.device)
        return obs, act, mask

    def init_code_activity_accumulator(self) -> dict[str, torch.Tensor]:
        """Create mutable per-chart code-count histograms for obs and act."""
        return {
            "obs": torch.zeros(
                self.agent.config.obs_encoder.num_charts,
                self.agent.config.obs_encoder.codes_per_chart,
                dtype=torch.long,
            ),
            "act": torch.zeros(
                self.agent.config.act_encoder.num_charts,
                self.agent.config.act_encoder.codes_per_chart,
                dtype=torch.long,
            ),
        }

    def update_code_activity_accumulator(
        self,
        accumulator: dict[str, torch.Tensor] | None,
        forward: dict[str, Any],
    ) -> None:
        """Fold one batch of hard chart/code assignments into histogram counts."""
        if accumulator is None:
            return

        obs_chart = forward["obs"]["chart_idx_valid"].reshape(-1).detach().cpu()
        obs_code = forward["obs"]["code_idx_valid"].reshape(-1).detach().cpu()
        act_chart = forward["act"]["chart_idx_valid"].reshape(-1).detach().cpu()
        act_code = forward["act"]["code_idx_valid"].reshape(-1).detach().cpu()

        for chart, code in zip(obs_chart.tolist(), obs_code.tolist(), strict=False):
            accumulator["obs"][int(chart), int(code)] += 1
        for chart, code in zip(act_chart.tolist(), act_code.tolist(), strict=False):
            accumulator["act"][int(chart), int(code)] += 1

    def finalize_code_activity(
        self,
        accumulator: dict[str, torch.Tensor] | None,
    ) -> dict[str, list[list[int]]]:
        """Convert a mutable activity accumulator into per-chart histograms."""
        if accumulator is None:
            return {"obs": [], "act": []}
        return {
            "obs": accumulator["obs"].tolist(),
            "act": accumulator["act"].tolist(),
        }

    def _phase1_valid_positions(self, encoded: dict[str, torch.Tensor]) -> torch.Tensor | None:
        """Select which valid frames contribute frame-local Phase-1 losses."""
        mode = self.config.phase1_frame_mode
        if mode == "all":
            return None
        if mode == "anchor":
            return encoded["anchor_valid_pos"]
        msg = "phase1_frame_mode must be one of {'all', 'anchor'}."
        raise ValueError(msg)

    def _select_valid_rows(
        self,
        value: torch.Tensor | None,
        valid_positions: torch.Tensor | None,
    ) -> torch.Tensor | None:
        """Select a subset of valid-frame rows when anchor-mode supervision is active."""
        if value is None or valid_positions is None:
            return value
        return value[valid_positions]

    def _compute_phase1_stack(
        self,
        encoded: dict[str, torch.Tensor],
        model: TopoEncoder,
        jump_operator: FactorizedJumpOperator,
        config: VLAConfig,
        *,
        epoch: int | None,
        prefix: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the Phase-1 encoder losses for one manifold."""
        valid_positions = self._phase1_valid_positions(encoded)
        x_valid = self._select_valid_rows(encoded["x_valid"], valid_positions)
        x_recon_valid = self._select_valid_rows(encoded["x_recon_valid"], valid_positions)
        enc_router_weights_valid = self._select_valid_rows(
            encoded["enc_router_weights_valid"],
            valid_positions,
        )
        dec_router_weights_valid = self._select_valid_rows(
            encoded["dec_router_weights_valid"],
            valid_positions,
        )
        z_geo_valid = self._select_valid_rows(encoded["z_geo_valid"], valid_positions)
        soft_router_weights_valid = self._select_valid_rows(
            encoded["soft_router_weights_valid"],
            valid_positions,
        )
        c_bar_valid = self._select_valid_rows(encoded["c_bar_valid"], valid_positions)
        v_local_valid = self._select_valid_rows(encoded["v_local_valid"], valid_positions)
        usage_router_weights_valid = self._select_valid_rows(
            encoded["usage_router_weights_valid"],
            valid_positions,
        )
        indices_stack_valid = self._select_valid_rows(
            encoded["indices_stack_valid"], valid_positions
        )
        router_scores_valid = self._select_valid_rows(
            encoded["router_scores_valid"], valid_positions
        )
        z_n_valid = self._select_valid_rows(encoded["z_n_valid"], valid_positions)
        z_tex_valid = self._select_valid_rows(encoded["z_tex_valid"], valid_positions)
        z_n_all_valid = self._select_valid_rows(encoded["z_n_all_valid"], valid_positions)

        selection_scale = 1.0
        if valid_positions is not None and encoded["x_valid"].shape[0] > 0:
            selection_scale = float(valid_positions.numel()) / float(encoded["x_valid"].shape[0])

        base_loss, zn_reg_loss, metrics = compute_phase1_loss(
            x_valid,
            x_recon_valid,
            encoded["vq_loss"] * selection_scale,
            enc_router_weights_valid,
            dec_router_weights_valid,
            z_geo_valid,
            model,
            config,
            router_reg_weights=soft_router_weights_valid,
            c_bar=c_bar_valid,
            v_local=v_local_valid,
            usage_router_weights=usage_router_weights_valid,
            indices_stack=indices_stack_valid,
            router_scores=router_scores_valid,
        )

        current_jump_weight = get_jump_weight_schedule(
            0 if epoch is None else epoch,
            warmup_end=config.w_jump_warmup,
            ramp_end=config.w_jump_ramp_end,
            final_weight=config.w_jump,
        )
        if current_jump_weight > 0:
            jump_loss, _ = compute_jump_consistency_loss(
                z_n_all_valid,
                enc_router_weights_valid,
                jump_operator,
            )
            zn_reg_loss = zn_reg_loss + current_jump_weight * jump_loss
        else:
            jump_loss = torch.zeros((), device=self.device)

        ortho_loss = orthogonality_loss(z_n_valid, z_tex_valid)
        base_loss = base_loss + getattr(config, "w_perp", 0.01) * ortho_loss
        total = base_loss + zn_reg_loss

        metrics.update({
            "jump": float(jump_loss.detach()),
            "ortho": float(ortho_loss.detach()),
            "jump_weight": float(current_jump_weight),
            "base_total": float(base_loss.detach()),
            "zn_reg_total": float(zn_reg_loss.detach()),
            "total": float(total.detach()),
        })
        return total, _prefixed_metrics(prefix, metrics)

    def compute_batch_losses(
        self,
        batch: Any,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        training: bool,
    ) -> dict[str, Any]:
        """Run the forward pass and assemble all training losses/metrics."""
        obs, act, mask = self._move_batch_to_device(batch)
        routing_tau = self.routing_tau_for_step(global_step, training=training)
        alpha = self.enclosure_alpha_for_step(global_step)
        self.agent.enclosure_probe.obs_grl.alpha.fill_(alpha)
        self.agent.enclosure_probe.act_grl.alpha.fill_(alpha)
        compute_enclosure = (
            self.config.weight_enclosure_encoder != 0.0
            or self.config.weight_enclosure_probe != 0.0
        )
        compute_markov_transition = self.config.weight_markov_transition != 0.0
        compute_markov_shape = self.config.weight_markov_shape != 0.0
        compute_macro = compute_markov_transition or compute_markov_shape

        forward = self.agent.forward_batch(
            obs,
            act,
            mask=mask,
            routing_tau=routing_tau,
            macro_chart_tau=self.config.macro_chart_tau,
            macro_code_tau=self.config.macro_code_tau,
            compute_macro=compute_macro,
        )

        obs_loss, obs_metrics = self._compute_phase1_stack(
            forward["obs"],
            self.agent.obs_encoder,
            self.agent.obs_jump_operator,
            self.agent.config.obs_encoder,
            epoch=epoch,
            prefix="obs",
        )
        act_loss, act_metrics = self._compute_phase1_stack(
            forward["act"],
            self.agent.act_encoder,
            self.agent.act_jump_operator,
            self.agent.config.act_encoder,
            epoch=epoch,
            prefix="act",
        )

        transitions = forward["transitions"]
        zero = torch.zeros((), device=self.device)
        enclosure_encoder_loss = zero
        enclosure_probe_loss = zero
        markov_transition_loss = zero
        markov_shape_loss = zero

        enclosure_metrics: dict[str, float] = {}
        markov_metrics: dict[str, float] = {}
        if transitions["num_valid"] > 0:
            if compute_enclosure:
                enclosure_encoder_loss, enclosure_probe_loss, enclosure_diag = (
                    compute_absolute_enclosure_loss(
                        self.agent.enclosure_probe,
                        obs_chart_centers=self.agent.obs_encoder.encoder.chart_centers,
                        obs_codebook=self.agent.obs_encoder.encoder.codebook,
                        obs_chart_t=transitions["obs_chart_t_valid"],
                        obs_code_t=transitions["obs_code_t_valid"],
                        obs_z_n_t=transitions["obs_z_n_t_valid"],
                        obs_z_tex_t=transitions["obs_z_tex_t_valid"],
                        act_chart_centers=self.agent.act_encoder.encoder.chart_centers,
                        act_codebook=self.agent.act_encoder.encoder.codebook,
                        act_chart_t=transitions["act_chart_t_valid"],
                        act_code_t=transitions["act_code_t_valid"],
                        act_z_n_t=transitions["act_z_n_t_valid"],
                        act_z_tex_t=transitions["act_z_tex_t_valid"],
                        obs_chart_tp1=transitions["obs_chart_tp1_valid"],
                        obs_code_tp1=transitions["obs_code_tp1_valid"],
                        obs_codes_per_chart=self.agent.config.obs_encoder.codes_per_chart,
                    )
                )
                enclosure_metrics = _prefixed_metrics("enclosure", enclosure_diag)
            else:
                enclosure_metrics = {
                    "enclosure/acc_base": 0.0,
                    "enclosure/acc_obs": 0.0,
                    "enclosure/acc_act": 0.0,
                    "enclosure/acc_both": 0.0,
                    "enclosure/defect_acc_obs": 0.0,
                    "enclosure/defect_acc_act": 0.0,
                    "enclosure/defect_acc_both": 0.0,
                    "enclosure/ce_base": 0.0,
                    "enclosure/ce_obs": 0.0,
                    "enclosure/ce_act": 0.0,
                    "enclosure/ce_both": 0.0,
                    "enclosure/defect_ce_obs": 0.0,
                    "enclosure/defect_ce_act": 0.0,
                    "enclosure/defect_ce_both": 0.0,
                    "enclosure/loss_encoder": 0.0,
                    "enclosure/loss_probe": 0.0,
                }

            if compute_macro:
                obs_geometry = {
                    "chart_centers": forward["macro"]["obs"]["chart_centers"],
                    "codebook": forward["macro"]["obs"]["codebook"],
                    "state_tangent_points": forward["macro"]["obs"]["state_tangent_points"],
                    "state_points": forward["macro"]["obs"]["state_points"],
                }
                act_geometry = {
                    "chart_centers": forward["macro"]["act"]["chart_centers"],
                    "codebook": forward["macro"]["act"]["codebook"],
                    "state_tangent_points": forward["macro"]["act"]["state_tangent_points"],
                    "state_points": forward["macro"]["act"]["state_points"],
                }
                if compute_markov_transition:
                    markov_transition_loss, transition_metrics, pred = (
                        compute_markov_transition_loss(
                            self.agent.macro_model,
                            transitions["obs_state_probs_t_valid"],
                            transitions["act_state_probs_t_valid"],
                            obs_geometry=obs_geometry,
                            act_geometry=act_geometry,
                            target_next_state_probs=transitions[
                                "obs_state_probs_tp1_valid"
                            ].detach(),
                            target_next_chart_idx=transitions["obs_chart_tp1_valid"],
                            target_next_code_idx=transitions["obs_code_tp1_valid"],
                            codes_per_chart=self.agent.config.obs_encoder.codes_per_chart,
                            metric_prefix="markov",
                        )
                    )
                else:
                    pred = self.agent.macro_model(
                        transitions["obs_state_probs_t_valid"],
                        transitions["act_state_probs_t_valid"],
                        obs_geometry=obs_geometry,
                        act_geometry=act_geometry,
                    )
                    transition_metrics = {
                        "markov/L_transition": 0.0,
                        "markov/transition_ce": 0.0,
                        "markov/state_ce": 0.0,
                        "markov/chart_ce": 0.0,
                        "markov/code_ce": 0.0,
                        "markov/transition_acc": 0.0,
                        "markov/chart_acc": 0.0,
                        "markov/code_acc": 0.0,
                        "markov/next_state_entropy": 0.0,
                        "markov/next_chart_entropy": 0.0,
                        "markov/next_code_entropy": 0.0,
                        "markov/next_state_top1_prob": float(
                            pred["next_state_top1_prob"].mean().detach()
                        ),
                        "markov/next_chart_top1_prob": float(
                            pred["next_chart_top1_prob"].mean().detach()
                        ),
                        "markov/next_code_top1_prob": float(
                            pred["next_code_top1_prob"].mean().detach()
                        ),
                        "markov/target_state_entropy": 0.0,
                    }

                if compute_markov_shape:
                    markov_shape_loss, shape_metrics = compute_markov_shape_loss(
                        pred["next_state_probs"],
                        transitions["obs_state_probs_tp1_valid"],
                        metric_prefix="markov/shape",
                    )
                else:
                    shape_metrics = {
                        "markov/shape/L_align": 0.0,
                        "markov/shape/align_ce": 0.0,
                        "markov/shape/align_kl": 0.0,
                        "markov/shape/agreement": 0.0,
                        "markov/shape/teacher_entropy": 0.0,
                        "markov/shape/student_entropy": 0.0,
                    }
                markov_metrics = transition_metrics | shape_metrics
            else:
                markov_metrics = {
                    "markov/L_transition": 0.0,
                    "markov/transition_ce": 0.0,
                    "markov/state_ce": 0.0,
                    "markov/chart_ce": 0.0,
                    "markov/code_ce": 0.0,
                    "markov/transition_acc": 0.0,
                    "markov/chart_acc": 0.0,
                    "markov/code_acc": 0.0,
                    "markov/next_state_entropy": 0.0,
                    "markov/next_state_top1_prob": 0.0,
                    "markov/next_chart_entropy": 0.0,
                    "markov/next_code_entropy": 0.0,
                    "markov/next_chart_top1_prob": 0.0,
                    "markov/next_code_top1_prob": 0.0,
                    "markov/target_state_entropy": 0.0,
                    "markov/shape/L_align": 0.0,
                    "markov/shape/align_ce": 0.0,
                    "markov/shape/align_kl": 0.0,
                    "markov/shape/agreement": 0.0,
                    "markov/shape/teacher_entropy": 0.0,
                    "markov/shape/student_entropy": 0.0,
                }
        else:
            enclosure_metrics = {
                "enclosure/acc_base": 0.0,
                "enclosure/acc_obs": 0.0,
                "enclosure/acc_act": 0.0,
                "enclosure/acc_both": 0.0,
                "enclosure/defect_acc_obs": 0.0,
                "enclosure/defect_acc_act": 0.0,
                "enclosure/defect_acc_both": 0.0,
                "enclosure/ce_base": 0.0,
                "enclosure/ce_obs": 0.0,
                "enclosure/ce_act": 0.0,
                "enclosure/ce_both": 0.0,
                "enclosure/defect_ce_obs": 0.0,
                "enclosure/defect_ce_act": 0.0,
                "enclosure/defect_ce_both": 0.0,
                "enclosure/loss_encoder": 0.0,
                "enclosure/loss_probe": 0.0,
            }
            markov_metrics = {
                "markov/L_transition": 0.0,
                "markov/transition_ce": 0.0,
                "markov/state_ce": 0.0,
                "markov/chart_ce": 0.0,
                "markov/code_ce": 0.0,
                "markov/transition_acc": 0.0,
                "markov/chart_acc": 0.0,
                "markov/code_acc": 0.0,
                "markov/next_state_entropy": 0.0,
                "markov/next_state_top1_prob": 0.0,
                "markov/next_chart_entropy": 0.0,
                "markov/next_code_entropy": 0.0,
                "markov/next_chart_top1_prob": 0.0,
                "markov/next_code_top1_prob": 0.0,
                "markov/target_state_entropy": 0.0,
                "markov/shape/L_align": 0.0,
                "markov/shape/align_ce": 0.0,
                "markov/shape/align_kl": 0.0,
                "markov/shape/agreement": 0.0,
                "markov/shape/teacher_entropy": 0.0,
                "markov/shape/student_entropy": 0.0,
            }

        main_loss = (
            self.config.weight_obs_phase1 * obs_loss
            + self.config.weight_act_phase1 * act_loss
            + self.config.weight_enclosure_encoder * enclosure_encoder_loss
            + self.config.weight_markov_transition * markov_transition_loss
            + self.config.weight_markov_shape * markov_shape_loss
        )
        probe_loss = self.config.weight_enclosure_probe * enclosure_probe_loss

        metrics = {
            "loss/main": float(main_loss.detach()),
            "loss/probe": float(probe_loss.detach()),
            "loss/obs": float(obs_loss.detach()),
            "loss/act": float(act_loss.detach()),
            "loss/enclosure_encoder": float(enclosure_encoder_loss.detach()),
            "loss/enclosure_probe": float(enclosure_probe_loss.detach()),
            "loss/markov_transition": float(markov_transition_loss.detach()),
            "loss/markov_shape": float(markov_shape_loss.detach()),
            "routing/tau": float(routing_tau),
            "enclosure/alpha": float(alpha),
            "transitions/num_valid": float(transitions["num_valid"]),
        }
        metrics.update(obs_metrics)
        metrics.update(act_metrics)
        metrics.update(enclosure_metrics)
        metrics.update(markov_metrics)

        return {
            "forward": forward,
            "main_loss": main_loss,
            "probe_loss": probe_loss,
            "metrics": metrics,
        }

    def train_step(
        self,
        batch: Any,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        code_activity_accumulator: dict[str, list[set[int]]] | None = None,
    ) -> dict[str, float]:
        """Run one optimization step."""
        self.agent.train()
        step_index = self.global_step if global_step is None else global_step
        outputs = self.compute_batch_losses(
            batch,
            epoch=epoch,
            global_step=step_index,
            training=True,
        )
        self.update_code_activity_accumulator(code_activity_accumulator, outputs["forward"])

        self.probe_optimizer.zero_grad()
        self.encoder_optimizer.zero_grad()
        self.markov_optimizer.zero_grad()
        outputs["main_loss"].backward()

        encoder_params = [
            param
            for group in self.encoder_optimizer.param_groups
            for param in group["params"]
            if param.requires_grad
        ]
        markov_params = [
            param for param in self.agent.macro_model.parameters() if param.requires_grad
        ]

        encoder_grad_norm = compute_grad_norm(encoder_params)
        encoder_param_norm = compute_param_norm(encoder_params)
        markov_grad_norm = compute_grad_norm(markov_params)
        markov_param_norm = compute_param_norm(markov_params)

        if self.config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(encoder_params, self.config.grad_clip)
            torch.nn.utils.clip_grad_norm_(markov_params, self.config.grad_clip)

        self.encoder_optimizer.step()
        self.markov_optimizer.step()

        probe_metrics = {
            "grad/probe_norm": 0.0,
            "param/probe_norm": compute_param_norm(
                [
                    param
                    for param in self.agent.enclosure_probe.parameters()
                    if param.requires_grad
                ],
            ),
        }
        if outputs["probe_loss"].requires_grad and outputs["probe_loss"].detach().item() > 0:
            self.probe_optimizer.zero_grad()
            outputs["probe_loss"].backward()
            probe_params = [
                param for param in self.agent.enclosure_probe.parameters() if param.requires_grad
            ]
            probe_metrics["grad/probe_norm"] = compute_grad_norm(probe_params)
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(probe_params, self.config.grad_clip)
            self.probe_optimizer.step()

        metrics = dict(outputs["metrics"])
        metrics.update({
            "grad/encoder_norm": float(encoder_grad_norm),
            "param/encoder_norm": float(encoder_param_norm),
            "grad/markov_norm": float(markov_grad_norm),
            "param/markov_norm": float(markov_param_norm),
        })
        metrics.update(probe_metrics)

        self.global_step = step_index + 1
        return metrics

    def eval_step(
        self,
        batch: Any,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
        code_activity_accumulator: dict[str, list[set[int]]] | None = None,
    ) -> dict[str, float]:
        """Evaluate one batch without parameter updates."""
        self.agent.eval()
        step_index = self.global_step if global_step is None else global_step
        with torch.no_grad():
            outputs = self.compute_batch_losses(
                batch,
                epoch=epoch,
                global_step=step_index,
                training=False,
            )
        self.update_code_activity_accumulator(code_activity_accumulator, outputs["forward"])
        return outputs["metrics"]

    def fit_epoch(
        self,
        loader: Any,
        *,
        epoch: int | None = None,
        start_global_step: int | None = None,
        step_scheduler: bool = True,
    ) -> dict[str, float]:
        """Convenience wrapper for one full training epoch."""
        metrics = []
        for batch_idx, batch in enumerate(loader):
            global_step = None if start_global_step is None else start_global_step + batch_idx
            metrics.append(self.train_step(batch, epoch=epoch, global_step=global_step))
        if self.encoder_scheduler is not None and step_scheduler:
            self.encoder_scheduler.step()
        return _mean_metrics(metrics)

    def evaluate_epoch(
        self,
        loader: Any,
        *,
        epoch: int | None = None,
        global_step: int | None = None,
    ) -> dict[str, float]:
        """Convenience wrapper for one full evaluation epoch."""
        metrics = [self.eval_step(batch, epoch=epoch, global_step=global_step) for batch in loader]
        return _mean_metrics(metrics)


__all__ = [
    "FragileAgent",
    "FragileAgentConfig",
    "FragileAgentTrainer",
    "FragileAgentTrainerConfig",
]
