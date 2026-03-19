"""Differentiable geometry-aware coarse Markov model on atlas macro symbols.

This module keeps the atlas symbols as the actual macro state:

- observations and actions are softly mapped to chart/code symbol
  distributions in their own Poincare balls,
- each flattened symbolic state has an absolute point ``c_k ⊕ q_{k,c}``,
- the coarse transition model reads the observation/action symbol geometry,
  predicts the next observation chart first, then the next code within that
  chart,
- helper losses fit the factorized model, shape the symbolic atlas, and can
  later supervise the micro world model.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import (
    BeliefGeometryEncoder,
    ChartTransitionRouter,
    ConditionalCodeRouter,
    NextStateQueryPredictor,
)
from fragile.layers.gauge import (
    hyperbolic_distance,
    log_map_zero,
    mobius_add,
    poincare_weighted_mean,
    project_to_ball,
)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one."""
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _state_index(
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
    codes_per_chart: int,
) -> torch.Tensor:
    """Flatten ``(chart, code)`` symbolic indices into one state id."""
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _normalize_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a non-negative tensor along the last axis."""
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)


def _validate_macro_geometry(chart_centers: torch.Tensor, codebook: torch.Tensor) -> None:
    """Validate chart/code tensors before composing symbolic coordinates."""
    if chart_centers.dim() != 2:
        msg = "chart_centers must have shape [N_c, D]."
        raise ValueError(msg)
    if codebook.dim() != 3:
        msg = "codebook must have shape [N_c, K, D]."
        raise ValueError(msg)
    if codebook.shape[0] != chart_centers.shape[0]:
        msg = "chart_centers and codebook must agree on the number of charts."
        raise ValueError(msg)
    if codebook.shape[-1] != chart_centers.shape[-1]:
        msg = "chart_centers and codebook must agree on latent dimension."
        raise ValueError(msg)


def _reshape_leading_dims(x: torch.Tensor, leading_shape: torch.Size) -> torch.Tensor:
    """Restore leading batch dimensions after a temporary flatten."""
    if len(leading_shape) == 0:
        if x.dim() <= 1:
            return x.reshape(()) if x.numel() == 1 else x
        return x.reshape(*x.shape[1:])
    return x.reshape(*leading_shape, *x.shape[1:])


def _state_chart_code_view(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> torch.Tensor:
    """View a flattened state distribution as ``[..., chart, code]``."""
    if state_probs.shape[-1] != num_charts * codes_per_chart:
        msg = "state_probs does not match the requested chart/code factorization."
        raise ValueError(msg)
    return state_probs.reshape(*state_probs.shape[:-1], num_charts, codes_per_chart)


def _state_probs_to_chart_probs(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> torch.Tensor:
    """Marginalize flattened state probabilities down to chart probabilities."""
    return _state_chart_code_view(state_probs, num_charts, codes_per_chart).sum(dim=-1)


def _state_probs_to_code_conditionals(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert flattened state probabilities into per-chart code conditionals."""
    view = _state_chart_code_view(state_probs, num_charts, codes_per_chart)
    chart_probs = view.sum(dim=-1, keepdim=True)
    code_probs = view / chart_probs.clamp(min=eps)
    uniform = code_probs.new_full(code_probs.shape, 1.0 / float(codes_per_chart))
    return torch.where(chart_probs <= eps, uniform, code_probs)


def _flatten_chart_code_probs(
    chart_probs: torch.Tensor,
    code_probs: torch.Tensor,
) -> torch.Tensor:
    """Flatten factorized chart/code probabilities back into one state axis."""
    if chart_probs.shape != code_probs.shape[:-1]:
        msg = "chart_probs and code_probs must agree on leading chart dimensions."
        raise ValueError(msg)
    return (chart_probs.unsqueeze(-1) * code_probs).reshape(*chart_probs.shape[:-1], -1)


def compose_absolute_macro_dictionary(
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compose the absolute point for every flattened symbolic state.

    The resulting dictionary lives in one manifold only. Use the observation
    chart/code tensors to build the observation symbol dictionary and the action
    tensors to build the action symbol dictionary.
    """
    _validate_macro_geometry(chart_centers, codebook)

    chart_centers_proj = project_to_ball(chart_centers)
    codebook_proj = project_to_ball(codebook).to(
        device=chart_centers_proj.device,
        dtype=chart_centers_proj.dtype,
    )

    state_points = project_to_ball(mobius_add(chart_centers_proj[:, None, :], codebook_proj))
    num_charts, codes_per_chart, latent_dim = codebook_proj.shape
    device = codebook_proj.device

    chart_idx = (
        torch.arange(num_charts, device=device)
        .unsqueeze(1)
        .expand(num_charts, codes_per_chart)
        .reshape(-1)
    )
    code_idx = (
        torch.arange(codes_per_chart, device=device)
        .unsqueeze(0)
        .expand(num_charts, codes_per_chart)
        .reshape(-1)
    )

    flat_state_points = state_points.reshape(num_charts * codes_per_chart, latent_dim)
    return {
        "state_points": flat_state_points,
        "state_tangent_points": log_map_zero(flat_state_points),
        "chart_idx": chart_idx,
        "code_idx": code_idx,
    }


def expected_macro_state(
    state_probs: torch.Tensor,
    state_points: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Map a soft symbolic state to one barycentric point in the same manifold."""
    if state_points.dim() != 2:
        msg = "state_points must have shape [S, D]."
        raise ValueError(msg)
    if state_probs.shape[-1] != state_points.shape[0]:
        msg = "state_probs and state_points must agree on the number of symbols."
        raise ValueError(msg)

    leading_shape = state_probs.shape[:-1]
    flat_probs = _normalize_probs(state_probs.reshape(-1, state_probs.shape[-1]), eps=eps)
    flat_points = project_to_ball(state_points).to(device=flat_probs.device, dtype=flat_probs.dtype)
    macro_mean = poincare_weighted_mean(flat_points, flat_probs, eps=eps)
    return _reshape_leading_dims(macro_mean, leading_shape)


def soft_macro_state_distribution(
    z_latent: torch.Tensor,
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
    *,
    chart_tau: float = 1.0,
    code_tau: float = 1.0,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Attach a differentiable coarse symbolic state to a manifold point."""
    _validate_macro_geometry(chart_centers, codebook)
    if z_latent.dim() < 2:
        msg = "z_latent must have shape [..., D]."
        raise ValueError(msg)
    if z_latent.shape[-1] != chart_centers.shape[-1]:
        msg = "z_latent and chart_centers must agree on latent dimension."
        raise ValueError(msg)

    leading_shape = z_latent.shape[:-1]
    latent_dim = z_latent.shape[-1]
    flat_z = project_to_ball(z_latent).reshape(-1, latent_dim)
    chart_centers_proj = project_to_ball(chart_centers).to(device=flat_z.device, dtype=flat_z.dtype)
    codebook_proj = project_to_ball(codebook).to(device=flat_z.device, dtype=flat_z.dtype)
    chart_tau = max(float(chart_tau), eps)
    code_tau = max(float(code_tau), eps)

    chart_dist = hyperbolic_distance(flat_z.unsqueeze(1), chart_centers_proj.unsqueeze(0))
    chart_logits = -chart_dist / chart_tau
    chart_log_probs = F.log_softmax(chart_logits, dim=-1)
    chart_probs = chart_log_probs.exp()

    c_bar = poincare_weighted_mean(chart_centers_proj, chart_probs, eps=eps)
    v_local = project_to_ball(mobius_add(-c_bar, flat_z))

    code_dist = hyperbolic_distance(v_local[:, None, None, :], codebook_proj[None, :, :, :])
    code_logits = -code_dist / code_tau
    code_log_probs = F.log_softmax(code_logits, dim=-1)
    code_probs = code_log_probs.exp()

    state_log_probs = (chart_log_probs.unsqueeze(-1) + code_log_probs).reshape(flat_z.shape[0], -1)
    state_probs = state_log_probs.exp()

    codes_per_chart = codebook_proj.shape[1]
    state_idx = state_probs.argmax(dim=-1)
    chart_idx = torch.div(state_idx, codes_per_chart, rounding_mode="floor")
    code_idx = state_idx.remainder(codes_per_chart)

    symbol_dict = compose_absolute_macro_dictionary(chart_centers_proj, codebook_proj)
    hard_state_point = symbol_dict["state_points"][state_idx]
    macro_state_mean = expected_macro_state(state_probs, symbol_dict["state_points"], eps=eps)
    state_entropy = -(state_probs * state_log_probs).sum(dim=-1)
    chart_entropy = -(chart_probs * chart_log_probs).sum(dim=-1)

    return {
        "z_latent": _reshape_leading_dims(flat_z, leading_shape),
        "router_weights": _reshape_leading_dims(chart_probs, leading_shape),
        "chart_logits": _reshape_leading_dims(chart_logits, leading_shape),
        "chart_probs": _reshape_leading_dims(chart_probs, leading_shape),
        "chart_idx": _reshape_leading_dims(chart_idx, leading_shape),
        "c_bar": _reshape_leading_dims(c_bar, leading_shape),
        "v_local": _reshape_leading_dims(v_local, leading_shape),
        "code_logits": _reshape_leading_dims(code_logits, leading_shape),
        "code_probs": _reshape_leading_dims(code_probs, leading_shape),
        "code_idx": _reshape_leading_dims(code_idx, leading_shape),
        "state_log_probs": _reshape_leading_dims(state_log_probs, leading_shape),
        "state_probs": _reshape_leading_dims(state_probs, leading_shape),
        "state_idx": _reshape_leading_dims(state_idx, leading_shape),
        "state_entropy": _reshape_leading_dims(state_entropy, leading_shape),
        "state_value_entropy": _reshape_leading_dims(state_entropy, leading_shape),
        "chart_entropy": _reshape_leading_dims(chart_entropy, leading_shape),
        "macro_state_mean": _reshape_leading_dims(macro_state_mean, leading_shape),
        "hard_state_point": _reshape_leading_dims(hard_state_point, leading_shape),
        "chart_centers": chart_centers_proj,
        "chart_tangent_points": log_map_zero(chart_centers_proj),
        "codebook": codebook_proj,
        "code_tangent_points": log_map_zero(codebook_proj),
        **symbol_dict,
    }


class MacroTransitionModel(nn.Module):
    """Geometry-aware stochastic symbolic dynamics model ``p(s_{t+1} | s_t, a_t)``.

    The model keeps the atlas symbols as the state space, but parameterizes the
    transition via the observation/action symbol geometry:

    1. summarize the current observation and action beliefs using their symbol
       tangent coordinates,
    2. fuse both summaries into a next-observation query point,
    3. score the next chart against the observation chart centers,
    4. score the next code inside each chart against that chart's local codebook.
    """

    def __init__(
        self,
        obs_latent_dim: int,
        act_latent_dim: int,
        num_obs_charts: int,
        obs_codes_per_chart: int,
        num_act_charts: int,
        act_codes_per_chart: int,
        *,
        hidden_dim: int = 128,
        feature_scale: float = 0.1,
        use_residual_transition: bool = True,
        residual_scale: float = 1.0,
        learn_reward: bool = True,
        learn_continuation: bool = True,
        initial_continuation: float = 0.99,
    ) -> None:
        super().__init__()
        self.obs_latent_dim = int(obs_latent_dim)
        self.act_latent_dim = int(act_latent_dim)
        self.num_obs_charts = int(num_obs_charts)
        self.obs_codes_per_chart = int(obs_codes_per_chart)
        self.num_act_charts = int(num_act_charts)
        self.act_codes_per_chart = int(act_codes_per_chart)
        self.num_states = self.num_obs_charts * self.obs_codes_per_chart
        self.num_actions = self.num_act_charts * self.act_codes_per_chart
        if self.num_states <= 0 or self.num_actions <= 0:
            msg = "num_states and num_actions must both be positive."
            raise ValueError(msg)

        self.hidden_dim = int(hidden_dim)
        self.obs_encoder = BeliefGeometryEncoder(self.obs_latent_dim, self.hidden_dim)
        self.act_encoder = BeliefGeometryEncoder(self.act_latent_dim, self.hidden_dim)
        self.query_predictor = NextStateQueryPredictor(self.hidden_dim, self.obs_latent_dim)
        self.chart_router = ChartTransitionRouter(
            self.obs_latent_dim,
            self.hidden_dim,
            feature_scale=feature_scale,
        )
        self.code_router = ConditionalCodeRouter(
            self.obs_latent_dim,
            self.hidden_dim,
            feature_scale=feature_scale,
        )

        self.residual_scale = float(residual_scale)
        if use_residual_transition:
            self.residual_transition_logits = nn.Parameter(
                torch.zeros(self.num_states, self.num_actions, self.num_states)
            )
        else:
            self.register_parameter("residual_transition_logits", None)

        if learn_reward:
            self.reward_table = nn.Parameter(torch.zeros(self.num_states, self.num_actions))
        else:
            self.register_parameter("reward_table", None)

        if learn_continuation:
            init_cont = torch.full(
                (self.num_states, self.num_actions),
                float(initial_continuation),
            ).clamp(min=1e-4, max=1.0 - 1e-4)
            self.continuation_logits = nn.Parameter(torch.logit(init_cont))
        else:
            self.register_parameter("continuation_logits", None)

    def _validate_inputs(self, state_probs: torch.Tensor, action_probs: torch.Tensor) -> None:
        if state_probs.dim() < 2 or action_probs.dim() < 2:
            msg = "state_probs and action_probs must have shape [..., num_symbols]."
            raise ValueError(msg)
        if state_probs.shape[:-1] != action_probs.shape[:-1]:
            msg = "state_probs and action_probs must share the same leading shape."
            raise ValueError(msg)
        if state_probs.shape[-1] != self.num_states:
            msg = "state_probs has the wrong number of states."
            raise ValueError(msg)
        if action_probs.shape[-1] != self.num_actions:
            msg = "action_probs has the wrong number of actions."
            raise ValueError(msg)

    def _validate_geometry(
        self,
        obs_geometry: dict[str, torch.Tensor],
        act_geometry: dict[str, torch.Tensor],
    ) -> None:
        required_obs = {"chart_centers", "codebook", "state_tangent_points"}
        required_act = {"state_tangent_points"}
        missing_obs = required_obs.difference(obs_geometry)
        missing_act = required_act.difference(act_geometry)
        if missing_obs:
            msg = f"obs_geometry is missing keys: {sorted(missing_obs)}."
            raise ValueError(msg)
        if missing_act:
            msg = f"act_geometry is missing keys: {sorted(missing_act)}."
            raise ValueError(msg)

    def continuation_table(self) -> torch.Tensor | None:
        """Return the coarse continuation probability for each state-action pair."""
        if self.continuation_logits is None:
            return None
        return torch.sigmoid(self.continuation_logits)

    def reward_from_probs(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the expected coarse reward under soft state/action distributions."""
        if self.reward_table is None:
            return state_probs.new_zeros(state_probs.shape[:-1])
        state_probs = _normalize_probs(state_probs)
        action_probs = _normalize_probs(action_probs)
        return torch.einsum("...s,sa,...a->...", state_probs, self.reward_table, action_probs)

    def continuation_from_probs(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the expected continuation probability under soft state/action distributions."""
        table = self.continuation_table()
        if table is None:
            return state_probs.new_ones(state_probs.shape[:-1])
        state_probs = _normalize_probs(state_probs)
        action_probs = _normalize_probs(action_probs)
        return torch.einsum("...s,sa,...a->...", state_probs, table, action_probs)

    def forward(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
        *,
        obs_geometry: dict[str, torch.Tensor],
        act_geometry: dict[str, torch.Tensor],
        eps: float = 1e-8,
    ) -> dict[str, torch.Tensor]:
        """Roll one coarse Markov step from soft symbolic state/action inputs."""
        self._validate_inputs(state_probs, action_probs)
        self._validate_geometry(obs_geometry, act_geometry)

        leading_shape = state_probs.shape[:-1]
        state_probs = _normalize_probs(state_probs, eps=eps)
        action_probs = _normalize_probs(action_probs, eps=eps)

        obs_state_tangent = obs_geometry["state_tangent_points"]
        act_state_tangent = act_geometry["state_tangent_points"]
        obs_chart_centers = obs_geometry["chart_centers"]
        obs_codebook = obs_geometry["codebook"]

        obs_summary = self.obs_encoder(state_probs, obs_state_tangent, eps=eps)
        act_summary = self.act_encoder(action_probs, act_state_tangent, eps=eps)
        query = self.query_predictor(obs_summary["summary"], act_summary["summary"])
        chart_out = self.chart_router(query["query_point"], query["context"], obs_chart_centers)
        code_out = self.code_router(
            query["query_point"],
            query["context"],
            obs_chart_centers,
            obs_codebook,
        )

        base_state_log_probs = (
            chart_out["chart_log_probs"].unsqueeze(-1) + code_out["code_log_probs"]
        ).reshape(*leading_shape, self.num_states)
        if self.residual_transition_logits is not None:
            residual_logits = torch.einsum(
                "...s,...a,san->...n",
                state_probs,
                action_probs,
                self.residual_transition_logits,
            )
            final_logits = base_state_log_probs + self.residual_scale * residual_logits
            next_state_log_probs = F.log_softmax(final_logits, dim=-1)
        else:
            residual_logits = None
            next_state_log_probs = base_state_log_probs
        next_state_probs = next_state_log_probs.exp()

        next_chart_probs = _state_probs_to_chart_probs(
            next_state_probs,
            self.num_obs_charts,
            self.obs_codes_per_chart,
        )
        next_chart_log_probs = next_chart_probs.clamp(min=eps).log()
        next_code_probs = _state_probs_to_code_conditionals(
            next_state_probs,
            self.num_obs_charts,
            self.obs_codes_per_chart,
            eps=eps,
        )
        next_code_log_probs = next_code_probs.clamp(min=eps).log()

        next_state_entropy = -(next_state_probs * next_state_log_probs).sum(dim=-1)
        next_chart_entropy = -(next_chart_probs * next_chart_log_probs).sum(dim=-1)
        code_entropy_per_chart = -(next_code_probs * next_code_log_probs).sum(dim=-1)
        next_code_entropy = (next_chart_probs * code_entropy_per_chart).sum(dim=-1)

        out = {
            "next_state_probs": next_state_probs,
            "next_state_log_probs": next_state_log_probs,
            "next_state_entropy": next_state_entropy,
            "next_state_top1_prob": next_state_probs.max(dim=-1).values,
            "next_chart_probs": next_chart_probs,
            "next_chart_log_probs": next_chart_log_probs,
            "next_chart_entropy": next_chart_entropy,
            "next_chart_top1_prob": next_chart_probs.max(dim=-1).values,
            "next_code_probs": next_code_probs,
            "next_code_log_probs": next_code_log_probs,
            "next_code_entropy": next_code_entropy,
            "next_code_top1_prob": (
                next_chart_probs * next_code_probs.max(dim=-1).values
            ).sum(dim=-1),
            "base_state_log_probs": base_state_log_probs,
            "base_state_probs": base_state_log_probs.exp(),
            "next_query_tangent": query["query_tangent"],
            "next_query_point": query["query_point"],
            "joint_context": query["context"],
            "next_chart_logits_base": chart_out["chart_logits"],
            "next_code_logits_base": code_out["code_logits"],
            "next_local_query": code_out["local_query"],
        }
        if residual_logits is not None:
            out["residual_transition_logits"] = residual_logits
        if self.reward_table is not None:
            out["reward"] = self.reward_from_probs(state_probs, action_probs)
        if self.continuation_logits is not None:
            out["continuation"] = self.continuation_from_probs(state_probs, action_probs)
        return out

    def conditional_from_indices(
        self,
        state_idx: torch.Tensor,
        action_idx: torch.Tensor,
        *,
        obs_geometry: dict[str, torch.Tensor],
        act_geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Read one conditional transition row from hard indices via one-hot inputs."""
        if state_idx.shape != action_idx.shape:
            msg = "state_idx and action_idx must have matching shapes."
            raise ValueError(msg)
        state_probs = F.one_hot(state_idx.long(), self.num_states).to(dtype=torch.float32)
        action_probs = F.one_hot(action_idx.long(), self.num_actions).to(dtype=torch.float32)
        return self(
            state_probs,
            action_probs,
            obs_geometry=obs_geometry,
            act_geometry=act_geometry,
        )

    def rollout(
        self,
        state_probs_0: torch.Tensor,
        action_probs_seq: torch.Tensor,
        *,
        obs_geometry: dict[str, torch.Tensor],
        act_geometry: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Roll the coarse model forward for a sequence of soft symbolic actions."""
        if state_probs_0.dim() < 2:
            msg = "state_probs_0 must have shape [..., S]."
            raise ValueError(msg)
        if action_probs_seq.dim() < 3:
            msg = "action_probs_seq must have shape [..., H, A]."
            raise ValueError(msg)
        if state_probs_0.shape[:-1] != action_probs_seq.shape[:-2]:
            msg = "state_probs_0 and action_probs_seq must share the same batch shape."
            raise ValueError(msg)

        current_state = _normalize_probs(state_probs_0)
        state_traj = [current_state]
        next_state_traj: list[torch.Tensor] = []
        entropy_traj: list[torch.Tensor] = []
        top1_traj: list[torch.Tensor] = []
        reward_traj: list[torch.Tensor] = []
        continuation_traj: list[torch.Tensor] = []

        horizon = action_probs_seq.shape[-2]
        for t in range(horizon):
            step_out = self(
                current_state,
                action_probs_seq[..., t, :],
                obs_geometry=obs_geometry,
                act_geometry=act_geometry,
            )
            current_state = step_out["next_state_probs"]
            state_traj.append(current_state)
            next_state_traj.append(step_out["next_state_probs"])
            entropy_traj.append(step_out["next_state_entropy"])
            top1_traj.append(step_out["next_state_top1_prob"])
            if "reward" in step_out:
                reward_traj.append(step_out["reward"])
            if "continuation" in step_out:
                continuation_traj.append(step_out["continuation"])

        out = {
            "state_probs": torch.stack(state_traj, dim=-2),
            "next_state_probs": torch.stack(next_state_traj, dim=-2),
            "next_state_entropy": torch.stack(entropy_traj, dim=-2),
            "next_state_top1_prob": torch.stack(top1_traj, dim=-2),
        }
        if reward_traj:
            out["reward"] = torch.stack(reward_traj, dim=-2)
        if continuation_traj:
            out["continuation"] = torch.stack(continuation_traj, dim=-2)
        return out


def compute_markov_transition_loss(
    model: MacroTransitionModel,
    state_probs: torch.Tensor,
    action_probs: torch.Tensor,
    *,
    obs_geometry: dict[str, torch.Tensor],
    act_geometry: dict[str, torch.Tensor],
    target_next_state_probs: torch.Tensor | None = None,
    target_next_chart_idx: torch.Tensor | None = None,
    target_next_code_idx: torch.Tensor | None = None,
    codes_per_chart: int | None = None,
    valid_mask: torch.Tensor | None = None,
    metric_prefix: str = "markov",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Fit the coarse model on replay next-symbol supervision."""
    pred = model(
        state_probs,
        action_probs,
        obs_geometry=obs_geometry,
        act_geometry=act_geometry,
        eps=eps,
    )
    flat_next_probs = pred["next_state_probs"].reshape(-1, model.num_states)
    flat_next_log_probs = pred["next_state_log_probs"].reshape(-1, model.num_states)
    flat_next_chart_probs = pred["next_chart_probs"].reshape(-1, model.num_obs_charts)
    flat_next_chart_log_probs = pred["next_chart_log_probs"].reshape(-1, model.num_obs_charts)
    flat_next_code_probs = pred["next_code_probs"].reshape(
        -1, model.num_obs_charts, model.obs_codes_per_chart
    )
    flat_next_code_log_probs = pred["next_code_log_probs"].reshape(
        -1, model.num_obs_charts, model.obs_codes_per_chart
    )

    if valid_mask is None:
        flat_valid = flat_next_probs.new_ones(flat_next_probs.shape[0])
    else:
        flat_valid = valid_mask.reshape(-1).to(flat_next_probs)

    target_state: torch.Tensor
    flat_target_chart: torch.Tensor
    flat_target_code: torch.Tensor
    target_entropy = flat_next_probs.new_zeros(flat_next_probs.shape[0])

    if target_next_state_probs is not None:
        flat_target_probs = _normalize_probs(
            target_next_state_probs.reshape(-1, model.num_states),
            eps=eps,
        )
        target_chart_probs = _state_probs_to_chart_probs(
            flat_target_probs,
            model.num_obs_charts,
            model.obs_codes_per_chart,
        )
        target_code_probs = _state_probs_to_code_conditionals(
            flat_target_probs,
            model.num_obs_charts,
            model.obs_codes_per_chart,
            eps=eps,
        )
        flat_chart_ce = -(target_chart_probs * flat_next_chart_log_probs).sum(dim=-1)
        flat_code_ce = -(
            target_chart_probs.unsqueeze(-1) * target_code_probs * flat_next_code_log_probs
        ).sum(dim=(-1, -2))
        flat_state_ce = -(flat_target_probs * flat_next_log_probs).sum(dim=-1)
        target_state = flat_target_probs.argmax(dim=-1)
        flat_target_chart = torch.div(target_state, model.obs_codes_per_chart, rounding_mode="floor")
        flat_target_code = target_state.remainder(model.obs_codes_per_chart)
        target_entropy = -(flat_target_probs * flat_target_probs.clamp(min=eps).log()).sum(dim=-1)
    else:
        if target_next_chart_idx is None or target_next_code_idx is None or codes_per_chart is None:
            msg = (
                "Provide either target_next_state_probs or the "
                "(target_next_chart_idx, target_next_code_idx, codes_per_chart) tuple."
            )
            raise ValueError(msg)
        flat_target_chart = target_next_chart_idx.reshape(-1).long()
        flat_target_code = target_next_code_idx.reshape(-1).long()
        target_state = _state_index(flat_target_chart, flat_target_code, codes_per_chart)
        flat_chart_ce = F.nll_loss(flat_next_chart_log_probs, flat_target_chart, reduction="none")
        code_rows = flat_next_code_log_probs[
            torch.arange(flat_next_code_log_probs.shape[0], device=flat_next_code_log_probs.device),
            flat_target_chart,
        ]
        flat_code_ce = F.nll_loss(code_rows, flat_target_code, reduction="none")
        flat_state_ce = F.nll_loss(flat_next_log_probs, target_state, reduction="none")

    flat_transition_ce = flat_chart_ce + flat_code_ce
    loss = _masked_mean(flat_transition_ce, flat_valid)
    pred_state = flat_next_probs.argmax(dim=-1)
    pred_chart = torch.div(pred_state, model.obs_codes_per_chart, rounding_mode="floor")
    pred_code = pred_state.remainder(model.obs_codes_per_chart)

    metrics = {
        f"{metric_prefix}/L_transition": float(loss.detach()),
        f"{metric_prefix}/transition_ce": float(_masked_mean(flat_transition_ce, flat_valid).detach()),
        f"{metric_prefix}/state_ce": float(_masked_mean(flat_state_ce, flat_valid).detach()),
        f"{metric_prefix}/chart_ce": float(_masked_mean(flat_chart_ce, flat_valid).detach()),
        f"{metric_prefix}/code_ce": float(_masked_mean(flat_code_ce, flat_valid).detach()),
        f"{metric_prefix}/transition_acc": float(
            _masked_mean((pred_state == target_state).to(flat_next_probs.dtype), flat_valid).detach()
        ),
        f"{metric_prefix}/chart_acc": float(
            _masked_mean((pred_chart == flat_target_chart).to(flat_next_probs.dtype), flat_valid).detach()
        ),
        f"{metric_prefix}/code_acc": float(
            _masked_mean((pred_code == flat_target_code).to(flat_next_probs.dtype), flat_valid).detach()
        ),
        f"{metric_prefix}/next_state_entropy": float(
            _masked_mean(pred["next_state_entropy"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_chart_entropy": float(
            _masked_mean(pred["next_chart_entropy"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_code_entropy": float(
            _masked_mean(pred["next_code_entropy"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_state_top1_prob": float(
            _masked_mean(pred["next_state_top1_prob"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_chart_top1_prob": float(
            _masked_mean(pred["next_chart_top1_prob"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_code_top1_prob": float(
            _masked_mean(pred["next_code_top1_prob"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/target_state_entropy": float(
            _masked_mean(target_entropy, flat_valid).detach()
        ),
    }

    if "residual_transition_logits" in pred:
        metrics[f"{metric_prefix}/residual_logit_norm"] = float(
            pred["residual_transition_logits"].norm(dim=-1).mean().detach()
        )

    return loss, metrics, pred


def compute_distribution_alignment_loss(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    metric_prefix: str = "markov/alignment",
    detach_teacher: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Align one symbolic distribution to another with masked KL/CE metrics."""
    if teacher_probs.shape != student_probs.shape:
        msg = "teacher_probs and student_probs must have the same shape."
        raise ValueError(msg)

    flat_teacher = _normalize_probs(teacher_probs.reshape(-1, teacher_probs.shape[-1]), eps=eps)
    flat_student = _normalize_probs(student_probs.reshape(-1, student_probs.shape[-1]), eps=eps)
    if detach_teacher:
        flat_teacher = flat_teacher.detach()

    if valid_mask is None:
        flat_valid = flat_student.new_ones(flat_student.shape[0])
    else:
        flat_valid = valid_mask.reshape(-1).to(flat_student)

    flat_student_log_probs = flat_student.clamp(min=eps).log()
    flat_teacher_log_probs = flat_teacher.clamp(min=eps).log()
    cross_entropy = -(flat_teacher * flat_student_log_probs).sum(dim=-1)
    teacher_entropy = -(flat_teacher * flat_teacher_log_probs).sum(dim=-1)
    kl = cross_entropy - teacher_entropy
    loss = _masked_mean(kl, flat_valid)

    teacher_idx = flat_teacher.argmax(dim=-1)
    student_idx = flat_student.argmax(dim=-1)
    student_entropy = -(flat_student * flat_student_log_probs).sum(dim=-1)

    metrics = {
        f"{metric_prefix}/L_align": float(loss.detach()),
        f"{metric_prefix}/align_ce": float(_masked_mean(cross_entropy, flat_valid).detach()),
        f"{metric_prefix}/align_kl": float(_masked_mean(kl, flat_valid).detach()),
        f"{metric_prefix}/agreement": float(
            _masked_mean((teacher_idx == student_idx).to(flat_student.dtype), flat_valid).detach()
        ),
        f"{metric_prefix}/teacher_entropy": float(_masked_mean(teacher_entropy, flat_valid).detach()),
        f"{metric_prefix}/student_entropy": float(_masked_mean(student_entropy, flat_valid).detach()),
    }
    return loss, metrics


def compute_markov_shape_loss(
    macro_next_state_probs: torch.Tensor,
    live_next_state_probs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    detach_teacher: bool = True,
    metric_prefix: str = "markov/shape",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Use the coarse model as a teacher for the live atlas symbolization."""
    return compute_distribution_alignment_loss(
        macro_next_state_probs,
        live_next_state_probs,
        valid_mask=valid_mask,
        metric_prefix=metric_prefix,
        detach_teacher=detach_teacher,
        eps=eps,
    )


def compute_markov_world_model_alignment_loss(
    macro_next_state_probs: torch.Tensor,
    wm_next_state_probs: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    detach_teacher: bool = True,
    metric_prefix: str = "markov/wm_align",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Make the micro world model follow the coarse symbolic transition."""
    return compute_distribution_alignment_loss(
        macro_next_state_probs,
        wm_next_state_probs,
        valid_mask=valid_mask,
        metric_prefix=metric_prefix,
        detach_teacher=detach_teacher,
        eps=eps,
    )


def compute_macro_auxiliary_loss(
    model: MacroTransitionModel,
    state_probs: torch.Tensor,
    action_probs: torch.Tensor,
    reward_target: torch.Tensor,
    continuation_target: torch.Tensor,
    *,
    weight_reward: float,
    weight_continuation: float,
    metric_prefix: str = "model",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Fit the reward and continuation tables on detached symbolic replay targets."""
    zero = reward_target.new_zeros(())
    total = zero
    metrics: dict[str, float] = {}

    if model.reward_table is not None and float(weight_reward) != 0.0:
        reward_pred = model.reward_from_probs(state_probs, action_probs)
        reward_loss = F.smooth_l1_loss(reward_pred, reward_target)
        total = total + float(weight_reward) * reward_loss
        metrics.update({
            f"{metric_prefix}/reward_loss": float(reward_loss.detach()),
            f"{metric_prefix}/reward_pred_mean": float(reward_pred.mean().detach()),
            f"{metric_prefix}/reward_target_mean": float(reward_target.mean().detach()),
        })
    else:
        metrics.update({
            f"{metric_prefix}/reward_loss": 0.0,
            f"{metric_prefix}/reward_pred_mean": 0.0,
            f"{metric_prefix}/reward_target_mean": float(reward_target.mean().detach()),
        })

    if model.continuation_logits is not None and float(weight_continuation) != 0.0:
        continuation_pred = model.continuation_from_probs(state_probs, action_probs)
        continuation_loss = F.binary_cross_entropy(
            continuation_pred.clamp(min=1e-6, max=1.0 - 1e-6),
            continuation_target,
        )
        total = total + float(weight_continuation) * continuation_loss
        metrics.update({
            f"{metric_prefix}/continuation_loss": float(continuation_loss.detach()),
            f"{metric_prefix}/continuation_pred_mean": float(continuation_pred.mean().detach()),
            f"{metric_prefix}/continuation_target_mean": float(continuation_target.mean().detach()),
        })
    else:
        metrics.update({
            f"{metric_prefix}/continuation_loss": 0.0,
            f"{metric_prefix}/continuation_pred_mean": 0.0,
            f"{metric_prefix}/continuation_target_mean": float(continuation_target.mean().detach()),
        })

    metrics[f"{metric_prefix}/aux_loss"] = float(total.detach())
    return total, metrics


__all__ = [
    "MacroTransitionModel",
    "compose_absolute_macro_dictionary",
    "compute_distribution_alignment_loss",
    "compute_macro_auxiliary_loss",
    "compute_markov_shape_loss",
    "compute_markov_transition_loss",
    "compute_markov_world_model_alignment_loss",
    "expected_macro_state",
    "soft_macro_state_distribution",
]
