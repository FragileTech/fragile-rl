"""Differentiable coarse Markov model on atlas macro symbols.

This module is the standalone symbolic planner discussed for Dreamer:

- observations and actions are softly mapped to chart/code symbol
  distributions in their own Poincare balls,
- each flattened symbolic state has an absolute point
  ``c_k ⊕ q_{k,c}`` attached to it,
- ``MacroTransitionModel`` learns a stochastic coarse transition
  ``p(s_{t+1} | s_t, a_t)``,
- helper losses let the coarse model fit replay transitions, shape the
  symbolic atlas, and supervise the micro world model later on.

The code here does not touch the current trainer yet. It only provides the
pieces needed to wire the symbolic Markov path in a later step.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

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


def compose_absolute_macro_dictionary(
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compose the absolute point for every flattened symbolic state.

    The resulting dictionary lives in one manifold only. Use the observation
    chart/code tensors to build the observation symbol dictionary and the action
    tensors to build the action symbol dictionary.

    Args:
        chart_centers: Absolute chart centers with shape ``[num_charts, latent_dim]``.
        codebook: Chart-local code centers with shape
            ``[num_charts, codes_per_chart, latent_dim]``.

    Returns:
        A dictionary with:
        - ``state_points``: absolute macro points ``c_k ⊕ q_{k,c}``, shape ``[S, D]``.
        - ``state_tangent_points``: origin-tangent coordinates of those points, shape ``[S, D]``.
        - ``chart_idx``: chart index attached to each flattened state, shape ``[S]``.
        - ``code_idx``: code index attached to each flattened state, shape ``[S]``.
    """
    _validate_macro_geometry(chart_centers, codebook)

    # Put both chart centers and chart-local code centers inside the valid
    # Poincare ball before composing them into absolute symbolic coordinates.
    chart_centers_proj = project_to_ball(chart_centers)
    codebook_proj = project_to_ball(codebook).to(
        device=chart_centers_proj.device,
        dtype=chart_centers_proj.dtype,
    )

    # Each flattened symbolic state is represented by the absolute point reached
    # by starting at a chart center and applying that chart's local code offset.
    state_points = project_to_ball(mobius_add(chart_centers_proj[:, None, :], codebook_proj))
    num_charts, codes_per_chart, latent_dim = codebook_proj.shape
    device = codebook_proj.device

    # Keep the original chart/code ids attached to the flattened state order so
    # later code can move freely between table indices and symbolic tuples.
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
    """Map a soft symbolic state to one barycentric point in the same manifold.

    This is a summary of the symbolic distribution, not the actual state used
    by the coarse planner. The coarse model should still reason in terms of the
    full distribution over symbols.
    """
    if state_points.dim() != 2:
        msg = "state_points must have shape [S, D]."
        raise ValueError(msg)
    if state_probs.shape[-1] != state_points.shape[0]:
        msg = "state_probs and state_points must agree on the number of symbols."
        raise ValueError(msg)

    # Collapse any batch/time axes, compute the hyperbolic barycenter of the
    # symbolic support points, then restore the original leading dimensions.
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
    """Attach a differentiable coarse symbolic state to a manifold point.

    The symbolizer mirrors the atlas factorization instead of snapping directly
    to one absolute macro point:

    1. score charts from the latent point,
    2. form the soft chart barycenter ``c_bar``,
    3. move the latent into chart-local coordinates,
    4. score chart-local codes,
    5. flatten ``chart_probs * code_probs`` into one symbolic distribution.

    Because the entire path is soft, gradients can flow back into the chart
    centers, codebook, and the latent that is being symbolized.

    Args:
        z_latent: Manifold points with shape ``[..., latent_dim]``.
        chart_centers: Absolute chart centers with shape ``[num_charts, latent_dim]``.
        codebook: Chart-local code centers with shape
            ``[num_charts, codes_per_chart, latent_dim]``.
        chart_tau: Temperature for the chart softmax.
        code_tau: Temperature for the code softmax inside each chart.
        eps: Numerical floor used when normalizing or taking logs.

    Returns:
        A dictionary containing soft chart/code/state distributions, hard
        argmax indices derived from the joint symbolic state, the absolute macro
        dictionary, and a barycentric ``macro_state_mean`` summary.
    """
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

    # First decide which charts are plausible for each latent point. This keeps
    # the symbolic state soft and differentiable instead of hard-routing early.
    chart_dist = hyperbolic_distance(flat_z.unsqueeze(1), chart_centers_proj.unsqueeze(0))
    chart_logits = -chart_dist / chart_tau
    chart_log_probs = F.log_softmax(chart_logits, dim=-1)
    chart_probs = chart_log_probs.exp()

    # Translate the latent point into the soft chart frame defined by the
    # chart barycenter. Codes are scored in these local coordinates.
    c_bar = poincare_weighted_mean(chart_centers_proj, chart_probs, eps=eps)
    v_local = project_to_ball(mobius_add(-c_bar, flat_z))

    # Within each chart, score the chart-local code centers against the local
    # latent coordinates. This yields a per-chart categorical code distribution.
    code_dist = hyperbolic_distance(v_local[:, None, None, :], codebook_proj[None, :, :, :])
    code_logits = -code_dist / code_tau
    code_log_probs = F.log_softmax(code_logits, dim=-1)
    code_probs = code_log_probs.exp()

    # Combine the chart and per-chart code probabilities into one flattened
    # symbolic state distribution over all `(chart, code)` pairs.
    state_log_probs = (chart_log_probs.unsqueeze(-1) + code_log_probs).reshape(
        flat_z.shape[0], -1
    )
    state_probs = state_log_probs.exp()

    # Decode the most likely symbolic state only for diagnostics/convenience.
    # The coarse model itself should still use the full soft distribution.
    codes_per_chart = codebook_proj.shape[1]
    state_idx = state_probs.argmax(dim=-1)
    chart_idx = torch.div(state_idx, codes_per_chart, rounding_mode="floor")
    code_idx = state_idx.remainder(codes_per_chart)

    # Build reusable absolute coordinates for every symbol, then expose both
    # the hard selected point and the soft barycentric summary of the state.
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
        **symbol_dict,
    }


class MacroTransitionModel(nn.Module):
    """Full stochastic symbolic dynamics model ``p(s_{t+1} | s_t, a_t)``.

    The model learns a dense transition tensor over observation symbols and
    action symbols. Planning stays cheap because rollouts operate directly on
    probability vectors instead of decoding the full micro latent.

    The class can also carry a coarse reward table and a continuation table so
    symbolic rollouts already expose the signals a planner usually needs.
    """

    def __init__(
        self,
        num_states: int,
        num_actions: int,
        *,
        learn_reward: bool = True,
        learn_continuation: bool = True,
        initial_continuation: float = 0.99,
    ) -> None:
        super().__init__()
        self.num_states = int(num_states)
        self.num_actions = int(num_actions)
        if self.num_states <= 0 or self.num_actions <= 0:
            msg = "num_states and num_actions must both be positive."
            raise ValueError(msg)

        self.transition_logits = nn.Parameter(
            torch.zeros(self.num_states, self.num_actions, self.num_states)
        )

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

    def transition_table(self) -> torch.Tensor:
        """Return the normalized transition tensor ``[S, A, S]``."""
        return F.softmax(self.transition_logits, dim=-1)

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
        # Average the symbolic reward table under the current belief over
        # coarse states and actions.
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
        # This mirrors `reward_from_probs`, but reads from the continuation
        # table so symbolic rollouts can also predict nonterminal probability.
        state_probs = _normalize_probs(state_probs)
        action_probs = _normalize_probs(action_probs)
        return torch.einsum("...s,sa,...a->...", state_probs, table, action_probs)

    def conditional_from_indices(
        self,
        state_idx: torch.Tensor,
        action_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Read one conditional transition row ``p(s' | s, a)`` from hard indices."""
        if state_idx.shape != action_idx.shape:
            msg = "state_idx and action_idx must have matching shapes."
            raise ValueError(msg)
        flat_state_idx = state_idx.reshape(-1).long()
        flat_action_idx = action_idx.reshape(-1).long()
        leading_shape = state_idx.shape

        # Hard indices simply read one row from the learned transition tensor.
        transition_row = self.transition_table()[flat_state_idx, flat_action_idx]
        out = {
            "next_state_probs": _reshape_leading_dims(transition_row, leading_shape),
            "next_state_log_probs": _reshape_leading_dims(
                transition_row.clamp(min=1e-8).log(),
                leading_shape,
            ),
        }
        if self.reward_table is not None:
            reward = self.reward_table[flat_state_idx, flat_action_idx]
            out["reward"] = _reshape_leading_dims(reward, leading_shape)
        if self.continuation_logits is not None:
            continuation = self.continuation_table()[flat_state_idx, flat_action_idx]
            out["continuation"] = _reshape_leading_dims(continuation, leading_shape)
        return out

    def forward(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Roll one coarse Markov step from soft symbolic state/action inputs."""
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

        # Normalize the incoming symbolic beliefs, then marginalize the full
        # `T[s, a, s']` tensor under them to get the next symbolic belief.
        state_probs = _normalize_probs(state_probs)
        action_probs = _normalize_probs(action_probs)
        transition = self.transition_table()

        next_state_probs = torch.einsum(
            "...s,...a,san->...n",
            state_probs,
            action_probs,
            transition,
        )
        next_state_log_probs = next_state_probs.clamp(min=1e-8).log()
        next_state_entropy = -(next_state_probs * next_state_log_probs).sum(dim=-1)
        next_state_top1_prob = next_state_probs.max(dim=-1).values

        # Keep a few diagnostics that are useful later when we want to know
        # whether the coarse model is confident or highly aliased.
        out = {
            "next_state_probs": next_state_probs,
            "next_state_log_probs": next_state_log_probs,
            "next_state_entropy": next_state_entropy,
            "next_state_top1_prob": next_state_top1_prob,
        }
        if self.reward_table is not None:
            # The same symbolic belief is used to read expected reward.
            out["reward"] = self.reward_from_probs(state_probs, action_probs)
        if self.continuation_logits is not None:
            # And optionally the expected continuation probability.
            out["continuation"] = self.continuation_from_probs(state_probs, action_probs)
        return out

    def rollout(
        self,
        state_probs_0: torch.Tensor,
        action_probs_seq: torch.Tensor,
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

        # Roll the belief state forward one symbolic action at a time and keep
        # the whole trajectory for planning or multi-step supervision.
        current_state = _normalize_probs(state_probs_0)
        state_traj = [current_state]
        next_state_traj: list[torch.Tensor] = []
        entropy_traj: list[torch.Tensor] = []
        top1_traj: list[torch.Tensor] = []
        reward_traj: list[torch.Tensor] = []
        continuation_traj: list[torch.Tensor] = []

        horizon = action_probs_seq.shape[-2]
        for t in range(horizon):
            step_out = self(current_state, action_probs_seq[..., t, :])
            current_state = step_out["next_state_probs"]
            state_traj.append(current_state)
            next_state_traj.append(step_out["next_state_probs"])
            entropy_traj.append(step_out["next_state_entropy"])
            top1_traj.append(step_out["next_state_top1_prob"])
            if "reward" in step_out:
                reward_traj.append(step_out["reward"])
            if "continuation" in step_out:
                continuation_traj.append(step_out["continuation"])

        # Return both the full state trajectory and the per-step outputs so the
        # caller can use this for planning, supervision, or diagnostics.
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
    target_next_state_probs: torch.Tensor | None = None,
    target_next_chart_idx: torch.Tensor | None = None,
    target_next_code_idx: torch.Tensor | None = None,
    codes_per_chart: int | None = None,
    valid_mask: torch.Tensor | None = None,
    metric_prefix: str = "markov",
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    """Fit the coarse model on replay next-symbol supervision.

    This helper does not detach inputs or targets. If the caller passes live
    symbolic probabilities from the atlas, the transition loss can reshape the
    symbol geometry. Detach before calling if a frozen target is desired.
    """
    pred = model(state_probs, action_probs)
    flat_next_probs = pred["next_state_probs"].reshape(-1, model.num_states)
    flat_next_log_probs = pred["next_state_log_probs"].reshape(-1, model.num_states)

    if valid_mask is None:
        flat_valid = flat_next_probs.new_ones(flat_next_probs.shape[0])
    else:
        flat_valid = valid_mask.reshape(-1).to(flat_next_probs)

    target_state: torch.Tensor | None = None
    target_entropy = flat_next_probs.new_zeros(flat_next_probs.shape[0])
    if target_next_state_probs is not None:
        # Soft targets let the macro model imitate another symbolic predictor or
        # a teacher distribution instead of one-hot replay labels.
        flat_target_probs = _normalize_probs(
            target_next_state_probs.reshape(-1, model.num_states),
            eps=eps,
        )
        flat_transition_ce = -(flat_target_probs * flat_next_log_probs).sum(dim=-1)
        target_state = flat_target_probs.argmax(dim=-1)
        target_entropy = -(flat_target_probs * flat_target_probs.clamp(min=eps).log()).sum(dim=-1)
    else:
        # Hard targets train against replay `(chart, code)` labels exactly the
        # same way a standard next-class model would.
        if target_next_chart_idx is None or target_next_code_idx is None or codes_per_chart is None:
            msg = (
                "Provide either target_next_state_probs or the "
                "(target_next_chart_idx, target_next_code_idx, codes_per_chart) tuple."
            )
            raise ValueError(msg)
        flat_target_chart = target_next_chart_idx.reshape(-1).long()
        flat_target_code = target_next_code_idx.reshape(-1).long()
        target_state = _state_index(flat_target_chart, flat_target_code, codes_per_chart)
        flat_transition_ce = F.nll_loss(flat_next_log_probs, target_state, reduction="none")

    # Optimize the masked average transition CE and report both full-state and
    # decomposed chart/code accuracy for debugging symbol failures.
    loss = _masked_mean(flat_transition_ce, flat_valid)
    pred_state = flat_next_probs.argmax(dim=-1)

    metrics = {
        f"{metric_prefix}/L_transition": float(loss.detach()),
        f"{metric_prefix}/transition_ce": float(_masked_mean(flat_transition_ce, flat_valid).detach()),
        f"{metric_prefix}/transition_acc": float(
            _masked_mean((pred_state == target_state).to(flat_next_probs.dtype), flat_valid).detach()
        ),
        f"{metric_prefix}/next_state_entropy": float(
            _masked_mean(pred["next_state_entropy"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/next_state_top1_prob": float(
            _masked_mean(pred["next_state_top1_prob"].reshape(-1), flat_valid).detach()
        ),
        f"{metric_prefix}/target_state_entropy": float(
            _masked_mean(target_entropy, flat_valid).detach()
        ),
    }

    if target_next_chart_idx is not None and target_next_code_idx is not None and codes_per_chart is not None:
        flat_target_chart = target_next_chart_idx.reshape(-1).long()
        flat_target_code = target_next_code_idx.reshape(-1).long()
        pred_chart = torch.div(pred_state, codes_per_chart, rounding_mode="floor")
        pred_code = pred_state.remainder(codes_per_chart)
        metrics[f"{metric_prefix}/chart_acc"] = float(
            _masked_mean((pred_chart == flat_target_chart).to(flat_next_probs.dtype), flat_valid).detach()
        )
        metrics[f"{metric_prefix}/code_acc"] = float(
            _masked_mean((pred_code == flat_target_code).to(flat_next_probs.dtype), flat_valid).detach()
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
    """Align one symbolic distribution to another with masked KL/CE metrics.

    The default is the teacher-student setup discussed for the macro model:
    the teacher is detached and the student receives the gradients.
    """
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

    # Compute teacher->student cross-entropy and KL. In the common use case the
    # teacher is detached, so this becomes a one-way shaping loss.
    flat_student_log_probs = flat_student.clamp(min=eps).log()
    flat_teacher_log_probs = flat_teacher.clamp(min=eps).log()
    cross_entropy = -(flat_teacher * flat_student_log_probs).sum(dim=-1)
    teacher_entropy = -(flat_teacher * flat_teacher_log_probs).sum(dim=-1)
    kl = cross_entropy - teacher_entropy
    loss = _masked_mean(kl, flat_valid)

    # Agreement and entropy metrics tell us whether the student is matching the
    # teacher's modes and whether it is collapsing or staying too diffuse.
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


__all__ = [
    "MacroTransitionModel",
    "compose_absolute_macro_dictionary",
    "compute_distribution_alignment_loss",
    "compute_markov_shape_loss",
    "compute_markov_transition_loss",
    "compute_markov_world_model_alignment_loss",
    "expected_macro_state",
    "soft_macro_state_distribution",
]
