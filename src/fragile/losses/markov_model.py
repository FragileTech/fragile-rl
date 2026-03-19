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
    """Compute the mean over entries where ``mask`` is one.

    Args:
        values: Tensor of arbitrary shape containing the values to average.
        mask: Tensor broadcastable to ``values`` with ones marking valid entries
            and zeros marking entries to ignore.

    Returns:
        torch.Tensor: Scalar mean of the masked entries. If no entries are valid
            the denominator is clamped to 1 to avoid division by zero.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _state_index(
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
    codes_per_chart: int,
) -> torch.Tensor:
    """Flatten ``(chart, code)`` symbolic indices into one state id.

    Args:
        chart_idx: Integer tensor of chart indices with arbitrary shape.
        code_idx: Integer tensor of code-within-chart indices, same shape as
            ``chart_idx``.
        codes_per_chart: Number of codes per chart (stride for the chart axis).

    Returns:
        torch.Tensor: Long tensor with the same shape as the inputs containing
            the flattened state index ``chart_idx * codes_per_chart + code_idx``.
    """
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _normalize_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a non-negative tensor along the last axis.

    Args:
        probs: Non-negative tensor of arbitrary shape to be normalized along
            its last dimension.
        eps: Small constant used to clamp the denominator and prevent division
            by zero.

    Returns:
        torch.Tensor: Tensor with the same shape as ``probs`` whose last axis
            sums to one.
    """
    return probs / probs.sum(dim=-1, keepdim=True).clamp(min=eps)


def _validate_macro_geometry(chart_centers: torch.Tensor, codebook: torch.Tensor) -> None:
    """Validate chart/code tensors before composing symbolic coordinates.

    Args:
        chart_centers: Tensor of shape ``[N_c, D]`` with the Poincare-ball
            center for each chart.
        codebook: Tensor of shape ``[N_c, K, D]`` with the local code
            embeddings for each chart.

    Returns:
        None. Raises ``ValueError`` if shapes are inconsistent.
    """
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
    """Restore leading batch dimensions after a temporary flatten.

    Args:
        x: Tensor whose first dimension was previously flattened from
            ``leading_shape``.
        leading_shape: The original leading dimensions to restore. An empty
            ``Size`` squeezes the leading dimension away.

    Returns:
        torch.Tensor: Tensor reshaped to ``(*leading_shape, *x.shape[1:])``.
    """
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
    """View a flattened state distribution as ``[..., chart, code]``.

    Args:
        state_probs: Tensor of shape ``[..., num_charts * codes_per_chart]``
            containing a flattened state probability distribution.
        num_charts: Number of charts in the atlas.
        codes_per_chart: Number of codes inside each chart.

    Returns:
        torch.Tensor: Reshaped tensor of shape ``[..., num_charts, codes_per_chart]``.
    """
    if state_probs.shape[-1] != num_charts * codes_per_chart:
        msg = "state_probs does not match the requested chart/code factorization."
        raise ValueError(msg)
    return state_probs.reshape(*state_probs.shape[:-1], num_charts, codes_per_chart)


def _state_probs_to_chart_probs(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> torch.Tensor:
    """Marginalize flattened state probabilities down to chart probabilities.

    Args:
        state_probs: Tensor of shape ``[..., num_charts * codes_per_chart]``
            containing a flattened state probability distribution.
        num_charts: Number of charts in the atlas.
        codes_per_chart: Number of codes inside each chart.

    Returns:
        torch.Tensor: Tensor of shape ``[..., num_charts]`` with the marginal
            chart probabilities obtained by summing over codes.
    """
    return _state_chart_code_view(state_probs, num_charts, codes_per_chart).sum(dim=-1)


def _state_probs_to_code_conditionals(
    state_probs: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert flattened state probabilities into per-chart code conditionals.

    For each chart, computes ``p(code | chart)`` by dividing the joint
    state probability by the marginal chart probability. Charts with
    near-zero marginal probability receive a uniform code distribution.

    Args:
        state_probs: Tensor of shape ``[..., num_charts * codes_per_chart]``
            containing a flattened state probability distribution.
        num_charts: Number of charts in the atlas.
        codes_per_chart: Number of codes inside each chart.
        eps: Small constant to prevent division by zero when a chart has
            negligible probability.

    Returns:
        torch.Tensor: Tensor of shape ``[..., num_charts, codes_per_chart]``
            with the conditional code probabilities for each chart.
    """
    view = _state_chart_code_view(state_probs, num_charts, codes_per_chart)
    chart_probs = view.sum(dim=-1, keepdim=True)
    code_probs = view / chart_probs.clamp(min=eps)
    uniform = code_probs.new_full(code_probs.shape, 1.0 / float(codes_per_chart))
    return torch.where(chart_probs <= eps, uniform, code_probs)


def _flatten_chart_code_probs(
    chart_probs: torch.Tensor,
    code_probs: torch.Tensor,
) -> torch.Tensor:
    """Flatten factorized chart/code probabilities back into one state axis.

    Computes the joint ``p(chart, code) = p(chart) * p(code | chart)`` and
    reshapes the last two dimensions into a single flattened state axis.

    Args:
        chart_probs: Tensor of shape ``[..., num_charts]`` with marginal chart
            probabilities.
        code_probs: Tensor of shape ``[..., num_charts, codes_per_chart]`` with
            conditional code probabilities for each chart.

    Returns:
        torch.Tensor: Tensor of shape ``[..., num_charts * codes_per_chart]``
            with the joint state probabilities.
    """
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

    Args:
        chart_centers: Tensor of shape ``[N_c, D]`` with the Poincare-ball
            center for each chart.
        codebook: Tensor of shape ``[N_c, K, D]`` with the local code
            embeddings for each chart.

    Returns:
        dict[str, torch.Tensor]: Dictionary with the following keys:
            - ``"state_points"``: Tensor of shape ``[N_c * K, D]`` with the
              absolute Poincare-ball point for every flattened state.
            - ``"state_tangent_points"``: Tensor of shape ``[N_c * K, D]`` with
              the tangent-space (log-map at origin) coordinates for every state.
            - ``"chart_idx"``: Long tensor of shape ``[N_c * K]`` mapping each
              flattened state back to its chart index.
            - ``"code_idx"``: Long tensor of shape ``[N_c * K]`` mapping each
              flattened state back to its code-within-chart index.
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
        torch
        .arange(num_charts, device=device)
        .unsqueeze(1)
        .expand(num_charts, codes_per_chart)
        .reshape(-1)
    )
    code_idx = (
        torch
        .arange(codes_per_chart, device=device)
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

    Computes the Poincare weighted mean of the state dictionary points using
    the given probability weights.

    Args:
        state_probs: Tensor of shape ``[..., S]`` with soft probability weights
            over the ``S`` symbolic states.
        state_points: Tensor of shape ``[S, D]`` with the absolute Poincare-ball
            point for each symbolic state.
        eps: Small constant for numerical stability in normalization and the
            weighted mean computation.

    Returns:
        torch.Tensor: Tensor of shape ``[..., D]`` with the barycentric point
            in the Poincare ball for each element of the batch.
    """
    if state_points.dim() != 2:
        msg = "state_points must have shape [S, D]."
        raise ValueError(msg)
    if state_probs.shape[-1] != state_points.shape[0]:
        msg = "state_probs and state_points must agree on the number of symbols."
        raise ValueError(msg)

    leading_shape = state_probs.shape[:-1]
    flat_probs = _normalize_probs(state_probs.reshape(-1, state_probs.shape[-1]), eps=eps)
    flat_points = project_to_ball(state_points).to(
        device=flat_probs.device, dtype=flat_probs.dtype
    )
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

    Given a latent point in the Poincare ball, computes a soft factorized
    distribution over ``(chart, code)`` symbols by measuring hyperbolic
    distances to the chart centers and local codebooks.

    Args:
        z_latent: Tensor of shape ``[..., D]`` with latent points in the
            Poincare ball.
        chart_centers: Tensor of shape ``[N_c, D]`` with the Poincare-ball
            center for each chart.
        codebook: Tensor of shape ``[N_c, K, D]`` with the local code
            embeddings for each chart.
        chart_tau: Temperature for the chart-level softmax. Lower values
            produce sharper chart assignments.
        code_tau: Temperature for the code-level softmax. Lower values
            produce sharper code assignments.
        eps: Small constant for numerical stability.

    Returns:
        dict[str, torch.Tensor]: Dictionary containing (all tensors restore the
            original leading batch dimensions of ``z_latent``):
            - ``"z_latent"``: Projected latent points, shape ``[..., D]``.
            - ``"router_weights"``: Same as ``chart_probs``, shape ``[..., N_c]``.
            - ``"chart_logits"``: Raw chart logits, shape ``[..., N_c]``.
            - ``"chart_probs"``: Soft chart probabilities, shape ``[..., N_c]``.
            - ``"chart_idx"``: Hard chart index (argmax), shape ``[...]``.
            - ``"c_bar"``: Weighted mean chart center, shape ``[..., D]``.
            - ``"v_local"``: Local residual after removing chart center, shape ``[..., D]``.
            - ``"code_logits"``: Raw code logits, shape ``[..., N_c, K]``.
            - ``"code_probs"``: Conditional code probabilities, shape ``[..., N_c, K]``.
            - ``"code_idx"``: Hard code index (argmax of joint), shape ``[...]``.
            - ``"state_log_probs"``: Log of the joint state distribution,
              shape ``[..., N_c * K]``.
            - ``"state_probs"``: Joint state probabilities, shape ``[..., N_c * K]``.
            - ``"state_idx"``: Hard state index (argmax of joint), shape ``[...]``.
            - ``"state_entropy"``: Entropy of the state distribution, shape ``[...]``.
            - ``"state_value_entropy"``: Same as ``state_entropy``, shape ``[...]``.
            - ``"chart_entropy"``: Entropy of the chart distribution, shape ``[...]``.
            - ``"macro_state_mean"``: Barycentric mean in the Poincare ball,
              shape ``[..., D]``.
            - ``"hard_state_point"``: Poincare-ball point for the hard state,
              shape ``[..., D]``.
            - ``"chart_centers"``: Projected chart centers, shape ``[N_c, D]``.
            - ``"chart_tangent_points"``: Tangent-space chart centers, shape ``[N_c, D]``.
            - ``"codebook"``: Projected codebook, shape ``[N_c, K, D]``.
            - ``"code_tangent_points"``: Tangent-space codebook, shape ``[N_c, K, D]``.
            - Plus all keys from ``compose_absolute_macro_dictionary``.
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
    chart_centers_proj = project_to_ball(chart_centers).to(
        device=flat_z.device, dtype=flat_z.dtype
    )
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
        """Initialize the macro transition model.

        Args:
            obs_latent_dim: Dimensionality of the observation Poincare-ball
                latent space.
            act_latent_dim: Dimensionality of the action Poincare-ball latent
                space.
            num_obs_charts: Number of observation charts in the atlas.
            obs_codes_per_chart: Number of codes inside each observation chart.
            num_act_charts: Number of action charts in the atlas.
            act_codes_per_chart: Number of codes inside each action chart.
            hidden_dim: Hidden layer width for the internal encoders and
                routers.
            feature_scale: Scale factor applied inside the chart and code
                routers.
            use_residual_transition: If ``True``, learn a residual correction
                table on top of the geometry-based transition.
            residual_scale: Multiplier applied to the residual transition
                logits before they are added to the base logits.
            learn_reward: If ``True``, learn a reward look-up table indexed by
                ``(state, action)``.
            learn_continuation: If ``True``, learn a continuation-probability
                look-up table indexed by ``(state, action)``.
            initial_continuation: Initial continuation probability used to
                initialize the continuation logits via the inverse sigmoid.

        Returns:
            None.
        """
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
        """Validate shapes of state and action probability inputs.

        Args:
            state_probs: Tensor of shape ``[..., num_states]`` with soft state
                probabilities.
            action_probs: Tensor of shape ``[..., num_actions]`` with soft
                action probabilities.

        Returns:
            None. Raises ``ValueError`` if shapes are invalid or inconsistent.
        """
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
        """Validate that geometry dictionaries contain required keys.

        Args:
            obs_geometry: Observation geometry dictionary. Must contain at least
                ``"chart_centers"``, ``"codebook"``, and
                ``"state_tangent_points"``.
            act_geometry: Action geometry dictionary. Must contain at least
                ``"state_tangent_points"``.

        Returns:
            None. Raises ``ValueError`` if required keys are missing.
        """
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
        """Return the coarse continuation probability for each state-action pair.

        Returns:
            torch.Tensor | None: Tensor of shape ``[num_states, num_actions]``
                with sigmoid-transformed continuation probabilities, or ``None``
                if the model was initialized with ``learn_continuation=False``.
        """
        if self.continuation_logits is None:
            return None
        return torch.sigmoid(self.continuation_logits)

    def reward_from_probs(
        self,
        state_probs: torch.Tensor,
        action_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Return the expected coarse reward under soft state/action distributions.

        Args:
            state_probs: Tensor of shape ``[..., num_states]`` with soft state
                probabilities.
            action_probs: Tensor of shape ``[..., num_actions]`` with soft
                action probabilities.

        Returns:
            torch.Tensor: Scalar-per-batch expected reward of shape ``[...]``.
                Returns zeros if the model has no reward table.
        """
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
        """Return the expected continuation probability under soft state/action distributions.

        Args:
            state_probs: Tensor of shape ``[..., num_states]`` with soft state
                probabilities.
            action_probs: Tensor of shape ``[..., num_actions]`` with soft
                action probabilities.

        Returns:
            torch.Tensor: Scalar-per-batch expected continuation probability of
                shape ``[...]``. Returns ones if the model has no continuation
                table.
        """
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
        """Roll one coarse Markov step from soft symbolic state/action inputs.

        Encodes the current observation and action beliefs via their symbol
        tangent coordinates, predicts a next-observation query in the Poincare
        ball, and scores the next chart and code to produce the predicted
        next-state distribution.

        Args:
            state_probs: Tensor of shape ``[..., num_states]`` with soft state
                probabilities for the current time step.
            action_probs: Tensor of shape ``[..., num_actions]`` with soft
                action probabilities for the current time step.
            obs_geometry: Observation geometry dictionary (must include
                ``"chart_centers"``, ``"codebook"``, ``"state_tangent_points"``).
            act_geometry: Action geometry dictionary (must include
                ``"state_tangent_points"``).
            eps: Small constant for numerical stability.

        Returns:
            dict[str, torch.Tensor]: Dictionary with the following keys (all
                tensors preserve the leading batch shape ``[...]``):
                - ``"next_state_probs"``: shape ``[..., S]``.
                - ``"next_state_log_probs"``: shape ``[..., S]``.
                - ``"next_state_entropy"``: shape ``[...]``.
                - ``"next_state_top1_prob"``: shape ``[...]``.
                - ``"next_chart_probs"``: shape ``[..., N_c]``.
                - ``"next_chart_log_probs"``: shape ``[..., N_c]``.
                - ``"next_chart_entropy"``: shape ``[...]``.
                - ``"next_chart_top1_prob"``: shape ``[...]``.
                - ``"next_code_probs"``: shape ``[..., N_c, K]``.
                - ``"next_code_log_probs"``: shape ``[..., N_c, K]``.
                - ``"next_code_entropy"``: shape ``[...]``.
                - ``"next_code_top1_prob"``: shape ``[...]``.
                - ``"base_state_log_probs"``: shape ``[..., S]``.
                - ``"base_state_probs"``: shape ``[..., S]``.
                - ``"next_query_tangent"``: shape ``[..., D]``.
                - ``"next_query_point"``: shape ``[..., D]``.
                - ``"joint_context"``: shape ``[..., H]``.
                - ``"next_chart_logits_base"``: shape ``[..., N_c]``.
                - ``"next_code_logits_base"``: shape ``[..., N_c, K]``.
                - ``"next_local_query"``: shape ``[..., D]``.
                - ``"residual_transition_logits"`` (optional): shape ``[..., S]``.
                - ``"reward"`` (optional): shape ``[...]``.
                - ``"continuation"`` (optional): shape ``[...]``.
        """
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
            "next_code_top1_prob": (next_chart_probs * next_code_probs.max(dim=-1).values).sum(
                dim=-1
            ),
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
        """Read one conditional transition row from hard indices via one-hot inputs.

        Converts hard state and action indices to one-hot distributions and
        delegates to :meth:`forward`.

        Args:
            state_idx: Integer tensor of shape ``[...]`` with hard state indices
                in ``[0, num_states)``.
            action_idx: Integer tensor of shape ``[...]`` with hard action
                indices in ``[0, num_actions)``. Must match the shape of
                ``state_idx``.
            obs_geometry: Observation geometry dictionary (see :meth:`forward`).
            act_geometry: Action geometry dictionary (see :meth:`forward`).

        Returns:
            dict[str, torch.Tensor]: Same dictionary as :meth:`forward`.
        """
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
        """Roll the coarse model forward for a sequence of soft symbolic actions.

        Iteratively applies :meth:`forward` for each action in the horizon,
        feeding the predicted next-state distribution back as the current state.

        Args:
            state_probs_0: Tensor of shape ``[..., S]`` with the initial soft
                state distribution.
            action_probs_seq: Tensor of shape ``[..., H, A]`` with the soft
                action distributions for each of the ``H`` time steps.
            obs_geometry: Observation geometry dictionary (see :meth:`forward`).
            act_geometry: Action geometry dictionary (see :meth:`forward`).

        Returns:
            dict[str, torch.Tensor]: Dictionary with the following keys:
                - ``"state_probs"``: Tensor of shape ``[..., H+1, S]`` with the
                  state distribution at every step (including the initial one).
                - ``"next_state_probs"``: Tensor of shape ``[..., H, S]`` with
                  the predicted next-state distributions.
                - ``"next_state_entropy"``: Tensor of shape ``[..., H]`` with
                  the entropy of each predicted next-state distribution.
                - ``"next_state_top1_prob"``: Tensor of shape ``[..., H]`` with
                  the maximum probability in each predicted next-state distribution.
                - ``"reward"`` (optional): Tensor of shape ``[..., H]`` with the
                  expected reward at each step, present only if the model has a
                  reward table.
                - ``"continuation"`` (optional): Tensor of shape ``[..., H]``
                  with the expected continuation probability at each step,
                  present only if the model has a continuation table.
        """
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
    """Fit the coarse model on replay next-symbol supervision.

    Supervision can be provided either as a soft target distribution
    (``target_next_state_probs``) or as hard chart/code indices
    (``target_next_chart_idx``, ``target_next_code_idx``, ``codes_per_chart``).

    Args:
        model: The macro transition model to evaluate.
        state_probs: Tensor of shape ``[..., S]`` with soft current-state
            probabilities.
        action_probs: Tensor of shape ``[..., A]`` with soft current-action
            probabilities.
        obs_geometry: Observation geometry dictionary (see
            :class:`MacroTransitionModel`).
        act_geometry: Action geometry dictionary (see
            :class:`MacroTransitionModel`).
        target_next_state_probs: Optional tensor of shape ``[..., S]`` with the
            soft target next-state distribution. Mutually exclusive with the
            hard-index arguments.
        target_next_chart_idx: Optional integer tensor of shape ``[...]`` with
            the hard target chart index. Required (together with
            ``target_next_code_idx`` and ``codes_per_chart``) when
            ``target_next_state_probs`` is ``None``.
        target_next_code_idx: Optional integer tensor of shape ``[...]`` with
            the hard target code index.
        codes_per_chart: Number of codes per chart, required when using hard
            index targets.
        valid_mask: Optional boolean/float tensor broadcastable to the batch
            shape, with ones marking valid transitions.
        metric_prefix: Prefix prepended to all metric keys.
        eps: Small constant for numerical stability.

    Returns:
        tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]: A
            three-element tuple:
            - **loss** -- Scalar tensor with the masked mean factorized
              cross-entropy loss (chart CE + code CE).
            - **metrics** -- Dictionary of scalar float metrics keyed by
              ``"{metric_prefix}/{name}"`` including transition CE, state CE,
              chart/code CE, accuracies, entropies, and top-1 probabilities.
            - **pred** -- The full prediction dictionary returned by
              ``model.forward``.
    """
    pred = model(
        state_probs,
        action_probs,
        obs_geometry=obs_geometry,
        act_geometry=act_geometry,
        eps=eps,
    )
    flat_next_probs = pred["next_state_probs"].reshape(-1, model.num_states)
    flat_next_log_probs = pred["next_state_log_probs"].reshape(-1, model.num_states)
    pred["next_chart_probs"].reshape(-1, model.num_obs_charts)
    flat_next_chart_log_probs = pred["next_chart_log_probs"].reshape(-1, model.num_obs_charts)
    pred["next_code_probs"].reshape(-1, model.num_obs_charts, model.obs_codes_per_chart)
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
        flat_target_chart = torch.div(
            target_state, model.obs_codes_per_chart, rounding_mode="floor"
        )
        flat_target_code = target_state.remainder(model.obs_codes_per_chart)
        target_entropy = -(flat_target_probs * flat_target_probs.clamp(min=eps).log()).sum(dim=-1)
    else:
        if (
            target_next_chart_idx is None
            or target_next_code_idx is None
            or codes_per_chart is None
        ):
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
            torch.arange(
                flat_next_code_log_probs.shape[0], device=flat_next_code_log_probs.device
            ),
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
        f"{metric_prefix}/transition_ce": float(
            _masked_mean(flat_transition_ce, flat_valid).detach()
        ),
        f"{metric_prefix}/state_ce": float(_masked_mean(flat_state_ce, flat_valid).detach()),
        f"{metric_prefix}/chart_ce": float(_masked_mean(flat_chart_ce, flat_valid).detach()),
        f"{metric_prefix}/code_ce": float(_masked_mean(flat_code_ce, flat_valid).detach()),
        f"{metric_prefix}/transition_acc": float(
            _masked_mean(
                (pred_state == target_state).to(flat_next_probs.dtype), flat_valid
            ).detach()
        ),
        f"{metric_prefix}/chart_acc": float(
            _masked_mean(
                (pred_chart == flat_target_chart).to(flat_next_probs.dtype), flat_valid
            ).detach()
        ),
        f"{metric_prefix}/code_acc": float(
            _masked_mean(
                (pred_code == flat_target_code).to(flat_next_probs.dtype), flat_valid
            ).detach()
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
    """Align one symbolic distribution to another with masked KL/CE metrics.

    Minimizes the forward KL divergence ``KL(teacher || student)`` over valid
    entries.

    Args:
        teacher_probs: Tensor of shape ``[..., S]`` with the target (teacher)
            probability distribution.
        student_probs: Tensor of shape ``[..., S]`` with the predicted
            (student) probability distribution. Must match the shape of
            ``teacher_probs``.
        valid_mask: Optional boolean/float tensor broadcastable to the batch
            shape, with ones marking valid entries.
        metric_prefix: Prefix prepended to all metric keys.
        detach_teacher: If ``True``, detach the teacher probabilities so that
            gradients only flow through the student.
        eps: Small constant for numerical stability in log and normalization.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A two-element tuple:
            - **loss** -- Scalar tensor with the masked mean KL divergence.
            - **metrics** -- Dictionary of scalar float metrics keyed by
              ``"{metric_prefix}/{name}"`` including alignment CE, KL,
              agreement rate, teacher entropy, and student entropy.
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
        f"{metric_prefix}/teacher_entropy": float(
            _masked_mean(teacher_entropy, flat_valid).detach()
        ),
        f"{metric_prefix}/student_entropy": float(
            _masked_mean(student_entropy, flat_valid).detach()
        ),
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
    """Use the coarse model as a teacher for the live atlas symbolization.

    Delegates to :func:`compute_distribution_alignment_loss` with the macro
    model's next-state distribution as teacher and the live symbolization's
    next-state distribution as student.

    Args:
        macro_next_state_probs: Tensor of shape ``[..., S]`` with the coarse
            model's predicted next-state distribution (teacher).
        live_next_state_probs: Tensor of shape ``[..., S]`` with the live
            atlas's next-state distribution (student).
        valid_mask: Optional boolean/float tensor broadcastable to the batch
            shape, with ones marking valid entries.
        detach_teacher: If ``True``, detach the teacher probabilities so
            gradients only flow through the student.
        metric_prefix: Prefix prepended to all metric keys.
        eps: Small constant for numerical stability.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A two-element tuple:
            - **loss** -- Scalar KL-divergence loss tensor.
            - **metrics** -- Dictionary of scalar float alignment metrics.
    """
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
    """Make the micro world model follow the coarse symbolic transition.

    Delegates to :func:`compute_distribution_alignment_loss` with the macro
    model's next-state distribution as teacher and the micro world model's
    next-state distribution as student.

    Args:
        macro_next_state_probs: Tensor of shape ``[..., S]`` with the coarse
            model's predicted next-state distribution (teacher).
        wm_next_state_probs: Tensor of shape ``[..., S]`` with the micro world
            model's predicted next-state distribution (student).
        valid_mask: Optional boolean/float tensor broadcastable to the batch
            shape, with ones marking valid entries.
        detach_teacher: If ``True``, detach the teacher probabilities so
            gradients only flow through the student.
        metric_prefix: Prefix prepended to all metric keys.
        eps: Small constant for numerical stability.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A two-element tuple:
            - **loss** -- Scalar KL-divergence loss tensor.
            - **metrics** -- Dictionary of scalar float alignment metrics.
    """
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
    """Fit the reward and continuation tables on detached symbolic replay targets.

    Computes weighted smooth-L1 loss for the reward table and binary
    cross-entropy loss for the continuation table, combining them into a
    single scalar.

    Args:
        model: The macro transition model whose reward and continuation tables
            are being trained.
        state_probs: Tensor of shape ``[..., S]`` with soft current-state
            probabilities.
        action_probs: Tensor of shape ``[..., A]`` with soft current-action
            probabilities.
        reward_target: Tensor of shape ``[...]`` with the ground-truth reward
            values.
        continuation_target: Tensor of shape ``[...]`` with the ground-truth
            continuation probabilities (values in ``[0, 1]``).
        weight_reward: Scalar weight applied to the reward loss term.
        weight_continuation: Scalar weight applied to the continuation loss
            term.
        metric_prefix: Prefix prepended to all metric keys.

    Returns:
        tuple[torch.Tensor, dict[str, float]]: A two-element tuple:
            - **total** -- Scalar tensor with the weighted sum of reward and
              continuation losses.
            - **metrics** -- Dictionary of scalar float metrics keyed by
              ``"{metric_prefix}/{name}"`` including individual losses and
              mean predictions/targets for both reward and continuation.
    """
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
            f"{metric_prefix}/continuation_target_mean": float(
                continuation_target.mean().detach()
            ),
        })
    else:
        metrics.update({
            f"{metric_prefix}/continuation_loss": 0.0,
            f"{metric_prefix}/continuation_pred_mean": 0.0,
            f"{metric_prefix}/continuation_target_mean": float(
                continuation_target.mean().detach()
            ),
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
