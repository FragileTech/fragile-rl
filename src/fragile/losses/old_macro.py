"""Macro / closure / symbolic transition losses.

Enclosure probe, dynamics transition model, gradient reversal, zeno
smoothness, and symbolic (chart+code) Markov losses.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Gradient Reversal
# ---------------------------------------------------------------------------


class GradientReversalFunction(torch.autograd.Function):
    """Identity forward, negates gradients backward with alpha scaling."""

    @staticmethod
    def forward(ctx, x, alpha):
        # Forward is a pure pass-through; the only effect of GRL is in backward.
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        # Multiply by -alpha so the upstream encoder is optimized to *hurt*
        # this probe while the probe itself still learns normally.
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wraps GradientReversalFunction as an nn.Module."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.register_buffer("alpha", torch.tensor(alpha))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.alpha)


# ---------------------------------------------------------------------------
# Enclosure Probe
# ---------------------------------------------------------------------------


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

        # The current code index is embedded and treated like another context
        # input when predicting the next symbolic state.
        self.code_embed = nn.Embedding(codes_per_chart, chart_dim)

        full_in = chart_dim + chart_dim + action_dim + ztex_dim
        # "Full" probe sees z_tex through GRL. If it outperforms the baseline,
        # z_tex contains transition information the encoder should remove.
        self.full_probe = nn.Sequential(
            nn.Linear(full_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.num_states),
        )

        base_in = chart_dim + chart_dim + action_dim
        # Baseline probe gets everything except z_tex, so it measures how much
        # transition prediction is possible without the texture residual.
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
        # Embed the current discrete code so both probes condition on the same
        # symbolic state information at time t.
        code_e = self.code_embed(code_idx)
        # The GRL leaves values unchanged here, but flips the gradient sign when
        # the loss flows back into the encoder that produced z_tex.
        ztex_rev = self.grl(z_tex)
        full_input = torch.cat([chart_embed, code_e, action, ztex_rev], dim=-1)
        base_input = torch.cat([chart_embed, code_e, action], dim=-1)
        return self.full_probe(full_input), self.baseline_probe(base_input)


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

    # Flatten the next symbolic state into one class id so the probe can use a
    # single standard cross-entropy over all (chart, code) combinations.
    target = K_chart_tp1.long() * codes_per_chart + K_code_tp1.long()

    # -- Encoder loss: gradients flow through GRL into structure filter --
    logits_full, _ = probe(chart_embed_t, action_t, ztex_t, K_code_t)
    ce_full = F.cross_entropy(logits_full, target)
    loss_encoder = ce_full

    # -- Probe loss: train probe on detached inputs --
    logits_full_det, logits_base_det = probe(
        chart_embed_t.detach(),
        action_t.detach(),
        ztex_t.detach(),
        K_code_t.detach(),
    )
    ce_full_det = F.cross_entropy(logits_full_det, target)
    ce_base_det = F.cross_entropy(logits_base_det, target)
    loss_probe = ce_full_det + ce_base_det

    # -- Diagnostics --
    with torch.no_grad():
        # The gap between the full and baseline probe tells us how much extra
        # predictive signal the probe can squeeze out of z_tex.
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
    # Start with a weak adversary and ramp it up as the representation becomes
    # more stable; otherwise the probe can overpower the encoder too early.
    return max_alpha * step / warmup_steps


# ---------------------------------------------------------------------------
# Dynamics Transition Model
# ---------------------------------------------------------------------------


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
            # Default path: look up a learned embedding for the current discrete
            # dynamics code.
            code_e = self.code_embed(code_idx)
        else:
            # Optional path: callers can pass the quantized dynamics feature
            # vector directly instead of going through the embedding table.
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
    # As above, flatten the next (chart, code) pair into one categorical target.
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


# ---------------------------------------------------------------------------
# Markov / Zeno losses
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

    # Quantize every chart-local latent into the auxiliary dynamics codebook.
    # This codebook is separate from the frozen atlas symbols used in Phase 1.
    z_q_dyn_flat, K_code_dyn_flat, _, vq_dyn_loss = atlas_encoder.dynamics_vq(
        v_local_all.reshape(B * H, D),
        router_weights_all.reshape(B * H, router_weights_all.shape[-1]),
    )
    z_q_dyn_all = z_q_dyn_flat.reshape(B, H, D)
    K_code_dyn_all = K_code_dyn_flat.reshape(B, H)

    # We can only supervise transitions where we have both a t -> t+1 pair and
    # an action aligned with time t.
    n_transitions = min(H - 1, actions.shape[1])
    if n_transitions < 1:
        metrics["dyn_vq"] = vq_dyn_loss.item()
        return vq_dyn_loss, metrics, K_code_dyn_all

    # Predict the next symbolic state from the current macro chart embedding,
    # current action, and current dynamics code / code feature.
    trans_logits = dyn_trans_model(
        chart_embed_all[:, :n_transitions].reshape(B * n_transitions, chart_dim),
        actions[:, :n_transitions].reshape(B * n_transitions, action_dim),
        K_code_dyn_all[:, :n_transitions].reshape(B * n_transitions),
        code_features=z_q_dyn_all[:, :n_transitions].reshape(B * n_transitions, D),
    )
    # The supervision target is the *next* chart together with the *next*
    # dynamics code assigned by the auxiliary VQ model.
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
        (K_code_dyn_all[:, 1 : n_transitions + 1] != K_code_dyn_all[:, :n_transitions])
        .float()
        .mean()
    )
    metrics["dyn_code_flip_rate"] = code_flips.item()

    if n_transitions > 1:
        probs = F.softmax(trans_logits, dim=-1).reshape(B, n_transitions, -1)
        pred_states = probs.argmax(dim=-1)
        # These metrics describe how sharp and how jumpy the transition model's
        # own predictions are over consecutive steps.
        state_entropy = -(probs * probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
        metrics["dyn_state_entropy"] = state_entropy.item()
        metrics["dyn_state_max_prob"] = probs.max(dim=-1).values.mean().item()
        metrics["dyn_state_flip_rate"] = (
            (pred_states[:, 1:] != pred_states[:, :-1]).float().mean().item()
        )
        if zeno_weight > 0:
            # Zeno regularization compares adjacent predicted distributions,
            # penalizing large step-to-step changes in the model's beliefs.
            dyn_zeno = zeno_loss(
                probs[:, 1:].reshape(-1, probs.shape[-1]),
                probs[:, :-1].reshape(-1, probs.shape[-1]),
                mode=zeno_mode,
            )
            total = total + zeno_weight * dyn_zeno
            metrics["dyn_zeno"] = dyn_zeno.item()

    return total, metrics, K_code_dyn_all


# ---------------------------------------------------------------------------
# Symbolic transition (from train_dreamer.py)
# ---------------------------------------------------------------------------


def _state_index(
    chart_idx: torch.Tensor, code_idx: torch.Tensor, codes_per_chart: int
) -> torch.Tensor:
    """Flatten `(chart, code)` symbolic state indices."""
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one."""
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _symbolic_transition_supervision_losses(
    *,
    state_probs: torch.Tensor,
    code_probs: torch.Tensor,
    target_charts: torch.Tensor,
    target_codes: torch.Tensor,
    valid_mask: torch.Tensor,
    codes_per_chart: int,
    metric_prefix: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Return replay-supervised code and full-symbol losses for a symbolic transition."""
    # Collapse batch/time axes so the bookkeeping below can operate on one flat
    # list of transitions regardless of the original rollout shape.
    flat_state_probs = state_probs.reshape(-1, state_probs.shape[-1])
    flat_code_probs = code_probs.reshape(-1, code_probs.shape[-2], code_probs.shape[-1])
    flat_target_charts = target_charts.reshape(-1).long()
    flat_target_codes = target_codes.reshape(-1).long()
    flat_target_state = _state_index(flat_target_charts, flat_target_codes, codes_per_chart)
    flat_valid = valid_mask.reshape(-1).to(flat_state_probs)
    batch_idx = torch.arange(flat_target_state.shape[0], device=flat_state_probs.device)

    # `code_probs` is conditional on the chart, so first select the row for the
    # target chart, then read out the probability of the target code inside it.
    target_chart_code_probs = flat_code_probs[batch_idx, flat_target_charts]
    target_code_log_prob = (
        target_chart_code_probs
        .gather(
            1,
            flat_target_codes.unsqueeze(-1),
        )
        .squeeze(-1)
        .clamp(min=1e-8)
        .log()
    )
    # `state_probs` already parameterizes the full flattened symbolic state, so
    # here we can gather the target (chart, code) class directly.
    target_state_log_prob = (
        flat_state_probs
        .gather(
            1,
            flat_target_state.unsqueeze(-1),
        )
        .squeeze(-1)
        .clamp(min=1e-8)
        .log()
    )

    L_code = _masked_mean(-target_code_log_prob, flat_valid)
    L_symbol = _masked_mean(-target_state_log_prob, flat_valid)

    # Decode the argmax symbolic state back into chart/code pieces so metrics can
    # report whether failures come from the chart prediction, the code within a
    # chart, or the full joint state.
    code_pred_target_chart = target_chart_code_probs.argmax(dim=-1)
    symbol_pred = flat_state_probs.argmax(dim=-1)
    symbol_pred_chart = torch.div(symbol_pred, codes_per_chart, rounding_mode="floor")
    symbol_pred_code = symbol_pred.remainder(codes_per_chart)

    symbol_entropy = -(flat_state_probs * flat_state_probs.clamp(min=1e-8).log()).sum(dim=-1)
    metrics = {
        f"{metric_prefix}/L_code": float(L_code.detach()),
        f"{metric_prefix}/L_symbol": float(L_symbol.detach()),
        f"{metric_prefix}/code_nll": float(L_code.detach()),
        f"{metric_prefix}/symbol_nll": float(L_symbol.detach()),
        f"{metric_prefix}/code_acc": float(
            _masked_mean(
                (code_pred_target_chart == flat_target_codes).to(flat_state_probs.dtype),
                flat_valid,
            ).detach(),
        ),
        f"{metric_prefix}/chart_acc_from_symbol": float(
            _masked_mean(
                (symbol_pred_chart == flat_target_charts).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/symbol_acc": float(
            _masked_mean(
                (symbol_pred == flat_target_state).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/symbol_code_acc": float(
            _masked_mean(
                (symbol_pred_code == flat_target_codes).to(flat_state_probs.dtype), flat_valid
            ).detach(),
        ),
        f"{metric_prefix}/state_entropy": float(_masked_mean(symbol_entropy, flat_valid).detach()),
    }
    return L_code, L_symbol, metrics
