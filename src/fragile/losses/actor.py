"""Actor / policy losses for geometric Dreamer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


if TYPE_CHECKING:
    from fragile.rl.config import DreamerConfig


# ---------------------------------------------------------------------------
# Shared helpers (small, duplicated where needed across loss modules)
# ---------------------------------------------------------------------------


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over entries where ``mask`` is one.

    Args:
        values: Tensor of arbitrary shape containing the values to average.
        mask: Boolean or binary tensor of the same shape as ``values``
            indicating which entries to include in the mean.

    Returns:
        A scalar tensor with the masked mean. The denominator is clamped to
        a minimum of 1.0 to avoid division by zero when the mask is empty.
    """
    denom = mask.sum().clamp(min=1.0)
    return (values * mask).sum() / denom


def _metric_inverse_scale(metric: nn.Module, z: torch.Tensor) -> torch.Tensor:
    """Return the inverse conformal metric scale ``lambda(z)^{-2}``.

    Args:
        metric: A metric module that exposes a ``conformal_factor(z)`` method
            and an optional ``epsilon`` attribute used for numerical stability.
        z: Latent-space coordinates at which to evaluate the conformal factor.

    Returns:
        A tensor of the same shape as the conformal factor, containing the
        inverse squared conformal scale ``1 / (lambda(z)^2 + epsilon)``.
    """
    cf = metric.conformal_factor(z)
    epsilon = getattr(metric, "epsilon", 1e-8)
    return 1.0 / (cf.pow(2) + epsilon)


# ---------------------------------------------------------------------------
# Actor losses
# ---------------------------------------------------------------------------


def _actor_return_trust(
    config: DreamerConfig,
    *,
    chart_acc: float,
    force_rel_err: float,
    policy_sync_err: float,
    hodge_conservative: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify how much imagined return the actor is allowed to trust.

    Combines four independent quality signals into a single multiplicative
    trust coefficient in [0, 1] that gates the imagined-return gradient
    flowing into the actor.

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``num_charts``, ``actor_return_chart_acc_target``,
            ``actor_return_force_err_scale``, and
            ``actor_return_policy_sync_scale``.
        chart_acc: Current chart-classification accuracy (scalar).
        force_rel_err: Relative error of the learned force field (scalar).
        policy_sync_err: Synchronization error between on-policy and
            off-policy action distributions (scalar).
        hodge_conservative: Hodge-decomposition conservativeness score in
            [0, 1] measuring how close the value gradient is to being
            curl-free (scalar).
        template: An existing tensor whose device and dtype are used to
            create new tensors on the correct device.

    Returns:
        A tuple of (trust, trust_metrics) where:
            trust: Scalar tensor in [0, 1] representing the overall return
                trust coefficient.
            trust_metrics: Dictionary mapping metric names to float values
                for each constituent trust factor.
    """
    chance = 1.0 / max(int(config.num_charts), 1)
    chart_target = max(float(config.actor_return_chart_acc_target), chance + 1e-6)
    chart_acc_t = template.new_tensor(chart_acc)
    force_rel_err_t = template.new_tensor(force_rel_err)
    policy_sync_err_t = template.new_tensor(policy_sync_err)
    hodge_conservative_t = template.new_tensor(hodge_conservative)

    chart_factor = ((chart_acc_t - chance) / (chart_target - chance)).clamp(0.0, 1.0)
    force_factor = torch.exp(
        -float(config.actor_return_force_err_scale) * force_rel_err_t.clamp(min=0.0),
    )
    policy_sync_factor = torch.exp(
        -float(config.actor_return_policy_sync_scale) * policy_sync_err_t.clamp(min=0.0),
    )
    conservative_factor = hodge_conservative_t.clamp(0.0, 1.0)
    trust = (chart_factor * force_factor * policy_sync_factor * conservative_factor).clamp(
        0.0, 1.0
    )
    trust_metrics = {
        "actor/return_trust": float(trust.detach()),
        "actor/return_trust_chart": float(chart_factor.detach()),
        "actor/return_trust_force": float(force_factor.detach()),
        "actor/return_trust_sync": float(policy_sync_factor.detach()),
        "actor/return_trust_conservative_exact": float(conservative_factor.detach()),
    }
    return trust, trust_metrics


def _categorical_entropy_varentropy(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return categorical entropy and varentropy from logits.

    Computes the Shannon entropy and the variance of the surprisal
    (varentropy) for a categorical distribution parameterized by logits.

    Args:
        logits: Unnormalized log-probabilities of shape ``(..., K)`` where
            ``K`` is the number of categories. Softmax is applied along the
            last dimension.

    Returns:
        A tuple of (entropy, varentropy) where:
            entropy: Tensor of shape ``(...)`` with the Shannon entropy
                ``H = -sum_k p_k log p_k`` for each distribution.
            varentropy: Tensor of shape ``(...)`` with the variance of the
                surprisal ``Var[-log p] = sum_k p_k (-log p_k - H)^2``.
    """
    probs = F.softmax(logits, dim=-1)
    surprisal = -probs.clamp_min(1e-8).log()
    entropy = (probs * surprisal).sum(dim=-1)
    varentropy = (probs * (surprisal - entropy.unsqueeze(-1)).pow(2)).sum(dim=-1)
    return entropy, varentropy


def _actor_curiosity_closure_gate(
    config: DreamerConfig,
    *,
    obs_state_acc: float,
    enclosure_defect_acc: float,
    enclosure_defect_ce: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Gate curiosity by grounded symbolic closure rather than raw novelty.

    Produces a multiplicative gate in [0, 1] that suppresses curiosity-driven
    exploration when the symbolic enclosure constraints are poorly satisfied.

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``num_charts``, ``actor_curiosity_closure_acc_target``,
            ``actor_curiosity_enclosure_defect_acc_scale``, and
            ``actor_curiosity_enclosure_defect_ce_scale``.
        obs_state_acc: Observation-to-state classification accuracy (scalar).
        enclosure_defect_acc: Enclosure defect measured via accuracy (scalar).
        enclosure_defect_ce: Enclosure defect measured via cross-entropy
            (scalar).
        template: An existing tensor whose device and dtype are used to
            create new tensors on the correct device.

    Returns:
        A tuple of (closure_gate, metrics) where:
            closure_gate: Scalar tensor in [0, 1] gating the curiosity
                bonus.
            metrics: Dictionary mapping metric names to float values for
                each constituent closure factor.
    """
    chance = 1.0 / max(int(config.num_charts), 1)
    acc_target = max(float(config.actor_curiosity_closure_acc_target), chance + 1e-6)
    obs_state_acc_t = template.new_tensor(obs_state_acc)
    enclosure_defect_acc_t = template.new_tensor(enclosure_defect_acc)
    enclosure_defect_ce_t = template.new_tensor(enclosure_defect_ce)

    obs_factor = ((obs_state_acc_t - chance) / (acc_target - chance)).clamp(0.0, 1.0)
    defect_acc_factor = torch.exp(
        -float(config.actor_curiosity_enclosure_defect_acc_scale)
        * enclosure_defect_acc_t.clamp(min=0.0),
    )
    defect_ce_factor = torch.exp(
        -float(config.actor_curiosity_enclosure_defect_ce_scale)
        * enclosure_defect_ce_t.clamp(min=0.0),
    )
    closure_gate = (obs_factor * defect_acc_factor * defect_ce_factor).clamp(0.0, 1.0)
    metrics = {
        "actor/curiosity_closure_gate": float(closure_gate.detach()),
        "actor/curiosity_closure_obs_factor": float(obs_factor.detach()),
        "actor/curiosity_closure_defect_acc_factor": float(defect_acc_factor.detach()),
        "actor/curiosity_closure_defect_ce_factor": float(defect_ce_factor.detach()),
    }
    return closure_gate, metrics


def _exact_control_gate(
    config: DreamerConfig,
    *,
    exact_increment_abs_err: float,
    exact_increment_target_mean: float,
    on_policy_covector_align_abs_err: float,
    on_policy_covector_target_mean: float,
    on_policy_exact_covector_norm_mean: float,
    on_policy_stiffness_target: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify whether the exact conservative field is calibrated enough to drive control.

    Combines three quality signals -- exact-increment relative error,
    on-policy covector alignment relative error, and a calibration ratio --
    into a single multiplicative gate in [0, 1].

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``actor_return_exact_increment_rel_scale`` and
            ``actor_return_exact_covector_rel_scale``.
        exact_increment_abs_err: Absolute error of the exact value increment
            (scalar).
        exact_increment_target_mean: Mean of the target exact value increment
            used to compute relative error (scalar).
        on_policy_covector_align_abs_err: Absolute alignment error between the
            on-policy covector and its target (scalar).
        on_policy_covector_target_mean: Mean of the target on-policy covector
            used to compute relative error (scalar).
        on_policy_exact_covector_norm_mean: Mean norm of the exact covector
            evaluated on-policy (scalar).
        on_policy_stiffness_target: Target stiffness scale for calibration
            (scalar).
        template: An existing tensor whose device and dtype are used to
            create new tensors on the correct device.

    Returns:
        A tuple of (exact_control_gate, metrics) where:
            exact_control_gate: Scalar tensor in [0, 1] indicating
                readiness of the exact conservative field for control.
            metrics: Dictionary mapping metric names to float diagnostic
                values for each constituent factor and intermediate quantity.
    """
    exact_increment_rel = exact_increment_abs_err / max(abs(exact_increment_target_mean), 1e-6)
    on_policy_covector_rel = on_policy_covector_align_abs_err / max(
        abs(on_policy_covector_target_mean),
        1e-6,
    )
    calibration_ratio = on_policy_exact_covector_norm_mean / max(on_policy_stiffness_target, 1e-6)

    exact_increment_rel_t = template.new_tensor(exact_increment_rel)
    on_policy_covector_rel_t = template.new_tensor(on_policy_covector_rel)
    calibration_ratio_t = template.new_tensor(calibration_ratio)

    exact_increment_factor = torch.exp(
        -float(config.actor_return_exact_increment_rel_scale)
        * exact_increment_rel_t.clamp(min=0.0),
    )
    on_policy_covector_factor = torch.exp(
        -float(config.actor_return_exact_covector_rel_scale)
        * on_policy_covector_rel_t.clamp(min=0.0),
    )
    calibration_factor = calibration_ratio_t.clamp(0.0, 1.0)
    exact_control_gate = (
        exact_increment_factor * on_policy_covector_factor * calibration_factor
    ).clamp(0.0, 1.0)
    metrics = {
        "actor/exact_control_gate": float(exact_control_gate.detach()),
        "actor/exact_control_increment_rel": float(exact_increment_rel_t.detach()),
        "actor/exact_control_covector_rel": float(on_policy_covector_rel_t.detach()),
        "actor/exact_control_calibration_ratio": float(calibration_ratio_t.detach()),
        "actor/exact_control_increment_factor": float(exact_increment_factor.detach()),
        "actor/exact_control_covector_factor": float(on_policy_covector_factor.detach()),
        "actor/exact_control_calibration_factor": float(calibration_factor.detach()),
    }
    return exact_control_gate, metrics


def _macro_control_gate(
    config: DreamerConfig,
    *,
    macro_exact_increment_abs_err: float,
    macro_exact_increment_target_mean: float,
    macro_on_policy_pullback_abs_err: float,
    macro_on_policy_value_std: float,
    macro_on_policy_exact_increment_pred_abs_mean: float,
    macro_target_scale: float,
    template: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Certify whether the symbolic macro value is calibrated enough to stage actor RL.

    If all macro loss weights are zero the gate is trivially 1 (always open).
    Otherwise three quality signals -- macro increment relative error,
    pullback relative error, and a calibration ratio -- are combined into
    a multiplicative gate in [0, 1].

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``w_macro_value``, ``w_macro_exact_increment``,
            ``w_macro_pullback``, ``w_macro_on_policy_pullback``,
            ``actor_return_macro_increment_rel_scale``, and
            ``actor_return_macro_pullback_rel_scale``.
        macro_exact_increment_abs_err: Absolute error of the macro exact
            value increment (scalar).
        macro_exact_increment_target_mean: Mean of the target macro exact
            increment used to compute relative error (scalar).
        macro_on_policy_pullback_abs_err: Absolute pullback error measured
            on-policy (scalar).
        macro_on_policy_value_std: Standard deviation of the macro
            on-policy value predictions (scalar).
        macro_on_policy_exact_increment_pred_abs_mean: Mean absolute
            predicted exact increment on-policy (scalar).
        macro_target_scale: Target scale for normalizing pullback error and
            computing the calibration ratio (scalar).
        template: An existing tensor whose device and dtype are used to
            create new tensors on the correct device.

    Returns:
        A tuple of (macro_control_gate, metrics) where:
            macro_control_gate: Scalar tensor in [0, 1] indicating readiness
                of the macro value for actor reinforcement learning.
            metrics: Dictionary mapping metric names to float diagnostic
                values for each constituent factor and intermediate quantity.
    """
    if (
        max(
            float(config.w_macro_value),
            float(config.w_macro_exact_increment),
            float(config.w_macro_pullback),
            float(config.w_macro_on_policy_pullback),
        )
        <= 0.0
    ):
        one = template.new_ones(())
        metrics = {
            "actor/macro_control_gate": 1.0,
            "actor/macro_control_increment_rel": 0.0,
            "actor/macro_control_pullback_rel": 0.0,
            "actor/macro_control_calibration_ratio": 1.0,
            "actor/macro_control_signal_scale": 1.0,
            "actor/macro_control_increment_factor": 1.0,
            "actor/macro_control_pullback_factor": 1.0,
            "actor/macro_control_calibration_factor": 1.0,
        }
        return one, metrics
    macro_increment_rel = macro_exact_increment_abs_err / max(
        abs(macro_exact_increment_target_mean), 1e-6
    )
    macro_pullback_rel = macro_on_policy_pullback_abs_err / max(abs(macro_target_scale), 1e-6)
    macro_signal_scale = max(
        abs(macro_on_policy_exact_increment_pred_abs_mean),
        abs(macro_on_policy_value_std),
    )
    macro_calibration_ratio = macro_signal_scale / max(abs(macro_target_scale), 1e-6)

    macro_increment_rel_t = template.new_tensor(macro_increment_rel)
    macro_pullback_rel_t = template.new_tensor(macro_pullback_rel)
    macro_calibration_ratio_t = template.new_tensor(macro_calibration_ratio)

    macro_increment_factor = torch.exp(
        -float(config.actor_return_macro_increment_rel_scale)
        * macro_increment_rel_t.clamp(min=0.0),
    )
    macro_pullback_factor = torch.exp(
        -float(config.actor_return_macro_pullback_rel_scale) * macro_pullback_rel_t.clamp(min=0.0),
    )
    macro_calibration_factor = macro_calibration_ratio_t.clamp(0.0, 1.0)
    macro_control_gate = (
        macro_increment_factor * macro_pullback_factor * macro_calibration_factor
    ).clamp(0.0, 1.0)
    metrics = {
        "actor/macro_control_gate": float(macro_control_gate.detach()),
        "actor/macro_control_increment_rel": float(macro_increment_rel_t.detach()),
        "actor/macro_control_pullback_rel": float(macro_pullback_rel_t.detach()),
        "actor/macro_control_calibration_ratio": float(macro_calibration_ratio_t.detach()),
        "actor/macro_control_signal_scale": float(macro_signal_scale),
        "actor/macro_control_increment_factor": float(macro_increment_factor.detach()),
        "actor/macro_control_pullback_factor": float(macro_pullback_factor.detach()),
        "actor/macro_control_calibration_factor": float(macro_calibration_factor.detach()),
    }
    return macro_control_gate, metrics


def _actor_supervise_scale(
    config: DreamerConfig,
    *,
    epoch: int,
    actor_return_gate: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Schedule replay-action supervision as a bootstrap term, not a permanent target.

    Combines a warmup/decay epoch schedule with the return-trust gate to
    produce a supervision scale that starts high and is gradually reduced
    as the actor becomes self-sufficient.

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``actor_supervise_min_scale``, ``actor_supervise_warmup_epochs``,
            and ``actor_supervise_decay_epochs``.
        epoch: Current training epoch (zero-indexed).
        actor_return_gate: Scalar tensor in [0, 1] from the return-trust
            computation. Higher values mean the actor trusts its own
            imagined returns more, reducing supervision.

    Returns:
        A tuple of (scale_t, metrics) where:
            scale_t: Scalar tensor with the supervision scale factor,
                always at least ``actor_supervise_min_scale``.
            metrics: Dictionary mapping metric names to float diagnostic
                values including the overall scale, warmup component, and
                gate-derived component.
    """
    gate = actor_return_gate.detach().clamp(0.0, 1.0)
    min_scale = float(np.clip(config.actor_supervise_min_scale, 0.0, 1.0))
    warmup_epochs = max(int(config.actor_supervise_warmup_epochs), 0)
    decay_epochs = max(int(config.actor_supervise_decay_epochs), 0)
    if epoch < warmup_epochs:
        warmup_scale = 1.0
    elif decay_epochs <= 0:
        warmup_scale = min_scale
    else:
        decay_progress = min(max((epoch - warmup_epochs + 1) / decay_epochs, 0.0), 1.0)
        warmup_scale = 1.0 - (1.0 - min_scale) * decay_progress
    gate_scale = float(1.0 - gate)
    scale = max(min_scale, warmup_scale * gate_scale)
    scale_t = gate.new_tensor(scale)
    metrics = {
        "actor/supervise_scale": float(scale_t.detach()),
        "actor/supervise_warmup_scale": float(warmup_scale),
        "actor/supervise_gate_scale": gate_scale,
    }
    return scale_t, metrics


def _scheduled_sigma_motor(
    config: DreamerConfig,
    *,
    epoch: int,
    exact_control_gate: float,
) -> tuple[float, dict[str, float]]:
    """Schedule thermal motor exploration with optional exact-field-aware cooling.

    Linearly anneals the motor exploration noise from an initial sigma to a
    final sigma, gated by the slower of epoch progress and exact-control-gate
    progress so that noise is not reduced before the exact field is ready.

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``sigma_motor``, ``sigma_motor_init``,
            ``sigma_motor_anneal_epochs``, and
            ``sigma_motor_exact_gate_target``.
        epoch: Current training epoch (zero-indexed).
        exact_control_gate: Scalar in [0, 1] indicating readiness of the
            exact conservative field (from ``_exact_control_gate``).

    Returns:
        A tuple of (sigma, metrics) where:
            sigma: Float value of the scheduled motor exploration noise
                standard deviation.
            metrics: Dictionary mapping metric names to float diagnostic
                values including initial, final, and current sigma as well
                as epoch and exact progress fractions.
    """
    sigma_final = max(float(config.sigma_motor), 0.0)
    sigma_init = (
        float(config.sigma_motor_init) if float(config.sigma_motor_init) > 0.0 else sigma_final
    )
    anneal_epochs = max(int(config.sigma_motor_anneal_epochs), 0)
    epoch_progress = 1.0 if anneal_epochs <= 0 else min(max((epoch + 1) / anneal_epochs, 0.0), 1.0)
    exact_target = max(float(config.sigma_motor_exact_gate_target), 0.0)
    exact_progress = (
        1.0 if exact_target <= 0.0 else min(max(exact_control_gate / exact_target, 0.0), 1.0)
    )
    cooling_progress = min(epoch_progress, exact_progress)
    sigma = sigma_init + (sigma_final - sigma_init) * cooling_progress
    metrics = {
        "policy/sigma_motor": float(sigma),
        "policy/sigma_motor_init": float(sigma_init),
        "policy/sigma_motor_final": float(sigma_final),
        "policy/sigma_motor_epoch_progress": float(epoch_progress),
        "policy/sigma_motor_exact_progress": float(exact_progress),
    }
    return float(sigma), metrics


def _actor_old_policy_kl_losses(
    actor_out: dict[str, torch.Tensor],
    actor_old_out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Penalize large discrete policy changes relative to the previous actor.

    Computes forward KL divergences (old || new) for both the chart
    selection head and the per-chart code selection head, weighting the
    code KL by the old chart probabilities.

    Args:
        actor_out: Dictionary from the current actor forward pass. Must
            contain ``"action_chart_logits"`` of shape ``(B, num_charts)``
            and ``"action_code_logits"`` of shape
            ``(B, num_charts, num_codes)``.
        actor_old_out: Dictionary from the previous (frozen) actor forward
            pass with the same keys and shapes as ``actor_out``. Gradients
            are detached from these tensors.

    Returns:
        A tuple of (L_chart_kl, L_code_kl) where:
            L_chart_kl: Scalar tensor with the batch-mean KL divergence
                between old and new chart distributions.
            L_code_kl: Scalar tensor with the chart-probability-weighted,
                batch-mean KL divergence between old and new code
                distributions.
    """
    old_chart_probs = F.softmax(actor_old_out["action_chart_logits"].detach(), dim=-1)
    new_chart_log_probs = F.log_softmax(actor_out["action_chart_logits"], dim=-1)
    L_chart_kl = F.kl_div(new_chart_log_probs, old_chart_probs, reduction="batchmean")

    old_code_probs = F.softmax(actor_old_out["action_code_logits"].detach(), dim=-1)
    new_code_log_probs = F.log_softmax(actor_out["action_code_logits"], dim=-1)
    code_kl_per_chart = (
        old_code_probs * (old_code_probs.clamp_min(1e-8).log() - new_code_log_probs)
    ).sum(dim=-1)
    L_code_kl = (old_chart_probs * code_kl_per_chart).sum(dim=-1).mean()
    return L_chart_kl, L_code_kl


def _actor_state_metric(
    config: DreamerConfig,
    *,
    metric: nn.Module,
    state_z_geo: torch.Tensor,
    actor_out: dict[str, torch.Tensor],
    obs_z_n: torch.Tensor,
    target_chart_idx: torch.Tensor,
    target_code_idx: torch.Tensor,
    exact_covector: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, bool, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Build the diagonal state-space metric proxy used by the actor update.

    Constructs a diagonal preconditioner from a combination of the value
    Hessian proxy (exact covector squared), a scaled Fisher information
    diagonal, and a regularization epsilon. The resulting metric controls
    the effective step size in each latent dimension during the actor
    gradient update.

    Args:
        config: Dreamer configuration object supplying hyper-parameters such as
            ``actor_metric_fisher_scale`` and ``actor_metric_epsilon``.
        metric: Geometric metric module used to obtain the inverse conformal
            scale at the current latent state.
        state_z_geo: Latent state in geometric coordinates, shape ``(B, D)``.
            Used only for evaluating the conformal factor (detached).
        actor_out: Dictionary from the actor forward pass containing
            ``"action_chart_logits"`` of shape ``(B, num_charts)`` and
            ``"action_code_logits"`` of shape
            ``(B, num_charts, num_codes)``.
        obs_z_n: Latent observation tensor of shape ``(B, D)`` that
            requires grad; used to compute the score function via
            ``torch.autograd.grad``.
        target_chart_idx: Integer tensor of shape ``(B,)`` with the
            ground-truth chart indices for each sample.
        target_code_idx: Integer tensor of shape ``(B,)`` with the
            ground-truth code indices for each sample.
        exact_covector: Tensor of shape ``(B, D)`` with the exact
            conservative covector field evaluated at the latent state.

    Returns:
        A tuple of (metric_diag, metric_inv, scale_certified, scale_trust,
        scale_barrier, metrics) where:
            metric_diag: Detached tensor of shape ``(D,)`` with the diagonal
                metric entries.
            metric_inv: Detached tensor of shape ``(D,)`` with the
                element-wise reciprocal of ``metric_diag``.
            scale_certified: Boolean indicating whether the value scale
                ``alpha`` exceeds the policy Fisher scale ``beta_pi``.
            scale_trust: Detached scalar tensor in [0, 1] derived from the
                scale barrier via ``exp(-barrier)``.
            scale_barrier: Detached scalar tensor measuring how much
                ``beta_pi`` exceeds ``alpha`` (clamped to non-negative).
            metrics: Dictionary mapping metric names to float diagnostic
                values.
    """
    sample_idx = torch.arange(obs_z_n.shape[0], device=obs_z_n.device)
    chart_log_probs = F.log_softmax(actor_out["action_chart_logits"], dim=-1)
    chart_logp = chart_log_probs[sample_idx, target_chart_idx.long()]
    code_logits = actor_out["action_code_logits"][sample_idx, target_chart_idx.long()]
    code_log_probs = F.log_softmax(code_logits, dim=-1)
    code_logp = code_log_probs[sample_idx, target_code_idx.long()]
    log_prob = chart_logp + code_logp
    state_score = torch.autograd.grad(
        log_prob.sum(),
        obs_z_n,
        retain_graph=True,
        create_graph=False,
    )[0]
    metric_scale = _metric_inverse_scale(metric, state_z_geo.detach()).mean()
    fisher_diag = metric_scale * state_score.detach().pow(2).mean(dim=0)
    fisher_scale = max(float(config.actor_metric_fisher_scale), 0.0)
    fisher_diag_scaled = fisher_scale * fisher_diag
    value_diag = metric_scale * exact_covector.detach().pow(2).mean(dim=0)
    metric_diag = value_diag + fisher_diag_scaled + float(config.actor_metric_epsilon)
    metric_inv = metric_diag.reciprocal()
    alpha = value_diag.mean().sqrt()
    beta_pi_raw = fisher_diag.mean().sqrt()
    beta_pi = fisher_diag_scaled.mean().sqrt()
    scale_barrier = (beta_pi - alpha).clamp(min=0.0) / (beta_pi + 1e-8)
    scale_trust = torch.exp(-scale_barrier)
    scale_certified = bool(float(alpha.detach()) > float(beta_pi.detach()))
    metrics = {
        "actor/state_alpha": float(alpha.detach()),
        "actor/state_beta_pi": float(beta_pi.detach()),
        "actor/state_beta_pi_raw": float(beta_pi_raw.detach()),
        "actor/state_scale_barrier": float(scale_barrier.detach()),
        "actor/state_scale_trust": float(scale_trust.detach()),
        "actor/state_metric_mean": float(metric_diag.mean().detach()),
        "actor/state_metric_inv_mean": float(metric_inv.mean().detach()),
        "actor/state_scale_certified": 1.0 if scale_certified else 0.0,
    }
    return (
        metric_diag.detach(),
        metric_inv.detach(),
        scale_certified,
        scale_trust.detach(),
        scale_barrier.detach(),
        metrics,
    )


def _relative_trust_region_scale(
    optimizer: torch.optim.Optimizer,
    parameters: list[torch.nn.Parameter],
    *,
    kappa: float,
    epsilon_theta: float,
) -> tuple[float, float, float, float]:
    """Apply the parameter-space Mach limit to the current gradients.

    Computes the ratio of the proposed gradient step norm to the maximum
    allowed step norm (``kappa * (||theta|| + epsilon_theta)``). If the
    step would exceed the limit, all parameter gradients are scaled down
    in-place so the step respects the trust region.

    Args:
        optimizer: The optimizer whose first param-group learning rate
            determines the base step size.
        parameters: List of model parameters to inspect. Only parameters
            with non-None ``.grad`` attributes are considered.
        kappa: Maximum allowed ratio of step norm to parameter norm
            (the "Mach number"). If non-positive, no clipping is applied.
        epsilon_theta: Small additive constant to the parameter norm to
            prevent a degenerate trust region when parameters are near zero.

    Returns:
        A tuple of (scale, param_norm, step_norm, max_step) where:
            scale: Float multiplicative factor applied to all gradients.
                Equals 1.0 when no clipping is needed or when kappa <= 0.
            param_norm: Float L2 norm of all parameters that have gradients.
            step_norm: Float L2 norm of the proposed gradient step
                (``grad_norm * lr``).
            max_step: Float maximum allowed step norm
                (``kappa * (param_norm + epsilon_theta)``).
    """
    grads = [param for param in parameters if param.grad is not None]
    if not grads or kappa <= 0.0:
        return 1.0, 0.0, 0.0, 0.0
    grad_norm_t = torch.norm(torch.stack([param.grad.detach().norm() for param in grads]), p=2)
    param_norm_t = torch.norm(torch.stack([param.detach().norm() for param in grads]), p=2)
    base_lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
    step_norm_t = grad_norm_t * base_lr
    max_step_t = float(kappa) * (param_norm_t + float(epsilon_theta))
    scale = min(1.0, float((max_step_t / (step_norm_t + 1e-12)).detach()))
    if scale < 1.0:
        for param in grads:
            param.grad.mul_(scale)
    return (
        scale,
        float(param_norm_t.detach()),
        float(step_norm_t.detach()),
        float(max_step_t.detach()),
    )
