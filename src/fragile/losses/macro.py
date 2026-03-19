"""Alternative macro / enclosure losses using absolute structured states.

This module implements the "absolute-state" enclosure probe discussed in the
Dreamer notes:

- observation and action symbols live in different Poincare balls,
- hard routing selects one chart and one code per manifold,
- the probe conditions on the absolute structured state
  ``u = c_K ⊕ q_{K,k} ⊕ exp_0(z_n)``,
- observation and action textures are tested separately and jointly for
  dynamics leakage.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers.gauge import exp_map_zero, mobius_add, project_to_ball
from fragile.losses.old_macro import GradientReversalLayer


def _state_index(
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
    codes_per_chart: int,
) -> torch.Tensor:
    """Flatten ``(chart, code)`` into one symbolic-state class index.

    Args:
        chart_idx: Hard chart assignment indices of shape ``[B]``.
        code_idx: Hard code assignment indices of shape ``[B]``.
        codes_per_chart: Number of codes per chart, used as the stride
            when flattening the two-level index.

    Returns:
        torch.Tensor: Flat class indices of shape ``[B]``, computed as
            ``chart_idx * codes_per_chart + code_idx``.
    """
    return chart_idx.long() * int(codes_per_chart) + code_idx.long()


def _validate_hard_symbol_inputs(
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
    z_n: torch.Tensor | None = None,
) -> None:
    """Check that chart/code tensors describe hard-routed symbolic states.

    Validates shapes and dimensional consistency of all inputs required to
    compose an absolute symbolic or structured state.

    Args:
        chart_centers: Chart centers in absolute manifold coordinates
            with expected shape ``[num_charts, latent_dim]``.
        codebook: Chart-local code centers with expected shape
            ``[num_charts, codes_per_chart, latent_dim]``.
        chart_idx: Hard chart assignment indices with expected shape ``[B]``.
        code_idx: Hard code assignment indices with expected shape ``[B]``.
        z_n: Optional tangent-space nuisance coordinates with expected shape
            ``[B, latent_dim]``. When provided, its batch size and latent
            dimension are also validated.

    Returns:
        None. Raises ``ValueError`` if any shape or consistency check fails.
    """
    if chart_centers.dim() != 2:
        msg = "chart_centers must have shape [N_c, D]."
        raise ValueError(msg)
    if codebook.dim() != 3:
        msg = "codebook must have shape [N_c, K, D]."
        raise ValueError(msg)
    if (
        codebook.shape[0] != chart_centers.shape[0]
        or codebook.shape[-1] != chart_centers.shape[-1]
    ):
        msg = "codebook must agree with chart_centers on chart count and latent dimension."
        raise ValueError(msg)
    if chart_idx.dim() != 1 or code_idx.dim() != 1:
        msg = "chart_idx and code_idx must both have shape [B]."
        raise ValueError(msg)
    if chart_idx.shape[0] != code_idx.shape[0]:
        msg = "chart_idx and code_idx must have the same batch size."
        raise ValueError(msg)
    if z_n is not None:
        if z_n.dim() != 2:
            msg = "z_n must have shape [B, D]."
            raise ValueError(msg)
        if z_n.shape[0] != chart_idx.shape[0] or z_n.shape[1] != chart_centers.shape[1]:
            msg = "z_n must match the batch size and latent dimension implied by chart_centers."
            raise ValueError(msg)


def compose_absolute_macro_state(
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
) -> torch.Tensor:
    """Compose the hard-routed absolute macro symbol ``c_K ⊕ q_{K,k}``.

    Args:
        chart_centers: Chart centers in absolute manifold coordinates
            with shape ``[num_charts, latent_dim]``.
        codebook: Chart-local code centers with shape
            ``[num_charts, codes_per_chart, latent_dim]``.
        chart_idx: Hard chart assignments of shape ``[batch]``.
        code_idx: Hard code assignments of shape ``[batch]``.

    Returns:
        Absolute macro symbols with shape ``[batch, latent_dim]``.
    """
    _validate_hard_symbol_inputs(chart_centers, codebook, chart_idx, code_idx)

    device = chart_idx.device
    chart_centers_proj = project_to_ball(chart_centers).to(device=device)
    codebook_proj = project_to_ball(codebook).to(device=device)

    chart_idx_long = chart_idx.long()
    code_idx_long = code_idx.long()
    selected_chart = chart_centers_proj[chart_idx_long]
    selected_code = codebook_proj[chart_idx_long, code_idx_long]
    return project_to_ball(mobius_add(selected_chart, selected_code))


def compose_absolute_structured_state(
    chart_centers: torch.Tensor,
    codebook: torch.Tensor,
    chart_idx: torch.Tensor,
    code_idx: torch.Tensor,
    z_n: torch.Tensor,
) -> torch.Tensor:
    """Compose the no-texture structured state ``c_K ⊕ (q_{K,k} ⊕ exp_0(z_n))``.

    This matches the encoder-side composition more closely than a raw
    ``(chart, code)`` tuple because it converts the hard-routed local symbol and
    nuisance coordinate into one absolute point in that manifold's Poincare
    ball.

    Args:
        chart_centers: Chart centers in absolute manifold coordinates
            with shape ``[num_charts, latent_dim]``.
        codebook: Chart-local code centers with shape
            ``[num_charts, codes_per_chart, latent_dim]``.
        chart_idx: Hard chart assignments of shape ``[batch]``.
        code_idx: Hard code assignments of shape ``[batch]``.
        z_n: Tangent nuisance coordinates with shape ``[batch, latent_dim]``.

    Returns:
        Absolute structured states with shape ``[batch, latent_dim]``.
    """
    _validate_hard_symbol_inputs(chart_centers, codebook, chart_idx, code_idx, z_n=z_n)

    device = z_n.device
    dtype = z_n.dtype
    chart_centers_proj = project_to_ball(chart_centers).to(device=device, dtype=dtype)
    codebook_proj = project_to_ball(codebook).to(device=device, dtype=dtype)

    chart_idx_long = chart_idx.long().to(device=device)
    code_idx_long = code_idx.long().to(device=device)
    selected_chart = chart_centers_proj[chart_idx_long]
    selected_code = codebook_proj[chart_idx_long, code_idx_long]

    # z_n is stored in the encoder as a tangent-space coordinate, so we map it
    # to the ball before composing it with the hard-routed code center.
    local_state = mobius_add(selected_code, exp_map_zero(z_n))
    return project_to_ball(mobius_add(selected_chart, local_state))


def _make_probe_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Module:
    """Build the small MLP used by each enclosure-probe head.

    Constructs a two-layer MLP with ReLU activation and dropout between the
    hidden and output layers.

    Args:
        input_dim: Dimensionality of the input features.
        hidden_dim: Number of units in the hidden layer.
        output_dim: Number of output logits (typically the number of
            symbolic-state classes).
        dropout: Dropout probability applied after the ReLU activation.

    Returns:
        nn.Module: An ``nn.Sequential`` module implementing
            ``Linear -> ReLU -> Dropout -> Linear``.
    """
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class AbsoluteEnclosureProbe(nn.Module):
    """Adversarial probe using absolute structured states from both manifolds.

    The probe predicts the next observation symbolic state from the current
    observation/action structured states. Three texture-bearing heads are used:

    - ``obs_texture_probe``: baseline plus observation texture,
    - ``act_texture_probe``: baseline plus action texture,
    - ``joint_texture_probe``: baseline plus both textures.

    Comparing those heads against the baseline quantifies how much extra
    transition information each texture channel contains beyond the intended
    structured state.
    """

    def __init__(
        self,
        obs_struct_dim: int,
        act_struct_dim: int,
        obs_tex_dim: int,
        act_tex_dim: int,
        num_obs_charts: int,
        obs_codes_per_chart: int,
        hidden_dim: int = 128,
        alpha: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize the absolute enclosure probe.

        Creates a baseline probe head and three texture-augmented probe heads,
        each predicting the next observation symbolic state. Gradient reversal
        layers are applied to the texture inputs so that the encoder is trained
        adversarially to suppress dynamics-relevant information in textures.

        Args:
            obs_struct_dim: Dimensionality of observation structured states.
            act_struct_dim: Dimensionality of action structured states.
            obs_tex_dim: Dimensionality of observation texture residuals.
            act_tex_dim: Dimensionality of action texture residuals.
            num_obs_charts: Number of observation charts (manifold regions).
            obs_codes_per_chart: Number of discrete codes per observation
                chart. The total number of prediction classes is
                ``num_obs_charts * obs_codes_per_chart``.
            hidden_dim: Number of hidden units in each probe MLP.
            alpha: Gradient reversal scaling factor applied to texture inputs.
            dropout: Dropout probability used in each probe MLP.

        Returns:
            None.
        """
        super().__init__()
        self.obs_codes_per_chart = obs_codes_per_chart
        self.num_obs_states = num_obs_charts * obs_codes_per_chart

        self.obs_grl = GradientReversalLayer(alpha=alpha)
        self.act_grl = GradientReversalLayer(alpha=alpha)

        baseline_dim = obs_struct_dim + act_struct_dim
        self.baseline_probe = _make_probe_mlp(
            baseline_dim,
            hidden_dim,
            self.num_obs_states,
            dropout,
        )
        self.obs_texture_probe = _make_probe_mlp(
            baseline_dim + obs_tex_dim,
            hidden_dim,
            self.num_obs_states,
            dropout,
        )
        self.act_texture_probe = _make_probe_mlp(
            baseline_dim + act_tex_dim,
            hidden_dim,
            self.num_obs_states,
            dropout,
        )
        self.joint_texture_probe = _make_probe_mlp(
            baseline_dim + obs_tex_dim + act_tex_dim,
            hidden_dim,
            self.num_obs_states,
            dropout,
        )

    def forward(
        self,
        u_obs: torch.Tensor,
        u_act: torch.Tensor,
        obs_z_tex: torch.Tensor,
        act_z_tex: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Run the baseline and texture-bearing enclosure heads.

        Args:
            u_obs: Observation structured states of shape ``[batch, obs_struct_dim]``.
            u_act: Action structured states of shape ``[batch, act_struct_dim]``.
            obs_z_tex: Observation texture residuals of shape ``[batch, obs_tex_dim]``.
            act_z_tex: Action texture residuals of shape ``[batch, act_tex_dim]``.

        Returns:
            dict[str, torch.Tensor]: A dictionary with four keys, each mapping
                to a logit tensor of shape ``[batch, num_obs_states]``:

                - ``"baseline"``: Logits from the structured-state-only head.
                - ``"obs"``: Logits from the observation-texture-augmented head.
                - ``"act"``: Logits from the action-texture-augmented head.
                - ``"both"``: Logits from the joint-texture-augmented head.
        """
        if u_obs.dim() != 2 or u_act.dim() != 2 or obs_z_tex.dim() != 2 or act_z_tex.dim() != 2:
            msg = "All probe inputs must have shape [B, D]."
            raise ValueError(msg)
        batch_size = u_obs.shape[0]
        if (
            u_act.shape[0] != batch_size
            or obs_z_tex.shape[0] != batch_size
            or act_z_tex.shape[0] != batch_size
        ):
            msg = "All probe inputs must share the same batch size."
            raise ValueError(msg)

        baseline_input = torch.cat([u_obs, u_act], dim=-1)
        obs_tex_rev = self.obs_grl(obs_z_tex)
        act_tex_rev = self.act_grl(act_z_tex)

        return {
            "baseline": self.baseline_probe(baseline_input),
            "obs": self.obs_texture_probe(torch.cat([baseline_input, obs_tex_rev], dim=-1)),
            "act": self.act_texture_probe(torch.cat([baseline_input, act_tex_rev], dim=-1)),
            "both": self.joint_texture_probe(
                torch.cat([baseline_input, obs_tex_rev, act_tex_rev], dim=-1),
            ),
        }


def compute_absolute_enclosure_loss(
    probe: AbsoluteEnclosureProbe,
    *,
    obs_chart_centers: torch.Tensor,
    obs_codebook: torch.Tensor,
    obs_chart_t: torch.Tensor,
    obs_code_t: torch.Tensor,
    obs_z_n_t: torch.Tensor,
    obs_z_tex_t: torch.Tensor,
    act_chart_centers: torch.Tensor,
    act_codebook: torch.Tensor,
    act_chart_t: torch.Tensor,
    act_code_t: torch.Tensor,
    act_z_n_t: torch.Tensor,
    act_z_tex_t: torch.Tensor,
    obs_chart_tp1: torch.Tensor,
    obs_code_tp1: torch.Tensor,
    obs_codes_per_chart: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Compute enclosure losses using absolute observation/action structured states.

    The predicted target is the next observation symbolic state. The encoder-side
    adversarial loss averages the three texture-bearing heads; the detached probe
    loss averages the baseline plus those same three heads.

    Args:
        probe: The absolute-state enclosure probe.
        obs_chart_centers: Observation chart centers ``[N_obs, D_obs]``.
        obs_codebook: Observation codebook ``[N_obs, K_obs, D_obs]``.
        obs_chart_t: Current observation hard chart ids ``[B]``.
        obs_code_t: Current observation hard code ids ``[B]``.
        obs_z_n_t: Current observation nuisance tangent ``[B, D_obs]``.
        obs_z_tex_t: Current observation texture residual ``[B, D_obs_tex]``.
        act_chart_centers: Action chart centers ``[N_act, D_act]``.
        act_codebook: Action codebook ``[N_act, K_act, D_act]``.
        act_chart_t: Current action hard chart ids ``[B]``.
        act_code_t: Current action hard code ids ``[B]``.
        act_z_n_t: Current action nuisance tangent ``[B, D_act]``.
        act_z_tex_t: Current action texture residual ``[B, D_act_tex]``.
        obs_chart_tp1: Next observation hard chart ids ``[B]``.
        obs_code_tp1: Next observation hard code ids ``[B]``.
        obs_codes_per_chart: Optional override for the observation symbol count.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]]: A three-element
            tuple ``(loss_encoder, loss_probe, diagnostics)``:

            - ``loss_encoder`` (torch.Tensor): Scalar adversarial encoder loss,
              the mean cross-entropy over the three texture-bearing heads.
            - ``loss_probe`` (torch.Tensor): Scalar probe training loss, the
              mean cross-entropy over all four heads (baseline + three texture).
            - ``diagnostics`` (dict[str, float]): Monitoring metrics with keys:

              - ``"acc_base"``: Baseline head accuracy.
              - ``"acc_obs"``: Observation-texture head accuracy.
              - ``"acc_act"``: Action-texture head accuracy.
              - ``"acc_both"``: Joint-texture head accuracy.
              - ``"defect_acc_obs"``: Accuracy gain of obs texture over baseline.
              - ``"defect_acc_act"``: Accuracy gain of act texture over baseline.
              - ``"defect_acc_both"``: Accuracy gain of joint texture over baseline.
              - ``"ce_base"``: Baseline cross-entropy.
              - ``"ce_obs"``: Observation-texture cross-entropy.
              - ``"ce_act"``: Action-texture cross-entropy.
              - ``"ce_both"``: Joint-texture cross-entropy.
              - ``"defect_ce_obs"``: CE reduction from obs texture vs baseline.
              - ``"defect_ce_act"``: CE reduction from act texture vs baseline.
              - ``"defect_ce_both"``: CE reduction from joint texture vs baseline.
              - ``"loss_encoder"``: Detached encoder loss value.
              - ``"loss_probe"``: Detached probe loss value.
    """
    if obs_codes_per_chart is None:
        obs_codes_per_chart = probe.obs_codes_per_chart

    target = _state_index(obs_chart_tp1, obs_code_tp1, obs_codes_per_chart)
    u_obs = compose_absolute_structured_state(
        obs_chart_centers,
        obs_codebook,
        obs_chart_t,
        obs_code_t,
        obs_z_n_t,
    )
    u_act = compose_absolute_structured_state(
        act_chart_centers,
        act_codebook,
        act_chart_t,
        act_code_t,
        act_z_n_t,
    )

    logits_live = probe(u_obs, u_act, obs_z_tex_t, act_z_tex_t)
    ce_obs = F.cross_entropy(logits_live["obs"], target)
    ce_act = F.cross_entropy(logits_live["act"], target)
    ce_both = F.cross_entropy(logits_live["both"], target)
    loss_encoder = (ce_obs + ce_act + ce_both) / 3.0

    logits_det = probe(
        u_obs.detach(),
        u_act.detach(),
        obs_z_tex_t.detach(),
        act_z_tex_t.detach(),
    )
    ce_base_det = F.cross_entropy(logits_det["baseline"], target)
    ce_obs_det = F.cross_entropy(logits_det["obs"], target)
    ce_act_det = F.cross_entropy(logits_det["act"], target)
    ce_both_det = F.cross_entropy(logits_det["both"], target)
    loss_probe = (ce_base_det + ce_obs_det + ce_act_det + ce_both_det) / 4.0

    with torch.no_grad():
        acc_base = (logits_det["baseline"].argmax(dim=-1) == target).float().mean().item()
        acc_obs = (logits_det["obs"].argmax(dim=-1) == target).float().mean().item()
        acc_act = (logits_det["act"].argmax(dim=-1) == target).float().mean().item()
        acc_both = (logits_det["both"].argmax(dim=-1) == target).float().mean().item()

    diagnostics = {
        "acc_base": acc_base,
        "acc_obs": acc_obs,
        "acc_act": acc_act,
        "acc_both": acc_both,
        "defect_acc_obs": acc_obs - acc_base,
        "defect_acc_act": acc_act - acc_base,
        "defect_acc_both": acc_both - acc_base,
        "ce_base": float(ce_base_det.detach()),
        "ce_obs": float(ce_obs_det.detach()),
        "ce_act": float(ce_act_det.detach()),
        "ce_both": float(ce_both_det.detach()),
        "defect_ce_obs": float((ce_base_det - ce_obs_det).detach()),
        "defect_ce_act": float((ce_base_det - ce_act_det).detach()),
        "defect_ce_both": float((ce_base_det - ce_both_det).detach()),
        "loss_encoder": float(loss_encoder.detach()),
        "loss_probe": float(loss_probe.detach()),
    }

    return loss_encoder, loss_probe, diagnostics


def zeno_loss(
    w_t: torch.Tensor,
    w_t_prev: torch.Tensor,
    mode: str = "jsd",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Penalize rapid changes in the soft routing distribution.

    Measures the divergence between consecutive routing weight vectors to
    encourage temporal smoothness.

    Args:
        w_t: Current routing weights after softmax, of shape
            ``[B, num_charts]``. Must sum to 1 along the last dimension.
        w_t_prev: Previous-timestep routing weights after softmax, of shape
            ``[B, num_charts]``. Must sum to 1 along the last dimension.
        mode: Divergence measure to use. ``"kl"`` computes
            ``D_KL(w_t || w_{t-1})``; ``"jsd"`` computes the
            Jensen-Shannon divergence between the two distributions.
        eps: Small floor value clamped onto weights to prevent ``log(0)``.

    Returns:
        torch.Tensor: Scalar loss (0-dim tensor), the mean divergence over
            the batch.
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


__all__ = [
    "AbsoluteEnclosureProbe",
    "compose_absolute_macro_state",
    "compose_absolute_structured_state",
    "compute_absolute_enclosure_loss",
    "zeno_loss",
]
