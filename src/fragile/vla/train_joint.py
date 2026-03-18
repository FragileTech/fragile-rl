"""Joint encoder + world model training with 3-phase structure.

Combines the detailed loss display from ``train_unsupervised.py`` with the
Boris-BAOAB world model from ``train.py``.  Each phase's epoch count is a
CLI arg (set to 0 to skip).

Phase 1: Encoder warmup (single frames, 14 losses)
Phase 2: World model warmup (encoder frozen, dynamics losses)
Phase 3: Joint fine-tuning (encoder + world model, combined losses)
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from fragile.checkpoints import (
    compute_grad_norm,
    compute_param_norm,
    count_parameters,
)
from fragile.layers import FactorizedJumpOperator
from fragile.layers.gauge import hyperbolic_distance, mobius_add, project_to_ball
from fragile.layers.jump_operator import compute_jump_consistency_loss
from fragile.layers.topoencoder import TopoEncoder
from fragile.losses.encoder import (
    _deterministic_st_router_weights,
    compute_phase1_loss,
    compute_router_information_metrics,
    compute_router_score_metrics,
    compute_router_sharpness_metrics,
    get_jump_weight_schedule,
    orthogonality_loss,
)
from fragile.losses.old_macro import (
    compute_dyn_transition_loss,
    compute_dynamics_markov_loss,
    compute_enclosure_loss,
    DynamicsTransitionModel,
    EnclosureProbe,
    grl_alpha_schedule,
    zeno_loss,
)
from fragile.losses.world_model import (
    compute_phase2_geodesic_diffusion_loss,
    compute_phase2_loss,
)
from fragile.vla.config import VLAConfig
from fragile.vla.covariant_world_model import GeometricWorldModel
from fragile.vla.extract_features import VLAFeatureDataset
from fragile.vla.optim import (
    build_encoder_param_groups,
    get_codebook_like_params,
)
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    Phase1AdaptiveState,
    update_phase1_adaptive_state,
)


# ── Tracked loss terms ────────────────────────────────────────────
ENCODER_LOSS_KEYS = [
    "recon",
    "vq",
    "entropy",
    "consistency",
    "chart_usage",
    "chart_ot",
    "uniformity",
    "radial_cal",
    "confidence_calibration",
    "hard_routing_nll",
    "router_margin",
    "v_tangent_barrier",
    "codebook_spread",
    "codebook_center",
    "chart_center_mean",
    "chart_center_radius",
    "chart_center_sep",
    "code_usage",
    "window",
    "jump",
    "ortho",
]

ENCLOSURE_DIAG_KEYS = [
    "encl/acc_full",
    "encl/acc_base",
    "encl/defect_acc",
    "encl/defect_ce",
    "encl/ce_full",
    "encl/ce_base",
]

ZENO_DIAG_KEYS = [
    "zeno/flip_rate",
    "zeno/routing_entropy",
    "zeno/max_weight",
    "zeno/mean_segment_length",
]

DYNAMICS_LOSS_KEYS = [
    "geodesic",
    "chart_transition",
    "momentum_reg",
    "energy_conservation",
    "screened_poisson",
    "hodge",
    # Geodesic diffusion keys (only populated when use_geodesic_diffusion=True)
    "position",
    "endpoint",
    "momentum_target",
    "hodge_perp",
    "geo_miss",
]

GEODIFF_DIAG_KEYS = [
    "mean_momentum",
    "mean_phi_eff",
    "hodge_cons",
    "hodge_sol",
    "hodge_harm",
    "same_chart_frac",
    "chart_accuracy",
    "n_same_chart_pairs",
]

DYN_SYMBOL_KEYS = [
    "dyn_vq",
    "dyn_trans_ce",
    "dyn_trans_acc",
    "dyn_zeno",
    "dyn_state_flip_rate",
    "dyn_state_entropy",
    "dyn_state_max_prob",
    "dyn_code_flip_rate",
]

INFO_KEYS = [
    "I_XK",
    "H_K",
    "H_K_given_X",
    "ot_target_top1_mean",
    "ot_plan_col_l1",
    "ot_plan_row_l1",
    "H_usage",
    "usage_perplexity",
    "usage_active",
    "H_code_usage",
    "code_usage_perplexity",
    "active_code_charts",
    "top1_prob_mean",
    "top1_prob_p10",
    "top1_prob_p90",
    "top2_prob_mean",
    "top1_gap_mean",
    "score_gap_mean",
    "score_gap_p50",
    "score_gap_p90",
    "score_gap_p99",
    "score_std",
    "score_mean_abs",
    "soft_equiv_log_ratio",
    "recon_quality_mean",
    "vq_quality_mean",
    "combined_quality_mean",
    "routing_confidence_mean",
    "radial_target_mean",
    "local_radius_mean",
    "v_boundary_frac",
    "v_local_clip_frac",
    "z_geo_clip_frac",
    "v_raw_r_p99",
    "v_local_raw_r_p99",
    "z_geo_raw_r_p99",
    "router_grad_norm",
    "codebook_grad_norm",
    "centers_grad_norm",
    "val_proj_grad_norm",
    "soft_equiv_grad_norm",
    "grad_norm",
    "param_norm",
    "update_ratio",
    "lr",
]


def _init_encoder_accumulators() -> dict[str, float]:
    return dict.fromkeys(ENCODER_LOSS_KEYS + INFO_KEYS + ["total"], 0.0)


BALL_MAX_NORM = 0.99


def _tensor_norms(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    t = tensor.detach()
    if t.numel() == 0:
        return None
    if t.ndim == 0:
        return t.reshape(1).float()
    return t.norm(dim=-1).reshape(-1).float()


def _norm_quantile(norms: torch.Tensor | None, q: float) -> float:
    if norms is None or norms.numel() == 0:
        return 0.0
    return float(torch.quantile(norms, q).item())


def _clip_fraction(norms: torch.Tensor | None, max_norm: float = BALL_MAX_NORM) -> float:
    if norms is None or norms.numel() == 0:
        return 0.0
    return float((norms > (max_norm + 1e-6)).float().mean().item())


def _boundary_fraction(
    norms: torch.Tensor | None,
    max_norm: float = BALL_MAX_NORM,
    atol: float = 1e-3,
) -> float:
    if norms is None or norms.numel() == 0:
        return 0.0
    return float((norms >= (max_norm - atol)).float().mean().item())


def _safe_grad_norm(params: list[torch.nn.Parameter]) -> float:
    grads = [p.grad.detach() for p in params if p is not None and p.grad is not None]
    if not grads:
        return 0.0
    total = torch.zeros((), device=grads[0].device)
    for grad in grads:
        total = total + (grad**2).sum()
    return float(torch.sqrt(total).item())


def _phase1_grad_breakdown(model: TopoEncoder) -> dict[str, float]:
    encoder = model.encoder
    router_params: list[torch.nn.Parameter] = []
    if getattr(encoder, "cov_router", None) is not None:
        router_params.extend(list(encoder.cov_router.parameters()))
    elif getattr(encoder, "key_proj", None) is not None:
        router_params.extend(list(encoder.key_proj.parameters()))
        if getattr(encoder, "chart_queries", None) is not None:
            router_params.append(encoder.chart_queries)

    val_proj_params = list(encoder.val_proj.parameters())
    if getattr(encoder, "val_proj_scale", None) is not None:
        val_proj_params.append(encoder.val_proj_scale)

    soft_equiv_params: list[torch.nn.Parameter] = []
    if getattr(encoder, "soft_equiv_layers", None) is not None:
        soft_equiv_params.extend(list(encoder.soft_equiv_layers.parameters()))

    return {
        "router_grad_norm": _safe_grad_norm(router_params),
        "codebook_grad_norm": _safe_grad_norm([encoder.codebook]),
        "centers_grad_norm": _safe_grad_norm([encoder.chart_centers]),
        "val_proj_grad_norm": _safe_grad_norm(val_proj_params),
        "soft_equiv_grad_norm": _safe_grad_norm(soft_equiv_params),
    }


def _phase1_debug_metrics(model: TopoEncoder) -> dict[str, float]:
    encoder = model.encoder
    router_scores = getattr(encoder, "_last_router_scores_live", None)
    if router_scores is not None:
        score_metrics = {
            key: float(value.item())
            for key, value in compute_router_score_metrics(router_scores).items()
        }
    else:
        score_metrics = {
            "score_gap_mean": 0.0,
            "score_gap_p50": 0.0,
            "score_gap_p90": 0.0,
            "score_gap_p99": 0.0,
            "score_std": 0.0,
            "score_mean_abs": 0.0,
        }

    soft_equiv = 0.0
    if hasattr(encoder, "soft_equiv_log_ratio_loss"):
        soft_equiv = float(encoder.soft_equiv_log_ratio_loss().detach().item())

    v_raw_norms = _tensor_norms(getattr(encoder, "_last_v_raw", None))
    v_projected_norms = _tensor_norms(getattr(encoder, "_last_v_projected", None))
    v_local_raw_norms = _tensor_norms(getattr(encoder, "_last_v_local_raw", None))
    z_geo_raw_norms = _tensor_norms(getattr(encoder, "_last_z_geo_raw", None))

    return {
        **score_metrics,
        "soft_equiv_log_ratio": soft_equiv,
        "v_boundary_frac": _boundary_fraction(v_projected_norms),
        "v_clip_frac": _boundary_fraction(v_projected_norms),
        "v_local_clip_frac": _clip_fraction(v_local_raw_norms),
        "z_geo_clip_frac": _clip_fraction(z_geo_raw_norms),
        "v_raw_r_p99": _norm_quantile(v_raw_norms, 0.99),
        "v_local_raw_r_p99": _norm_quantile(v_local_raw_norms, 0.99),
        "z_geo_raw_r_p99": _norm_quantile(z_geo_raw_norms, 0.99),
    }


WM_DIAG_KEYS = [
    "mean_momentum",
    "energy_var",
    "jump_frac",
    "mean_phi_eff",
    "hodge_cons",
    "hodge_sol",
    "hodge_harm",
]


def _init_dynamics_accumulators() -> dict[str, float]:
    return dict.fromkeys(
        DYNAMICS_LOSS_KEYS
        + DYN_SYMBOL_KEYS
        + WM_DIAG_KEYS
        + GEODIFF_DIAG_KEYS
        + ["grad_norm", "param_norm", "update_ratio", "lr", "total"],
        0.0,
    )


def _init_joint_accumulators() -> dict[str, float]:
    keys = (
        [f"enc/{k}" for k in ENCODER_LOSS_KEYS]
        + [f"dyn/{k}" for k in DYNAMICS_LOSS_KEYS]
        + [f"wm/{k}" for k in WM_DIAG_KEYS]
        + [f"gd/{k}" for k in GEODIFF_DIAG_KEYS]
        + INFO_KEYS
        + ["enc_total", "dyn_total", "total"]
        + ENCLOSURE_DIAG_KEYS
        + ["encl_encoder", "encl_probe"]
        + ZENO_DIAG_KEYS
        + ["zeno"]
    )
    return dict.fromkeys(keys, 0.0)


def _wm_diagnostics(wm_output: dict[str, torch.Tensor]) -> dict[str, float]:
    """Extract diagnostic scalars from world model output."""
    momenta = wm_output["momenta"]
    phi_eff = wm_output["phi_eff"]
    diag = {
        "mean_momentum": momenta.norm(dim=-1).mean().item(),
        "energy_var": wm_output.get("energy_var", torch.tensor(0.0)).item(),
        "jump_frac": wm_output["jumped"].float().mean().item() if "jumped" in wm_output else 0.0,
        "mean_phi_eff": phi_eff.mean().item(),
        "hodge_cons": 0.0,
        "hodge_sol": 0.0,
        "hodge_harm": 0.0,
    }
    if "hodge_conservative_ratio" in wm_output:
        diag["hodge_cons"] = wm_output["hodge_conservative_ratio"].mean().item()
        diag["hodge_sol"] = wm_output["hodge_solenoidal_ratio"].mean().item()
        diag["hodge_harm"] = wm_output["hodge_harmonic_ratio"].mean().item()
    return diag


def _bind_world_model_to_encoder_atlas(
    model: TopoEncoder,
    world_model: GeometricWorldModel,
) -> None:
    """Bind the Phase-2 world model to the frozen Phase-1 chart atlas."""
    phase1_centers = getattr(model.encoder, "chart_centers", None)
    if phase1_centers is None:
        return
    world_model.bind_chart_centers(phase1_centers, freeze=True)


def _chart_stats_from_tensor(
    K_all: torch.Tensor,
    num_charts: int,
) -> tuple[np.ndarray, float, int]:
    """Compute usage, perplexity, active count from hard chart assignments."""
    K_np = K_all.detach().cpu().reshape(-1).numpy()
    usage = np.zeros(num_charts)
    for c in K_np:
        usage[int(c)] += 1
    usage /= usage.sum() + 1e-8
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    return usage, perplexity, active


def _chart_stats_from_probs(
    router_weights: torch.Tensor,
    num_charts: int,
) -> tuple[np.ndarray, float, int]:
    """Compute usage, perplexity, active count from soft routing probabilities."""
    rw = router_weights.detach().cpu().reshape(-1, num_charts)
    usage = rw.mean(dim=0).numpy()
    usage /= usage.sum() + 1e-8
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    return usage, perplexity, active


# ── Hard-routing tau annealing ────────────────────────────────────


def _get_hard_routing_tau(args: argparse.Namespace, epoch: int, total_epochs: int) -> float:
    """Compute annealed hard-routing temperature for the current epoch.

    Linear anneal from ``args.hard_routing_tau`` to ``args.hard_routing_tau_end``
    over ``args.hard_routing_tau_anneal_epochs`` epochs (defaults to *total_epochs*).
    Returns constant ``args.hard_routing_tau`` when ``tau_end`` is None.
    """
    warmup_epochs = max(int(getattr(args, "hard_routing_warmup_epochs", 0) or 0), 0)
    effective_epoch = max(epoch - warmup_epochs, 0)
    effective_total_epochs = max(total_epochs - warmup_epochs, 1)
    if args.hard_routing_tau < 0:
        # Negative tau means deterministic straight-through argmax. Keep it
        # fixed unless the user explicitly opts into another negative endpoint.
        if args.hard_routing_tau_end is None or args.hard_routing_tau_end >= 0:
            return args.hard_routing_tau
    if args.hard_routing_tau_end is None:
        return args.hard_routing_tau
    anneal_epochs = args.hard_routing_tau_anneal_epochs or effective_total_epochs
    if anneal_epochs <= 0:
        return args.hard_routing_tau_end
    t = min(effective_epoch / anneal_epochs, 1.0)
    return args.hard_routing_tau + t * (args.hard_routing_tau_end - args.hard_routing_tau)


def _use_hard_routing(args: argparse.Namespace, epoch: int) -> bool:
    """Warm-start with a soft partition of unity, then continue to hard routing."""
    if not args.hard_routing:
        return False
    warmup_epochs = max(int(getattr(args, "hard_routing_warmup_epochs", 0) or 0), 0)
    return epoch >= warmup_epochs


def _phase1_config_from_args(
    args: argparse.Namespace,
    phase1_state: Phase1AdaptiveState | None = None,
) -> VLAConfig:
    """Build the shared Phase 1 loss config from CLI args."""
    scales = phase1_effective_weight_scales(args, phase1_state)
    return VLAConfig(
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
        w_feature_recon=args.w_recon,
        w_vq=args.w_vq,
        w_entropy=args.w_entropy * scales["entropy_scale"],
        w_diversity=args.w_diversity * scales["chart_usage_scale"],
        chart_usage_entropy_low=getattr(args, "chart_usage_h_low", None),
        chart_usage_entropy_high=getattr(args, "chart_usage_h_high", None),
        w_chart_ot=args.w_chart_ot * scales["chart_ot_scale"],
        chart_ot_epsilon=getattr(args, "chart_ot_epsilon", 0.05),
        chart_ot_iters=getattr(args, "chart_ot_iters", 20),
        w_uniformity=args.w_uniformity,
        w_radial_calibration=args.w_radial_cal,
        w_confidence_calibration=args.w_confidence_calibration,
        w_hard_routing_nll=args.w_hard_routing_nll,
        w_router_margin=args.w_router_margin,
        router_margin_target=args.router_margin_target,
        radial_quality_alpha=args.radial_quality_alpha,
        radial_vq_alpha=args.radial_vq_alpha,
        radial_quality_rank_mix=args.radial_quality_rank_mix,
        radial_recon_quality_weight=args.radial_recon_quality_weight,
        radial_quality_mix=args.radial_quality_mix,
        radial_quality_base_weight=args.radial_quality_base_weight,
        radial_calibration_rho_max=args.radial_calibration_rho_max,
        radial_calibration_band_width=args.radial_calibration_band_width,
        w_v_tangent_barrier=args.w_v_tangent_barrier,
        v_tangent_barrier_radius=args.v_tangent_barrier_radius,
        w_codebook_spread=args.w_codebook_spread,
        w_codebook_center=args.w_codebook_center,
        w_chart_center_mean=args.w_chart_center_mean,
        w_chart_center_radius=args.w_chart_center_radius,
        chart_center_radius_max=args.chart_center_radius_max,
        w_chart_center_sep=args.w_chart_center_sep,
        chart_center_sep_margin=args.chart_center_sep_margin,
        w_chart_collapse=args.w_chart_collapse,
        w_code_collapse=args.w_code_collapse * scales["code_usage_scale"],
        code_usage_entropy_low=getattr(args, "code_usage_h_low", None),
        code_usage_entropy_high=getattr(args, "code_usage_h_high", None),
        w_code_collapse_temperature=getattr(args, "code_usage_temperature", 1.0),
        w_window=args.w_window,
        w_window_eps_ground=getattr(args, "w_window_eps_ground", 0.1),
        w_consistency=args.w_consistency,
        lr_chart_centers_scale=args.lr_chart_centers_scale,
        lr_codebook_scale=args.lr_codebook_scale,
    )


# ── Encoder loss computation (same as train_unsupervised.py) ──────


def _compute_encoder_losses(
    x: torch.Tensor,
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    args: argparse.Namespace,
    epoch: int,
    routing_tau: float = 1.0,
    phase1_config: VLAConfig | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute all Phase 1 encoder losses with split encoder/decoder calls.

    Returns (base_loss, zn_reg_loss, metrics, z_geo, enc_w, K_chart,
             z_n, z_tex, c_bar, K_code, z_q_blended, v_local) so callers
    can reuse encoder outputs without re-encoding.
    """
    (
        K_chart,
        K_code,
        z_n,
        z_tex,
        enc_w,
        z_geo,
        vq_loss,
        indices,
        z_n_all,
        c_bar,
        v_local,
        z_q_blended,
    ) = model.encoder(x, routing_tau=routing_tau)

    # When hard routing is on, pass encoder weights to decoder so both use the
    # same one-hot assignment (matching TopoEncoder.forward).  Without
    # this the decoder draws an independent Gumbel sample, consistency loss
    # explodes, and training diverges.
    router_override = enc_w
    x_recon, dec_w, _aux_losses = model.decoder(
        z_geo,
        chart_index=None,
        router_weights=router_override,
        routing_tau=routing_tau,
    )
    usage_router_weights = enc_w
    router_scores_live = getattr(model.encoder, "_last_router_scores_live", None)
    if router_scores_live is not None:
        # Utilization losses should see the router's deterministic hard
        # preference; a single Gumbel draw can look balanced by noise.
        usage_router_weights = _deterministic_st_router_weights(router_scores_live)

    phase1_config = phase1_config or _phase1_config_from_args(args)
    base_loss, zn_reg_loss, metrics = compute_phase1_loss(
        x,
        x_recon,
        vq_loss,
        enc_w,
        dec_w,
        z_geo,
        model,
        phase1_config,
        c_bar=c_bar,
        v_local=v_local,
        usage_router_weights=usage_router_weights,
        indices_stack=indices,
        router_scores=getattr(model.encoder, "_last_router_scores_live", None),
    )
    metrics.update(_phase1_debug_metrics(model))

    current_jump_weight = get_jump_weight_schedule(
        epoch,
        warmup_end=args.w_jump_warmup,
        ramp_end=args.w_jump_ramp_end,
        final_weight=args.w_jump,
    )
    if current_jump_weight > 0:
        jump_loss, _jump_info = compute_jump_consistency_loss(z_n_all, enc_w, jump_op)
        zn_reg_loss = zn_reg_loss + current_jump_weight * jump_loss
    else:
        jump_loss = torch.zeros((), device=x.device)

    # Orthogonality loss
    ortho_loss = orthogonality_loss(z_n, z_tex)
    base_loss = base_loss + getattr(args, "w_perp", 0.01) * ortho_loss

    total = base_loss + zn_reg_loss

    metrics["jump"] = jump_loss.item()
    metrics["ortho"] = ortho_loss.item()
    metrics["jump_weight"] = current_jump_weight
    metrics["total"] = total.item()
    return (
        base_loss,
        zn_reg_loss,
        metrics,
        z_geo,
        enc_w,
        K_chart,
        z_n,
        z_tex,
        c_bar,
        K_code,
        z_q_blended,
        v_local,
    )


# ── Eval pass (chart usage, perplexity, mean_r) ──────────────────


def _eval_pass(
    model: TopoEncoder,
    loader: DataLoader,
    K: int,
    device: torch.device,
    *,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
) -> tuple[np.ndarray, float, int, np.ndarray, float, int, float, dict]:
    """Compute hard/soft chart stats, mean radius, and extra diagnostics."""
    model.eval()
    all_charts: list[torch.Tensor] = []
    all_soft_router_weights: list[torch.Tensor] = []
    all_router_scores: list[torch.Tensor] = []
    all_radii: list[torch.Tensor] = []
    all_v_raw_norms: list[torch.Tensor] = []
    all_v_projected_norms: list[torch.Tensor] = []
    all_v_local_raw_norms: list[torch.Tensor] = []
    all_z_geo_raw_norms: list[torch.Tensor] = []
    all_vq_dists: list[torch.Tensor] = []
    all_code_indices: list[torch.Tensor] = []
    all_soft_equiv: list[float] = []
    with torch.no_grad():
        for batch in loader:
            x = batch["feature"].to(device)
            eval_tau = -1.0 if hard_routing else hard_routing_tau
            (
                K_ch,
                _K_code,
                _z_n,
                _z_tex,
                enc_w,
                z_g,
                _vq_loss,
                indices,
                _z_n_all,
                _c_bar,
                v_local,
                _,
            ) = model.encoder(
                x,
                routing_tau=eval_tau,
            )
            all_charts.append(K_ch.cpu())
            soft_router_weights = getattr(model.encoder, "_last_soft_router_weights_live", None)
            if soft_router_weights is None:
                soft_router_weights = enc_w
            all_soft_router_weights.append(soft_router_weights.detach().cpu())
            router_scores = getattr(model.encoder, "_last_router_scores_live", None)
            if router_scores is not None:
                all_router_scores.append(router_scores.detach().cpu())
            all_radii.append(z_g.cpu().norm(dim=-1))
            v_raw_norms = _tensor_norms(getattr(model.encoder, "_last_v_raw", None))
            if v_raw_norms is not None:
                all_v_raw_norms.append(v_raw_norms.cpu())
            v_projected_norms = _tensor_norms(getattr(model.encoder, "_last_v_projected", None))
            if v_projected_norms is not None:
                all_v_projected_norms.append(v_projected_norms.cpu())
            v_local_raw_norms = _tensor_norms(getattr(model.encoder, "_last_v_local_raw", None))
            if v_local_raw_norms is not None:
                all_v_local_raw_norms.append(v_local_raw_norms.cpu())
            z_geo_raw_norms = _tensor_norms(getattr(model.encoder, "_last_z_geo_raw", None))
            if z_geo_raw_norms is not None:
                all_z_geo_raw_norms.append(z_geo_raw_norms.cpu())
            if hasattr(model.encoder, "soft_equiv_log_ratio_loss"):
                all_soft_equiv.append(
                    float(model.encoder.soft_equiv_log_ratio_loss().detach().cpu().item())
                )
            # Compute per-sample VQ distance (nearest code distance)
            codebook = project_to_ball(model.encoder.codebook)  # [N_c, K_codes, D]
            v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
            cb_exp = codebook.unsqueeze(0)  # [1, N_c, K_codes, D]
            diff = v_exp - cb_exp  # Euclidean approx for diagnostics
            dists_sq = (diff**2).sum(-1)  # [B, N_c, K_codes]
            min_dist = dists_sq.min(dim=-1).values  # [B, N_c]
            # Weight by router for per-sample nearest distance
            weighted_dist = (min_dist * soft_router_weights).sum(dim=-1)  # [B]
            all_vq_dists.append(weighted_dist.cpu())
            all_code_indices.append(indices.cpu())  # [B, N_c]

    charts_t = torch.cat(all_charts)
    charts_np = charts_t.numpy()
    router_weights_t = torch.cat(all_soft_router_weights)
    radii_np = torch.cat(all_radii).numpy()
    vq_dists_np = torch.cat(all_vq_dists).numpy()
    code_indices = torch.cat(all_code_indices)  # [N_total, N_c]
    router_scores_t = torch.cat(all_router_scores) if all_router_scores else None
    v_raw_norms_t = torch.cat(all_v_raw_norms) if all_v_raw_norms else None
    v_projected_norms_t = torch.cat(all_v_projected_norms) if all_v_projected_norms else None
    v_local_raw_norms_t = torch.cat(all_v_local_raw_norms) if all_v_local_raw_norms else None
    z_geo_raw_norms_t = torch.cat(all_z_geo_raw_norms) if all_z_geo_raw_norms else None

    usage = np.zeros(K)
    for c in charts_np:
        usage[int(c)] += 1
    usage /= usage.sum() + 1e-8

    hard_entropy = float(-np.sum(usage * np.log(usage + 1e-8)))
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    soft_usage, soft_perplexity, soft_active = _chart_stats_from_probs(
        router_weights_t,
        K,
    )
    soft_info = compute_router_information_metrics(router_weights_t)
    soft_sharpness = compute_router_sharpness_metrics(router_weights_t)
    if router_scores_t is not None:
        score_metrics = {
            key: float(value.item())
            for key, value in compute_router_score_metrics(router_scores_t).items()
        }
    else:
        score_metrics = {
            "score_gap_mean": 0.0,
            "score_gap_p50": 0.0,
            "score_gap_p90": 0.0,
            "score_gap_p99": 0.0,
            "score_std": 0.0,
            "score_mean_abs": 0.0,
        }
    mean_r = float(radii_np.mean())

    # Extra diagnostics
    codebook_raw_cpu = model.encoder.codebook.detach().cpu()
    codebook_cpu = project_to_ball(model.encoder.codebook).detach().cpu()
    cb_radii = codebook_cpu.norm(dim=-1)  # [N_c, K_codes]
    cb_raw_radii = codebook_raw_cpu.norm(dim=-1)
    chart_centers_raw_cpu = model.encoder.chart_centers.detach().cpu()
    chart_centers_cpu = project_to_ball(model.encoder.chart_centers).detach().cpu()
    cc_radii = chart_centers_cpu.norm(dim=-1)  # [N_c]
    cc_raw_radii = chart_centers_raw_cpu.norm(dim=-1)

    # Code utilization: how many unique codes are used per chart
    codes_per_chart = codebook_cpu.shape[1]
    unique_codes_per_chart = []
    code_entropy_per_chart = []
    code_perplexity_per_chart = []
    for c in range(K):
        mask = charts_t == c
        if mask.sum() > 0:
            codes_for_chart = code_indices[mask, c]
            codes_used = codes_for_chart.unique().numel()
            counts = torch.bincount(codes_for_chart, minlength=codes_per_chart).float()
            probs = counts / counts.sum().clamp(min=1.0)
            entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
            code_entropy_per_chart.append(entropy)
            code_perplexity_per_chart.append(float(np.exp(entropy)))
        else:
            codes_used = 0
            code_entropy_per_chart.append(0.0)
            code_perplexity_per_chart.append(1.0)
        unique_codes_per_chart.append(codes_used)

    active_chart_mask = usage > 0.01
    if np.any(active_chart_mask):
        code_entropy_mean_active = float(
            np.mean(np.array(code_entropy_per_chart)[active_chart_mask])
        )
        code_perplexity_mean_active = float(
            np.mean(np.array(code_perplexity_per_chart)[active_chart_mask])
        )
    else:
        code_entropy_mean_active = 0.0
        code_perplexity_mean_active = 1.0

    extra = {
        "hard_entropy": hard_entropy,
        "r_std": float(radii_np.std()),
        "r_min": float(radii_np.min()),
        "r_max": float(radii_np.max()),
        "r_p10": float(np.percentile(radii_np, 10)),
        "r_p90": float(np.percentile(radii_np, 90)),
        "vq_dist_mean": float(vq_dists_np.mean()),
        "vq_dist_std": float(vq_dists_np.std()),
        "vq_dist_p90": float(np.percentile(vq_dists_np, 90)),
        "vq_dist_p99": float(np.percentile(vq_dists_np, 99)),
        "vq_dist_max": float(vq_dists_np.max()),
        "cb_r_mean": float(cb_radii.mean()),
        "cb_r_std": float(cb_radii.std()),
        "cb_r_max": float(cb_radii.max()),
        "cb_raw_r_p99": _norm_quantile(cb_raw_radii.reshape(-1), 0.99),
        "cb_clip_frac": _clip_fraction(cb_raw_radii.reshape(-1)),
        "cc_r_mean": float(cc_radii.mean()),
        "cc_r_max": float(cc_radii.max()),
        "cc_raw_r_p99": _norm_quantile(cc_raw_radii.reshape(-1), 0.99),
        "cc_clip_frac": _clip_fraction(cc_raw_radii.reshape(-1)),
        "codes_per_chart": unique_codes_per_chart,
        "codes_per_chart_total": codes_per_chart,
        "code_entropy_per_chart": code_entropy_per_chart,
        "code_perplexity_per_chart": code_perplexity_per_chart,
        "code_entropy_mean_active": code_entropy_mean_active,
        "code_perplexity_mean_active": code_perplexity_mean_active,
        "soft_usage": soft_usage,
        "soft_perplexity": soft_perplexity,
        "soft_active": soft_active,
        "soft_H_K": float(soft_info["H_K"].item()),
        "soft_H_K_given_X": float(soft_info["H_K_given_X"].item()),
        "soft_I_XK": float(soft_info["I_XK"].item()),
        "soft_top1_prob_mean": float(soft_sharpness["top1_prob_mean"].item()),
        "soft_top1_prob_p10": float(soft_sharpness["top1_prob_p10"].item()),
        "soft_top1_prob_p90": float(soft_sharpness["top1_prob_p90"].item()),
        "soft_top2_prob_mean": float(soft_sharpness["top2_prob_mean"].item()),
        "soft_top1_gap_mean": float(soft_sharpness["top1_gap_mean"].item()),
        "soft_equiv_log_ratio": float(np.mean(all_soft_equiv)) if all_soft_equiv else 0.0,
        "v_boundary_frac": _boundary_fraction(v_projected_norms_t),
        "v_clip_frac": _boundary_fraction(v_projected_norms_t),
        "v_local_clip_frac": _clip_fraction(v_local_raw_norms_t),
        "z_geo_clip_frac": _clip_fraction(z_geo_raw_norms_t),
        "v_raw_r_p99": _norm_quantile(v_raw_norms_t, 0.99),
        "v_local_raw_r_p99": _norm_quantile(v_local_raw_norms_t, 0.99),
        "z_geo_raw_r_p99": _norm_quantile(z_geo_raw_norms_t, 0.99),
        **score_metrics,
    }
    return usage, perplexity, active, soft_usage, soft_perplexity, soft_active, mean_r, extra


# ── Minimum length scale measurement ─────────────────────────────


@torch.no_grad()
def _measure_min_length(
    model: TopoEncoder,
    seq_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> float:
    """Measure average consecutive geodesic distance d_hyp(z_t, z_{t+1}).

    This gives the *action resolution* — the typical geodesic distance
    the latent state moves in one time step.  Used as the minimum
    resolvable length scale ℓ_min for CFL-derived force / velocity /
    noise bounds.
    """
    model.eval()
    dists: list[torch.Tensor] = []
    for i, batch in enumerate(seq_loader):
        if i >= max_batches:
            break
        features = batch["features"].to(device)  # [B, H, D_feat]
        B, H, D_feat = features.shape
        flat = features.reshape(B * H, D_feat)
        _, _, _, _, _, z_flat, *_ = model.encoder(flat)
        z_seq = z_flat.reshape(B, H, -1)  # [B, H, D]
        # Consecutive geodesic distances
        for t in range(H - 1):
            d = hyperbolic_distance(z_seq[:, t, :], z_seq[:, t + 1, :])  # [B]
            dists.append(d)
    all_dists = torch.cat(dists)
    median_dist = float(all_dists.median().item())
    mean_dist = float(all_dists.mean().item())
    print(
        f"  ℓ_min measurement: median={median_dist:.4f}  mean={mean_dist:.4f}  "
        f"(from {all_dists.numel()} pairs)"
    )
    return median_dist


def _update_world_model_min_length(
    world_model: GeometricWorldModel,
    min_length: float,
) -> None:
    """Update world model CFL bounds from a measured minimum length scale."""
    import math as _math

    world_model.min_length = min_length
    dt = world_model.dt
    gamma = world_model.gamma
    c2 = world_model.c2
    D = world_model.latent_dim

    world_model.V_alg = min_length / dt
    world_model.F_max = 2.0 * gamma * min_length / dt
    denom = max(c2, 1e-12) * _math.sqrt(D) * dt
    world_model.cf_max = max(2.0 * min_length / denom, 2.0)

    print(f"  CFL bounds from ℓ_min={min_length:.4f}:")
    print(f"    F_max  = {world_model.F_max:.2f}  (force squashing)")
    print(f"    V_alg  = {world_model.V_alg:.2f}  (velocity squashing)")
    print(f"    cf_max = {world_model.cf_max:.2f}  (thermostat noise cap)")


# ── Phase 1: Encoder warmup ──────────────────────────────────────


def _run_phase1(
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    single_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
    seq_loader: DataLoader | None = None,
    resume_probe_state: dict | None = None,
) -> dict[str, float]:
    """Phase 1: Train encoder + jump_op.

    Enclosure and zeno losses are disabled in Phase 1 (they are redundant
    with the dual-codebook architecture).  Single-frame training only.
    """
    # Enclosure / zeno always off in Phase 1 — dual codebook makes them redundant
    w_enclosure = 0.0
    w_zeno = 0.0
    use_sequences = False

    print("\n" + "=" * 60)
    print("Phase 1: Encoder warmup (single frames)")
    print("=" * 60)

    optimizer = torch.optim.Adam(
        build_encoder_param_groups(
            model,
            jump_op,
            base_lr=args.lr,
            lr_chart_centers_scale=args.lr_chart_centers_scale,
            lr_codebook_scale=args.lr_codebook_scale,
        )
    )
    all_params = [param for group in optimizer.param_groups for param in group["params"]]
    scheduler = None
    if args.use_scheduler or args.phase1_cosine_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.phase1_epochs,
            eta_min=args.phase1_eta_min,
        )

    # Enclosure probe (if enabled)
    probe = None
    probe_optimizer = None
    if w_enclosure > 0 and use_sequences:
        probe = EnclosureProbe(
            chart_dim=args.latent_dim,
            ztex_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=args.num_charts,
            codes_per_chart=args.codes_per_chart,
            hidden_dim=getattr(args, "enclosure_probe_hidden_dim", 128),
            alpha=0.0,
        ).to(device)
        probe_optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=getattr(args, "enclosure_probe_lr", 3e-3),
        )
        if resume_probe_state is not None:
            if resume_probe_state.get("probe") is not None:
                probe.load_state_dict(resume_probe_state["probe"])
                print("  Restored enclosure probe weights from checkpoint")
            if resume_probe_state.get("probe_optimizer") is not None:
                probe_optimizer.load_state_dict(resume_probe_state["probe_optimizer"])
                print("  Restored enclosure probe optimizer from checkpoint")

    train_loader = seq_loader if use_sequences else single_loader
    K = args.num_charts
    last_metrics: dict[str, float] = {}

    # Build accumulator keys — always include enclosure/zeno slots
    def _init_p1_accumulators() -> dict[str, float]:
        keys = list(ENCODER_LOSS_KEYS) + list(INFO_KEYS) + ["total"]
        keys += [*list(ENCLOSURE_DIAG_KEYS), "encl_encoder", "encl_probe"]
        keys += [*list(ZENO_DIAG_KEYS), "zeno"]
        return dict.fromkeys(keys, 0.0)

    phase1_state = init_phase1_adaptive_state(args)

    for epoch in tqdm(range(args.phase1_epochs), desc="Phase 1"):
        model.train()
        jump_op.train()
        acc = _init_p1_accumulators()
        n_batches = 0
        current_hard_routing = _use_hard_routing(args, global_epoch_offset + epoch)
        current_tau = _get_hard_routing_tau(
            args,
            global_epoch_offset + epoch,
            total_epochs_all_phases,
        )
        phase1_config = _phase1_config_from_args(args, phase1_state)

        for batch in train_loader:
            if use_sequences:
                # --- Sequence-based training ---
                features = batch["features"].to(device)  # [B, H, D_feat]
                actions = batch["actions"].to(device)  # [B, H, A]
                B, H, D_feat = features.shape

                # Frame 0: full encoder losses (recon, VQ, etc.)
                (
                    base_loss,
                    zn_reg_loss,
                    metrics,
                    _z_geo_0,
                    enc_w_0,
                    K_ch_0,
                    zn_0,
                    ztex_0,
                    c_bar_0,
                    K_code_0,
                    _,
                    _,
                ) = _compute_encoder_losses(
                    features[:, 0, :],
                    model,
                    jump_op,
                    args,
                    epoch,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                    phase1_config=phase1_config,
                )

                # Encode remaining frames
                if H > 1:
                    rest = features[:, 1:, :].reshape(B * (H - 1), D_feat)
                    (
                        K_rest,
                        Kcode_rest,
                        zn_rest,
                        ztex_rest,
                        rw_rest,
                        z_rest,
                        _,
                        _,
                        _,
                        c_bar_rest,
                        _,
                        _,
                    ) = model.encoder(
                        rest,
                        hard_routing=current_hard_routing,
                        hard_routing_tau=current_tau,
                    )
                    z_rest.reshape(B, H - 1, -1)
                    K_rest = K_rest.reshape(B, H - 1)
                    Kcode_rest = Kcode_rest.reshape(B, H - 1)
                    zn_rest = zn_rest.reshape(B, H - 1, -1)
                    ztex_rest = ztex_rest.reshape(B, H - 1, -1)
                    c_bar_rest = c_bar_rest.reshape(B, H - 1, -1)
                    rw_rest_reshaped = rw_rest.reshape(B, H - 1, -1)
                    K_all = torch.cat([K_ch_0.unsqueeze(1), K_rest], dim=1)
                    Kcode_all = torch.cat([K_code_0.unsqueeze(1), Kcode_rest], dim=1)
                    torch.cat([zn_0.unsqueeze(1), zn_rest], dim=1)
                    ztex_all = torch.cat([ztex_0.unsqueeze(1), ztex_rest], dim=1)
                    c_bar_all = torch.cat([c_bar_0.unsqueeze(1), c_bar_rest], dim=1)
                    rw_all = torch.cat([enc_w_0.unsqueeze(1), rw_rest_reshaped], dim=1)
                else:
                    K_all = K_ch_0.unsqueeze(1)
                    Kcode_all = K_code_0.unsqueeze(1)
                    zn_0.unsqueeze(1)
                    ztex_all = ztex_0.unsqueeze(1)
                    c_bar_all = c_bar_0.unsqueeze(1)
                    rw_all = enc_w_0.unsqueeze(1)

                # Enclosure loss
                L_encl_encoder = None
                L_encl_probe = None
                encl_diag_avg = {}
                if probe is not None and H > 1:
                    global_step = epoch * len(train_loader) + n_batches
                    probe.grl.alpha.fill_(
                        grl_alpha_schedule(
                            global_step,
                            warmup_steps=getattr(args, "enclosure_grl_warmup_steps", 5000),
                            max_alpha=getattr(args, "enclosure_grl_max_alpha", 1.0),
                        )
                    )

                    encl_enc_losses = []
                    encl_probe_losses = []
                    all_encl_diag = []
                    for t in range(H - 1):
                        loss_enc_t, loss_probe_t, diag_t = compute_enclosure_loss(
                            probe=probe,
                            chart_embed_t=c_bar_all[:, t],
                            action_t=actions[:, t],
                            ztex_t=ztex_all[:, t],
                            K_chart_tp1=K_all[:, t + 1],
                            K_code_t=Kcode_all[:, t],
                            K_code_tp1=Kcode_all[:, t + 1],
                            codes_per_chart=args.codes_per_chart,
                        )
                        encl_enc_losses.append(loss_enc_t)
                        encl_probe_losses.append(loss_probe_t)
                        all_encl_diag.append(diag_t)

                    L_encl_encoder = torch.stack(encl_enc_losses).mean()
                    L_encl_probe = torch.stack(encl_probe_losses).mean()
                    for key in all_encl_diag[0]:
                        encl_diag_avg[f"encl/{key}"] = sum(d[key] for d in all_encl_diag) / len(
                            all_encl_diag
                        )

                # Zeno loss
                L_zeno = None
                zeno_diag = {}
                if w_zeno > 0 and H > 1:
                    zeno_mode = getattr(args, "zeno_mode", "jsd")
                    zeno_losses = []
                    for t in range(1, H):
                        zeno_t = zeno_loss(rw_all[:, t], rw_all[:, t - 1], mode=zeno_mode)
                        zeno_losses.append(zeno_t)
                    L_zeno = torch.stack(zeno_losses).mean()

                    with torch.no_grad():
                        flips = (K_all[:, 1:] != K_all[:, :-1]).float()
                        ent = -(rw_all * rw_all.clamp(min=1e-8).log()).sum(dim=-1)
                        zeno_diag["zeno/flip_rate"] = flips.mean().item()
                        zeno_diag["zeno/routing_entropy"] = ent.mean().item()
                        zeno_diag["zeno/max_weight"] = rw_all.max(dim=-1).values.mean().item()
                        seg_lens = []
                        for b in range(min(B, 8)):
                            charts_b = K_all[b].tolist()
                            seg = 1
                            for i in range(1, len(charts_b)):
                                if charts_b[i] == charts_b[i - 1]:
                                    seg += 1
                                else:
                                    seg_lens.append(seg)
                                    seg = 1
                            seg_lens.append(seg)
                        zeno_diag["zeno/mean_segment_length"] = (
                            sum(seg_lens) / len(seg_lens) if seg_lens else 0.0
                        )

                # Assemble total loss
                total = base_loss + zn_reg_loss
                if L_encl_encoder is not None:
                    total = total + w_enclosure * L_encl_encoder
                if L_zeno is not None:
                    total = total + w_zeno * L_zeno

                # Backward + step for encoder
                optimizer.zero_grad()
                total.backward()
                metrics.update(_phase1_grad_breakdown(model))
                grad_norm = compute_grad_norm(all_params)
                param_norm = compute_param_norm(all_params)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                optimizer.step()

                # Probe optimizer step (separate)
                if probe is not None and L_encl_probe is not None:
                    probe_optimizer.zero_grad()
                    L_encl_probe.backward()
                    probe_optimizer.step()

                # Accumulate
                if L_encl_encoder is not None:
                    acc["encl_encoder"] += L_encl_encoder.item()
                    acc["encl_probe"] += L_encl_probe.item()
                    for k, v in encl_diag_avg.items():
                        acc[k] += v
                if L_zeno is not None:
                    acc["zeno"] += L_zeno.item()
                    for k, v in zeno_diag.items():
                        acc[k] += v

            else:
                # --- Single-frame training (original path) ---
                x = batch["feature"].to(device)

                base_loss, zn_reg_loss, metrics, _, _, _, _, _, _, _, _, _ = (
                    _compute_encoder_losses(
                        x,
                        model,
                        jump_op,
                        args,
                        epoch,
                        routing_tau=current_tau,
                        phase1_config=phase1_config,
                    )
                )
                total = base_loss + zn_reg_loss

                optimizer.zero_grad()
                total.backward()
                metrics.update(_phase1_grad_breakdown(model))
                grad_norm = compute_grad_norm(all_params)
                param_norm = compute_param_norm(all_params)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
                optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0

            acc["total"] += metrics["total"]
            for k in ENCODER_LOSS_KEYS:
                acc[k] += metrics[k]
            for k in INFO_KEYS:
                if k in {"grad_norm", "param_norm", "update_ratio", "lr"}:
                    continue
                acc[k] += metrics.get(k, 0.0)
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase1_epochs - 1)
        need_eval = should_log or (phase1_state is not None)
        if need_eval:
            (
                hard_usage,
                hard_perplexity,
                hard_active,
                soft_usage,
                soft_perplexity,
                soft_active,
                mean_r,
                extra,
            ) = _eval_pass(
                model,
                single_loader,
                K,
                device,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )
        else:
            hard_usage = np.zeros(K, dtype=np.float64)
            hard_perplexity = 0.0
            hard_active = 0
            soft_usage = np.zeros(K, dtype=np.float64)
            soft_perplexity = 0.0
            soft_active = 0
            mean_r = 0.0
            extra = {}

        if should_log:
            print(
                f"P1 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"| LR: {acc['lr']:.2e} | tau: {current_tau:.3f}"
            )
            print(f"  Hard usage: {np.array2string(hard_usage, precision=2, separator=', ')}")
            print(f"  Soft usage: {np.array2string(soft_usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['recon']:.3f} vq={acc['vq']:.3f} "
                f"entropy={acc['entropy']:.3f} consist={acc['consistency']:.3f} "
                f"chart_use={acc['chart_usage']:.3f} "
                f"hard_nll={acc['hard_routing_nll']:.3f} "
                f"margin={acc['router_margin']:.3f}"
            )
            print(
                f"  Info: I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f}"
            )
            print(
                f"  Qual: recon_q={acc['recon_quality_mean']:.3f} "
                f"vq_q={acc['vq_quality_mean']:.3f} q={acc['combined_quality_mean']:.3f} "
                f"conf={acc['routing_confidence_mean']:.3f} "
                f"target_r={acc['radial_target_mean']:.3f} "
                f"local_r={acc['local_radius_mean']:.3f}"
            )
            print(
                f"  Sharp: top1={acc['top1_prob_mean']:.3f} "
                f"gap={acc['top1_gap_mean']:.3f} top2={acc['top2_prob_mean']:.3f} "
                f"p10={acc['top1_prob_p10']:.3f} p90={acc['top1_prob_p90']:.3f}"
            )
            print(
                f"  Logits: gap50={acc['score_gap_p50']:.3f} "
                f"gap90={acc['score_gap_p90']:.3f} gap99={acc['score_gap_p99']:.3f} "
                f"std={acc['score_std']:.3f} abs={acc['score_mean_abs']:.3f} "
                f"soft_equiv={acc['soft_equiv_log_ratio']:.3f}"
            )
            print(
                f"  Geo: unif={acc['uniformity']:.3f} "
                f"rad_cal={acc['radial_cal']:.3f} "
                f"conf_cal={acc['confidence_calibration']:.3f} "
                f"v_tan={acc['v_tangent_barrier']:.3f} "
                f"cb_spread={acc['codebook_spread']:.3f} "
                f"cb_center={acc['codebook_center']:.3f} "
                f"cc_mean={acc['chart_center_mean']:.3f} "
                f"cc_rad={acc['chart_center_radius']:.3f} "
                f"cc_sep={acc['chart_center_sep']:.3f}"
            )
            print(
                f"  Proj: v_boundary={acc['v_boundary_frac']:.3f} "
                f"v_local_clip={acc['v_local_clip_frac']:.3f} "
                f"z_geo_clip={acc['z_geo_clip_frac']:.3f} "
                f"raw_p99=({acc['v_raw_r_p99']:.3f}, {acc['v_local_raw_r_p99']:.3f}, "
                f"{acc['z_geo_raw_r_p99']:.3f})"
            )
            print(f"  Usage: code={acc['code_usage']:.4f}")
            print(
                f"  Train usage: H_hard={acc['H_usage']:.3f} "
                f"perp={acc['usage_perplexity']:.2f}/{K} "
                f"active={acc['usage_active']:.2f}/{K} "
                f"code_H={acc['H_code_usage']:.3f} "
                f"code_perp={acc['code_usage_perplexity']:.2f}/{args.codes_per_chart} "
                f"active_code_charts={acc['active_code_charts']:.2f}"
            )
            print(f"  Ortho: {acc['ortho']:.4f} (w={getattr(args, 'w_perp', 0.01):.3f})")
            if use_sequences:
                grl_alpha = probe.grl.alpha.item() if probe is not None else 0.0
                print(
                    f"  Enclosure: enc={acc.get('encl_encoder', 0):.4f} "
                    f"probe={acc.get('encl_probe', 0):.4f} "
                    f"defect_acc={acc.get('encl/defect_acc', 0):.4f} "
                    f"defect_ce={acc.get('encl/defect_ce', 0):.4f} "
                    f"grl_alpha={grl_alpha:.3f} "
                    f"(w={w_enclosure:.3f})"
                )
                print(
                    f"  Zeno: loss={acc.get('zeno', 0):.4f} "
                    f"flip_rate={acc.get('zeno/flip_rate', 0):.4f} "
                    f"seg_len={acc.get('zeno/mean_segment_length', 0):.2f} "
                    f"entropy={acc.get('zeno/routing_entropy', 0):.4f} "
                    f"max_w={acc.get('zeno/max_weight', 0):.4f} "
                    f"(w={w_zeno:.3f})"
                )
            print(f"  Window: {acc['window']:.3f} (w={args.w_window:.3f})")
            print(f"  Jump: {acc['jump']:.3f} (lambda={metrics.get('jump_weight', 0.0):.3f})")
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} lr={acc['lr']:.2e}"
            )
            print(
                f"  Grad parts: router={acc['router_grad_norm']:.2e} "
                f"val_proj={acc['val_proj_grad_norm']:.2e} "
                f"codebook={acc['codebook_grad_norm']:.2e} "
                f"centers={acc['centers_grad_norm']:.2e} "
                f"soft_equiv={acc['soft_equiv_grad_norm']:.2e}"
            )
            if phase1_state is not None:
                print(
                    f"  Ctrl: w_ent={phase1_config.w_entropy:.3f} "
                    f"w_chart={phase1_config.w_diversity:.3f} "
                    f"w_ot={phase1_config.w_chart_ot:.3f} "
                    f"w_code={phase1_config.w_code_collapse:.3f}"
                )
            print(
                f"  Metrics: hard_perplexity={hard_perplexity:.2f}/{K} "
                f"hard_active={hard_active}/{K} "
                f"soft_perplexity={soft_perplexity:.2f}/{K} "
                f"soft_active={soft_active}/{K} mean_r={mean_r:.3f}"
            )
            print(
                f"  Eval info: hard_H={extra['hard_entropy']:.3f} "
                f"soft_I_XK={extra['soft_I_XK']:.3f} "
                f"soft_H_K={extra['soft_H_K']:.3f} "
                f"soft_H_K|X={extra['soft_H_K_given_X']:.3f}"
            )
            print(
                f"  Eval sharp: top1={extra['soft_top1_prob_mean']:.3f} "
                f"gap={extra['soft_top1_gap_mean']:.3f} "
                f"top2={extra['soft_top2_prob_mean']:.3f} "
                f"p10={extra['soft_top1_prob_p10']:.3f} "
                f"p90={extra['soft_top1_prob_p90']:.3f}"
            )
            print(
                f"  Eval logits: gap50={extra['score_gap_p50']:.3f} "
                f"gap90={extra['score_gap_p90']:.3f} gap99={extra['score_gap_p99']:.3f} "
                f"std={extra['score_std']:.3f} abs={extra['score_mean_abs']:.3f} "
                f"soft_equiv={extra['soft_equiv_log_ratio']:.3f}"
            )
            print(
                f"  OT: loss={acc['chart_ot']:.3f} "
                f"target_top1={acc['ot_target_top1_mean']:.3f} "
                f"col_l1={acc['ot_plan_col_l1']:.3e}"
            )
            # Extra diagnostics
            print(
                f"  Radii: mean={mean_r:.3f} std={extra['r_std']:.3f} "
                f"[{extra['r_min']:.3f}, {extra['r_max']:.3f}] "
                f"p10={extra['r_p10']:.3f} p90={extra['r_p90']:.3f}"
            )
            print(
                f"  Eval geom: v_boundary={extra['v_boundary_frac']:.3f} "
                f"v_local_clip={extra['v_local_clip_frac']:.3f} "
                f"z_geo_clip={extra['z_geo_clip_frac']:.3f} "
                f"raw_p99=({extra['v_raw_r_p99']:.3f}, {extra['v_local_raw_r_p99']:.3f}, "
                f"{extra['z_geo_raw_r_p99']:.3f})"
            )
            print(
                f"  VQ dist: mean={extra['vq_dist_mean']:.4f} "
                f"std={extra['vq_dist_std']:.4f} p90={extra['vq_dist_p90']:.4f} "
                f"p99={extra['vq_dist_p99']:.4f} max={extra['vq_dist_max']:.4f}"
            )
            print(
                f"  Codebook: cb_r={extra['cb_r_mean']:.3f}±{extra['cb_r_std']:.3f} "
                f"(max={extra['cb_r_max']:.3f}) "
                f"centers_r={extra['cc_r_mean']:.3f} (max={extra['cc_r_max']:.3f}) "
                f"raw_p99=({extra['cb_raw_r_p99']:.3f}, {extra['cc_raw_r_p99']:.3f}) "
                f"clip=({extra['cb_clip_frac']:.3f}, {extra['cc_clip_frac']:.3f})"
            )
            print(
                f"  Code stats: H={extra['code_entropy_mean_active']:.3f} "
                f"perp={extra['code_perplexity_mean_active']:.2f}/{args.codes_per_chart}"
            )
            cpc = extra["codes_per_chart"]
            total_codes = extra["codes_per_chart_total"]
            print(f"  Code util: {cpc} / {total_codes} per chart")
            print("-" * 60)

        if phase1_state is not None:
            update_phase1_adaptive_state(
                phase1_state,
                args,
                train_metrics=acc,
                eval_metrics=extra,
                epoch=epoch,
            )

        should_save = (
            epoch > 0 and epoch % args.save_every == 0
        ) or epoch == args.phase1_epochs - 1
        if should_save:
            _save_checkpoint(
                args,
                model,
                jump_op,
                None,
                optimizer,
                scheduler,
                epoch,
                1,
                acc,
                probe=probe,
                probe_optimizer=probe_optimizer,
            )

        last_metrics = acc

    return last_metrics, probe


# ── Phase 2: World model warmup ──────────────────────────────────


def _run_phase2(
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel,
    seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
) -> dict[str, float]:
    """Phase 2: Train world model with frozen encoder."""
    print("\n" + "=" * 60)
    print("Phase 2: World model warmup (encoder frozen)")
    print("=" * 60)

    _bind_world_model_to_encoder_atlas(model, world_model)
    phase2_use_jump = world_model.use_jump
    world_model.use_jump = False
    if phase2_use_jump:
        print("  Phase 2 warmup disables active world-model jumps; learning transport only.")
    print("  Phase 2 atlas is bound to the frozen Phase 1 router/chart geometry.")

    # Freeze encoder
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for p in jump_op.parameters():
        p.requires_grad_(False)

    # Dynamics codebook: unfreeze if present
    dyn_codes = getattr(args, "dyn_codes_per_chart", 0)
    dyn_trans_model = None
    if dyn_codes > 0 and model.encoder.codebook_dyn is not None:
        model.encoder.codebook_dyn.requires_grad_(True)
        dyn_trans_model = DynamicsTransitionModel(
            chart_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=args.num_charts,
            dyn_codes_per_chart=dyn_codes,
            hidden_dim=getattr(args, "dyn_transition_hidden_dim", 128),
        ).to(device)
        lr_dyn = getattr(args, "lr_dyn_codebook", 1e-3)
        optimizer = torch.optim.Adam([
            {"params": world_model.parameters(), "lr": args.lr_wm},
            {"params": [model.encoder.codebook_dyn], "lr": lr_dyn},
            {"params": dyn_trans_model.parameters(), "lr": lr_dyn},
        ])
    else:
        optimizer = torch.optim.Adam(world_model.parameters(), lr=args.lr_wm)

    scheduler = None
    if args.use_scheduler or args.phase2_cosine_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.phase2_epochs,
            eta_min=args.phase2_eta_min,
        )

    # Build config namespace for compute_phase2_loss / geodesic diffusion
    config_ns = SimpleNamespace(
        w_geodesic=args.w_geodesic,
        w_chart_transition=args.w_chart_transition,
        w_momentum_reg=args.w_momentum_reg,
        w_energy_conservation=args.w_energy_conservation,
        w_screened_poisson=args.w_screened_poisson,
        wm_screening_kappa=args.wm_screening_kappa,
        w_hodge=args.w_hodge,
        # Geodesic diffusion fields
        wm_diffusion_substeps=getattr(args, "wm_diffusion_substeps", 8),
        w_position=getattr(args, "w_position", 1.0),
        w_endpoint=getattr(args, "w_endpoint", 2.0),
        w_momentum_target=getattr(args, "w_momentum_target", 0.1),
        w_hodge_perp=getattr(args, "w_hodge_perp", 0.01),
        use_geodesic_diffusion=getattr(args, "use_geodesic_diffusion", False),
        wm_dt=args.wm_dt,
    )

    K = args.num_charts
    last_metrics: dict[str, float] = {}

    try:
        for epoch in tqdm(range(args.phase2_epochs), desc="Phase 2"):
            world_model.train()
            acc = _init_dynamics_accumulators()
            n_batches = 0
            epoch_K_all: list[torch.Tensor] = []
            epoch_soft_rw_all: list[torch.Tensor] = []
            epoch_radii: list[torch.Tensor] = []
            current_hard_routing = _use_hard_routing(
                args,
                global_epoch_offset + epoch,
            )
            current_tau = _get_hard_routing_tau(
                args,
                global_epoch_offset + epoch,
                total_epochs_all_phases,
            )

            for batch in seq_loader:
                features = batch["features"].to(device)  # [B, H, D_feat]
                actions = batch["actions"].to(device)  # [B, H, A]
                B, H, D_feat = features.shape

                # Encode all frames in one batched pass (frozen encoder/atlas)
                with torch.no_grad():
                    flat = features.reshape(B * H, D_feat)
                    K_flat, _, _, _, rw_flat, z_flat, _, _, _, c_bar_flat, v_local_flat, _ = (
                        model.encoder(
                            flat,
                            hard_routing=current_hard_routing,
                            hard_routing_tau=current_tau,
                        )
                    )
                    z_all = z_flat.reshape(B, H, -1)
                    K_all = K_flat.reshape(B, H)
                    rw_0 = rw_flat[:B]  # router weights for first frame
                    v_local_all_p2 = v_local_flat.reshape(B, H, -1)
                    rw_all_p2_full = rw_flat.reshape(B, H, -1)
                    soft_rw_all_p2 = getattr(
                        model.encoder,
                        "_last_soft_router_weights",
                        rw_flat.detach(),
                    ).reshape(B, H, -1)
                    c_bar_all_p2 = c_bar_flat.reshape(B, H, -1)
                    epoch_K_all.append(K_all.detach().cpu())
                    epoch_soft_rw_all.append(soft_rw_all_p2.detach().cpu())
                    epoch_radii.append(z_all.detach().norm(dim=-1).cpu())

                z_0 = z_all[:, 0, :]

                wm_output = None
                if getattr(config_ns, "use_geodesic_diffusion", False):
                    loss, metrics = compute_phase2_geodesic_diffusion_loss(
                        world_model,
                        z_all,
                        rw_all_p2_full,
                        K_all,
                        actions,
                        config_ns,
                    )
                else:
                    pred_actions = actions[:, :-1, :]
                    z_targets = z_all[:, 1:, :]
                    chart_targets = K_all[:, 1:]
                    wm_output = world_model(z_0, pred_actions, rw_0)
                    loss, metrics = compute_phase2_loss(
                        wm_output,
                        z_targets,
                        chart_targets,
                        config_ns,
                    )
                wm_diag = (
                    _wm_diagnostics(wm_output)
                    if wm_output is not None
                    else dict.fromkeys(WM_DIAG_KEYS, 0.0)
                )

                dyn_symbol_loss = z_all.new_tensor(0.0)
                dyn_symbol_metrics: dict[str, float] = {}
                if dyn_trans_model is not None and H > 1:
                    dyn_trans_model.train()
                    closure_weight = (
                        getattr(args, "w_enclosure", 0.0)
                        if getattr(args, "w_enclosure", 0.0) > 0
                        else getattr(args, "w_dyn_transition", 0.5)
                    )
                    dyn_symbol_loss, dyn_symbol_metrics, _ = compute_dynamics_markov_loss(
                        model.encoder,
                        dyn_trans_model,
                        v_local_all_p2.detach(),
                        rw_all_p2_full.detach(),
                        c_bar_all_p2.detach(),
                        K_all,
                        actions,
                        transition_weight=closure_weight,
                        zeno_weight=getattr(args, "w_zeno", 0.0),
                        zeno_mode=getattr(args, "zeno_mode", "jsd"),
                    )
                    loss = loss + dyn_symbol_loss
                    metrics.update(dyn_symbol_metrics)

                optimizer.zero_grad()
                loss.backward()
                all_opt_params = list(world_model.parameters())
                if dyn_trans_model is not None:
                    all_opt_params += [
                        model.encoder.codebook_dyn,
                        *list(dyn_trans_model.parameters()),
                    ]
                grad_norm = compute_grad_norm(all_opt_params)
                param_norm = compute_param_norm(all_opt_params)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_opt_params, args.grad_clip)
                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]
                update_ratio = (
                    current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0
                )

                acc["total"] += metrics["total"]
                for k in DYNAMICS_LOSS_KEYS:
                    if k in metrics:
                        acc[k] += metrics[k]
                for k in DYN_SYMBOL_KEYS:
                    if k in metrics:
                        acc[k] += metrics[k]
                for k in WM_DIAG_KEYS:
                    acc[k] += wm_diag[k]
                for k in GEODIFF_DIAG_KEYS:
                    if k in metrics:
                        acc[k] += metrics[k]
                acc["grad_norm"] += grad_norm
                acc["param_norm"] += param_norm
                acc["update_ratio"] += update_ratio
                acc["lr"] += current_lr
                n_batches += 1

            if scheduler is not None:
                scheduler.step()

            for k in acc:
                acc[k] /= max(n_batches, 1)

            # Chart usage from all batches in epoch
            hard_usage, hard_perplexity, hard_active = _chart_stats_from_tensor(
                torch.cat(epoch_K_all),
                K,
            )
            soft_usage, soft_perplexity, soft_active = _chart_stats_from_probs(
                torch.cat(epoch_soft_rw_all),
                K,
            )
            mean_r = torch.cat(epoch_radii).mean().item()

            use_geodiff = getattr(args, "use_geodesic_diffusion", False)
            should_log = (epoch % args.log_every == 0) or (epoch == args.phase2_epochs - 1)
            if should_log:
                print(f"P2 E{epoch:5d} | Loss: {acc['total']:.4f} | LR: {acc['lr']:.2e}")
                print(f"  Hard usage: {np.array2string(hard_usage, precision=2, separator=', ')}")
                print(f"  Soft usage: {np.array2string(soft_usage, precision=2, separator=', ')}")
                if use_geodiff:
                    print(
                        f"  GeoDiff: pos={acc.get('position', 0):.4f} "
                        f"endpoint={acc.get('endpoint', 0):.4f} "
                        f"mom_target={acc.get('momentum_target', 0):.4f} "
                        f"hodge_perp={acc.get('hodge_perp', 0):.4f} "
                        f"geo_miss={acc.get('geo_miss', 0):.4f}"
                    )
                    print(
                        f"  Chart: transition_ce={acc['chart_transition']:.4f} "
                        f"accuracy={acc.get('chart_accuracy', 0):.3f} "
                        f"same_chart%={acc.get('same_chart_frac', 0):.3f}"
                    )
                    print(
                        f"  Energy: conservation={acc.get('energy_conservation', 0):.4f} "
                        f"|p|={acc.get('mean_momentum', 0):.4f} "
                        f"phi_eff={acc.get('mean_phi_eff', 0):.4f}"
                    )
                    print(
                        f"  Hodge decomp: cons={acc.get('hodge_cons', 0):.4f} "
                        f"sol={acc.get('hodge_sol', 0):.4f} "
                        f"harm={acc.get('hodge_harm', 0):.4f}"
                    )
                else:
                    print(
                        f"  Dynamics: geo={acc['geodesic']:.4f} "
                        f"chart={acc['chart_transition']:.4f} "
                        f"mom_reg={acc['momentum_reg']:.4f}"
                    )
                    print(
                        f"  Energy: conservation={acc['energy_conservation']:.4f} "
                        f"screened_poisson={acc.get('screened_poisson', 0):.4f} "
                        f"hodge_loss={acc.get('hodge', 0):.4f}"
                    )
                    print(
                        f"  WM diag: |p|={acc['mean_momentum']:.4f} "
                        f"E_var={acc['energy_var']:.4f} "
                        f"jmp%={acc['jump_frac']:.4f} "
                        f"phi_eff={acc['mean_phi_eff']:.4f}"
                    )
                    print(
                        f"  Hodge decomp: cons={acc['hodge_cons']:.4f} "
                        f"sol={acc['hodge_sol']:.4f} "
                        f"harm={acc['hodge_harm']:.4f}"
                    )
                if dyn_trans_model is not None:
                    print(
                        f"  Macro: vq={acc.get('dyn_vq', 0):.4f} "
                        f"closure_ce={acc.get('dyn_trans_ce', 0):.4f} "
                        f"acc={acc.get('dyn_trans_acc', 0):.3f} "
                        f"zeno={acc.get('dyn_zeno', 0):.4f}"
                    )
                    print(
                        f"  Macro diag: state_flip={acc.get('dyn_state_flip_rate', 0):.4f} "
                        f"state_H={acc.get('dyn_state_entropy', 0):.4f} "
                        f"state_max={acc.get('dyn_state_max_prob', 0):.4f} "
                        f"code_flip={acc.get('dyn_code_flip_rate', 0):.4f}"
                    )
                print(
                    f"  Train: grad={acc['grad_norm']:.2e} "
                    f"upd_ratio={acc['update_ratio']:.2e} "
                    f"param={acc['param_norm']:.2e}"
                )
                print(
                    f"  Metrics: hard_perplexity={hard_perplexity:.2f}/{K} "
                    f"hard_active={hard_active}/{K} "
                    f"soft_perplexity={soft_perplexity:.2f}/{K} "
                    f"soft_active={soft_active}/{K} mean_r={mean_r:.3f}"
                )
                print("-" * 60)

            should_save = (
                epoch > 0 and epoch % args.save_every == 0
            ) or epoch == args.phase2_epochs - 1
            if should_save:
                _save_checkpoint(
                    args,
                    model,
                    jump_op,
                    world_model,
                    optimizer,
                    scheduler,
                    epoch,
                    2,
                    acc,
                    dyn_trans_model=dyn_trans_model,
                )

            last_metrics = acc
    finally:
        world_model.use_jump = phase2_use_jump

    # Unfreeze encoder for Phase 3
    for p in model.parameters():
        p.requires_grad_(True)
    for p in jump_op.parameters():
        p.requires_grad_(True)

    return last_metrics, dyn_trans_model


# ── Phase 3: Joint fine-tuning ────────────────────────────────────


def _run_phase3(
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel,
    seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    global_epoch_offset: int = 0,
    total_epochs_all_phases: int = 1,
    resume_probe_state: dict | None = None,
    dyn_trans_model: DynamicsTransitionModel | None = None,
) -> dict[str, float]:
    """Phase 3: Joint fine-tuning of encoder + world model."""
    print("\n" + "=" * 60)
    print("Phase 3: Joint fine-tuning")
    print("=" * 60)

    # Separate optimizers for alternating optimization
    optimizer_enc = torch.optim.Adam(
        build_encoder_param_groups(
            model,
            jump_op,
            base_lr=args.lr_joint_encoder,
            lr_chart_centers_scale=args.lr_chart_centers_scale,
            lr_codebook_scale=args.lr_codebook_scale,
        )
    )
    enc_params = [param for group in optimizer_enc.param_groups for param in group["params"]]
    optimizer_wm = torch.optim.Adam(world_model.parameters(), lr=args.lr_joint_wm)

    # Codebook dynamics optimizer (Option D): only codebook params
    w_codebook_dynamics = getattr(args, "w_codebook_dynamics", 0.0)
    optimizer_cb = None
    if w_codebook_dynamics > 0 and hasattr(model.encoder, "codebook"):
        cb_params = get_codebook_like_params(model)
        optimizer_cb = torch.optim.Adam(
            cb_params,
            lr=args.lr_joint_encoder * args.lr_codebook_scale,
        )
    scheduler_enc = None
    scheduler_wm = None
    if args.use_scheduler or args.phase3_cosine_lr:
        scheduler_enc = CosineAnnealingLR(
            optimizer_enc,
            T_max=args.phase3_epochs,
            eta_min=args.phase3_eta_min,
        )
        scheduler_wm = CosineAnnealingLR(
            optimizer_wm,
            T_max=args.phase3_epochs,
            eta_min=args.phase3_eta_min,
        )

    # Config namespace for dynamics losses
    config_ns = SimpleNamespace(
        w_geodesic=args.w_geodesic,
        w_chart_transition=args.w_chart_transition,
        w_momentum_reg=args.w_momentum_reg,
        w_energy_conservation=args.w_energy_conservation,
        w_screened_poisson=args.w_screened_poisson,
        wm_screening_kappa=args.wm_screening_kappa,
        w_hodge=args.w_hodge,
        # Geodesic diffusion fields
        wm_diffusion_substeps=getattr(args, "wm_diffusion_substeps", 8),
        w_position=getattr(args, "w_position", 1.0),
        w_endpoint=getattr(args, "w_endpoint", 2.0),
        w_momentum_target=getattr(args, "w_momentum_target", 0.1),
        w_hodge_perp=getattr(args, "w_hodge_perp", 0.01),
        use_geodesic_diffusion=getattr(args, "use_geodesic_diffusion", False),
        wm_dt=args.wm_dt,
    )

    K = args.num_charts
    last_metrics: dict[str, float] = {}

    # Enclosure probe (if enabled)
    probe = None
    probe_optimizer = None
    if getattr(args, "w_enclosure", 0.0) > 0:
        probe = EnclosureProbe(
            chart_dim=args.latent_dim,
            ztex_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=args.num_charts,
            codes_per_chart=args.codes_per_chart,
            hidden_dim=getattr(args, "enclosure_probe_hidden_dim", 128),
            alpha=0.0,  # starts at 0, warmed up
        ).to(device)
        probe_optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=getattr(args, "enclosure_probe_lr", 3e-3),
        )
        if resume_probe_state is not None:
            if resume_probe_state.get("probe") is not None:
                probe.load_state_dict(resume_probe_state["probe"])
                print("  Restored enclosure probe weights from checkpoint")
            if resume_probe_state.get("probe_optimizer") is not None:
                probe_optimizer.load_state_dict(resume_probe_state["probe_optimizer"])
                print("  Restored enclosure probe optimizer from checkpoint")

    for epoch in tqdm(range(args.phase3_epochs), desc="Phase 3"):
        model.train()
        jump_op.train()
        world_model.train()
        acc = _init_joint_accumulators()
        n_batches = 0
        epoch_K_all: list[torch.Tensor] = []
        epoch_soft_rw_all: list[torch.Tensor] = []
        epoch_radii: list[torch.Tensor] = []
        current_hard_routing = _use_hard_routing(
            args,
            global_epoch_offset + epoch,
        )
        current_tau = _get_hard_routing_tau(
            args,
            global_epoch_offset + epoch,
            total_epochs_all_phases,
        )

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)  # [B, H, A]
            B, H, D_feat = features.shape

            # Frame 0: full encoder losses + reuse outputs (1 encoder call)
            (
                base_loss,
                zn_reg_loss,
                enc_metrics_0,
                z_geo_0,
                enc_w_0,
                K_ch_0,
                zn_0,
                ztex_0,
                c_bar_0,
                K_code_0,
                zq_blended_0,
                v_local_0,
            ) = _compute_encoder_losses(
                features[:, 0, :],
                model,
                jump_op,
                args,
                epoch,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )
            soft_rw_0 = getattr(model.encoder, "_last_soft_router_weights", enc_w_0.detach())

            # Frames 1..H-1: batched encoding (1 encoder call)
            if H > 1:
                rest = features[:, 1:, :].reshape(B * (H - 1), D_feat)
                (
                    K_rest,
                    Kcode_rest,
                    zn_rest,
                    ztex_rest,
                    rw_rest,
                    z_rest,
                    _,
                    _,
                    _,
                    c_bar_rest,
                    v_local_rest,
                    zq_blended_rest,
                ) = model.encoder(
                    rest,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                )
                soft_rw_rest = getattr(
                    model.encoder, "_last_soft_router_weights", rw_rest.detach()
                )
                z_geo_rest = z_rest.reshape(B, H - 1, -1)
                K_rest = K_rest.reshape(B, H - 1)
                Kcode_rest = Kcode_rest.reshape(B, H - 1)
                zn_rest = zn_rest.reshape(B, H - 1, -1)
                ztex_rest = ztex_rest.reshape(B, H - 1, -1)
                c_bar_rest = c_bar_rest.reshape(B, H - 1, -1)
                rw_rest_reshaped = rw_rest.reshape(B, H - 1, -1)
                soft_rw_rest_reshaped = soft_rw_rest.reshape(B, H - 1, -1)
                zq_blended_rest = zq_blended_rest.reshape(B, H - 1, -1)
                v_local_rest = v_local_rest.reshape(B, H - 1, -1)
                z_all = torch.cat([z_geo_0.unsqueeze(1), z_geo_rest], dim=1)
                K_all = torch.cat([K_ch_0.unsqueeze(1), K_rest], dim=1)
                Kcode_all = torch.cat([K_code_0.unsqueeze(1), Kcode_rest], dim=1)
                torch.cat([zn_0.unsqueeze(1), zn_rest], dim=1)
                ztex_all = torch.cat([ztex_0.unsqueeze(1), ztex_rest], dim=1)
                c_bar_all = torch.cat([c_bar_0.unsqueeze(1), c_bar_rest], dim=1)
                rw_all = torch.cat([enc_w_0.unsqueeze(1), rw_rest_reshaped], dim=1)
                soft_rw_all = torch.cat([soft_rw_0.unsqueeze(1), soft_rw_rest_reshaped], dim=1)
                zq_blended_all = torch.cat([zq_blended_0.unsqueeze(1), zq_blended_rest], dim=1)
                v_local_all = torch.cat([v_local_0.unsqueeze(1), v_local_rest], dim=1)
            else:
                z_all = z_geo_0.unsqueeze(1)
                K_all = K_ch_0.unsqueeze(1)
                Kcode_all = K_code_0.unsqueeze(1)
                zn_0.unsqueeze(1)
                ztex_all = ztex_0.unsqueeze(1)
                c_bar_all = c_bar_0.unsqueeze(1)
                rw_all = enc_w_0.unsqueeze(1)
                soft_rw_all = soft_rw_0.unsqueeze(1)
                zq_blended_all = zq_blended_0.unsqueeze(1)
                v_local_all = v_local_0.unsqueeze(1)

            # Enclosure loss (over consecutive frame pairs)
            L_encl_encoder = None
            L_encl_probe = None
            encl_diag_avg = {}
            if probe is not None:
                global_step = epoch * len(seq_loader) + n_batches
                probe.grl.alpha.fill_(
                    grl_alpha_schedule(
                        global_step,
                        warmup_steps=getattr(args, "enclosure_grl_warmup_steps", 5000),
                        max_alpha=getattr(args, "enclosure_grl_max_alpha", 1.0),
                    )
                )

                encl_enc_losses = []
                encl_probe_losses = []
                all_encl_diag = []

                for t in range(H - 1):
                    loss_enc_t, loss_probe_t, diag_t = compute_enclosure_loss(
                        probe=probe,
                        chart_embed_t=c_bar_all[:, t],
                        action_t=actions[:, t],
                        ztex_t=ztex_all[:, t],
                        K_chart_tp1=K_all[:, t + 1],
                        K_code_t=Kcode_all[:, t],
                        K_code_tp1=Kcode_all[:, t + 1],
                        codes_per_chart=args.codes_per_chart,
                    )
                    encl_enc_losses.append(loss_enc_t)
                    encl_probe_losses.append(loss_probe_t)
                    all_encl_diag.append(diag_t)

                L_encl_encoder = torch.stack(encl_enc_losses).mean()
                L_encl_probe = torch.stack(encl_probe_losses).mean()

                # Average diagnostics
                for key in all_encl_diag[0]:
                    encl_diag_avg[f"encl/{key}"] = sum(d[key] for d in all_encl_diag) / len(
                        all_encl_diag
                    )

            # Zeno loss (routing distribution smoothness)
            L_zeno = None
            zeno_diag = {}
            w_zeno = getattr(args, "w_zeno", 0.0)
            if w_zeno > 0 and H > 1:
                zeno_mode = getattr(args, "zeno_mode", "jsd")
                zeno_losses = []
                for t in range(1, H):
                    zeno_t = zeno_loss(rw_all[:, t], rw_all[:, t - 1], mode=zeno_mode)
                    zeno_losses.append(zeno_t)
                L_zeno = torch.stack(zeno_losses).mean()

                with torch.no_grad():
                    flips = (K_all[:, 1:] != K_all[:, :-1]).float()
                    ent = -(rw_all * rw_all.clamp(min=1e-8).log()).sum(dim=-1)
                    zeno_diag["zeno/flip_rate"] = flips.mean().item()
                    zeno_diag["zeno/routing_entropy"] = ent.mean().item()
                    zeno_diag["zeno/max_weight"] = rw_all.max(dim=-1).values.mean().item()
                    # Mean segment length (sample up to 8 batch elements)
                    seg_lens = []
                    for b in range(min(B, 8)):
                        charts_b = K_all[b].tolist()
                        seg = 1
                        for i in range(1, len(charts_b)):
                            if charts_b[i] == charts_b[i - 1]:
                                seg += 1
                            else:
                                seg_lens.append(seg)
                                seg = 1
                        seg_lens.append(seg)
                    zeno_diag["zeno/mean_segment_length"] = (
                        sum(seg_lens) / len(seg_lens) if seg_lens else 0.0
                    )

            # --- Encoder step (WM frozen) ---
            optimizer_enc.zero_grad()
            L_enc = args.phase3_encoder_scale * base_loss + args.phase3_zn_reg_scale * zn_reg_loss
            if L_encl_encoder is not None:
                L_enc = L_enc + getattr(args, "w_enclosure", 0.0) * L_encl_encoder
            if L_zeno is not None:
                L_enc = L_enc + w_zeno * L_zeno
            L_enc.backward()
            enc_grad_norm = compute_grad_norm(enc_params)
            enc_param_norm = compute_param_norm(enc_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(enc_params, args.grad_clip)
            optimizer_enc.step()

            # --- WM step (encoder detached) ---
            optimizer_wm.zero_grad()
            z_all_det = z_all.detach()
            rw_all_det = rw_all.detach()
            K_all_det = K_all.detach()

            if getattr(config_ns, "use_geodesic_diffusion", False):
                dyn_loss, dyn_metrics = compute_phase2_geodesic_diffusion_loss(
                    world_model,
                    z_all_det,
                    rw_all_det,
                    K_all_det,
                    actions,
                    config_ns,
                )
                wm_diag = dict.fromkeys(WM_DIAG_KEYS, 0.0)
            else:
                rw_0 = enc_w_0.detach()
                pred_actions = actions[:, :-1, :]
                z_targets = z_all_det[:, 1:, :]
                chart_targets = K_all_det[:, 1:]
                wm_output = world_model(z_all_det[:, 0], pred_actions, rw_0)
                dyn_loss, dyn_metrics = compute_phase2_loss(
                    wm_output,
                    z_targets,
                    chart_targets,
                    config_ns,
                )
                wm_diag = _wm_diagnostics(wm_output)

            (args.phase3_dynamics_scale * dyn_loss).backward()
            wm_params = list(world_model.parameters())
            wm_grad_norm = compute_grad_norm(wm_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(wm_params, args.grad_clip)
            optimizer_wm.step()

            # Probe optimizer step (separate backward on detached inputs)
            if probe is not None and L_encl_probe is not None:
                probe_optimizer.zero_grad()
                L_encl_probe.backward()
                probe_optimizer.step()

            # --- Codebook dynamics step (Option D) ---
            # Flow WM prediction gradients into codebook (not other encoder params).
            # WM weights frozen; c_bar detached so router doesn't get dynamics grads.
            L_cb_dyn = torch.tensor(0.0, device=device)
            if optimizer_cb is not None and H > 1:
                optimizer_cb.zero_grad()
                # Build coarse latent from detached c_bar + live codebook codes
                z_coarse_0 = project_to_ball(
                    mobius_add(c_bar_all[:, 0].detach(), zq_blended_all[:, 0])
                )
                rw_0_cb = rw_all[:, 0].detach()
                # Forward WM with frozen weights but live codebook
                with torch.no_grad():
                    for p in world_model.parameters():
                        p.requires_grad_(False)
                wm_out_cb = world_model(z_coarse_0, actions[:, :-1, :], rw_0_cb)
                z_pred_cb = wm_out_cb["z_pred"]  # [B, H-1, D]
                with torch.no_grad():
                    for p in world_model.parameters():
                        p.requires_grad_(True)
                L_cb_dyn = hyperbolic_distance(
                    z_pred_cb.reshape(-1, z_pred_cb.shape[-1]),
                    z_all_det[:, 1:].reshape(-1, z_all_det.shape[-1]),
                ).mean()
                # Dynamics codebook transition loss (Phase 3)
                if dyn_trans_model is not None and model.encoder.codebook_dyn is not None:
                    v_local_det_p3 = v_local_all.detach()
                    rw_det_p3 = rw_all.detach()
                    vq_dyn_losses_p3 = []
                    K_code_dyn_list_p3 = []
                    for t in range(H):
                        _, K_code_dyn_t, _, vq_dyn_t = model.encoder.dynamics_vq(
                            v_local_det_p3[:, t],
                            rw_det_p3[:, t],
                        )
                        vq_dyn_losses_p3.append(vq_dyn_t)
                        K_code_dyn_list_p3.append(K_code_dyn_t)
                    vq_dyn_loss_p3 = torch.stack(vq_dyn_losses_p3).mean()
                    K_code_dyn_all_p3 = torch.stack(K_code_dyn_list_p3, dim=1)

                    trans_losses_p3 = []
                    for t in range(H - 1):
                        t_loss, _ = compute_dyn_transition_loss(
                            dyn_trans_model,
                            c_bar_all[:, t].detach(),
                            actions[:, t],
                            K_code_dyn_all_p3[:, t],
                            K_all_det[:, t + 1],
                            K_code_dyn_all_p3[:, t + 1],
                        )
                        trans_losses_p3.append(t_loss)
                    trans_loss_p3 = torch.stack(trans_losses_p3).mean()
                    w_dyn_transition = getattr(args, "w_dyn_transition", 0.5)
                    L_dyn_cb_extra = vq_dyn_loss_p3 + w_dyn_transition * trans_loss_p3
                    L_dyn_cb_extra.backward()

                (w_codebook_dynamics * L_cb_dyn).backward()
                cb_clip_params = [model.encoder.codebook]
                if model.encoder.codebook_dyn is not None:
                    cb_clip_params.append(model.encoder.codebook_dyn)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(cb_clip_params, args.grad_clip)
                optimizer_cb.step()

            grad_norm = enc_grad_norm + wm_grad_norm
            param_norm = enc_param_norm + compute_param_norm(wm_params)
            current_lr = optimizer_enc.param_groups[0]["lr"]
            update_ratio = current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0

            total = L_enc.item() + args.phase3_dynamics_scale * dyn_loss.item()
            acc["total"] += total
            acc["enc_total"] += enc_metrics_0["total"]
            acc["dyn_total"] += dyn_metrics["total"]
            for k in ENCODER_LOSS_KEYS:
                acc[f"enc/{k}"] += enc_metrics_0[k]
            for k in DYNAMICS_LOSS_KEYS:
                if k in dyn_metrics:
                    acc[f"dyn/{k}"] += dyn_metrics[k]
            for k in WM_DIAG_KEYS:
                acc[f"wm/{k}"] += wm_diag[k]
            for k in GEODIFF_DIAG_KEYS:
                if k in dyn_metrics:
                    acc[f"gd/{k}"] += dyn_metrics[k]
            if L_encl_encoder is not None:
                acc["encl_encoder"] += L_encl_encoder.item()
                acc["encl_probe"] += L_encl_probe.item()
                for k, v in encl_diag_avg.items():
                    acc[k] += v
            if L_zeno is not None:
                acc["zeno"] += L_zeno.item()
                for k, v in zeno_diag.items():
                    acc[k] += v
            for k in INFO_KEYS:
                if k in {"grad_norm", "param_norm", "update_ratio", "lr"}:
                    continue
                acc[k] += enc_metrics_0.get(k, 0.0)
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            epoch_K_all.append(K_all.detach().cpu())
            epoch_soft_rw_all.append(soft_rw_all.detach().cpu())
            epoch_radii.append(z_all.detach().norm(dim=-1).cpu())
            n_batches += 1

        if scheduler_enc is not None:
            scheduler_enc.step()
        if scheduler_wm is not None:
            scheduler_wm.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        # Chart stats from all batches in epoch
        hard_usage, hard_perplexity, hard_active = _chart_stats_from_tensor(
            torch.cat(epoch_K_all),
            K,
        )
        soft_usage, soft_perplexity, soft_active = _chart_stats_from_probs(
            torch.cat(epoch_soft_rw_all),
            K,
        )
        mean_r = torch.cat(epoch_radii).mean().item()

        should_log = (epoch % args.log_every == 0) or (epoch == args.phase3_epochs - 1)
        if should_log:
            grl_alpha = probe.grl.alpha.item() if probe is not None else 0.0
            print(
                f"P3 E{epoch:5d} | Loss: {acc['total']:.4f} "
                f"(enc={acc['enc_total']:.4f} dyn={acc['dyn_total']:.4f}) "
                f"| LR: {acc['lr']:.2e} | tau: {current_tau:.3f}"
            )
            print(f"  Hard usage: {np.array2string(hard_usage, precision=2, separator=', ')}")
            print(f"  Soft usage: {np.array2string(soft_usage, precision=2, separator=', ')}")
            # --- Encoder losses ---
            print(
                f"  Core: recon={acc['enc/recon']:.3f} "
                f"vq={acc['enc/vq']:.3f} "
                f"entropy={acc['enc/entropy']:.3f} "
                f"consist={acc['enc/consistency']:.3f} "
                f"chart_use={acc['enc/chart_usage']:.3f}"
            )
            print(
                f"  Info: I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f}"
            )
            print(
                f"  Sharp: top1={acc['top1_prob_mean']:.3f} "
                f"gap={acc['top1_gap_mean']:.3f} top2={acc['top2_prob_mean']:.3f} "
                f"p10={acc['top1_prob_p10']:.3f} p90={acc['top1_prob_p90']:.3f}"
            )
            print(
                f"  Geo: unif={acc['enc/uniformity']:.3f} "
                f"rad_cal={acc['enc/radial_cal']:.3f} "
                f"conf_cal={acc['enc/confidence_calibration']:.3f} "
                f"v_tan={acc['enc/v_tangent_barrier']:.3f} "
                f"cb_spread={acc['enc/codebook_spread']:.3f} "
                f"cb_center={acc['enc/codebook_center']:.3f} "
                f"cc_mean={acc['enc/chart_center_mean']:.3f} "
                f"cc_rad={acc['enc/chart_center_radius']:.3f} "
                f"cc_sep={acc['enc/chart_center_sep']:.3f}"
            )
            print(f"  Usage: code={acc['enc/code_usage']:.4f}")
            print(
                f"  Train usage: H_hard={acc['H_usage']:.3f} "
                f"perp={acc['usage_perplexity']:.2f}/{K} "
                f"active={acc['usage_active']:.2f}/{K} "
                f"code_H={acc['H_code_usage']:.3f} "
                f"code_perp={acc['code_usage_perplexity']:.2f}/{args.codes_per_chart} "
                f"active_code_charts={acc['active_code_charts']:.2f}"
            )
            print(
                f"  Ortho: {acc.get('enc/ortho', 0):.4f} (w={getattr(args, 'w_perp', 0.01):.3f})"
            )
            print(
                f"  Enclosure: enc={acc.get('encl_encoder', 0):.4f} "
                f"probe={acc.get('encl_probe', 0):.4f} "
                f"defect_acc={acc.get('encl/defect_acc', 0):.4f} "
                f"defect_ce={acc.get('encl/defect_ce', 0):.4f} "
                f"grl_alpha={grl_alpha:.3f} "
                f"(w={getattr(args, 'w_enclosure', 0.0):.3f})"
            )
            print(
                f"  Zeno: loss={acc.get('zeno', 0):.4f} "
                f"flip_rate={acc.get('zeno/flip_rate', 0):.4f} "
                f"seg_len={acc.get('zeno/mean_segment_length', 0):.2f} "
                f"entropy={acc.get('zeno/routing_entropy', 0):.4f} "
                f"max_w={acc.get('zeno/max_weight', 0):.4f} "
                f"(w={getattr(args, 'w_zeno', 0.0):.3f})"
            )
            print(f"  Window: {acc['enc/window']:.3f} (w={args.w_window:.3f})")
            print(
                f"  Jump: {acc['enc/jump']:.3f} "
                f"(lambda={enc_metrics_0.get('jump_weight', 0.0):.3f})"
            )
            # --- Dynamics losses ---
            use_geodiff = getattr(args, "use_geodesic_diffusion", False)
            if use_geodiff:
                print(
                    f"  GeoDiff: pos={acc.get('dyn/position', 0):.4f} "
                    f"endpoint={acc.get('dyn/endpoint', 0):.4f} "
                    f"mom_target={acc.get('dyn/momentum_target', 0):.4f} "
                    f"hodge_perp={acc.get('dyn/hodge_perp', 0):.4f} "
                    f"geo_miss={acc.get('dyn/geo_miss', 0):.4f}"
                )
                print(
                    f"  Chart: transition_ce={acc.get('dyn/chart_transition', 0):.4f} "
                    f"accuracy={acc.get('gd/chart_accuracy', 0):.3f} "
                    f"same_chart%={acc.get('gd/same_chart_frac', 0):.3f}"
                )
                print(
                    f"  Energy: conservation={acc.get('dyn/energy_conservation', 0):.4f} "
                    f"|p|={acc.get('gd/mean_momentum', 0):.4f} "
                    f"phi_eff={acc.get('gd/mean_phi_eff', 0):.4f}"
                )
                print(
                    f"  Hodge decomp: cons={acc.get('gd/hodge_cons', 0):.4f} "
                    f"sol={acc.get('gd/hodge_sol', 0):.4f} "
                    f"harm={acc.get('gd/hodge_harm', 0):.4f}"
                )
            else:
                print(
                    f"  Dynamics: geo={acc['dyn/geodesic']:.4f} "
                    f"chart={acc['dyn/chart_transition']:.4f} "
                    f"mom_reg={acc['dyn/momentum_reg']:.4f}"
                )
                print(
                    f"  Energy: conservation={acc['dyn/energy_conservation']:.4f} "
                    f"screened_poisson={acc.get('dyn/screened_poisson', 0):.4f} "
                    f"hodge_loss={acc.get('dyn/hodge', 0):.4f}"
                )
                print(
                    f"  WM diag: |p|={acc['wm/mean_momentum']:.4f} "
                    f"E_var={acc['wm/energy_var']:.4f} "
                    f"jmp%={acc['wm/jump_frac']:.4f} "
                    f"phi_eff={acc['wm/mean_phi_eff']:.4f}"
                )
                print(
                    f"  Hodge decomp: cons={acc['wm/hodge_cons']:.4f} "
                    f"sol={acc['wm/hodge_sol']:.4f} "
                    f"harm={acc['wm/hodge_harm']:.4f}"
                )
            # --- Training diagnostics ---
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} lr={acc['lr']:.2e}"
            )
            print(
                f"  Metrics: hard_perplexity={hard_perplexity:.2f}/{K} "
                f"hard_active={hard_active}/{K} "
                f"soft_perplexity={soft_perplexity:.2f}/{K} "
                f"soft_active={soft_active}/{K} mean_r={mean_r:.3f}"
            )
            print("-" * 60)

        should_save = (
            epoch > 0 and epoch % args.save_every == 0
        ) or epoch == args.phase3_epochs - 1
        if should_save:
            _save_checkpoint(
                args,
                model,
                jump_op,
                world_model,
                optimizer_enc,
                scheduler_enc,
                epoch,
                3,
                acc,
                probe=probe,
                probe_optimizer=probe_optimizer,
                dyn_trans_model=dyn_trans_model,
            )

        last_metrics = acc

    return last_metrics, probe


# ── Checkpoint helpers ────────────────────────────────────────────


def _save_checkpoint(
    args: argparse.Namespace,
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    world_model: GeometricWorldModel | None,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    epoch: int,
    phase: int,
    metrics: dict[str, float],
    probe: EnclosureProbe | None = None,
    probe_optimizer: torch.optim.Optimizer | None = None,
    dyn_trans_model: DynamicsTransitionModel | None = None,
) -> None:
    ckpt_path = os.path.join(args.output_dir, f"p{phase}_epoch_{epoch:05d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "phase": phase,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": world_model.state_dict() if world_model is not None else None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "probe": probe.state_dict() if probe is not None else None,
            "probe_optimizer": probe_optimizer.state_dict()
            if probe_optimizer is not None
            else None,
            "dyn_trans_model": dyn_trans_model.state_dict()
            if dyn_trans_model is not None
            else None,
            "args": vars(args),
            "metrics": metrics,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")


# ── Final diagnostics ────────────────────────────────────────────


def _run_diagnostics(
    model: TopoEncoder,
    single_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    had_world_model: bool,
    final_dyn_metrics: dict[str, float] | None,
) -> None:
    """Run end-of-training diagnostics."""
    print("\n=== Final Diagnostics ===")
    model.eval()

    K = args.num_charts
    all_features: list[torch.Tensor] = []
    all_ep_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in single_loader:
            all_features.append(batch["feature"])
            all_ep_ids.append(batch["episode_id"])
    all_features_t = torch.cat(all_features)
    all_ep_ids_t = torch.cat(all_ep_ids)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from fragile.vla.visualize import full_diagnostic, plot_chart_transitions

    config_ns = SimpleNamespace(num_charts=K)
    full_diagnostic(model, all_features_t, config_ns, save_dir=args.output_dir)

    with torch.no_grad():
        K_chart_all, *_ = model.encoder(all_features_t.to(device))

    fig = plot_chart_transitions(K_chart_all.cpu(), all_ep_ids_t)
    fig.savefig(
        os.path.join(args.output_dir, "chart_transitions.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)

    # World model dynamics summary
    if had_world_model and final_dyn_metrics is not None:
        print("\n--- World Model Summary ---")
        for k in DYNAMICS_LOSS_KEYS:
            # Try both prefixed and unprefixed keys
            val = final_dyn_metrics.get(f"dyn/{k}", final_dyn_metrics.get(k, 0.0))
            print(f"  {k}: {val:.4f}")

    print(f"\nAll outputs saved to {args.output_dir}")


# ── Main training function ────────────────────────────────────────


def train_joint(args: argparse.Namespace) -> None:  # noqa: C901
    """Run the 3-phase joint encoder + world model training."""
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data loaders ──────────────────────────────────────────
    single_loader = None
    seq_loader = None

    if args.phase1_epochs > 0:
        single_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="train")
        single_loader = DataLoader(
            single_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,
        )
        input_dim = single_ds[0]["feature"].shape[0]
        print(
            f"Single-frame train dataset: {len(single_ds)} frames, {len(single_loader)} batches",
        )
    else:
        # Need input_dim from a probe
        probe_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="train")
        input_dim = probe_ds[0]["feature"].shape[0]
        # Still build single_loader for diagnostics
        single_loader = DataLoader(
            probe_ds,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        print("Skipping Phase 1 (single-frame loader built for diagnostics only)")

    # Build seq_loader if P2/P3 need it (Phase 1 is always single-frame)
    if args.phase2_epochs > 0 or args.phase3_epochs > 0:
        seq_ds = VLAFeatureDataset(
            args.feature_cache_dir,
            sequence_length=args.sequence_length,
            split="train",
        )
        seq_loader = DataLoader(
            seq_ds,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
        )
        print(
            f"Sequence train dataset: {len(seq_ds)} windows "
            f"(seq_len={args.sequence_length}), {len(seq_loader)} batches",
        )

    print(f"Feature dim: {input_dim}")

    # ── Models ────────────────────────────────────────────────
    K = args.num_charts
    model = TopoEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=K,
        codes_per_chart=args.codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
        commitment_beta=getattr(args, "commitment_beta", 0.25),
        codebook_loss_weight=getattr(args, "codebook_loss_weight", 1.0),
    ).to(device)

    jump_op = FactorizedJumpOperator(
        num_charts=K,
        latent_dim=args.latent_dim,
    ).to(device)

    world_model = None
    if args.phase2_epochs > 0 or args.phase3_epochs > 0:
        risk_alpha = getattr(args, "wm_risk_metric_alpha", None)
        if risk_alpha is None:
            risk_alpha = 0.0
        world_model = GeometricWorldModel(
            latent_dim=args.latent_dim,
            action_dim=args.action_dim,
            num_charts=K,
            d_model=args.wm_d_model,
            hidden_dim=args.wm_hidden_dim,
            dt=args.wm_dt,
            gamma_friction=args.wm_gamma_friction,
            T_c=args.wm_T_c,
            alpha_potential=args.wm_alpha_potential,
            beta_curl=args.wm_beta_curl,
            gamma_risk=args.wm_gamma_risk,
            use_boris=args.wm_use_boris,
            use_jump=args.wm_use_jump,
            n_refine_steps=args.wm_refine_steps,
            jump_beta=args.wm_jump_beta,
            min_length=max(args.wm_min_length, 0.0),  # -1 (auto) deferred until after P1
            risk_metric_alpha=risk_alpha,
        ).to(device)

    # ── Parameter breakdown ──────────────────────────────────
    n_enc = count_parameters(model.encoder)
    n_dec = count_parameters(model.decoder)
    n_jump = count_parameters(jump_op)
    n_total = n_enc + n_dec + n_jump
    print(f"  Encoder:  {n_enc:>10,} params")
    print(f"  Decoder:  {n_dec:>10,} params")
    print(f"  Jump op:  {n_jump:>10,} params")
    if world_model is not None:
        n_wm = count_parameters(world_model)
        n_total += n_wm
        print("  World model breakdown:")
        for name, child in world_model.named_children():
            nc = count_parameters(child)
            if nc > 0:
                print(f"    {name:25s} {nc:>8,} params")
        print(f"  World model total: {n_wm:>7,} params")
        if world_model.min_length > 0:
            print(f"  CFL bounds (ℓ_min={world_model.min_length:.4f}):")
            print(
                f"    F_max={world_model.F_max:.2f}  V_alg={world_model.V_alg:.2f}  "
                f"cf_max={world_model.cf_max:.2f}"
            )
        elif args.wm_min_length < 0:
            print("  CFL bounds: auto-measure after Phase 1")
        else:
            print("  CFL bounds: disabled (no squashing)")
    print(f"  TOTAL: {n_total:>13,} params")

    # ── Resume ────────────────────────────────────────────────
    resume_probe_state = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        if ckpt.get("world_model") is not None and world_model is not None:
            world_model.load_state_dict(ckpt["world_model"])
        if ckpt.get("probe") is not None:
            resume_probe_state = {
                "probe": ckpt["probe"],
                "probe_optimizer": ckpt.get("probe_optimizer"),
            }
        print(
            f"Resumed from {args.resume} (phase {ckpt.get('phase', '?')}, "
            f"epoch {ckpt.get('epoch', '?')})"
        )

    # ── Output dir ────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Warm-start chart centers ───────────────────────────────
    if not args.resume:
        if args.hard_routing:
            print(
                "\nSkipping k-means chart warm-start under hard routing; "
                "keeping the encoder's quasi-uniform atlas initialization.",
            )
        else:
            print("\nWarm-starting chart centers with k-means...")
            model.warmstart_chart_centers(
                single_loader,
                device,
                max_batches=10,
                radius_floor=0.0,
            )

    # ── Run phases ────────────────────────────────────────────
    last_dyn_metrics = None
    had_world_model = False
    total_epochs_all_phases = args.phase1_epochs + args.phase2_epochs + args.phase3_epochs

    final_probe = None

    if args.phase1_epochs > 0:
        _, probe_from_p1 = _run_phase1(
            model,
            jump_op,
            single_loader,
            args,
            device,
            global_epoch_offset=0,
            total_epochs_all_phases=total_epochs_all_phases,
            seq_loader=seq_loader,
            resume_probe_state=resume_probe_state,
        )
        final_probe = probe_from_p1
        # Propagate trained probe state to Phase 3 (warm start)
        if probe_from_p1 is not None and resume_probe_state is None:
            resume_probe_state = {
                "probe": probe_from_p1.state_dict(),
                "probe_optimizer": None,  # fresh optimizer for Phase 3
            }

    # Auto-measure ℓ_min from encoder if requested (--wm-min-length -1)
    if args.wm_min_length < 0 and world_model is not None and seq_loader is not None:
        print("\n  Auto-measuring minimum length scale from encoder...")
        ell_min = _measure_min_length(model, seq_loader, device)
        _update_world_model_min_length(world_model, ell_min)

    dyn_trans_model_p2 = None
    if args.phase2_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics, dyn_trans_model_p2 = _run_phase2(
            model,
            jump_op,
            world_model,
            seq_loader,
            args,
            device,
            global_epoch_offset=args.phase1_epochs,
            total_epochs_all_phases=total_epochs_all_phases,
        )
        had_world_model = True

    if args.phase3_epochs > 0:
        assert world_model is not None
        assert seq_loader is not None
        last_dyn_metrics, probe_from_p3 = _run_phase3(
            model,
            jump_op,
            world_model,
            seq_loader,
            args,
            device,
            global_epoch_offset=args.phase1_epochs + args.phase2_epochs,
            total_epochs_all_phases=total_epochs_all_phases,
            resume_probe_state=resume_probe_state,
            dyn_trans_model=dyn_trans_model_p2,
        )
        had_world_model = True
        if probe_from_p3 is not None:
            final_probe = probe_from_p3

    # ── Final checkpoint ──────────────────────────────────────
    final_phase = 3 if args.phase3_epochs > 0 else (2 if args.phase2_epochs > 0 else 1)
    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    torch.save(
        {
            "epoch": -1,
            "phase": final_phase,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": world_model.state_dict() if world_model is not None else None,
            "probe": final_probe.state_dict() if final_probe is not None else None,
            "dyn_trans_model": dyn_trans_model_p2.state_dict()
            if dyn_trans_model_p2 is not None
            else None,
            "optimizer": None,
            "scheduler": None,
            "args": vars(args),
            "metrics": last_dyn_metrics,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    # ── Diagnostics ───────────────────────────────────────────
    _run_diagnostics(
        model,
        single_loader,
        args,
        device,
        had_world_model,
        last_dyn_metrics,
    )


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for joint encoder + world model training."""
    p = argparse.ArgumentParser(
        description="Joint encoder + world model training (3-phase)",
    )

    # Data / output
    p.add_argument("--feature-cache-dir", default="outputs/vla/features")
    p.add_argument("--output-dir", default="outputs/vla/joint")

    # Architecture
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument(
        "--hard-routing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use hard routing by default (one-hot forward, ST gradients)",
    )
    p.add_argument(
        "--hard-routing-warmup-epochs",
        type=int,
        default=5,
        help="Epochs of soft partition-of-unity routing before hard ST continuation",
    )
    p.add_argument(
        "--hard-routing-tau",
        type=float,
        default=1.0,
        help="Starting temperature for hard routing; positive = Gumbel-softmax, "
        "negative = deterministic straight-through argmax (no Gumbel noise)",
    )
    p.add_argument(
        "--hard-routing-tau-end",
        type=float,
        default=0.3,
        help="Final tau after annealing (None = no annealing, use constant tau)",
    )
    p.add_argument(
        "--hard-routing-tau-anneal-epochs",
        type=int,
        default=200,
        help="Anneal tau linearly over this many epochs (None = total epochs)",
    )

    # Phase epochs (0 = skip)
    p.add_argument("--phase1-epochs", type=int, default=100)
    p.add_argument("--phase2-epochs", type=int, default=50)
    p.add_argument("--phase3-epochs", type=int, default=50)

    # Learning rates
    p.add_argument("--lr", type=float, default=1e-3, help="Phase 1 encoder LR")
    p.add_argument("--lr-wm", type=float, default=1e-3, help="Phase 2 world model LR")
    p.add_argument("--lr-joint-encoder", type=float, default=1e-4, help="Phase 3 encoder LR")
    p.add_argument("--lr-joint-wm", type=float, default=1e-3, help="Phase 3 world model LR")
    p.add_argument(
        "--lr-chart-centers-scale",
        type=float,
        default=0.1,
        help="LR scale for chart_centers relative to the base encoder LR",
    )
    p.add_argument(
        "--lr-codebook-scale",
        type=float,
        default=0.5,
        help="LR scale for codebook parameters relative to the base encoder LR",
    )

    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--use-scheduler",
        action="store_true",
        help="Enable cosine LR scheduler for all phases (legacy flag)",
    )
    p.add_argument("--phase1-cosine-lr", action="store_true", help="Cosine anneal LR in Phase 1")
    p.add_argument(
        "--phase1-eta-min", type=float, default=1e-6, help="Minimum LR for Phase 1 cosine schedule"
    )
    p.add_argument("--phase2-cosine-lr", action="store_true", help="Cosine anneal LR in Phase 2")
    p.add_argument(
        "--phase2-eta-min", type=float, default=1e-6, help="Minimum LR for Phase 2 cosine schedule"
    )
    p.add_argument("--phase3-cosine-lr", action="store_true", help="Cosine anneal LR in Phase 3")
    p.add_argument(
        "--phase3-eta-min", type=float, default=1e-6, help="Minimum LR for Phase 3 cosine schedule"
    )

    # World model params
    p.add_argument(
        "--wm-hidden-dim",
        type=int,
        default=None,
        help="World model MLP width (defaults to --hidden-dim)",
    )
    p.add_argument("--wm-dt", type=float, default=0.01)
    p.add_argument("--wm-gamma-friction", type=float, default=1.0)
    p.add_argument("--wm-T-c", type=float, default=0.1)
    p.add_argument("--wm-alpha-potential", type=float, default=0.5)
    p.add_argument("--wm-beta-curl", type=float, default=0.1)
    p.add_argument("--wm-gamma-risk", type=float, default=0.01)
    p.add_argument("--wm-use-boris", action="store_true", default=True)
    p.add_argument("--wm-no-boris", dest="wm_use_boris", action="store_false")
    p.add_argument("--wm-use-jump", action="store_true", default=True)
    p.add_argument("--wm-no-jump", dest="wm_use_jump", action="store_false")
    p.add_argument(
        "--wm-refine-steps",
        type=int,
        default=3,
        help="Number of BAOAB sub-steps per horizon step (WFR W2 component)",
    )
    p.add_argument(
        "--wm-jump-beta",
        type=float,
        default=1.0,
        help="Inverse temperature for Boltzmann chart selection",
    )
    p.add_argument(
        "--wm-min-length",
        type=float,
        default=0.03,
        help="Minimum geodesic length scale (derives F_max, V_alg, cf_max; "
        "0=off, -1=auto-measure from encoder after Phase 1)",
    )
    p.add_argument(
        "--wm-d-model", type=int, default=128, help="CovariantAttention width for world model"
    )
    p.add_argument(
        "--wm-risk-metric-alpha",
        type=float,
        default=None,
        help="Risk-metric coupling alpha; None uses config default (0=off)",
    )

    # Phase 3 scaling
    p.add_argument("--phase3-encoder-scale", type=float, default=0.1)
    p.add_argument("--phase3-dynamics-scale", type=float, default=1.0)
    p.add_argument(
        "--phase3-zn-reg-scale",
        type=float,
        default=0.1,
        help="Scale for z_n regularization in phase 3 (0=disable)",
    )

    # Encoder loss weights (same defaults as train_unsupervised)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-vq", type=float, default=1.0)
    p.add_argument("--w-entropy", type=float, default=0.3)
    p.add_argument("--w-consistency", type=float, default=0.0)
    p.add_argument(
        "--w-diversity",
        type=float,
        default=1.0,
        help="Weight for hard/ST chart-usage entropy band",
    )
    p.add_argument(
        "--chart-usage-h-low",
        type=float,
        default=None,
        help="Minimum hard chart-usage entropy (None = auto from num_charts)",
    )
    p.add_argument(
        "--chart-usage-h-high",
        type=float,
        default=None,
        help="Optional maximum hard chart-usage entropy",
    )
    p.add_argument(
        "--w-chart-ot",
        type=float,
        default=1.0,
        help="Entropic OT chart-balancing auxiliary weight",
    )
    p.add_argument(
        "--chart-ot-epsilon",
        type=float,
        default=0.05,
        help="Entropic OT epsilon for chart balancing",
    )
    p.add_argument(
        "--chart-ot-iters",
        type=int,
        default=20,
        help="Number of Sinkhorn iterations for chart balancing",
    )
    p.add_argument("--w-uniformity", type=float, default=0.05)
    p.add_argument("--w-radial-cal", type=float, default=0.1)
    p.add_argument(
        "--w-confidence-calibration",
        type=float,
        default=0.05,
        help="Align router confidence with per-sample reconstruction quality",
    )
    p.add_argument(
        "--w-hard-routing-nll",
        type=float,
        default=0.5,
        help="Sharpen the deterministic hard chart partition in score space",
    )
    p.add_argument(
        "--w-router-margin",
        type=float,
        default=2.0,
        help="Weight for enforcing a positive hard-routing score margin",
    )
    p.add_argument(
        "--router-margin-target",
        type=float,
        default=0.05,
        help="Minimum desired gap between the winning and runner-up chart scores",
    )
    p.add_argument(
        "--radial-quality-alpha",
        type=float,
        default=2.0,
        help="Sharpness of the reconstruction-quality target used by radial calibration",
    )
    p.add_argument(
        "--radial-vq-alpha",
        type=float,
        default=1.0,
        help="Sharpness of the VQ-quality target used by radial calibration",
    )
    p.add_argument(
        "--radial-quality-rank-mix",
        type=float,
        default=0.75,
        help="Blend from absolute error quality (0) to batch-rank quality (1)",
    )
    p.add_argument(
        "--radial-recon-quality-weight",
        type=float,
        default=0.7,
        help="Weight of reconstruction quality when combining recon and VQ targets",
    )
    p.add_argument(
        "--radial-quality-mix",
        type=float,
        default=1.0,
        help="Blend from confidence-only radial targets (0) to quality-gated targets (1)",
    )
    p.add_argument(
        "--radial-quality-base-weight",
        type=float,
        default=0.0,
        help="Optional quality-driven base shell weight (default off for theory-aligned runs)",
    )
    p.add_argument(
        "--radial-calibration-rho-max",
        type=float,
        default=4.0,
        help="Maximum hyperbolic radius assigned to confident, low-error samples",
    )
    p.add_argument(
        "--radial-calibration-band-width",
        type=float,
        default=0.75,
        help="Half-width of the acceptable local hyperbolic-radius band",
    )
    p.add_argument(
        "--w-v-tangent-barrier",
        type=float,
        default=0.01,
        help="Weight for the pre-squash tangent barrier on v_raw",
    )
    p.add_argument(
        "--v-tangent-barrier-radius",
        type=float,
        default=0.9,
        help="Projected-radius threshold where the v_raw barrier activates",
    )
    p.add_argument("--w-codebook-spread", type=float, default=0.05)
    p.add_argument("--w-codebook-center", type=float, default=0.02)
    p.add_argument(
        "--w-chart-center-mean",
        type=float,
        default=0.02,
        help="Weight for tangent-space atlas barycenter anchoring",
    )
    p.add_argument(
        "--w-chart-center-radius",
        type=float,
        default=0.05,
        help="Weight for hyperbolic safe-harbor radius regularization",
    )
    p.add_argument(
        "--chart-center-radius-max",
        type=float,
        default=2.0,
        help="Maximum hyperbolic radius before chart centers are penalized",
    )
    p.add_argument(
        "--w-chart-center-sep",
        type=float,
        default=0.02,
        help="Weight for hyperbolic chart-center separation",
    )
    p.add_argument(
        "--chart-center-sep-margin",
        type=float,
        default=1.0,
        help="Minimum pairwise hyperbolic separation between chart centers",
    )
    p.add_argument(
        "--w-chart-collapse",
        type=float,
        default=0.0,
        help="Deprecated; no longer part of the active Phase 1 stack",
    )
    p.add_argument(
        "--w-code-collapse",
        type=float,
        default=0.5,
        help="Weight for hard/ST per-chart code-usage entropy band",
    )
    p.add_argument(
        "--code-usage-h-low",
        type=float,
        default=None,
        help="Minimum per-chart code-usage entropy (None = auto from codes_per_chart)",
    )
    p.add_argument(
        "--code-usage-h-high",
        type=float,
        default=None,
        help="Optional maximum per-chart code-usage entropy",
    )
    p.add_argument(
        "--code-usage-temperature",
        type=float,
        default=1.0,
        help="Soft/ST temperature used for code-usage regularization",
    )
    p.add_argument("--w-window", type=float, default=0.0)
    p.add_argument("--w-jump", type=float, default=0.0)
    p.add_argument("--w-jump-warmup", type=int, default=20)
    p.add_argument("--w-jump-ramp-end", type=int, default=50)
    p.add_argument(
        "--phase1-adaptive-multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive multipliers for Phase 1 routing losses",
    )
    p.add_argument(
        "--phase1-multiplier-max",
        type=float,
        default=8.0,
        help="Maximum adaptive multiplier scale in Phase 1",
    )
    p.add_argument(
        "--phase1-multiplier-decay",
        type=float,
        default=0.05,
        help="Relaxation rate back toward base weights when constraints are satisfied",
    )
    p.add_argument(
        "--conf-target-top1",
        type=float,
        default=0.55,
        help="Target mean soft top-1 routing probability",
    )
    p.add_argument(
        "--conf-multiplier-lr",
        type=float,
        default=1.5,
        help="Adaptive multiplier update rate for routing confidence",
    )
    p.add_argument(
        "--chart-multiplier-lr",
        type=float,
        default=1.0,
        help="Adaptive multiplier update rate for hard chart usage",
    )
    p.add_argument(
        "--chart-ot-i-target",
        type=float,
        default=0.35,
        help="Target soft mutual information for OT balancing pressure",
    )
    p.add_argument(
        "--chart-ot-multiplier-lr",
        type=float,
        default=1.0,
        help="Adaptive multiplier update rate for OT chart balancing",
    )
    p.add_argument(
        "--code-usage-gate-h",
        type=float,
        default=1.25,
        help="Enable code-usage pressure only after hard chart entropy exceeds this value",
    )
    p.add_argument(
        "--code-usage-ramp-epochs",
        type=int,
        default=50,
        help="Epochs to ramp code-usage pressure after the chart gate opens",
    )
    p.add_argument(
        "--code-multiplier-lr",
        type=float,
        default=0.5,
        help="Adaptive multiplier update rate for code usage",
    )

    # Dynamics codebook
    p.add_argument(
        "--dyn-codes-per-chart",
        type=int,
        default=0,
        help="Dynamics codebook codes per chart (0=disabled)",
    )
    p.add_argument(
        "--dyn-codebook-loss-weight", type=float, default=1.0, help="Dynamics codebook loss weight"
    )
    p.add_argument(
        "--dyn-commitment-beta", type=float, default=0.25, help="Dynamics codebook commitment beta"
    )
    p.add_argument(
        "--w-dyn-transition", type=float, default=0.5, help="DynamicsTransitionModel CE weight"
    )
    p.add_argument(
        "--dyn-transition-hidden-dim",
        type=int,
        default=128,
        help="Dynamics transition model MLP hidden dim",
    )
    p.add_argument(
        "--lr-dyn-codebook",
        type=float,
        default=1e-3,
        help="Phase 2/3 LR for dynamics codebook + transition model",
    )

    # Dynamics loss weights
    p.add_argument("--w-geodesic", type=float, default=1.0)
    p.add_argument("--w-chart-transition", type=float, default=0.5)
    p.add_argument("--w-momentum-reg", type=float, default=0.01)
    p.add_argument("--w-energy-conservation", type=float, default=0.01)
    p.add_argument(
        "--w-screened-poisson",
        type=float,
        default=0.0,
        help="Screened Poisson PDE residual weight; 0 = disabled",
    )
    p.add_argument(
        "--wm-screening-kappa",
        type=float,
        default=1.0,
        help="Screening mass kappa for screened Poisson loss",
    )
    p.add_argument(
        "--w-hodge", type=float, default=0.0, help="Hodge consistency loss weight; 0 = disabled"
    )

    # Supervised geodesic diffusion
    p.add_argument(
        "--wm-diffusion-substeps",
        type=int,
        default=8,
        help="Number of waypoints between z_t and z_{t+1}",
    )
    p.add_argument(
        "--w-position", type=float, default=1.0, help="Position loss weight (waypoint matching)"
    )
    p.add_argument(
        "--w-endpoint", type=float, default=2.0, help="Endpoint loss weight (z_N == z_{t+1})"
    )
    p.add_argument(
        "--w-momentum-target", type=float, default=0.1, help="Momentum supervision weight"
    )
    p.add_argument("--w-hodge-perp", type=float, default=0.01, help="Harmonic force penalty")
    p.add_argument(
        "--use-geodesic-diffusion",
        action="store_true",
        default=False,
        help="Use supervised geodesic diffusion for Phase 2/3 WM training",
    )

    # Orthogonality & enclosure
    p.add_argument(
        "--w-perp",
        type=float,
        default=0.01,
        help="Orthogonality loss weight (z_n vs z_tex decorrelation)",
    )
    p.add_argument(
        "--w-enclosure", type=float, default=0.0, help="Causal enclosure loss weight (0=disabled)"
    )
    p.add_argument("--enclosure-probe-hidden-dim", type=int, default=128)
    p.add_argument("--enclosure-probe-lr", type=float, default=3e-3)
    p.add_argument("--enclosure-grl-max-alpha", type=float, default=1.0)
    p.add_argument("--enclosure-grl-warmup-steps", type=int, default=5000)

    # Zeno loss (routing smoothness)
    p.add_argument(
        "--w-zeno",
        type=float,
        default=0.0,
        help="Zeno loss weight (routing distribution smoothness; 0=disabled)",
    )
    p.add_argument(
        "--zeno-mode",
        type=str,
        default="jsd",
        choices=["jsd", "kl"],
        help="Zeno divergence mode: 'jsd' (recommended) or 'kl'",
    )

    # Logging / saving / resume
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default="", help="Checkpoint path to resume from")
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    if args.wm_hidden_dim is None:
        args.wm_hidden_dim = args.hidden_dim
    train_joint(args)


if __name__ == "__main__":
    main()
