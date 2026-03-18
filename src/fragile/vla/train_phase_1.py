"""Standalone Phase 1 VLA training.

This module extracts the encoder-only training path from ``train_joint.py``
and keeps only the code required for Phase 1.
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

from fragile.checkpoints import compute_grad_norm, compute_param_norm, count_parameters
from fragile.layers import FactorizedJumpOperator
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
from fragile.vla.config import VLAConfig
from fragile.vla.extract_features import VLAFeatureDataset
from fragile.vla.optim import build_encoder_param_groups
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    Phase1AdaptiveState,
    update_phase1_adaptive_state,
)


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


BALL_MAX_NORM = 0.99


def _init_encoder_accumulators() -> dict[str, float]:
    return dict.fromkeys(ENCODER_LOSS_KEYS + INFO_KEYS + ["total"], 0.0)


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


def _chart_stats_from_probs(
    router_weights: torch.Tensor,
    num_charts: int,
) -> tuple[np.ndarray, float, int]:
    rw = router_weights.detach().cpu().reshape(-1, num_charts)
    usage = rw.mean(dim=0).numpy()
    usage /= usage.sum() + 1e-8
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    return usage, perplexity, active


def _get_hard_routing_tau(args: argparse.Namespace, epoch: int, total_epochs: int) -> float:
    warmup_epochs = max(int(getattr(args, "hard_routing_warmup_epochs", 0) or 0), 0)
    effective_epoch = max(epoch - warmup_epochs, 0)
    effective_total_epochs = max(total_epochs - warmup_epochs, 1)
    if args.hard_routing_tau < 0:
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
    if not args.hard_routing:
        return False
    warmup_epochs = max(int(getattr(args, "hard_routing_warmup_epochs", 0) or 0), 0)
    return epoch >= warmup_epochs


def _phase1_config_from_args(
    args: argparse.Namespace,
    phase1_state: Phase1AdaptiveState | None = None,
) -> VLAConfig:
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
    ) = model.encoder(
        x,
        routing_tau=routing_tau,
    )

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
        router_scores=router_scores_live,
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


def _eval_pass(
    model: TopoEncoder,
    loader: DataLoader,
    num_charts: int,
    device: torch.device,
    *,
    hard_routing: bool = False,
    hard_routing_tau: float = 1.0,
) -> tuple[np.ndarray, float, int, np.ndarray, float, int, float, dict[str, float | list[int]]]:
    from fragile.layers.gauge import project_to_ball

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
                    float(model.encoder.soft_equiv_log_ratio_loss().detach().item())
                )

            codebook = project_to_ball(model.encoder.codebook)
            v_exp = v_local.unsqueeze(1).unsqueeze(2)
            cb_exp = codebook.unsqueeze(0)
            diff = v_exp - cb_exp
            dists_sq = (diff**2).sum(-1)
            min_dist = dists_sq.min(dim=-1).values
            weighted_dist = (min_dist * soft_router_weights).sum(dim=-1)
            all_vq_dists.append(weighted_dist.cpu())
            all_code_indices.append(indices.cpu())

    charts_t = torch.cat(all_charts)
    charts_np = charts_t.numpy()
    router_weights_t = torch.cat(all_soft_router_weights)
    radii_np = torch.cat(all_radii).numpy()
    vq_dists_np = torch.cat(all_vq_dists).numpy()
    code_indices = torch.cat(all_code_indices)
    router_scores_t = torch.cat(all_router_scores) if all_router_scores else None
    v_raw_norms_t = torch.cat(all_v_raw_norms) if all_v_raw_norms else None
    v_projected_norms_t = torch.cat(all_v_projected_norms) if all_v_projected_norms else None
    v_local_raw_norms_t = torch.cat(all_v_local_raw_norms) if all_v_local_raw_norms else None
    z_geo_raw_norms_t = torch.cat(all_z_geo_raw_norms) if all_z_geo_raw_norms else None

    usage = np.zeros(num_charts)
    for chart in charts_np:
        usage[int(chart)] += 1
    usage /= usage.sum() + 1e-8

    hard_entropy = float(-np.sum(usage * np.log(usage + 1e-8)))
    perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
    active = int((usage > 0.01).sum())
    soft_usage, soft_perplexity, soft_active = _chart_stats_from_probs(
        router_weights_t, num_charts
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

    codebook_raw_cpu = model.encoder.codebook.detach().cpu()
    codebook_cpu = project_to_ball(model.encoder.codebook).detach().cpu()
    cb_radii = codebook_cpu.norm(dim=-1)
    cb_raw_radii = codebook_raw_cpu.norm(dim=-1)
    chart_centers_raw_cpu = model.encoder.chart_centers.detach().cpu()
    chart_centers_cpu = project_to_ball(model.encoder.chart_centers).detach().cpu()
    cc_radii = chart_centers_cpu.norm(dim=-1)
    cc_raw_radii = chart_centers_raw_cpu.norm(dim=-1)

    codes_per_chart = codebook_cpu.shape[1]
    unique_codes_per_chart = []
    code_entropy_per_chart = []
    code_perplexity_per_chart = []
    for chart_idx in range(num_charts):
        mask = charts_t == chart_idx
        if mask.sum() > 0:
            codes_for_chart = code_indices[mask, chart_idx]
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
            np.mean(np.array(code_entropy_per_chart)[active_chart_mask]),
        )
        code_perplexity_mean_active = float(
            np.mean(np.array(code_perplexity_per_chart)[active_chart_mask]),
        )
    else:
        code_entropy_mean_active = 0.0
        code_perplexity_mean_active = 1.0

    extra: dict[str, float | list[int]] = {
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


def _save_checkpoint(
    args: argparse.Namespace,
    model: TopoEncoder,
    jump_op: FactorizedJumpOperator,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineAnnealingLR | None,
    epoch: int,
    metrics: dict[str, float],
) -> None:
    ckpt_path = os.path.join(args.output_dir, f"p1_epoch_{epoch:05d}.pt")
    torch.save(
        {
            "epoch": epoch,
            "phase": 1,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": None,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else None,
            "probe": None,
            "probe_optimizer": None,
            "dyn_trans_model": None,
            "args": vars(args),
            "metrics": metrics,
        },
        ckpt_path,
    )
    print(f"  Saved checkpoint: {ckpt_path}")


def _run_diagnostics(
    model: TopoEncoder,
    eval_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    print("\n=== Final Diagnostics ===")
    model.eval()

    all_features: list[torch.Tensor] = []
    all_ep_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in eval_loader:
            all_features.append(batch["feature"])
            all_ep_ids.append(batch["episode_id"])
    all_features_t = torch.cat(all_features)
    all_ep_ids_t = torch.cat(all_ep_ids)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    from fragile.vla.visualize import full_diagnostic, plot_chart_transitions

    config_ns = SimpleNamespace(num_charts=args.num_charts)
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
    print(f"\nAll outputs saved to {args.output_dir}")


def train_phase_1(args: argparse.Namespace) -> None:  # noqa: C901
    """Run Phase 1 encoder warmup only."""
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    dataset = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="train")
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    input_dim = dataset[0]["feature"].shape[0]
    print(f"Single-frame train dataset: {len(dataset)} frames, {len(train_loader)} batches")
    print(f"Feature dim: {input_dim}")

    model = TopoEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
        commitment_beta=getattr(args, "commitment_beta", 0.25),
        codebook_loss_weight=getattr(args, "codebook_loss_weight", 1.0),
    ).to(device)
    jump_op = FactorizedJumpOperator(
        num_charts=args.num_charts,
        latent_dim=args.latent_dim,
    ).to(device)

    n_enc = count_parameters(model.encoder)
    n_dec = count_parameters(model.decoder)
    n_jump = count_parameters(jump_op)
    print(f"  Encoder:  {n_enc:>10,} params")
    print(f"  Decoder:  {n_dec:>10,} params")
    print(f"  Jump op:  {n_jump:>10,} params")
    print(f"  TOTAL: {n_enc + n_dec + n_jump:>13,} params")

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
            T_max=args.epochs,
            eta_min=args.phase1_eta_min,
        )

    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        if ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if scheduler and ckpt.get("scheduler") is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = max(int(ckpt.get("epoch", -1)) + 1, 0)
        print(
            f"Resumed from {args.resume} "
            f"(phase {ckpt.get('phase', '?')}, epoch {ckpt.get('epoch', '?')})"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    if not args.resume:
        if args.hard_routing:
            print(
                "\nSkipping k-means chart warm-start under hard routing; "
                "keeping the encoder's quasi-uniform atlas initialization.",
            )
        elif hasattr(model, "warmstart_chart_centers"):
            print("\nWarm-starting chart centers with k-means...")
            model.warmstart_chart_centers(
                eval_loader,
                device,
                max_batches=10,
                radius_floor=0.0,
            )
        else:
            print("\nSkipping chart-center warm-start; this encoder does not expose that helper.")

    phase1_state = init_phase1_adaptive_state(args)
    last_metrics = _init_encoder_accumulators()
    last_batch_metrics: dict[str, float] = {}

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Phase 1"):
        model.train()
        jump_op.train()
        acc = _init_encoder_accumulators()
        n_batches = 0
        current_tau = _get_hard_routing_tau(args, epoch, args.epochs)
        phase1_config = _phase1_config_from_args(args, phase1_state)

        for batch in train_loader:
            x = batch["feature"].to(device)

            base_loss, zn_reg_loss, metrics, *_ = _compute_encoder_losses(
                x,
                model,
                jump_op,
                args,
                epoch,
                routing_tau=current_tau,
                phase1_config=phase1_config,
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
            last_batch_metrics = metrics

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0

            acc["total"] += metrics["total"]
            for key in ENCODER_LOSS_KEYS:
                acc[key] += metrics[key]
            for key in INFO_KEYS:
                if key in {"grad_norm", "param_norm", "update_ratio", "lr"}:
                    continue
                acc[key] += metrics.get(key, 0.0)
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for key in acc:
            acc[key] /= max(n_batches, 1)

        should_log = (epoch % args.log_every == 0) or (epoch == args.epochs - 1)
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
                eval_loader,
                args.num_charts,
                device,
                hard_routing=_use_hard_routing(args, epoch),
                hard_routing_tau=current_tau,
            )
        else:
            hard_usage = np.zeros(args.num_charts, dtype=np.float64)
            hard_perplexity = 0.0
            hard_active = 0
            soft_usage = np.zeros(args.num_charts, dtype=np.float64)
            soft_perplexity = 0.0
            soft_active = 0
            mean_r = 0.0
            extra = {}

        if should_log:
            print(
                f"P1 E{epoch:5d} | Loss: {acc['total']:.4f} | LR: {acc['lr']:.2e} | tau: {current_tau:.3f}"
            )
            print(f"  Hard usage: {np.array2string(hard_usage, precision=2, separator=', ')}")
            print(f"  Soft usage: {np.array2string(soft_usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['recon']:.3f} vq={acc['vq']:.3f} "
                f"entropy={acc['entropy']:.3f} consist={acc['consistency']:.3f} "
                f"chart_use={acc['chart_usage']:.3f} hard_nll={acc['hard_routing_nll']:.3f} "
                f"margin={acc['router_margin']:.3f}"
            )
            print(
                f"  Info: I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f}"
            )
            print(
                f"  Qual: recon_q={acc['recon_quality_mean']:.3f} "
                f"vq_q={acc['vq_quality_mean']:.3f} q={acc['combined_quality_mean']:.3f} "
                f"conf={acc['routing_confidence_mean']:.3f} target_r={acc['radial_target_mean']:.3f} "
                f"local_r={acc['local_radius_mean']:.3f}"
            )
            print(
                f"  Sharp: top1={acc['top1_prob_mean']:.3f} gap={acc['top1_gap_mean']:.3f} "
                f"top2={acc['top2_prob_mean']:.3f} p10={acc['top1_prob_p10']:.3f} "
                f"p90={acc['top1_prob_p90']:.3f}"
            )
            print(
                f"  Logits: gap50={acc['score_gap_p50']:.3f} gap90={acc['score_gap_p90']:.3f} "
                f"gap99={acc['score_gap_p99']:.3f} std={acc['score_std']:.3f} "
                f"abs={acc['score_mean_abs']:.3f} soft_equiv={acc['soft_equiv_log_ratio']:.3f}"
            )
            print(
                f"  Geo: unif={acc['uniformity']:.3f} rad_cal={acc['radial_cal']:.3f} "
                f"conf_cal={acc['confidence_calibration']:.3f} v_tan={acc['v_tangent_barrier']:.3f} "
                f"cb_spread={acc['codebook_spread']:.3f} cb_center={acc['codebook_center']:.3f} "
                f"cc_mean={acc['chart_center_mean']:.3f} cc_rad={acc['chart_center_radius']:.3f} "
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
                f"perp={acc['usage_perplexity']:.2f}/{args.num_charts} "
                f"active={acc['usage_active']:.2f}/{args.num_charts} "
                f"code_H={acc['H_code_usage']:.3f} "
                f"code_perp={acc['code_usage_perplexity']:.2f}/{args.codes_per_chart} "
                f"active_code_charts={acc['active_code_charts']:.2f}"
            )
            print(f"  Ortho: {acc['ortho']:.4f} (w={getattr(args, 'w_perp', 0.01):.3f})")
            print(f"  Window: {acc['window']:.3f} (w={args.w_window:.3f})")
            print(
                f"  Jump: {acc['jump']:.3f} (lambda={last_batch_metrics.get('jump_weight', 0.0):.3f})"
            )
            print(
                f"  Train: grad={acc['grad_norm']:.2e} upd_ratio={acc['update_ratio']:.2e} "
                f"lr={acc['lr']:.2e}"
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
                f"  Metrics: hard_perplexity={hard_perplexity:.2f}/{args.num_charts} "
                f"hard_active={hard_active}/{args.num_charts} "
                f"soft_perplexity={soft_perplexity:.2f}/{args.num_charts} "
                f"soft_active={soft_active}/{args.num_charts} mean_r={mean_r:.3f}"
            )
            print(
                f"  Eval info: hard_H={extra['hard_entropy']:.3f} "
                f"soft_I_XK={extra['soft_I_XK']:.3f} soft_H_K={extra['soft_H_K']:.3f} "
                f"soft_H_K|X={extra['soft_H_K_given_X']:.3f}"
            )
            print(
                f"  Eval sharp: top1={extra['soft_top1_prob_mean']:.3f} "
                f"gap={extra['soft_top1_gap_mean']:.3f} top2={extra['soft_top2_prob_mean']:.3f} "
                f"p10={extra['soft_top1_prob_p10']:.3f} p90={extra['soft_top1_prob_p90']:.3f}"
            )
            print(
                f"  Eval logits: gap50={extra['score_gap_p50']:.3f} "
                f"gap90={extra['score_gap_p90']:.3f} gap99={extra['score_gap_p99']:.3f} "
                f"std={extra['score_std']:.3f} abs={extra['score_mean_abs']:.3f} "
                f"soft_equiv={extra['soft_equiv_log_ratio']:.3f}"
            )
            print(
                f"  OT: loss={acc['chart_ot']:.3f} target_top1={acc['ot_target_top1_mean']:.3f} "
                f"col_l1={acc['ot_plan_col_l1']:.3e}"
            )
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
                f"  VQ dist: mean={extra['vq_dist_mean']:.4f} std={extra['vq_dist_std']:.4f} "
                f"p90={extra['vq_dist_p90']:.4f} p99={extra['vq_dist_p99']:.4f} "
                f"max={extra['vq_dist_max']:.4f}"
            )
            print(
                f"  Codebook: cb_r={extra['cb_r_mean']:.3f}±{extra['cb_r_std']:.3f} "
                f"(max={extra['cb_r_max']:.3f}) centers_r={extra['cc_r_mean']:.3f} "
                f"(max={extra['cc_r_max']:.3f}) raw_p99=({extra['cb_raw_r_p99']:.3f}, "
                f"{extra['cc_raw_r_p99']:.3f}) clip=({extra['cb_clip_frac']:.3f}, "
                f"{extra['cc_clip_frac']:.3f})"
            )
            print(
                f"  Code stats: H={extra['code_entropy_mean_active']:.3f} "
                f"perp={extra['code_perplexity_mean_active']:.2f}/{args.codes_per_chart}"
            )
            print(
                f"  Code util: {extra['codes_per_chart']} / "
                f"{extra['codes_per_chart_total']} per chart"
            )
            print("-" * 60)

        if phase1_state is not None:
            update_phase1_adaptive_state(
                phase1_state,
                args,
                train_metrics=acc,
                eval_metrics=extra,
                epoch=epoch,
            )

        should_save = (epoch > 0 and epoch % args.save_every == 0) or epoch == args.epochs - 1
        if should_save:
            _save_checkpoint(
                args,
                model,
                jump_op,
                optimizer,
                scheduler,
                epoch,
                acc,
            )

        last_metrics = acc

    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    torch.save(
        {
            "epoch": -1,
            "phase": 1,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": None,
            "probe": None,
            "dyn_trans_model": None,
            "optimizer": None,
            "scheduler": None,
            "args": vars(args),
            "metrics": last_metrics,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    _run_diagnostics(model, eval_loader, args, device)


def main() -> None:
    """CLI entry point for Phase 1-only VLA training."""
    p = argparse.ArgumentParser(description="Phase 1 VLA encoder training")

    p.add_argument("--feature-cache-dir", default="outputs/vla/features")
    p.add_argument("--output-dir", default="outputs/vla/phase1")

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
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
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument(
        "--cosine-lr",
        dest="phase1_cosine_lr",
        action="store_true",
        help="Cosine anneal LR during Phase 1",
    )
    p.add_argument(
        "--eta-min",
        dest="phase1_eta_min",
        type=float,
        default=1e-6,
        help="Minimum LR for cosine scheduling",
    )

    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)

    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default="", help="Checkpoint path to resume from")
    p.add_argument("--device", default="auto")

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
        help="Starting hard-routing temperature; negative means deterministic straight-through argmax",
    )
    p.add_argument(
        "--hard-routing-tau-end",
        type=float,
        default=0.3,
        help="Final tau after annealing (None = no annealing)",
    )
    p.add_argument(
        "--hard-routing-tau-anneal-epochs",
        type=int,
        default=200,
        help="Anneal tau linearly over this many epochs",
    )

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
    p.add_argument("--chart-usage-h-low", type=float, default=None)
    p.add_argument("--chart-usage-h-high", type=float, default=None)
    p.add_argument(
        "--w-chart-ot",
        type=float,
        default=1.0,
        help="Entropic OT chart-balancing auxiliary weight",
    )
    p.add_argument("--chart-ot-epsilon", type=float, default=0.05)
    p.add_argument("--chart-ot-iters", type=int, default=20)
    p.add_argument("--w-uniformity", type=float, default=0.05)
    p.add_argument("--w-radial-cal", type=float, default=0.1)
    p.add_argument("--w-confidence-calibration", type=float, default=0.05)
    p.add_argument("--w-hard-routing-nll", type=float, default=0.5)
    p.add_argument("--w-router-margin", type=float, default=2.0)
    p.add_argument("--router-margin-target", type=float, default=0.05)
    p.add_argument("--radial-quality-alpha", type=float, default=2.0)
    p.add_argument("--radial-vq-alpha", type=float, default=1.0)
    p.add_argument("--radial-quality-rank-mix", type=float, default=0.75)
    p.add_argument("--radial-recon-quality-weight", type=float, default=0.7)
    p.add_argument("--radial-quality-mix", type=float, default=1.0)
    p.add_argument("--radial-quality-base-weight", type=float, default=0.0)
    p.add_argument("--radial-calibration-rho-max", type=float, default=4.0)
    p.add_argument("--radial-calibration-band-width", type=float, default=0.75)
    p.add_argument("--w-v-tangent-barrier", type=float, default=0.01)
    p.add_argument("--v-tangent-barrier-radius", type=float, default=0.9)
    p.add_argument("--w-codebook-spread", type=float, default=0.05)
    p.add_argument("--w-codebook-center", type=float, default=0.02)
    p.add_argument("--w-chart-center-mean", type=float, default=0.02)
    p.add_argument("--w-chart-center-radius", type=float, default=0.05)
    p.add_argument("--chart-center-radius-max", type=float, default=2.0)
    p.add_argument("--w-chart-center-sep", type=float, default=0.02)
    p.add_argument("--chart-center-sep-margin", type=float, default=1.0)
    p.add_argument("--w-chart-collapse", type=float, default=0.0)
    p.add_argument("--w-code-collapse", type=float, default=0.5)
    p.add_argument("--code-usage-h-low", type=float, default=None)
    p.add_argument("--code-usage-h-high", type=float, default=None)
    p.add_argument("--code-usage-temperature", type=float, default=1.0)
    p.add_argument("--w-window", type=float, default=0.0)
    p.add_argument("--w-window-eps-ground", type=float, default=0.1)
    p.add_argument("--w-jump", type=float, default=0.0)
    p.add_argument("--w-jump-warmup", type=int, default=20)
    p.add_argument("--w-jump-ramp-end", type=int, default=50)
    p.add_argument("--w-perp", type=float, default=0.01)

    p.add_argument(
        "--adaptive-multipliers",
        dest="phase1_adaptive_multipliers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable adaptive multipliers for routing losses",
    )
    p.add_argument(
        "--multiplier-max",
        dest="phase1_multiplier_max",
        type=float,
        default=8.0,
        help="Maximum adaptive multiplier scale",
    )
    p.add_argument(
        "--multiplier-decay",
        dest="phase1_multiplier_decay",
        type=float,
        default=0.05,
        help="Relaxation rate back toward base weights when constraints are satisfied",
    )
    p.add_argument("--conf-target-top1", type=float, default=0.55)
    p.add_argument("--conf-multiplier-lr", type=float, default=1.5)
    p.add_argument("--chart-multiplier-lr", type=float, default=1.0)
    p.add_argument("--chart-ot-i-target", type=float, default=0.35)
    p.add_argument("--chart-ot-multiplier-lr", type=float, default=1.0)
    p.add_argument("--code-usage-gate-h", type=float, default=1.25)
    p.add_argument("--code-usage-ramp-epochs", type=int, default=50)
    p.add_argument("--code-multiplier-lr", type=float, default=0.5)

    args = p.parse_args()
    train_phase_1(args)


if __name__ == "__main__":
    main()
