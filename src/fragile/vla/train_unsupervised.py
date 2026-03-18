"""Unsupervised TopoEncoder training on cached VLA features.

Follows the same loss-logging pattern as ``topoencoder_mnist.py``: split
encoder/decoder calls for full internal access, per-epoch detailed loss
display with tiers, info metrics, and training dynamics.
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
from fragile.layers import FactorizedJumpOperator, TopoEncoder
from fragile.layers.jump_operator import compute_jump_consistency_loss
from fragile.losses.encoder import (
    compute_phase1_loss,
    compute_router_information_metrics,
    compute_router_sharpness_metrics,
    get_jump_weight_schedule,
)
from fragile.vla.config import VLAConfig
from fragile.vla.extract_features import VLAFeatureDataset
from fragile.vla.optim import build_encoder_param_groups
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    phase1_effective_weight_scales,
    update_phase1_adaptive_state,
)


# ── Tracked loss terms ─────────────────────────────────────────────
LOSS_KEYS = [
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
]

# ── Info metrics (7) ───────────────────────────────────────────────
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
    "recon_quality_mean",
    "vq_quality_mean",
    "combined_quality_mean",
    "routing_confidence_mean",
    "radial_target_mean",
    "local_radius_mean",
    "grad_norm",
    "param_norm",
    "update_ratio",
    "lr",
]


def _init_accumulators() -> dict[str, float]:
    return dict.fromkeys(LOSS_KEYS + INFO_KEYS + ["total"], 0.0)


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


def _phase1_config_from_args(args: argparse.Namespace, phase1_state=None) -> VLAConfig:
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


# ── Main training function ─────────────────────────────────────────


def train_unsupervised(args: argparse.Namespace) -> None:  # noqa: C901
    """Train a TopoEncoder unsupervised on cached VLA features."""
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Dataset ────────────────────────────────────────────────
    dataset = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="train")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    input_dim = dataset[0]["feature"].shape[0]
    print(f"Dataset: {len(dataset)} frames, {len(loader)} batches (bs={args.batch_size})")
    print(f"Feature dim: {input_dim}")

    # ── Model ──────────────────────────────────────────────────
    K = args.num_charts
    model = TopoEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=K,
        codes_per_chart=args.codes_per_chart,
        soft_equiv_metric=True,
        film_conditioning=True,
    ).to(device)

    jump_op = FactorizedJumpOperator(
        num_charts=K,
        latent_dim=args.latent_dim,
    ).to(device)

    print(
        f"Model: {count_parameters(model):,} params | Jump: {count_parameters(jump_op):,} params"
    )

    # ── Optimizer & scheduler ──────────────────────────────────
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
    if args.use_scheduler:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Resume ─────────────────────────────────────────────────
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        if scheduler and "scheduler" in ckpt and ckpt["scheduler"] is not None:
            scheduler.load_state_dict(ckpt["scheduler"])
        print(f"Resumed from {args.resume} (epoch {start_epoch})")

    # ── Output dir ─────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Training loop ──────────────────────────────────────────
    phase1_state = init_phase1_adaptive_state(args)
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training"):
        model.train()
        jump_op.train()
        acc = _init_accumulators()
        n_batches = 0
        current_hard_routing = _use_hard_routing(args, epoch)
        current_tau = _get_hard_routing_tau(args, epoch, args.epochs)
        phase1_config = _phase1_config_from_args(args, phase1_state)

        for batch in loader:
            x = batch["feature"].to(device)

            # ── Encoder forward (split call) ───────────────
            (
                _K_chart,
                _K_code,
                _z_n,
                _z_tex,
                enc_w,
                z_geo,
                vq_loss,
                indices,
                z_n_all,
                c_bar,
                v_local,
                _z_q_blended,
            ) = model.encoder(
                x,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )

            # ── Decoder forward (dreaming mode) ────────────
            # When hard routing is on, pass encoder weights to decoder so both
            # use the same one-hot assignment (avoids consistency loss explosion).
            router_override = enc_w if current_hard_routing else None
            x_recon, dec_w, _aux_losses = model.decoder(
                z_geo,
                chart_index=None,
                router_weights=router_override,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )

            # ── Loss computation (shared Phase 1 stack) ─────
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
                indices_stack=indices,
                router_scores=getattr(model.encoder, "_last_router_scores_live", None),
            )

            # Jump (scheduled weight)
            current_jump_weight = get_jump_weight_schedule(
                epoch,
                warmup_end=args.w_jump_warmup,
                ramp_end=args.w_jump_ramp_end,
                final_weight=args.w_jump,
            )
            jump_loss, _jump_info = compute_jump_consistency_loss(
                z_n_all,
                enc_w,
                jump_op,
            )
            zn_reg_loss = zn_reg_loss + current_jump_weight * jump_loss

            # ── Total loss ─────────────────────────────────
            total = base_loss + zn_reg_loss

            # ── Gradient step ──────────────────────────────
            optimizer.zero_grad()
            total.backward()
            grad_norm = compute_grad_norm(all_params)
            param_norm = compute_param_norm(all_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0

            # ── Accumulate ─────────────────────────────────
            acc["total"] += total.item()
            for key in LOSS_KEYS:
                if key == "jump":
                    continue
                acc[key] += metrics.get(key, 0.0)
            acc["jump"] += jump_loss.item()
            for key in INFO_KEYS:
                if key in {"grad_norm", "param_norm", "update_ratio", "lr"}:
                    continue
                acc[key] += float(metrics.get(key, 0.0))
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        # ── Scheduler step ─────────────────────────────────
        if scheduler is not None:
            scheduler.step()

        # ── Average over batches ───────────────────────────
        for k in acc:
            acc[k] /= max(n_batches, 1)

        # ── Usage / perplexity (eval pass) ─────────────────
        model.eval()
        all_charts: list[torch.Tensor] = []
        all_router_weights: list[torch.Tensor] = []
        all_radii: list[torch.Tensor] = []
        all_code_indices: list[torch.Tensor] = []
        with torch.no_grad():
            for batch in loader:
                x_eval = batch["feature"].to(device)
                K_ch, _, _, _, enc_w, z_g, _, indices, *_ = model.encoder(x_eval)
                all_charts.append(K_ch.cpu())
                all_router_weights.append(enc_w.cpu())
                all_radii.append(z_g.cpu().norm(dim=-1))
                all_code_indices.append(indices.cpu())
        charts_t = torch.cat(all_charts)
        charts_np = charts_t.numpy()
        router_weights_t = torch.cat(all_router_weights)
        radii_np = torch.cat(all_radii).numpy()
        code_indices = torch.cat(all_code_indices)

        usage = np.zeros(K)
        for c in charts_np:
            usage[int(c)] += 1
        usage /= usage.sum() + 1e-8

        hard_entropy = float(-np.sum(usage * np.log(usage + 1e-8)))
        perplexity = float(np.exp(-np.sum(usage * np.log(usage + 1e-8))))
        soft_usage = router_weights_t.mean(dim=0).numpy()
        soft_usage /= soft_usage.sum() + 1e-8
        soft_perplexity = float(np.exp(-np.sum(soft_usage * np.log(soft_usage + 1e-8))))
        soft_active = int((soft_usage > 0.01).sum())
        soft_info = compute_router_information_metrics(router_weights_t)
        soft_sharpness = compute_router_sharpness_metrics(router_weights_t)
        mean_r = float(radii_np.mean())

        codes_per_chart = model.encoder.codebook.shape[1]
        unique_codes_per_chart = []
        code_entropy_per_chart = []
        code_perplexity_per_chart = []
        for c in range(K):
            mask = charts_t == c
            if mask.sum() > 0:
                codes_for_chart = code_indices[mask, c]
                unique_codes_per_chart.append(codes_for_chart.unique().numel())
                counts = torch.bincount(codes_for_chart, minlength=codes_per_chart).float()
                probs = counts / counts.sum().clamp(min=1.0)
                entropy = float(-(probs * torch.log(probs + 1e-8)).sum().item())
                code_entropy_per_chart.append(entropy)
                code_perplexity_per_chart.append(float(np.exp(entropy)))
            else:
                unique_codes_per_chart.append(0)
                code_entropy_per_chart.append(0.0)
                code_perplexity_per_chart.append(1.0)

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

        # ── Logging ────────────────────────────────────────
        should_log = (epoch % args.log_every == 0) or (epoch == args.epochs - 1)
        if should_log:
            print(f"Epoch {epoch:5d} | Loss: {acc['total']:.4f} | LR: {acc['lr']:.2e}")
            print(f"  Hard usage: {np.array2string(usage, precision=2, separator=', ')}")
            print(f"  Soft usage: {np.array2string(soft_usage, precision=2, separator=', ')}")
            print(
                f"  Core: recon={acc['recon']:.3f} vq={acc['vq']:.3f} "
                f"entropy={acc['entropy']:.3f} consist={acc['consistency']:.3f} "
                f"hard_nll={acc['hard_routing_nll']:.3f} "
                f"margin={acc['router_margin']:.3f}"
            )
            print(
                f"  Info: I_XK={acc['I_XK']:.3f} "
                f"H_K={acc['H_K']:.3f} H_K|X={acc['H_K_given_X']:.3f}"
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
            print(f"  Usage: chart={acc['chart_usage']:.4f} code={acc['code_usage']:.4f}")
            print(
                f"  OT: loss={acc['chart_ot']:.3f} "
                f"target_top1={acc['ot_target_top1_mean']:.3f} "
                f"col_l1={acc['ot_plan_col_l1']:.3e}"
            )
            print(
                f"  Train usage: H_hard={acc['H_usage']:.3f} "
                f"perp={acc['usage_perplexity']:.2f}/{K} "
                f"active={acc['usage_active']:.2f}/{K} "
                f"code_H={acc['H_code_usage']:.3f} "
                f"code_perp={acc['code_usage_perplexity']:.2f}/{args.codes_per_chart} "
                f"active_code_charts={acc['active_code_charts']:.2f}"
            )
            print(f"  Window: {acc['window']:.3f} (w={args.w_window:.3f})")
            print(f"  Jump: {acc['jump']:.3f} (lambda={current_jump_weight:.3f})")
            print(
                f"  Train: grad={acc['grad_norm']:.2e} "
                f"upd_ratio={acc['update_ratio']:.2e} "
                f"lr={acc['lr']:.2e}"
            )
            if phase1_state is not None:
                print(
                    f"  Ctrl: w_ent={phase1_config.w_entropy:.3f} "
                    f"w_chart={phase1_config.w_diversity:.3f} "
                    f"w_ot={phase1_config.w_chart_ot:.3f} "
                    f"w_code={phase1_config.w_code_collapse:.3f}"
                )
            print(
                f"  Metrics: hard_perplexity={perplexity:.2f}/{K} "
                f"hard_active={int((usage > 0.01).sum())}/{K} "
                f"soft_perplexity={soft_perplexity:.2f}/{K} "
                f"soft_active={soft_active}/{K} mean_r={mean_r:.3f}"
            )
            print(
                f"  Eval info: hard_H={hard_entropy:.3f} "
                f"soft_I_XK={soft_info['I_XK'].item():.3f} "
                f"soft_H_K={soft_info['H_K'].item():.3f} "
                f"soft_H_K|X={soft_info['H_K_given_X'].item():.3f}"
            )
            print(
                f"  Eval sharp: top1={soft_sharpness['top1_prob_mean'].item():.3f} "
                f"gap={soft_sharpness['top1_gap_mean'].item():.3f} "
                f"top2={soft_sharpness['top2_prob_mean'].item():.3f} "
                f"p10={soft_sharpness['top1_prob_p10'].item():.3f} "
                f"p90={soft_sharpness['top1_prob_p90'].item():.3f}"
            )
            print(
                f"  Code stats: H={code_entropy_mean_active:.3f} "
                f"perp={code_perplexity_mean_active:.2f}/{args.codes_per_chart}"
            )
            print(f"  Code util: {unique_codes_per_chart} / {codes_per_chart} per chart")
            print("-" * 60)

        if phase1_state is not None:
            update_phase1_adaptive_state(
                phase1_state,
                args,
                train_metrics=acc,
                eval_metrics={
                    "hard_entropy": hard_entropy,
                    "soft_I_XK": float(soft_info["I_XK"].item()),
                    "soft_top1_prob_mean": float(soft_sharpness["top1_prob_mean"].item()),
                    "code_entropy_mean_active": code_entropy_mean_active,
                },
                epoch=epoch,
            )

        # ── Checkpoint ─────────────────────────────────────
        should_save = (epoch > 0 and epoch % args.save_every == 0) or epoch == args.epochs - 1
        if should_save:
            ckpt_path = os.path.join(args.output_dir, f"epoch_{epoch:05d}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "jump_op": jump_op.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": (scheduler.state_dict() if scheduler else None),
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  Saved checkpoint: {ckpt_path}")

    # ── Final diagnostics ──────────────────────────────────
    print("\n=== Final Diagnostics ===")
    model.eval()
    all_features: list[torch.Tensor] = []
    all_ep_ids: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
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

    # Chart transition matrix
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


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    """CLI entry point for unsupervised VLA training."""
    p = argparse.ArgumentParser(
        description="Unsupervised TopoEncoder training on VLA features",
    )

    # Data / output
    p.add_argument(
        "--feature-cache-dir",
        default="outputs/vla/features",
        help="Path to cached VLA features",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/vla/unsupervised",
        help="Output directory for checkpoints and plots",
    )

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
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

    # Architecture
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)

    # Logging / saving
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)

    # Resume / device
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

    # ── Loss weights ───────────────────────────────────────
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

    args = p.parse_args()
    train_unsupervised(args)


if __name__ == "__main__":
    main()
