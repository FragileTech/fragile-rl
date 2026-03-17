"""Self-contained unified training: encoder + dynamics with a shared codebook.

Instead of the 3-phase pipeline (Phase 1 repr → Phase 2 dynamics → Phase 3
joint), this script trains a single ``SharedDynTopoEncoder`` where the
reconstruction codebook is simultaneously used for Markov-transition
prediction and zeno smoothness.  Dynamics are a first-class citizen from
epoch 0.

Run::

    uv run python -m fragile.vla.shared_dyn.train \\
        --output-dir /tmp/shared_dyn --phase1-epochs 20 \\
        --batch-size 64 --codes-per-chart 8 --hidden-dim 256 \\
        --hard-routing --w-dyn-transition 0.5 --w-zeno 0.1 --log-every 1
"""

from __future__ import annotations

import argparse
import os

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
from fragile.losses.macro import (
    compute_dynamics_markov_loss,
    DynamicsTransitionModel,
)
from fragile.vla.extract_features import VLAFeatureDataset
from fragile.vla.optim import build_encoder_param_groups
from fragile.vla.phase1_control import (
    init_phase1_adaptive_state,
    update_phase1_adaptive_state,
)
from fragile.vla.train_joint import (
    _compute_encoder_losses,
    _eval_pass,
    _get_hard_routing_tau,
    _phase1_config_from_args,
    _phase1_grad_breakdown,
    _run_diagnostics,
    _save_checkpoint,
    _use_hard_routing,
    DYN_SYMBOL_KEYS,
    ENCLOSURE_DIAG_KEYS,
    ENCODER_LOSS_KEYS,
    INFO_KEYS,
    ZENO_DIAG_KEYS,
)

from .encoder import SharedDynTopoEncoder


# ── Accumulators ──────────────────────────────────────────────────


def _init_shared_dyn_accumulators() -> dict[str, float]:
    keys = (
        list(ENCODER_LOSS_KEYS)
        + list(INFO_KEYS)
        + ["total"]
        + list(DYN_SYMBOL_KEYS)
        + list(ENCLOSURE_DIAG_KEYS)
        + ["encl_encoder", "encl_probe"]
        + list(ZENO_DIAG_KEYS)
        + ["zeno"]
    )
    return dict.fromkeys(keys, 0.0)


# ── Symbol usage statistics ───────────────────────────────────────


def _symbol_usage_stats(
    K_chart: torch.Tensor,
    K_code_dyn: torch.Tensor,
    num_charts: int,
    codes_per_chart: int,
) -> dict[str, float]:
    """Compute symbol (chart x code) usage statistics.

    Args:
        K_chart: [N] hard chart assignments (flattened over batch & time).
        K_code_dyn: [N] dynamics code assignments (same shape).
        num_charts: Total number of charts.
        codes_per_chart: Codes per chart (shared codebook size).

    Returns:
        Dict with symbol_active, symbol_entropy, symbol_perplexity,
        per_chart_code_entropy, per_chart_code_perplexity,
        per_chart_active_codes.
    """
    K_chart.numel()
    num_states = num_charts * codes_per_chart

    # Joint (chart, code) flat index
    flat_state = K_chart.long() * codes_per_chart + K_code_dyn.long()
    counts = torch.bincount(flat_state.reshape(-1), minlength=num_states).float()
    probs = counts / counts.sum().clamp(min=1.0)
    active = int((counts > 0).sum().item())
    entropy = -(probs * probs.clamp(min=1e-8).log()).sum().item()
    perplexity = float(np.exp(entropy))

    # Per-chart code usage
    per_chart_entropy = []
    per_chart_perplexity = []
    per_chart_active = []
    for c in range(num_charts):
        mask = K_chart == c
        if mask.sum() > 0:
            codes_c = K_code_dyn[mask]
            c_counts = torch.bincount(codes_c.long(), minlength=codes_per_chart).float()
            c_probs = c_counts / c_counts.sum().clamp(min=1.0)
            c_ent = -(c_probs * c_probs.clamp(min=1e-8).log()).sum().item()
            per_chart_entropy.append(c_ent)
            per_chart_perplexity.append(float(np.exp(c_ent)))
            per_chart_active.append(int((c_counts > 0).sum().item()))
        else:
            per_chart_entropy.append(0.0)
            per_chart_perplexity.append(0.0)
            per_chart_active.append(0)

    # Average over active charts
    active_charts = [i for i in range(num_charts) if per_chart_active[i] > 0]
    if active_charts:
        avg_code_ent = np.mean([per_chart_entropy[i] for i in active_charts])
        avg_code_perp = np.mean([per_chart_perplexity[i] for i in active_charts])
        avg_code_active = np.mean([per_chart_active[i] for i in active_charts])
    else:
        avg_code_ent = avg_code_perp = avg_code_active = 0.0

    return {
        "symbol_active": active,
        "symbol_total": num_states,
        "symbol_entropy": entropy,
        "symbol_perplexity": perplexity,
        "code_entropy_mean": float(avg_code_ent),
        "code_perplexity_mean": float(avg_code_perp),
        "code_active_mean": float(avg_code_active),
        "per_chart_active_codes": per_chart_active,
    }


# ── Test-set evaluation ──────────────────────────────────────────


@torch.no_grad()
def _test_eval_dynamics(
    model: SharedDynTopoEncoder,
    dyn_trans_model: DynamicsTransitionModel,
    test_seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    hard_routing: bool,
    hard_routing_tau: float,
) -> dict[str, float]:
    """Compute reconstruction + dynamics metrics on the held-out test set.

    Returns a flat dict with keys prefixed by ``test/``.
    """
    model.eval()
    dyn_trans_model.eval()

    recon_sum = 0.0
    vq_sum = 0.0
    dyn_ce_sum = 0.0
    dyn_acc_sum = 0.0
    dyn_zeno_sum = 0.0
    dyn_code_flip_sum = 0.0
    n_batches = 0
    all_K_chart: list[torch.Tensor] = []
    all_K_code_dyn: list[torch.Tensor] = []

    for batch in test_seq_loader:
        features = batch["features"].to(device)  # [B, H, D_feat]
        actions = batch["actions"].to(device)  # [B, H, A]
        B, H, D_feat = features.shape

        # Encode all frames
        flat = features.reshape(B * H, D_feat)
        (
            K_flat,
            _K_code_flat,
            _z_n_flat,
            _z_tex_flat,
            rw_flat,
            z_geo_flat,
            vq_loss_flat,
            _,
            _,
            c_bar_flat,
            v_local_flat,
            _z_q_flat,
        ) = model.encoder(
            flat,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )

        # Reconstruction loss on all frames (MSE between input and decoder output)
        router_override = rw_flat if hard_routing else None
        x_recon, _, _ = model.decoder(
            z_geo_flat,
            chart_index=None,
            router_weights=router_override,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        recon_loss = torch.nn.functional.mse_loss(x_recon, flat)
        recon_sum += recon_loss.item()
        vq_sum += vq_loss_flat.item()

        # Dynamics metrics on sequences
        if H > 1:
            v_local_all = v_local_flat.reshape(B, H, -1)
            rw_all = rw_flat.reshape(B, H, -1)
            c_bar_all = c_bar_flat.reshape(B, H, -1)
            K_all = K_flat.reshape(B, H)

            _, dyn_metrics, K_code_dyn_all = compute_dynamics_markov_loss(
                model.encoder,
                dyn_trans_model,
                v_local_all,
                rw_all,
                c_bar_all,
                K_all,
                actions,
                transition_weight=args.w_dyn_transition,
                zeno_weight=args.w_zeno,
                zeno_mode=getattr(args, "zeno_mode", "jsd"),
            )
            dyn_ce_sum += dyn_metrics.get("dyn_trans_ce", 0.0)
            dyn_acc_sum += dyn_metrics.get("dyn_trans_acc", 0.0)
            dyn_zeno_sum += dyn_metrics.get("dyn_zeno", 0.0)
            dyn_code_flip_sum += dyn_metrics.get("dyn_code_flip_rate", 0.0)

            if K_code_dyn_all is not None:
                all_K_chart.append(K_all.cpu())
                all_K_code_dyn.append(K_code_dyn_all.cpu())

        n_batches += 1

    n = max(n_batches, 1)
    result: dict[str, float] = {
        "test/recon": recon_sum / n,
        "test/vq": vq_sum / n,
        "test/dyn_trans_ce": dyn_ce_sum / n,
        "test/dyn_trans_acc": dyn_acc_sum / n,
        "test/dyn_zeno": dyn_zeno_sum / n,
        "test/dyn_code_flip_rate": dyn_code_flip_sum / n,
    }

    # Symbol usage on test set
    if all_K_chart:
        K_chart_cat = torch.cat(all_K_chart).reshape(-1)
        K_code_cat = torch.cat(all_K_code_dyn).reshape(-1)
        sym = _symbol_usage_stats(
            K_chart_cat,
            K_code_cat,
            num_charts=args.num_charts,
            codes_per_chart=args.codes_per_chart,
        )
        result["test/symbol_active"] = sym["symbol_active"]
        result["test/symbol_total"] = sym["symbol_total"]
        result["test/symbol_entropy"] = sym["symbol_entropy"]
        result["test/symbol_perplexity"] = sym["symbol_perplexity"]
        result["test/code_entropy_mean"] = sym["code_entropy_mean"]
        result["test/code_perplexity_mean"] = sym["code_perplexity_mean"]
        result["test/code_active_mean"] = sym["code_active_mean"]

    return result


@torch.no_grad()
def _train_symbol_usage(
    model: SharedDynTopoEncoder,
    seq_loader: DataLoader,
    args: argparse.Namespace,
    device: torch.device,
    hard_routing: bool,
    hard_routing_tau: float,
    max_batches: int = 50,
) -> dict[str, float]:
    """Compute symbol usage statistics on the training set (subsampled)."""
    model.eval()
    all_K_chart: list[torch.Tensor] = []
    all_K_code_dyn: list[torch.Tensor] = []

    for i, batch in enumerate(seq_loader):
        if i >= max_batches:
            break
        features = batch["features"].to(device)
        B, H, D_feat = features.shape
        flat = features.reshape(B * H, D_feat)

        K_flat, _, _, _, rw_flat, _, _, _, _, _, v_local_flat, _ = model.encoder(
            flat,
            hard_routing=hard_routing,
            hard_routing_tau=hard_routing_tau,
        )
        K_all = K_flat.reshape(B, H)
        v_local_all = v_local_flat.reshape(B, H, -1)
        rw_all = rw_flat.reshape(B, H, -1)

        # Get dynamics codes via shared codebook VQ
        K_code_list = []
        for t in range(H):
            _, K_code_t, _, _ = model.encoder.dynamics_vq(
                v_local_all[:, t],
                rw_all[:, t],
            )
            K_code_list.append(K_code_t)
        K_code_dyn = torch.stack(K_code_list, dim=1)  # [B, H]

        all_K_chart.append(K_all.cpu())
        all_K_code_dyn.append(K_code_dyn.cpu())

    if not all_K_chart:
        return {}

    K_chart_cat = torch.cat(all_K_chart).reshape(-1)
    K_code_cat = torch.cat(all_K_code_dyn).reshape(-1)
    return _symbol_usage_stats(
        K_chart_cat,
        K_code_cat,
        num_charts=args.num_charts,
        codes_per_chart=args.codes_per_chart,
    )


# ── Main training loop ───────────────────────────────────────────


def _run_unified(
    model: SharedDynTopoEncoder,
    jump_op: FactorizedJumpOperator,
    single_loader: DataLoader,
    seq_loader: DataLoader,
    dyn_trans_model: DynamicsTransitionModel,
    args: argparse.Namespace,
    device: torch.device,
    test_single_loader: DataLoader | None = None,
    test_seq_loader: DataLoader | None = None,
) -> dict[str, float]:
    """Unified encoder + dynamics training with shared codebook."""

    print("\n" + "=" * 60)
    print("Shared-dynamics unified training (encoder + Markov closure)")
    print("=" * 60)

    # -- Optimizer: encoder + jump_op + dynamics transition model ------
    encoder_groups = build_encoder_param_groups(
        model,
        jump_op,
        base_lr=args.lr,
        lr_chart_centers_scale=args.lr_chart_centers_scale,
        lr_codebook_scale=args.lr_codebook_scale,
    )
    # Add dynamics transition model parameters
    lr_dyn = getattr(args, "lr_dyn_transition", 3e-3)
    encoder_groups.append({
        "params": list(dyn_trans_model.parameters()),
        "lr": lr_dyn,
    })
    optimizer = torch.optim.Adam(encoder_groups)
    all_params = [p for g in optimizer.param_groups for p in g["params"]]

    scheduler = None
    if args.use_scheduler or args.phase1_cosine_lr:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.phase1_epochs,
            eta_min=args.phase1_eta_min,
        )

    K = args.num_charts
    total_epochs = args.phase1_epochs
    last_metrics: dict[str, float] = {}
    phase1_state = init_phase1_adaptive_state(args)

    for epoch in tqdm(range(total_epochs), desc="Unified"):
        model.train()
        jump_op.train()
        dyn_trans_model.train()

        acc = _init_shared_dyn_accumulators()
        n_batches = 0
        current_hard_routing = _use_hard_routing(args, epoch)
        current_tau = _get_hard_routing_tau(args, epoch, total_epochs)
        phase1_config = _phase1_config_from_args(args, phase1_state)

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)  # [B, H, A]
            B, H, D_feat = features.shape

            # ── 1. Full encoder losses on frame 0 ─────────────────
            (
                base_loss,
                zn_reg_loss,
                metrics,
                _z_geo_0,
                enc_w_0,
                K_ch_0,
                _zn_0,
                _ztex_0,
                c_bar_0,
                _K_code_0,
                _,
                v_local_0,
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

            # ── 2. Encode remaining frames (no reconstruction) ────
            if H > 1:
                rest = features[:, 1:, :].reshape(B * (H - 1), D_feat)
                (
                    K_rest,
                    _Kcode_rest,
                    _zn_rest,
                    _ztex_rest,
                    rw_rest,
                    _z_rest,
                    _,
                    _,
                    _,
                    c_bar_rest,
                    v_local_rest,
                    _,
                ) = model.encoder(
                    rest,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                )
                # Reshape to [B, H-1, ...]
                K_rest = K_rest.reshape(B, H - 1)
                rw_rest = rw_rest.reshape(B, H - 1, -1)
                c_bar_rest = c_bar_rest.reshape(B, H - 1, -1)
                v_local_rest = v_local_rest.reshape(B, H - 1, -1)

                # Assemble full-horizon tensors
                K_all = torch.cat([K_ch_0.unsqueeze(1), K_rest], dim=1)
                rw_all = torch.cat([enc_w_0.unsqueeze(1), rw_rest], dim=1)
                c_bar_all = torch.cat([c_bar_0.unsqueeze(1), c_bar_rest], dim=1)
                v_local_all = torch.cat([v_local_0.unsqueeze(1), v_local_rest], dim=1)
            else:
                K_all = K_ch_0.unsqueeze(1)
                rw_all = enc_w_0.unsqueeze(1)
                c_bar_all = c_bar_0.unsqueeze(1)
                v_local_all = v_local_0.unsqueeze(1)

            # ── 3. Dynamics Markov loss (transition CE + zeno) ────
            L_dyn = v_local_all.new_tensor(0.0)
            dyn_metrics: dict[str, float] = {}
            if H > 1:
                L_dyn, dyn_metrics, _ = compute_dynamics_markov_loss(
                    model.encoder,
                    dyn_trans_model,
                    v_local_all,
                    rw_all,
                    c_bar_all,
                    K_all,
                    actions,
                    transition_weight=args.w_dyn_transition,
                    zeno_weight=args.w_zeno,
                    zeno_mode=getattr(args, "zeno_mode", "jsd"),
                )

            # ── 4. Total loss ─────────────────────────────────────
            total = base_loss + zn_reg_loss + L_dyn

            # ── 5. Backward + step ────────────────────────────────
            optimizer.zero_grad()
            total.backward()
            metrics.update(_phase1_grad_breakdown(model))
            grad_norm = compute_grad_norm(all_params)
            param_norm = compute_param_norm(all_params)
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(all_params, args.grad_clip)
            optimizer.step()

            # ── 6. Accumulate ─────────────────────────────────────
            current_lr = optimizer.param_groups[0]["lr"]
            update_ratio = current_lr * grad_norm / (param_norm + 1e-12) if param_norm > 0 else 0.0

            acc["total"] += total.item()
            for k in ENCODER_LOSS_KEYS:
                acc[k] += metrics.get(k, 0.0)
            for k in INFO_KEYS:
                if k in {"grad_norm", "param_norm", "update_ratio", "lr"}:
                    continue
                acc[k] += metrics.get(k, 0.0)
            for k in DYN_SYMBOL_KEYS:
                acc[k] += dyn_metrics.get(k, 0.0)
            acc["grad_norm"] += grad_norm
            acc["param_norm"] += param_norm
            acc["update_ratio"] += update_ratio
            acc["lr"] += current_lr
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        for k in acc:
            acc[k] /= max(n_batches, 1)

        # ── Eval pass (train set) ─────────────────────────────────
        should_log = (epoch % args.log_every == 0) or (epoch == total_epochs - 1)
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
            hard_perplexity = soft_perplexity = mean_r = 0.0
            hard_active = soft_active = 0
            soft_usage = np.zeros(K, dtype=np.float64)
            extra = {}

        # ── Train symbol usage ────────────────────────────────────
        train_sym: dict[str, float] = {}
        if should_log:
            train_sym = _train_symbol_usage(
                model,
                seq_loader,
                args,
                device,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )

        # ── Eval pass (test set) ──────────────────────────────────
        test_metrics: dict[str, float] = {}
        if should_log and test_seq_loader is not None:
            test_metrics = _test_eval_dynamics(
                model,
                dyn_trans_model,
                test_seq_loader,
                args,
                device,
                hard_routing=current_hard_routing,
                hard_routing_tau=current_tau,
            )
            if test_single_loader is not None:
                (
                    _,
                    test_perplexity,
                    test_active,
                    _,
                    test_soft_perplexity,
                    test_soft_active,
                    test_mean_r,
                    _test_extra,
                ) = _eval_pass(
                    model,
                    test_single_loader,
                    K,
                    device,
                    hard_routing=current_hard_routing,
                    hard_routing_tau=current_tau,
                )
                test_metrics["test/perplexity"] = test_perplexity
                test_metrics["test/active"] = test_active
                test_metrics["test/soft_perplexity"] = test_soft_perplexity
                test_metrics["test/soft_active"] = test_soft_active
                test_metrics["test/mean_r"] = test_mean_r

        if should_log:
            print(
                f"E{epoch:5d} | Loss: {acc['total']:.4f} "
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
                f"  Dynamics: vq={acc['dyn_vq']:.4f} "
                f"closure_ce={acc['dyn_trans_ce']:.4f} "
                f"acc={acc['dyn_trans_acc']:.3f} "
                f"zeno={acc['dyn_zeno']:.4f}"
            )
            print(
                f"  Dyn diag: state_flip={acc['dyn_state_flip_rate']:.4f} "
                f"state_H={acc['dyn_state_entropy']:.4f} "
                f"state_max={acc['dyn_state_max_prob']:.4f} "
                f"code_flip={acc['dyn_code_flip_rate']:.4f}"
            )
            print(
                f"  Info: I_XK={acc['I_XK']:.3f} H_K={acc['H_K']:.3f} "
                f"H_K|X={acc['H_K_given_X']:.3f}"
            )
            print(
                f"  Sharp: top1={acc['top1_prob_mean']:.3f} "
                f"gap={acc['top1_gap_mean']:.3f} top2={acc['top2_prob_mean']:.3f}"
            )
            print(
                f"  Geo: unif={acc['uniformity']:.3f} "
                f"rad_cal={acc['radial_cal']:.3f} "
                f"cb_spread={acc['codebook_spread']:.3f} "
                f"cb_center={acc['codebook_center']:.3f}"
            )
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
            # ── Symbol usage (train) ──────────────────────────────
            CPC = args.codes_per_chart
            if train_sym:
                print(
                    f"  Symbols (train): "
                    f"active={train_sym['symbol_active']}/{train_sym['symbol_total']} "
                    f"H={train_sym['symbol_entropy']:.3f} "
                    f"perp={train_sym['symbol_perplexity']:.1f}"
                )
                print(
                    f"  Codes/chart (train): "
                    f"H={train_sym['code_entropy_mean']:.3f} "
                    f"perp={train_sym['code_perplexity_mean']:.1f}/{CPC} "
                    f"active={train_sym['code_active_mean']:.1f}/{CPC}"
                )
                pac = train_sym["per_chart_active_codes"]
                print(
                    f"  Per-chart active codes: {np.array2string(np.array(pac), separator=', ')}"
                )
            if extra.get("code_entropy_mean_active") is not None:
                print(
                    f"  VQ codes (eval): "
                    f"H={extra['code_entropy_mean_active']:.3f} "
                    f"perp={extra['code_perplexity_mean_active']:.1f}/{CPC}"
                )
            # ── Test metrics ──────────────────────────────────────
            if test_metrics:
                print(
                    f"  TEST recon={test_metrics['test/recon']:.4f} "
                    f"vq={test_metrics['test/vq']:.4f} "
                    f"dyn_ce={test_metrics['test/dyn_trans_ce']:.4f} "
                    f"dyn_acc={test_metrics['test/dyn_trans_acc']:.3f} "
                    f"zeno={test_metrics['test/dyn_zeno']:.4f} "
                    f"code_flip={test_metrics['test/dyn_code_flip_rate']:.4f}"
                )
                if "test/symbol_active" in test_metrics:
                    print(
                        f"  TEST symbols: "
                        f"active={test_metrics['test/symbol_active']:.0f}"
                        f"/{test_metrics['test/symbol_total']:.0f} "
                        f"H={test_metrics['test/symbol_entropy']:.3f} "
                        f"perp={test_metrics['test/symbol_perplexity']:.1f} "
                        f"code_H={test_metrics['test/code_entropy_mean']:.3f} "
                        f"code_perp={test_metrics['test/code_perplexity_mean']:.1f}/{CPC} "
                        f"code_active={test_metrics['test/code_active_mean']:.1f}/{CPC}"
                    )
                if "test/perplexity" in test_metrics:
                    print(
                        f"  TEST charts: "
                        f"perp={test_metrics['test/perplexity']:.2f}/{K} "
                        f"active={test_metrics['test/active']:.0f}/{K} "
                        f"soft_perp={test_metrics['test/soft_perplexity']:.2f}/{K} "
                        f"soft_active={test_metrics['test/soft_active']:.0f}/{K} "
                        f"mean_r={test_metrics['test/mean_r']:.3f}"
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

        should_save = (epoch > 0 and epoch % args.save_every == 0) or epoch == total_epochs - 1
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
                dyn_trans_model=dyn_trans_model,
            )

        last_metrics = acc

    return last_metrics


# ── Orchestrator ──────────────────────────────────────────────────


def train_shared_dyn(args: argparse.Namespace) -> None:
    """Entry point for shared-dynamics unified training."""

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Data loaders ──────────────────────────────────────────
    # Train
    single_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="train")
    single_loader = DataLoader(
        single_ds,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    input_dim = single_ds[0]["feature"].shape[0]
    print(f"Train single-frame: {len(single_ds)} frames, {len(single_loader)} batches")

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
        f"Train sequences: {len(seq_ds)} windows "
        f"(seq_len={args.sequence_length}), {len(seq_loader)} batches"
    )

    # Test
    test_single_ds = VLAFeatureDataset(args.feature_cache_dir, sequence_length=1, split="test")
    test_single_loader = DataLoader(
        test_single_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    test_seq_ds = VLAFeatureDataset(
        args.feature_cache_dir,
        sequence_length=args.sequence_length,
        split="test",
    )
    test_seq_loader = DataLoader(
        test_seq_ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    print(f"Test single-frame: {len(test_single_ds)} frames, {len(test_single_loader)} batches")
    print(f"Test sequences: {len(test_seq_ds)} windows, {len(test_seq_loader)} batches")
    print(f"Feature dim: {input_dim}")

    # ── Model ─────────────────────────────────────────────────
    K = args.num_charts
    model = SharedDynTopoEncoder(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_charts=K,
        codes_per_chart=args.codes_per_chart,
        covariant_attn=True,
        covariant_attn_tensorization="full",
        soft_equiv_metric=True,
        conv_backbone=False,
        film_conditioning=True,
        commitment_beta=args.commitment_beta,
        codebook_loss_weight=getattr(args, "codebook_loss_weight", 1.0),
    ).to(device)

    jump_op = FactorizedJumpOperator(
        num_charts=K,
        latent_dim=args.latent_dim,
    ).to(device)

    # Dynamics transition model — operates on the *main* codebook size
    dyn_codes = args.codes_per_chart  # shared codebook
    dyn_trans_model = DynamicsTransitionModel(
        chart_dim=args.latent_dim,
        action_dim=args.action_dim,
        num_charts=K,
        codes_per_chart=dyn_codes,
        hidden_dim=args.dyn_transition_hidden_dim,
    ).to(device)

    # ── Parameter breakdown ───────────────────────────────────
    n_enc = count_parameters(model.encoder)
    n_dec = count_parameters(model.decoder)
    n_jump = count_parameters(jump_op)
    n_dyn = count_parameters(dyn_trans_model)
    n_total = n_enc + n_dec + n_jump + n_dyn
    print(f"  Encoder:        {n_enc:>10,} params")
    print(f"  Decoder:        {n_dec:>10,} params")
    print(f"  Jump op:        {n_jump:>10,} params")
    print(f"  Dyn transition: {n_dyn:>10,} params")
    print(f"  TOTAL:          {n_total:>10,} params")
    print(
        f"  (codebook_dyn is None — dynamics uses main codebook "
        f"[{K} charts x {args.codes_per_chart} codes])"
    )

    # ── Resume ────────────────────────────────────────────────
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        jump_op.load_state_dict(ckpt["jump_op"])
        if ckpt.get("dyn_trans_model") is not None:
            dyn_trans_model.load_state_dict(ckpt["dyn_trans_model"])
        print(f"Resumed from {args.resume} (epoch {ckpt.get('epoch', '?')})")

    # ── Output dir ────────────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Warm-start chart centers ──────────────────────────────
    if not args.resume:
        if args.hard_routing:
            print(
                "\nSkipping k-means chart warm-start under hard routing; "
                "keeping quasi-uniform atlas initialization."
            )
        else:
            print("\nWarm-starting chart centers with k-means...")
            model.warmstart_chart_centers(
                single_loader,
                device,
                max_batches=10,
                radius_floor=0.0,
            )

    # ── Run unified training ──────────────────────────────────
    last_metrics = _run_unified(
        model,
        jump_op,
        single_loader,
        seq_loader,
        dyn_trans_model,
        args,
        device,
        test_single_loader=test_single_loader,
        test_seq_loader=test_seq_loader,
    )

    # ── Final checkpoint ──────────────────────────────────────
    final_path = os.path.join(args.output_dir, "checkpoint_final.pt")
    torch.save(
        {
            "epoch": -1,
            "phase": 1,
            "model": model.state_dict(),
            "jump_op": jump_op.state_dict(),
            "world_model": None,
            "dyn_trans_model": dyn_trans_model.state_dict(),
            "optimizer": None,
            "scheduler": None,
            "args": vars(args),
            "metrics": last_metrics,
        },
        final_path,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    # ── Diagnostics ───────────────────────────────────────────
    _run_diagnostics(model, single_loader, args, device, False, None)


# ── CLI ───────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="Unified encoder + dynamics training with shared codebook",
    )

    # Data / output
    p.add_argument("--feature-cache-dir", default="outputs/vla/features")
    p.add_argument("--output-dir", default="outputs/vla/shared_dyn")

    # Architecture
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--num-charts", type=int, default=16)
    p.add_argument("--codes-per-chart", type=int, default=64)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--sequence-length", type=int, default=8)
    p.add_argument("--hard-routing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--hard-routing-warmup-epochs", type=int, default=5)
    p.add_argument("--hard-routing-tau", type=float, default=1.0)
    p.add_argument("--hard-routing-tau-end", type=float, default=0.3)
    p.add_argument("--hard-routing-tau-anneal-epochs", type=int, default=200)

    # Epochs
    p.add_argument(
        "--phase1-epochs", type=int, default=100, help="Number of unified training epochs"
    )

    # Learning rates
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lr-chart-centers-scale", type=float, default=0.1)
    p.add_argument("--lr-codebook-scale", type=float, default=0.5)
    p.add_argument(
        "--lr-dyn-transition", type=float, default=3e-3, help="LR for dynamics transition model"
    )

    # Training
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--use-scheduler", action="store_true")
    p.add_argument("--phase1-cosine-lr", action="store_true")
    p.add_argument("--phase1-eta-min", type=float, default=1e-6)

    # Dynamics loss weights (shared codebook)
    p.add_argument(
        "--w-dyn-transition", type=float, default=0.5, help="Markov transition CE weight"
    )
    p.add_argument("--w-zeno", type=float, default=0.1, help="Zeno routing-smoothness weight")
    p.add_argument("--zeno-mode", type=str, default="jsd", choices=["jsd", "kl"])
    p.add_argument("--dyn-transition-hidden-dim", type=int, default=128)
    p.add_argument(
        "--commitment-beta", type=float, default=0.25, help="VQ commitment loss weight (beta)"
    )

    # Encoder loss weights (same as train_joint)
    p.add_argument("--w-recon", type=float, default=1.0)
    p.add_argument("--w-vq", type=float, default=1.0)
    p.add_argument("--w-entropy", type=float, default=0.3)
    p.add_argument("--w-consistency", type=float, default=0.0)
    p.add_argument("--w-diversity", type=float, default=1.0)
    p.add_argument("--chart-usage-h-low", type=float, default=None)
    p.add_argument("--chart-usage-h-high", type=float, default=None)
    p.add_argument("--w-chart-ot", type=float, default=1.0)
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

    # Adaptive multipliers
    p.add_argument(
        "--phase1-adaptive-multipliers", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--phase1-multiplier-max", type=float, default=8.0)
    p.add_argument("--phase1-multiplier-decay", type=float, default=0.05)
    p.add_argument("--conf-target-top1", type=float, default=0.55)
    p.add_argument("--conf-multiplier-lr", type=float, default=1.5)
    p.add_argument("--chart-multiplier-lr", type=float, default=1.0)
    p.add_argument("--chart-ot-i-target", type=float, default=0.35)
    p.add_argument("--chart-ot-multiplier-lr", type=float, default=1.0)
    p.add_argument("--code-usage-gate-h", type=float, default=1.25)
    p.add_argument("--code-usage-ramp-epochs", type=int, default=50)
    p.add_argument("--code-multiplier-lr", type=float, default=0.5)

    # Logging / saving / resume
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", default="")
    p.add_argument("--device", default="auto")

    args = p.parse_args()
    train_shared_dyn(args)


if __name__ == "__main__":
    main()
