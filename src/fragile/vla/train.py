"""3-phase training loop for the TopoEncoder x SmolVLA experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, fields
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from fragile.core.layers import TopoEncoder
from fragile.core.layers.topology import FactorizedJumpOperator
from fragile.hyperbolic_losses import (
    compute_jump_consistency_loss as compute_jump_consistency_loss_hyp,
    get_jump_weight_schedule,
)

from .config import VLAConfig
from .covariant_world_model import GeometricWorldModel
from .extract_features import VLAFeatureDataset
from .losses import (
    compute_dyn_transition_loss,
    compute_dynamics_markov_loss,
    compute_enclosure_loss,
    compute_phase1_loss,
    compute_phase2_geodesic_diffusion_loss,
    compute_phase2_loss,
    DynamicsTransitionModel,
    EnclosureProbe,
    grl_alpha_schedule,
    orthogonality_loss,
    zeno_loss,
)
from .optim import build_encoder_param_groups, get_codebook_like_params


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: str,
    encoder: nn.Module,
    jump_op: nn.Module,
    world_model: nn.Module | None,
    optimizers: dict,
    epoch: int,
    phase: int,
    config: VLAConfig,
    metrics: dict | None = None,
    probe: nn.Module | None = None,
    probe_optimizer: torch.optim.Optimizer | None = None,
    dyn_trans_model: nn.Module | None = None,
) -> None:
    """Save a VLA training checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "epoch": epoch,
        "phase": phase,
        "config": asdict(config),
        "encoder": {k: v.cpu() for k, v in encoder.state_dict().items()},
        "jump_op": {k: v.cpu() for k, v in jump_op.state_dict().items()},
        "world_model": (
            {k: v.cpu() for k, v in world_model.state_dict().items()}
            if world_model is not None
            else None
        ),
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
        "probe": (
            {k: v.cpu() for k, v in probe.state_dict().items()} if probe is not None else None
        ),
        "probe_optimizer": (probe_optimizer.state_dict() if probe_optimizer is not None else None),
        "dyn_trans_model": (
            {k: v.cpu() for k, v in dyn_trans_model.state_dict().items()}
            if dyn_trans_model is not None
            else None
        ),
        "metrics": metrics or {},
    }
    torch.save(payload, path)


def _log_metrics(
    metrics: dict[str, float],
    epoch: int,
    phase: int,
    mlflow_enabled: bool,
) -> None:
    """Print and optionally MLflow-log metrics."""
    parts = [f"P{phase} E{epoch:03d}"]
    for k, v in metrics.items():
        if k == "total":
            parts.insert(1, f"loss={v:.4f}")
        elif not k.startswith("H_") and not k.startswith("I_"):
            parts.append(f"{k}={v:.4f}")
    print(" | ".join(parts))

    if mlflow_enabled:
        try:
            from fragile.mlflow_logging import log_mlflow_metrics

            prefixed = {f"phase{phase}/{k}": v for k, v in metrics.items()}
            log_mlflow_metrics(prefixed, step=epoch, enabled=True)
        except ImportError:
            pass


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------


def _run_phase1(
    encoder: nn.Module,
    jump_op: nn.Module,
    train_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
) -> dict[str, float]:
    """Phase 1: Train encoder on single-frame features."""
    device = torch.device(config.device)
    encoder.to(device)
    jump_op.to(device)

    optimizer = optim.Adam(
        build_encoder_param_groups(
            encoder,
            jump_op,
            base_lr=config.lr_encoder,
            lr_chart_centers_scale=config.lr_chart_centers_scale,
            lr_codebook_scale=config.lr_codebook_scale,
        )
    )

    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.phase1_epochs + 1):
        encoder.train()
        jump_op.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["feature"].to(device)

            # Forward through encoder
            (
                x_recon,
                vq_loss,
                enc_router_weights,
                dec_router_weights,
                _K_chart,
                z_geo,
                _z_n,
                _c_bar,
                _aux_losses,
            ) = encoder(x)
            inner_enc = encoder.encoder if hasattr(encoder, "encoder") else encoder
            router_reg_weights = getattr(
                inner_enc,
                "_last_soft_router_weights_live",
                enc_router_weights,
            )
            v_local = getattr(inner_enc, "_last_v_local", None)

            # Phase 1 loss
            base_loss, zn_reg_loss, metrics = compute_phase1_loss(
                x,
                x_recon,
                vq_loss,
                enc_router_weights,
                dec_router_weights,
                z_geo,
                encoder,
                config,
                router_reg_weights=router_reg_weights,
                v_local=v_local,
            )
            loss = base_loss + zn_reg_loss

            # Inner encoder call for z_n/z_tex (orthogonality) and jump
            jump_w = get_jump_weight_schedule(
                epoch,
                warmup_end=config.w_jump_warmup,
                ramp_end=config.w_jump_ramp_end,
                final_weight=config.w_jump,
            )
            if (config.w_perp > 0 or jump_w > 0) and hasattr(encoder, "encoder"):
                enc_out = encoder.encoder(x)
                z_n_enc = enc_out[2]  # z_n
                z_tex_enc = enc_out[3]  # z_tex
                z_n_all_charts = enc_out[8]

                # Orthogonality loss
                if config.w_perp > 0:
                    loss_ortho = orthogonality_loss(z_n_enc, z_tex_enc)
                    loss = loss + config.w_perp * loss_ortho
                    metrics["ortho"] = loss_ortho.item()

                # Jump consistency loss (with warmup schedule)
                if jump_w > 0:
                    loss_jump = compute_jump_consistency_loss_hyp(
                        jump_op,
                        z_n_all_charts,
                        enc_router_weights,
                    )
                    loss = loss + jump_w * loss_jump
                    metrics["jump"] = loss_jump.item()

            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(jump_op.parameters()),
                    config.grad_clip,
                )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        last_metrics = metrics
        last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

        if epoch % config.log_every == 0 or epoch == config.phase1_epochs:
            _log_metrics(last_metrics, epoch, phase=1, mlflow_enabled=mlflow_enabled)

        if config.save_every > 0 and epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir,
                f"checkpoint_p1_e{epoch:04d}.pt",
            )
            _save_checkpoint(
                ckpt_path,
                encoder,
                jump_op,
                None,
                {"encoder_jump": optimizer},
                epoch,
                1,
                config,
                last_metrics,
            )

    return last_metrics


def _run_phase2(
    encoder: nn.Module,
    world_model: nn.Module,
    seq_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
) -> tuple[dict[str, float], DynamicsTransitionModel | None]:
    """Phase 2: Train world model with frozen encoder."""
    device = torch.device(config.device)
    encoder.to(device)
    world_model.to(device)
    inner_enc = encoder.encoder if hasattr(encoder, "encoder") else encoder
    phase1_centers = getattr(inner_enc, "chart_centers", None)
    if phase1_centers is not None and hasattr(world_model, "bind_chart_centers"):
        world_model.bind_chart_centers(phase1_centers, freeze=True)
    phase2_use_jump = world_model.use_jump
    world_model.use_jump = False

    # Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    # Dynamics codebook
    dyn_codes = getattr(config, "dyn_codes_per_chart", 0)
    dyn_trans_model = None
    inner_enc = encoder.encoder if hasattr(encoder, "encoder") else encoder
    if dyn_codes > 0 and inner_enc is not None and inner_enc.codebook_dyn is not None:
        inner_enc.codebook_dyn.requires_grad_(True)
        dyn_trans_model = DynamicsTransitionModel(
            chart_dim=config.latent_dim,
            action_dim=config.action_dim,
            num_charts=config.num_charts,
            dyn_codes_per_chart=dyn_codes,
            hidden_dim=getattr(config, "dyn_transition_hidden_dim", 128),
        ).to(device)
        lr_dyn = getattr(config, "lr_dyn_codebook", 1e-3)
        optimizer = optim.Adam([
            {"params": world_model.parameters(), "lr": config.lr_wm},
            {"params": [inner_enc.codebook_dyn], "lr": lr_dyn},
            {"params": dyn_trans_model.parameters(), "lr": lr_dyn},
        ])
    else:
        optimizer = optim.Adam(world_model.parameters(), lr=config.lr_wm)

    last_metrics: dict[str, float] = {}
    try:
        for epoch in range(1, config.phase2_epochs + 1):
            world_model.train()
            if dyn_trans_model is not None:
                dyn_trans_model.train()
            epoch_loss = 0.0
            n_batches = 0
            epoch_metric_totals: dict[str, float] = {}

            for batch in seq_loader:
                features = batch["features"].to(device)  # [B, H, D_feat]
                actions = batch["actions"].to(device)  # [B, H, A]

                _B, H, _ = features.shape

                # Encode all frames with the frozen Phase-1 atlas.
                with torch.no_grad():
                    z_list = []
                    rw_list = []
                    K_list = []
                    v_local_list = []
                    c_bar_list = []
                    for t in range(H):
                        (
                            _x_recon,
                            _vq_loss,
                            enc_rw,
                            _dec_rw,
                            K_chart,
                            z_geo,
                            _z_n,
                            c_bar,
                            _aux,
                        ) = encoder(features[:, t, :])
                        z_list.append(z_geo)
                        rw_list.append(enc_rw)
                        K_list.append(K_chart)
                        c_bar_list.append(c_bar)
                        if inner_enc is not None and hasattr(inner_enc, "_last_v_local"):
                            v_local_list.append(inner_enc._last_v_local)

                    z_all = torch.stack(z_list, dim=1)  # [B, H, D_lat]
                    K_all = torch.stack(K_list, dim=1)  # [B, H]
                    rw_all = torch.stack(rw_list, dim=1)  # [B, H, K]
                    c_bar_all = torch.stack(c_bar_list, dim=1)  # [B, H, D]
                    v_local_all = (
                        torch.stack(v_local_list, dim=1) if len(v_local_list) == H else None
                    )

                z_0 = z_all[:, 0, :]
                rw_0 = rw_all[:, 0]

                if getattr(config, "use_geodesic_diffusion", False):
                    loss, metrics = compute_phase2_geodesic_diffusion_loss(
                        world_model,
                        z_all,
                        rw_all,
                        K_all,
                        actions,
                        config,
                    )
                else:
                    pred_actions = actions[:, :-1, :]  # [B, H-1, A]
                    z_targets = z_all[:, 1:, :]  # [B, H-1, D]
                    chart_targets = K_all[:, 1:]  # [B, H-1]
                    wm_output = world_model(z_0, pred_actions, rw_0)
                    loss, metrics = compute_phase2_loss(
                        wm_output,
                        z_targets,
                        chart_targets,
                        config,
                    )

                if (
                    dyn_trans_model is not None
                    and inner_enc is not None
                    and v_local_all is not None
                    and H > 1
                ):
                    closure_weight = (
                        getattr(config, "w_enclosure", 0.0)
                        if getattr(config, "w_enclosure", 0.0) > 0
                        else getattr(config, "w_dyn_transition", 0.5)
                    )
                    dyn_symbol_loss, dyn_symbol_metrics, _ = compute_dynamics_markov_loss(
                        inner_enc,
                        dyn_trans_model,
                        v_local_all.detach(),
                        rw_all.detach(),
                        c_bar_all.detach(),
                        K_all,
                        actions,
                        transition_weight=closure_weight,
                        zeno_weight=getattr(config, "w_zeno", 0.0),
                        zeno_mode=getattr(config, "zeno_mode", "jsd"),
                    )
                    loss = loss + dyn_symbol_loss
                    metrics.update(dyn_symbol_metrics)

                optimizer.zero_grad()
                loss.backward()
                if config.grad_clip > 0:
                    all_params = list(world_model.parameters())
                    if dyn_trans_model is not None:
                        all_params += [inner_enc.codebook_dyn, *list(dyn_trans_model.parameters())]
                    nn.utils.clip_grad_norm_(all_params, config.grad_clip)
                optimizer.step()

                batch_metrics = dict(metrics)
                batch_metrics["total"] = loss.item()
                epoch_loss += loss.item()
                for key, value in batch_metrics.items():
                    epoch_metric_totals[key] = epoch_metric_totals.get(key, 0.0) + float(value)
                n_batches += 1

            last_metrics = {
                key: value / max(n_batches, 1) for key, value in epoch_metric_totals.items()
            }
            last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

            if epoch % config.log_every == 0 or epoch == config.phase2_epochs:
                _log_metrics(last_metrics, epoch, phase=2, mlflow_enabled=mlflow_enabled)

            if config.save_every > 0 and epoch % config.save_every == 0:
                ckpt_path = os.path.join(
                    config.output_dir,
                    f"checkpoint_p2_e{epoch:04d}.pt",
                )
                _save_checkpoint(
                    ckpt_path,
                    encoder,
                    FactorizedJumpOperator(
                        config.num_charts,
                        config.latent_dim,
                    ),
                    world_model,
                    {"wm": optimizer},
                    epoch,
                    2,
                    config,
                    last_metrics,
                    dyn_trans_model=dyn_trans_model,
                )
    finally:
        world_model.use_jump = phase2_use_jump
        for p in encoder.parameters():
            p.requires_grad_(True)

    return last_metrics, dyn_trans_model


def _run_phase3(
    encoder: nn.Module,
    jump_op: nn.Module,
    world_model: nn.Module,
    seq_loader: DataLoader,
    config: VLAConfig,
    mlflow_enabled: bool = False,
    dyn_trans_model: DynamicsTransitionModel | None = None,
) -> dict[str, float]:
    """Phase 3: Joint fine-tuning of encoder + world model."""
    device = torch.device(config.device)
    encoder.to(device)
    jump_op.to(device)
    world_model.to(device)

    # Separate optimizers for alternating optimization
    optimizer_enc = optim.Adam(
        build_encoder_param_groups(
            encoder,
            jump_op,
            base_lr=config.lr_joint_encoder,
            lr_chart_centers_scale=config.lr_chart_centers_scale,
            lr_codebook_scale=config.lr_codebook_scale,
        )
    )
    optimizer_wm = optim.Adam(
        world_model.parameters(),
        lr=config.lr_joint_wm,
    )

    # Codebook dynamics optimizer (Option D)
    w_codebook_dynamics = getattr(config, "w_codebook_dynamics", 0.0)
    optimizer_cb = None
    if (
        w_codebook_dynamics > 0
        and hasattr(encoder, "encoder")
        and hasattr(encoder.encoder, "codebook")
    ):
        cb_params = get_codebook_like_params(encoder)
        optimizer_cb = optim.Adam(
            cb_params,
            lr=config.lr_joint_encoder * config.lr_codebook_scale,
        )

    # Enclosure probe
    probe = None
    probe_optimizer = None
    if config.w_enclosure > 0:
        probe = EnclosureProbe(
            chart_dim=config.latent_dim,
            ztex_dim=config.latent_dim,
            action_dim=config.action_dim,
            num_charts=config.num_charts,
            codes_per_chart=config.codes_per_chart,
            hidden_dim=config.enclosure_probe_hidden_dim,
            alpha=0.0,
        ).to(device)
        probe_optimizer = torch.optim.Adam(
            probe.parameters(),
            lr=config.enclosure_probe_lr,
        )

    last_metrics: dict[str, float] = {}

    for epoch in range(1, config.phase3_epochs + 1):
        encoder.train()
        jump_op.train()
        world_model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in seq_loader:
            features = batch["features"].to(device)  # [B, H, D_feat]
            actions = batch["actions"].to(device)  # [B, H, A]

            _B, H, _D_feat = features.shape

            # Encode all frames (with gradients) using inner encoder
            z_list = []
            rw_list = []
            K_list = []
            Kcode_list = []
            zn_list = []
            ztex_list = []
            c_bar_list = []
            x_recon_0 = None
            vq_loss_0 = None
            enc_rw_0 = None
            dec_rw_0 = None
            z_geo_0 = None
            z_n_0 = None
            z_tex_0 = None
            router_reg_w_0 = None
            v_local_0 = None

            zq_blended_list = []
            v_local_list = []
            for t in range(H):
                enc_out_t = encoder.encoder(features[:, t, :])
                K_chart_t = enc_out_t[0]
                K_code_t = enc_out_t[1]
                z_n_t = enc_out_t[2]
                z_tex_t = enc_out_t[3]
                enc_rw_t = enc_out_t[4]
                z_geo_t = enc_out_t[5]
                vq_loss_t = enc_out_t[6]
                c_bar_t = enc_out_t[9]
                v_local_t = enc_out_t[10]
                zq_blended_t = enc_out_t[11]

                z_list.append(z_geo_t)
                rw_list.append(enc_rw_t)
                K_list.append(K_chart_t)
                Kcode_list.append(K_code_t)
                zn_list.append(z_n_t)
                ztex_list.append(z_tex_t)
                c_bar_list.append(c_bar_t)
                zq_blended_list.append(zq_blended_t)
                v_local_list.append(v_local_t)

                if t == 0:
                    # Decoder for reconstruction loss (z_tex not used)
                    x_recon_0, dec_rw_0, _ = encoder.decoder(z_geo_t)
                    vq_loss_0 = vq_loss_t
                    enc_rw_0 = enc_rw_t
                    z_geo_0 = z_geo_t
                    z_n_0 = z_n_t
                    z_tex_0 = z_tex_t
                    router_reg_w_0 = getattr(
                        encoder.encoder,
                        "_last_soft_router_weights_live",
                        enc_rw_t,
                    )
                    v_local_0 = v_local_t

            z_all = torch.stack(z_list, dim=1)
            K_all = torch.stack(K_list, dim=1)
            Kcode_all = torch.stack(Kcode_list, dim=1)  # [B, H]
            torch.stack(zn_list, dim=1)  # [B, H, D]
            ztex_all = torch.stack(ztex_list, dim=1)  # [B, H, D]
            c_bar_all = torch.stack(c_bar_list, dim=1)  # [B, H, D]

            # Encoder-side losses
            from .losses import compute_phase1_loss as _p1_loss

            base_enc, zn_reg, enc_metrics = _p1_loss(
                features[:, 0, :],
                x_recon_0,
                vq_loss_0,
                enc_rw_0,
                dec_rw_0,
                z_geo_0,
                encoder,
                config,
                router_reg_weights=router_reg_w_0,
                v_local=v_local_0,
            )
            metrics: dict[str, float] = {}
            for k, v in enc_metrics.items():
                metrics[f"enc/{k}"] = v

            # Orthogonality loss (add to base encoder loss)
            if config.w_perp > 0 and z_n_0 is not None and z_tex_0 is not None:
                loss_ortho = orthogonality_loss(z_n_0, z_tex_0)
                base_enc = base_enc + config.w_perp * loss_ortho
                metrics["enc/ortho"] = loss_ortho.item()

            # Zeno loss (routing distribution smoothness)
            L_zeno = torch.zeros((), device=device)
            if config.w_zeno > 0 and H > 1:
                rw_all = torch.stack(rw_list, dim=1)  # [B, H, N_c]
                zeno_losses = []
                for t in range(1, H):
                    zeno_t = zeno_loss(rw_all[:, t], rw_all[:, t - 1], mode=config.zeno_mode)
                    zeno_losses.append(zeno_t)
                L_zeno = torch.stack(zeno_losses).mean()
                metrics["enc/zeno"] = L_zeno.item()

                with torch.no_grad():
                    flips = (K_all[:, 1:] != K_all[:, :-1]).float()
                    metrics["zeno/flip_rate"] = flips.mean().item()
                    ent = -(rw_all * rw_all.clamp(min=1e-8).log()).sum(dim=-1)
                    metrics["zeno/routing_entropy"] = ent.mean().item()
                    metrics["zeno/max_weight"] = rw_all.max(dim=-1).values.mean().item()

            # Enclosure loss
            L_encl_encoder = None
            L_encl_probe = None
            if probe is not None:
                global_step = epoch * len(seq_loader) + n_batches
                probe.grl.alpha.fill_(
                    grl_alpha_schedule(
                        global_step,
                        warmup_steps=config.enclosure_grl_warmup_steps,
                        max_alpha=config.enclosure_grl_max_alpha,
                    )
                )

                encl_enc_losses = []
                encl_probe_losses = []
                for t in range(H - 1):
                    le, lp, encl_diag = compute_enclosure_loss(
                        probe,
                        c_bar_all[:, t],
                        actions[:, t],
                        ztex_all[:, t],
                        K_all[:, t + 1],
                        K_code_t=Kcode_all[:, t],
                        K_code_tp1=Kcode_all[:, t + 1],
                        codes_per_chart=config.codes_per_chart,
                    )
                    encl_enc_losses.append(le)
                    encl_probe_losses.append(lp)

                L_encl_encoder = torch.stack(encl_enc_losses).mean()
                L_encl_probe = torch.stack(encl_probe_losses).mean()
                metrics["enc/encl_enc"] = L_encl_encoder.item()
                metrics["enc/encl_probe"] = L_encl_probe.item()
                metrics["enc/encl_defect"] = encl_diag["defect_acc"]

            # --- Encoder step (WM frozen) ---
            optimizer_enc.zero_grad()
            L_enc = config.phase3_encoder_scale * base_enc + config.phase3_zn_reg_scale * zn_reg
            if L_encl_encoder is not None:
                L_enc = L_enc + config.w_enclosure * L_encl_encoder
            if config.w_perp > 0:
                pass  # already added to base_enc above
            if config.w_zeno > 0 and H > 1:
                L_enc = L_enc + config.w_zeno * L_zeno
            L_enc.backward()
            enc_params = list(encoder.parameters()) + list(jump_op.parameters())
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(enc_params, config.grad_clip)
            optimizer_enc.step()

            # --- WM step (encoder detached) ---
            optimizer_wm.zero_grad()
            rw_all_det = torch.stack(rw_list, dim=1).detach()  # [B, H, K]
            z_all_det = z_all.detach()
            K_all_det = K_all.detach()

            if getattr(config, "use_geodesic_diffusion", False):
                dyn_loss, dyn_metrics = compute_phase2_geodesic_diffusion_loss(
                    world_model,
                    z_all_det,
                    rw_all_det,
                    K_all_det,
                    actions,
                    config,
                )
            else:
                pred_actions = actions[:, :-1, :]
                z_targets = z_all_det[:, 1:, :]
                chart_targets = K_all_det[:, 1:]
                rw_0 = rw_all_det[:, 0]
                wm_output = world_model(z_all_det[:, 0], pred_actions, rw_0)
                dyn_loss, dyn_metrics = compute_phase2_loss(
                    wm_output,
                    z_targets,
                    chart_targets,
                    config,
                )

            (config.phase3_dynamics_scale * dyn_loss).backward()
            if config.grad_clip > 0:
                nn.utils.clip_grad_norm_(world_model.parameters(), config.grad_clip)
            optimizer_wm.step()

            for k, v in dyn_metrics.items():
                metrics[f"dyn/{k}"] = v

            # Probe optimizer step (separate from main optimizers)
            if probe is not None and L_encl_probe is not None:
                probe_optimizer.zero_grad()
                L_encl_probe.backward()
                probe_optimizer.step()

            # --- Codebook dynamics step (Option D) ---
            if optimizer_cb is not None and H > 1:
                from fragile.core.layers.atlas import _project_to_ball, mobius_add
                from fragile.core.layers.gauge import hyperbolic_distance as _hyp_dist

                optimizer_cb.zero_grad()
                zq_blended_all = torch.stack(zq_blended_list, dim=1)  # [B, H, D]
                c_bar_all_t = torch.stack(c_bar_list, dim=1)  # [B, H, D]
                z_coarse_0 = _project_to_ball(
                    mobius_add(c_bar_all_t[:, 0].detach(), zq_blended_all[:, 0])
                )
                rw_0_cb = rw_list[0].detach()
                with torch.no_grad():
                    for p in world_model.parameters():
                        p.requires_grad_(False)
                wm_out_cb = world_model(z_coarse_0, actions[:, :-1, :], rw_0_cb)
                z_pred_cb = wm_out_cb["z_pred"]
                with torch.no_grad():
                    for p in world_model.parameters():
                        p.requires_grad_(True)
                L_cb_dyn = _hyp_dist(
                    z_pred_cb.reshape(-1, z_pred_cb.shape[-1]),
                    z_all_det[:, 1:].reshape(-1, z_all_det.shape[-1]),
                ).mean()
                # Dynamics codebook transition loss (Phase 3)
                if dyn_trans_model is not None and encoder.encoder.codebook_dyn is not None:
                    v_local_all_p3 = torch.stack(v_local_list, dim=1).detach()
                    rw_all_p3 = torch.stack(rw_list, dim=1).detach()
                    vq_dyn_losses_p3 = []
                    K_code_dyn_list_p3 = []
                    for t in range(H):
                        _, K_code_dyn_t, _, vq_dyn_t = encoder.encoder.dynamics_vq(
                            v_local_all_p3[:, t],
                            rw_all_p3[:, t],
                        )
                        vq_dyn_losses_p3.append(vq_dyn_t)
                        K_code_dyn_list_p3.append(K_code_dyn_t)
                    vq_dyn_loss_p3 = torch.stack(vq_dyn_losses_p3).mean()
                    K_code_dyn_all_p3 = torch.stack(K_code_dyn_list_p3, dim=1)

                    trans_losses_p3 = []
                    for t in range(H - 1):
                        t_loss, _ = compute_dyn_transition_loss(
                            dyn_trans_model,
                            c_bar_all_t[:, t].detach(),
                            actions[:, t],
                            K_code_dyn_all_p3[:, t],
                            K_all_det[:, t + 1],
                            K_code_dyn_all_p3[:, t + 1],
                        )
                        trans_losses_p3.append(t_loss)
                    trans_loss_p3 = torch.stack(trans_losses_p3).mean()
                    w_dyn_transition = getattr(config, "w_dyn_transition", 0.5)
                    L_dyn_cb_extra = vq_dyn_loss_p3 + w_dyn_transition * trans_loss_p3
                    L_dyn_cb_extra.backward()

                (w_codebook_dynamics * L_cb_dyn).backward()
                cb_clip_params = [encoder.encoder.codebook]
                if encoder.encoder.codebook_dyn is not None:
                    cb_clip_params.append(encoder.encoder.codebook_dyn)
                if config.grad_clip > 0:
                    nn.utils.clip_grad_norm_(cb_clip_params, config.grad_clip)
                optimizer_cb.step()

            total = L_enc.item() + config.phase3_dynamics_scale * dyn_loss.item()
            metrics["total"] = total
            epoch_loss += total
            n_batches += 1

        last_metrics = metrics
        last_metrics["epoch_loss"] = epoch_loss / max(n_batches, 1)

        if epoch % config.log_every == 0 or epoch == config.phase3_epochs:
            _log_metrics(last_metrics, epoch, phase=3, mlflow_enabled=mlflow_enabled)

        if config.save_every > 0 and epoch % config.save_every == 0:
            ckpt_path = os.path.join(
                config.output_dir,
                f"checkpoint_p3_e{epoch:04d}.pt",
            )
            _save_checkpoint(
                ckpt_path,
                encoder,
                jump_op,
                world_model,
                {"enc": optimizer_enc, "wm": optimizer_wm},
                epoch,
                3,
                config,
                last_metrics,
                probe=probe,
                probe_optimizer=probe_optimizer,
                dyn_trans_model=dyn_trans_model,
            )

    return last_metrics, probe


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def train_vla(config: VLAConfig) -> dict:
    """Run the full 3-phase VLA training pipeline.

    Returns:
        Dict with final metrics from each phase.
    """
    os.makedirs(config.output_dir, exist_ok=True)

    # MLflow setup
    mlflow_enabled = False
    if config.mlflow:
        try:
            import mlflow

            if config.mlflow_tracking_uri:
                mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            if config.mlflow_experiment:
                mlflow.set_experiment(config.mlflow_experiment)
            run_name = config.mlflow_run_name or "vla_train"
            mlflow.start_run(run_name=run_name)
            safe_params = {}
            for k, v in asdict(config).items():
                safe_params[k] = str(v) if not isinstance(v, (int, float, str, bool)) else v
            mlflow.log_params(safe_params)
            mlflow_enabled = True
        except ImportError:
            print("MLflow not available, skipping.")

    # --- Instantiate models ---
    print("Building encoder …")
    encoder = TopoEncoder(
        input_dim=config.input_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_charts=config.num_charts,
        codes_per_chart=config.codes_per_chart,
        covariant_attn=config.covariant_attn,
        covariant_attn_tensorization=config.covariant_attn_tensorization,
        covariant_attn_rank=config.covariant_attn_rank,
        covariant_attn_tau_min=config.covariant_attn_tau_min,
        covariant_attn_denom_min=config.covariant_attn_denom_min,
        covariant_attn_use_transport=config.covariant_attn_use_transport,
        covariant_attn_transport_eps=config.covariant_attn_transport_eps,
        soft_equiv_metric=config.soft_equiv_metric,
        soft_equiv_bundle_size=config.soft_equiv_bundle_size or None,
        soft_equiv_hidden_dim=config.soft_equiv_hidden_dim,
        soft_equiv_use_spectral_norm=config.soft_equiv_use_spectral_norm,
        soft_equiv_zero_self_mixing=config.soft_equiv_zero_self_mixing,
        soft_equiv_soft_assign=config.soft_equiv_soft_assign,
        soft_equiv_temperature=config.soft_equiv_temperature,
        conv_backbone=config.conv_backbone,
        commitment_beta=config.commitment_beta,
        codebook_loss_weight=config.codebook_loss_weight,
        dyn_codes_per_chart=config.dyn_codes_per_chart,
        dyn_commitment_beta=config.dyn_commitment_beta,
        dyn_codebook_loss_weight=config.dyn_codebook_loss_weight,
    )

    print("Building jump operator …")
    jump_op = FactorizedJumpOperator(
        num_charts=config.num_charts,
        latent_dim=config.latent_dim,
    )

    print("Building world model …")
    world_model = GeometricWorldModel(
        latent_dim=config.latent_dim,
        action_dim=config.action_dim,
        num_charts=config.num_charts,
        hidden_dim=config.wm_hidden_dim,
        dt=config.wm_dt,
        gamma_friction=config.wm_gamma_friction,
        T_c=config.wm_T_c,
        alpha_potential=config.wm_alpha_potential,
        beta_curl=config.wm_beta_curl,
        gamma_risk=config.wm_gamma_risk,
        use_boris=config.wm_use_boris,
        use_jump=config.wm_use_jump,
        n_refine_steps=config.wm_refine_steps,
        jump_beta=config.wm_jump_beta,
    )

    # Count parameters
    n_enc = sum(p.numel() for p in encoder.parameters())
    n_jump = sum(p.numel() for p in jump_op.parameters())
    n_wm = sum(p.numel() for p in world_model.parameters())
    print(
        f"Parameters: encoder={n_enc:,}  jump={n_jump:,}  wm={n_wm:,}  total={n_enc + n_jump + n_wm:,}"
    )

    # --- Data loaders ---
    print(f"Loading features from {config.feature_cache_dir} …")

    single_ds = VLAFeatureDataset(config.feature_cache_dir, sequence_length=1, split="train")
    single_loader = DataLoader(
        single_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    seq_ds = VLAFeatureDataset(
        config.feature_cache_dir,
        sequence_length=config.sequence_length,
        split="train",
    )
    seq_loader = DataLoader(
        seq_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    results: dict[str, dict] = {}

    # --- Warm-start chart centers ---
    print("\nWarm-starting chart centers with k-means...")
    encoder.warmstart_chart_centers(single_loader, config.device, max_batches=10)

    # --- Phase 1: Encoder training ---
    print("\n" + "=" * 60)
    print("Phase 1: Encoder training")
    print("=" * 60)
    results["phase1"] = _run_phase1(
        encoder,
        jump_op,
        single_loader,
        config,
        mlflow_enabled,
    )

    # --- Phase 2: World model training (encoder frozen) ---
    print("\n" + "=" * 60)
    print("Phase 2: World model training (encoder frozen)")
    print("=" * 60)
    p2_result = _run_phase2(
        encoder,
        world_model,
        seq_loader,
        config,
        mlflow_enabled,
    )
    if isinstance(p2_result, tuple):
        results["phase2"], dyn_trans_model_p2 = p2_result
    else:
        results["phase2"], dyn_trans_model_p2 = p2_result, None

    # --- Phase 3: Joint fine-tuning ---
    print("\n" + "=" * 60)
    print("Phase 3: Joint fine-tuning")
    print("=" * 60)
    results["phase3"], probe = _run_phase3(
        encoder,
        jump_op,
        world_model,
        seq_loader,
        config,
        mlflow_enabled,
        dyn_trans_model=dyn_trans_model_p2,
    )

    # Save final checkpoint
    final_path = os.path.join(config.output_dir, "checkpoint_final.pt")
    _save_checkpoint(
        final_path,
        encoder,
        jump_op,
        world_model,
        {},
        0,
        3,
        config,
        results.get("phase3"),
        probe=probe,
        dyn_trans_model=dyn_trans_model_p2,
    )
    print(f"\nFinal checkpoint saved to {final_path}")

    if mlflow_enabled:
        try:
            import mlflow

            mlflow.end_run()
        except ImportError:
            pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point: parse args into VLAConfig and run training."""
    parser = argparse.ArgumentParser(description="Train TopoEncoder x SmolVLA")

    # Add all VLAConfig fields as CLI arguments
    import dataclasses

    MISSING = dataclasses.MISSING
    for f in fields(VLAConfig):
        default = f.default if f.default is not MISSING else None
        if f.type is bool:
            parser.add_argument(
                f"--{f.name}",
                type=lambda x: x.lower() in {"true", "1", "yes"},
                default=default,
                help=f"{f.name} (default: {default})",
            )
        elif f.type is int:
            parser.add_argument(f"--{f.name}", type=int, default=default)
        elif f.type is float:
            parser.add_argument(f"--{f.name}", type=float, default=default)
        elif f.type is str:
            parser.add_argument(f"--{f.name}", type=str, default=default)

    args = parser.parse_args()
    config = VLAConfig(**{k: v for k, v in vars(args).items() if v is not None})
    train_vla(config)


if __name__ == "__main__":
    main()
