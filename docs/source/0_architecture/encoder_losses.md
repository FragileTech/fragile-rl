# Encoder Losses Reference

This document maps every loss function defined in `src/fragile/losses/encoder.py` to the training scripts and test files that actually call it. The central orchestrator is `compute_phase1_loss`, which assembles most individual losses into a single Phase 1 training step; some losses are called directly by training loops outside that function.

## Notation

| Abbreviation | File |
|---|---|
| **encoder.py** | `src/fragile/losses/encoder.py` |
| **train_joint.py** | `src/fragile/vla/train_joint.py` |
| **train_phase_1.py** | `src/fragile/vla/train_phase_1.py` |
| **train_unsupervised.py** | `src/fragile/vla/train_unsupervised.py` |
| **shared_dyn/train.py** | `src/fragile/vla/shared_dyn/train.py` |
| **topology.py** | `src/fragile/layers/topology.py` |

---

(sec-phase1-loss-assembly)=
## Phase 1 Loss Assembly (`compute_phase1_loss`)

`compute_phase1_loss` (encoder.py:1348) is the single function that wires together most individual losses into `base_loss` + `zn_reg_loss`. It is called by:

| Caller | Location |
|---|---|
| `_compute_encoder_losses` | train_joint.py:565 |
| `_compute_encoder_losses` | train_phase_1.py:375 |
| training loop | train_unsupervised.py:310 |
| tests | `tests/test_hyperbolic_losses.py` (lines 541, 551, 603, 666, 741, 835, 847, 923, 1007, 1021) |
| tests | `tests/test_train_joint_diagnostics.py` (lines 189, 201) |

The `_compute_encoder_losses` wrapper in train_joint.py and train_phase_1.py adds **jump consistency** and **orthogonality** losses on top of `compute_phase1_loss`, then returns all encoder outputs for downstream use. `shared_dyn/train.py` calls `_compute_encoder_losses` from train_joint.py.

---

(sec-losses-inside-phase1)=
## Losses Called Inside `compute_phase1_loss`

These losses are assembled automatically when `compute_phase1_loss` runs. They are gated by their corresponding `config.w_*` weight — a weight of 0 skips the loss entirely.

### Reconstruction

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Feature reconstruction (MSE) | `F.mse_loss` (inline) | `x` [B,D], `x_recon` [B,D] | — | `w_feature_recon` | `recon` |

### VQ

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| VQ loss | passed in from encoder | `vq_loss` [] (scalar) | — | `w_vq` | `vq` |

### Routing Sharpness

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Routing entropy $H(K \mid X)$ | `compute_routing_entropy` | `router_weights` [B,K] | `eps=1e-6` | `w_entropy` | `entropy` |
| Router margin (score gap) | `compute_router_margin_loss` | `router_scores` [B,K] | `margin` (from `router_margin_target`) | `w_router_margin` | `router_margin` |
| Hard routing NLL | `compute_hard_routing_nll` | `router_scores` [B,K] | — | `w_hard_routing_nll` | `hard_routing_nll` |
| Confidence calibration | `compute_confidence_calibration_loss` | `router_weights` [B,K], `quality_target` [B] | `num_charts`, `eps=1e-6` | `w_confidence_calibration` | `confidence_calibration` |

### Chart Balancing

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Chart usage band | `compute_chart_usage_band_loss` | `router_weights` [B,K] | `num_charts`, `h_low`, `h_high`, `eps=1e-6` | `w_diversity` | `chart_usage` |
| Sinkhorn OT chart balancing | `compute_sinkhorn_balanced_chart_loss` | `router_scores` [B,K] | `epsilon` (from `chart_ot_epsilon`), `num_iters` (from `chart_ot_iters`), `eps=1e-8` | `w_chart_ot` | `chart_ot` |

### Geometry Regularization

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Hyperbolic uniformity | `compute_hyperbolic_uniformity_loss` | `z_geo` [B,D] | `eps=1e-6` | `w_uniformity` | `uniformity` |
| Radial calibration | `compute_radial_calibration_loss` | `z_geo` [B,D], `router_weights` [B,K], `center_points` [B,D] (optional), `quality_target` [B] (optional) | `num_charts`, `quality_mix`, `quality_base_weight`, `rho_max` (from `radial_calibration_rho_max`), `rho_band_width` (from `radial_calibration_band_width`), `use_hyperbolic_radius=True`, `eps=1e-6` | `w_radial_calibration` | `radial_cal` |
| Pre-squash tangent barrier | `compute_v_tangent_barrier_loss` | `v_raw` [B,D] | `target_radius` (from `v_tangent_barrier_radius`), `max_norm=0.99` | `w_v_tangent_barrier` | `v_tangent_barrier` |

### Codebook & Chart Centers

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Codebook spread | `compute_codebook_spread_loss` | `codebook` [K,C,D] | `margin` (from `w_codebook_spread_margin`) | `w_codebook_spread` | `codebook_spread` |
| Codebook centering | `compute_codebook_centering_loss` | `codebook` [K,C,D] | — | `w_codebook_center` | `codebook_center` |
| Chart center mean | `compute_chart_center_mean_loss` | `chart_centers` [K,D] | — | `w_chart_center_mean` | `chart_center_mean` |
| Chart center radius | `compute_chart_center_radius_loss` | `chart_centers` [K,D] | `radius_max` (from `chart_center_radius_max`), `barrier_beta=4.0` | `w_chart_center_radius` | `chart_center_radius` |
| Chart center separation | `compute_chart_center_separation_loss` | `chart_centers` [K,D] | `margin` (from `chart_center_sep_margin`) | `w_chart_center_sep` | `chart_center_sep` |

### Code Usage

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Code usage band | `compute_code_usage_band_loss` | `v_local` [B,D], `codebook` [K,C,D], `router_weights` [B,K], `hard_code_indices` [B,K] (optional) | `h_low`, `h_high`, `temperature` (from `w_code_collapse_temperature`), `eps=1e-6` | `w_code_collapse` | `code_usage` |

### Encoder–Decoder Consistency

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| KL(enc ∥ dec) routing consistency | inline KL | `enc_router_weights` [B,K], `dec_router_weights` [B,K] | `eps=1e-6` | `w_consistency` | `consistency` |

### Information Window

| Loss | Function | Tensor inputs | Scalar params | Config weight | Metric key |
|---|---|---|---|---|---|
| Window loss $I(X;K) \geq \varepsilon$ | `compute_window_loss` | `router_weights` [B,K] | `num_charts`, `eps_ground` (from `w_window_eps_ground`), `eps=1e-6` | `w_window` | `window` |

### Info-Only Metrics (computed but not added to loss)

These are computed inside `compute_phase1_loss` for logging only:

| Function | Tensor inputs | Scalar params | Metric keys |
|---|---|---|---|
| `compute_router_information_metrics` | `router_weights` [B,K] | `eps=1e-6` | `H_K`, `H_K_given_X`, `I_XK` |
| `compute_router_sharpness_metrics` | `router_weights` [B,K] | — | `top1_prob_mean`, `top1_prob_p10`, `top1_prob_p90`, `top2_prob_mean`, `top1_gap_mean` |
| `compute_error_quality_targets` | `per_sample_error` [B] | `alpha` (from `radial_quality_alpha` or `radial_vq_alpha`), `eps=1e-6` | feeds `recon_quality_mean`, `vq_quality_mean` |
| `compute_rank_quality_targets` | `per_sample_error` [B] | — | feeds quality target blending |
| `mix_quality_targets` | `absolute_quality` [B], `rank_quality` [B] | `rank_mix` (from `radial_quality_rank_mix`) | blends absolute + rank quality |
| `combine_quality_targets` | `primary_quality` [B], `secondary_quality` [B] | `primary_weight` (from `radial_recon_quality_weight`) | `combined_quality_mean` |
| `compute_routing_confidence` | `router_weights` [B,K] | `num_charts`, `eps=1e-6` | `routing_confidence_mean` |

---

(sec-losses-outside-phase1)=
## Losses Called Outside `compute_phase1_loss`

These losses are added by the `_compute_encoder_losses` wrappers or directly by training loops, **after** `compute_phase1_loss` returns.

### Jump Consistency

| Function | Tensor inputs | Scalar params | Callers |
|---|---|---|---|
| `get_jump_weight_schedule` | — | `epoch`, `warmup_end`, `ramp_end`, `final_weight` | train_joint.py:582, train_phase_1.py:392, train_unsupervised.py:326 |
| `compute_jump_consistency_loss` (from `layers.topology`) | `z_n_all_charts` [B,K,D], `router_weights` [B,K], `jump_op` (module) | — | train_joint.py:589, train_phase_1.py:399, train_unsupervised.py:332 |

The jump loss is scheduled: zero during warmup, linearly ramped, then held at `w_jump`.

### Orthogonality ($z_n \perp z_{\text{tex}}$)

| Function | Tensor inputs | Scalar params | Callers |
|---|---|---|---|
| `orthogonality_loss` | `zn` [B,D], `ztex` [B,D] | — | train_joint.py:595 (weight: `w_perp`), train_phase_1.py:404 |

### Router Score Diagnostics

| Function | Tensor inputs | Scalar params | Callers |
|---|---|---|---|
| `compute_router_score_metrics` | `router_scores` [B,K] | — | train_joint.py:280, 732; train_phase_1.py:199, 537 |

These are logging-only calls outside the loss assembly.

### Deterministic ST Router Weights

| Function | Tensor inputs | Scalar params | Callers |
|---|---|---|---|
| `_deterministic_st_router_weights` | `router_scores` [B,K] | — | train_joint.py:562, train_phase_1.py:372 |

Used to build deterministic hard-assignment weights for utilization losses from live router scores.

---

(sec-eval-only-metrics)=
## Eval-Only Metric Functions

These are called during eval passes (not training) for logging:

| Function | Tensor inputs | Scalar params | Callers |
|---|---|---|---|
| `compute_router_information_metrics` | `router_weights` [B,K] | `eps=1e-6` | train_joint.py:727, train_phase_1.py:532, train_unsupervised.py:410 |
| `compute_router_sharpness_metrics` | `router_weights` [B,K] | — | train_joint.py:728, train_phase_1.py:533, train_unsupervised.py:411 |
| `compute_router_score_metrics` | `router_scores` [B,K] | — | train_joint.py:732, train_phase_1.py:537 |

---

(sec-unused-losses)=
## Unused Losses (Exported but Not Called)

These functions are defined and exported in `encoder.py` but have **no call sites** in any training script. They may be used for future experiments or are vestiges of earlier iterations.

| Function | Tensor inputs | Scalar params | Purpose |
|---|---|---|---|
| `compute_diversity_loss` | `router_weights` [B,K] | `num_charts`, `eps=1e-6` | Chart collapse via $\log K - H(K)$ (superseded by `compute_chart_usage_band_loss`) |
| `compute_code_entropy_loss` | `indices_stack` [B,K] | `num_codes` | Global code entropy maximization (superseded by `compute_code_usage_band_loss`) |
| `compute_per_chart_code_entropy_loss` | `indices_stack` [B,K], `K_chart` [B] | `num_charts`, `num_codes` | Per-chart code entropy via bincount (non-differentiable; superseded by band loss) |
| `compute_residual_scale_loss` | `z_n` [B,D] | `assume_tangent=True` | Penalize $\lVert z_n \rVert^2$ (dropped from active stack) |
| `compute_vq_geodesic_loss` | `z_q_all` [B,K,D], `v_local` [B,D], `router_weights` [B,K] | `commitment_cost=0.25` | VQ loss using hyperbolic distance (not wired into Phase 1) |
| `compute_hyperbolic_contrastive_loss` | `z_geo` [B,D], `labels` [B] | `margin=2.0` | Supervised contrastive in geodesic space (requires labels) |
| `compute_symbol_purity_loss` | `K_chart` [B], `indices_stack` [B,K], `labels` [B], `router_weights` [B,K] | `num_charts`, `num_codes`, `eps=1e-6` | $H(Y \mid \text{chart}, \text{code})$ (requires labels) |
| `compute_symbol_calibration_loss` | `z_geo` [B,D], `K_chart` [B], `indices_stack` [B,K] | `num_charts`, `num_codes` | Radial consistency within symbols (requires labels) |
| `compute_chart_collapse_penalty` | `router_weights` [B,K] | `num_charts` | $\max(p_k) - 1/K$ (superseded by band loss) |
| `compute_code_collapse_penalty` | `v_local` [B,D], `codebook` [K,C,D], `router_weights` [B,K] | `temperature=1.0`, `eps=1e-6` | Differentiable code entropy penalty (superseded by band loss) |
| `compute_orthogonality_loss` | `model` (nn.Module) | `max_svd_dim=64`, `eps=1e-6` | SVD-based weight anisotropy penalty |
| `get_loss_schedule` | — | `epoch`, `warmup`, `ramp_end`, `final_weight=1.0` | Generic warmup schedule utility |
| `SupervisedTopologyLoss` | `router_weights` [B,K], `y_true` [B], `z_latent` [B,D] (optional) | `num_charts`, `num_classes`, `lambda_purity=0.1`, `lambda_balance=0.01`, `lambda_metric=0.01`, `margin=1.0`, `temperature=1.0` | Supervised purity + balance + contrastive (requires labels) |

---

(sec-loss-flow-summary)=
## Loss Flow Summary

```
Training loop (train_joint / train_phase_1 / train_unsupervised)
│
├─ encoder.forward()  →  z_geo, z_tex, z_n, enc_w, vq_loss, ...
├─ decoder.forward()  →  x_recon, dec_w
│
├─ compute_phase1_loss(x, x_recon, vq_loss, enc_w, dec_w, z_geo, ...)
│   ├─ base_loss  ← recon + vq + entropy + margin + hard_nll
│   │               + chart_usage + chart_ot + confidence_cal
│   │               + v_tangent + codebook_spread + codebook_center
│   │               + chart_center_{mean,radius,sep}
│   │               + code_usage + window + consistency
│   └─ zn_reg_loss ← uniformity + radial_cal
│
├─ + jump_weight * compute_jump_consistency_loss(...)   [scheduled]
├─ + w_perp * orthogonality_loss(z_n, z_tex)
│
└─ total = base_loss + zn_reg_loss + jump + ortho
```
