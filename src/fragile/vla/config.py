"""VLA experiment configuration dataclass."""

from dataclasses import dataclass, field

import torch


@dataclass
class VLAConfig:
    """Configuration for the TopoEncoder x SmolVLA experiment."""

    # --- Feature extraction ---
    smolvla_model_id: str = "lerobot/smolvla_base"
    dataset_name: str = "lerobot/svla_so100_pickplace"
    feature_dim: int = 720  # lm_expert hidden_size (SmolVLM2-500M VLM=960, expert=720)
    feature_cache_dir: str = "outputs/vla/features"
    pooling: str = "mean"  # "mean" or "modality_aware"
    max_episodes: int = 0  # 0 = all
    held_out_test_episodes: int = 5  # reserve the last N cached episodes for evaluation

    # --- Encoder architecture (maps to TopoEncoderPrimitives args) ---
    input_dim: int = 720  # must match feature_dim
    hidden_dim: int = 256
    latent_dim: int = 16
    num_charts: int = 8
    codes_per_chart: int = 32
    covariant_attn: bool = True
    covariant_attn_tensorization: str = "full"
    covariant_attn_rank: int = 8
    covariant_attn_tau_min: float = 1e-2
    covariant_attn_denom_min: float = 1e-3
    covariant_attn_use_transport: bool = True
    covariant_attn_transport_eps: float = 1e-3
    conv_backbone: bool = False

    # --- VQ codebook training ---
    commitment_beta: float = 0.25         # commitment loss weight (encoder tracks codes)
    codebook_loss_weight: float = 1.0     # codebook pulled toward encoder outputs
    w_codebook_dynamics: float = 1.0      # weight for coarse dynamics loss on codebook (Phase 3)

    # --- Soft equivariant metric ---
    soft_equiv_metric: bool = True
    soft_equiv_bundle_size: int = 0  # 0 = use latent_dim
    soft_equiv_hidden_dim: int = 64
    soft_equiv_use_spectral_norm: bool = True
    soft_equiv_zero_self_mixing: bool = False
    soft_equiv_soft_assign: bool = True
    soft_equiv_temperature: float = 1.0

    # --- World model ---
    action_dim: int = 6  # SO100: 6 joints (shoulder_pan/lift, elbow, wrist_flex/roll, gripper)
    wm_hidden_dim: int = 256
    wm_dt: float = 0.01
    wm_gamma_friction: float = 1.0
    wm_T_c: float = 0.1
    wm_prediction_horizon: int = 4
    wm_alpha_potential: float = 0.5     # generation vs control balance in Phi_eff
    wm_beta_curl: float = 0.1           # Value Curl coupling for Lorentz/Boris
    wm_gamma_risk: float = 0.01         # risk-stress penalty weight
    wm_use_boris: bool = True           # enable Boris rotation
    wm_use_jump: bool = True            # enable conditional chart jump (WFR Fisher-Rao)
    wm_refine_steps: int = 3            # N BAOAB sub-steps per horizon step (WFR W2)
    wm_jump_beta: float = 1.0           # inverse temperature for Boltzmann chart selection
    wm_min_length: float = 0.03        # minimum geodesic length scale; derives F_max, V_alg, cf_max
    wm_risk_metric_alpha: float = 0.1  # risk-metric coupling

    # --- Phase 1 loss weights (encoder) ---
    w_feature_recon: float = 1.0
    w_vq: float = 1.0
    w_entropy: float = 0.3           # local routing entropy H(K|X): lower = more confident routing
    w_diversity: float = 1.0         # hard/ST chart-usage entropy-band weight
    chart_usage_entropy_low: float | None = None   # defaults to log(0.9 * num_charts)
    chart_usage_entropy_high: float | None = None  # optional upper saturation guard
    w_chart_ot: float = 1.0          # entropic OT chart-balancing auxiliary
    chart_ot_epsilon: float = 0.05
    chart_ot_iters: int = 20
    w_uniformity: float = 0.05
    w_radial_calibration: float = 0.1
    w_confidence_calibration: float = 0.05
    w_hard_routing_nll: float = 0.5
    w_router_margin: float = 2.0
    router_margin_target: float = 0.05
    radial_quality_alpha: float = 2.0
    radial_vq_alpha: float = 1.0
    radial_quality_rank_mix: float = 0.75
    radial_recon_quality_weight: float = 0.7
    radial_quality_mix: float = 1.0
    radial_quality_base_weight: float = 0.0
    radial_calibration_rho_max: float = 4.0
    radial_calibration_band_width: float = 0.75
    w_v_tangent_barrier: float = 0.01   # keep the pre-squash latent out of the saturated shell
    v_tangent_barrier_radius: float = 0.9
    w_codebook_spread: float = 0.05
    w_codebook_spread_margin: float = 1.0
    w_codebook_center: float = 0.02
    w_chart_center_mean: float = 0.02
    w_chart_center_radius: float = 0.05
    chart_center_radius_max: float = 2.0
    w_chart_center_sep: float = 0.02
    chart_center_sep_margin: float = 1.0
    w_chart_collapse: float = 0.0    # deprecated; no longer part of the active stack
    w_code_collapse: float = 0.5     # hard/ST per-chart code-usage entropy-band weight
    code_usage_entropy_low: float | None = None    # defaults to log(0.75 * codes_per_chart)
    code_usage_entropy_high: float | None = None   # optional upper saturation guard
    w_code_collapse_temperature: float = 1.0       # code-usage soft/ST temperature
    w_window: float = 0.0
    w_window_eps_ground: float = 0.1
    w_consistency: float = 0.0
    w_jump: float = 0.0
    w_jump_warmup: int = 20
    w_jump_ramp_end: int = 40

    # --- Orthogonality loss ---
    w_perp: float = 0.01                      # orthogonality loss weight (z_n vs z_tex decorrelation)

    # --- Dynamics codebook ---
    dyn_codes_per_chart: int = 0            # 0 = disabled; e.g. 8 for coarse dynamics
    dyn_codebook_loss_weight: float = 1.0   # codebook pulled toward data
    dyn_commitment_beta: float = 0.25       # encoder commitment
    w_dyn_transition: float = 0.5           # DynamicsTransitionModel CE weight
    dyn_transition_hidden_dim: int = 128    # transition model MLP hidden dim
    lr_dyn_codebook: float = 1e-3           # Phase 2 LR for dynamics codebook + transition model

    # --- Causal enclosure loss ---
    w_enclosure: float = 0.0                   # enclosure loss weight (0=disabled; redundant with dual codebook)
    enclosure_probe_hidden_dim: int = 128      # hidden dim of enclosure probe MLP
    enclosure_probe_lr: float = 3e-3           # probe learning rate (3x encoder LR)
    enclosure_grl_max_alpha: float = 1.0       # max gradient reversal strength
    enclosure_grl_warmup_steps: int = 5000     # GRL warmup steps

    # --- Zeno loss (routing smoothness) ---
    w_zeno: float = 0.0                        # Zeno loss weight (0=disabled; redundant with dual codebook)
    zeno_mode: str = "jsd"                     # "jsd" (recommended) or "kl"

    # --- Phase 2 loss weights (dynamics) ---
    w_geodesic: float = 1.0
    w_chart_transition: float = 0.5
    w_momentum_reg: float = 0.01
    w_energy_conservation: float = 0.01
    w_screened_poisson: float = 0.01  # screened Poisson PDE residual weight
    wm_screening_kappa: float = 1.0  # screening mass κ
    w_hodge: float = 0.01  # Hodge consistency loss weight

    # --- Supervised geodesic diffusion ---
    wm_diffusion_substeps: int = 8          # N waypoints between z_t and z_{t+1}
    w_position: float = 1.0                 # position loss weight (waypoint matching)
    w_endpoint: float = 2.0                 # endpoint loss weight (z_N == z_{t+1})
    w_momentum_target: float = 0.1          # momentum supervision weight
    w_hodge_perp: float = 0.01              # harmonic force penalty
    use_geodesic_diffusion: bool = True      # toggle new vs old Phase 2

    # --- Phase 3 joint scaling ---
    phase3_encoder_scale: float = 0.1
    phase3_dynamics_scale: float = 1.0
    phase3_zn_reg_scale: float = 0.1

    # --- Training ---
    phase1_epochs: int = 100
    phase2_epochs: int = 50
    phase3_epochs: int = 30
    lr_encoder: float = 1e-3
    lr_wm: float = 1e-3
    lr_joint_encoder: float = 1e-4
    lr_joint_wm: float = 1e-3
    lr_chart_centers_scale: float = 0.1
    lr_codebook_scale: float = 0.5
    batch_size: int = 256
    sequence_length: int = 8
    grad_clip: float = 1.0

    # --- Logging / output ---
    output_dir: str = "outputs/vla"
    save_every: int = 25
    log_every: int = 10
    mlflow: bool = False
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "vla"
    mlflow_run_name: str = ""

    # --- Device ---
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu",
    )
