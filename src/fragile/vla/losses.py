"""Backward-compat shim. Losses moved to fragile.losses.*."""

# Encoder losses (originally defined here or re-exported from hyperbolic_losses)
# Re-export _project_to_ball for any code that imported it from here
from fragile.losses._helpers import _project_to_ball  # noqa: F401
from fragile.losses.encoder import (  # noqa: F401
    _deterministic_st_router_weights,
    combine_quality_targets,
    compute_chart_center_mean_loss,
    compute_chart_center_radius_loss,
    compute_chart_center_separation_loss,
    compute_chart_usage_band_loss,
    compute_code_usage_band_loss,
    compute_codebook_centering_loss,
    compute_codebook_spread_loss,
    compute_confidence_calibration_loss,
    compute_error_quality_targets,
    compute_hard_routing_nll,
    compute_hyperbolic_uniformity_loss,
    compute_phase1_loss,
    compute_radial_calibration_loss,
    compute_rank_quality_targets,
    compute_router_information_metrics,
    compute_router_margin_loss,
    compute_router_sharpness_metrics,
    compute_routing_confidence,
    compute_routing_entropy,
    compute_sinkhorn_balanced_chart_loss,
    compute_v_tangent_barrier_loss,
    compute_window_loss,
    mix_quality_targets,
    orthogonality_loss,
)

# Macro / closure losses
from fragile.losses.macro import (  # noqa: F401
    compute_dyn_transition_loss,
    compute_dynamics_markov_loss,
    compute_enclosure_loss,
    DynamicsTransitionModel,
    EnclosureProbe,
    GradientReversalFunction,
    GradientReversalLayer,
    grl_alpha_schedule,
    zeno_loss,
)

# World model / dynamics losses
from fragile.losses.world_model import (  # noqa: F401
    compute_dynamics_chart_loss,
    compute_dynamics_geodesic_loss,
    compute_energy_conservation_loss,
    compute_hodge_consistency_loss,
    compute_momentum_regularization,
    compute_phase2_loss,
    compute_screened_poisson_loss,
    hyperbolic_laplacian,
)
