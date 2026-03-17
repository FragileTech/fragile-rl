"""Backward-compat shim. Geodesic losses moved to fragile.losses.world_model."""

from fragile.losses.world_model import (  # noqa: F401
    compute_momentum_targets,
    compute_phase2_geodesic_diffusion_loss,
    compute_supervised_wm_loss,
    endpoint_loss,
    geodesic_interpolation,
    momentum_loss,
    position_loss,
)
