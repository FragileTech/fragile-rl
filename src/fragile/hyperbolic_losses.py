"""Backward-compat shim. All losses moved to fragile.losses.encoder."""

from fragile.losses._helpers import _as_tangent, _project_to_ball  # noqa: F401
from fragile.losses.encoder import *  # noqa: F401,F403
