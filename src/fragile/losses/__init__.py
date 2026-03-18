"""Centralized loss library for fragile-rl.

Organized by agent component:
- encoder: encoder / router / codebook / atlas losses
- world_model: dynamics / geodesic / Hamiltonian losses
- macro: closure / macro model / symbolic transition losses
- critic: value-field / covector alignment losses
- actor: policy losses
- reward: reward decomposition losses

Use explicit submodule imports, e.g.::

    from fragile.losses.encoder import compute_phase1_loss
    from fragile.losses.world_model import compute_phase2_loss
    from fragile.losses.macro import EnclosureProbe
"""

import importlib as _importlib


def __getattr__(name: str):
    """Lazy re-export: ``from fragile.losses import X`` works for any public symbol."""
    submodules = [
        "fragile.losses.encoder",
        "fragile.losses.encoder_unused",
        "fragile.losses.world_model",
        "fragile.losses.old_macro",
        "fragile.losses.macro",
        "fragile.losses.markov_model",
        "fragile.losses.new_macro",
        "fragile.losses.critic",
        "fragile.losses.actor",
        "fragile.losses.reward",
    ]
    for mod_name in submodules:
        mod = _importlib.import_module(mod_name)
        if hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(f"module 'fragile.losses' has no attribute {name!r}")
