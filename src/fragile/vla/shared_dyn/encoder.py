"""Encoder variants that share the main codebook for dynamics VQ.

Instead of maintaining a separate ``codebook_dyn`` parameter, these
subclasses route ``dynamics_vq()`` through the *same* codebook used for
reconstruction.  This lets a single codebook learn symbols that are
simultaneously good for both decoding and Markov-transition prediction.
"""

from __future__ import annotations

from fragile.layers.topoencoder import (
    AttentiveAtlasEncoder,
    TopoEncoder,
)


class SharedDynAtlasEncoder(AttentiveAtlasEncoder):
    """Atlas encoder where ``dynamics_vq`` uses the main codebook.

    The parent class creates a separate ``codebook_dyn`` when
    ``dyn_codes_per_chart > 0``.  We bypass that entirely by forcing
    ``dyn_codes_per_chart=0`` (no extra parameter) and overriding
    ``dynamics_vq`` to quantize against ``self.codebook`` with zero
    VQ commitment loss (the main forward pass already trains the
    codebook via reconstruction).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # -- dynamics_vq override ------------------------------------------

    def dynamics_vq(self, v_local, router_weights):
        """VQ *v_local* against the **main** codebook with zero VQ loss.

        Returns the same 4-tuple as the parent:
            z_q_blended, K_code, indices, vq_loss
        """
        return self._hyperbolic_vq(
            v_local,
            self.codebook,
            router_weights,
            0.0,  # commitment_beta  — zero; main pass handles training
            0.0,  # codebook_loss_weight — zero
            use_soft_equiv=False,
        )[:4]

    @property
    def effective_dyn_codes_per_chart(self):
        """Number of dynamics codes equals the main codebook size."""
        return self.codes_per_chart


class SharedDynTopoEncoder(TopoEncoder):
    """``TopoEncoder`` whose inner encoder uses the shared codebook.

    ``super().__init__()`` builds both ``self.encoder`` (a
    ``AttentiveAtlasEncoder``) and ``self.decoder``.  We then
    replace ``self.encoder`` with a ``SharedDynAtlasEncoder`` that has
    the identical architecture but overrides ``dynamics_vq``.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Collect the kwargs that AttentiveAtlasEncoder accepts.
        encoder_kwargs = {
            "input_dim": kwargs.get("input_dim", 2),
            "hidden_dim": kwargs.get("hidden_dim", 32),
            "latent_dim": kwargs.get("latent_dim", 2),
            "num_charts": kwargs.get("num_charts", 3),
            "codes_per_chart": kwargs.get("codes_per_chart", 21),
            "bundle_size": kwargs.get("bundle_size", None),
            "covariant_attn": kwargs.get("covariant_attn", True),
            "covariant_attn_tau_min": kwargs.get("covariant_attn_tau_min", 1e-2),
            "covariant_attn_denom_min": kwargs.get("covariant_attn_denom_min", 1e-3),
            "covariant_attn_transport_eps": kwargs.get("covariant_attn_transport_eps", 1e-3),
            "soft_equiv_metric": kwargs.get("soft_equiv_metric", False),
            "soft_equiv_bundle_size": kwargs.get("soft_equiv_bundle_size", None),
            "soft_equiv_hidden_dim": kwargs.get("soft_equiv_hidden_dim", 64),
            "soft_equiv_use_spectral_norm": kwargs.get("soft_equiv_use_spectral_norm", True),
            "soft_equiv_zero_self_mixing": kwargs.get("soft_equiv_zero_self_mixing", False),
            "soft_equiv_soft_assign": kwargs.get("soft_equiv_soft_assign", True),
            "soft_equiv_temperature": kwargs.get("soft_equiv_temperature", 1.0),
            "commitment_beta": kwargs.get("commitment_beta", 0.25),
            "codebook_loss_weight": kwargs.get("codebook_loss_weight", 1.0),
        }

        # Replace the parent's AttentiveAtlasEncoder with our
        # shared-dynamics variant.  The decoder is left unchanged.
        self.encoder = SharedDynAtlasEncoder(**encoder_kwargs)
