import math

import torch
from torch import nn
import torch.nn.functional as F

from fragile.layers import IsotropicBlock, NormGatedGELU, SoftEquivariantLayer, SpectralLinear
from fragile.layers.gauge import (
    exp_map_zero,
    log_map_zero,
    mobius_add,
    poincare_weighted_mean,
    poincare_weighted_mean_per_chart,
    project_to_ball,
    smooth_tangent_to_ball,
)
from fragile.layers.initialization import (
    init_soft_equiv_layers,
    resolve_bundle_params,
    spread_codebook,
    spread_directions,
)
from fragile.layers.router import CovariantChartRouter


class GlobalAffineMap(nn.Module):
    """Global per-dimension affine map shared by encoder input and decoder output.

    The map is deterministic and invertible as long as every scale is positive.
    It can be initialized from dataset statistics and optionally left frozen, so
    dreamed latents can still be decoded back to the original raw coordinate
    system without needing an accompanying input sample.
    """

    def __init__(
        self,
        dim: int,
        *,
        enabled: bool = False,
        learnable: bool = False,
        min_scale: float = 1e-3,
    ) -> None:
        """Initialize the global affine map.

        Args:
            dim: Dimensionality of the input/output space.
            enabled: Whether the affine normalization is active on construction.
            learnable: Whether the offset and log_scale parameters require
                gradients.
            min_scale: Minimum allowed per-dimension scale to prevent
                division-by-zero during normalization.
        """
        super().__init__()
        self.dim = int(dim)
        self.min_scale = float(min_scale)
        self.register_buffer(
            "_enabled",
            torch.tensor(bool(enabled), dtype=torch.bool),
            persistent=True,
        )
        self.offset = nn.Parameter(torch.zeros(dim), requires_grad=learnable)
        self.log_scale = nn.Parameter(torch.zeros(dim), requires_grad=learnable)

    @property
    def enabled(self) -> bool:
        """Whether the affine normalization is active.

        Returns:
            bool: True if the affine map is applied during normalize/denormalize.
        """
        return bool(self._enabled.item())

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the affine map.

        Args:
            enabled: If True, the affine normalization is applied in
                ``normalize`` and ``denormalize``. If False, those methods
                become identity functions.
        """
        self._enabled.fill_(bool(enabled))

    def set_learnable(self, learnable: bool) -> None:
        """Toggle gradient updates for the affine parameters.

        Args:
            learnable: If True, offset and log_scale will accumulate
                gradients during back-propagation.
        """
        self.offset.requires_grad_(learnable)
        self.log_scale.requires_grad_(learnable)

    def scale(self) -> torch.Tensor:
        """Return the positive per-dimension scale.

        Returns:
            torch.Tensor: Per-dimension scale factors of shape ``[dim]``,
                clamped to be at least ``min_scale``.
        """
        return self.log_scale.exp().clamp_min(self.min_scale)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw inputs into the normalized model space.

        Args:
            x: Input tensor of shape ``[..., dim]`` in the original data
                coordinate system.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as ``x``.
                When the map is disabled, returns ``x`` unchanged.
        """
        if not self.enabled:
            return x
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        scale = self.scale().to(device=x.device, dtype=x.dtype)
        return (x - offset) / scale

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Map normalized model outputs back to the raw data space.

        Args:
            x: Normalized tensor of shape ``[..., dim]``.

        Returns:
            torch.Tensor: De-normalized tensor of the same shape as ``x``.
                When the map is disabled, returns ``x`` unchanged.
        """
        if not self.enabled:
            return x
        offset = self.offset.to(device=x.device, dtype=x.dtype)
        scale = self.scale().to(device=x.device, dtype=x.dtype)
        return x * scale + offset

    @torch.no_grad()
    def set_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        """Initialize the affine map from dataset mean/std statistics.

        Sets the offset to ``mean`` and the scale to ``std`` (clamped by
        ``min_scale``), then enables the map.  This method runs under
        ``torch.no_grad()``.

        Args:
            mean: Per-dimension mean of shape ``[dim]``.
            std: Per-dimension standard deviation of shape ``[dim]``.

        Raises:
            ValueError: If ``mean`` or ``std`` do not match the expected
                shape ``[dim]``.
        """
        mean_t = torch.as_tensor(mean, device=self.offset.device, dtype=self.offset.dtype)
        std_t = torch.as_tensor(std, device=self.offset.device, dtype=self.offset.dtype)
        if mean_t.shape != self.offset.shape:
            msg = "mean must have shape [input_dim]."
            raise ValueError(msg)
        if std_t.shape != self.offset.shape:
            msg = "std must have shape [input_dim]."
            raise ValueError(msg)
        self.offset.copy_(mean_t)
        self.log_scale.copy_(std_t.clamp_min(self.min_scale).log())
        self.set_enabled(True)

    def extra_repr(self) -> str:
        """Return a human-readable summary of the module configuration.

        Returns:
            str: String containing dim, enabled state, learnable flag, and
                min_scale.
        """
        return (
            f"dim={self.dim}, enabled={self.enabled}, "
            f"learnable={self.offset.requires_grad}, min_scale={self.min_scale}"
        )


class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder using gauge-covariant primitives."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn: bool = True,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
    ) -> None:
        """Initialize the attentive atlas encoder.

        Args:
            input_dim: Dimensionality of the raw observation space.
            hidden_dim: Width of the internal feature-extraction MLP.
            latent_dim: Dimensionality of the Poincare-ball latent space.
            num_charts: Number of atlas charts (routing targets).
            codes_per_chart: Number of VQ codebook entries per chart.
            bundle_size: Fiber bundle size for gated activations. If ``None``,
                it is resolved automatically from ``hidden_dim`` and
                ``latent_dim``.
            covariant_attn: Whether to use the covariant attention router.
            covariant_attn_tau_min: Minimum temperature for the covariant
                router softmax.
            covariant_attn_denom_min: Minimum denominator clamp inside the
                covariant router.
            covariant_attn_transport_eps: Epsilon used for parallel-transport
                numerical stability in the router.
            soft_equiv_metric: If True, build per-chart soft-equivariant
                layers for a learned distance metric in VQ.
            soft_equiv_bundle_size: Bundle size for the soft-equivariant
                layers. Defaults to ``latent_dim`` when ``None``.
            soft_equiv_hidden_dim: Hidden width of each soft-equivariant
                layer.
            soft_equiv_use_spectral_norm: Apply spectral normalization inside
                the soft-equivariant layers.
            soft_equiv_zero_self_mixing: Zero-initialize the self-mixing
                weights of soft-equivariant layers.
            soft_equiv_soft_assign: Use a soft (temperature-weighted) VQ
                assignment when the soft-equivariant metric is active.
            soft_equiv_temperature: Temperature for the soft VQ assignment.
                Must be positive when ``soft_equiv_soft_assign`` is True.
            commitment_beta: Commitment loss weight for the VQ objective.
            codebook_loss_weight: Codebook loss weight for the VQ objective.
        """
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self.covariant_attn = covariant_attn
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.feature_extractor = nn.Sequential(
            SpectralLinear(input_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
            SpectralLinear(hidden_dim, hidden_dim, bias=True),
            NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles),
        )

        self.cov_router = CovariantChartRouter(
            latent_dim=latent_dim,
            key_dim=hidden_dim,
            num_charts=num_charts,
            feature_dim=hidden_dim,
            tau_min=covariant_attn_tau_min,
            tau_denom_min=covariant_attn_denom_min,
            transport_eps=covariant_attn_transport_eps,
        )
        self.key_proj = None
        self.chart_queries = None
        self.scale = None

        self.val_proj = SpectralLinear(hidden_dim, latent_dim, bias=True)
        self.val_proj_scale = nn.Parameter(torch.tensor(2.0))  # learnable pre-squash scale

        # Quasi-uniform chart centers: Fibonacci sphere (3-D) or repulsion
        # init so every chart starts with a distinct, well-separated catchment
        # region — prevents softmax winner-take-all collapse at epoch 0.
        self.chart_centers = nn.Parameter(spread_directions(num_charts, latent_dim) * 0.5)

        # Spread codebook codes around the local origin of each chart so that
        # VQ does not instantly collapse to a single nearest-neighbor.
        self.codebook = nn.Parameter(
            spread_codebook(num_charts, codes_per_chart, latent_dim, radius=0.3)
        )

        self.soft_equiv_layers: nn.ModuleList | None = None
        if soft_equiv_metric:
            bundle_size = soft_equiv_bundle_size or latent_dim
            if bundle_size <= 0:
                msg = "soft_equiv_bundle_size must be positive."
                raise ValueError(msg)
            if latent_dim % bundle_size != 0:
                msg = "latent_dim must be divisible by soft_equiv_bundle_size."
                raise ValueError(msg)
            n_bundles = latent_dim // bundle_size
            self.soft_equiv_layers = nn.ModuleList([
                SoftEquivariantLayer(
                    n_bundles=n_bundles,
                    bundle_dim=bundle_size,
                    hidden_dim=soft_equiv_hidden_dim,
                    use_spectral_norm=soft_equiv_use_spectral_norm,
                    zero_self_mixing=soft_equiv_zero_self_mixing,
                )
                for _ in range(num_charts)
            ])
            init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)

        self.structure_filter = nn.Sequential(
            IsotropicBlock(latent_dim, latent_dim, bundle_size=latent_dim),
            SpectralLinear(latent_dim, latent_dim, bias=True),
        )
        self._last_v_raw: torch.Tensor | None = None
        self._last_v_projected: torch.Tensor | None = None
        self._last_v_local_raw: torch.Tensor | None = None
        self._last_z_geo_raw: torch.Tensor | None = None
        self._last_c_bar: torch.Tensor | None = None
        self._last_v_local: torch.Tensor | None = None
        self._last_indices_stack: torch.Tensor | None = None
        self._last_v_raw: torch.Tensor | None = None
        self._last_soft_router_weights: torch.Tensor | None = None
        self._last_soft_router_weights_live: torch.Tensor | None = None
        self._last_router_scores: torch.Tensor | None = None
        self._last_router_scores_live: torch.Tensor | None = None

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract hidden features from raw (normalized) input.

        Args:
            x: Input tensor of shape ``[B, input_dim]``.

        Returns:
            torch.Tensor: Feature tensor of shape ``[B, hidden_dim]``.
        """
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
        """Compute a learned distance metric via soft-equivariant layers.

        When ``soft_equiv_layers`` is ``None``, falls back to the squared
        Euclidean norm.  Otherwise each chart's soft-equivariant layer
        transforms the tangent-space difference and the squared norm of the
        output is returned.  A log-ratio regularization loss is cached in
        ``_last_soft_equiv_log_ratio``.

        Args:
            diff: Tangent-space difference tensor of shape
                ``[B, num_charts, codes_per_chart, latent_dim]``.

        Returns:
            torch.Tensor: Squared distance tensor of shape
                ``[B, num_charts, codes_per_chart]``.
        """
        if self.soft_equiv_layers is None:
            self._last_soft_equiv_log_ratio = None
            return (diff**2).sum(dim=-1)
        batch_size, _num_charts, num_codes, latent_dim = diff.shape
        ratio_max = 50.0
        eps = 1e-6
        transformed = []
        log_ratio_losses = []
        for chart_idx, layer in enumerate(self.soft_equiv_layers):
            diff_chart = diff[:, chart_idx].reshape(-1, latent_dim)
            diff_chart = torch.nan_to_num(diff_chart, nan=0.0, posinf=0.0, neginf=0.0)
            diff_out = layer(diff_chart)
            diff_out = torch.nan_to_num(diff_out, nan=0.0, posinf=0.0, neginf=0.0)
            in_norm = diff_chart.norm(dim=-1, keepdim=True).clamp(min=eps)
            out_norm = diff_out.norm(dim=-1, keepdim=True)
            ratio = out_norm / in_norm
            ratio_clamped = ratio.clamp(max=ratio_max)
            scale = torch.where(ratio > 0, ratio_clamped / ratio, torch.ones_like(ratio))
            diff_out = diff_out * scale
            log_ratio = torch.log(ratio.clamp(min=eps, max=ratio_max))
            log_ratio_losses.append((log_ratio**2).mean())
            transformed.append(diff_out.view(batch_size, num_codes, latent_dim))
        diff_out = torch.stack(transformed, dim=1)
        if log_ratio_losses:
            self._last_soft_equiv_log_ratio = torch.stack(log_ratio_losses).mean()
        else:
            self._last_soft_equiv_log_ratio = torch.tensor(0.0, device=diff.device)
        return (diff_out**2).sum(dim=-1)

    def soft_equiv_l1_loss(self) -> torch.Tensor:
        """Compute the mean L1 sparsity loss across all soft-equivariant layers.

        Returns:
            torch.Tensor: Scalar L1 loss averaged over charts. Returns
                ``0.0`` when no soft-equivariant layers exist.
        """
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        """Return the cached log-ratio regularization loss.

        The value is computed and cached during ``_apply_soft_equiv_metric``.
        It penalizes the soft-equivariant layers from changing the norm of
        their input too aggressively.

        Returns:
            torch.Tensor: Scalar log-ratio loss. Returns ``0.0`` when no
                soft-equivariant layers exist or when no forward pass has
                been executed yet.
        """
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def _hyperbolic_vq(
        self,
        v_local: torch.Tensor,
        codebook_param: torch.Tensor,
        router_weights: torch.Tensor,
        commitment_beta: float,
        codebook_loss_weight: float,
        use_soft_equiv: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform hyperbolic vector quantization against a codebook.

        Computes the nearest codebook entry per chart for each sample in the
        batch using Mobius arithmetic in the Poincare ball, then blends
        results with router weights.

        Args:
            v_local: Chart-local latent points of shape ``[B, D]`` in the
                Poincare ball.
            codebook_param: Raw codebook parameters of shape
                ``[num_charts, codes_per_chart, D]`` (projected to the ball
                internally).
            router_weights: Soft chart-routing weights of shape
                ``[B, num_charts]``.
            commitment_beta: Scalar weight for the commitment loss term.
            codebook_loss_weight: Scalar weight for the codebook loss term.
            use_soft_equiv: If True, use the learned soft-equivariant metric
                for distance computation; otherwise use squared Euclidean
                distance in the tangent space.

        Returns:
            tuple: A 5-tuple containing:
                - **z_q_blended** (*torch.Tensor*): Router-weighted code blend
                  of shape ``[B, D]``.
                - **k_code** (*torch.Tensor*): Winning code index for the
                  winning chart, of shape ``[B]``.
                - **indices** (*torch.Tensor*): Nearest code index per chart,
                  of shape ``[B, num_charts]``.
                - **vq_loss** (*torch.Tensor*): Scalar VQ loss combining
                  commitment and codebook terms.
                - **z_q_all** (*torch.Tensor*): Nearest code per chart (full
                  tensor) of shape ``[B, num_charts, D]``.
        """
        codebook = project_to_ball(codebook_param)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)

        if use_soft_equiv:
            dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        else:
            dist = (diff_tan**2).sum(dim=-1)  # [B, N_c, K]

        indices = torch.argmin(dist, dim=-1)  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v_local.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]

        if use_soft_equiv and self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = poincare_weighted_mean_per_chart(codebook, weights)
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ objective weighted by routing.
        weights = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * weights).mean(dim=(0, 2)).sum()
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss_val = (delta_codebook**2 * weights).mean(dim=(0, 2)).sum()
        vq_loss = codebook_loss_weight * codebook_loss_val + commitment_beta * commitment

        k_chart = torch.argmax(router_weights, dim=1)  # [B]
        z_q_blended = poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        k_code = indices.gather(1, k_chart.unsqueeze(1)).squeeze(1)  # [B]

        return z_q_blended, k_code, indices, vq_loss, z_q_all

    def dynamics_vq(
        self,
        v_local: torch.Tensor,
        router_weights: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Quantize chart-local latents against the dynamics codebook.

        This is a convenience wrapper around ``_hyperbolic_vq`` that uses
        the dynamics-specific codebook and loss weights. The
        soft-equivariant metric is **not** used for dynamics quantization.

        Args:
            v_local: Chart-local latent points of shape ``[B, D]`` in the
                Poincare ball.
            router_weights: Soft chart-routing weights of shape
                ``[B, num_charts]``.

        Returns:
            tuple: A 4-tuple containing:
                - **z_q_dyn_blended** (*torch.Tensor*): Router-weighted
                  dynamics code blend of shape ``[B, D]``.
                - **K_code_dyn** (*torch.Tensor*): Winning dynamics code
                  index of shape ``[B]``.
                - **indices_dyn** (*torch.Tensor*): Nearest dynamics code per
                  chart of shape ``[B, num_charts]``.
                - **vq_loss_dyn** (*torch.Tensor*): Scalar VQ loss for the
                  dynamics codebook.

        Raises:
            AssertionError: If the dynamics codebook has not been initialized
                (i.e. ``dyn_codes_per_chart=0``).
        """
        assert self.codebook_dyn is not None, (
            "dynamics codebook not initialized (dyn_codes_per_chart=0)"
        )
        return self._hyperbolic_vq(
            v_local,
            self.codebook_dyn,
            router_weights,
            self._dyn_commitment_beta,
            self._dyn_codebook_loss_weight,
            use_soft_equiv=False,
        )[:4]

    def forward(
        self,
        x: torch.Tensor,
        routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Forward pass through the attentive atlas encoder.

        Extracts features, routes to charts, computes chart-local
        coordinates, performs vector quantization, and separates the latent
        into geometric (structure + nuisance) and texture components.

        Args:
            x: Normalized input tensor of shape ``[B, input_dim]``.
            routing_tau: Temperature scaling for the chart router softmax.

        Returns:
            tuple: A 12-tuple containing:
                - **K_chart** (*torch.Tensor*): Winning chart index per sample,
                  shape ``[B]``.
                - **K_code** (*torch.Tensor*): Winning codebook code index for
                  the winning chart, shape ``[B]``.
                - **z_n_tan** (*torch.Tensor*): Router-blended nuisance
                  component in tangent space, shape ``[B, latent_dim]``.
                - **z_tex** (*torch.Tensor*): Texture residual in tangent
                  space, shape ``[B, latent_dim]``.
                - **router_weights** (*torch.Tensor*): Soft chart-routing
                  weights, shape ``[B, num_charts]``.
                - **z_geo** (*torch.Tensor*): Full geometric latent in the
                  Poincare ball, shape ``[B, latent_dim]``.
                - **vq_loss** (*torch.Tensor*): Scalar vector-quantization
                  loss.
                - **indices_stack** (*torch.Tensor*): Nearest codebook index
                  per chart, shape ``[B, num_charts]``.
                - **z_n_all_charts** (*torch.Tensor*): Per-chart nuisance
                  embeddings in the Poincare ball, shape
                  ``[B, num_charts, latent_dim]``.
                - **c_bar** (*torch.Tensor*): Router-blended chart center in
                  the Poincare ball, shape ``[B, latent_dim]``.
                - **v_local** (*torch.Tensor*): Chart-local latent point in
                  the Poincare ball, shape ``[B, latent_dim]``.
                - **z_q_blended** (*torch.Tensor*): Router-blended VQ code in
                  the Poincare ball, shape ``[B, latent_dim]``.
        """
        # Extract features and map into chart coordinates (Poincare ball).
        features = self._encode_features(x)  # [B, H]
        v_raw = self.val_proj(features) * self.val_proj_scale
        v = smooth_tangent_to_ball(v_raw)  # [B, D]
        chart_centers = project_to_ball(self.chart_centers)  # [N_c, D]
        router_weights, K_chart = self.cov_router(
            v,
            features=features,
            chart_tokens=chart_centers,
            routing_tau=routing_tau,
        )
        self._last_soft_router_weights = self.cov_router._last_soft_router_weights
        self._last_soft_router_weights_live = self.cov_router._last_soft_router_weights_live
        self._last_router_scores = self.cov_router._last_router_scores
        self._last_router_scores_live = self.cov_router._last_router_scores_live

        c_bar = poincare_weighted_mean(chart_centers, router_weights)  # [B, D]
        v_local_raw = mobius_add(-c_bar, v)
        v_local = project_to_ball(v_local_raw)  # [B, D]

        # Per-chart codebook lookup via shared VQ helper.
        z_q_blended, K_code, indices_stack, vq_loss, z_q_all = self._hyperbolic_vq(
            v_local,
            self.codebook,
            router_weights,
            self._commitment_beta,
            self._codebook_loss_weight,
            use_soft_equiv=True,
        )

        # Structure filter extracts nuisance; remainder is texture.
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_local = mobius_add(z_q_st, exp_map_zero(z_n_tan))
        z_geo_raw = mobius_add(c_bar, z_local)
        z_geo = project_to_ball(z_geo_raw)  # [B, D]

        # Cache the chart-local latent so auxiliary losses can read the exact
        # codebook input from this forward pass instead of reconstructing it.
        self._last_v_raw = v_raw
        self._last_v_projected = v
        self._last_v_local_raw = v_local_raw
        self._last_z_geo_raw = z_geo_raw
        self._last_c_bar = c_bar
        self._last_v_local = v_local
        self._last_indices_stack = indices_stack

        return (
            K_chart,
            K_code,
            z_n_tan,
            z_tex,
            router_weights,
            z_geo,
            vq_loss,
            indices_stack,
            z_n_all_charts,
            c_bar,
            v_local,
            z_q_blended,
        )


class _ChartFiLM1d(nn.Module):
    """Per-chart FiLM conditioning for 1-D feature vectors [B, H].

    Each chart contributes a multiplicative (gamma) and additive (beta)
    modulation that is blended via the router weights.
    """

    def __init__(self, num_charts: int, dim: int) -> None:
        """Initialize per-chart FiLM conditioning parameters.

        Args:
            num_charts: Number of atlas charts.
            dim: Feature dimensionality to modulate.
        """
        super().__init__()
        self.gammas = nn.Parameter(torch.zeros(num_charts, dim))
        self.betas = nn.Parameter(torch.zeros(num_charts, dim))

    def forward(self, h: torch.Tensor, router_weights: torch.Tensor) -> torch.Tensor:
        """Apply FiLM conditioning blended by router weights.

        Args:
            h: Feature tensor of shape ``[B, dim]``.
            router_weights: Soft chart-routing weights of shape
                ``[B, num_charts]``.

        Returns:
            torch.Tensor: Modulated feature tensor of shape ``[B, dim]``.
        """
        gamma = router_weights @ self.gammas  # [B, H]
        beta = router_weights @ self.betas
        return h * (1.0 + gamma) + beta


class TopologicalDecoder(nn.Module):
    """Topological decoder using gauge-covariant primitives.

    Reconstructs observations from a geometric latent living in the Poincare
    ball by routing through per-chart linear projections, applying optional
    FiLM conditioning, and rendering through a two-layer MLP with a skip
    connection.
    """

    def __init__(
        self,
        latent_dim: int = 2,
        hidden_dim: int = 32,
        num_charts: int = 3,
        output_dim: int = 2,
        bundle_size: int | None = None,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
        film_conditioning: bool = False,
    ) -> None:
        """Initialize the topological decoder.

        Args:
            latent_dim: Dimensionality of the Poincare-ball latent space.
            hidden_dim: Width of the internal rendering MLP.
            num_charts: Number of atlas charts.
            output_dim: Dimensionality of the reconstructed observation.
            bundle_size: Fiber bundle size for gated activations. If ``None``,
                it is resolved automatically from ``hidden_dim`` and
                ``latent_dim``.
            covariant_attn_tau_min: Minimum temperature for the decoder's
                covariant router softmax.
            covariant_attn_denom_min: Minimum denominator clamp inside the
                covariant router.
            covariant_attn_transport_eps: Epsilon for parallel-transport
                numerical stability in the router.
            film_conditioning: If True, insert per-chart FiLM layers after
                the first and second hidden layers.
        """
        super().__init__()
        self.num_charts = num_charts
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.router_tau_min = covariant_attn_tau_min
        self.router_tau_denom_min = covariant_attn_denom_min
        self.router_transport_eps = covariant_attn_transport_eps

        bundle_size, n_bundles = resolve_bundle_params(hidden_dim, latent_dim, bundle_size)

        self.chart_projectors = nn.ModuleList([
            SpectralLinear(latent_dim, hidden_dim, bias=False) for _ in range(num_charts)
        ])
        self.chart_gate = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )
        self.cov_router = CovariantChartRouter(
            latent_dim=latent_dim,
            key_dim=hidden_dim,
            num_charts=num_charts,
            feature_dim=None,
            tau_min=covariant_attn_tau_min,
            tau_denom_min=covariant_attn_denom_min,
            transport_eps=covariant_attn_transport_eps,
        )

        self.render_fc1 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act1 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_fc2 = SpectralLinear(hidden_dim, hidden_dim, bias=True)
        self.render_act2 = NormGatedGELU(bundle_size=bundle_size, n_bundles=n_bundles)
        self.render_out = SpectralLinear(hidden_dim, output_dim, bias=True)
        self.render_skip = SpectralLinear(hidden_dim, output_dim, bias=True)
        if film_conditioning:
            self.film1 = _ChartFiLM1d(num_charts, hidden_dim)
            self.film2 = _ChartFiLM1d(num_charts, hidden_dim)
        else:
            self.film1 = None
            self.film2 = None

    def forward(
        self,
        z_geo: torch.Tensor,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        routing_tau: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Decode from latent geometry to observation space.

        Determines chart routing (from provided weights, a hard chart index,
        or the internal covariant router), then renders through per-chart
        projections and a shared MLP with optional FiLM conditioning.

        Args:
            z_geo: Geometric latent tensor of shape ``[B, latent_dim]`` in
                the Poincare ball.
            chart_index: Optional hard chart assignment of shape ``[B]``.
                Converted to one-hot router weights when provided.
            router_weights: Optional pre-computed soft chart-routing weights
                of shape ``[B, num_charts]``. Takes priority over
                ``chart_index``.
            routing_tau: Temperature scaling for the covariant router softmax.
                Only used when neither ``router_weights`` nor ``chart_index``
                is supplied.

        Returns:
            tuple: A 3-tuple containing:
                - **x_hat** (*torch.Tensor*): Reconstructed observation of
                  shape ``[B, output_dim]``.
                - **router_weights** (*torch.Tensor*): Chart-routing weights
                  used for decoding, shape ``[B, num_charts]``.
                - **aux_losses** (*dict[str, torch.Tensor]*): Dictionary of
                  auxiliary losses (currently empty).

        Raises:
            ValueError: If ``router_weights`` has an unexpected shape.
        """
        aux_losses: dict[str, torch.Tensor] = {}

        # Clamp geometry to chart range (Poincare ball).
        z_geo = project_to_ball(z_geo)
        chart_centers = project_to_ball(self.chart_centers)
        if router_weights is not None:
            if router_weights.ndim != 2 or router_weights.shape[1] != self.num_charts:
                msg = "router_weights must have shape [B, N_c]."
                raise ValueError(msg)
        elif chart_index is not None:
            router_weights = F.one_hot(
                chart_index, num_classes=self.num_charts
            ).float()  # [B, N_c]
        else:
            # Covariant router predicts chart membership from geometry.
            router_weights, _ = self.cov_router(
                z_geo,
                chart_tokens=chart_centers,
                routing_tau=routing_tau,
            )

        # Chart-specific projections + gauge-covariant gating.
        h_stack = torch.stack(
            [proj(z_geo) for proj in self.chart_projectors], dim=1
        )  # [B, N_c, H]
        h_stack = self.chart_gate(h_stack.view(-1, self.hidden_dim)).view(
            z_geo.shape[0], self.num_charts, self.hidden_dim
        )
        h_global = (h_stack * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, H]

        # FC mode: layer-by-layer with optional FiLM conditioning
        h = self.render_fc1(h_global)
        if self.film1 is not None:
            h = self.film1(h, router_weights)
        h = self.render_act1(h)
        h = self.render_fc2(h)
        if self.film2 is not None:
            h = self.film2(h, router_weights)
        h = self.render_act2(h)
        x_hat = self.render_out(h) + self.render_skip(h_global)

        return x_hat, router_weights, aux_losses


class TopoEncoder(nn.Module):
    """Attentive Atlas encoder + topological decoder.

    Combines an ``AttentiveAtlasEncoder`` and a ``TopologicalDecoder`` into a
    single autoencoder module, with an optional ``GlobalAffineMap`` for
    input/output normalization.
    """

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
        bundle_size: int | None = None,
        covariant_attn_tau_min: float = 1e-2,
        covariant_attn_denom_min: float = 1e-3,
        covariant_attn_transport_eps: float = 1e-3,
        soft_equiv_metric: bool = False,
        soft_equiv_bundle_size: int | None = None,
        soft_equiv_hidden_dim: int = 64,
        soft_equiv_use_spectral_norm: bool = True,
        soft_equiv_zero_self_mixing: bool = False,
        soft_equiv_soft_assign: bool = True,
        soft_equiv_temperature: float = 1.0,
        film_conditioning: bool = False,
        commitment_beta: float = 0.25,
        codebook_loss_weight: float = 1.0,
        input_affine_enabled: bool = False,
        input_affine_learnable: bool = False,
        input_affine_min_scale: float = 1e-3,
    ) -> None:
        """Initialize the TopoEncoder autoencoder.

        Args:
            input_dim: Dimensionality of the raw observation space.
            hidden_dim: Width of the internal feature-extraction and
                rendering MLPs.
            latent_dim: Dimensionality of the Poincare-ball latent space.
            num_charts: Number of atlas charts (routing targets).
            codes_per_chart: Number of VQ codebook entries per chart.
            bundle_size: Fiber bundle size for gated activations. If ``None``,
                it is resolved automatically.
            covariant_attn_tau_min: Minimum temperature for the covariant
                router softmax (shared by encoder and decoder).
            covariant_attn_denom_min: Minimum denominator clamp inside the
                covariant router.
            covariant_attn_transport_eps: Epsilon for parallel-transport
                numerical stability.
            soft_equiv_metric: If True, use per-chart soft-equivariant layers
                for VQ distance in the encoder.
            soft_equiv_bundle_size: Bundle size for the soft-equivariant
                layers. Defaults to ``latent_dim`` when ``None``.
            soft_equiv_hidden_dim: Hidden width of each soft-equivariant
                layer.
            soft_equiv_use_spectral_norm: Apply spectral normalization in the
                soft-equivariant layers.
            soft_equiv_zero_self_mixing: Zero-initialize self-mixing weights
                of soft-equivariant layers.
            soft_equiv_soft_assign: Use a soft VQ assignment when the
                soft-equivariant metric is active.
            soft_equiv_temperature: Temperature for the soft VQ assignment.
            film_conditioning: If True, insert per-chart FiLM layers in the
                decoder.
            commitment_beta: Commitment loss weight for the VQ objective.
            codebook_loss_weight: Codebook loss weight for the VQ objective.
            input_affine_enabled: Whether the input/output affine
                normalization is active on construction.
            input_affine_learnable: Whether the affine parameters are
                learnable.
            input_affine_min_scale: Minimum scale for the affine map.
        """
        super().__init__()
        self.num_charts = num_charts
        self.io_affine = GlobalAffineMap(
            input_dim,
            enabled=input_affine_enabled,
            learnable=input_affine_learnable,
            min_scale=input_affine_min_scale,
        )

        self.encoder = AttentiveAtlasEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_charts=num_charts,
            codes_per_chart=codes_per_chart,
            bundle_size=bundle_size,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            soft_equiv_metric=soft_equiv_metric,
            soft_equiv_bundle_size=soft_equiv_bundle_size,
            soft_equiv_hidden_dim=soft_equiv_hidden_dim,
            soft_equiv_use_spectral_norm=soft_equiv_use_spectral_norm,
            soft_equiv_zero_self_mixing=soft_equiv_zero_self_mixing,
            soft_equiv_soft_assign=soft_equiv_soft_assign,
            soft_equiv_temperature=soft_equiv_temperature,
            commitment_beta=commitment_beta,
            codebook_loss_weight=codebook_loss_weight,
        )
        self.decoder = TopologicalDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_charts=num_charts,
            output_dim=input_dim,
            bundle_size=bundle_size,
            covariant_attn_tau_min=covariant_attn_tau_min,
            covariant_attn_denom_min=covariant_attn_denom_min,
            covariant_attn_transport_eps=covariant_attn_transport_eps,
            film_conditioning=film_conditioning,
        )

    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Project raw inputs into the model's normalized coordinate space.

        Args:
            x: Raw input tensor of shape ``[..., input_dim]``.

        Returns:
            torch.Tensor: Normalized tensor of the same shape as ``x``.
        """
        return self.io_affine.normalize(x)

    def denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Project normalized decoder outputs back into raw data coordinates.

        Args:
            x: Normalized tensor of shape ``[..., input_dim]``.

        Returns:
            torch.Tensor: De-normalized tensor of the same shape as ``x``.
        """
        return self.io_affine.denormalize(x)

    def loss_space_pair(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the reconstruction pair in the normalized training space.

        Both tensors are mapped through ``normalize_input`` so that
        reconstruction losses are computed in the model's internal
        coordinate frame.

        Args:
            x: Raw input tensor of shape ``[B, input_dim]``.
            x_recon: Raw reconstruction tensor of shape ``[B, input_dim]``.

        Returns:
            tuple: A 2-tuple of:
                - **x_norm** (*torch.Tensor*): Normalized input, shape
                  ``[B, input_dim]``.
                - **x_recon_norm** (*torch.Tensor*): Normalized
                  reconstruction, shape ``[B, input_dim]``.
        """
        return self.normalize_input(x), self.normalize_input(x_recon)

    @torch.no_grad()
    def set_io_affine_stats(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
        *,
        learnable: bool | None = None,
    ) -> None:
        """Initialize the optional affine map from dataset-level statistics.

        Delegates to ``GlobalAffineMap.set_stats`` and optionally toggles
        the learnable flag.  Runs under ``torch.no_grad()``.

        Args:
            mean: Per-dimension mean of shape ``[input_dim]``.
            std: Per-dimension standard deviation of shape ``[input_dim]``.
            learnable: If not ``None``, set the affine parameters' gradient
                requirement accordingly.
        """
        self.io_affine.set_stats(mean, std)
        if learnable is not None:
            self.io_affine.set_learnable(learnable)

    def decode(
        self,
        z_geo: torch.Tensor,
        chart_index: torch.Tensor | None = None,
        router_weights: torch.Tensor | None = None,
        routing_tau: float = 1.0,
        *,
        return_model_space: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Decode a latent state and optionally expose the normalized output.

        Runs the ``TopologicalDecoder`` and then de-normalizes the result
        back to the raw data coordinate system via ``denormalize_output``.

        Args:
            z_geo: Geometric latent tensor of shape ``[B, latent_dim]`` in
                the Poincare ball.
            chart_index: Optional hard chart assignment of shape ``[B]``.
            router_weights: Optional pre-computed soft chart-routing weights
                of shape ``[B, num_charts]``.
            routing_tau: Temperature for the decoder's covariant router.
            return_model_space: If True, the normalized (pre-denormalization)
                reconstruction is included in ``aux_losses`` under the key
                ``"x_model"``.

        Returns:
            tuple: A 3-tuple containing:
                - **x_raw** (*torch.Tensor*): Reconstruction in raw data
                  coordinates, shape ``[B, input_dim]``.
                - **dec_router_weights** (*torch.Tensor*): Decoder routing
                  weights, shape ``[B, num_charts]``.
                - **aux_losses** (*dict[str, torch.Tensor]*): Auxiliary loss
                  dictionary. Contains ``"x_model"`` of shape
                  ``[B, input_dim]`` when ``return_model_space`` is True.
        """
        x_model, dec_router_weights, aux_losses = self.decoder(
            z_geo,
            chart_index=chart_index,
            router_weights=router_weights,
            routing_tau=routing_tau,
        )
        x_raw = self.denormalize_output(x_model)
        if return_model_space:
            aux_losses = dict(aux_losses)
            aux_losses["x_model"] = x_model
        return x_raw, dec_router_weights, aux_losses

    def forward(
        self,
        x: torch.Tensor,
        routing_tau: float = 1.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        """Run the full encode-decode forward pass.

        Normalizes the input, encodes through the ``AttentiveAtlasEncoder``,
        decodes via the ``TopologicalDecoder`` (reusing the encoder's router
        weights), and de-normalizes the reconstruction.

        Args:
            x: Raw input tensor of shape ``[B, input_dim]``.
            routing_tau: Temperature scaling for both the encoder and decoder
                covariant router softmax.

        Returns:
            tuple: A 9-tuple containing:
                - **x_recon** (*torch.Tensor*): Reconstruction in raw data
                  coordinates, shape ``[B, input_dim]``.
                - **vq_loss** (*torch.Tensor*): Scalar vector-quantization
                  loss from the encoder.
                - **enc_router_weights** (*torch.Tensor*): Encoder routing
                  weights, shape ``[B, num_charts]``.
                - **dec_router_weights** (*torch.Tensor*): Decoder routing
                  weights, shape ``[B, num_charts]``.
                - **K_chart** (*torch.Tensor*): Winning chart index per
                  sample, shape ``[B]``.
                - **z_geo** (*torch.Tensor*): Geometric latent in the
                  Poincare ball, shape ``[B, latent_dim]``.
                - **z_n** (*torch.Tensor*): Router-blended nuisance component
                  in tangent space, shape ``[B, latent_dim]``.
                - **c_bar** (*torch.Tensor*): Router-blended chart center in
                  the Poincare ball, shape ``[B, latent_dim]``.
                - **aux_losses** (*dict[str, torch.Tensor]*): Auxiliary loss
                  dictionary from the decoder.
        """
        (
            K_chart,
            _K_code,
            z_n,
            _z_tex,
            enc_router_weights,
            z_geo,
            vq_loss,
            _indices,
            _z_n_all,
            c_bar,
            _v_local,
            _z_q_blended,
        ) = self.encoder(
            self.normalize_input(x),
            routing_tau=routing_tau,
        )

        router_override = enc_router_weights
        x_recon, dec_router_weights, aux_losses = self.decode(
            z_geo,
            chart_index=None,
            router_weights=router_override,
            routing_tau=routing_tau,
        )

        return (
            x_recon,
            vq_loss,
            enc_router_weights,
            dec_router_weights,
            K_chart,
            z_geo,
            z_n,
            c_bar,
            aux_losses,
        )

    def compute_consistency_loss(
        self, enc_weights: torch.Tensor, dec_weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """Compute KL-divergence consistency loss between encoder and decoder routing.

        Encourages the decoder's chart routing to agree with the encoder's
        by minimizing ``KL(enc_weights || dec_weights)``.

        Args:
            enc_weights: Encoder routing weights of shape
                ``[B, num_charts]``.
            dec_weights: Decoder routing weights of shape
                ``[B, num_charts]``.
            eps: Small constant added to both distributions for numerical
                stability of the logarithm.

        Returns:
            torch.Tensor: Scalar mean KL-divergence across the batch.
        """
        kl = (enc_weights * torch.log((enc_weights + eps) / (dec_weights + eps))).sum(dim=-1)
        return kl.mean()

    def compute_perplexity(self, K_chart: torch.Tensor) -> float:
        """Compute the perplexity of the chart usage distribution.

        Perplexity measures how uniformly the charts are utilized. A value
        equal to ``num_charts`` indicates perfectly uniform usage.

        Args:
            K_chart: Winning chart indices of shape ``[B]`` (integer tensor).

        Returns:
            float: Perplexity of the empirical chart distribution,
                computed as ``exp(entropy)``.
        """
        counts = torch.bincount(K_chart, minlength=self.num_charts).float()
        probs = counts / counts.sum()
        probs = probs[probs > 0]
        entropy = -(probs * torch.log(probs)).sum()
        return math.exp(entropy.item())
