class AttentiveAtlasEncoder(nn.Module):
    """Attentive Atlas encoder with cross-attention routing."""

    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 32,
        latent_dim: int = 2,
        num_charts: int = 3,
        codes_per_chart: int = 21,
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
        super().__init__()
        self.num_charts = num_charts
        self.latent_dim = latent_dim
        self.codes_per_chart = codes_per_chart
        self._commitment_beta = commitment_beta
        self._codebook_loss_weight = codebook_loss_weight

        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.val_proj = nn.Linear(hidden_dim, latent_dim)

        # Unit-sphere init: each chart gets a distinct catchment region from
        # the first forward pass, preventing softmax winner-take-all collapse.
        self.chart_centers = nn.Parameter(
            torch.nn.functional.normalize(torch.randn(num_charts, latent_dim), dim=-1)
        )

        self.codebook = nn.Parameter(torch.randn(num_charts, codes_per_chart, latent_dim) * 0.02)

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
            _init_soft_equiv_layers(self.soft_equiv_layers)

        self.soft_equiv_soft_assign = soft_equiv_soft_assign
        self.soft_equiv_temperature = soft_equiv_temperature
        if soft_equiv_metric and self.soft_equiv_soft_assign and self.soft_equiv_temperature <= 0:
            msg = "soft_equiv_temperature must be positive when soft_equiv_soft_assign is enabled."
            raise ValueError(msg)
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None
        self._last_soft_equiv_log_ratio: torch.Tensor | None = None

        self.structure_filter = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2 if latent_dim > 2 else latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim // 2 if latent_dim > 2 else latent_dim, latent_dim),
        )

    def _encode_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)

    def _apply_soft_equiv_metric(self, diff: torch.Tensor) -> torch.Tensor:
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
        if self.soft_equiv_layers is None:
            return torch.tensor(0.0, device=self.codebook.device)
        total = torch.zeros((), device=self.codebook.device)
        for layer in self.soft_equiv_layers:
            total += layer.l1_loss()
        return total / len(self.soft_equiv_layers)

    def soft_equiv_log_ratio_loss(self) -> torch.Tensor:
        if self.soft_equiv_layers is None or self._last_soft_equiv_log_ratio is None:
            return torch.tensor(0.0, device=self.codebook.device)
        return self._last_soft_equiv_log_ratio

    def forward(
        self,
        x: torch.Tensor,
        hard_routing: bool = False,
        hard_routing_tau: float = 1.0,
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
    ]:
        """Forward pass through the attentive atlas.

        Returns:
            K_chart: [B] chart assignment
            K_code: [B] code index within chart
            z_n: [B, D] nuisance latent
            z_tex: [B, D] texture residual
            router_weights: [B, N_c] routing weights
            z_geo: [B, D] geometric latent
            vq_loss: [] VQ loss
            indices_stack: [B, N_c] code indices per chart
            z_n_all_charts: [B, N_c, D] per-chart nuisance
            c_bar: [B, D] chart center mixture
            v_local: [B, D] local residual (for code collapse penalty)
        """
        # Extract features via fully-connected front-end.
        features = self._encode_features(x)  # [B, H]

        # Map features into chart-space coordinates.
        v = self.val_proj(features)  # [B, D]
        scores = torch.matmul(v, self.chart_centers.t()) / math.sqrt(self.latent_dim)  # [B, N_c]
        # Chart routing distributes mass across atlas charts.
        router_weights = _routing_weights(scores, hard_routing, hard_routing_tau)  # [B, N_c]
        K_chart = torch.argmax(router_weights, dim=1)  # [B]

        # Chart-center mixture defines the macro coordinate (hyperbolic barycenter).
        c_bar = _poincare_weighted_mean(self.chart_centers, router_weights)  # [B, D]
        v_local = _project_to_ball(mobius_add(-c_bar, v))  # [B, D]

        # Per-chart codebook lookup using hyperbolic geometry.
        codebook = _project_to_ball(self.codebook)  # [N_c, K, D]
        v_exp = v_local.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D]
        codebook_exp = codebook.unsqueeze(0)  # [1, N_c, K, D]
        diff = mobius_add(-codebook_exp, v_exp)  # [B, N_c, K, D]
        diff_tan = log_map_zero(diff)
        dist = self._apply_soft_equiv_metric(diff_tan)  # [B, N_c, K]
        indices = torch.argmin(dist, dim=-1)  # [B, N_c]
        indices_stack = indices  # [B, N_c]

        indices_exp = indices.unsqueeze(-1).unsqueeze(-1)  # [B, N_c, 1, 1]
        indices_exp = indices_exp.expand(-1, -1, 1, self.latent_dim)  # [B, N_c, 1, D]
        z_q_all = torch.gather(codebook.expand(v.shape[0], -1, -1, -1), 2, indices_exp)
        z_q_all = z_q_all.squeeze(2)  # [B, N_c, D]
        if self.soft_equiv_layers is not None and self.soft_equiv_soft_assign:
            temperature = max(self.soft_equiv_temperature, 1e-6)
            weights = F.softmax(-dist / temperature, dim=-1)
            z_q_soft = _poincare_weighted_mean_per_chart(codebook, weights)
            # Straight-through soft assignment so gradients reach the metric network.
            z_q_all = z_q_all + z_q_soft - z_q_soft.detach()

        # VQ loss using tangent-space distances in the Poincaré ball.
        w = router_weights.unsqueeze(-1).detach()  # [B, N_c, 1]
        v_bc = v_local.unsqueeze(1)  # [B, 1, D]
        delta_commit = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))
        commitment = (delta_commit**2 * w).mean(dim=(0, 2)).sum()  # []
        delta_codebook = log_map_zero(mobius_add(-v_bc.detach(), z_q_all))
        codebook_loss = (delta_codebook**2 * w).mean(dim=(0, 2)).sum()  # []
        vq_loss = self._codebook_loss_weight * codebook_loss + self._commitment_beta * commitment

        # Blend chart codes to form macro latent (hyperbolic barycenter).
        z_q_blended = _poincare_weighted_mean(z_q_all, router_weights)  # [B, D]
        K_code = indices_stack.gather(1, K_chart.unsqueeze(1)).squeeze(1)  # [B]

        # Structure filter extracts nuisance in tangent space.
        delta = log_map_zero(mobius_add(-z_q_all.detach(), v_bc))  # [B, N_c, D]
        z_n_all = self.structure_filter(delta.reshape(-1, self.latent_dim))  # [B*N_c, D]
        z_n_all_charts_tan = z_n_all.view(v.shape[0], self.num_charts, self.latent_dim)
        z_n_all_charts = _project_to_ball(exp_map_zero(z_n_all_charts_tan))  # [B, N_c, D]

        # Texture residual is what's left after nuisance subtraction.
        z_n_tan = (z_n_all_charts_tan * router_weights.unsqueeze(-1)).sum(dim=1)  # [B, D]
        delta_blended = log_map_zero(mobius_add(-z_q_blended.detach(), v_local))  # [B, D]
        z_tex = delta_blended - z_n_tan  # [B, D]

        # Geometric latent = chart center + macro code + nuisance (Möbius sums).
        delta_to_code = log_map_zero(mobius_add(-v_local, z_q_blended))
        z_q_st = mobius_add(v_local, exp_map_zero(delta_to_code.detach()))
        z_geo = _project_to_ball(
            mobius_add(c_bar, mobius_add(z_q_st, exp_map_zero(z_n_tan)))
        )  # [B, D]

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
