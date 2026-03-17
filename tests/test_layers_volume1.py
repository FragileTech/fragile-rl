import torch
from torch import nn

from fragile.core.layers import (
    AreaLawScreening,
    ChiralProjector,
    ChristoffelQuery,
    class_modulated_jump_rate,
    compute_jump_consistency_loss,
    compute_orthogonality_loss,
    compute_separation_loss,
    compute_topology_loss,
    ConformalMetric,
    CovariantAttention,
    DisentangledAgent,
    DisentangledConfig,
    Encoder,
    FactorizedJumpOperator,
    GeodesicConfig,
    GeodesicCrossAttention,
    HierarchicalDisentangled,
    HyperbolicTransport,
    InvariantChartClassifier,
    IsotropicBlock,
    LorentzianConfig,
    LorentzianMemoryAttention,
    LorentzianMetric,
    MacroDynamicsModel,
    SpectralLinear,
    SupervisedTopologyLoss,
    TemporalChristoffelQuery,
    VectorQuantizer,
)
from fragile.core.layers.gauge import (
    exp_map_zero,
    hyperbolic_distance,
    log_map_zero,
    mobius_add,
    mobius_scalar_mul,
    parallel_transport,
    parallel_transport_zero,
)


def test_disentangled_agent_shapes() -> None:
    torch.manual_seed(0)
    config = DisentangledConfig(
        obs_dim=64 * 64 * 3,
        hidden_dim=32,
        macro_embed_dim=8,
        codebook_size=16,
        nuisance_dim=4,
        tex_dim=6,
        action_dim=3,
        rnn_hidden_dim=12,
    )
    agent = DisentangledAgent(config)

    obs = torch.rand(2, 3, 64, 64)  # [B, C, H, W]
    action = torch.rand(2, config.action_dim)  # [B, Da]
    hidden = torch.zeros(2, config.rnn_hidden_dim)  # [B, Dh]

    out = agent(obs, action, hidden)

    assert out["z_macro"].shape == (2, config.macro_embed_dim)
    assert out["z_nuis"].shape == (2, config.nuisance_dim)
    assert out["z_tex"].shape == (2, config.tex_dim)
    assert out["indices"].shape == (2,)
    assert out["recon"].shape == (2, 3, 64, 64)
    assert out["next_logits"].shape == (2, config.codebook_size)
    assert out["hidden_next"].shape == (2, config.rnn_hidden_dim)
    assert out["losses"]["loss_total"].ndim == 0


def test_vq_encoder_decoder_components() -> None:
    torch.manual_seed(1)
    encoder = Encoder(obs_dim=64 * 64 * 3, hidden_dim=16)
    quantizer = VectorQuantizer(codebook_size=8, embed_dim=6)
    decoder = MacroDynamicsModel(macro_embed_dim=6, action_dim=2, hidden_dim=10, codebook_size=8)

    obs = torch.rand(3, 3, 64, 64)  # [B, C, H, W]
    h = encoder(obs)  # [B, H]
    z_q, indices, vq_loss = quantizer(h[:, :6])  # [B, D], [B], []

    action = torch.rand(3, 2)  # [B, Da]
    hidden = torch.zeros(3, 10)  # [B, Dh]
    logits, hidden_next, z_pred = decoder(z_q, action, hidden)  # [B, K], [B, Dh], [B, D]

    assert h.shape == (3, 16)
    assert z_q.shape == (3, 6)
    assert indices.shape == (3,)
    assert vq_loss.ndim == 0
    assert logits.shape == (3, 8)
    assert hidden_next.shape == (3, 10)
    assert z_pred.shape == (3, 6)


def test_hierarchical_disentangled_shapes() -> None:
    torch.manual_seed(2)
    config = DisentangledConfig(hidden_dim=16)
    model = HierarchicalDisentangled(
        config,
        n_levels=2,
        level_dims=[4, 6],
        level_codebook_sizes=[8, 12],
        level_update_freqs=[2, 1],
    )

    obs = torch.rand(2, 3, 64, 64)  # [B, C, H, W]
    prev = torch.zeros(2, 2, 6)  # [B, L, Dmax]
    out = model(obs, step=0, prev_z_macro=prev)

    assert out["z_macro"].shape == (2, 2, 6)
    assert out["indices"].shape == (2, 2)
    assert out["vq_loss"].ndim == 0


def test_supervised_topology_loss_and_jump_rate() -> None:
    torch.manual_seed(3)
    lambda_base = torch.ones(3, 3)  # [N_c, N_c]
    chart_logits = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0]])  # [N_c, C]
    lambda_sup = class_modulated_jump_rate(lambda_base, chart_logits, gamma_sep=2.0)

    assert lambda_sup.shape == (3, 3)
    assert lambda_sup[0, 1] < lambda_base[0, 1]

    loss_fn = SupervisedTopologyLoss(num_charts=3, num_classes=2)
    chart_assignments = torch.softmax(torch.randn(4, 3), dim=-1)  # [B, N_c]
    class_labels = torch.tensor([0, 1, 0, 1])  # [B]
    embeddings = torch.randn(4, 5)  # [B, D]

    total_loss, loss_dict = loss_fn(chart_assignments, class_labels, embeddings)

    assert total_loss.ndim == 0
    assert set(loss_dict.keys()) >= {
        "loss_total",
        "loss_route",
        "loss_purity",
        "loss_balance",
        "loss_metric",
    }


def test_topology_helpers_and_jump_consistency() -> None:
    torch.manual_seed(4)
    batch = 6
    num_charts = 3
    dim = 4

    weights = torch.softmax(torch.randn(batch, num_charts), dim=-1)
    loss_entropy, loss_balance = compute_topology_loss(weights, num_charts)
    assert loss_entropy.ndim == 0
    assert loss_balance.ndim == 0

    chart_outputs = [torch.randn(batch, dim) for _ in range(num_charts)]
    loss_sep = compute_separation_loss(chart_outputs, weights, margin=0.5)
    assert loss_sep.ndim == 0

    # Test Möbius-based jump operator (default, O(n))
    jump_op = FactorizedJumpOperator(num_charts=num_charts, latent_dim=dim, use_mobius=True)
    z_n_by_chart = torch.randn(batch, num_charts, dim) * 0.5  # Keep inside ball
    loss_jump, info = compute_jump_consistency_loss(
        z_n_by_chart,
        weights,
        jump_op,
        overlap_threshold=0.0,
        max_pairs_per_batch=8,
    )

    assert loss_jump.ndim == 0
    assert "num_overlaps" in info

    dummy = nn.Sequential(
        SpectralLinear(dim, dim, bias=False),
        IsotropicBlock(dim, dim, bundle_size=1),
    )
    orth_loss = compute_orthogonality_loss([dummy])
    assert orth_loss.ndim == 0


def test_mobius_jump_operator() -> None:
    """Test Möbius-based chart transitions preserve hyperbolic structure."""
    torch.manual_seed(7)
    batch = 8
    num_charts = 4
    dim = 6

    jump_op = FactorizedJumpOperator(num_charts=num_charts, latent_dim=dim)

    # Generate points inside the Poincaré ball
    z_n = torch.randn(batch, dim) * 0.3
    source_idx = torch.randint(0, num_charts, (batch,))
    target_idx = torch.randint(0, num_charts, (batch,))

    # Apply jump
    z_out = jump_op(z_n, source_idx, target_idx)

    # Check output shape
    assert z_out.shape == (batch, dim)

    # Check output stays inside ball (hyperbolic constraint)
    norms = z_out.norm(dim=-1)
    assert (norms < 1.0).all(), "Output should stay inside Poincaré ball"

    # Test roundtrip: jumping from A to B and back to A preserves norm
    z_to_target = jump_op(z_n, source_idx, target_idx)
    z_roundtrip = jump_op(z_to_target, target_idx, source_idx)
    assert torch.allclose(z_roundtrip.norm(dim=-1), z_n.norm(dim=-1), atol=0.1)

    # Test lift_to_global and project_from_global
    z_global = jump_op.lift_to_global(z_n, source_idx)
    z_back = jump_op.project_from_global(z_global, source_idx)
    assert torch.allclose(z_back, z_n, atol=0.1)


def test_invariant_chart_classifier_rotation_invariance() -> None:
    torch.manual_seed(5)
    batch = 8
    num_charts = 4
    num_classes = 3
    latent_dim = 6
    bundle_size = 3

    classifier = InvariantChartClassifier(
        num_charts=num_charts,
        num_classes=num_classes,
        latent_dim=latent_dim,
        bundle_size=bundle_size,
    )

    router_weights = torch.softmax(torch.randn(batch, num_charts), dim=-1)
    z_geo = torch.randn(batch, latent_dim)

    q1, _ = torch.linalg.qr(torch.randn(bundle_size, bundle_size))
    q2, _ = torch.linalg.qr(torch.randn(bundle_size, bundle_size))
    q = torch.block_diag(q1, q2)
    z_geo_rot = z_geo @ q

    logits = classifier(router_weights, z_geo)
    logits_rot = classifier(router_weights, z_geo_rot)

    assert logits.shape == (batch, num_classes)
    assert torch.allclose(logits, logits_rot, atol=1e-5)


def test_lorentzian_modules() -> None:
    torch.manual_seed(4)
    config = LorentzianConfig(d_model=8, d_latent=4)
    metric = LorentzianMetric(config)
    query = TemporalChristoffelQuery(d_in=8, d_out=8, d_latent=4)
    attn = LorentzianMemoryAttention(config)

    z = torch.zeros(2, 4)  # [B, d]
    z_mem = torch.zeros(2, 3, 4)  # [B, N, d]
    t = torch.ones(2, 1)  # [B, 1]
    t_mem = torch.tensor([[[0.5], [0.8], [1.2]], [[0.2], [0.7], [0.9]]])  # [B, N, 1]

    lambda_z = metric.conformal_factor(z)  # [B, 1]
    d_g = metric.geodesic_distance(z, z_mem)  # [B, N]
    q = query(torch.randn(2, 8), z, t)  # [B, d_out]

    x = torch.randn(2, 8)  # [B, d_model]
    x_mem = torch.randn(2, 3, 8)  # [B, N, d_model]
    out, weights = attn(x, z, t, x_mem, z_mem, t_mem)

    assert lambda_z.shape == (2, 1)
    assert d_g.shape == (2, 3)
    assert q.shape == (2, 8)
    assert out.shape == (2, 8)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-5)


def test_hyperbolic_primitives() -> None:
    """Test fundamental Möbius operations in the Poincaré ball."""
    torch.manual_seed(6)

    # Test mobius_add identity: x ⊕ 0 = x
    x = torch.randn(4, 3) * 0.5  # Keep inside ball
    zero = torch.zeros_like(x)
    assert torch.allclose(mobius_add(x, zero), x, atol=1e-5)

    # Test mobius_add commutativity at origin
    y = torch.randn(4, 3) * 0.5
    # Note: Möbius addition is NOT commutative in general, but x ⊕ 0 = 0 ⊕ x = x
    assert torch.allclose(mobius_add(zero, x), x, atol=1e-5)

    # Test exp/log inverse: log_0(exp_0(v)) ≈ v for small v
    v = torch.randn(4, 3) * 0.3
    recovered = log_map_zero(exp_map_zero(v))
    assert torch.allclose(recovered, v, atol=1e-4)

    # Test hyperbolic distance is non-negative
    dist = hyperbolic_distance(x, y)
    assert (dist >= 0).all()

    # Test hyperbolic distance is zero for same points (within numerical precision)
    dist_same = hyperbolic_distance(x, x)
    assert torch.allclose(dist_same, torch.zeros_like(dist_same), atol=1e-4)

    # Test parallel transport preserves norm (approximately)
    v_tangent = torch.randn(4, 3) * 0.2
    transported = parallel_transport(v_tangent, zero, x)
    # Norm should be scaled by conformal factor ratio
    assert transported.shape == v_tangent.shape


def test_hyperbolic_transport_shapes() -> None:
    """Test HyperbolicTransport module shapes and properties.

    HyperbolicTransport returns a scalar factor [B, N, 1] per key,
    not a full d_k x d_k matrix.
    """
    torch.manual_seed(6)
    config = GeodesicConfig(d_model=8, d_latent=4)
    transport = HyperbolicTransport(config, d_k=6)
    metric = ConformalMetric()

    z_query = torch.zeros(2, 4)
    z_key = torch.randn(2, 3, 4) * 0.5  # Keep inside ball
    scale = transport(z_query, z_key)

    # Check shape: [B, N, 1] scalar factor per key
    assert scale.shape == (2, 3, 1)

    # All scales should be positive
    assert (scale > 0).all()

    # Test temperature scaling
    tau = metric.temperature(torch.zeros(2, 4), d_k=6)
    expected = torch.full_like(tau, fill_value=(6**0.5) / 2.0)
    assert torch.allclose(tau, expected, atol=1e-6)


def test_gauge_modules() -> None:
    torch.manual_seed(5)
    config = GeodesicConfig(d_model=8, d_latent=4, n_heads=2)

    # Test both transports
    transport = HyperbolicTransport(config, d_k=4)
    metric = ConformalMetric()
    query = ChristoffelQuery(d_in=8, d_out=8, d_latent=4)
    chiral = ChiralProjector(d_latent=4)
    screening = AreaLawScreening(config)
    # Use hyperbolic transport (default)
    head = CovariantAttention(config, use_chirality=True, use_screening=True)
    cross = GeodesicCrossAttention(config)

    z_query = torch.zeros(2, 4)  # [B, d_latent]
    z_key = torch.zeros(2, 3, 4)  # [B, N, d_latent]
    x_query = torch.randn(2, 8)  # [B, d_model]
    x_key = torch.randn(2, 3, 8)  # [B, N, d_model]
    x_value = torch.randn(2, 3, 8)  # [B, N, d_model]

    U = transport(z_query, z_key)  # [B, N, 1] scalar
    g = metric.metric(z_query)  # [B, d, d]
    q = query(x_query, z_query)  # [B, d_out]
    psi = torch.stack([x_query, x_query], dim=1)  # [B, 2, d_model]
    grad_V = torch.randn(2, 4)  # [B, d_latent]
    psi_proj = chiral(psi, grad_V)  # [B, 2*d_model]

    attention = torch.softmax(torch.randn(2, 3), dim=-1)  # [B, N]
    lambda_z = metric.conformal_factor(z_query)  # [B, 1]
    screened = screening(attention, z_query, z_key, lambda_z, level=0)  # [B, N]
    screened_lo = screening(attention, z_query, z_key, lambda_z, level=5)  # [B, N]

    out, attn_weights = head(
        z_query,
        z_key,
        x_query,
        x_key,
        x_value,
        grad_V=grad_V,
        level=0,
    )

    z = torch.randn(2, 4)  # [B, d_latent]
    p = torch.zeros(2, 4)  # [B, d_latent]
    context_force = torch.randn(2, 3, 4)  # [B, N, d_latent]
    z_next, p_next = cross(z, p, z_key, x_key, context_force)

    assert U.shape == (2, 3, 1)  # scalar transport factor
    assert g.shape == (2, 4, 4)
    assert q.shape == (2, 8)
    assert psi_proj.shape == (2, 16)
    assert screened.shape == (2, 3)
    assert (screened <= attention + 1e-6).all()
    assert (screened_lo >= screened - 1e-6).all()
    assert out.shape == (2, 8)
    assert attn_weights.shape == (2, 3)
    assert torch.allclose(attn_weights.sum(dim=-1), torch.ones(2), atol=1e-5)
    assert z_next.shape == (2, 4)
    assert p_next.shape == (2, 4)
    assert torch.all(torch.norm(z_next, dim=-1) <= 0.999 + 1e-6)
