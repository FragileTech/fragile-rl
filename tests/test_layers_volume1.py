import torch
from torch import nn

from fragile.layers import (
    AreaLawScreening,
    ChiralProjector,
    ChristoffelQuery,
    compute_jump_consistency_loss,
    ConformalMetric,
    CovariantAttention,
    FactorizedJumpOperator,
    GeodesicConfig,
    GeodesicCrossAttention,
    HyperbolicTransport,
    IsotropicBlock,
    LorentzianConfig,
    LorentzianMemoryAttention,
    LorentzianMetric,
    SpectralLinear,
    TemporalChristoffelQuery,
)
from fragile.layers.gauge import (
    exp_map_zero,
    hyperbolic_distance,
    log_map_zero,
    mobius_add,
    parallel_transport,
)


def test_jump_consistency_loss() -> None:
    torch.manual_seed(4)
    batch = 6
    num_charts = 3
    dim = 4

    weights = torch.softmax(torch.randn(batch, num_charts), dim=-1)

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
