import torch

from fragile.core.layers import (
    AdaptiveL1Scheduler,
    BundleConfig,
    compute_diagnostics,
    CovariantAttentionLayer,
    FactoredTensorLayer,
    GramInteractionLayer,
    L1Scheduler,
    log_sparsity_diagnostics,
    NormInteractionLayer,
    SoftEquivariantLayer,
    train_ugn,
    UGNConfig,
    UniversalGeometricNetwork,
)


def test_factored_tensor_layer_shape() -> None:
    torch.manual_seed(0)
    layer = FactoredTensorLayer(d_C=3, d_L=2, d_Y=4, rank=5, d_out=7)
    z_C = torch.randn(4, 3)
    z_L = torch.randn(4, 2)
    z_Y = torch.randn(4, 4)
    out = layer(z_C, z_L, z_Y)
    assert out.shape == (4, 7)


def test_interaction_layers_shapes() -> None:
    torch.manual_seed(1)
    z = torch.randn(3, 4, 6)
    norm_layer = NormInteractionLayer(n_bundles=4, hidden_dim=16)
    gram_layer = GramInteractionLayer(n_bundles=4, hidden_dim=16)

    norm_out = norm_layer(z)
    gram_out = gram_layer(z)

    assert norm_out.shape == z.shape
    assert gram_out.shape == z.shape
    assert torch.isfinite(norm_out).all()
    assert torch.isfinite(gram_out).all()


def test_soft_equivariant_layer_shapes_and_losses() -> None:
    torch.manual_seed(2)
    layer = SoftEquivariantLayer(n_bundles=3, bundle_dim=4, zero_self_mixing=True)
    z = torch.randn(2, 3, 4)
    out = layer(z)
    out_flat = layer(z.view(2, -1))

    assert out.shape == z.shape
    assert out_flat.shape == (2, 12)
    assert layer.l1_loss().ndim == 0
    assert layer.mixing_strength() > 0


def test_ugn_forward_and_losses() -> None:
    torch.manual_seed(3)
    config = UGNConfig(
        input_dim=6,
        output_dim=5,
        n_bundles=2,
        bundle_dim=4,
        n_latent_layers=2,
        lambda_equiv=0.0,
        use_spectral_norm=False,
    )
    model = UniversalGeometricNetwork(config)
    x = torch.randn(4, 6)
    y = torch.randn(4, 5)

    out = model(x)
    losses = model.total_loss(x, y)

    assert out.shape == (4, 5)
    assert losses["total"].ndim == 0
    assert losses["task"].ndim == 0
    assert losses["l1"].ndim == 0

    bundles = [BundleConfig(name="a", dim=3), BundleConfig(name="b", dim=5)]
    config_het = UGNConfig(
        input_dim=7,
        output_dim=2,
        bundles=bundles,
        n_latent_layers=1,
        encoder_hidden_dim=8,
        decoder_hidden_dim=8,
        use_spectral_norm=False,
    )
    model_het = UniversalGeometricNetwork(config_het)
    out_het = model_het(torch.randn(2, 7))
    assert out_het.shape == (2, 2)
    assert model_het.equivariance_violation().ndim == 0


def test_l1_schedulers() -> None:
    scheduler = L1Scheduler(lambda_init=0.1, target_violation=0.1, adaptation_rate=0.5)
    higher = scheduler.step(0.2)
    lower = scheduler.step(0.0)
    assert higher > 0.1
    assert lower < higher

    adaptive = AdaptiveL1Scheduler(
        initial_lambda=0.05,
        target_violation=0.1,
        learning_rate=0.5,
        min_lambda=0.01,
        max_lambda=0.2,
    )
    updated = adaptive.step(0.3)
    assert adaptive.history["lambda_l1"][-1] == updated
    assert adaptive.history["violation"][-1] == 0.3
    assert adaptive.min_lambda <= updated <= adaptive.max_lambda


def test_covariant_attention_layer_shape() -> None:
    torch.manual_seed(4)
    layer = CovariantAttentionLayer(bundle_dims=[4, 4], n_heads=1, use_wilson_lines=False)
    z = torch.randn(2, 8)
    context = torch.randn(2, 3, 8)
    out = layer(z, context=context)
    assert out.shape == (2, 8)


def test_ugn_diagnostics_and_training_helpers() -> None:
    torch.manual_seed(5)
    config = UGNConfig(
        input_dim=4,
        output_dim=3,
        n_bundles=2,
        bundle_dim=2,
        n_latent_layers=1,
        encoder_hidden_dim=4,
        decoder_hidden_dim=4,
        use_spectral_norm=False,
    )
    model = UniversalGeometricNetwork(config)
    x_sample = torch.randn(3, 4)

    diagnostics = compute_diagnostics(model, x_sample)
    assert "node_67_gauge_invariance" in diagnostics
    assert "l1_mixing_strength" in diagnostics
    assert "equivariance_violation" in diagnostics

    logs = log_sparsity_diagnostics(model, step=1, logger=None)
    assert any(key.endswith("/sparsity") for key in logs)

    dataset = torch.utils.data.TensorDataset(
        torch.randn(6, 4),
        torch.randn(6, 3),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=3)
    history = train_ugn(
        model,
        loader,
        loader,
        n_epochs=1,
        lr=1e-3,
        use_adaptive_l1=False,
        device="cpu",
    )
    assert len(history["train_loss"]) == 1
