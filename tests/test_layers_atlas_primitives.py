import torch

from fragile.layers import (
    AttentiveAtlasEncoder,
    TopoEncoder,
    TopologicalDecoder,
)


def test_attentive_atlas_encoder_shapes() -> None:
    torch.manual_seed(0)
    encoder = AttentiveAtlasEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(4, 3)
    (
        K_chart,
        K_code,
        z_n,
        z_tex,
        router_weights,
        z_geo,
        vq_loss,
        indices_stack,
        z_n_all_charts,
        c_bar,
        _v_local,
        _z_q_blended,
    ) = encoder(x)

    assert K_chart.shape == (4,)
    assert K_code.shape == (4,)
    assert z_n.shape == (4, 2)
    assert z_tex.shape == (4, 2)
    assert router_weights.shape == (4, 3)
    assert z_geo.shape == (4, 2)
    assert vq_loss.ndim == 0
    assert indices_stack.shape == (4, 3)
    assert z_n_all_charts.shape == (4, 3, 2)
    assert c_bar.shape == (4, 2)


def test_topological_decoder_shapes() -> None:
    torch.manual_seed(1)
    decoder = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=3,
        output_dim=3,
    )
    z_geo = torch.randn(4, 2)
    x_hat, router_weights, aux_losses = decoder(z_geo)

    assert x_hat.shape == (4, 3)
    assert router_weights.shape == (4, 3)
    assert torch.allclose(router_weights.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert isinstance(aux_losses, dict)


def test_topoencoder_forward_and_losses() -> None:
    torch.manual_seed(2)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(5, 3)
    x_recon, vq_loss, enc_weights, dec_weights, K_chart, z_geo, z_n, c_bar, aux_losses = model(x)

    assert x_recon.shape == (5, 3)
    assert vq_loss.ndim == 0
    assert enc_weights.shape == (5, 3)
    assert dec_weights.shape == (5, 3)
    assert K_chart.shape == (5,)
    assert z_geo.shape == (5, 2)
    assert z_n.shape == (5, 2)
    assert c_bar.shape == (5, 2)
    assert isinstance(aux_losses, dict)

    consistency = model.compute_consistency_loss(enc_weights, dec_weights)
    assert consistency.ndim == 0
    assert model.compute_perplexity(K_chart) > 0.0


def test_decoder_film_conditioning() -> None:
    """Conv decoder with FiLM conditioning produces correct shapes."""
    torch.manual_seed(3)
    decoder = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=5,
        output_dim=784,
        film_conditioning=True,
    )
    z_geo = torch.randn(4, 2)
    x_hat, router_weights, aux_losses = decoder(z_geo)

    assert x_hat.shape == (4, 784)
    assert router_weights.shape == (4, 5)
    assert isinstance(aux_losses, dict)


def test_decoder_conformal_freq_gating() -> None:
    """Conformal frequency gating produces correct shapes and modifies output."""
    torch.manual_seed(4)
    decoder_plain = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=3,
        output_dim=784,
    )
    decoder_gated = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=3,
        output_dim=784,
    )
    # Copy weights for fair comparison
    decoder_gated.load_state_dict(decoder_plain.state_dict())

    z_geo = torch.randn(4, 2)
    x_plain, _, _ = decoder_plain(z_geo)
    x_gated, _, _ = decoder_gated(z_geo)

    assert x_gated.shape == (4, 784)
    # Gating should modify the output
    assert not torch.allclose(x_plain, x_gated, atol=1e-6)


def test_decoder_all_features_combined() -> None:
    """All features together produce correct shapes."""
    torch.manual_seed(6)
    decoder = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=32,
        num_charts=5,
        output_dim=784,
        film_conditioning=True,
    )
    z_geo = torch.randn(4, 2)
    x_hat, router_weights, aux_losses = decoder(z_geo)

    assert x_hat.shape == (4, 784)
    assert router_weights.shape == (4, 5)
    assert isinstance(aux_losses, dict)


def test_routing_produces_valid_weights() -> None:
    """Routing produces valid probability distributions with correct shapes."""
    torch.manual_seed(10)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(8, 3)
    x_recon, vq_loss, enc_weights, dec_weights, K_chart, z_geo, _z_n, _c_bar, _aux_losses = model(
        x
    )

    # Shapes unchanged
    assert x_recon.shape == (8, 3)
    assert enc_weights.shape == (8, 3)
    assert dec_weights.shape == (8, 3)
    assert K_chart.shape == (8,)
    assert z_geo.shape == (8, 2)

    # Encoder weights should be a valid probability distribution
    assert torch.allclose(enc_weights.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (enc_weights >= 0).all()

    # Outputs finite
    assert torch.isfinite(x_recon).all()
    assert torch.isfinite(vq_loss)


def test_routing_gradients_flow() -> None:
    """Gradients flow through routing to encoder parameters."""
    torch.manual_seed(11)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(8, 3)
    x_recon, vq_loss, _enc_weights, _dec_weights, _K_chart, _z_geo, _z_n, _c_bar, _aux_losses = (
        model(x)
    )

    loss = torch.nn.functional.mse_loss(x_recon, x) + vq_loss
    loss.backward()

    # Encoder parameters should receive gradients
    has_grad = False
    for p in model.encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowed to encoder parameters"


def test_forward_outputs_valid_and_gradients_flow() -> None:
    """Forward pass produces valid outputs and gradients flow."""
    torch.manual_seed(12)
    model = TopoEncoder(
        input_dim=3,
        hidden_dim=32,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(8, 3)

    out = model(x)
    x_recon, vq_loss, enc_w = out[0], out[1], out[2]

    # Valid probability distribution
    assert torch.allclose(enc_w.sum(dim=-1), torch.ones(8), atol=1e-5)
    assert (enc_w >= 0).all()

    # Outputs are finite
    assert torch.isfinite(x_recon).all()
    assert torch.isfinite(vq_loss)

    # Gradients flow
    loss = torch.nn.functional.mse_loss(x_recon, x) + vq_loss
    loss.backward()

    has_grad = False
    for p in model.encoder.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients through forward path"
