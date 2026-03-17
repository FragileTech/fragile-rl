import torch

from fragile.core.layers import (
    AttentiveAtlasEncoder,
    HierarchicalAtlasStack,
    TopoEncoder,
    TopoEncoderAttachments,
    TopologicalDecoder,
)
from fragile.core.layers.vision import StandardResNetBackbone


def test_attentive_atlas_encoder_shapes() -> None:
    torch.manual_seed(0)
    encoder = AttentiveAtlasEncoder(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(6, 4)
    (
        k_chart,
        k_code,
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

    assert k_chart.shape == (6,)
    assert k_code.shape == (6,)
    assert z_n.shape == (6, 2)
    assert z_tex.shape == (6, 2)
    assert router_weights.shape == (6, 3)
    assert z_geo.shape == (6, 2)
    assert vq_loss.ndim == 0
    assert indices_stack.shape == (6, 3)
    assert z_n_all_charts.shape == (6, 3, 2)
    assert c_bar.shape == (6, 2)
    assert torch.allclose(router_weights.sum(dim=-1), torch.ones(6), atol=1e-5)


def test_topological_decoder_routing() -> None:
    torch.manual_seed(1)
    decoder = TopologicalDecoder(
        latent_dim=2,
        hidden_dim=8,
        num_charts=3,
        output_dim=4,
    )
    z_geo = torch.randn(5, 2)
    x_hat, router_weights = decoder(z_geo)

    assert x_hat.shape == (5, 4)
    assert router_weights.shape == (5, 3)
    assert torch.allclose(router_weights.sum(dim=-1), torch.ones(5), atol=1e-5)

    chart_index = torch.tensor([0, 1, 2, 1, 0])
    _, router_hard = decoder(z_geo, chart_index=chart_index)
    expected = torch.nn.functional.one_hot(chart_index, num_classes=3).float()
    assert torch.allclose(router_hard, expected, atol=1e-6)


def test_topoencoder_forward_shapes() -> None:
    torch.manual_seed(2)
    model = TopoEncoder(
        input_dim=4,
        hidden_dim=8,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
    )
    x = torch.randn(4, 4)
    x_recon, vq_loss, enc_w, dec_w, k_chart, z_geo, z_n, c_bar = model(x)

    assert x_recon.shape == (4, 4)
    assert vq_loss.ndim == 0
    assert enc_w.shape == (4, 3)
    assert dec_w.shape == (4, 3)
    assert k_chart.shape == (4,)
    assert z_geo.shape == (4, 2)
    assert z_n.shape == (4, 2)
    assert c_bar.shape == (4, 2)
    assert torch.allclose(enc_w.sum(dim=-1), torch.ones(4), atol=1e-5)
    assert torch.allclose(dec_w.sum(dim=-1), torch.ones(4), atol=1e-5)

    consistency = model.compute_consistency_loss(enc_w, dec_w)
    assert consistency.ndim == 0
    assert model.compute_perplexity(k_chart) > 0.0


def test_hierarchical_atlas_stack_outputs() -> None:
    torch.manual_seed(3)
    model = HierarchicalAtlasStack(
        input_dim=4,
        hidden_dim=8,
        latent_dim=[2, 3],
        num_charts=[3, 2],
        codes_per_chart=[4, 3],
        n_levels=2,
        level_update_freqs=[1, 1],
        covariant_attn=True,
        covariant_attn_tensorization="sum",
    )
    x = torch.randn(3, 4)
    outputs = model(x)

    assert len(outputs) == 2
    for level, out in enumerate(outputs):
        latent_dim = [2, 3][level]
        num_charts = [3, 2][level]
        assert out["x_recon"].shape == (3, 4)
        assert out["vq_loss"].ndim == 0
        assert out["enc_router_weights"].shape == (3, num_charts)
        assert out["dec_router_weights"].shape == (3, num_charts)
        assert out["K_chart"].shape == (3,)
        assert out["K_code"].shape == (3,)
        assert out["z_geo"].shape == (3, latent_dim)
        assert out["z_n"].shape == (3, latent_dim)
        assert out["z_tex"].shape == (3, latent_dim)
        assert out["indices_stack"].shape == (3, num_charts)
        assert out["z_n_all_charts"].shape == (3, num_charts, latent_dim)
        assert out["c_bar"].shape == (3, latent_dim)


def test_topoencoder_attachments_classifier() -> None:
    torch.manual_seed(4)
    attachments = TopoEncoderAttachments(
        num_charts=3,
        latent_dim=4,
        num_classes=2,
        enable_jump=False,
        enable_classifier=True,
    )
    router_weights = torch.softmax(torch.randn(5, 3), dim=-1)
    z_geo = torch.randn(5, 4)
    outputs = attachments(router_weights=router_weights, z_geo=z_geo)

    assert outputs["classifier_logits"].shape == (5, 2)


def test_standard_resnet_backbone_shape() -> None:
    torch.manual_seed(5)
    backbone = StandardResNetBackbone(in_channels=3, out_dim=16, base_channels=8)
    x = torch.randn(2, 3, 32, 32)
    out = backbone(x)

    assert out.shape == (2, 16)
