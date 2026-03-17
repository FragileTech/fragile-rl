import torch

from fragile.core.layers import HierarchicalAtlasStack, TopoEncoderAttachments


def test_hierarchical_atlas_stack_shapes_and_jump() -> None:
    torch.manual_seed(0)
    model = HierarchicalAtlasStack(
        input_dim=3,
        hidden_dim=16,
        latent_dim=2,
        num_charts=3,
        codes_per_chart=5,
        n_levels=2,
        enable_cross_level_jump=True,
    )
    x = torch.randn(4, 3)
    outputs = model(x)

    assert len(outputs) == 2
    for level in outputs:
        assert level["x_recon"].shape == (4, 3)
        assert level["vq_loss"].ndim == 0
        assert level["enc_router_weights"].shape == (4, 3)
        assert level["dec_router_weights"].shape == (4, 3)
        assert level["K_chart"].shape == (4,)
        assert level["K_code"].shape == (4,)
        assert level["z_geo"].shape == (4, 2)
        assert level["z_n"].shape == (4, 2)
        assert level["z_tex"].shape == (4, 2)
        assert level["indices_stack"].shape == (4, 3)
        assert level["z_n_all_charts"].shape == (4, 3, 2)
        assert level["c_bar"].shape == (4, 2)
        assert level["z_n_local"].shape == (4, 2)

    assert outputs[0]["z_n_jump_to_next"].shape == (4, 2)
    assert outputs[1]["z_n_jump_from_prev"].shape == (4, 2)


def test_topoencoder_attachments_classifier_output() -> None:
    torch.manual_seed(1)
    attachments = TopoEncoderAttachments(
        num_charts=3,
        latent_dim=2,
        num_classes=4,
        enable_classifier=True,
        enable_jump=True,
    )
    router_weights = torch.softmax(torch.randn(5, 3), dim=-1)
    z_geo = torch.randn(5, 2)
    outputs = attachments(router_weights=router_weights, z_geo=z_geo)

    assert outputs["classifier_logits"].shape == (5, 4)
    assert attachments.jump_operator is not None
