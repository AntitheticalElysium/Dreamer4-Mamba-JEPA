import torch

from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.objectives import (
    cdp_cosine_loss,
    optimizer_groups,
    reconstruction_anchor_loss,
)
from d4_mamba_jepa.tests.test_baseline import tiny_config


def test_cdp_predictor_shape_and_stop_gradient_target():
    torch.manual_seed(31)
    cfg = tiny_config(representation_objective="cdp")
    world = D4LiteWorld(cfg)
    clean = torch.randn(
        1, 2, cfg.n_spatial, cfg.d_spatial, requires_grad=True
    )
    actions = torch.tensor([[-1, 4]])
    loss, prediction = cdp_cosine_loss(
        world, clean=clean, led_to_actions=actions
    )
    assert prediction.shape == (1, 1, cfg.n_spatial, cfg.d_spatial)
    loss.backward()
    assert clean.grad is not None
    assert clean.grad[:, 0].abs().sum().item() > 0.0
    # With T=2, slot one appears only as the detached future target.
    assert clean.grad[:, 1].abs().sum().item() == 0.0


def test_cdp_updates_encoder_context_but_not_frozen_decoder():
    torch.manual_seed(37)
    cfg = tiny_config(representation_objective="cdp")
    world = D4LiteWorld(cfg)
    frames = torch.randint(
        0, 256, (2, cfg.sequence_length, 3, 16, 16), dtype=torch.uint8
    )
    encoded = world.encode_frames(frames, frozen=False)
    actions = torch.tensor([[-1, 0, 1, 2], [-1, 3, 4, 5]])
    cdp, _ = cdp_cosine_loss(
        world, clean=encoded.packed, led_to_actions=actions
    )
    anchor = reconstruction_anchor_loss(
        world, frames=frames, bottleneck=encoded.bottleneck
    )
    (cfg.cdp_weight * cdp + cfg.reconstruction_anchor_weight * anchor).backward()
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in world.encoder.parameters()
    )
    assert all(
        not parameter.requires_grad and parameter.grad is None
        for parameter in world.decoder.parameters()
    )
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum().item() > 0
        for parameter in world.cdp_predictor.parameters()
    )


def test_base_arm_freezes_tokenizer_and_has_no_cdp_predictor():
    world = D4LiteWorld(tiny_config())
    assert world.cdp_predictor is None
    assert all(not parameter.requires_grad for parameter in world.encoder.parameters())
    assert all(not parameter.requires_grad for parameter in world.decoder.parameters())


def test_cdp_optimizer_groups_are_disjoint_and_use_slow_encoder():
    cfg = tiny_config(representation_objective="cdp", encoder_lr_ratio=0.3)
    world = D4LiteWorld(cfg)
    groups = optimizer_groups(world, 1e-4)
    assert [group["name"] for group in groups] == ["encoder", "main"]
    assert groups[0]["lr"] == 3e-5
    assert groups[1]["lr"] == 1e-4
    ids = [id(parameter) for group in groups for parameter in group["params"]]
    assert len(ids) == len(set(ids))
    decoder_ids = {id(parameter) for parameter in world.decoder.parameters()}
    assert decoder_ids.isdisjoint(ids)
