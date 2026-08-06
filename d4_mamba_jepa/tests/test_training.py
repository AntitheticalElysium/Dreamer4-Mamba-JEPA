import torch

from d4_mamba_jepa.tests.synthetic import moving_square_batch
from d4_mamba_jepa.model import D4LiteWorld
from d4_mamba_jepa.objectives import optimizer_groups
from d4_mamba_jepa.tests.test_baseline import tiny_config
from d4_mamba_jepa.training import (
    LossWeights,
    WorldLossNormalizer,
    reward_mtp_loss,
    world_loss,
)


def test_reward_mtp_uses_led_to_current_and_future_slots():
    # Centers [-1,0,1], predict zero everywhere. Only verify alignment/masking
    # through finite gradients; operator correctness is upstream-owned.
    logits = torch.zeros(1, 4, 2, 3, requires_grad=True)
    centers = torch.tensor([-1.0, 0.0, 1.0])
    rewards = torch.tensor([[0.0, 0.0, 1.0, -1.0]])
    valid = torch.tensor([[False, True, True, True]])
    loss = reward_mtp_loss(logits, centers, rewards, valid)
    loss.backward()
    assert torch.isfinite(loss)
    assert logits.grad[0, 0, 0].abs().sum().item() == 0.0
    assert logits.grad[0, 0, 1].abs().sum().item() > 0.0


def test_world_loss_base_keeps_tokenizer_frozen():
    torch.manual_seed(41)
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    batch = moving_square_batch(cfg, batch_size=2, device="cpu", seed=3)
    normalizer = WorldLossNormalizer()
    loss, metrics = world_loss(world, batch, normalizer=normalizer)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in world.encoder.parameters())
    assert metrics["loss/cdp"].item() == 0.0
    assert metrics["loss/reconstruction"].item() == 0.0


def test_world_loss_cdp_has_only_registered_encoder_routes():
    torch.manual_seed(43)
    cfg = tiny_config(representation_objective="cdp")
    world = D4LiteWorld(cfg)
    batch = moving_square_batch(cfg, batch_size=2, device="cpu", seed=5)
    normalizer = WorldLossNormalizer()
    loss, metrics = world_loss(
        world,
        batch,
        normalizer=normalizer,
        weights=LossWeights(flow=1.0, reward=1.0, continuation=1.0),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert metrics["loss/cdp"].item() > 0.0
    assert metrics["loss/reconstruction"].item() > 0.0
    assert any(parameter.grad is not None for parameter in world.encoder.parameters())
    assert all(parameter.grad is None for parameter in world.decoder.parameters())


def test_optimizer_groups_cover_every_trainable_parameter_once():
    world = D4LiteWorld(tiny_config(representation_objective="cdp"))
    groups = optimizer_groups(world, 1e-4)
    grouped = [parameter for group in groups for parameter in group["params"]]
    expected = [parameter for parameter in world.parameters() if parameter.requires_grad]
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in expected
    }
    assert len(grouped) == len({id(parameter) for parameter in grouped})


def test_task_heads_share_the_noised_flow_forward(monkeypatch):
    torch.manual_seed(89)
    cfg = tiny_config()
    world = D4LiteWorld(cfg)
    batch = moving_square_batch(cfg, batch_size=2, device="cpu", seed=17)

    def reject_second_clean_forward(*args, **kwargs):
        raise AssertionError("world_loss made a second clean task-head pass")

    monkeypatch.setattr(world, "forward_dynamics", reject_second_clean_forward)
    loss, _ = world_loss(
        world,
        batch,
        normalizer=WorldLossNormalizer(),
    )
    loss.backward()
    assert world.reward_head.out.weight.grad is not None
    assert world.continuation_head.out.weight.grad is not None
