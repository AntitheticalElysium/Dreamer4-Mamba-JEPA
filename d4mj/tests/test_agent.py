import pytest
import torch

from d4mj.agent import Heads, head_loss, head_targets, twohot

from .conftest import latent_batch


def readout_for(config, batch):
    torch.manual_seed(0)
    heads = Heads(config)
    agent = torch.randn(
        batch.led_to_action.shape[0],
        batch.led_to_action.shape[1],
        config.n_agent,
        config.d_model,
        generator=torch.Generator().manual_seed(1),
    )
    return heads, heads(agent) | {"centers": heads.centers}


def test_behaviour_cloning_uses_the_relevant_half_only(config):
    """§4.1 applies BC to the relevant fraction; reward and continuation see both."""
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    heads, readout = readout_for(config, base)

    moved_uniform = latent_batch(config, 4, 6, relevant=half)
    moved_uniform.led_to_action[2:] = 5
    moved_relevant = latent_batch(config, 4, 6, relevant=half)
    moved_relevant.led_to_action[:2] = 5

    reference = head_loss(readout, head_targets(base, config), config)
    uniform = head_loss(readout, head_targets(moved_uniform, config), config)
    relevant = head_loss(readout, head_targets(moved_relevant, config), config)

    assert torch.equal(reference["policy"], uniform["policy"])
    assert not torch.equal(reference["policy"], relevant["policy"])


def test_reward_head_uses_both_halves(config):
    """The paper restricts BC and dynamics by name and says the mixture amplifies
    signal *for* reward modelling, so halving that head's data would be an addition."""
    half = [True, True, False, False]
    base = latent_batch(config, 4, 6, relevant=half)
    heads, readout = readout_for(config, base)
    for rows in (slice(0, 2), slice(2, 4)):
        moved = latent_batch(config, 4, 6, relevant=half)
        moved.reward[rows] = 7.0
        assert not torch.equal(
            head_loss(readout, head_targets(base, config), config)["reward"],
            head_loss(readout, head_targets(moved, config), config)["reward"],
        )


def test_behaviour_cloning_requires_the_mixture(config):
    """Pretraining batches carry no roles, and BC must not silently score them all."""
    with pytest.raises(AssertionError, match="mixture"):
        head_targets(latent_batch(config, 4, 6), config)


def test_reward_lead_zero_is_the_arriving_reward(config):
    """The reward caused by the action at block t is lead 0 of block t+1, never
    lead 0 of block t."""
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.reward[:] = torch.arange(6).float()
    targets = head_targets(batch, config)
    assert torch.equal(targets["reward"][0, :, 0], torch.arange(6).float())


def test_policy_lead_zero_is_the_outgoing_action(config):
    """Led-to storage puts the outgoing action one block later."""
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.led_to_action[:] = torch.arange(6)
    targets = head_targets(batch, config)
    assert torch.equal(targets["action"][0, :-1, 0], torch.arange(1, 6).float())


def test_twohot_is_exact_between_centres(config):
    centers = torch.linspace(-2.0, 2.0, 9)
    values = torch.tensor([[-2.0, 0.0, 0.5, 2.0]])
    weights = twohot(values, centers)
    assert torch.allclose(weights.sum(-1), torch.ones_like(values))
    assert torch.allclose((weights * centers).sum(-1), values, atol=1e-6)
