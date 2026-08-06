import pytest
import torch

from d4mj.agent import Heads, head_loss, head_targets, terminal_loss, twohot

from .conftest import latent_batch


def readout_for(config, batch):
    """The reward and value heads ship zero-initialised, which makes their output
    uniform -- and a uniform distribution has cross-entropy log(bins) against *any*
    target. These tests are about which rows a loss reads, so they need a head whose
    predictions actually vary."""
    torch.manual_seed(0)
    heads = Heads(config)
    generator = torch.Generator().manual_seed(7)
    for head in (heads.reward, heads.value):
        head.weight.data.normal_(0.0, 0.02, generator=generator)
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


def test_continuation_is_one_step(config):
    batch = latent_batch(config, 2, 6, relevant=[True, False])
    batch.terminated[0, 4] = True
    heads, predictions = readout_for(config, batch)
    targets = head_targets(batch, config)
    assert predictions["continuation"].shape == (2, 6, 1)
    assert targets["continuation"].shape == (2, 6, 1)
    assert targets["continuation"][0, 4, 0] == 0


def test_head_output_scales_match_the_pinned_config(config):
    """DreamerV3 ships `rewhead` and `value` at outscale 0.0, `policy` at 0.01 and
    `conhead` at 1.0. A value head starting at random emits random advantages on
    Phase 3's first steps, and PMPO reads only their sign."""
    torch.manual_seed(0)
    heads = Heads(config)
    for head in (heads.reward, heads.value):
        assert float(head.weight.detach().abs().max()) == 0.0
        assert float(head.bias.detach().abs().max()) == 0.0
    policy = float(heads.policy.weight.detach().abs().max())
    continuation = float(heads.continuation.weight.detach().abs().max())
    assert 0.0 < policy < continuation, "policy is scaled down, continuation is not"


def test_zero_initialised_heads_predict_a_flat_distribution(config):
    """The consequence worth stating: a uniform reward head has cross-entropy
    log(bins) against every target, so it starts with no preference at all."""
    import math

    torch.manual_seed(0)
    heads = Heads(config)
    agent = torch.randn(2, 3, config.n_agent, config.d_model, generator=torch.Generator().manual_seed(1))
    logits = heads(agent)["reward"]
    assert torch.allclose(logits, torch.zeros_like(logits))
    entropy = -torch.log_softmax(logits, -1).mean()
    assert abs(float(entropy.detach()) - math.log(config.bins)) < 1e-4


def test_terminal_loss_is_bce_over_the_stratified_tail(config):
    batch = latent_batch(config, 2, 8, relevant=[False, False], support=[True, True])
    batch.terminated[:, -1] = True
    heads, predictions = readout_for(config, batch)
    targets = head_targets(batch, config)
    reference = terminal_loss(predictions, targets)

    valid = targets["continuation_valid"].bool()
    alive = targets["continuation"].bool() & valid
    changed_alive = dict(predictions)
    changed_alive["continuation"] = predictions["continuation"].clone()
    changed_alive["continuation"][alive] += 20.0
    assert not torch.equal(reference, terminal_loss(changed_alive, targets))

    changed_terminal = dict(predictions)
    changed_terminal["continuation"] = predictions["continuation"].clone()
    changed_terminal["continuation"][~alive & valid] -= 20.0
    assert terminal_loss(changed_terminal, targets) < reference


def test_terminal_stratum_has_bounded_positive_mass(config):
    fractions = []
    for step in range(64):
        finetune = step >= 64 * (1 - config.long_only_fraction)
        long = finetune or (step + 1) % config.long_batch_every == 0
        length = config.sequence_long if long else config.sequence
        batch = latent_batch(config, 1, length, relevant=[False], support=[True])
        batch.terminated[:, -1] = True
        targets = head_targets(batch, config)
        valid = targets["continuation_valid"]
        fractions.append(float(((1 - targets["continuation"]) * valid).sum() / valid.sum()))
    mass = config.terminal_loss_mass * sum(fractions) / len(fractions)
    assert 0.008 < mass < 0.009


def test_twohot_is_exact_between_centres(config):
    centers = torch.linspace(-2.0, 2.0, 9)
    values = torch.tensor([[-2.0, 0.0, 0.5, 2.0]])
    weights = twohot(values, centers)
    assert torch.allclose(weights.sum(-1), torch.ones_like(values))
    assert torch.allclose((weights * centers).sum(-1), values, atol=1e-6)
