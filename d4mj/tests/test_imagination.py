from dataclasses import replace

import pytest
import torch

from d4mj.agent import Heads
from d4mj.imagination import imagine
from d4mj.state import WorldState
from d4mj.transition import World, commit_inputs

ARMS = ("flow", "direct")


def rollout(config, transition, policy_seed=5, model_seed=7):
    arm = replace(config, transition=transition)
    torch.manual_seed(0)
    world, heads = World(arm).eval(), Heads(arm).eval()
    rng = torch.Generator().manual_seed(model_seed)

    latents = torch.randn(2, 4, arm.n_spatial, arm.d_spatial,
                          generator=torch.Generator().manual_seed(1)).tanh()
    actions = torch.zeros(2, 4, dtype=torch.long)
    with torch.no_grad():
        committed, conditioning = commit_inputs(latents, rng, arm)
        features, agent, memory = world(None, actions, committed, conditioning)
        state = WorldState(latents[:, -1:], memory, 4, features[:, -1:])
        return imagine(world, heads, state, agent[:, -1:], rng,
                       torch.Generator().manual_seed(policy_seed), arm), arm


@pytest.mark.parametrize("transition", ARMS)
def test_shapes_follow_the_horizon(config, transition):
    """Values and readouts span horizon + 1 states; actions and rewards span the
    horizon transitions between them."""
    trajectory, arm = rollout(config, transition)
    assert trajectory.action.shape == (2, arm.horizon)
    assert trajectory.reward.shape == (2, arm.horizon)
    assert trajectory.continuation.shape == (2, arm.horizon)
    assert trajectory.value.shape == (2, arm.horizon + 1)
    assert trajectory.agent.shape[:2] == (2, arm.horizon + 1)


@pytest.mark.parametrize("transition", ARMS)
def test_continuation_is_a_probability(config, transition):
    trajectory, _ = rollout(config, transition)
    assert ((trajectory.continuation >= 0) & (trajectory.continuation <= 1)).all()


@pytest.mark.parametrize("transition", ARMS)
def test_everything_is_finite(config, transition):
    trajectory, _ = rollout(config, transition)
    for name in ("logits", "reward", "continuation", "value"):
        assert torch.isfinite(getattr(trajectory, name)).all(), name


def test_policy_stream_is_independent_of_the_world_stream(config):
    """Direct draws no world noise at all, so moving the model seed must leave its
    action sequence untouched. If the two shared one stream, world draws would
    advance the policy stream and the same policy seed would give different
    actions -- which is what desynchronises the arms in a paired comparison."""
    one, _ = rollout(config, "direct", policy_seed=5, model_seed=7)
    other, _ = rollout(config, "direct", policy_seed=5, model_seed=99)
    assert torch.equal(one.action, other.action)


def test_flow_world_noise_moves_its_own_rollout(config):
    """The contrast: flow does consume model noise, so its trajectory depends on
    the model seed even with the policy seed fixed.

    Compared on the agent readout, not on reward: the reward head ships
    zero-initialised, so at initialisation it emits the same value everywhere and
    would hide any difference in the states underneath it."""
    one, _ = rollout(config, "flow", policy_seed=5, model_seed=7)
    other, _ = rollout(config, "flow", policy_seed=5, model_seed=99)
    assert not torch.equal(one.agent, other.agent)


def test_changing_the_policy_seed_changes_the_actions(config):
    one, _ = rollout(config, "direct", policy_seed=5)
    other, _ = rollout(config, "direct", policy_seed=6)
    assert not torch.equal(one.action, other.action)


def test_horizon_is_a_smoke_default_not_a_settled_value(config):
    """S54: the final horizon is selected on DEV from the candidate set using the
    full-context multistep diagnostic, never blessed by being the default."""
    assert config.horizon in config.horizon_candidates
