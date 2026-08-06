from dataclasses import replace

import pytest
import torch

from d4mj import backbone, gates

ARMS = ("flow", "direct")


@pytest.mark.parametrize("transition", ARMS)
def test_recurrent_carry_catches_inert_memory(config, transition):
    """This passed for the wrong reason before: `initial` and `advance` commit at
    different points in the noise stream, so for the flow arm that draw alone moved
    the output and the gate passed with memory entirely ignored."""
    arm = replace(config, transition=transition)
    original = backbone.Backbone.forward

    def inert(self, x, memory=None, offset=0):
        return original(self, x, None, 0)

    try:
        gates.recurrent_carry(arm)
        backbone.Backbone.forward = inert
        with pytest.raises(AssertionError):
            gates.recurrent_carry(arm)
    finally:
        backbone.Backbone.forward = original


@pytest.mark.parametrize("transition", ARMS)
def test_reset_parity_catches_state_crossing_a_boundary(config, transition):
    """The complementary claim, and the reason the two gates are not duplicates:
    `recurrent_carry` shows memory matters, this shows a reset removes it. Inert
    memory passes here -- correctly -- so the mutation is a *leaking* reset that
    threads the previous episode's state into the new one."""
    arm = replace(config, transition=transition)
    original = gates.initial
    seen: list = []

    def leaky(world, latent, action, rng, config):
        state, agent = original(world, latent, action, rng, config)
        seen.append(state.memory)
        if len(seen) == 2:
            state = type(state)(state.latent, seen[0], state.step, state.features)
        return state, agent

    try:
        gates.reset_parity(arm)
        gates.initial = leaky
        with pytest.raises(AssertionError):
            gates.reset_parity(arm)
    finally:
        gates.initial = original
        seen.clear()


@pytest.mark.parametrize("transition", ARMS)
def test_alignment_holds(config, transition):
    gates.alignment(replace(config, transition=transition))


@pytest.mark.parametrize("transition", ARMS)
def test_firewall_holds(config, transition):
    gates.firewall(replace(config, transition=transition))


def test_alignment_catches_a_shifted_action(config, monkeypatch):
    """The gate must fail when the led-to convention is violated, not merely pass
    when it holds."""
    from d4mj import data

    original = data._window

    def shifted(episode, start, length, config):
        row = original(episode, start, length, config)
        row["led_to_action"] = row["led_to_action"].roll(1)
        return row

    monkeypatch.setattr(data, "_window", shifted)
    with pytest.raises(AssertionError):
        gates.alignment(config)
