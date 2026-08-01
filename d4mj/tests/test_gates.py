from dataclasses import replace

import pytest
import torch

from d4mj import backbone, gates

ARMS = ("flow", "direct")
HISTORY = ("reset_parity", "recurrent_carry")


@pytest.mark.parametrize("transition", ARMS)
@pytest.mark.parametrize("name", HISTORY)
def test_history_gates_catch_inert_memory(config, transition, name):
    """These passed for the wrong reason before: `initial` and `advance` commit at
    different points in the noise stream, so for the flow arm that draw alone moved
    the output and the gate passed with memory entirely ignored."""
    arm = replace(config, transition=transition)
    original = backbone.Backbone.forward

    def inert(self, x, memory=None, offset=0):
        return original(self, x, None, 0)

    try:
        getattr(gates, name)(arm)
        backbone.Backbone.forward = inert
        with pytest.raises(AssertionError):
            getattr(gates, name)(arm)
    finally:
        backbone.Backbone.forward = original


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
