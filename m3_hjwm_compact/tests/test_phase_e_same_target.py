"""Indexing contracts for the same-target Phase-E depth diagnostic."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))

from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import suffix_partition, target_rows, window_arrays  # noqa: E402


def _episode(transitions: int = 32):
    # Pixel value, action, and reward expose every index directly.
    obs = np.arange(transitions + 1, dtype=np.uint8)[:, None, None, None]
    return {
        "obs": obs,
        "actions": np.arange(transitions, dtype=np.int64),
        "rewards": np.arange(transitions, dtype=np.float32),
        "continues": np.ones(transitions, dtype=np.float32),
    }


def test_common_target_window_and_previous_action_alignment():
    episodes = [_episode()]
    rows = target_rows(episodes)
    assert rows[0] == {"episode": 0, "transition": 14}
    arrays = window_arrays(episodes, [rows[0], rows[6]])

    np.testing.assert_array_equal(arrays["obs"][0, :, 0, 0, 0],
                                  np.arange(16))
    np.testing.assert_array_equal(arrays["actions"][0], np.arange(15))
    np.testing.assert_array_equal(arrays["previous_actions"][0],
                                  np.r_[-1, np.arange(15)])
    assert arrays["rewards"][0] == 14

    # Target j=20 starts at observation 6, so previous action 5 initializes
    # the first real state and action 20 causes the target reward.
    np.testing.assert_array_equal(arrays["obs"][1, :, 0, 0, 0],
                                  np.arange(6, 22))
    np.testing.assert_array_equal(arrays["actions"][1], np.arange(6, 21))
    np.testing.assert_array_equal(arrays["previous_actions"][1],
                                  np.arange(5, 21))
    assert arrays["rewards"][1] == 20


def test_suffix_replacement_keeps_total_updates_and_target_action_fixed():
    expected = {
        0: (list(range(8, 16)), []),
        1: (list(range(8, 15)), [14]),
        2: (list(range(8, 14)), [13, 14]),
        4: (list(range(8, 12)), [11, 12, 13, 14]),
        8: ([], list(range(7, 15))),
    }
    for depth, wanted in expected.items():
        real, imagined = suffix_partition(depth)
        assert (list(real), list(imagined)) == wanted
        assert 8 + len(real) + len(imagined) == 16
        if depth:
            assert list(imagined)[-1] == 14


def test_continuation_supplement_uses_the_identical_target_transition():
    episode = _episode()
    episode["continues"] = np.arange(
        len(episode["rewards"]), dtype=np.float32)
    rows = target_rows([episode])
    targets = continuation_targets([episode], [rows[0], rows[6]])
    np.testing.assert_array_equal(targets, np.asarray([14, 20]))
