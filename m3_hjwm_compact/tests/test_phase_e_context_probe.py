"""Selection and label contracts for the frozen-context Phase-E probe."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))

from phase_e_context_probe import (  # noqa: E402
    balanced_task_indices,
    binary_targets,
    task_labels,
)


def test_task_probe_labels_use_the_target_transition_only():
    episode = {
        "rewards": np.asarray([0.0, -0.2, 1.0], dtype=np.float32),
        "continues": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
    }
    rows = [
        {"episode": 0, "transition": 0},
        {"episode": 0, "transition": 1},
        {"episode": 0, "transition": 2},
    ]
    labels = task_labels([episode], rows)
    np.testing.assert_array_equal(labels["reward_event"],
                                  [False, True, True])
    np.testing.assert_array_equal(labels["reward_positive"],
                                  [False, False, True])
    np.testing.assert_array_equal(labels["reward_negative"],
                                  [False, True, False])
    np.testing.assert_array_equal(labels["terminal"],
                                  [False, False, True])


def test_probe_selection_is_balanced_and_binary_targets_match():
    labels = {
        "reward_event": np.asarray([1, 1, 0, 0, 0, 0], dtype=bool),
        "reward_positive": np.asarray([1, 0, 0, 0, 0, 0], dtype=bool),
        "reward_negative": np.asarray([0, 1, 1, 0, 0, 0], dtype=bool),
        "terminal": np.asarray([0, 0, 0, 1, 0, 0], dtype=bool),
    }
    selected = balanced_task_indices(labels, np.random.default_rng(3))
    for task, index in selected.items():
        target = binary_targets(task, labels, index)
        assert len(target) % 2 == 0
        assert int(target.sum()) == len(target) // 2
