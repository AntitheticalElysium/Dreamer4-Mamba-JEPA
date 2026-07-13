"""Contracts for the corrected open-loop instrument (amendment S3-v2)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "verification"))

from openloop_v2 import (  # noqa: E402
    HORIZON,
    PREFIX,
    cluster_bootstrap,
    paired_difference_gate,
    window_manifest,
)


def fake_episodes(lengths):
    return [
        {"obs": np.zeros((n + 1, 3, 8, 8), dtype=np.uint8),
         "actions": np.zeros(n, dtype=np.int64)}
        for n in lengths
    ]


def test_manifest_windows_are_deterministic_nonoverlapping_and_tagged():
    episodes = fake_episodes([120, 60, 30])
    manifest = window_manifest(episodes)
    assert manifest == window_manifest(episodes), "manifest must be deterministic"
    span = PREFIX + HORIZON
    seen = set()
    for w in manifest:
        key = (w["episode"], w["start"])
        assert key not in seen, "duplicate window"
        seen.add(key)
        assert w["start"] >= 1
    by_episode = {}
    for w in manifest:
        by_episode.setdefault(w["episode"], []).append(w["start"])
    for starts in by_episode.values():
        starts.sort()
        for a, b in zip(starts, starts[1:]):
            assert b - a >= span, "overlapping windows within an episode"
    # short episode (30 < span+2) contributes at most one window
    assert len(by_episode.get(2, [])) <= 1


def test_cluster_bootstrap_wider_than_iid_under_episode_correlation():
    rng = np.random.default_rng(0)
    episodes = np.repeat(np.arange(10), 6)
    episode_effect = np.repeat(rng.normal(0, 1.0, 10), 6)
    values = episode_effect + rng.normal(0, 0.1, 60)

    cluster = cluster_bootstrap(values, episodes, seed=1)
    iid_rng = np.random.default_rng(1)
    iid = np.array([
        values[iid_rng.integers(0, len(values), size=len(values))].mean()
        for _ in range(2000)
    ])
    iid_width = float(np.quantile(iid, 0.975) - np.quantile(iid, 0.025))
    cluster_width = cluster[1] - cluster[0]
    assert cluster_width > 1.5 * iid_width, (
        f"cluster CI ({cluster_width:.3f}) should be materially wider than "
        f"iid CI ({iid_width:.3f}) under episode-level correlation"
    )


def test_paired_difference_gate_uses_shared_valid_windows():
    manifest = [{"episode": e, "start": 1 + 24 * i} for e in range(4) for i in range(3)]
    n = len(manifest)
    margins_roll = [0.10] * n
    margins_base = [0.05] * n
    valid = [True] * n
    valid[0] = False  # invalid in one arm must drop the pair
    per_k_roll = [{"k": 8, "window_margins": margins_roll, "window_valid": valid}]
    per_k_base = [{"k": 8, "window_margins": margins_base, "window_valid": [True] * n}]
    out = paired_difference_gate(per_k_roll, per_k_base, manifest, k=8)
    assert out["paired_diff_mean"] == pytest.approx(0.05)
    assert out["pass"] is True


def test_loo_oracle_matches_brute_force():
    import torch
    from fork_oracle_v2 import loo_oracle_error

    torch.manual_seed(3)
    branch = torch.randn(6, 5, 8)
    fast = loo_oracle_error(branch)          # [B, S] per-branch
    unit = torch.nn.functional.normalize(branch.float(), dim=-1)
    slow = []
    for b in range(6):
        others = torch.cat([unit[:b], unit[b + 1:]])
        p = torch.nn.functional.normalize(others.mean(0), dim=-1)
        slow.append(1.0 - (p * unit[b]).sum(-1))
    slow = torch.stack(slow)                 # [B, S]
    assert fast.shape == (6, 5), "must stay per-branch until masks are applied"
    assert torch.allclose(fast, slow, atol=1e-5)
    # branch-specific masking control: masking branch 0's row must only use
    # branch 0's error (2026-07-13 correction)
    mask = torch.zeros(5, dtype=torch.bool); mask[2] = True
    assert torch.allclose(fast[0][mask], slow[0][mask])


def test_shift_copy_geometry():
    import numpy as np
    from fork_oracle_v2 import shift_copy_frame

    frame = np.zeros((3, 64, 64), dtype=np.uint8)
    frame[:, 21:28, 21:28] = 200          # one bright world tile
    frame[:, 50:, :] = 77                 # HUD must never move
    right = shift_copy_frame(frame, 2)    # player moves right -> content shifts left
    assert right[:, 21:28, 14:21].max() == 200 and right[:, 21:28, 21:28].max() == 0
    down = shift_copy_frame(frame, 4)     # player moves down -> content shifts up
    assert down[:, 14:21, 21:28].max() == 200
    noop = shift_copy_frame(frame, 0)
    assert np.array_equal(noop, frame)
    for moved in (right, down):
        assert np.array_equal(moved[:, 50:, :], frame[:, 50:, :]), "HUD moved"


def test_all_pairs_divergence_counts_pairs_not_reference():
    from fork_oracle_v2 import all_pairs_divergence

    outcomes = [{"x": 0}, {"x": 0}, {"x": 1}]
    # pairs: (0,1) same, (0,2) diff, (1,2) diff -> 2/3
    assert abs(all_pairs_divergence(outcomes, "x") - 2 / 3) < 1e-9
