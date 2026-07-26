"""Tests for the Stage-0 resolution-parity logic (no craftax/JAX)."""
from __future__ import annotations

import numpy as np

from d4_mamba_jepa.craftax_resolution import (
    _three_way_episode_split,
    paired_r2_difference,
)


def _episodes(n_ep=12, per=20):
    return np.repeat(np.arange(n_ep), per)


def test_three_way_split_is_episode_disjoint():
    ep = _episodes()
    train, val, test = _three_way_episode_split(ep, seed=0)
    assert not (train & val).any() and not (train & test).any() and not (val & test).any()
    assert (train | val | test).all()
    for a, b in [(train, val), (train, test), (val, test)]:
        assert set(ep[a]).isdisjoint(set(ep[b]))


def test_paired_difference_zero_when_predictions_identical():
    rng = np.random.default_rng(0)
    ep = _episodes()
    y = rng.normal(size=(ep.shape[0], 3))
    pred = y + 0.1 * rng.normal(size=y.shape)
    res = paired_r2_difference(
        y, pred, pred.copy(), ep, seed=0,
        target_names=["a", "b", "c"], margin=0.05,
    )
    for t in res.values():
        assert abs(t["diff_64_minus_144"]) < 1e-9
        assert t["non_inferior"] is True


def test_paired_difference_flags_inferior_predictor():
    rng = np.random.default_rng(1)
    ep = _episodes()
    y = rng.normal(size=(ep.shape[0], 2))
    good = y.copy()                          # perfect
    bad = np.zeros_like(y) + y.mean(0)       # constant -> R^2 ~ 0
    # a=bad (64), b=good (144): 64 is far inferior -> must be flagged.
    res = paired_r2_difference(
        y, bad, good, ep, seed=0, target_names=["a", "b"], margin=0.05
    )
    for t in res.values():
        assert t["diff_64_minus_144"] < -0.5
        assert t["non_inferior"] is False
    # a=good (64), b=bad (144): 64 superior -> non-inferior holds.
    res2 = paired_r2_difference(
        y, good, bad, ep, seed=0, target_names=["a", "b"], margin=0.05
    )
    for t in res2.values():
        assert t["non_inferior"] is True
