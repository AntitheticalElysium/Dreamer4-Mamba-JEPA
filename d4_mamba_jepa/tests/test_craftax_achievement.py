"""Tests for the executed-achievement paired scoring (no craftax/JAX)."""
from __future__ import annotations

from d4_mamba_jepa.craftax_achievement import _paired_score_ci


def _rows(seeds, unlocked):
    ach = {"collect_wood": int(unlocked), "collect_stone": int(unlocked)}
    return {s: {"achievements": dict(ach)} for s in seeds}


def test_paired_score_ci_detects_advantage():
    seeds = list(range(10))
    winner = _rows(seeds, True)     # unlocks both every episode
    loser = _rows(seeds, False)     # unlocks nothing
    point, ci = _paired_score_ci(winner, loser, seeds, seed=0, draws=500)
    assert point > 0.0
    assert ci[0] > 0.0              # winner strictly better, CI excludes zero


def test_paired_score_ci_identical_is_zero():
    seeds = list(range(8))
    rows = _rows(seeds, True)
    point, ci = _paired_score_ci(rows, rows, seeds, seed=0, draws=200)
    assert point == 0.0
    assert ci == [0.0, 0.0]
