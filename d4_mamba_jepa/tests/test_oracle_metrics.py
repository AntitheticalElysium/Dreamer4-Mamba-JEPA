"""Exact correctness tests for the oracle metric primitives."""
from __future__ import annotations

import math

import numpy as np
import pytest

from d4_mamba_jepa.oracle_metrics import (
    auroc,
    average_precision,
    average_ranks,
    brier,
    episode_bootstrap_ci,
    mean_r2,
    r2_per_target,
    ridge_predict,
    select_ridge_lambda,
)


# --- AUROC: the exact cases the protocol requires -------------------------
def test_auroc_perfect_and_reversed():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert auroc(scores, labels) == 1.0
    assert auroc(-scores, labels) == 0.0


def test_auroc_all_equal_is_half_not_spurious():
    # A constant / fully-tied predictor must score exactly 0.5.
    scores = np.ones(6)
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=bool)
    assert auroc(scores, labels) == 0.5


def test_auroc_degenerate_single_class_is_nan():
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([1, 1], dtype=bool)))
    assert math.isnan(auroc(np.array([0.1, 0.9]), np.array([0, 0], dtype=bool)))


def test_auroc_one_pos_one_neg():
    assert auroc(np.array([0.2, 0.8]), np.array([0, 1], dtype=bool)) == 1.0
    assert auroc(np.array([0.8, 0.2]), np.array([0, 1], dtype=bool)) == 0.0


def test_auroc_tie_averaging_matches_probabilistic_definition():
    # pos={1.0,1.0}, neg={1.0,0.0}. P(pos>neg)+0.5 P(tie): one clear win, two
    # ties with the 1.0 negative -> (1 + 0.5*2)/4 = 0.5 ... verify explicitly.
    scores = np.array([1.0, 1.0, 1.0, 0.0])
    labels = np.array([1, 1, 0, 0], dtype=bool)
    # pairs (pos,neg): (1,1)tie .5, (1,0)win 1, (1,1)tie .5, (1,0)win 1 -> 3/4
    assert auroc(scores, labels) == pytest.approx(0.75)


def test_average_ranks_ties():
    assert average_ranks(np.array([1.0, 1.0, 3.0])).tolist() == [1.5, 1.5, 3.0]


# --- Average precision / Brier --------------------------------------------
def test_average_precision_perfect_and_none():
    scores = np.array([0.1, 0.2, 0.9, 0.8])
    labels = np.array([0, 0, 1, 1], dtype=bool)
    assert average_precision(scores, labels) == pytest.approx(1.0)
    assert math.isnan(average_precision(scores, np.zeros(4, dtype=bool)))


def test_average_precision_known_value():
    # ranking: p, n, p  -> precisions at recall steps 0.5 (1/1) and 1.0 (2/3)
    scores = np.array([0.9, 0.8, 0.7])
    labels = np.array([1, 0, 1], dtype=bool)
    assert average_precision(scores, labels) == pytest.approx(0.5 * 1.0 + 0.5 * (2 / 3))


def test_brier_known():
    assert brier(np.array([1.0, 0.0]), np.array([1, 0], dtype=bool)) == 0.0
    assert brier(np.array([0.5, 0.5]), np.array([1, 0], dtype=bool)) == 0.25


# --- R^2 -------------------------------------------------------------------
def test_r2_perfect_and_constant_prediction():
    y = np.array([[1.0], [2.0], [3.0], [4.0]])
    assert mean_r2(y, y) == pytest.approx(1.0)
    const = np.full_like(y, y.mean())
    assert mean_r2(y, const) == pytest.approx(0.0)


def test_r2_zero_variance_target_is_zero():
    y = np.ones((5, 1))
    assert r2_per_target(y, y).tolist() == [0.0]


# --- Ridge: correct for p < n AND p >> n ----------------------------------
def test_ridge_recovers_linear_signal_small_p():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    w = rng.normal(size=(5, 2))
    y = x @ w + 3.0
    pred = ridge_predict(x[:150], y[:150], x[150:], lam=1e-3)
    assert mean_r2(y[150:], pred) > 0.99


def test_ridge_stable_when_features_exceed_samples():
    # p (150) > n_train (100): the economy SVD path must stay finite and, on a
    # clean low-rank signal, recover it held out (beating a constant predictor).
    rng = np.random.default_rng(1)
    x = rng.normal(size=(140, 150))
    w = np.zeros((150, 1)); w[:5, 0] = rng.normal(size=5)
    y = x @ w
    pred = ridge_predict(x[:100], y[:100], x[100:], lam=1e-3)
    assert np.isfinite(pred).all()
    assert mean_r2(y[100:], pred) > 0.5


def test_select_ridge_lambda_prefers_regularization_under_noise():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(120, 50))
    w = np.zeros((50, 1)); w[:3, 0] = rng.normal(size=3)
    y = x @ w + 0.5 * rng.normal(size=(120, 1))
    lam = select_ridge_lambda(x[:60], y[:60], x[60:90], y[60:90])
    assert lam >= 1.0  # noise + many features -> non-trivial regularization


# --- Episode bootstrap -----------------------------------------------------
def test_episode_bootstrap_point_and_interval():
    rng = np.random.default_rng(3)
    episode_id = np.repeat(np.arange(30), 10)
    values = rng.normal(size=300)
    point, (lo, hi) = episode_bootstrap_ci(
        values, episode_id, np.mean, seed=0, draws=500
    )
    assert point == pytest.approx(values.mean())
    assert lo < point < hi


def test_episode_bootstrap_resamples_episodes_not_frames():
    # If a single episode is extreme, resampling episodes must sometimes exclude
    # it, widening the interval well beyond a naive per-frame bootstrap.
    episode_id = np.repeat(np.arange(10), 5)
    values = np.zeros(50)
    values[episode_id == 0] = 100.0  # one extreme episode
    _, (lo, hi) = episode_bootstrap_ci(values, episode_id, np.mean, seed=1, draws=1000)
    assert hi - lo > 5.0
