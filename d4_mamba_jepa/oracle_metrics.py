"""Rigorous, self-contained probe metrics for the oracle (pure numpy).

Everything the representation/dynamics oracles report is built on these
primitives, so they are unit-tested against exact, hand-checkable cases:

- ``auroc`` uses TIE-AWARE average ranks (a constant or quantized predictor is
  scored 0.5, not spuriously high/low).
- ``average_precision`` and ``brier`` for rare binary targets, where the
  negative class dominates and AUROC alone is misleading.
- ``ridge_predict`` uses an economy SVD, so it is correct and stable when the
  feature dimension exceeds the sample count (raw-pixel ceilings have p >> n).
- ``select_ridge_lambda`` does nested, held-out ridge selection so different
  feature sources (latent vs pixels) are capacity-matched, not fixed at one
  arbitrary lambda.
- ``episode_bootstrap_ci`` resamples whole EPISODES (frames within an episode
  are correlated and are not independent samples).
"""
from __future__ import annotations

from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Ranking / binary metrics.
# ---------------------------------------------------------------------------
def average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks in ``[1, n]`` with ties assigned their average rank (scipy-style)."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank of the tie block
        ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware AUROC via the Mann-Whitney U statistic. NaN if degenerate."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(scores)
    auc = (ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (step definition). NaN if no pos."""
    labels = np.asarray(labels).astype(bool)
    scores = np.asarray(scores, dtype=np.float64)
    n_pos = int(labels.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")  # descending, stable
    y = labels[order]
    tp = np.cumsum(y)
    precision = tp / (np.arange(len(y)) + 1.0)
    recall = tp / n_pos
    ap = 0.0
    prev_recall = 0.0
    for p, r in zip(precision, recall):
        ap += (r - prev_recall) * p
        prev_recall = r
    return float(ap)


def brier(prob: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared error between predicted probability and the binary label."""
    prob = np.clip(np.asarray(prob, dtype=np.float64), 0.0, 1.0)
    labels = np.asarray(labels).astype(np.float64)
    return float(np.mean((prob - labels) ** 2))


# ---------------------------------------------------------------------------
# Regression metrics.
# ---------------------------------------------------------------------------
def r2_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-target R^2; a target with zero variance yields R^2 = 0."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    ss_tot = (centered ** 2).sum(axis=0)
    return np.where(ss_tot < 1e-12, 0.0, 1.0 - ss_res / np.maximum(ss_tot, 1e-12))


def mean_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(r2_per_target(y_true, y_pred)))


# ---------------------------------------------------------------------------
# Ridge probe (economy SVD -> correct for p >> n).
# ---------------------------------------------------------------------------
def _standardize(train: np.ndarray, *others):
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return tuple((a - mean) / std for a in (train, *others))


class RidgeProbe:
    """Economy-SVD ridge that decomposes the (standardized) train features ONCE.

    The SVD does not depend on the ridge strength or the targets, so caching it
    makes lambda selection and the final refit-predict essentially free instead
    of re-decomposing a p >> n feature matrix per lambda.
    """

    def __init__(self, x_train: np.ndarray):
        self.mean = x_train.mean(axis=0, keepdims=True)
        std = x_train.std(axis=0, keepdims=True)
        self.std = np.where(std < 1e-8, 1.0, std)
        xtr = (x_train - self.mean) / self.std
        self.u, self.s, self.vt = np.linalg.svd(xtr, full_matrices=False)

    def predict(self, x_test: np.ndarray, y_train: np.ndarray, lam: float) -> np.ndarray:
        if y_train.ndim == 1:
            y_train = y_train[:, None]
        y_mean = y_train.mean(axis=0, keepdims=True)
        d = self.s / (self.s ** 2 + lam)
        w = (self.vt.T * d) @ (self.u.T @ (y_train - y_mean))   # [p, k]
        xte = (x_test - self.mean) / self.std
        return xte @ w + y_mean


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Ridge via economy SVD; correct for any (n, p) including p >> n."""
    return RidgeProbe(x_train).predict(x_test, y_train, lam)


def select_ridge_lambda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    lambdas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3),
) -> float:
    """Pick lambda maximizing held-out mean R^2 (one cached SVD)."""
    probe = RidgeProbe(x_train)
    y_val2 = y_val if y_val.ndim > 1 else y_val[:, None]
    best_lam, best_score = float(lambdas[0]), -np.inf
    for lam in lambdas:
        score = mean_r2(y_val2, probe.predict(x_val, y_train, float(lam)))
        if score > best_score:
            best_score, best_lam = score, float(lam)
    return best_lam


def select_and_predict(
    x_train, y_train, x_val, y_val, x_test,
    lambdas=(1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3),
):
    """Select lambda on val and predict test, decomposing train ONCE."""
    probe = RidgeProbe(x_train)
    y_val2 = y_val if y_val.ndim > 1 else y_val[:, None]
    best_lam, best_score = float(lambdas[0]), -np.inf
    for lam in lambdas:
        score = mean_r2(y_val2, probe.predict(x_val, y_train, float(lam)))
        if score > best_score:
            best_score, best_lam = score, float(lam)
    return probe.predict(x_test, y_train, best_lam), best_lam


# ---------------------------------------------------------------------------
# Episode-level bootstrap.
# ---------------------------------------------------------------------------
def episode_bootstrap_ci(
    values: np.ndarray,
    episode_id: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    *,
    seed: int,
    draws: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, list[float]]:
    """Bootstrap a per-sample statistic by resampling whole EPISODES.

    ``statistic`` maps a 1-D array of per-sample values to a scalar. Frames from
    one episode are correlated, so episodes -- not frames -- are the unit.
    """
    values = np.asarray(values, dtype=np.float64)
    episode_id = np.asarray(episode_id)
    unique = np.unique(episode_id)
    by_ep = {int(e): values[episode_id == e] for e in unique}
    rng = np.random.default_rng(seed)
    point = float(statistic(values))
    boots = np.empty(draws, dtype=np.float64)
    n_ep = len(unique)
    for b in range(draws):
        pick = rng.integers(0, n_ep, size=n_ep)
        sample = np.concatenate([by_ep[int(unique[i])] for i in pick])
        boots[b] = statistic(sample)
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return point, [lo, hi]


__all__ = [
    "average_ranks",
    "auroc",
    "average_precision",
    "brier",
    "r2_per_target",
    "mean_r2",
    "RidgeProbe",
    "ridge_predict",
    "select_ridge_lambda",
    "select_and_predict",
    "episode_bootstrap_ci",
]
