"""Executed-evaluation score helpers.

The original danijar/crafter random-shooting planner harness that lived here was
removed in the Craftax migration: it depended on the danijar CrafterAdapter and
is superseded by the native Craftax achievement evaluation. Only the pure,
dependency-free scoring/serialization helpers remain, so this module imports
neither torch, JAX, nor any environment.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _self_sha256(path: Path | None = None) -> str:
    target = Path(__file__) if path is None else Path(path)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _crafter_score(episodes: list[dict]) -> tuple[float, dict[str, float]]:
    """Official Crafter/Craftax geometric-mean achievement score.

    ``score = exp(mean_k log(1 + success_rate_k)) - 1`` with success rates in
    percent. This is byte-equivalent to ``danijar/crafter``
    ``analysis/common.py:compute_scores`` and to Craftax-Classic's own
    ``compute_score`` formula, so it is comparable across both implementations.
    """
    names = sorted(
        {name for episode in episodes for name in episode["achievements"]}
    )
    success_rates = {
        name: 100.0
        * sum(episode["achievements"].get(name, 0) > 0 for episode in episodes)
        / len(episodes)
        for name in names
    }
    score = (
        math.exp(
            sum(math.log1p(rate) for rate in success_rates.values())
            / len(success_rates)
        )
        - 1.0
        if success_rates
        else 0.0
    )
    return float(score), success_rates


__all__ = ["_atomic_json", "_self_sha256", "_pearson", "_crafter_score"]
