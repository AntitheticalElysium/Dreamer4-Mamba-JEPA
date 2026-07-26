"""Stage-0 resolution-parity oracle: prove 63->64 loses no information vs 144.

The 63x63 agent render and the 144x144 dataset render show the SAME 9x9 tiles at
different pixel densities. This module renders an identical set of Craftax states
at both resolutions and asks, per privileged target, whether a linear pixel
probe recovers the target from the padded 64x64 frame NON-INFERIORLY to the
144x144 frame -- with a paired, episode-level bootstrap on the R^2 difference and
a preregistered margin. Run BEFORE any world-model training.

Linear probe = "is the information similarly easy to extract". Raw pixels are
used (no pooling, which would erase the small HUD/inventory cells the labels
live in); the economy-SVD ridge in ``oracle_metrics`` handles p >> n.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .oracle_metrics import (
    episode_bootstrap_ci,
    r2_per_target,
    select_and_predict,
)


@dataclass
class MultiResProbe:
    frames_64: np.ndarray    # uint8 [N,3,64,64]
    frames_144: np.ndarray   # uint8 [N,3,144,144]
    targets: dict            # name -> float32 [N, k]
    episode_id: np.ndarray   # int [N]


def collect_multires_probe(
    *, seeds: list[int], action_fn_factory, max_steps: int
) -> MultiResProbe:
    """Roll episodes and render each visited state at 64 (padded 63) and 144."""
    from .craftax_env import CraftaxPixelEnv

    f64, f144, vitals, inventory, episode_id = [], [], [], [], []
    for ep_idx, seed in enumerate(seeds):
        env = CraftaxPixelEnv(seed=int(seed), target_size=64)
        obs = env.reset()
        action_fn = action_fn_factory(int(seed))

        def record(frame64):
            labels = env.privileged()
            f64.append(frame64)
            f144.append(env.high_res())
            vitals.append(labels["vitals"])
            inventory.append(labels["inventory"])
            episode_id.append(ep_idx)

        record(obs)
        for t in range(int(max_steps)):
            result = env.step(int(action_fn(obs, t)))
            obs = result.obs
            record(obs)
            if result.done:
                break
    return MultiResProbe(
        frames_64=np.stack(f64).astype(np.uint8),
        frames_144=np.stack(f144).astype(np.uint8),
        targets={
            "vitals": np.stack(vitals).astype(np.float32),
            "inventory": np.stack(inventory).astype(np.float32),
        },
        episode_id=np.asarray(episode_id, dtype=np.int64),
    )


def _flatten01(frames: np.ndarray) -> np.ndarray:
    return frames.reshape(frames.shape[0], -1).astype(np.float32) / 255.0


def _three_way_episode_split(episode_id, *, seed, val_frac=0.2, test_frac=0.3):
    ids = np.unique(episode_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    n_val = max(1, int(round(val_frac * len(ids))))
    n_test = max(1, int(round(test_frac * len(ids))))
    test_ids = set(int(i) for i in ids[:n_test])
    val_ids = set(int(i) for i in ids[n_test:n_test + n_val])
    train = np.array([int(e) not in test_ids and int(e) not in val_ids for e in episode_id])
    val = np.array([int(e) in val_ids for e in episode_id])
    test = np.array([int(e) in test_ids for e in episode_id])
    return train, val, test


def _probe_predictions(features, targets, train, val, test):
    """Nested-lambda ridge: select on val and predict test (one cached SVD)."""
    return select_and_predict(
        features[train], targets[train], features[val], targets[val], features[test]
    )


def paired_r2_difference(
    y_true, pred_a, pred_b, episode_id, *, seed, target_names, margin
):
    """Per-target R^2(a) - R^2(b) with a paired episode bootstrap CI.

    Non-inferior when the CI lower bound of (a - b) exceeds ``-margin``.
    """
    n_targets = y_true.shape[1]
    # per-sample index bootstrap handled through a stacked value array: we
    # resample episodes and recompute R^2 for both predictors on the SAME
    # resampled rows (paired).
    results = {}
    err_a = (y_true - pred_a)
    err_b = (y_true - pred_b)
    for k in range(n_targets):
        yk = y_true[:, k:k + 1]
        pak = pred_a[:, k:k + 1]
        pbk = pred_b[:, k:k + 1]

        def diff_stat(idx_values):
            # idx_values holds row indices for a bootstrap resample.
            idx = idx_values.astype(int)
            r_a = r2_per_target(yk[idx], pak[idx])[0]
            r_b = r2_per_target(yk[idx], pbk[idx])[0]
            return float(r_a - r_b)

        row_index = np.arange(y_true.shape[0], dtype=np.float64)
        point, ci = episode_bootstrap_ci(
            row_index, episode_id, diff_stat, seed=seed + k, draws=1000
        )
        r_a_full = float(r2_per_target(yk, pak)[0])
        r_b_full = float(r2_per_target(yk, pbk)[0])
        results[target_names[k]] = {
            "r2_64": r_a_full,
            "r2_144": r_b_full,
            "diff_64_minus_144": r_a_full - r_b_full,
            "diff_ci": ci,
            "non_inferior": bool(ci[0] > -margin),
        }
    return results


TARGET_NAMES = {
    "vitals": ["health", "food", "drink", "energy"],
    "inventory": [
        "wood", "stone", "coal", "iron", "diamond", "sapling",
        "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
        "wood_sword", "stone_sword", "iron_sword",
    ],
}


def resolution_parity(
    probe: MultiResProbe, *, split_seed: int = 20260726, margin: float = 0.05
) -> dict:
    """Full Stage-0 verdict: is 64 non-inferior to 144, per target?"""
    train, val, test = _three_way_episode_split(probe.episode_id, seed=split_seed)
    x64, x144 = _flatten01(probe.frames_64), _flatten01(probe.frames_144)
    report = {"margin": margin, "groups": {}, "all_non_inferior": True}
    for group, targets in probe.targets.items():
        pred64, lam64 = _probe_predictions(x64, targets, train, val, test)
        pred144, lam144 = _probe_predictions(x144, targets, train, val, test)
        per_target = paired_r2_difference(
            targets[test], pred64, pred144, probe.episode_id[test],
            seed=split_seed, target_names=TARGET_NAMES[group], margin=margin,
        )
        report["groups"][group] = {
            "lambda_64": lam64, "lambda_144": lam144, "targets": per_target,
        }
        for t in per_target.values():
            if not t["non_inferior"]:
                report["all_non_inferior"] = False
    return report


__all__ = [
    "MultiResProbe",
    "collect_multires_probe",
    "paired_r2_difference",
    "resolution_parity",
    "TARGET_NAMES",
]
