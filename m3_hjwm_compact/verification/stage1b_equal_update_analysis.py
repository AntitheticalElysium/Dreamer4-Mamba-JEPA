"""Paired analysis for the registered Stage-1b equal-update factorial.

This consumes only the hash-pinned Stage-1 evaluation data and the raw rows
written by stage1b_equal_update_control.py. It does not fit or select a model.
All uncertainty intervals resample environment/episode clusters and preserve
the pairing between arms.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata as scipy_rankdata

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from phase_e_continuation_depth import continuation_targets  # noqa: E402
from stage1_head_adaptation import (  # noqa: E402
    BATCH,
    MANIFEST,
    NATURAL,
    PREFIX,
    TERMINAL,
    UPDATES,
    WINDOW,
    window_index,
)
from stage1b_equal_update_control import (  # noqa: E402
    RAW_PATH,
    REPORT_PATH,
)
from step3_temporal import load_scaled_data  # noqa: E402

ARTIFACTS = REPO_ROOT / "reviews" / "artifacts"
STAGE1_REPORT = ARTIFACTS / "stage1_report.json"
OUTPUT = ARTIFACTS / "stage1b_equal_update_analysis.json"
BOOTSTRAP_DRAWS = 2_000
SEEDS = (505, 606, 707)
FAMILIES = OrderedDict((("mamba2", "X-FLM"), ("gru", "X-FLG")))
ARMS = ("R1", "H1", "R2", "H2")
CONTRASTS = OrderedDict((
    ("generated_natural", {"H1": 1.0, "R1": -1.0}),
    ("generated_event", {"H2": 1.0, "R2": -1.0}),
    ("event_real", {"R2": 1.0, "R1": -1.0}),
    ("event_generated", {"H2": 1.0, "H1": -1.0}),
    ("factorial_interaction",
     {"H2": 1.0, "H1": -1.0, "R2": -1.0, "R1": 1.0}),
))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rankdata(values: np.ndarray) -> np.ndarray:
    # scipy's C/NumPy-backed implementation matters here: this function is
    # called inside thousands of cluster-bootstrap replicates.
    return scipy_rankdata(values, method="average") - 1.0


def corr(left: np.ndarray, right: np.ndarray, ranked: bool = False):
    if ranked:
        left, right = rankdata(left), rankdata(right)
    if left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def auroc(scores: np.ndarray, labels: np.ndarray):
    labels = labels.astype(bool)
    positives = int(labels.sum())
    negatives = int((~labels).sum())
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(scores) + 1.0
    numerator = ranks[labels].sum() - positives * (positives + 1) / 2
    return float(numerator / (positives * negatives))


def average_precision(scores: np.ndarray, labels: np.ndarray):
    labels = labels.astype(bool)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, labels = scores[order], labels[order]
    true_positive = false_positive = 0
    recall_before = result = 0.0
    start = 0
    while start < len(scores):
        stop = start + 1
        while stop < len(scores) and scores[stop] == scores[start]:
            stop += 1
        true_positive += int(labels[start:stop].sum())
        false_positive += int(stop - start - labels[start:stop].sum())
        recall = true_positive / positives
        result += (
            (recall - recall_before)
            * true_positive / (true_positive + false_positive)
        )
        recall_before = recall
        start = stop
    return float(result)


def reward_metrics(predicted: np.ndarray, actual: np.ndarray) -> dict:
    event = np.abs(actual) > 1e-6
    zero = ~event
    positive = actual > 1e-6
    negative = actual < -1e-6
    event_predicted = predicted[event]
    event_actual = actual[event]
    return {
        "event_auroc": auroc(np.abs(predicted), event),
        "event_average_precision": average_precision(
            np.abs(predicted), event),
        "reward_pearson": corr(predicted, actual),
        "reward_spearman": corr(predicted, actual, ranked=True),
        "mae_zero": float(np.abs(predicted[zero]).mean()),
        "mae_event": float(
            np.abs(event_predicted - event_actual).mean()),
        "decoded_abs_event_mean": float(
            np.abs(event_predicted).mean()),
        "decoded_positive_mean": float(predicted[positive].mean()),
        "decoded_negative_mean": float(predicted[negative].mean()),
        "reward_sign_accuracy": float(
            (np.sign(event_predicted) == np.sign(event_actual)).mean()),
        "reward_sign_auroc": auroc(
            event_predicted, event_actual > 0),
    }


def reward_bootstrap_metrics(
    predicted: np.ndarray,
    actual: np.ndarray,
) -> dict:
    """Primary paired readouts; omit AP/Spearman to keep bootstrap tractable."""
    event = np.abs(actual) > 1e-6
    zero = ~event
    event_predicted = predicted[event]
    event_actual = actual[event]
    return {
        "event_auroc": auroc(np.abs(predicted), event),
        "reward_pearson": corr(predicted, actual),
        "mae_zero": float(np.abs(predicted[zero]).mean()),
        "mae_event": float(
            np.abs(event_predicted - event_actual).mean()),
        "decoded_abs_event_mean": float(
            np.abs(event_predicted).mean()),
        "reward_sign_accuracy": float(
            (np.sign(event_predicted) == np.sign(event_actual)).mean()),
        "reward_sign_auroc": auroc(
            event_predicted, event_actual > 0),
    }


def continuation_metrics(
    predicted_continue: np.ndarray,
    actual_continue: np.ndarray,
) -> dict:
    terminal = actual_continue < 0.5
    predicted_terminal = 1.0 - predicted_continue
    brier = float(
        np.mean((predicted_continue - actual_continue) ** 2))
    climatology = float(
        np.mean((actual_continue - actual_continue.mean()) ** 2))
    return {
        "terminal_auroc": auroc(predicted_terminal, terminal),
        "brier_skill": 1.0 - brier / climatology,
        "predicted_termination_terminal_mean": float(
            predicted_terminal[terminal].mean()),
        "predicted_termination_nonterminal_mean": float(
            predicted_terminal[~terminal].mean()),
    }


def ranking_arrays(rows: list[dict]) -> tuple[dict, np.ndarray]:
    differing = [row for row in rows if row["differs"]]
    return {
        "chosen_minus_random": np.asarray(
            [row["chosen_minus_random"] for row in differing]),
        "regret": np.asarray([row["regret"] for row in differing]),
    }, np.asarray([row["env_seed"] for row in differing])


def zero_suffix_arrays(rows: list[dict]) -> tuple[dict, np.ndarray]:
    signed, absolute, clusters = [], [], []
    for row in rows:
        for name, actual in row["actual"].items():
            if abs(actual) <= 1e-9:
                value = float(row["j_sum"][name])
                signed.append(value)
                absolute.append(abs(value))
                clusters.append(row["env_seed"])
    return {
        "zero_suffix_predicted_sum": np.asarray(signed),
        "zero_suffix_abs_predicted_sum": np.asarray(absolute),
    }, np.asarray(clusters)


def mean_metrics(arrays: dict[str, np.ndarray]) -> dict:
    return {name: float(values.mean()) for name, values in arrays.items()}


def cluster_bootstrap_indices(
    clusters: np.ndarray,
    seed: int,
    draws: int = BOOTSTRAP_DRAWS,
) -> list[np.ndarray]:
    unique = np.unique(clusters)
    members = {
        cluster: np.flatnonzero(clusters == cluster)
        for cluster in unique
    }
    rng = np.random.default_rng(seed)
    return [
        np.concatenate([
            members[cluster]
            for cluster in rng.choice(unique, len(unique), replace=True)
        ])
        for _ in range(draws)
    ]


def contrast(
    arm_metrics: dict[str, dict],
    coefficients: dict[str, float],
) -> dict:
    names = next(iter(arm_metrics.values())).keys()
    return {
        name: float(sum(
            coefficient * arm_metrics[arm][name]
            for arm, coefficient in coefficients.items()
        ))
        for name in names
        if all(
            arm_metrics[arm][name] is not None
            for arm in coefficients
        )
    }


def bootstrap_metric_contrasts(
    arm_values: dict[str, np.ndarray],
    labels: np.ndarray,
    clusters: np.ndarray,
    metric_fn,
    seed: int,
) -> dict:
    point = {
        arm: metric_fn(values, labels)
        for arm, values in arm_values.items()
    }
    boot = {
        name: {metric: [] for metric in contrast(point, coefficients)}
        for name, coefficients in CONTRASTS.items()
    }
    for indices in cluster_bootstrap_indices(clusters, seed):
        sampled = {
            arm: metric_fn(values[indices], labels[indices])
            for arm, values in arm_values.items()
        }
        for name, coefficients in CONTRASTS.items():
            values = contrast(sampled, coefficients)
            for metric, value in values.items():
                if np.isfinite(value):
                    boot[name][metric].append(value)
    output = {}
    for name, coefficients in CONTRASTS.items():
        values = contrast(point, coefficients)
        output[name] = {}
        for metric, value in values.items():
            samples = boot[name][metric]
            output[name][metric] = {
                "delta": value,
                "ci95": [
                    float(x) for x in np.percentile(
                        samples, (2.5, 97.5))
                ],
            }
    return output


def bootstrap_array_contrasts(
    arm_arrays: dict[str, dict[str, np.ndarray]],
    clusters: np.ndarray,
    seed: int,
) -> dict:
    point = {
        arm: mean_metrics(arrays)
        for arm, arrays in arm_arrays.items()
    }
    boot = {
        name: {metric: [] for metric in contrast(point, coefficients)}
        for name, coefficients in CONTRASTS.items()
    }
    for indices in cluster_bootstrap_indices(clusters, seed):
        sampled = {
            arm: mean_metrics({
                metric: values[indices]
                for metric, values in arrays.items()
            })
            for arm, arrays in arm_arrays.items()
        }
        for name, coefficients in CONTRASTS.items():
            for metric, value in contrast(
                    sampled, coefficients).items():
                boot[name][metric].append(value)
    output = {}
    for name, coefficients in CONTRASTS.items():
        output[name] = {}
        for metric, value in contrast(point, coefficients).items():
            output[name][metric] = {
                "delta": value,
                "ci95": [
                    float(x) for x in np.percentile(
                        boot[name][metric], (2.5, 97.5))
                ],
            }
    return output


def merge_contrasts(*groups: dict) -> dict:
    output = {name: {} for name in CONTRASTS}
    for group in groups:
        for name, metrics in group.items():
            overlap = set(output[name]) & set(metrics)
            if overlap:
                raise RuntimeError(f"duplicate contrast metrics: {overlap}")
            output[name].update(metrics)
    return output


def training_schedule_audit(train: list[dict]) -> dict:
    uniform, event = window_index(train)
    output = {
        "uniform_pool_windows": len(uniform),
        "event_pool_windows": len(event),
        "event_pool_rate": len(event) / len(uniform),
        "realized": {},
    }
    for seed in SEEDS:
        for arm in ("H1", "H2"):
            rng = np.random.default_rng(10_000 + seed)
            schedule = []
            for _ in range(UPDATES):
                if arm == "H2":
                    half = BATCH // 2
                    schedule.extend(
                        uniform[int(rng.integers(len(uniform)))]
                        for _ in range(half)
                    )
                    schedule.extend(
                        event[int(rng.integers(len(event)))]
                        for _ in range(half)
                    )
                else:
                    schedule.extend(
                        uniform[int(rng.integers(len(uniform)))]
                        for _ in range(BATCH)
                    )
            labels = np.concatenate([
                train[episode]["rewards"][start:start + WINDOW - 1]
                for episode, start in schedule
            ])
            continues = np.concatenate([
                train[episode]["continues"][start:start + WINDOW - 1]
                for episode, start in schedule
            ])
            event_windows = sum(
                np.max(np.abs(
                    train[episode]["rewards"][
                        start + PREFIX - 1:start + PREFIX + 1
                    ]
                )) > 1e-6
                for episode, start in schedule
            )
            output["realized"][f"s{seed}_{arm}"] = {
                "event_window_fraction": event_windows / len(schedule),
                "unique_windows": len(set(schedule)),
                "all_label_event_fraction": float(
                    np.mean(np.abs(labels) > 1e-6)),
                "all_label_terminal_fraction": float(
                    np.mean(continues < 0.5)),
                "all_label_mean_reward": float(labels.mean()),
            }
    return output


def acceptance_direction_counts(stage1: dict, arm: str) -> dict:
    reward = (
        ("event_average_precision", 1),
        ("event_auroc", 1),
        ("reward_pearson", 1),
        ("reward_spearman", 1),
        ("mae_event", -1),
        ("nll_event", -1),
        ("decoded_abs_event_mean", 1),
    )
    continuation = (
        ("brier_skill", 1),
        ("terminal_average_precision", 1),
        ("terminal_auroc", 1),
        ("predicted_termination_terminal_mean", 1),
    )
    output = {}
    for label, metrics, block in (
        ("reward", reward, "reward_depth"),
        ("continuation", continuation, "continuation_depth"),
    ):
        deltas = []
        for kind in FAMILIES.values():
            for seed in SEEDS:
                adapted = stage1[f"{kind}_s{seed}_{arm}"]
                base = stage1[f"{kind}_s{seed}_H0"]
                for depth in ("k1", "k2", "k4", "k8"):
                    deltas.extend(
                        direction * (
                            adapted[block][depth][metric]
                            - base[block][depth][metric]
                        )
                        for metric, direction in metrics
                    )
        output[label] = {
            "improved": sum(value > 0 for value in deltas),
            "equal": sum(value == 0 for value in deltas),
            "worsened": sum(value < 0 for value in deltas),
        }
    ranking = []
    for kind in FAMILIES.values():
        for seed in SEEDS:
            adapted = stage1[f"{kind}_s{seed}_{arm}"]["ranking"]
            base = stage1[f"{kind}_s{seed}_H0"]["ranking"]
            ranking.extend((
                adapted["chosen_minus_random_mean"]
                - base["chosen_minus_random_mean"],
                base["regret_mean"] - adapted["regret_mean"],
            ))
    output["ranking"] = {
        "improved": sum(value > 0 for value in ranking),
        "equal": sum(value == 0 for value in ranking),
        "worsened": sum(value < 0 for value in ranking),
    }
    return output


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if sha256(RAW_PATH) != report["provenance"]["raw_sha256"]:
        raise RuntimeError("Stage-1b raw artifact drift")
    if len(report["results"]) != 24 or len(raw["results"]) != 24:
        raise RuntimeError("Stage-1b factorial is incomplete")

    natural_episodes = torch.load(NATURAL, weights_only=False)
    terminal_episodes = torch.load(TERMINAL, weights_only=False)
    natural_arrays = window_arrays(
        natural_episodes, target_rows(natural_episodes))
    terminal_rows = target_rows(terminal_episodes)
    terminal_arrays = window_arrays(terminal_episodes, terminal_rows)
    actual_continue = continuation_targets(
        terminal_episodes, terminal_rows)
    actual_reward = natural_arrays["rewards"]
    train, _ = load_scaled_data()
    stage1 = json.loads(STAGE1_REPORT.read_text())["results"]

    output = {
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "stage1b_report_sha256": sha256(REPORT_PATH),
            "stage1b_raw_sha256": sha256(RAW_PATH),
            "stage1_report_sha256": sha256(STAGE1_REPORT),
            "manifest_sha256": sha256(MANIFEST),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_unit": {
                "reward": "episode",
                "continuation": "episode",
                "ranking_and_suffix": "environment seed",
            },
        },
        "data": {
            "reward_targets": len(actual_reward),
            "reward_events": int(
                (np.abs(actual_reward) > 1e-6).sum()),
            "actual_abs_event_mean": float(np.abs(
                actual_reward[np.abs(actual_reward) > 1e-6]).mean()),
            "actual_positive_mean": float(
                actual_reward[actual_reward > 1e-6].mean()),
            "actual_negative_mean": float(
                actual_reward[actual_reward < -1e-6].mean()),
            "continuation_targets": len(actual_continue),
            "terminal_targets": int((actual_continue < 0.5).sum()),
        },
        "training_schedule": training_schedule_audit(train),
        "registered_acceptance_direction_audit": {
            arm: acceptance_direction_counts(stage1, arm)
            for arm in ("H1", "H2")
        },
        "checkpoints": {},
        "family_summary": {},
    }

    family_points = {
        family: {name: {} for name in CONTRASTS}
        for family in FAMILIES
    }
    for family_index, (family, kind) in enumerate(FAMILIES.items()):
        for seed in SEEDS:
            base = f"{kind}_s{seed}"
            reward_predictions = {}
            continue_predictions = {}
            ranking = {}
            zero_suffix = {}
            arm_points = {}
            for arm in ARMS:
                tag = f"{base}_{arm}"
                rows = raw["results"][tag]
                reward_predictions[arm] = np.asarray(
                    rows["reward_predictions"]["k8"])
                continue_predictions[arm] = np.asarray(
                    rows["continuation_predictions"]["k8"])
                ranking[arm], ranking_clusters = ranking_arrays(
                    rows["ranking_rows"])
                zero_suffix[arm], zero_clusters = zero_suffix_arrays(
                    rows["ranking_rows"])
                arm_points[arm] = {
                    "reward_k8": reward_metrics(
                        reward_predictions[arm], actual_reward),
                    "continuation_k8": continuation_metrics(
                        continue_predictions[arm], actual_continue),
                    "ranking": mean_metrics(ranking[arm]),
                    "zero_reward_suffix": mean_metrics(zero_suffix[arm]),
                }

            reward_contrasts = bootstrap_metric_contrasts(
                reward_predictions,
                actual_reward,
                natural_arrays["episodes"],
                reward_bootstrap_metrics,
                seed=18_100 + family_index * 100 + seed,
            )
            continuation_contrasts = bootstrap_metric_contrasts(
                continue_predictions,
                actual_continue,
                terminal_arrays["episodes"],
                continuation_metrics,
                seed=18_200 + family_index * 100 + seed,
            )
            ranking_contrasts = bootstrap_array_contrasts(
                ranking,
                ranking_clusters,
                seed=18_300 + family_index * 100 + seed,
            )
            zero_contrasts = bootstrap_array_contrasts(
                zero_suffix,
                zero_clusters,
                seed=18_400 + family_index * 100 + seed,
            )
            contrasts = merge_contrasts(
                reward_contrasts,
                continuation_contrasts,
                ranking_contrasts,
                zero_contrasts,
            )
            output["checkpoints"][base] = {
                "arms": arm_points,
                "paired_contrasts": contrasts,
            }
            for contrast_name, metrics in contrasts.items():
                for metric, block in metrics.items():
                    family_points[family][contrast_name].setdefault(
                        metric, []).append(block["delta"])

        output["family_summary"][family] = {}
        for contrast_name, metrics in family_points[family].items():
            output["family_summary"][family][contrast_name] = {
                metric: {
                    "training_seed_deltas": values,
                    "mean_delta": float(np.mean(values)),
                    "positive_seeds": int(
                        np.sum(np.asarray(values) > 0)),
                    "negative_seeds": int(
                        np.sum(np.asarray(values) < 0)),
                    "zero_seeds": int(
                        np.sum(np.asarray(values) == 0)),
                }
                for metric, values in metrics.items()
            }

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(OUTPUT, sha256(OUTPUT))


if __name__ == "__main__":
    main()
