"""Paired cluster-bootstrap analysis of the registered Stage-1c D8-D2 run."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

COMPACT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = COMPACT_ROOT.parent
sys.path.insert(0, str(COMPACT_ROOT))
sys.path.insert(0, str(COMPACT_ROOT / "verification"))

from phase_e_continuation_depth import continuation_targets  # noqa: E402
from phase_e_same_target import target_rows, window_arrays  # noqa: E402
from stage1_head_adaptation import NATURAL, TERMINAL  # noqa: E402
from stage1b_equal_update_analysis import (  # noqa: E402
    cluster_bootstrap_indices,
    continuation_metrics,
    mean_metrics,
    ranking_arrays,
    reward_bootstrap_metrics,
    reward_metrics,
    zero_suffix_arrays,
)
from stage1c_head_depth_ceiling import (  # noqa: E402
    PREFIX,
    RAW_PATH,
    REPORT_PATH,
    WINDOW,
    build_schedule,
)
from step3_temporal import load_scaled_data  # noqa: E402

OUTPUT = REPO_ROOT / \
    "reviews/artifacts/stage1c_head_depth_analysis.json"
DEPENDENCIES = (
    COMPACT_ROOT / "verification/stage1b_equal_update_analysis.py",
    COMPACT_ROOT / "verification/stage1c_head_depth_ceiling.py",
)
KINDS = ("X-FLM", "X-FLG")
DEPTH_KEYS = ("k1", "k8")
BOOTSTRAP_DRAWS = 2_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_metric(
    d2: np.ndarray,
    d8: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    metric_fn,
    seed: int,
) -> dict:
    point2 = metric_fn(d2, labels)
    point8 = metric_fn(d8, labels)
    names = {
        name for name in point2
        if point2[name] is not None and point8[name] is not None
    }
    samples = {name: [] for name in names}
    for indices in cluster_bootstrap_indices(
            clusters, seed, draws=BOOTSTRAP_DRAWS):
        metric2 = metric_fn(d2[indices], labels[indices])
        metric8 = metric_fn(d8[indices], labels[indices])
        for name in names:
            left, right = metric2[name], metric8[name]
            if left is not None and right is not None:
                samples[name].append(right - left)
    return {
        name: {
            "delta": float(point8[name] - point2[name]),
            "ci95": [
                float(value) for value in np.percentile(
                    samples[name], (2.5, 97.5))
            ],
        }
        for name in names
    }


def paired_arrays(
    d2: dict[str, np.ndarray],
    d8: dict[str, np.ndarray],
    clusters: np.ndarray,
    seed: int,
) -> dict:
    point2, point8 = mean_metrics(d2), mean_metrics(d8)
    samples = {name: [] for name in point2}
    for indices in cluster_bootstrap_indices(
            clusters, seed, draws=BOOTSTRAP_DRAWS):
        metric2 = mean_metrics({
            name: values[indices] for name, values in d2.items()
        })
        metric8 = mean_metrics({
            name: values[indices] for name, values in d8.items()
        })
        for name in samples:
            samples[name].append(metric8[name] - metric2[name])
    return {
        name: {
            "delta": float(point8[name] - point2[name]),
            "ci95": [
                float(value) for value in np.percentile(
                    samples[name], (2.5, 97.5))
            ],
        }
        for name in point2
    }


def schedule_audit() -> dict:
    train, _ = load_scaled_data()
    schedule, digest = build_schedule(train)
    output = {"schedule_sha256": digest, "arms": {}}
    for arm, depth in (("D2", 2), ("D8", 8)):
        rewards, continues = [], []
        terminal_by_generated_depth = [0] * depth
        for episode_index, start in schedule:
            episode = train[episode_index]
            indices = (
                list(range(start, start + PREFIX - 1))
                + list(range(
                    start + PREFIX - 1,
                    start + PREFIX - 1 + depth,
                ))
            )
            rewards.extend(np.asarray(episode["rewards"])[indices])
            continues.extend(np.asarray(episode["continues"])[indices])
            for generated_depth in range(depth):
                value = episode["continues"][
                    start + PREFIX - 1 + generated_depth]
                terminal_by_generated_depth[generated_depth] += \
                    int(value < 0.5)
        rewards = np.asarray(rewards)
        continues = np.asarray(continues)
        output["arms"][arm] = {
            "targets": len(rewards),
            "event_targets": int(
                (np.abs(rewards) > 1e-6).sum()),
            "event_fraction": float(
                np.mean(np.abs(rewards) > 1e-6)),
            "terminal_targets": int((continues < 0.5).sum()),
            "terminal_fraction": float(
                np.mean(continues < 0.5)),
            "terminal_by_generated_depth":
                terminal_by_generated_depth,
        }
    return output


def main() -> None:
    report = json.loads(REPORT_PATH.read_text())
    raw = json.loads(RAW_PATH.read_text())
    if sha256(RAW_PATH) != report["provenance"]["raw_sha256"]:
        raise RuntimeError("Stage-1c raw artifact drift")
    if len(report["results"]) != 4 or len(raw["results"]) != 4:
        raise RuntimeError("Stage-1c run is incomplete")

    natural_episodes = torch.load(NATURAL, weights_only=False)
    terminal_episodes = torch.load(TERMINAL, weights_only=False)
    reward_arrays = window_arrays(
        natural_episodes, target_rows(natural_episodes))
    terminal_rows = target_rows(terminal_episodes)
    terminal_arrays = window_arrays(terminal_episodes, terminal_rows)
    actual_reward = reward_arrays["rewards"]
    actual_continue = continuation_targets(
        terminal_episodes, terminal_rows)

    output = {
        "provenance": {
            "script_sha256": sha256(Path(__file__)),
            "dependency_sha256": {
                str(path.relative_to(REPO_ROOT)): sha256(path)
                for path in DEPENDENCIES
            },
            "report_sha256": sha256(REPORT_PATH),
            "raw_sha256": sha256(RAW_PATH),
            "bootstrap_draws": BOOTSTRAP_DRAWS,
        },
        "schedule": schedule_audit(),
        "results": {},
    }

    for kind_index, kind in enumerate(KINDS):
        base = f"{kind}_s505"
        rows2 = raw["results"][f"{base}_D2"]
        rows8 = raw["results"][f"{base}_D8"]
        block = {"arm_points": {}, "d8_minus_d2": {}}

        for arm, rows in (("D2", rows2), ("D8", rows8)):
            ranking, _ = ranking_arrays(rows["ranking_rows"])
            zero_suffix, _ = zero_suffix_arrays(
                rows["ranking_rows"])
            block["arm_points"][arm] = {
                "reward": {
                    depth: reward_metrics(
                        np.asarray(
                            rows["reward_predictions"][depth]),
                        actual_reward,
                    )
                    for depth in DEPTH_KEYS
                },
                "continuation": {
                    depth: continuation_metrics(
                        np.asarray(
                            rows["continuation_predictions"][depth]),
                        actual_continue,
                    )
                    for depth in DEPTH_KEYS
                },
                "ranking": mean_metrics(ranking),
                "zero_reward_suffix": mean_metrics(zero_suffix),
            }

        for depth_index, depth in enumerate(DEPTH_KEYS):
            block["d8_minus_d2"][f"reward_{depth}"] = paired_metric(
                np.asarray(rows2["reward_predictions"][depth]),
                np.asarray(rows8["reward_predictions"][depth]),
                actual_reward,
                reward_arrays["episodes"],
                reward_bootstrap_metrics,
                seed=18_510 + kind_index * 100 + depth_index,
            )
            block["d8_minus_d2"][
                f"continuation_{depth}"
            ] = paired_metric(
                np.asarray(
                    rows2["continuation_predictions"][depth]),
                np.asarray(
                    rows8["continuation_predictions"][depth]),
                actual_continue,
                terminal_arrays["episodes"],
                continuation_metrics,
                seed=18_520 + kind_index * 100 + depth_index,
            )

        ranking2, ranking_clusters = ranking_arrays(
            rows2["ranking_rows"])
        ranking8, _ = ranking_arrays(rows8["ranking_rows"])
        block["d8_minus_d2"]["ranking"] = paired_arrays(
            ranking2,
            ranking8,
            ranking_clusters,
            seed=18_530 + kind_index * 100,
        )
        zero2, zero_clusters = zero_suffix_arrays(
            rows2["ranking_rows"])
        zero8, _ = zero_suffix_arrays(rows8["ranking_rows"])
        block["d8_minus_d2"]["zero_reward_suffix"] = \
            paired_arrays(
                zero2,
                zero8,
                zero_clusters,
                seed=18_540 + kind_index * 100,
            )
        output["results"][base] = block

    OUTPUT.write_text(json.dumps(output, indent=2))
    print(OUTPUT, sha256(OUTPUT))


if __name__ == "__main__":
    main()
