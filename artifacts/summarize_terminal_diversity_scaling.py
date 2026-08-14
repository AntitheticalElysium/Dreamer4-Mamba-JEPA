"""Join training, held-out geometry, delta, and policy-fork scaling evidence."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from artifacts.evaluate_phase1b_archive_geometry import score_data
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from d4mj.config import Config


def compact_cell(
    name: str,
    training: dict,
    archive: dict,
    archive_features: Path,
    policy: dict,
    delta: dict,
    prepared: dict,
    step: int,
    seed: int,
) -> dict:
    world = f"{name}_{step // 1000:03d}k"
    dev = archive["worlds"][world]
    features = torch.load(
        archive_features / f"{world}.pt", weights_only=False, map_location="cpu"
    )
    train = score_data(
        features["train_paths"]["reset16"],
        prepared,
        seed=seed,
        bootstraps=1000,
    )["combined"]
    support = dev["paths"]["reset16"]["support"]
    output = {
        "unique_terminal_episodes": training["contract"]["cell"]
        ["unique_terminal_episodes"],
        "replicate": training["contract"]["cell"]["replicate"],
        "step": step,
        "train": {
            "fixed_direction_auc": train["predicted_separation"]["auc"],
            "fixed_direction_auc_ci95": train["predicted_separation"]["auc_ci95"],
            "total_mse": train["total_mse"],
            "direction_mse_over_variance": train[
                "direction_mse_over_target_variance"
            ],
        },
        "heldout_support": {
            "fixed_direction_auc": support["predicted_separation"]["auc"],
            "fixed_direction_auc_ci95": support["predicted_separation"]["auc_ci95"],
            "fresh_probe_auc": dev["generated_probe"]["reset16"]["support"]["auc"],
            "fresh_probe_auc_ci95": dev["generated_probe"]["reset16"]["support"][
                "auc_ci95"
            ],
            "total_mse": support["total_mse"],
            "direction_mse_over_variance": support[
                "direction_mse_over_target_variance"
            ],
        },
        "delta": delta["worlds"][world]["archive"]["reset16"]["support"],
    }
    if world in policy["worlds"]:
        probe = policy["worlds"][world]["generated_latent_probe"]
        output["policy_forks"] = {
            "trajectory_action_auc": probe["trajectory_action"]["auc"],
            "other_16_actions_auc": probe["other_16_actions"]["auc"],
            "delta": delta["worlds"][world]["policy_forks"],
        }
    return output


def aggregate(cells: dict[str, dict]) -> dict:
    grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for value in cells.values():
        grouped[(value["unique_terminal_episodes"], value["step"])].append(value)
    output = {}
    for (size, step), rows in sorted(grouped.items()):
        metrics = {
            "train_fixed_auc": [row["train"]["fixed_direction_auc"] for row in rows],
            "heldout_fixed_auc": [
                row["heldout_support"]["fixed_direction_auc"] for row in rows
            ],
            "heldout_fresh_auc": [
                row["heldout_support"]["fresh_probe_auc"] for row in rows
            ],
            "heldout_mse": [row["heldout_support"]["total_mse"] for row in rows],
            "predicted_fatal_delta": [
                row["delta"]["predicted_delta"]["fatal"]["mean"] for row in rows
            ],
            "heldout_conditional_contrast": [
                row["delta"]["conditional_consequence"][
                    "predicted_fatal_minus_safe"
                ]
                for row in rows
            ],
            "heldout_conditional_slope": [
                row["delta"]["conditional_consequence"][
                    "within_group_predicted_vs_true_slope"
                ]
                for row in rows
            ],
            "heldout_recovered_fraction": [
                row["delta"]["conditional_consequence"]["recovered_fraction"]
                for row in rows
            ],
        }
        if all("policy_forks" in row for row in rows):
            metrics |= {
                "trajectory_action_auc": [
                    row["policy_forks"]["trajectory_action_auc"] for row in rows
                ],
                "other_16_actions_auc": [
                    row["policy_forks"]["other_16_actions_auc"] for row in rows
                ],
            }
        output[f"k{size:04d}_{step // 1000:03d}k"] = {
            "replicates": len(rows),
            "metrics": {
                name: {
                    "mean": sum(values) / len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
                for name, values in metrics.items()
            },
        }
    return output


def endpoint_log_slope(aggregate_rows: dict) -> dict:
    points = []
    for name, row in aggregate_rows.items():
        if not name.endswith("020k"):
            continue
        size = int(name.split("_")[0][1:])
        points.append(
            (
                math.log(size),
                row["metrics"]["heldout_conditional_contrast"]["mean"],
            )
        )
    x = torch.tensor([value[0] for value in points])
    y = torch.tensor([value[1] for value in points])
    centered = x - x.mean()
    slope = float(
        (centered * (y - y.mean())).sum()
        / centered.square().sum().clamp(min=1e-12)
    )
    return {
        "heldout_conditional_contrast_per_log_unique_episode": slope,
        "descriptive_only": True,
        "rule": (
            "The primary trend is conditional fatal-minus-matched-safe movement; "
            "AUC is deliberately secondary."
        ),
    }


def contrast_vector(features: Path, prepared: dict) -> torch.Tensor:
    payload = torch.load(features, weights_only=False, map_location="cpu")["paths"]["reset16"]
    support = torch.tensor([value == "support" for value in payload["pool"]])
    start = torch.cat([
        record["latents"][record["transitions"]] for record in prepared["records"]
    ]).flatten(1).float()[support]
    predicted = payload["predicted"].flatten(1).float()[support]
    label = payload["label"].bool()[support]
    group = payload["group"].long()[support]
    direction = prepared["direction"].flatten().float()
    delta = (predicted - start) @ direction
    contrast = []
    for value in group.unique():
        rows = group == value
        contrast.append(delta[rows][label[rows]].mean() - delta[rows][~label[rows]].mean())
    return torch.stack(contrast)


def saturation_verdict(
    training_summary: dict,
    archive_features: Path,
    prepared: dict,
    *,
    samples: int = 2000,
) -> dict:
    sizes = sorted({
        report["contract"]["cell"]["unique_terminal_episodes"]
        for report in (
            json.loads(Path(path).read_text())
            for path in training_summary["reports"].values()
        )
        if report["contract"]["cell"]["replicate"] == 0
    })
    vectors = {
        size: contrast_vector(
            archive_features / f"k{size:04d}_r0_020k.pt", prepared
        )
        for size in sizes
    }
    true_groups = defaultdict(lambda: {"score": [], "label": []})
    direction = prepared["direction"].flatten().float()
    for record in prepared["records"]:
        steps = record["transitions"]
        delta = (
            record["latents"][steps + 1].flatten(1)
            - record["latents"][steps].flatten(1)
        ) @ direction
        group = int(record.get("group", record["episode_index"]))
        true_groups[group]["score"].append(delta)
        true_groups[group]["label"].append(record["labels"].bool())
    true_values = []
    for value in true_groups.values():
        score = torch.cat(value["score"])
        label = torch.cat(value["label"])
        true_values.append(score[label].mean() - score[~label].mean())
    minimum_effect = 0.05 * abs(float(torch.stack(true_values).mean()))
    rng = torch.Generator().manual_seed(Config().seed + 10_400)
    increments = []
    for smaller, larger in zip(sizes, sizes[1:]):
        difference = vectors[larger] - vectors[smaller]
        estimates = []
        for _ in range(samples):
            chosen = torch.randint(len(difference), (len(difference),), generator=rng)
            estimates.append(difference[chosen].mean())
        distribution = torch.stack(estimates)
        increments.append({
            "from": smaller,
            "to": larger,
            "mean": float(difference.mean()),
            "ci95": [
                float(distribution.quantile(0.025)),
                float(distribution.quantile(0.975)),
            ],
        })
    top = increments[-2:]
    equivalent = lambda row: (
        row["ci95"][0] > -minimum_effect
        and row["ci95"][1] < minimum_effect
    )
    changed = lambda row: (
        row["ci95"][0] > minimum_effect
        or row["ci95"][1] < -minimum_effect
    )
    if len(top) == 2 and all(equivalent(row) for row in top):
        verdict = "saturated"
        approximate = top[0]["from"]
    elif top and changed(top[-1]):
        verdict = "not_saturated"
        approximate = None
    else:
        verdict = "inconclusive"
        approximate = None
    return {
        "verdict": verdict,
        "approximately_saturated_by_unique_episodes": approximate,
        "minimum_meaningful_increment": minimum_effect,
        "minimum_effect_rule": "5% of the held-out true fatal-minus-matched-safe delta",
        "criterion": (
            "saturated only when each of the final two paired 95% intervals lies "
            "inside the two-sided practical-equivalence band"
        ),
        "increments": increments,
    }


def plot_scaling(aggregate_rows: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for step, style in ((5, ":"), (10, "--"), (20, "-")):
        rows = []
        for name, value in aggregate_rows.items():
            if not name.endswith(f"{step:03d}k"):
                continue
            size = int(name.split("_")[0][1:])
            rows.append((size, value))
        rows.sort()
        sizes = [item[0] for item in rows]
        for axis, metric, label in (
            (axes[0], "heldout_conditional_contrast", "DEV conditional delta"),
            (axes[1], "heldout_fixed_auc", "DEV fixed AUC"),
            (axes[1], "heldout_fresh_auc", "DEV fresh-probe AUC"),
        ):
            means = [item[1]["metrics"][metric]["mean"] for item in rows]
            low = [item[1]["metrics"][metric]["minimum"] for item in rows]
            high = [item[1]["metrics"][metric]["maximum"] for item in rows]
            axis.plot(sizes, means, linestyle=style, marker="o", label=f"{label} {step}k")
            axis.fill_between(sizes, low, high, alpha=0.12)
    axes[0].axhline(0.0, color="black", linewidth=0.7)
    axes[1].axhline(0.5, color="black", linewidth=0.7)
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("unique terminal episodes")
        axis.legend(fontsize=7)
    axes[0].set_ylabel("fatal minus matched-safe predicted delta")
    axes[1].set_ylabel("fatality AUC (secondary)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-features", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--delta", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    training_summary = json.loads(args.training.read_text())
    archive = json.loads(args.archive.read_text())
    policy = json.loads(args.policy.read_text())
    delta = json.loads(args.delta.read_text())
    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    if not all(training_summary["invariants"].values()):
        raise AssertionError("terminal-diversity training was not matched")
    reports = {
        name: json.loads(Path(path).read_text())
        for name, path in training_summary["reports"].items()
    }
    cells = {}
    for index, (name, report) in enumerate(reports.items()):
        for step in training_summary["common"]["milestones"]:
            key = f"{name}_{step // 1000:03d}k"
            cells[key] = compact_cell(
                name,
                report,
                archive,
                args.archive_features,
                policy,
                delta,
                prepared,
                step,
                seed=Config().seed + 10_200 + index * 100 + step,
            )
    aggregated = aggregate(cells)
    contract = {
        "version": "terminal-diversity-scaling-summary-v1",
        "inputs": {
            "training": file_digest(args.training),
            "prepared": file_digest(args.prepared),
            "archive": file_digest(args.archive),
            "policy": file_digest(args.policy),
            "delta": file_digest(args.delta),
        },
        "implementation": implementation_digests(Path(__file__)),
    }
    output = {
        "contract": contract,
        "matching": training_summary["invariants"],
        "primary_evaluation": (
            "whole held-out support DEV episodes; fixed TRAIN fatality direction; "
            "FINAL remains untouched"
        ),
        "cells": cells,
        "aggregate": aggregated,
        "endpoint_trend": endpoint_log_slope(aggregated),
        "saturation": saturation_verdict(
            training_summary, args.archive_features, prepared
        ),
    }
    atomic_json(args.out, output)
    plot_scaling(aggregated, args.out.parent / "terminal_diversity_curve.png")
    print(f"complete: {args.out}", flush=True)

if __name__ == "__main__":
    main()
