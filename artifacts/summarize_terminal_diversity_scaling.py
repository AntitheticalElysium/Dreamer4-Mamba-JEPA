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
                row["metrics"]["heldout_fixed_auc"]["mean"],
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
        "heldout_fixed_auc_per_log_unique_episode": slope,
        "descriptive_only": True,
        "rule": (
            "A positive descriptive slope motivates new unique death collection; "
            "a flat curve with a persistent TRAIN-DEV gap favors objective or model repair."
        ),
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
            (axes[0], "train_fixed_auc", "TRAIN"),
            (axes[0], "heldout_fixed_auc", "DEV fixed"),
            (axes[1], "heldout_fresh_auc", "DEV fresh probe"),
        ):
            means = [item[1]["metrics"][metric]["mean"] for item in rows]
            low = [item[1]["metrics"][metric]["minimum"] for item in rows]
            high = [item[1]["metrics"][metric]["maximum"] for item in rows]
            axis.plot(sizes, means, linestyle=style, marker="o", label=f"{label} {step}k")
            axis.fill_between(sizes, low, high, alpha=0.12)
    for axis in axes:
        axis.axhline(0.5, color="black", linewidth=0.7)
        axis.set_xscale("log")
        axis.set_xlabel("unique terminal episodes")
        axis.set_ylabel("fatality AUC")
        axis.legend(fontsize=7)
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
    }
    atomic_json(args.out, output)
    plot_scaling(aggregated, args.out.parent / "terminal_diversity_curve.png")
    print(f"complete: {args.out}", flush=True)

if __name__ == "__main__":
    main()
