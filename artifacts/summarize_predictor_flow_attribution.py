"""Summarize paired predictor-topology and Flow-diversity effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.evaluate_fatality_direction_delta import archive_current
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import finite_json
from d4mj.config import Config


def contrast_vector(
    data: dict, current: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    predicted = data["predicted"].flatten(1).float()
    label = data["label"].bool()
    group = data["group"].long()
    score = (predicted - current.flatten(1).float()) @ direction.flatten().float()
    values = []
    for current in group.unique():
        rows = group == current
        values.append(score[rows][label[rows]].mean() - score[rows][~label[rows]].mean())
    return torch.stack(values)


def fork_contrast_vector(data: dict, direction: torch.Tensor) -> torch.Tensor:
    score = data["predicted"].flatten(1).float() @ direction.flatten().float()
    label = data["target"].bool()
    group = data["group"].long()
    values = []
    for current in group.unique():
        rows = group == current
        values.append(score[rows][label[rows]].mean() - score[rows][~label[rows]].mean())
    return torch.stack(values)


def paired_difference(
    intervention: torch.Tensor,
    control: torch.Tensor,
    *,
    minimum_effect: float,
    seed: int,
    samples: int,
) -> dict:
    if intervention.shape != control.shape:
        raise ValueError("paired contrast vectors differ in shape")
    difference = intervention - control
    rng = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(samples):
        chosen = torch.randint(len(difference), (len(difference),), generator=rng)
        estimates.append(difference[chosen].mean())
    distribution = torch.stack(estimates)
    interval = [
        float(distribution.quantile(0.025)),
        float(distribution.quantile(0.975)),
    ]
    return {
        "groups": len(difference),
        "mean": float(difference.mean()),
        "ci95": interval,
        "minimum_meaningful_effect": minimum_effect,
        "material_improvement": interval[0] > minimum_effect,
        "practically_equivalent": (
            interval[0] > -minimum_effect and interval[1] < minimum_effect
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-features", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--fork-features", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    training = json.loads(args.training.read_text())
    archive = json.loads(args.archive.read_text())
    forks = json.loads(args.forks.read_text())
    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    direction = prepared["direction"]
    archive_start = archive_current(prepared["records"])
    inputs = {
        "training": file_digest(args.training),
        "prepared": file_digest(args.prepared),
        "archive": file_digest(args.archive),
        "forks": file_digest(args.forks),
    }
    contract = {
        "version": "predictor-flow-attribution-summary-v1",
        "inputs": inputs,
        "implementation": implementation_digests(Path(__file__)),
        "primary": "fatal-minus-safe movement along the fixed TRAIN fatality direction",
        "uncertainty": "paired whole DEV episode or whole policy-fork state bootstrap",
        "minimum_effect": "5% of the corresponding true held-out consequence contrast",
        "decision_rule": (
            "a rescue requires the paired lower 95% bound to exceed the minimum "
            "effect on both logged DEV and policy forks; equivalence requires the "
            "whole interval inside the two-sided band"
        ),
    }

    archive_data = {}
    fork_data = {}
    compact = {}
    for name, row in archive["worlds"].items():
        archive_payload = torch.load(
            args.archive_features / f"{name}.pt", weights_only=False, map_location="cpu"
        )
        archive_data[name] = {
            path: contrast_vector(data, archive_start, direction)
            for path, data in archive_payload["paths"].items()
        }
        fork_payload = torch.load(
            args.fork_features / f"{name}.pt", weights_only=False, map_location="cpu"
        )
        fork_data[name] = {
            variant: fork_contrast_vector(data, direction)
            for variant, data in fork_payload["variants"].items()
        }
        compact[name] = {
            "transition": row["transition"],
            "predictor": row["predictor"],
            "archive": {},
            "policy_forks": {},
        }
        for path, result in row["paths"].items():
            support = result["geometry"]["support"]
            compact[name]["archive"][path] = {
                "mse": support["total_mse"],
                "fixed_direction_auc": support["predicted_separation"]["auc"],
                "fresh_probe_auc": result["fresh_probe"]["support"]["auc"],
                "conditional_consequence": result["delta"]["support"][
                    "conditional_consequence"
                ],
            }
        for variant, splits in forks["worlds"][name]["variants"].items():
            compact[name]["policy_forks"][variant] = {
                split: {
                    "probe_auc": value["probe"]["auc"],
                    "mse": value["latent_mse"]["all"],
                    "conditional_consequence": value["delta"][
                        "conditional_consequence"
                    ],
                }
                for split, value in splits.items()
            }

    direct_names = {
        row["predictor"]: name
        for name, row in archive["worlds"].items()
        if row["transition"] == "direct"
    }
    flow_names = [
        name for name, row in archive["worlds"].items() if row["transition"] == "flow"
    ]
    if set(direct_names) != {"current", "deep_mlp", "token_transformer"}:
        raise ValueError("Direct topology cells are incomplete")
    if len(flow_names) != 2:
        raise ValueError("Flow diversity endpoints are incomplete")
    flow_names.sort(key=lambda name: training["reports"][name]["contract"]["cell"]["unique_terminal_episodes"])

    true_archive = compact[direct_names["current"]]["archive"]["reset16"][
        "conditional_consequence"
    ]["true_fatal_minus_safe"]
    true_policy = compact[direct_names["current"]]["policy_forks"]["generated"][
        "all_actions"
    ]["conditional_consequence"]["true_fatal_minus_safe"]
    minimum_archive = 0.05 * abs(true_archive)
    minimum_policy = 0.05 * abs(true_policy)

    comparisons = {"direct": {}, "flow": {}}
    for label, intervention, control in (
        ("deep_mlp_minus_current", "deep_mlp", "current"),
        ("token_transformer_minus_current", "token_transformer", "current"),
        ("token_transformer_minus_deep_mlp", "token_transformer", "deep_mlp"),
    ):
        first, second = direct_names[intervention], direct_names[control]
        comparisons["direct"][label] = {
            "archive": paired_difference(
                archive_data[first]["reset16"],
                archive_data[second]["reset16"],
                minimum_effect=minimum_archive,
                seed=Config().seed + 16_000 + len(comparisons["direct"]),
                samples=args.bootstraps,
            ),
            "policy_forks": paired_difference(
                fork_data[first]["generated"],
                fork_data[second]["generated"],
                minimum_effect=minimum_policy,
                seed=Config().seed + 16_500 + len(comparisons["direct"]),
                samples=args.bootstraps,
            ),
        }

    small, full = flow_names
    for variant, archive_path in (
        ("generated_first", "reset16_first"),
        ("generated_mean", "reset16_mean"),
    ):
        comparisons["flow"][f"full_minus_small_{variant}"] = {
            "archive": paired_difference(
                archive_data[full][archive_path],
                archive_data[small][archive_path],
                minimum_effect=minimum_archive,
                seed=Config().seed + 17_000 + len(comparisons["flow"]),
                samples=args.bootstraps,
            ),
            "policy_forks": paired_difference(
                fork_data[full][variant],
                fork_data[small][variant],
                minimum_effect=minimum_policy,
                seed=Config().seed + 17_500 + len(comparisons["flow"]),
                samples=args.bootstraps,
            ),
        }

    def both(row: dict, key: str) -> bool:
        return row["archive"][key] and row["policy_forks"][key]

    deep = comparisons["direct"]["deep_mlp_minus_current"]
    token = comparisons["direct"]["token_transformer_minus_current"]
    token_over_deep = comparisons["direct"]["token_transformer_minus_deep_mlp"]
    if both(token, "material_improvement") and both(token_over_deep, "material_improvement"):
        direct_verdict = "token_topology_rescue"
    elif both(deep, "material_improvement") and not both(token_over_deep, "material_improvement"):
        direct_verdict = "capacity_rescue_without_token_specific_evidence"
    elif both(deep, "practically_equivalent") and both(token, "practically_equivalent"):
        direct_verdict = "neither_capacity_nor_token_topology_rescues"
    else:
        direct_verdict = "mixed_or_inconclusive"

    flow_first = comparisons["flow"]["full_minus_small_generated_first"]
    if both(flow_first, "material_improvement"):
        flow_verdict = "terminal_diversity_rescue"
    elif both(flow_first, "practically_equivalent"):
        flow_verdict = "terminal_diversity_not_a_material_flow_effect"
    else:
        flow_verdict = "mixed_or_inconclusive"

    report = finite_json(
        {
            "contract": contract,
            "matching": training["invariants"],
            "cells": compact,
            "comparisons": comparisons,
            "verdicts": {"direct": direct_verdict, "flow": flow_verdict},
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "report.json", report)
    print(f"Direct verdict: {direct_verdict}", flush=True)
    print(f"Flow verdict: {flow_verdict}", flush=True)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
