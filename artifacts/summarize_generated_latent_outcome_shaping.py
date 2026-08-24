"""Summarize the matched generated-latent outcome-shaping experiment."""

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
from artifacts.summarize_predictor_flow_attribution import (
    contrast_vector,
    fork_contrast_vector,
    paired_difference,
)
from d4mj.config import Config


def cell_metrics(archive: dict, forks: dict, heads: dict, name: str) -> dict:
    logged = archive["worlds"][name]["paths"]["reset16"]
    policy = forks["worlds"][name]["variants"]["generated"]["all_actions"]
    return {
        "archive_support": {
            "mse": logged["geometry"]["support"]["total_mse"],
            "fixed_direction_auc": logged["geometry"]["support"][
                "predicted_separation"
            ]["auc"],
            "fresh_probe_auc": logged["fresh_probe"]["support"]["auc"],
            "conditional_consequence": logged["delta"]["support"][
                "conditional_consequence"
            ],
            "trained_head": heads["models"][name]["archive"]["support"],
        },
        "policy_forks": {
            "mse": policy["latent_mse"]["all"],
            "fresh_probe_auc": policy["probe"]["auc"],
            "conditional_consequence": policy["delta"]["conditional_consequence"],
            "trained_head": heads["models"][name]["policy_forks"],
        },
    }


def head_valid(heads: dict, names: tuple[str, str]) -> dict:
    checks = {}
    for name in names:
        observed_archive = heads["models"][name]["archive"]["support"]["target"]
        observed_forks = heads["models"][name]["policy_forks"]["observed"]
        checks[name] = {
            "archive_observed_auc_ci_above_chance": observed_archive[
                "death_auc_ci95"
            ][0]
            > 0.5,
            "fork_observed_auc_ci_above_chance": observed_forks[
                "death_auc_ci95"
            ][0]
            > 0.5,
        }
    return {
        "rule": (
            "each final trained head's observed-latent death-AUC lower clustered "
            "95% bound exceeds chance on archive support and policy forks"
        ),
        "checks": checks,
        "passed": all(all(row.values()) for row in checks.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-features", type=Path, required=True)
    parser.add_argument("--forks", type=Path, required=True)
    parser.add_argument("--fork-features", type=Path, required=True)
    parser.add_argument("--heads", type=Path, required=True)
    parser.add_argument("--old-gradient-training", type=Path, required=True)
    parser.add_argument("--old-gradient-probe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstraps", type=int, default=2000)
    args = parser.parse_args()

    training = json.loads(args.training.read_text())
    archive = json.loads(args.archive.read_text())
    forks = json.loads(args.forks.read_text())
    heads = json.loads(args.heads.read_text())
    old_training = json.loads(args.old_gradient_training.read_text())
    old_probe = json.loads(args.old_gradient_probe.read_text())
    prepared = torch.load(args.prepared, weights_only=False, map_location="cpu")
    milestones = tuple(training["contract"]["milestones"])
    direction = prepared["direction"]
    archive_start = archive_current(prepared["records"])
    inputs = {
        "training": file_digest(args.training),
        "prepared": file_digest(args.prepared),
        "archive": file_digest(args.archive),
        "forks": file_digest(args.forks),
        "heads": file_digest(args.heads),
        "old_gradient_training": file_digest(args.old_gradient_training),
        "old_gradient_probe": file_digest(args.old_gradient_probe),
    }
    contract = {
        "version": "generated-latent-outcome-shaping-summary-v1",
        "inputs": inputs,
        "implementation": implementation_digests(
            Path(__file__),
            Path("artifacts/summarize_predictor_flow_attribution.py"),
        ),
        "primary": (
            "allowed-minus-stopped fatal-minus-safe generated-latent movement "
            "along the fixed TRAIN fatality direction"
        ),
        "primary_endpoint": milestones[-1],
        "supporting_endpoints": list(milestones[:-1]),
        "uncertainty": "paired whole DEV episode or policy-fork-state bootstrap",
        "minimum_effect": "5% of the corresponding true held-out contrast",
        "rescue_rule": (
            "at the final endpoint the lower paired 95% bound exceeds the minimum "
            "effect on both archive support and policy forks, after the trained-head "
            "validity check passes"
        ),
        "equivalence_rule": (
            "both paired intervals lie wholly inside their corresponding two-sided "
            "minimum-effect bands"
        ),
        "trained_head_role": "validity check only, never a rescue endpoint",
    }

    compact, comparisons = {}, {}
    archive_vectors, fork_vectors = {}, {}
    for step in milestones:
        tag = f"{step // 1000:03d}k"
        for variant in ("allowed", "stopped"):
            name = f"{variant}_{tag}"
            compact[name] = cell_metrics(archive, forks, heads, name)
            archive_payload = torch.load(
                args.archive_features / f"{name}.pt",
                weights_only=False,
                map_location="cpu",
            )["paths"]["reset16"]
            fork_payload = torch.load(
                args.fork_features / f"{name}.pt",
                weights_only=False,
                map_location="cpu",
            )["variants"]["generated"]
            support = torch.tensor(
                [pool == "support" for pool in archive_payload["pool"]]
            )
            archive_vectors[name] = contrast_vector(
                {
                    key: (
                        [value for value, keep in zip(archive_payload[key], support) if keep]
                        if key == "pool"
                        else archive_payload[key][support]
                    )
                    for key in archive_payload
                },
                archive_start[support],
                direction,
            )
            fork_vectors[name] = fork_contrast_vector(fork_payload, direction)

        allowed, stopped = f"allowed_{tag}", f"stopped_{tag}"
        true_archive = compact[stopped]["archive_support"]["conditional_consequence"][
            "true_fatal_minus_safe"
        ]
        true_forks = compact[stopped]["policy_forks"]["conditional_consequence"][
            "true_fatal_minus_safe"
        ]
        comparisons[str(step)] = {
            "archive_support": paired_difference(
                archive_vectors[allowed],
                archive_vectors[stopped],
                minimum_effect=0.05 * abs(true_archive),
                seed=Config().seed + 18_700 + step,
                samples=args.bootstraps,
            ),
            "policy_forks": paired_difference(
                fork_vectors[allowed],
                fork_vectors[stopped],
                minimum_effect=0.05 * abs(true_forks),
                seed=Config().seed + 18_800 + step,
                samples=args.bootstraps,
            ),
        }

    final_tag = f"{milestones[-1] // 1000:03d}k"
    validity = head_valid(heads, (f"allowed_{final_tag}", f"stopped_{final_tag}"))
    final = comparisons[str(milestones[-1])]
    both_rescue = all(row["material_improvement"] for row in final.values())
    both_equivalent = all(row["practically_equivalent"] for row in final.values())
    either_harmful = any(
        row["ci95"][1] < -row["minimum_meaningful_effect"]
        for row in final.values()
    )
    if not validity["passed"]:
        verdict = "invalid_outcome_head_did_not_generalize"
    elif both_rescue:
        verdict = "generated_latent_terminal_shaping_rescue"
    elif both_equivalent:
        verdict = "no_material_generated_latent_terminal_shaping_effect"
    elif either_harmful:
        verdict = "generated_latent_terminal_shaping_harmful_or_mixed"
    else:
        verdict = "mixed_or_inconclusive"

    adverse = {
        "description": (
            "earlier terminal BCE attached after Direct's generated agent-token "
            "path; it was not a direct generated-latent objective"
        ),
        "gradient_preflight": old_training["gradient_preflight"],
        "allowed_same_action_auc": old_probe["worlds"]["world_gradient"][
            "generated_latent_probe"
        ]["same_action"]["conditional"]["pooled_pair_auc"],
        "stopped_same_action_auc": old_probe["worlds"]["stopped_world_gradient"][
            "generated_latent_probe"
        ]["same_action"]["conditional"]["pooled_pair_auc"],
        "allowed_mse": old_probe["worlds"]["world_gradient"][
            "latent_prediction_error"
        ]["all"],
        "stopped_mse": old_probe["worlds"]["stopped_world_gradient"][
            "latent_prediction_error"
        ]["all"],
    }
    report = finite_json(
        {
            "contract": contract,
            "matching": {
                "shared_initial_world": True,
                "shared_initial_head": True,
                "stopped_matches_s78_at_every_milestone": all(
                    training["control_matches"].values()
                ),
                "allowed_and_stopped_worlds_differ": training["worlds_differ"],
                "gradient_preflight": training["gradient_preflight"],
            },
            "head_validity": validity,
            "cells": compact,
            "comparisons": comparisons,
            "adverse_control": adverse,
            "verdict": verdict,
        }
    )
    args.out.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out / "report.json", report)
    print(f"verdict: {verdict}", flush=True)
    print(f"complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
