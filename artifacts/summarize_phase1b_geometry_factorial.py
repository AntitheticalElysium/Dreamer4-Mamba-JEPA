"""Compact the matched Phase-1B factorial into endpoint effects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.phase1b_diagnostic_common import atomic_json, stored_state_digest


CELLS = (
    "ordinary_ordinary",
    "whitened_ordinary",
    "ordinary_terminal",
    "whitened_terminal",
)


def endpoint(archive: dict, policy: dict, name: str, pool: str) -> dict:
    logged = archive["worlds"][name]
    geometry = logged["paths"]["reset16"][pool]
    generated = policy["worlds"][name]["generated_latent_probe"]
    return {
        "archive_total_mse": geometry["total_mse"],
        "archive_direction_mse_over_variance": geometry[
            "direction_mse_over_target_variance"
        ],
        "archive_fixed_direction_auc": geometry["predicted_separation"]["auc"],
        "archive_fresh_probe_auc": logged["generated_probe"]["reset16"][pool]["auc"],
        "trajectory_action_auc": generated["trajectory_action"]["auc"],
        "other_16_actions_auc": generated["other_16_actions"]["auc"],
    }


def effects(values: dict[str, dict]) -> dict:
    output = {}
    for metric in next(iter(values.values())):
        oo = values["ordinary_ordinary"][metric]
        wo = values["whitened_ordinary"][metric]
        ot = values["ordinary_terminal"][metric]
        wt = values["whitened_terminal"][metric]
        output[metric] = {
            "whitening_at_ordinary_sampling": wo - oo,
            "whitening_at_terminal_sampling": wt - ot,
            "terminal_sampling_at_ordinary_metric": ot - oo,
            "terminal_sampling_at_whitened_metric": wt - wo,
            "factorial_interaction": wt - wo - ot + oo,
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    training = json.loads(args.training.read_text())
    archive = json.loads(args.archive.read_text())
    policy = json.loads(args.policy.read_text())
    if not all(training["invariants"].values()):
        raise AssertionError("training cells were not matched")

    reports = {
        name: json.loads(Path(path).read_text())
        for name, path in training["reports"].items()
    }
    expected_trained = set(CELLS) - {"ordinary_ordinary"}
    if set(reports) != expected_trained:
        raise ValueError(f"expected trained cells {sorted(expected_trained)}")
    for name, report in reports.items():
        expected_rows = report["contract"]["steps"] if name.endswith("_terminal") else 0
        if report["terminal_rows_scored"] != expected_rows:
            raise AssertionError(f"wrong terminal exposure in {name}")

    names = {
        f"{cell}_{step}"
        for cell in CELLS
        for step in ("005k", "020k")
    }
    if not names.issubset(archive["worlds"]) or not names.issubset(policy["worlds"]):
        raise ValueError("evaluation is missing a factorial cell or milestone")

    baseline_path = Path(
        archive["worlds"]["ordinary_ordinary_020k"]["checkpoint"]
    )
    baseline = torch.load(baseline_path, weights_only=False, map_location="cpu")
    baseline_digest = stored_state_digest(baseline["modules"]["part0"])
    reference_digest = next(iter(reports.values()))["reference"]["world_sha256"]
    if baseline_digest != reference_digest:
        raise AssertionError("ordinary factorial control is not the production world")

    by_step = {}
    for step in ("005k", "020k"):
        by_step[step] = {
            pool: {
                cell: endpoint(archive, policy, f"{cell}_{step}", pool)
                for cell in CELLS
            }
            for pool in ("combined", "support")
        }
    report = {
        "version": "phase1b-geometry-factorial-summary-v1",
        "matching": training["invariants"]
        | {"ordinary_control_matches_production_world": True},
        "interpretation": (
            "Effects are raw paired endpoint differences, not confidence intervals. "
            "For AUC positive is improvement; for MSE negative is improvement."
        ),
        "milestones": by_step,
        "endpoint_20k_effects": {
            pool: effects(by_step["020k"][pool])
            for pool in ("combined", "support")
        },
    }
    atomic_json(args.out, report)
    print(f"complete: {args.out}", flush=True)


if __name__ == "__main__":
    main()
