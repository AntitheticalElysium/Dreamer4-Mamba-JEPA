"""Verify semantic equality of two completed Stage-M1 experiment directories."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import file_sha256


def _scientific_report(report: dict) -> dict:
    """Remove only explicitly operational, machine-dependent report fields."""
    return {
        "format": report["format"],
        "status": report["status"],
        "claim_boundary": report["claim_boundary"],
        "provenance": report["provenance"],
        "pairing": report["pairing"],
        "decision": report["decision"],
        "arms": {
            name: {
                "arm": arm["arm"],
                "config": arm["config"],
                "parameters": arm["parameters"],
                "training": {
                    key: value
                    for key, value in arm["training"].items()
                    if key
                    not in {
                        "seconds",
                        "updates_per_second",
                        "peak_vram_bytes",
                    }
                },
                "evaluation": arm["evaluation"],
                "world_state_sha256": arm["checkpoint"][
                    "world_state_sha256"
                ],
                "strict_reload_step": arm["checkpoint"][
                    "strict_reload_step"
                ],
            }
            for name, arm in report["arms"].items()
        },
    }


def _equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return (
            isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)):
        return (
            isinstance(right, type(left))
            and len(left) == len(right)
            and all(_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, float) and isinstance(right, float):
        if math.isnan(left) and math.isnan(right):
            return True
    return left == right


def verify(first: Path, second: Path) -> dict:
    reports = [
        json.loads((directory / "report.json").read_text())
        for directory in (first, second)
    ]
    result = {
        "first": str(first),
        "second": str(second),
        "scientific_reports_exact": (
            _scientific_report(reports[0])
            == _scientific_report(reports[1])
        ),
        "arms": {},
    }
    for arm in ("t_base", "m_base"):
        paths = [directory / f"{arm}.pt" for directory in (first, second)]
        payloads = [
            torch.load(path, map_location="cpu", weights_only=False)
            for path in paths
        ]
        result["arms"][arm.upper().replace("_", "-")] = {
            "decoded_checkpoint_payloads_exact": _equal(
                payloads[0], payloads[1]
            ),
            "world_state_sha256": [
                reports[index]["arms"][arm.upper().replace("_", "-")][
                    "checkpoint"
                ]["world_state_sha256"]
                for index in (0, 1)
            ],
            "raw_checkpoint_sha256": [
                file_sha256(path) for path in paths
            ],
        }
    result["pass"] = (
        result["scientific_reports_exact"]
        and all(
            arm["decoded_checkpoint_payloads_exact"]
            and len(set(arm["world_state_sha256"])) == 1
            for arm in result["arms"].values()
        )
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    result = verify(args.first, args.second)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
