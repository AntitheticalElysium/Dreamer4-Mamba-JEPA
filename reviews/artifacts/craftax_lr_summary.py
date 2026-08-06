"""Aggregate the preregistered Craftax LR/SIGReg queue without interpreting it."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.checkpoint import (
    file_sha256, implementation_sha256,
)

SEEDS = (20260727, 20260728, 20260729)
ORACLE_METRICS = (
    "inventory_linear_mean",
    "inventory_nonlinear_mean",
    "vitals_linear_mean",
    "vitals_nonlinear_mean",
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _mean_sd(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "sample_sd": (
            float(array.std(ddof=1)) if array.size > 1 else None
        ),
        "values": [float(value) for value in array],
    }


def _grid_cell(path: Path) -> dict:
    payload = _read(path)
    steps = sorted(int(step) for step in payload["curve"])
    first = payload["curve"][str(steps[0])]
    last = payload["curve"][str(steps[-1])]
    delta = {
        key: float(last["summary"][key] - first["summary"][key])
        for key in ORACLE_METRICS
    }
    delta["dev_cosine"] = float(
        last["dev_cosine"] - first["dev_cosine"]
    )
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "first_step": steps[0],
        "last_step": steps[-1],
        "delta": delta,
        "final": {
            **{key: float(last["summary"][key]) for key in ORACLE_METRICS},
            "dev_cosine": float(last["dev_cosine"]),
            "verdicts": last["summary"]["verdicts"],
            "audit_pass": bool(last["full_report"]["audit"]["pass"]),
        },
        "config": payload["config"],
        "checkpoint": payload.get("checkpoint"),
    }


def _grid_group(
    grid_dir: Path, tags: list[str], missing: list[str]
) -> dict:
    cells = []
    for tag in tags:
        path = grid_dir / f"{tag}.json"
        if path.is_file():
            cells.append(_grid_cell(path))
        else:
            missing.append(str(path))
    aggregate = {}
    if cells:
        for key in (*ORACLE_METRICS, "dev_cosine"):
            aggregate[key] = _mean_sd(
                [cell["delta"][key] for cell in cells]
            )
    return {"cells": cells, "delta_aggregate": aggregate}


def _arm_metrics(path: Path) -> dict:
    report = _read(path)
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "world": {
            key: report["world"].get(key)
            for key in (
                "updates", "learning_rate", "encoder_learning_rate",
                "final_jepa_loss", "train_cosine_last_100",
                "online_std_last_100", "dev_cosine", "seconds",
                "checkpoint_sha256",
            )
        },
        "bc": {
            key: report["bc"].get(key)
            for key in (
                "updates", "first_loss", "last_loss",
                "dev_action_accuracy", "seconds",
            )
        },
        "imagination": {
            "updates": report["imagination"].get("updates"),
            "seconds": report["imagination"].get("seconds"),
            "final": report["imagination"].get("final"),
        },
    }


def _numeric_delta(slow, full):
    if isinstance(slow, dict) and isinstance(full, dict):
        return {
            key: _numeric_delta(slow[key], full[key])
            for key in slow.keys() & full.keys()
        }
    if (
        isinstance(slow, (int, float))
        and not isinstance(slow, bool)
        and isinstance(full, (int, float))
        and not isinstance(full, bool)
    ):
        return float(slow - full)
    return None


def _oracle_summary(path: Path) -> dict:
    payload = _read(path)
    arms = {}
    for key, report in payload["arms"].items():
        groups = {}
        for group in ("vitals", "inventory"):
            entries = report[group]["per_target"]
            counts: dict[str, int] = {}
            for entry in entries.values():
                verdict = entry["verdict"]
                counts[verdict] = counts.get(verdict, 0) + 1
            groups[group] = {
                "verdicts": counts,
                "latent_linear_mean": float(np.mean([
                    entry["latent_linear_r2"] for entry in entries.values()
                ])),
                "latent_nonlinear_mean": float(np.mean([
                    entry["latent_nonlinear_r2"] for entry in entries.values()
                ])),
            }
        arms[key] = {
            "audit_pass": bool(report["audit"]["pass"]),
            "groups": groups,
            "checkpoint": report.get("checkpoint"),
        }
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "arms": arms,
    }


def _executed_summary(path: Path) -> dict:
    payload = _read(path)
    result = payload["result"]
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "claim_boundary": payload["claim_boundary"],
        "run": payload["run"],
        "evaluation": payload["evaluation"],
        "summary": result["summary"],
        "actor_minus_bc": result["actor_minus_bc"],
        "actor_minus_bc_ci": result["actor_minus_bc_ci"],
        "actor_beats_bc": result["actor_beats_bc"],
        "actor_minus_random": result["actor_minus_random"],
        "actor_minus_random_ci": result["actor_minus_random_ci"],
        "actor_beats_random": result["actor_beats_random"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--grid-dir", type=Path,
        default=REPO_ROOT / "reviews/artifacts/lr_objective_grid",
    )
    parser.add_argument(
        "--baseline-dir", type=Path,
        default=REPO_ROOT / "outputs/d4_mamba_jepa/craftax_expert_v1",
    )
    parser.add_argument(
        "--slow-dir", type=Path,
        default=REPO_ROOT / "outputs/d4_mamba_jepa/craftax_slowenc_v1",
    )
    parser.add_argument(
        "--oracle", type=Path,
        default=REPO_ROOT / "reviews/artifacts/craftax_slowenc_oracle.json",
    )
    parser.add_argument(
        "--executed-dir", type=Path,
        default=REPO_ROOT / "reviews/artifacts/craftax_executed_lr",
    )
    parser.add_argument(
        "--output", type=Path,
        default=REPO_ROOT
        / "reviews/artifacts/craftax_lr_experiment_summary.json",
    )
    args = parser.parse_args()
    missing: list[str] = []

    groups = {
        "ema_actual_full": _grid_group(
            args.grid_dir,
            [f"transfer_ema_full_s{seed}" for seed in SEEDS],
            missing,
        ),
        "ema_actual_slow": _grid_group(
            args.grid_dir,
            [f"transfer_ema_slow_s{seed}" for seed in SEEDS],
            missing,
        ),
        "sigreg_clean_full": _grid_group(
            args.grid_dir,
            [f"sigreg_clean_full_s{seed}" for seed in SEEDS],
            missing,
        ),
        "sigreg_clean_slow": _grid_group(
            args.grid_dir,
            [f"sigreg_clean_slow_s{seed}" for seed in SEEDS],
            missing,
        ),
        "ema_clean_spatial_slow": _grid_group(
            args.grid_dir,
            [f"spatial_slow_s{seed}" for seed in SEEDS],
            missing,
        ),
        "sigreg_actual_full": _grid_group(
            args.grid_dir,
            ["sigreg_actual_full_s20260727"],
            missing,
        ),
        "sigreg_actual_slow": _grid_group(
            args.grid_dir,
            ["sigreg_actual_slow_s20260727"],
            missing,
        ),
    }

    full_pipeline = {}
    for arm in ("t_jepa", "m_jepa"):
        baseline_path = args.baseline_dir / arm / "arm_report.json"
        slow_path = args.slow_dir / arm / "arm_report.json"
        if baseline_path.is_file() and slow_path.is_file():
            full = _arm_metrics(baseline_path)
            slow = _arm_metrics(slow_path)
            full_pipeline[arm] = {
                "full_encoder_lr": full,
                "slow_encoder_lr": slow,
                "slow_minus_full_numeric": _numeric_delta(slow, full),
            }
        else:
            for path in (baseline_path, slow_path):
                if not path.is_file():
                    missing.append(str(path))

    oracle = None
    if args.oracle.is_file():
        oracle = _oracle_summary(args.oracle)
    else:
        missing.append(str(args.oracle))

    executed = {}
    for label in ("full_t", "full_m", "slow_t", "slow_m"):
        path = args.executed_dir / f"{label}.json"
        if path.is_file():
            executed[label] = _executed_summary(path)
        else:
            missing.append(str(path))

    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    payload = {
        "format": "craftax_lr_experiment_summary_v1",
        "status": "completed" if not missing else "partial",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_head": git_head,
            "implementation_sha256": implementation_sha256(),
            "runner_sha256": file_sha256(Path(__file__)),
        },
        "grid": groups,
        "full_pipeline": full_pipeline,
        "oracle": oracle,
        "executed": executed,
        "missing": sorted(set(missing)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(
        f"wrote {args.output}: status={payload['status']} "
        f"missing={len(payload['missing'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
