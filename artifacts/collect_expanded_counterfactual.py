"""Collect a larger evaluation-only all-action fork set on fresh seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from d4mj.config import Config
from d4mj.counterfactual import collect_outcome_forks, outcome_metrics


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--phase2", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=13_000)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--steps", default="0,15,40,80,110")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/counterfactual_expanded_s76"),
    )
    args = parser.parse_args()

    base = Config()
    model_config = replace(
        base,
        transition="direct",
        time_mixer="attention",
    )
    encoder, world, heads = load_models(
        args.phase1a, args.phase2, base, model_config
    )
    gate_config = replace(
        model_config,
        outcome_gate_seeds=tuple(
            range(args.seed_start, args.seed_start + args.seeds)
        ),
        outcome_gate_steps=tuple(
            int(value) for value in args.steps.split(",") if value
        ),
        outcome_gate_limit=args.limit,
    )

    forks = collect_outcome_forks(world, encoder, heads, gate_config)
    metrics = outcome_metrics(forks, heads, gate_config)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fork_path = args.out_dir / "direct-attention.outcome_forks.pt"
    torch.save(vars(forks), fork_path)
    manifest = {
        "format": "d4mj_expanded_counterfactual_v1",
        "evaluation_only": True,
        "arm": "direct-attention",
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "scheduled_steps": list(gate_config.outcome_gate_steps),
        "limit": args.limit,
        "phase1a": str(args.phase1a.resolve()),
        "phase2": str(args.phase2.resolve()),
        "phase1a_sha256": _digest(args.phase1a),
        "phase2_sha256": _digest(args.phase2),
        "fork_sha256": _digest(fork_path),
        "rows": int(len(forks.seed)),
        "terminal_opportunity_states": int(
            (forks.true_death.any(1) & (~forks.true_death).any(1)).sum()
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    print(json.dumps({"manifest": manifest, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
