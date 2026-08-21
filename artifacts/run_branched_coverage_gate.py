"""Run branched competent-state collection and its supervised transfer gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def execute(arguments: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("artifacts/branched_coverage_gate"))
    parser.add_argument("--seed-start", type=int, default=14_000)
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--probe-steps", type=int, default=2000)
    parser.add_argument("--probe-seeds", type=int, default=3)
    args = parser.parse_args()
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    python = sys.executable
    phase1a = Path("artifacts/stage_a_terminalfix/phase1a.pt")
    phase2 = Path("artifacts/stage_a_s76_terminal_only/direct-attention.2.pt")
    forks = Path("artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt")
    fork_starts = Path("artifacts/terminal_diversity_diagnostics/fork_start_latents.pt")
    reference = Path("artifacts/identifiability_gate_v2/v2_scaling/report.json")
    reference_features = Path("artifacts/identifiability_gate_v2/v2_scaling/features.pt")
    required = (phase1a, phase2, forks, fork_starts, reference, reference_features)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        parser.error(f"missing prerequisite artifacts: {missing}")

    collection = args.out / "collection"
    execute([
        python,
        "-m",
        "artifacts.collect_branched_policy_states",
        "--phase1a",
        str(phase1a),
        "--trajectory-phase2",
        str(phase2),
        "--out",
        str(collection),
        "--seed-start",
        str(args.seed_start),
        "--seeds",
        str(args.seeds),
        "--limit",
        str(args.limit),
    ], env)
    execute([
        python,
        "-m",
        "artifacts.probe_branched_policy_states",
        "--collection",
        str(collection),
        "--fork-starts",
        str(fork_starts),
        "--forks",
        str(forks),
        "--reference",
        str(reference),
        "--reference-features",
        str(reference_features),
        "--out",
        str(args.out / "probe"),
        "--steps",
        str(args.probe_steps),
        "--seeds",
        str(args.probe_seeds),
    ], env)


if __name__ == "__main__":
    main()
