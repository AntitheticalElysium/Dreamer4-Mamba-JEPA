"""Run the representation ladder, then the support-v2 coverage ladder."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def execute(arguments: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env=env)


def main() -> None:
    python = sys.executable
    root = Path("artifacts/identifiability_gate_v2")
    root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["MPLCONFIGDIR"] = str((root / "matplotlib-cache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    phase1a = "artifacts/stage_a_terminalfix/phase1a.pt"
    phase2 = "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"
    forks = "artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt"
    starts = "artifacts/terminal_diversity_diagnostics/fork_start_latents.pt"
    prepared = "artifacts/phase1b_archive_geometry/preparation/prepared.pt"
    policy_features = "artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/probe/features/baseline_020k.pt"
    execute([
        python, "-m", "artifacts.diagnose_fork_representation_identifiability",
        "--phase1a", phase1a, "--trajectory-phase2", phase2,
        "--forks", forks, "--fixed-z", starts,
        "--out", str(root / "fork_representations"),
    ], env)
    execute([
        python, "-m", "artifacts.diagnose_v2_fatality_identifiability",
        "--phase1a", phase1a,
        "--support", "artifacts/craftax_support_v2",
        "--prepared", prepared,
        "--policy-features", policy_features,
        "--fork-starts", starts,
        "--forks", forks,
        "--out", str(root / "v2_scaling"),
    ], env)
    print(f"identifiability gate complete: {root}", flush=True)


if __name__ == "__main__":
    main()
