"""Run the generated-latent outcome-shaping experiment end to end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def execute(arguments: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--support", type=Path, default=Path("artifacts/craftax_support_v2")
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("artifacts/generated_latent_outcome_shaping"),
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu"
    env["MPLCONFIGDIR"] = str((args.out / "matplotlib-cache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    python = sys.executable
    phase1a = Path("artifacts/stage_a_terminalfix/phase1a.pt")
    reference_phase1b = Path("artifacts/stage_a_terminalfix/direct-attention.1b.pt")
    control = Path("artifacts/terminal_diversity_v2/training/k7501_r0")
    prepared = Path("artifacts/terminal_diversity_v2/preparation/prepared.pt")
    trajectory_phase2 = Path(
        "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"
    )
    forks = Path(
        "artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/"
        "paired_trajectory_forks.pt"
    )
    training = args.out / "training"
    command = [
        python,
        "-m",
        "artifacts.train_generated_latent_outcome_shaping",
        "--phase1a",
        str(phase1a),
        "--reference-phase1b",
        str(reference_phase1b),
        "--control",
        str(control),
        "--support",
        str(args.support),
        "--cache",
        "artifacts/terminal_diversity_v2/train_latent_cache",
        "--out",
        str(training),
    ]
    if args.smoke:
        command.extend(["--smoke", "--steps", "2", "--milestones", "1", "2"])
    execute(command, env)
    if args.smoke:
        print("smoke complete; fixed DEV evaluation was not inspected", flush=True)
        return

    report = json.loads((training / "training_report.json").read_text())
    worlds, models = [], []
    for step in report["contract"]["milestones"]:
        tag = f"{step // 1000:03d}k"
        for variant in ("allowed", "stopped"):
            name = f"{variant}_{tag}"
            row = report["milestones"][variant][str(step)]
            worlds.extend(["--world", name, "direct", "current", row["world"]])
            models.extend(["--model", name, row["model"]])

    archive = args.out / "archive_evaluation"
    execute(
        [
            python,
            "-m",
            "artifacts.evaluate_predictor_flow_archive",
            "--prepared",
            str(prepared),
            *worlds,
            "--out",
            str(archive),
        ],
        env,
    )
    fork_out = args.out / "policy_fork_evaluation"
    execute(
        [
            python,
            "-m",
            "artifacts.evaluate_predictor_flow_forks",
            "--prepared",
            str(prepared),
            "--phase1a",
            str(phase1a),
            "--trajectory-phase2",
            str(trajectory_phase2),
            "--forks",
            str(forks),
            "--fork-starts",
            str(fork_out / "fork_start_latents.pt"),
            *worlds,
            "--out",
            str(fork_out),
        ],
        env,
    )
    head_out = args.out / "head_evaluation"
    execute(
        [
            python,
            "-m",
            "artifacts.evaluate_generated_latent_outcome_heads",
            "--archive-features",
            str(archive / "features"),
            "--fork-features",
            str(fork_out / "features"),
            *models,
            "--out",
            str(head_out),
        ],
        env,
    )
    execute(
        [
            python,
            "-m",
            "artifacts.summarize_generated_latent_outcome_shaping",
            "--training",
            str(training / "training_report.json"),
            "--prepared",
            str(prepared),
            "--archive",
            str(archive / "report.json"),
            "--archive-features",
            str(archive / "features"),
            "--forks",
            str(fork_out / "report.json"),
            "--fork-features",
            str(fork_out / "features"),
            "--heads",
            str(head_out / "report.json"),
            "--old-gradient-training",
            "artifacts/phase1b_causal_diagnostics/consequence_gradient/training_report.json",
            "--old-gradient-probe",
            "artifacts/phase1b_causal_diagnostics/consequence_gradient_probe/report.json",
            "--out",
            str(args.out),
        ],
        env,
    )


if __name__ == "__main__":
    main()
