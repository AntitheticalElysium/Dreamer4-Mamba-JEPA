"""Run the Direct-topology and Flow-diversity attribution package."""

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
        "--out", type=Path, default=Path("artifacts/predictor_flow_attribution")
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
    direct_reference = Path("artifacts/stage_a_terminalfix/direct-attention.1b.pt")
    flow_reference = Path("artifacts/stage_a_terminalfix/flow-attention.1b.pt")
    direct_control = Path("artifacts/terminal_diversity_v2/training/k7501_r0")
    prepared = Path("artifacts/terminal_diversity_v2/preparation/prepared.pt")
    trajectory_phase2 = Path(
        "artifacts/stage_a_s76_terminal_only/direct-attention.2.pt"
    )
    forks = Path(
        "artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/"
        "paired_trajectory_forks.pt"
    )
    training = args.out / "training"

    training_command = [
        python,
        "-m",
        "artifacts.train_predictor_flow_attribution",
        "--phase1a",
        str(phase1a),
        "--direct-reference",
        str(direct_reference),
        "--flow-reference",
        str(flow_reference),
        "--support",
        str(args.support),
        "--cache",
        "artifacts/terminal_diversity_v2/train_latent_cache",
        "--out",
        str(training),
    ]
    if args.smoke:
        training_command.extend(
            ["--smoke", "--steps", "2", "--milestones", "1", "2"]
        )
    else:
        training_command.extend(["--direct-control", str(direct_control)])
    execute(training_command, env)
    if args.smoke:
        print("smoke complete; full evaluation intentionally not launched", flush=True)
        return

    training_report = json.loads((training / "training_report.json").read_text())
    endpoint = training_report["common"]["milestones"][-1]
    specifications = []
    for name in training_report["cells"]:
        row = training_report["reports"][name]
        if "external_control" in row:
            checkpoint = Path(row["external_control"]["milestones"][str(endpoint)])
            transition, predictor = "direct", "current"
        else:
            checkpoint = training / name / f"world_{endpoint:06d}.pt"
            transition = row["contract"]["cell"]["transition"]
            predictor = row["contract"]["cell"]["predictor"]
        specifications.extend(
            ["--world", name, transition, predictor, str(checkpoint)]
        )

    archive = args.out / "archive_evaluation"
    execute(
        [
            python,
            "-m",
            "artifacts.evaluate_predictor_flow_archive",
            "--prepared",
            str(prepared),
            *specifications,
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
            *specifications,
            "--out",
            str(fork_out),
        ],
        env,
    )
    execute(
        [
            python,
            "-m",
            "artifacts.summarize_predictor_flow_attribution",
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
            "--out",
            str(args.out),
        ],
        env,
    )


if __name__ == "__main__":
    main()
