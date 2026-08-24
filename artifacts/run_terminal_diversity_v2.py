"""Wait for support-v2, then run the fixed-exposure diversity ladder end to end."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from artifacts.phase1b_diagnostic_common import file_digest


def execute(arguments: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(arguments), flush=True)
    subprocess.run(arguments, check=True, env=env)


def fork_start_cache(output: Path) -> Path:
    """Keep derived fork starts inside the experiment that owns them."""
    return output / "delta_evaluation" / "fork_start_latents.pt"


def completed_evaluation(
    output: Path,
    expected_inputs: dict,
    expected_worlds: list[str],
) -> bool:
    """Accept a completed evaluation only when its full contract is still current."""
    contract_path = output / "contract.json"
    report_path = output / "report.json"
    if not contract_path.exists() or not report_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text())
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if report.get("contract") != contract:
        return False
    if contract.get("inputs") != expected_inputs:
        return False
    if list(report.get("worlds", {})) != expected_worlds:
        return False
    for source, digest in contract.get("implementation", {}).items():
        path = Path(source)
        if not path.exists() or file_digest(path) != digest:
            return False
    return all((output / "features" / f"{name}.pt").exists() for name in expected_worlds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--support", type=Path, default=Path("artifacts/craftax_support_v2"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/terminal_diversity_v2"))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--sizes", type=int, nargs="+", default=(300, 900, 3800))
    parser.add_argument("--replicates", type=int, default=2)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str((args.out / "matplotlib-cache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    python = sys.executable
    phase1a = Path("artifacts/stage_a_terminalfix/phase1a.pt")
    reference = Path("artifacts/stage_a_terminalfix/direct-attention.1b.pt")
    base_prepared = Path("artifacts/phase1b_archive_geometry/preparation/prepared.pt")
    trajectory_phase2 = Path("artifacts/stage_a_s76_terminal_only/direct-attention.2.pt")
    forks = Path("artifacts/phase1b_causal_diagnostics/paired_trajectory_actions/paired_trajectory_forks.pt")
    fork_starts = fork_start_cache(args.out)

    manifest_path = args.support / "manifest.json"
    while True:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("complete"):
                print(
                    f"support-v2 complete: {manifest['terminal_episodes']} terminal "
                    f"episodes in {manifest['episodes']} retained rollouts",
                    flush=True,
                )
                break
            print(
                f"waiting for support-v2: {manifest.get('terminal_episodes', 0)}/"
                f"{manifest.get('target_terminal_episodes', '?')}",
                flush=True,
            )
        else:
            print("waiting for support-v2 manifest", flush=True)
        time.sleep(args.poll_seconds)

    preparation = args.out / "preparation"
    execute(
        [
            python,
            "-m",
            "artifacts.prepare_terminal_diversity_evaluation",
            "--phase1a",
            str(phase1a),
            "--base-prepared",
            str(base_prepared),
            "--support",
            str(args.support),
            "--cache",
            str(preparation / "dev_latent_cache"),
            "--out",
            str(preparation),
        ],
        env,
    )
    prepared = preparation / "prepared.pt"
    training = args.out / "training"
    execute(
        [
            python,
            "-m",
            "artifacts.train_terminal_diversity_scaling",
            "--phase1a",
            str(phase1a),
            "--reference-phase1b",
            str(reference),
            "--support",
            str(args.support),
            "--cache",
            str(args.out / "train_latent_cache"),
            "--out",
            str(training),
            "--sizes",
            *(str(value) for value in args.sizes),
            "--replicates",
            str(args.replicates),
        ],
        env,
    )
    training_report = json.loads((training / "training_report.json").read_text())
    milestones = training_report["common"]["milestones"]
    cells = training_report["cells"]

    archive_arguments = []
    archive_delta_arguments = []
    policy_arguments = []
    policy_delta_arguments = []
    for cell in cells:
        for step in milestones:
            name = f"{cell}_{step // 1000:03d}k"
            checkpoint = training / cell / f"world_{step:06d}.pt"
            archive_arguments.extend(["--world", name, str(checkpoint)])
            archive_delta_arguments.extend(["--archive-world", name])
        endpoint = milestones[-1]
        name = f"{cell}_{endpoint // 1000:03d}k"
        checkpoint = training / cell / f"world_{endpoint:06d}.pt"
        policy_arguments.extend(["--world", name, str(checkpoint)])
        policy_delta_arguments.extend(["--policy-world", name])

    archive_out = args.out / "archive_evaluation"
    archive_worlds = {
        archive_arguments[index + 1]: file_digest(Path(archive_arguments[index + 2]))
        for index in range(0, len(archive_arguments), 3)
    }
    archive_inputs = {"prepared": file_digest(prepared), "worlds": archive_worlds}
    if completed_evaluation(archive_out, archive_inputs, list(archive_worlds)):
        print(f"verified complete: {archive_out / 'report.json'}", flush=True)
    else:
        execute(
            [
                python,
                "-m",
                "artifacts.evaluate_phase1b_archive_geometry",
                "--prepared",
                str(prepared),
                *archive_arguments,
                "--out",
                str(archive_out),
            ],
            env,
        )
    policy_out = args.out / "policy_trajectory_evaluation"
    policy_worlds = {
        policy_arguments[index + 1]: file_digest(Path(policy_arguments[index + 2]))
        for index in range(0, len(policy_arguments), 3)
    }
    policy_inputs = {
        "phase1a": file_digest(phase1a),
        "trajectory_phase2": file_digest(trajectory_phase2),
        "forks": file_digest(forks),
        "worlds": policy_worlds,
    }
    if completed_evaluation(policy_out, policy_inputs, list(policy_worlds)):
        print(f"verified complete: {policy_out / 'report.json'}", flush=True)
    else:
        execute(
            [
                python,
                "-m",
                "artifacts.probe_paired_trajectory_forks",
                "--phase1a",
                str(phase1a),
                "--trajectory-phase2",
                str(trajectory_phase2),
                "--forks",
                str(forks),
                *policy_arguments,
                "--out",
                str(policy_out),
            ],
            env,
        )
    delta_out = args.out / "delta_evaluation"
    execute(
        [
            python,
            "-m",
            "artifacts.evaluate_fatality_direction_delta",
            "--prepared",
            str(prepared),
            "--archive-features",
            str(archive_out / "features"),
            "--policy-features",
            str(policy_out / "features"),
            "--phase1a",
            str(phase1a),
            "--trajectory-phase2",
            str(trajectory_phase2),
            "--forks",
            str(forks),
            "--fork-starts",
            str(fork_starts),
            *archive_delta_arguments,
            *policy_delta_arguments,
            "--out",
            str(delta_out),
        ],
        env,
    )
    execute(
        [
            python,
            "-m",
            "artifacts.summarize_terminal_diversity_scaling",
            "--training",
            str(training / "training_report.json"),
            "--prepared",
            str(prepared),
            "--archive",
            str(archive_out / "report.json"),
            "--archive-features",
            str(archive_out / "features"),
            "--policy",
            str(policy_out / "report.json"),
            "--delta",
            str(delta_out / "report.json"),
            "--out",
            str(args.out / "report.json"),
        ],
        env,
    )
    print(f"terminal diversity v2 complete: {args.out / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
