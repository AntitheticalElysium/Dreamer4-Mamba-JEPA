"""Run paired executed Craftax evaluation for one saved world/BC/actor arm.

This is an exploratory fresh-seed diagnostic, not a newly selected sealed tier.
It verifies every checkpoint digest and policy/world pairing before interacting
with the environment, then reports the official geometric-mean Crafter score,
episode return, and paired actor-minus-BC/random score intervals.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from d4_mamba_jepa.common import load_bc_policy
from d4_mamba_jepa.checkpoint import (
    file_sha256, implementation_sha256, load_checkpoint,
)
from d4_mamba_jepa.craftax_achievement import evaluate_craftax_achievement
from d4_mamba_jepa.source import craftax_source_report


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("t_jepa", "m_jepa"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=2_500)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--mode", choices=("sample", "greedy"), default="sample")
    parser.add_argument("--policy-seed-base", type=int, default=7_000_000)
    parser.add_argument(
        "--allow-implementation-drift", action="store_true",
        help=(
            "permit loading an older, digest-pinned checkpoint after a verified "
            "default-path-preserving implementation change"
        ),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.episodes < 2:
        raise ValueError("at least two episodes are required for paired evaluation")
    if args.max_steps < 1:
        raise ValueError("max_steps must be positive")

    device = torch.device(args.device)
    arm_dir = args.run_dir / args.arm
    world_report = _read_json(arm_dir / "world_report.json")
    bc_report = _read_json(arm_dir / "bc_report.json")
    actor_report = _read_json(arm_dir / "imagination_report.json")
    world_sha = world_report["world_checkpoint_sha256"]
    bc_sha = bc_report["bc_checkpoint_sha256"]
    actor_sha = actor_report["actor_checkpoint_sha256"]

    world, _, world_payload = load_checkpoint(
        arm_dir / "world.pt",
        device=device,
        expected_sha256=world_sha,
        strict_implementation=not args.allow_implementation_drift,
    )
    bc, _ = load_bc_policy(
        arm_dir / "bc.pt",
        expected_sha256=bc_sha,
        expected_world_sha256=world_sha,
        device=device,
    )
    actor, _ = load_bc_policy(
        arm_dir / "actor.pt",
        expected_sha256=actor_sha,
        expected_world_sha256=world_sha,
        device=device,
    )
    world.eval()
    bc.eval()
    actor.eval()

    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    print(
        f"run={args.run_dir} arm={args.arm} seeds={seeds[0]}..{seeds[-1]} "
        f"max_steps={args.max_steps} mode={args.mode} device={device}",
        flush=True,
    )
    started = time.perf_counter()
    result = evaluate_craftax_achievement(
        world=world,
        bc_policy=bc,
        actor_policy=actor,
        seeds=seeds,
        context=args.context,
        max_steps=args.max_steps,
        device=device,
        policy_seed_base=args.policy_seed_base,
        mode=args.mode,
        progress=True,
    )
    seconds = time.perf_counter() - started
    payload = {
        "format": "d4_mamba_jepa_craftax_achievement_eval_v1",
        "status": "completed",
        "claim_boundary": (
            "fresh fixed exploratory environment seeds, disjoint from replay "
            "seeds 0..319 and probe seed family 90000; official metric but not "
            "a sealed holdout or a selection-valid confirmatory tier"
        ),
        "run": {
            "path": str(args.run_dir),
            "arm": args.arm,
            "world_checkpoint_sha256": world_sha,
            "bc_checkpoint_sha256": bc_sha,
            "actor_checkpoint_sha256": actor_sha,
            "stored_implementation_sha256": world_payload["provenance"][
                "implementation_sha256"
            ],
            "current_implementation_sha256": implementation_sha256(),
            "implementation_drift_allowed": args.allow_implementation_drift,
        },
        "evaluation": {
            "seeds": seeds,
            "context": args.context,
            "max_steps": args.max_steps,
            "mode": args.mode,
            "policy_seed_base": args.policy_seed_base,
            "seconds": seconds,
            "device": str(device),
        },
        "provenance": {
            "git_head": _git_head(),
            "runner_sha256": file_sha256(Path(__file__)),
            "craftax": craftax_source_report(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "world_file_sha256": file_sha256(arm_dir / "world.pt"),
            "bc_file_sha256": file_sha256(arm_dir / "bc.pt"),
            "actor_file_sha256": file_sha256(arm_dir / "actor.pt"),
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=float) + "\n"
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(
        f"actor-BC {result['actor_minus_bc']:+.3f} "
        f"{result['actor_minus_bc_ci']}; actor-random "
        f"{result['actor_minus_random']:+.3f} "
        f"{result['actor_minus_random_ci']}; {seconds / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
