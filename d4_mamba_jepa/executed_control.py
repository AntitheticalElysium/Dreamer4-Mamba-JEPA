"""Bounded, no-learning Crafter control check for a frozen world model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

import numpy as np
import torch

from .checkpoint import file_sha256, load_checkpoint
from .data import CrafterAdapter
from .rollout import categorical_random_shooting, shortcut_schedule
from .source import source_report


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "outputs/d4_mamba_jepa/preflight_t_base_5k/world_t_base.pt"
)
DEFAULT_CHECKPOINT_SHA256 = (
    "6d4a2a18ed968ab29b0ef32d02f656284647b50714b25d54abfd90884ed079e4"
)
FORMAT = "d4_mamba_jepa_executed_control_v1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _self_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(x) != len(y):
        return None
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _crafter_score(episodes: list[dict]) -> tuple[float, dict[str, float]]:
    """Official geometric-mean formula, on this deliberately tiny sample."""
    names = sorted(
        {
            name
            for episode in episodes
            for name in episode["achievements"]
        }
    )
    success_rates = {
        name: 100.0
        * sum(episode["achievements"].get(name, 0) > 0 for episode in episodes)
        / len(episodes)
        for name in names
    }
    score = (
        math.exp(
            sum(math.log1p(rate) for rate in success_rates.values())
            / len(success_rates)
        )
        - 1.0
        if success_rates
        else 0.0
    )
    return float(score), success_rates


@torch.inference_mode()
def _planner_action(
    world,
    *,
    observations: list[np.ndarray],
    led_to_actions: list[int],
    context: int,
    horizon: int,
    candidates: int,
    schedule: dict,
    discount: float,
    generator: torch.Generator,
    device: torch.device,
) -> tuple[int, dict]:
    frames = torch.from_numpy(
        np.stack(observations[-context:])
    )[None].to(device)
    action_context = torch.tensor(
        led_to_actions[-context:],
        device=device,
        dtype=torch.long,
    )[None]
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        packed = world.encode_frames(frames, frozen=True).packed
        result = categorical_random_shooting(
            world,
            context_packed=packed,
            context_led_to_actions=action_context,
            horizon=horizon,
            candidates=candidates,
            schedule=schedule,
            discount=discount,
            use_cache=True,
            generator=generator,
        )
    scores = result.scores.float().cpu()
    return result.action, {
        "selected_score": result.score,
        "candidate_score_mean": float(scores.mean().item()),
        "candidate_score_std": float(scores.std(unbiased=False).item()),
        "candidate_score_range": float(
            (scores.max() - scores.min()).item()
        ),
    }


def _run_episode(
    *,
    world,
    policy: str,
    env_seed: int,
    device: torch.device,
    max_steps: int,
    context: int,
    horizon: int,
    candidates: int,
    denoise_steps: int,
    discount: float,
    policy_seed: int,
) -> dict:
    environment = CrafterAdapter(seed=env_seed)
    observation = environment.reset()
    observations = [observation]
    led_to_actions = [-1]
    random_rng = np.random.default_rng(policy_seed)
    planner_rng = torch.Generator(device=device).manual_seed(policy_seed)
    schedule = shortcut_schedule(world.cfg.k_max, denoise_steps)
    actions: list[int] = []
    rewards: list[float] = []
    selected_scores: list[float] = []
    candidate_stds: list[float] = []
    candidate_ranges: list[float] = []
    final_info: dict = {"achievements": {}}

    start = time.perf_counter()
    for _ in range(max_steps):
        if policy == "random":
            action = int(random_rng.integers(world.cfg.n_actions))
            planner_info = None
        elif policy == "planner":
            action, planner_info = _planner_action(
                world,
                observations=observations,
                led_to_actions=led_to_actions,
                context=min(context, len(observations)),
                horizon=horizon,
                candidates=candidates,
                schedule=schedule,
                discount=discount,
                generator=planner_rng,
                device=device,
            )
        else:
            raise ValueError(f"unsupported policy {policy!r}")

        observation, reward, continuation, final_info = environment.step(action)
        observations.append(observation)
        led_to_actions.append(action)
        actions.append(action)
        rewards.append(float(reward))
        if planner_info is not None:
            selected_scores.append(planner_info["selected_score"])
            candidate_stds.append(planner_info["candidate_score_std"])
            candidate_ranges.append(planner_info["candidate_score_range"])
        if continuation == 0.0:
            break

    elapsed = time.perf_counter() - start
    histogram = Counter(actions)
    probabilities = np.asarray(
        [histogram.get(action, 0) / len(actions) for action in range(world.cfg.n_actions)],
        dtype=np.float64,
    )
    positive = probabilities[probabilities > 0]
    entropy = float(-(positive * np.log(positive)).sum())
    return {
        "policy": policy,
        "env_seed": env_seed,
        "policy_seed": policy_seed,
        "steps": len(actions),
        "terminated": len(actions) < max_steps,
        "return": float(sum(rewards)),
        "nonzero_reward_steps": int(np.count_nonzero(rewards)),
        "achievement_count": int(
            sum(value > 0 for value in final_info["achievements"].values())
        ),
        "achievements": {
            name: int(value)
            for name, value in final_info["achievements"].items()
        },
        "action_histogram": {
            str(action): int(histogram.get(action, 0))
            for action in range(world.cfg.n_actions)
        },
        "action_entropy_nats": entropy,
        "max_action_fraction": float(probabilities.max()),
        "planner": (
            {
                "mean_selected_predicted_return": float(
                    np.mean(selected_scores)
                ),
                "mean_candidate_score_std": float(np.mean(candidate_stds)),
                "mean_candidate_score_range": float(
                    np.mean(candidate_ranges)
                ),
                "predicted_return_vs_immediate_reward_pearson": _pearson(
                    selected_scores, rewards
                ),
            }
            if selected_scores
            else None
        ),
        "wall_seconds": elapsed,
        "steps_per_second": len(actions) / max(elapsed, 1e-12),
    }


def _summarize(rows: list[dict]) -> dict:
    score, success_rates = _crafter_score(rows)
    return {
        "episodes": len(rows),
        "mean_return": float(np.mean([row["return"] for row in rows])),
        "mean_achievements": float(
            np.mean([row["achievement_count"] for row in rows])
        ),
        "mean_length": float(np.mean([row["steps"] for row in rows])),
        "mean_max_action_fraction": float(
            np.mean([row["max_action_fraction"] for row in rows])
        ),
        "crafter_score_percent_tiny_truncated_sample": score,
        "success_rates_percent": success_rates,
        "total_wall_seconds": float(sum(row["wall_seconds"] for row in rows)),
    }


def run(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    output: Path,
    seeds: list[int],
    device: torch.device,
    max_steps: int,
    context: int,
    horizon: int,
    candidates: int,
    denoise_steps: int,
    discount: float,
    policy_seed_base: int,
) -> dict:
    world, _, checkpoint_payload = load_checkpoint(
        checkpoint,
        device=device,
        expected_sha256=checkpoint_sha256,
    )
    world.eval()
    if candidates < world.cfg.n_actions:
        raise ValueError("candidate count must cover all first actions")

    rows = []
    for policy_index, policy in enumerate(("random", "planner")):
        for env_seed in seeds:
            rows.append(
                _run_episode(
                    world=world,
                    policy=policy,
                    env_seed=env_seed,
                    device=device,
                    max_steps=max_steps,
                    context=context,
                    horizon=horizon,
                    candidates=candidates,
                    denoise_steps=denoise_steps,
                    discount=discount,
                    policy_seed=(
                        policy_seed_base + 10_000 * policy_index + env_seed
                    ),
                )
            )

    random_rows = [row for row in rows if row["policy"] == "random"]
    planner_rows = [row for row in rows if row["policy"] == "planner"]
    random_by_seed = {row["env_seed"]: row for row in random_rows}
    planner_by_seed = {row["env_seed"]: row for row in planner_rows}
    paired = [
        {
            "env_seed": seed,
            "planner_minus_random_return": (
                planner_by_seed[seed]["return"] - random_by_seed[seed]["return"]
            ),
            "planner_minus_random_achievements": (
                planner_by_seed[seed]["achievement_count"]
                - random_by_seed[seed]["achievement_count"]
            ),
            "planner_minus_random_length": (
                planner_by_seed[seed]["steps"] - random_by_seed[seed]["steps"]
            ),
        }
        for seed in seeds
    ]
    payload = {
        "format": FORMAT,
        "status": "completed",
        "claim_boundary": (
            "tiny truncated executed preflight; not an official Crafter "
            "evaluation, not checkpoint selection, and not evidence of "
            "architecture superiority"
        ),
        "protocol": {
            "seeds": seeds,
            "policies": ["uniform_random", "categorical_random_shooting"],
            "max_steps": max_steps,
            "context": context,
            "horizon": horizon,
            "candidates": candidates,
            "denoise_steps": denoise_steps,
            "discount": discount,
            "policy_seed_base": policy_seed_base,
            "execute_first_action_only": True,
            "learning": False,
        },
        "provenance": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_step": checkpoint_payload["step"],
            "checkpoint_implementation_sha256": checkpoint_payload[
                "provenance"
            ]["implementation_sha256"],
            "evaluation_implementation_sha256": _self_sha256(),
            "sources": source_report(),
        },
        "rows": rows,
        "summary": {
            "random": _summarize(random_rows),
            "planner": _summarize(planner_rows),
            "paired": paired,
            "mean_paired_return_delta": float(
                np.mean(
                    [row["planner_minus_random_return"] for row in paired]
                )
            ),
            "mean_paired_achievement_delta": float(
                np.mean(
                    [
                        row["planner_minus_random_achievements"]
                        for row in paired
                    ]
                )
            ),
        },
    }
    _atomic_json(output, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--checkpoint-sha256", default=DEFAULT_CHECKPOINT_SHA256
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            REPO_ROOT
            / "outputs/d4_mamba_jepa/executed_control_t_base_5k.json"
        ),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2000, 2001, 2002])
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--context", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--candidates", type=int, default=34)
    parser.add_argument("--denoise-steps", type=int, default=4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--policy-seed-base", type=int, default=20260720)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    payload = run(
        checkpoint=args.checkpoint,
        checkpoint_sha256=args.checkpoint_sha256,
        output=args.output,
        seeds=args.seeds,
        device=torch.device(args.device),
        max_steps=args.max_steps,
        context=args.context,
        horizon=args.horizon,
        candidates=args.candidates,
        denoise_steps=args.denoise_steps,
        discount=args.discount,
        policy_seed_base=args.policy_seed_base,
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
