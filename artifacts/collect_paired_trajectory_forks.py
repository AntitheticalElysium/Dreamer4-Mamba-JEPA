"""Collect paired safe/fatal trajectory actions with all-action simulator truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from artifacts.phase1b_diagnostic_common import atomic_json, file_digest, implementation_digests
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe


@torch.no_grad()
def collect(
    encoder,
    world,
    heads,
    config: Config,
    *,
    seed_start: int,
    seed_count: int,
    limit: int,
    reference: dict,
) -> tuple[dict, dict]:
    reference_keys = {
        (int(seed), int(step)): row
        for row, (seed, step) in enumerate(zip(reference["seed"], reference["step"]))
    }
    reproduced = set()
    candidates = []

    for seed in range(seed_start, seed_start + seed_count):
        observation, env_state = reset(seed)
        state = None
        incoming = torch.full(
            (1, 1), config.n_actions, dtype=torch.long, device=config.device
        )
        world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
        policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)

        for index in range(limit):
            patches = patchify(observation[None, None], config.patch).to(config.device)
            state, agent = observe(
                world, encoder, state, incoming, patches, world_rng, config
            )
            logits = heads(agent)["policy"][:, -1, 0]
            chosen = int(torch.multinomial(logits.softmax(-1), 1, generator=policy_rng))

            successors = [
                env_step(env_state, action, seed + index + 1)
                for action in range(config.n_actions)
            ]
            truth = torch.tensor([value[3] for value in successors], dtype=torch.bool)
            key = (seed, index)
            if key in reference_keys:
                row = reference_keys[key]
                if not torch.equal(truth, reference["true_death"][row].bool()):
                    raise AssertionError(f"saved fork truth changed at {key}")
                reproduced.add(key)

            if bool(truth.any()) and not bool(truth.all()):
                candidates.append(
                    {
                        "seed": seed,
                        "step": index,
                        "true_death": truth,
                        "trajectory_action": chosen,
                        "trajectory_death": bool(truth[chosen]),
                    }
                )

            observation, env_state, _, terminated, truncated = successors[chosen]
            incoming.fill_(chosen)
            if terminated or truncated:
                break

    if reproduced != set(reference_keys):
        missing = sorted(set(reference_keys) - reproduced)
        raise RuntimeError(f"failed to reproduce saved trajectory states: {missing}")

    by_seed: dict[int, list[dict]] = {}
    for row in candidates:
        by_seed.setdefault(row["seed"], []).append(row)

    pairs = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        fatal = [row for row in rows if row["trajectory_death"]]
        for dead in fatal:
            safe = [
                row
                for row in rows
                if not row["trajectory_death"] and row["step"] < dead["step"]
            ]
            if safe:
                pairs.append((max(safe, key=lambda row: row["step"]), dead))

    if not pairs:
        raise RuntimeError("no trajectory has both a safe and fatal opportunity action")
    selected = []
    for pair, (safe, dead) in enumerate(pairs):
        selected.extend([(pair, safe), (pair, dead)])

    payload = {
        "seed": torch.tensor([row["seed"] for _, row in selected]),
        "step": torch.tensor([row["step"] for _, row in selected]),
        "true_death": torch.stack([row["true_death"] for _, row in selected]),
        "trajectory_action": torch.tensor(
            [row["trajectory_action"] for _, row in selected]
        ),
        "trajectory_death": torch.tensor(
            [row["trajectory_death"] for _, row in selected]
        ),
        "pair": torch.tensor([pair for pair, _ in selected]),
    }
    report = {
        "candidate_opportunity_states": len(candidates),
        "candidate_safe_trajectory_actions": sum(
            not row["trajectory_death"] for row in candidates
        ),
        "candidate_fatal_trajectory_actions": sum(
            row["trajectory_death"] for row in candidates
        ),
        "paired_trajectories": len(pairs),
        "selected_states": len(selected),
        "selected_safe_trajectory_actions": int((~payload["trajectory_death"]).sum()),
        "selected_fatal_trajectory_actions": int(payload["trajectory_death"].sum()),
        "safe_selection": "nearest preceding terminal-opportunity state in the same trajectory",
        "saved_reference_truth_reproduced_exactly": True,
    }
    return payload, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--reference-forks", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=13_000)
    parser.add_argument("--seeds", type=int, default=128)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    contract = {
        "version": "paired-trajectory-forks-v1",
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "reference_forks": file_digest(args.reference_forks),
        "implementation": implementation_digests(Path(__file__)),
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "limit": args.limit,
        "evaluation_only": True,
    }
    encoder, world, heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    reference = torch.load(args.reference_forks, weights_only=False)
    payload, collection = collect(
        encoder,
        world,
        heads,
        config,
        seed_start=args.seed_start,
        seed_count=args.seeds,
        limit=args.limit,
        reference=reference,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    fork_path = args.out / "paired_trajectory_forks.pt"
    torch.save(payload, fork_path)
    report = {
        "contract": contract,
        "fork_path": str(fork_path.resolve()),
        "fork_sha256": file_digest(fork_path),
        "collection": collection,
    }
    atomic_json(args.out / "collection_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()

