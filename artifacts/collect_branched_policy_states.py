"""Collect all-action outcomes at states reached by the frozen BC policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from artifacts.localize_counterfactual import load_models
from artifacts.phase1b_diagnostic_common import (
    atomic_json,
    file_digest,
    implementation_digests,
)
from artifacts.phase1b_geometry_common import atomic_torch
from d4mj.config import Config
from d4mj.data import patchify
from d4mj.env import reset, step as env_step
from d4mj.transition import observe

VERSION = "branched-policy-states-v1"


@torch.no_grad()
def collect_seed(encoder, world, heads, config: Config, seed: int, limit: int) -> dict:
    observation, env_state = reset(seed)
    state = None
    incoming = torch.full(
        (1, 1), config.n_actions, dtype=torch.long, device=config.device
    )
    world_rng = torch.Generator(device=config.device).manual_seed(seed + 2**21)
    policy_rng = torch.Generator(device=config.device).manual_seed(seed + 2**20)
    values = {
        "latent": [],
        "true_death": [],
        "true_reward": [],
        "trajectory_action": [],
        "step": [],
    }

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
        values["latent"].append(state.world.latent[0, -1].cpu())
        values["true_death"].append(
            torch.tensor([successor[3] for successor in successors], dtype=torch.bool)
        )
        values["true_reward"].append(
            torch.tensor([successor[2] for successor in successors], dtype=torch.float32)
        )
        values["trajectory_action"].append(chosen)
        values["step"].append(index)

        observation, env_state, _, terminated, truncated = successors[chosen]
        incoming.fill_(chosen)
        if terminated or truncated:
            break

    if not values["latent"]:
        raise RuntimeError(f"policy seed {seed} produced no states")
    death = torch.stack(values["true_death"])
    action = torch.tensor(values["trajectory_action"], dtype=torch.long)
    return {
        "seed": seed,
        "latent": torch.stack(values["latent"]),
        "true_death": death,
        "true_reward": torch.stack(values["true_reward"]),
        "trajectory_action": action,
        "trajectory_death": death.gather(1, action[:, None]).squeeze(1),
        "step": torch.tensor(values["step"], dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1a", type=Path, required=True)
    parser.add_argument("--trajectory-phase2", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=14_000)
    parser.add_argument("--seeds", type=int, default=512)
    parser.add_argument("--limit", type=int, default=400)
    args = parser.parse_args()
    if args.seeds < 2 or args.limit < 1:
        parser.error("seeds must be at least two and limit must be positive")

    base = Config()
    config = Config(transition="direct", time_mixer="attention")
    eval_seeds = set(config.outcome_gate_seeds) | set(range(13_000, 13_128))
    collection_seeds = set(range(args.seed_start, args.seed_start + args.seeds))
    if collection_seeds & eval_seeds:
        parser.error("branched TRAIN seeds overlap a fixed DEV fork seed")

    contract = {
        "version": VERSION,
        "phase1a": file_digest(args.phase1a),
        "trajectory_phase2": file_digest(args.trajectory_phase2),
        "seed_start": args.seed_start,
        "seed_count": args.seeds,
        "limit": args.limit,
        "policy": "frozen Direct-Attention Phase-2 BC policy, categorical temperature 1",
        "branches": "all 17 actions from every reached state with one common environment key",
        "split": "TRAIN collection seeds disjoint from fixed 12000 and 13000-series DEV forks",
        "implementation": implementation_digests(Path(__file__)),
    }
    manifest_path = args.out / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("contract") != contract:
            raise ValueError("branched-state collection contract changed")
        if manifest.get("complete"):
            print(f"already complete: {manifest_path}", flush=True)
            return
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        manifest = {
            "contract": contract,
            "complete": False,
            "seeds_complete": [],
            "shards": [],
            "states": 0,
            "opportunity_states": 0,
            "fatal_actions": 0,
            "trajectory_deaths": 0,
        }
        atomic_json(manifest_path, manifest)

    completed = set(int(seed) for seed in manifest["seeds_complete"])
    encoder, world, heads = load_models(
        args.phase1a, args.trajectory_phase2, base, config
    )
    for position, seed in enumerate(
        range(args.seed_start, args.seed_start + args.seeds), start=1
    ):
        if seed in completed:
            continue
        payload = collect_seed(encoder, world, heads, config, seed, args.limit)
        shard = args.out / f"seed-{seed:06d}.pt"
        if shard.exists():
            raise FileExistsError(f"unregistered branch shard exists: {shard}")
        atomic_torch(shard, payload)
        death = payload["true_death"]
        opportunity = death.any(1) & (~death).any(1)
        record = {
            "file": shard.name,
            "sha256": file_digest(shard),
            "seed": seed,
            "states": len(death),
            "opportunity_states": int(opportunity.sum()),
            "fatal_actions": int(death.sum()),
            "trajectory_deaths": int(payload["trajectory_death"].sum()),
        }
        manifest["seeds_complete"].append(seed)
        manifest["shards"].append(record)
        for key in (
            "states",
            "opportunity_states",
            "fatal_actions",
            "trajectory_deaths",
        ):
            manifest[key] += record[key]
        atomic_json(manifest_path, manifest)
        print(
            f"seed {position}/{args.seeds}: states={manifest['states']} "
            f"opportunities={manifest['opportunity_states']} "
            f"fatal_actions={manifest['fatal_actions']}",
            flush=True,
        )

    if len(manifest["seeds_complete"]) != args.seeds:
        raise AssertionError("branched-state collection did not finish every seed")
    manifest["complete"] = True
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
