"""Assemble branched-collection opportunity states into the fixed fork-file layout.

The recursive evaluator selects action-dependent-death states itself and replays
each seed, so only `seed`, `step` and `true_death` are load-bearing; `pair` is
never read. The branched collection used the same frozen BC checkpoint as the
DEV forks, so replay reproduces the collected states exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collection",
        type=Path,
        default=Path("artifacts/branched_coverage_gate/collection"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.collection / "manifest.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("branched collection is incomplete")

    seeds, steps, deaths, actions, trajectory_deaths = [], [], [], [], []
    for shard in sorted(args.collection.glob("seed-*.pt")):
        payload = torch.load(shard, weights_only=False, map_location="cpu")
        death = payload["true_death"]
        varies = death.any(1) & (~death).any(1)
        if not bool(varies.any()):
            continue
        rows = varies.nonzero().flatten()
        seeds.append(torch.full((len(rows),), int(payload["seed"]), dtype=torch.long))
        steps.append(payload["step"][rows].long())
        deaths.append(death[rows])
        actions.append(payload["trajectory_action"][rows].long())
        trajectory_deaths.append(payload["trajectory_death"][rows])

    forks = {
        "seed": torch.cat(seeds),
        "step": torch.cat(steps),
        "true_death": torch.cat(deaths),
        "trajectory_action": torch.cat(actions),
        "trajectory_death": torch.cat(trajectory_deaths),
        "pair": torch.arange(len(torch.cat(seeds))),
    }
    keys = {(int(s), int(t)) for s, t in zip(forks["seed"], forks["step"])}
    if len(keys) != len(forks["seed"]):
        raise ValueError("branched fork set contains duplicate opportunity states")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(forks, args.out)
    digest = hashlib.sha256(args.out.read_bytes()).hexdigest()
    print(
        f"states={len(forks['seed'])} seeds={len(set(forks['seed'].tolist()))} "
        f"source_manifest={manifest['contract']['trajectory_phase2'][:16]} sha256={digest[:16]}"
    )


if __name__ == "__main__":
    main()
