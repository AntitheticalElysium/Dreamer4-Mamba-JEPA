"""Which terminal tails actually teach an action-conditioned decision?

The death-transfer comparison showed all-17 supervision teaches state-conditioned death
while ordinary trajectories do not, which makes terminal exposure the obvious next
lever. But the terminal corpus is not what that framing assumes. It was collected by
targeting terminal *episodes* under four epsilon-greedy policies, retaining every
rollout; it never selected for states where one action kills and another survives. A
stratified sample put the actionable rate near 43%, with a median of zero safe
alternatives -- so a raw tail arm would spend much of its budget teaching unavoidable
death, and a raw "tail versus all-17" matchup would confound supervision type with
whether the example contains a decision at all.

This is the census that lets those be separated. For every TRAIN terminal episode it
replays to the pre-terminal state and executes all 17 actions, recording:

  actionable        does any action survive where the taken one died
  safe_actions      how many, and which
  taken_action      what the policy did, and whether it was among the lethal ones
  covariates        health, food, drink, energy, light, timestep, mob counts, inventory
  epsilon           the collection policy, from the shard manifest

The output indexes the archive into two strata -- unavoidable tails, which can teach
terminal appearance and continuation, and actionable tails, which are the only ones
carrying the mechanic the production model fails to learn. No new collection: this is
replay over data already on disk.

Resumable by shard, since the full pass is ~8,000 episodes each needing a compiled scan
over its whole trajectory.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay

OUT = HERE / "terminal_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=int, nargs="+", default=None)
    parser.add_argument("--limit", type=int, default=0, help="episodes per shard, 0 = all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    manifest = replay.manifest()
    _, _, _, step, _ = replay.env_and_render()   # params already bound
    import jax

    shards = args.shards if args.shards is not None else range(len(manifest["shards"]))
    for shard_index in shards:
        target = OUT / f"shard-{shard_index:03d}.json"
        if target.exists():
            continue
        entry = manifest["shards"][shard_index]
        payload = replay._shard(shard_index)
        epsilon = entry.get("epsilon")
        rows, started = [], time.time()
        episodes = payload["episodes"]
        count = len(episodes) if not args.limit else min(args.limit, len(episodes))
        for slot in range(count):
            fields = episodes[slot]
            if fields.get("split") not in (None, "train"):
                continue
            actions = fields["actions_taken"].numpy()
            n = len(actions)
            if n < 2:
                continue
            state = replay.advance_to(shard_index, slot, n - 1)
            if replay.is_dead(state):
                continue                      # already terminal before the last action
            key = jax.random.PRNGKey(replay.SUPPORT_SEED + shard_index + n)
            lethal = []
            for action in range(17):
                _, nxt, _, done, _ = step(key, state, action)
                lethal.append(bool(done) or replay.is_dead(nxt))
            taken = int(actions[n - 1])
            covariates = replay.scalars(state)
            rows.append({
                "shard": shard_index, "slot": slot, "steps": n, "epsilon": epsilon,
                "taken_action": taken, "taken_lethal": lethal[taken],
                "lethal": lethal, "n_lethal": int(sum(lethal)),
                "safe_actions": [a for a, x in enumerate(lethal) if not x],
                "actionable": bool(lethal[taken] and not all(lethal)),
                "all_lethal": bool(all(lethal)),
                **covariates,
            })
        if args.limit:
            # a truncated pass must never satisfy the skip guard on a later full run
            print(f"  --limit set: {len(rows)} rows NOT written", flush=True)
        else:
            target.write_text(json.dumps(rows))
        rate = len(rows) / max(time.time() - started, 1e-9)
        print(f"shard {shard_index}: {len(rows)} terminal episodes audited, "
              f"{rate:.1f}/s, epsilon {epsilon}", flush=True)


if __name__ == "__main__":
    main()
