"""Successors at the corrected actionable terminal roots, for the three-arm comparison.

These are support-v2 roots, cached fresh: the existing 3,651-root fork corpus is a
different selection (damage-choice roots on a fixed production trajectory) and reusing
it would change the population as well as the supervision.

Every branch is stepped with the episode's own final-step environment key, so the 17
successors differ only in the action taken. That is the whole point of the comparison,
and the first census got it wrong by inventing a key.

Written per root:

  history       the 32 stored observations ending at the pre-terminal state
  render_drift  max pixel difference between the replayed pre-terminal state and the
                stored final observation; must be 0 or the branches are not this
                trajectory's
  successors    all 17 successor frames under the original key
  terminated    which of them end the episode
  lethal_action the action the policy actually took, which was fatal
  safe_actions  the alternatives that survive

The factual arm consumes `lethal_action` alone at each root; the counterfactual arm
cycles through all 17 across repeated presentations of the same root, so the two see the
same roots the same number of times with the same number of targets per update.
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

HISTORY = 32
OUT = HERE / "actionable_roots"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strata", type=Path, default=HERE / "terminal_strata.json")
    parser.add_argument("--shard-size", type=int, default=200)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    roots = json.loads(args.strata.read_text())["actionable"]
    print(f"{len(roots)} actionable roots", flush=True)
    _, _, _, step, frame = replay.env_and_render()

    rows, shard, started = [], 0, time.time()
    done = {int(p.stem.split("-")[1]) for p in OUT.glob("shard-*.pt")}
    for index, root in enumerate(roots):
        if index // args.shard_size in done:
            continue
        shard_index, slot, steps = root["shard"], root["slot"], root["steps"]
        _, env_keys = replay._slot_keys(shard_index, slot)
        key = env_keys[steps - 1]

        # the causal window is already on disk as rendered observations, so it costs
        # nothing; replaying it frame by frame would be 32 full scans per root
        fields = replay.episode_fields(shard_index, slot)
        observations = fields["observations"]
        start = max(0, steps - HISTORY)
        history = observations[start:steps]
        state = replay.advance_to(shard_index, slot, steps - 1)

        # the replayed pre-terminal state must render to the stored final observation,
        # or the branch successors do not belong to this trajectory
        drift = int(np.abs(np.asarray(frame(state)).astype(np.int32)
                           - observations[steps - 1].numpy().astype(np.int32)).max())

        successors, terminated = [], []
        for action in range(17):
            _, nxt, _, done_flag, _ = step(key, state, action)
            successors.append(np.asarray(frame(nxt)))
            terminated.append(bool(done_flag) or replay.is_dead(nxt))
        rows.append({
            "shard": shard_index, "slot": slot, "steps": steps,
            "history": history.clone(),
            "render_drift": drift,
            "successors": torch.from_numpy(np.stack(successors)),
            "terminated": torch.tensor(terminated),
            "lethal_action": root["lethal_action"],
            "safe_actions": root["safe_actions"],
            "epsilon": root["epsilon"],
        })
        if len(rows) >= args.shard_size:
            torch.save(rows, OUT / f"shard-{shard:03d}.pt")
            shard, rows = shard + 1, []
        if (index + 1) % 100 == 0:
            rate = (index + 1) / (time.time() - started)
            print(f"  {index+1}/{len(roots)} [{time.time()-started:.0f}s, "
                  f"{(len(roots)-index-1)/rate:.0f}s left]", flush=True)
    if rows:
        torch.save(rows, OUT / f"shard-{shard:03d}.pt")
        shard += 1
    (OUT / "manifest.json").write_text(json.dumps(
        {"roots": len(roots), "shards": shard, "history": HISTORY,
         "key": "episode final-step environment key"}, indent=2))
    print(f"wrote {shard} shards", flush=True)


if __name__ == "__main__":
    main()
