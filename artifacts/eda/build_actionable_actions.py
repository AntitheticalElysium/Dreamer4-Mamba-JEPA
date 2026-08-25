"""Real causal action histories for the actionable terminal roots.

The same omission as the original fork trainer: feeding BOS at every history block means
the backbone never sees which actions produced the trajectory. `build_fork_actions` fixed
it there; this is the equivalent for the terminal roots.

Convention as everywhere else: block i holds the action that produced observation i, so a
root whose 32-frame window starts at absolute step `start` takes

    led[0]   = actions[start - 1], or BOS when the window starts at the episode start
    led[i>0] = actions[start + i - 1]

Keyed by (shard, slot) to join the latent shards.
"""

from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import replay

HISTORY = 32
N_ACTIONS = 17
OUT = HERE / "actionable_actions.pt"


def main() -> None:
    keys = []
    for path in sorted(glob.glob(str(HERE / "actionable_roots" / "shard-*.pt"))):
        for root in torch.load(path, weights_only=False):
            keys.append((int(root["shard"]), int(root["slot"]), int(root["steps"]),
                         int(root["history"].shape[0])))

    actions, short = {}, 0
    for shard_index, slot, steps, length in keys:
        if length != HISTORY:
            short += 1
            continue
        taken = replay.episode_fields(shard_index, slot)["actions_taken"].long()
        start = steps - HISTORY
        first = torch.tensor([N_ACTIONS if start == 0 else int(taken[start - 1])])
        led = torch.cat([first, taken[start : steps - 1]])
        assert len(led) == HISTORY, f"{shard_index}:{slot} gave {len(led)}"
        actions[(shard_index, slot)] = led

    torch.save(actions, OUT)
    bos = sum(int((v == N_ACTIONS).sum()) for v in actions.values())
    total = len(actions) * HISTORY
    print(f"wrote {len(actions):,} action histories ({short} short-history roots skipped)")
    print(f"BOS occupies {bos}/{total} positions ({bos/total:.3%}) -- only windows that "
          f"start at step 0")


if __name__ == "__main__":
    main()
