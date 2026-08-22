"""The real causal action history for every fork root.

The Phase-1B fork pipeline fed the world backbone `torch.full(..., config.n_actions)`
-- the BOS/null token -- at every one of a root's 32 history blocks. So the backbone
never saw which actions produced the trajectory, and the ordinary teacher-forced term
conditioned its readout on the null token exclusively, leaving the two loss terms on
disjoint action tokens: null for `ordinary`, 0-16 for the all-17 fork term.

The fix needs no re-encoding. `z_history` and `z_branch` come from the encoder, which
consumes patches and never sees an action, so the latents are unaffected. Only the
action stream was wrong, and `branched_damage/seed-*.pt` already carries the full
`led_to_action` trajectory per seed, BOS at index 0, on exactly the fixed production
rollout the roots were drawn from.

Emits {(seed, step): LongTensor(32)} under the project's convention: block i holds the
action that produced observation i, so a root at absolute `step` takes the closed slice
[step - 31, step].
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

HISTORY = 32
OUT = HERE / "fork_actions.pt"


def main() -> None:
    actions: dict[tuple[int, int], torch.Tensor] = {}
    short = 0
    for path in sorted((HERE / "branched_damage").glob("seed-*.pt")):
        payload = torch.load(path, weights_only=False)
        seed, led = int(payload["seed"]), payload["led_to_action"].long()
        assert int(led[0]) == 17, f"{seed}: history does not start at BOS"
        for row in payload["rows"]:
            step = int(row["step"])
            start = step - HISTORY + 1
            if start < 0:                       # dropped by load_forkset's full-history filter
                short += 1
                continue
            window = led[start : step + 1]
            assert len(window) == HISTORY, f"{seed}:{step} gave {len(window)}"
            actions[(seed, step)] = window.clone()
    torch.save(actions, OUT)
    print(f"wrote {len(actions):,} action histories ({short:,} roots too near an episode "
          f"start, matching the forkset's own filter)")


if __name__ == "__main__":
    main()
