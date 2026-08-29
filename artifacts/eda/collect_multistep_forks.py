"""Sealed multi-step counterfactual rollouts: does a first action's consequence persist?

Every reading in this line of work is one-step. `_direct_loss` trains two generated
states and its own docstring caps imagination there, and `production_1b_evaluation`
measures a second generated step only on ordinary held-out trajectories -- never on a
counterfactual branch. So nothing yet says whether Direct propagates the consequence of a
*chosen* action beyond the step it was taken.

That matters more than one-step fidelity for this project. Compounding sits at 1.75-1.77x
for every model ever trained here -- production, terminal, broad, both seeds -- and not
one intervention moved it. Improving one-step successors while compounding stays fixed is
not obviously progress toward imagination RL.

The design isolates the first action and nothing else:

  root        one of the 197 held-out roots every other reading uses, so the numbers are
              comparable. Whole-seed test split; these seeds were never trained on.
  branch      all 17 first actions from the identical state under one shared env key,
              the same intervention protocol as the one-step forks
  continue    K-1 NOOP steps, identical across all 17 branches, so any divergence
              between branches at step k is attributable to the first action alone

NOOP rather than a policy continuation on purpose: a policy would choose different
actions per branch, and the model would then be scored partly on predicting the policy.
Holding the continuation fixed asks only whether the branch difference survives.

Histories are not re-collected -- `branched_965.pt` already holds the exact 32 frames
each root was scored with, so reusing them guarantees the multi-step reading sits on the
same context as the one-step one.

No model is loaded. `branched_damage` stores each rollout's full `led_to_action`, so the
trajectory replays deterministically from the stored actions, and every root's arrival
frame is asserted against the one saved with it. The alternative -- re-running the frozen
BC policy -- would need the pre-promotion World and pre-64-slot Config to be reconstructed
exactly, and would make reaching the right state depend on reproducing a sampler.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from train_phase1b_fork import seed_split

from d4mj.env import reset, step as env_step

N_ACTIONS = 17
NOOP = 0
OUT = HERE / "multistep_forks"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=4, help="rollout depth, first action included")
    parser.add_argument("--limit", type=int, default=0, help="roots, for smoking")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)

    # the rollout policy predates the 64-slot default and must be loaded under the
    # config it was written with; it only reaches the root states, and they are the same
    # states every other reading uses
    # exactly the roots every other evaluation scores, so nothing new is held out
    successors = {}
    for path in sorted(glob.glob(str(HERE / "fork_successors" / "shard-*.pt"))):
        for row in torch.load(path, weights_only=False):
            successors[(int(row["seed"]), int(row["step"]))] = row
    keys = set(successors)
    rows = [r for r in torch.load(HERE / "fork_histories" / "branched_965.pt", weights_only=False)
            if (int(r["seed"]), int(r["step"])) in keys
            and seed_split(int(r["seed"])) == "test"]
    if args.limit:
        rows = rows[: args.limit]
    wanted: dict[int, set[int]] = {}
    saved: dict[tuple[int, int], torch.Tensor] = {}
    for r in rows:
        wanted.setdefault(int(r["seed"]), set()).add(int(r["step"]))
        saved[(int(r["seed"]), int(r["step"]))] = r["frames"][-1]
    print(f"{len(rows)} sealed test roots across {len(wanted)} rollout seeds, "
          f"depth {args.steps}", flush=True)

    # the stored action stream per rollout seed, under the a_{t-1} convention: index 0
    # is the BOS token and index i is the action that produced observation i
    actions = {}
    for path in sorted(glob.glob(str(HERE / "branched_damage" / "seed-*.pt"))):
        payload = torch.load(path, weights_only=False)
        actions[int(payload["seed"])] = payload["led_to_action"]
    missing = [s for s in wanted if s not in actions]
    assert not missing, f"{len(missing)} seeds without a stored action stream"

    collected, started = [], time.time()
    for order, seed in enumerate(sorted(wanted)):
        stream = actions[seed]
        observation, env_state = reset(seed)
        for index in range(max(wanted[seed]) + 1):
            if index in wanted[seed]:
                # the branch is the right counterfactual only if the replay is on the
                # trajectory the root was saved from, so compare against the frame stored
                # with it rather than trusting the replay
                assert torch.equal(observation.cpu(), saved[(seed, index)].cpu()), (
                    f"replay diverged at seed {seed} step {index}")
                frames, dead, done_at = [], [], []
                for action in range(N_ACTIONS):
                    branch, branch_state, ended = [], env_state, -1
                    for depth in range(args.steps):
                        frame, branch_state, _, terminated, truncated = env_step(
                            branch_state, action if depth == 0 else NOOP,
                            seed + index + depth + 1)
                        branch.append(frame.clone())
                        if terminated or truncated:
                            ended = depth
                            # a finished branch is held at its last frame rather than
                            # stepped on, which would reset the env underneath us
                            branch += [frame.clone()] * (args.steps - depth - 1)
                            break
                    frames.append(torch.stack(branch))
                    dead.append(ended >= 0)
                    done_at.append(ended)
                done_at = torch.tensor(done_at)
                # `terminated_first` is the one-step fact the existing forks record;
                # `terminated_any` folds in the NOOP continuation. Keeping both named
                # apart stops the two being compared as if they meant the same thing.
                row = {"seed": seed, "step": index,
                       "successors": torch.stack(frames),
                       "terminated_first": done_at == 0,
                       "terminated_any": torch.tensor(dead),
                       "terminated_at": done_at,
                       "continuation": NOOP, "depth": args.steps}
                reference = successors[(seed, index)]
                assert torch.equal(row["successors"][:, 0].cpu(),
                                   reference["successors"].cpu()), (
                    f"depth-0 frames disagree with the saved one-step successors at "
                    f"seed {seed} step {index}")
                assert torch.equal(row["terminated_first"].cpu(),
                                   reference["terminated"].cpu()), (
                    f"depth-0 deaths disagree with the saved forks at seed {seed} "
                    f"step {index}")
                collected.append(row)
            if index + 1 < len(stream):
                observation, env_state, _, terminated, truncated = env_step(
                    env_state, int(stream[index + 1]), seed + index + 1)
                if terminated or truncated:
                    break
        if (order + 1) % 10 == 0:
            rate = (order + 1) / (time.time() - started)
            print(f"  seed {order+1}/{len(wanted)}, {len(collected)} roots "
                  f"[{time.time()-started:.0f}s, {(len(wanted)-order-1)/rate:.0f}s left]",
                  flush=True)

    assert len(collected) == len(rows), f"collected {len(collected)} of {len(rows)} roots"
    torch.save(collected, OUT / "rollouts.pt")
    (OUT / "manifest.json").write_text(json.dumps(
        {"roots": len(collected), "depth": args.steps, "continuation": "NOOP",
         "split": "test", "seeds": len(wanted)}, indent=2))
    print(f"wrote {len(collected)} roots in {time.time()-started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
