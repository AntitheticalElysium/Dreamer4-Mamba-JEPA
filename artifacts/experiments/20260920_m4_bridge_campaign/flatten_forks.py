"""Stage 1 (alternative): broad_forks_v2 flattened into ORDINARY LeWM transitions.

Two different things can be meant by "give Raw/TC the fork data", and they answer different
questions:

  GROUPED    Direct's branch-training contract: 4 roots x all 17 actions + surviving second steps,
             computed every update as a separate weighted term. ~136 extra differentiable
             transitions and 152 extra encoder frames per update, which roughly doubles joint
             training (0.32 -> 0.59 s/update, measured). `fork_mass` weights the LOSS, not the
             compute -- all 136 transitions are computed to obtain it. This answers: *can LeWM work
             end to end when given Direct-style counterfactual supervision?*
  FLATTENED  each (root, action) counterfactual becomes one ordinary episode in the corpus, drawn
             through the normal `JointSampler` minibatch path at the same batch size. No extra
             per-update work at all -- just a larger, differently distributed dataset. This answers
             the purer question: *can native LeWM work when simply exposed to the same examples?*

This builds the FLATTENED corpus. It is not a cheaper approximation of the grouped one: the world
never sees two actions from the same root in one batch, which is exactly the contrast the grouped
form supplies.

Prior evidence does NOT already cover this. The 2026-09-19 A' arm is often read as the flattened
control, but it trained on `z_branch[pick, factual_action[pick]]` -- the LOGGED action only, one
per root. All 17 counterfactual transitions as ordinary minibatch examples has never been run.

Layout, per (root, action) pair, at the recipe's 4-frame window:

    frames  = history[-3:] + [successor[a]]            (+ [second[a]] when it survived)
    actions = led_to_action[-2:] + [a]                 (+ [NOOP] when it survived)

`second_valid` was verified to mean exactly "the first step survived", so the only terminal a
flattened episode can carry is its last transition -- which is what `validate_episode` requires.

REWARDS ARE NOT COMPLETE HERE. The fork store records the counterfactual step's reward and the
second step's, but the history transitions' rewards were never collected, and are written as zero.
That is harmless for joint WORLD training, which reads only frames and actions, and it is why this
corpus must never reach the bridge's reward head. `bc_eligible` is False for the same reason: a
counterfactual branch is not logged expert behaviour.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from d4mj.config import load_recipe
from d4mj.data import STORE_FORMAT, Episode, atomic_manifest, save_episode_shard
from d4mj.lewm_config import window_layout

FORKS = ROOT / "artifacts/eda/broad_forks_v2"
N_ACTIONS = 17
NOOP = 0
EVAL_SEEDS = range(13000, 14512)


def episodes_of(row, span: int, split: str):
    """Every action from one root as its own ordinary episode."""
    history = row["frames"][-(span - 1):]
    past = row["led_to_action"][-(span - 2):]
    seed, step = int(row["seed"]), int(row["step"])
    out = []
    for action in range(N_ACTIONS):
        survived = bool(row["second_valid"][action])
        frames = [history, row["successors"][action][None]]
        actions = [past, torch.tensor([action])]
        reward = [torch.zeros(span - 2), row["reward"][action][None].float()]
        terminated = [torch.zeros(span - 2, dtype=torch.bool), row["terminated"][action][None]]
        truncated = [torch.zeros(span - 2, dtype=torch.bool), row["truncated"][action][None]]
        if survived:
            frames.append(row["second"][action][None])
            actions.append(torch.tensor([NOOP]))
            reward.append(row["second_reward"][action][None].float())
            terminated.append(row["second_terminated"][action][None])
            truncated.append(row["second_truncated"][action][None])
        terminated = torch.cat(terminated)
        truncated = torch.cat(truncated)
        if not bool(terminated[-1] or truncated[-1]):
            # The branch ends because the collector stopped, not because the agent died.
            truncated = truncated.clone()
            truncated[-1] = True
        out.append(Episode(
            observations=torch.cat(frames), actions_taken=torch.cat(actions).long(),
            rewards=torch.cat(reward), terminated=terminated, truncated=truncated,
            events=None, uniform_eligible=True, bc_eligible=False, epsilon=None, split=split,
            episode_id=f"fork_v2:{seed:06d}:{step:05d}:a{action:02d}",
            terminal_cause="death" if bool(terminated[-1]) else "branch_end"))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT / "d4mj/recipes/lewm_mamba_raw.json")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/craftax_forks_flat_v1")
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--per-shard", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=None, help="root files, for a smoke build")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)

    recipe = load_recipe(args.recipe)
    offsets = window_layout(recipe.joint)[0]
    span = offsets[-1] + 1
    if offsets != tuple(range(span)):
        raise SystemExit("flattening assumes a consecutive encoded window (stride 1)")

    paths = sorted(glob.glob(str(FORKS / "seed-*.pt")))[:args.limit]
    # Seeds come from the FILENAMES, so the split can be decided without holding the corpus in
    # memory. Loading all 1,505 root files first peaked at 12 GB RSS; this streams one file at a
    # time and stays in the hundreds of megabytes.
    seeds = sorted({int(Path(f).stem.split("-")[1]) for f in paths})
    if any(s in EVAL_SEEDS for s in seeds):
        raise RuntimeError("a fork root sits on a sealed evaluation seed")
    order = torch.randperm(len(seeds), generator=torch.Generator().manual_seed(args.seed))
    held = {seeds[i] for i in order[: int(round(args.holdout * len(seeds)))].tolist()}

    built, shards, counts, roots = [], [], {"train": 0, "dev": 0}, 0
    def flush():
        if not built:
            return
        path = args.out / f"shard-{len(shards):04d}.pt"
        shards.append(save_episode_shard(path, built))
        built.clear()

    for position, path in enumerate(paths):
        for row in torch.load(path, map_location="cpu", weights_only=False):
            seed = int(row["seed"])
            if seed in EVAL_SEEDS:
                raise RuntimeError("a fork root sits on a sealed evaluation seed")
            split = "dev" if seed in held else "train"
            made = episodes_of(row, span, split)
            counts[split] += len(made)
            roots += 1
            built.extend(made)
            if len(built) >= args.per_shard:
                flush()
        if position % 150 == 0:
            print(json.dumps({"stage": "flatten", "file": position, "of": len(paths),
                              "roots": roots, "shards": len(shards)}), flush=True)
    flush()

    manifest = {"format": STORE_FORMAT, "kind": "d4mj_forks_flat_v1", "complete": True,
                "episodes": sum(s["episodes"] for s in shards), "shards": shards,
                "transitions": sum(s["transitions"] for s in shards),
                "terminal_episodes": sum(s["terminal_episodes"] for s in shards),
                "split_episode_counts": counts, "roots": roots, "actions": N_ACTIONS,
                "holdout_by": "whole seed", "holdout": args.holdout, "seed": args.seed,
                "bc_eligible": False, "uniform_eligible": True,
                "rewards": "counterfactual and second steps carry true rewards; history "
                           "transitions were never collected and are zero -- WORLD TRAINING ONLY, "
                           "never the bridge's reward head",
                "source": str(FORKS), "collector_training_access": "unknown"}
    atomic_manifest(args.out / "manifest.json", manifest)
    report = {k: v for k, v in manifest.items() if k != "shards"}
    (args.evidence / "forks_flat.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "flatten_complete", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
