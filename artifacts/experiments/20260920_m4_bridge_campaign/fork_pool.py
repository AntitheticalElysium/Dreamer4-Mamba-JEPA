"""Stage 1: the all-action fork corpus, as LeWM windows rather than Direct latents.

Direct's counterfactual arm trains on `broad_forks_v2`: hazard-choice roots carrying all 17 first
successors, plus a second NOOP successor wherever the first left the branch alive, gated by
`second_valid`. Matching Direct's data contract means matching that -- coverage, outcomes and the
surviving second steps -- and the user asked for exactly that match.

The store holds RAW uint8 frames, which is what makes this possible at all. Direct consumed it
through a frozen tokenizer and cached latents; LeWM trains its encoder jointly, so latents cannot
be cached and the frames themselves are what a LeWM arm needs. This reads them into the window
layout `JointSampler` produces, so the fork term and the factual term speak the same language:

  frames[-w:]            the w-frame context, w = the recipe's encoded window
  led_to_action[-w+1:]   the w-1 actions between those frames -- the alignment verified in the
                         2026-09-19 arms experiment, not re-derived here
  successors[a]          the first successor for every one of the 17 actions
  second[a]              the second (NOOP) successor, valid only where `second_valid`

Roots are held out by SEED, not by row, so no evaluation seed can reach training. The sealed M03
evaluation seeds are refused outright: a fork root on one of those would contaminate the panel the
whole campaign is read against.

Written as SHARDS, not one 6.8 GB tensor. Joint training already holds a mmap-backed episode
corpus, a ViT and a Mamba world on a 6 GB card with 20 GB of host RAM; a monolithic fork pool
resident alongside them is the difference between fitting and not. Shards are mmap-loaded and
indexed per batch, so only the pages of the four sampled roots are ever touched.
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
from d4mj.data import _sha256
from d4mj.lewm_config import window_layout

FORKS = ROOT / "artifacts/eda/broad_forks_v2"
N_ACTIONS = 17
# The M03 evaluation panel's seeds. A fork root here would contaminate the readout panel.
EVAL_SEEDS = range(13000, 14512)


def windows(recipe, holdout: float, seed: int, limit: int | None = None) -> dict:
    """Every fork root as a LeWM window, split fit/held-out by whole seed."""
    offsets = window_layout(recipe.joint)[0]
    span = offsets[-1] + 1
    # `led_to_action[t]` is the action that LED TO frame t, so a w-frame window carries w-1
    # actions -- the last w-1 of the history. That is the alignment the 2026-09-19 arms
    # experiment verified against the gate's own prefill path, and it is only this simple while
    # the window is consecutive, which `validate_recipe` already forces for a research recipe.
    if offsets != tuple(range(span)):
        raise SystemExit("fork windows assume a consecutive encoded window (stride 1)")
    paths = sorted(glob.glob(str(FORKS / "seed-*.pt")))
    if not paths:
        raise SystemExit("no broad_forks_v2 roots; nothing to match Direct against")
    frames, past, first, second, valid = [], [], [], [], []
    terminated, reward, health, achievement, factual, seeds = [], [], [], [], [], []
    second_terminated, second_reward = [], []
    for position, path in enumerate(paths):
        for row in torch.load(path, map_location="cpu", weights_only=False):
            if int(row["seed"]) in EVAL_SEEDS:
                raise RuntimeError(f"fork root on a sealed evaluation seed: {row['seed']}")
            if len(row["frames"]) < span:
                continue
            frames.append(row["frames"][-span:])
            past.append(row["led_to_action"][-span + 1:])
            first.append(row["successors"])
            second.append(row["second"])
            valid.append(row["second_valid"])
            terminated.append(row["terminated"])
            reward.append(row["reward"])
            health.append(row["health_delta"])
            achievement.append(row["achievement_delta"])
            second_terminated.append(row["second_terminated"])
            second_reward.append(row["second_reward"])
            factual.append(int(row["bc_action"]))
            seeds.append(int(row["seed"]))
        if limit is not None and len(frames) >= limit:
            break
        if position % 250 == 0:
            print(json.dumps({"stage": "fork_load", "shard": position, "of": len(paths),
                              "roots": len(frames)}), flush=True)

    seed_tensor = torch.tensor(seeds)
    unique = torch.unique(seed_tensor)
    order = unique[torch.randperm(len(unique), generator=torch.Generator().manual_seed(seed))]
    held = set(order[: int(round(holdout * len(order)))].tolist())
    is_held = torch.tensor([s in held for s in seeds])
    shuffle = torch.randperm(len(frames), generator=torch.Generator().manual_seed(seed + 1))
    def pick(values, dtype=None):
        out = torch.stack([values[i] for i in shuffle.tolist()])
        return out if dtype is None else out.to(dtype)
    pool = {"frames": pick(frames), "past_actions": pick(past),
            "first": pick(first), "second": pick(second),
            "second_valid": pick(valid, torch.bool),
            "terminated": pick(terminated, torch.bool),
            "reward": pick(reward, torch.float32),
            "health_delta": pick(health, torch.float32),
            "achievement_delta": pick(achievement, torch.long),
            "second_terminated": pick(second_terminated, torch.bool),
            "second_reward": pick(second_reward, torch.float32),
            "factual_action": torch.tensor(factual)[shuffle],
            "seed": seed_tensor[shuffle], "held_out": is_held[shuffle]}
    # A zeroed post-terminal slot must never be trained on; Direct asserts the same invariant.
    if bool(pool["second"][~pool["second_valid"]].abs().sum()):
        raise AssertionError("an invalid second successor is not zero")
    return pool


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", type=Path, default=ROOT / "d4mj/recipes/lewm_mamba_raw.json")
    parser.add_argument("--out", type=Path, default=HERE / "cache/forks")
    parser.add_argument("--per-shard", type=int, default=512)
    parser.add_argument("--evidence", type=Path, default=HERE / "evidence")
    parser.add_argument("--holdout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)

    pool = windows(load_recipe(args.recipe), args.holdout, args.seed, args.limit)
    total, shards = len(pool["frames"]), []
    for position in range(0, total, args.per_shard):
        stop = min(total, position + args.per_shard)
        path = args.out / f"fork-{position // args.per_shard:04d}.pt"
        torch.save({k: v[position:stop].clone() for k, v in pool.items()}, path)
        # Hash every shard: an unhashed store cannot be shown to be the one a run consumed.
        shards.append({"file": path.name, "roots": stop - position, "sha256": _sha256(path)})
        print(json.dumps({"stage": "fork_shard", "index": len(shards), "roots": stop - position}),
              flush=True)
    (args.out / "manifest.json").write_text(json.dumps(
        {"format": "d4mj_m4_forkpool_v1", "roots": total, "shards": shards,
         "holdout": args.holdout, "seed": args.seed}, indent=2) + "\n")
    fatal = pool["terminated"]
    report = {"schema": "d4mj_m4_forkpool_v1", "roots": len(pool["frames"]),
              "seeds": int(pool["seed"].unique().numel()),
              "held_out_roots": int(pool["held_out"].sum()),
              "held_out_seeds": int(pool["seed"][pool["held_out"]].unique().numel()),
              "window_frames": int(pool["frames"].shape[1]),
              "actions": int(pool["first"].shape[1]),
              "second_valid_share": round(float(pool["second_valid"].float().mean()), 4),
              "fatal_pair_share": round(float(fatal.float().mean()), 4),
              "roots_all_safe": int((fatal.sum(1) == 0).sum()),
              "roots_all_fatal": int((fatal.sum(1) == fatal.shape[1]).sum()),
              "rewarding_pair_share": round(float((pool["reward"] > 0).float().mean()), 4),
              "eval_seed_collisions": 0}
    (args.evidence / "fork_pool.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "fork_pool_complete", **report}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
