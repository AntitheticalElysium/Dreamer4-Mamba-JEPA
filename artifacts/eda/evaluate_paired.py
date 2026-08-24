"""Two evaluation categories, reported apart.

  interpolation -- hazard-choice roots from held-out S82 seeds, the same distribution
                   the paired data come from;
  transfer      -- the 677 support hazard forks (damage truth), the 965 saved
                   opportunity roots restricted to held-out seeds, and the 104 policy
                   forks (death truth), none of which entered training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
import corpus
from evaluate_damage_classifier import report, score_roots
from train_damage_classifier import DamageHead
from train_paired_scaling import load_pool, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World


def hazard_split_of(record) -> str:
    key = f"consequence-probe:{record['shard']}:{record['slot']}:{record['t']}"
    draw = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little") % 10
    return "fit" if draw < 6 else ("tune" if draw < 8 else "test")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = Config(transition="direct", time_mixer="attention")
    world, head = World(config).to(config.device), DamageHead(config).to(config.device)
    load(args.model, config, part0=world, part1=head)
    world.eval(); head.eval()
    results = {}
    print(f"\nmodel {args.model}")

    # ------------------------------------------------ interpolation: held-out S82 seeds
    roots, trajectories = load_pool()
    held = [r for r in roots if r["split"] == "test" and r["hazard"]]
    for name, subset in (("S82 held-out seeds (damage)", held),):
        histories, leds, labels = [], [], []
        for record in subset:
            latents, led = trajectories[record["seed"]]
            t = record["step"]
            start = max(0, t - config.sequence_long + 1)
            histories.append(latents[start : t + 1])
            leds.append(led[start : t + 1])
            labels.append(record["label"])
        scores = score_roots(world, head, config, histories, leds)
        results["interpolation_s82"] = report(name, scores, np.stack(labels), 41)

    # ------------------------------------------------------------------- transfer sets
    records = []
    for path in sorted((HERE / "latent_forks").glob("shard-*.pt")):
        records += torch.load(path, weights_only=False)
    rows = corpus.train_rows()
    lookup = {(r["shard"], r["slot"]): i for i, r in enumerate(rows) if r["source"] == "support"}
    cached = {i: v for i, v in corpus.iter_cached_latents()}
    manifest = json.loads((corpus.SUPPORT / "manifest.json").read_text())
    action_cache: dict[int, dict] = {}
    histories, leds, labels = [], [], []
    for record in records:
        health, dead = record["health"].numpy(), record["dead"].numpy()
        positives = (health <= -1) | dead
        if not positives.any() or not ((health >= 0) & ~dead).any():
            continue
        if hazard_split_of(record) != "test":
            continue
        shard = record["shard"]
        if shard not in action_cache:
            payload = torch.load(corpus.SUPPORT / manifest["shards"][shard]["file"],
                                 weights_only=False, mmap=True)
            action_cache[shard] = {s: f["actions_taken"].numpy()
                                   for s, f in enumerate(payload["episodes"])}
            del payload
        acts = action_cache[shard][record["slot"]]
        t = record["t"]
        start = max(0, t - config.sequence_long + 1)
        histories.append(cached[lookup[(shard, record["slot"])]][start : t + 1].clone())
        led = np.concatenate([[config.n_actions] if start == 0 else [acts[start - 1]],
                              acts[start : t]]).astype(np.int64)
        leds.append(torch.from_numpy(led))
        labels.append(positives.astype(float))
    scores = score_roots(world, head, config, histories, leds)
    results["transfer_hazard_forks"] = report("677 hazard forks, test split (damage)",
                                              scores, np.stack(labels), 42)

    for name, key, seed in (("965 opportunity roots, held-out seeds (death)",
                             "branched_965", 43),
                            ("104 policy forks (death)", "policy_fork_104", 44)):
        path = HERE / "fork_histories" / f"{key}.pt"
        if not path.exists():
            continue
        stored = torch.load(path, weights_only=False)
        if key == "branched_965":
            stored = [r for r in stored if seed_split(int(r["seed"])) == "test"]
            if not stored:
                continue
        scores = score_roots(world, head, config, [r["history"] for r in stored],
                             [r["led_to_action"] for r in stored])
        truth = np.stack([r["true_death"].numpy().astype(float) for r in stored])
        results[f"transfer_{key}"] = report(name, scores, truth, seed)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "evaluation.json").write_text(json.dumps(results, indent=2))
    print("wrote evaluation.json")


if __name__ == "__main__":
    main()
