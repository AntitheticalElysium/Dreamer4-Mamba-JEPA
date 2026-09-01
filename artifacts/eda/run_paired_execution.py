"""Native-horizon paired execution: every actor against its own BC, on shared DEV seeds.

The imagination metrics say what the model believes; this says what the policy does in
Craftax. Every policy runs the same 512 DEV seeds at the native 10000-step cap, so the
comparisons are paired, and the raw episode rows are kept for re-analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.execution import Result, evaluate, run_episode, run_random
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=HERE / "v2_paired_execution")
    parser.add_argument("--seed-base", type=int, default=30_000)
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument("--limit", type=int, default=10_000)
    parser.add_argument("--arms", nargs="+", default=["attention", "mamba"])
    parser.add_argument("--budgets", nargs="+", default=["", "_10k"])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    encoder.eval()

    present = {}
    for arm in args.arms:
        saved = replace(base, transition="direct", time_mixer=arm)
        config = replace(saved, horizon=saved.direct_rollout)
        world, prior = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
        load(HERE / f"v2_phase2_{arm}" / "phase2_final.pt", saved, part0=world, part1=prior)
        world.eval()
        present[f"{arm}_bc"] = (world, prior.eval(), config)
        for tag in args.budgets:
            folder = HERE / f"v2_phase3_{arm}{tag}"
            if not (folder / "phase3_final.pt").exists():
                print(f"skipping {folder.name}: no phase3_final.pt", flush=True)
                continue
            steps = json.loads((folder / "training_report.json").read_text())["steps"]
            actor = Heads(saved).to(DEVICE)
            load(folder / "phase3_final.pt", config, part0=World(saved).to(DEVICE), part1=actor)
            present[f"{arm}_actor{steps}"] = (world, actor.eval(), config)

    # Every episode is written before the next one starts, so an interruption hours in
    # costs one episode and a rerun picks up where it stopped. The same store also lets
    # a 64-seed preliminary pass be extended to 512 without re-executing anything.
    store = args.out / "episodes.pt"
    rows = torch.load(store, weights_only=False) if store.exists() else {}

    def cached(name, run):
        done = rows.setdefault(name, {})
        def episode(seed: int) -> Result:
            if seed not in done:
                done[seed] = asdict(run(seed))
                temporary = store.with_suffix(".pt.tmp")
                torch.save(rows, temporary)
                temporary.replace(store)
                if len(done) % 16 == 0:
                    print(f"  {name}: {len(done)} episodes", flush=True)
            return Result(**done[seed])
        return episode

    policies = {name: cached(name, (lambda w, h, c: lambda seed: run_episode(
        w, encoder, h, seed, c, limit=args.limit))(*value))
        for name, value in present.items()}
    policies["random"] = cached("random", lambda seed: run_random(seed, base, limit=args.limit))

    seeds = list(range(args.seed_base, args.seed_base + args.episodes))
    resumed = sum(len(set(seeds) & set(rows.get(name, {}))) for name in policies)
    print(f"executing {len(policies)} policies x {len(seeds)} DEV seeds "
          f"(native cap {args.limit}, {resumed} cached): {', '.join(policies)}", flush=True)
    start = time.time()
    scores = evaluate(policies, seeds, base)
    print(f"executed in {(time.time() - start) / 60:.1f} min", flush=True)

    for entry in scores.values():
        entry.pop("episodes")
    (args.out / f"paired_execution_{args.episodes}.json").write_text(json.dumps(
        {"seed_base": args.seed_base, "episodes": args.episodes, "limit": args.limit,
         "policies": list(policies), "evaluation": scores}, indent=2, default=float))

    print(f"\n{'policy':<20}{'achieve':>9}{'score':>9}{'reward':>9}"
          f"{'term':>7}{'length':>9}{'vs own BC achievements (95%)':>34}")
    for name, entry in scores.items():
        control = f"{name.split('_')[0]}_bc"
        versus = entry.get(f"versus_{control}") if control in scores else None
        gap = (f"{versus['achievements_gap']:+.3f} "
               f"[{versus['achievements_interval'][0]:+.3f},"
               f"{versus['achievements_interval'][1]:+.3f}]" if versus else "")
        print(f"{name:<20}{entry['achievements']:>9.3f}{entry['score']:>9.3f}"
              f"{entry['reward']:>9.3f}{entry['terminated']:>7.3f}"
              f"{entry['length']:>9.1f}{gap:>34}")


if __name__ == "__main__":
    main()
