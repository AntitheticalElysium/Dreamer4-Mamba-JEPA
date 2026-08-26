"""The repaired Direct stack, trained through the real production objective.

Every recent Direct experiment used the diagnostic objective -- teacher-forced MSE plus
a one-step all-17 fork term -- which never exercises the generated-prefix path that
production `_direct_loss` actually trains:

    teacher forcing + first generated successor + second generated-prefix successor

So we do not yet know whether the repaired encoder and the promoted mixer solve the
problem Direct faces during generation. This runs them through it.

  encoder     64x16, capacity6k/n64d16_s1 at 6,000 steps -- the repaired geometry
  transition  direct, attention, with the promoted action-token mixer and the tanh
  corpus      the expert archive plus support-v2, on its declared train/dev split.
              `run_stage_a` still defaults to support-v1 because prior Stage-A results
              are digest-bound to it, but 22ff300 collected v2 precisely because v1's
              400 terminal events "could not resolve the 0.03-0.05 effects every recent
              experiment turned on" -- which is the range our repairs live in, and the
              corpus every diagnostic here was built on.
  objective   production transition_loss, unchanged
  budget      the production default of 20,000 dynamics steps

Nothing new: no Flow, no extra loss term, no additional data. The latent cache has to
be rebuilt because the repaired geometry is 64 slots and the existing cache is 32, and
it is written to a resumable mmap store rather than held in memory -- 3.18M frames at
1024 wide is 13 GB.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts"))
from run_stage_a import corpus

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.representation import Encoder
from d4mj.train import cache_latents, cache_latents_to_store, train_dynamics

ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--expert", type=int, default=320)
    parser.add_argument("--out", type=Path, default=HERE / "production_1b")
    parser.add_argument("--cache", type=Path, default=HERE / "latent_cache_64")
    parser.add_argument("--support", type=Path,
                        default=ROOT / "artifacts/craftax_support_v2")
    parser.add_argument("--seed", type=int, default=None,
                        help="the training seed only -- initialisation, sampler and "
                             "model noise. The corpus split stays on the default seed: "
                             "`episode_splits` keys off `config.seed`, so moving it "
                             "would reshuffle train/dev and put evaluation episodes into "
                             "training, which is not what a training-seed replication "
                             "asks. It also keeps the latent cache reusable.")
    parser.add_argument("--smoke", type=int, default=0,
                        help="episodes per split for a shape/plumbing check; skips the "
                             "bootstrap-start guard since it is not a real run")
    args = parser.parse_args()
    # `main` chdirs to ROOT so `run_stage_a` can resolve its repo-relative corpora, which
    # silently invalidates any relative path taken from the command line afterwards
    args.out, args.cache = args.out.resolve(), args.cache.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.time() - started:7.0f}s] {message}"
        print(line, flush=True)
        (args.out / "run.log").open("a").write(line + "\n")

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    config = replace(base, transition="direct", time_mixer="attention")
    if args.seed is not None:            # `base` keeps the default: split and cache
        config = replace(config, seed=args.seed)
    if not args.smoke and args.steps <= base.bootstrap_start:
        raise SystemExit(f"--steps must exceed bootstrap_start={base.bootstrap_start}")

    os.chdir(ROOT)          # run_stage_a resolves ARCHIVE and SUPPORT repo-relative
    train_set, dev_set = corpus(base, args.expert, log, support=args.support)
    if args.smoke:
        train_set, dev_set = list(train_set)[: args.smoke], list(dev_set)[: args.smoke]
    log(f"corpus: {len(train_set)} train, {len(dev_set)} dev episodes")

    # the repaired tokenizer, restored under the config it was written with
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(base.device)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]), part0=encoder)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    log(f"encoder restored from {ENCODER.name} ({stored['steps']} steps, batch {stored['batch']})")

    args.cache.mkdir(parents=True, exist_ok=True)
    cached_train = cache_latents_to_store(
        encoder, train_set, base, args.cache,
        source_contract={"kind": "eda_production_1b", "encoder": str(ENCODER)})
    log(f"train latents cached to {args.cache}")
    cached_dev = cache_latents(encoder, dev_set, base)
    log(f"dev latents cached in memory, digest {cached_train[0].latent_digest}")

    encoder.cpu()
    torch.cuda.empty_cache()
    world = train_dynamics(cached_train, args.steps, config,
                           checkpoint=args.out / "phase1b.pt")
    log(f"phase 1B done, {args.steps} steps under the production objective")
    torch.save({"world": world.state_dict()}, args.out / "world.pt")
    (args.out / "done.json").write_text(json.dumps(
        {"steps": args.steps, "seed": config.seed, "seconds": time.time() - started,
         "train_episodes": len(cached_train), "dev_episodes": len(cached_dev),
         "encoder": str(ENCODER)}, indent=2))


if __name__ == "__main__":
    main()
