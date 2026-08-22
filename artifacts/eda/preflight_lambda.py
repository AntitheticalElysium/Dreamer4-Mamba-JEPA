"""Is lambda = 2.3761 still the right weight under the corrected protocol?

The original calibration is not reusable. It fed null action histories, measured on
`load_forkset(folder)[:8]` -- the first eight rows in shard order, not restricted to
the fit split -- and built its world under the *tokenizer's* seed (20260732) rather
than the seed the diagnostic trainer actually uses for world initialisation
(Config().seed = 20260731).

This mirrors `train_phase1b_fork` exactly instead: real causal action histories, the
same config, the same world initialisation and RNG streams, and batches drawn from
the trainer's own numpy stream over the fit split at the training batch size.

Registered before the result. The statistic is

    rho = lambda * |g_fork| / |g_ordinary|      at lambda = 2.3761

averaged over BATCHES matched fit batches. Lambda is EQUIVALENT iff mean rho lies in
[0.67, 1.50] -- within 1.5x of parity either way -- and is additionally flagged
UNSTABLE, not a clean pass, if the mean is inside the band while more than 25% of
batches fall outside [0.5, 2.0]. The point estimate is not chased: the question is
whether the existing weight still balances the two terms, not what its optimum is.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from train_phase1b_fork import fork_actions, load_forkset, seed_split

from d4mj.config import Config
from d4mj.train import _share_initialisation
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"
LAMBDA = 2.3761
BATCHES = 16
BAND = (0.67, 1.50)
WIDE = (0.5, 2.0)


def grad_norm(world, loss):
    world.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = sum(float(p.grad.pow(2).sum()) for p in world.parameters() if p.grad is not None)
    world.zero_grad(set_to_none=True)
    return total ** 0.5


def main() -> None:
    suffix, n_latents = "s1fix", 64
    rows = load_forkset(HERE / f"forkset_{suffix}_n{n_latents}")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    fit = np.where(splits == "fit")[0]
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    led_history = fork_actions(rows)

    # identical to the trainer: config, world initialisation, both RNG streams
    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=n_latents, d_bottleneck=16, batch=4, seed=Config().seed)
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    milestone = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if milestone:   # supplementary only: does the imbalance persist after training?
        from d4mj.checkpoint import load
        load(HERE / f"phase1b_{suffix}_n{n_latents}" / f"world_{milestone:06d}.pt",
             config, part0=world)
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 1001)
    draw = np.random.default_rng(config.seed + 91)
    actions = torch.arange(17, device=DEVICE)
    spatial, d = config.n_spatial, config.d_spatial
    steps = history.shape[1]

    print(f"arm {n_latents}x16 {suffix}: world seed {config.seed + 1}, "
          f"{len(fit)} fit roots, batch {config.batch}, {BATCHES} batches", flush=True)
    print(f"{'batch':>6}{'ordinary':>12}{'fork':>12}{'|g_ord|':>12}{'|g_fork|':>12}"
          f"{'ratio':>9}{'rho':>9}")

    records = []
    for index in range(BATCHES):
        chosen = draw.choice(fit, config.batch, replace=False)
        z = history[chosen].to(DEVICE)
        target = branch[chosen].to(DEVICE)
        led = led_history[chosen].to(DEVICE)
        committed, conditioning = commit_inputs(
            z.view(config.batch, steps, spatial, d), rng, config)
        features, _, _ = world(None, led, committed, conditioning)

        ordinary = (world.predict(features[:, :-1], led[:, 1:]).flatten(2)
                    - z[:, 1:]).pow(2).mean()
        last = features[:, -1:]
        fork = (world.predict(last.expand(config.batch, 17, *last.shape[2:]),
                              actions[None].expand(config.batch, -1)).flatten(2)
                - target).pow(2).mean()
        g_ord, g_fork = grad_norm(world, ordinary), grad_norm(world, fork)
        rho = LAMBDA * g_fork / g_ord
        records.append({"ordinary": float(ordinary), "fork": float(fork),
                        "g_ordinary": g_ord, "g_fork": g_fork,
                        "ratio": g_ord / g_fork, "rho": rho})
        print(f"{index:>6}{float(ordinary):12.5f}{float(fork):12.5f}"
              f"{g_ord:12.5f}{g_fork:12.5f}{g_ord / g_fork:9.3f}{rho:9.3f}", flush=True)

    rho = np.array([r["rho"] for r in records])
    outside = float(np.mean((rho < WIDE[0]) | (rho > WIDE[1])))
    equivalent = BAND[0] <= rho.mean() <= BAND[1]
    unstable = equivalent and outside > 0.25
    verdict = "EQUIVALENT" if equivalent and not unstable else (
        "UNSTABLE" if unstable else "NOT EQUIVALENT")
    print(f"\n  mean rho {rho.mean():.3f}  (median {np.median(rho):.3f}, "
          f"min {rho.min():.3f}, max {rho.max():.3f})")
    print(f"  band {BAND}, {outside:.1%} of batches outside {WIDE}  ->  {verdict}")
    print(f"  implied lambda at parity: {LAMBDA / rho.mean():.4f}")
    name = f"lambda_preflight_corrected{f'_{milestone:06d}' if milestone else ''}.json"
    (HERE / name).write_text(json.dumps({
        "milestone": milestone,
        "suffix": suffix, "n_latents": n_latents, "lambda": LAMBDA, "batches": BATCHES,
        "band": list(BAND), "wide": list(WIDE), "mean_rho": float(rho.mean()),
        "median_rho": float(np.median(rho)), "fraction_outside_wide": outside,
        "equivalent": bool(equivalent), "unstable": bool(unstable), "verdict": verdict,
        "implied_lambda_at_parity": float(LAMBDA / rho.mean()),
        "records": records}, indent=2))


if __name__ == "__main__":
    main()
