"""Are the two Phase-1B loss terms comparably normalised, and what is lambda?

Both terms must be means over target scalars, not sums, or the fork term's weight
would depend on how many successors a root has and on the latent width -- which
differs between the two geometries and would silently make lambda geometry-dependent.
Reports the per-scalar normalisation and the lambda=1 gradient ratio for both arms so
a single fixed lambda can be chosen once and applied to both.
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
from d4mj.config import Config
from d4mj.train import _share_initialisation
from d4mj.transition import World, commit_inputs

DEVICE = "cuda"


def losses_for(arm_dir: Path, rows, config):
    """Ordinary teacher-forced dynamics loss and the all-17 fork loss, both as means
    over target scalars so they are on one scale regardless of geometry."""
    torch.manual_seed(config.seed + 1)
    world = _share_initialisation(World(config), config).to(DEVICE)
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 1001)
    spatial, d = config.n_spatial, config.d_spatial

    history = torch.stack([r["z_history"] for r in rows]).to(DEVICE)
    branch = torch.stack([r["z_branch"] for r in rows]).to(DEVICE)
    steps = history.shape[1]
    led = torch.full((len(rows), steps), config.n_actions, dtype=torch.long, device=DEVICE)
    committed, conditioning = commit_inputs(history.view(len(rows), steps, spatial, d),
                                            rng, config)
    features, _, _ = world(None, led, committed, conditioning)

    # ordinary: predict z_{t+1} from block t, teacher forced across the window
    ordinary_pred = world.predict(features[:, :-1], led[:, 1:]).flatten(2)
    ordinary = (ordinary_pred - history[:, 1:]).pow(2).mean()

    # fork: predict all 17 successors from the final block
    last = features[:, -1:]
    actions = torch.arange(17, device=DEVICE)[None].expand(len(rows), -1)
    fork_pred = world.predict(last.expand(len(rows), 17, *last.shape[2:]), actions).flatten(2)
    fork = (fork_pred - branch).pow(2).mean()
    return world, ordinary, fork


def grad_norm(world, loss):
    world.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    total = sum(float(p.grad.pow(2).sum()) for p in world.parameters() if p.grad is not None)
    world.zero_grad(set_to_none=True)
    return total ** 0.5


def main() -> None:
    print(f"{'arm':<8}{'z dim':>7}{'ordinary':>12}{'fork':>12}"
          f"{'|g_ord|':>12}{'|g_fork|':>12}{'ratio':>9}")
    summary = {}
    for n_latents in (32, 64):
        folder = HERE / f"forkset_s1_n{n_latents}"
        manifest = json.loads((folder / "manifest.json").read_text())
        from train_phase1b_fork import load_forkset

        rows = load_forkset(folder)[:8]
        report = json.loads((HERE / "capacity6k" /
                             f"n{n_latents}d16_s1" / "training_report.json").read_text())
        config = replace(Config(transition="direct", time_mixer="attention"),
                         n_latents=n_latents, d_bottleneck=16, seed=report["seed"])
        torch.cuda.empty_cache()
        world, ordinary, fork = losses_for(folder, rows, config)
        g_ord, g_fork = grad_norm(world, ordinary), grad_norm(world, fork)
        print(f"{f'{n_latents}x16':<8}{manifest['z_dim']:>7}{float(ordinary):>12.5f}"
              f"{float(fork):>12.5f}{g_ord:>12.5f}{g_fork:>12.5f}{g_ord/g_fork:>9.3f}")
        summary[f"{n_latents}x16"] = {"ordinary": float(ordinary), "fork": float(fork),
                                      "grad_ordinary": g_ord, "grad_fork": g_fork,
                                      "ratio": g_ord / g_fork, "z_dim": manifest["z_dim"]}
        del world
        torch.cuda.empty_cache()
    print("\nboth losses are means over target scalars, so neither depends on latent")
    print("width or successor count; the ratio below is what lambda must offset.")
    ratios = [v["ratio"] for v in summary.values()]
    print(f"lambda = {sum(ratios)/len(ratios):.3f} equalises gradient magnitude on average "
          f"(per-arm {ratios[0]:.3f}, {ratios[1]:.3f})")
    (HERE / "lambda_preflight.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
