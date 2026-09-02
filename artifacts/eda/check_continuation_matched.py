"""Continuation heads on identical states for both arms, split by regime.

The live gate collects its own forks under each arm's BC policy, so T and M were scored on
zero shared states -- T's population held two escape-rich roots, M's was entirely
trap-heavy, and their marginals are not comparable. This uses the fixed 965-root
collection and its whole-seed split, so both arms see the same roots.

Intercept-only calibration, fitted on fit+tune and read once on the held-out roots: the
earlier slopes were near one, and one parameter is far harder to overfit than two here.
"""

from __future__ import annotations

import glob
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from evaluate_death_transfer import DEVICE
from train_phase1b_fork import seed_split

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.state import WorldState
from d4mj.train import _repeat_memory
from d4mj.transition import World, advance, commit_inputs, initial

N = 17
EPS = 1e-6


def within_auc(score, truth):
    out = []
    for s, y in zip(score, truth):
        pos, neg = s[y > 0], s[y == 0]
        if len(pos) and len(neg):
            out.append(float((pos[:, None] > neg[None]).mean()
                             + 0.5 * (pos[:, None] == neg[None]).mean()))
    return float(np.mean(out)) if out else float("nan")


def bce(logit, y):
    p = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
    p = np.clip(p, EPS, 1 - EPS)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


@torch.no_grad()
def readouts(world, heads, config, z_hist, led, true_z):
    spatial, d = config.n_spatial, config.d_spatial
    rng = torch.Generator(device=DEVICE).manual_seed(config.seed + 4242)
    z = z_hist[None].to(DEVICE)
    t = z.shape[1]
    committed, conditioning = commit_inputs(z.view(1, t, spatial, d), rng, config)
    features, _, memory = world(None, led[None].to(DEVICE), committed, conditioning)
    actions = torch.arange(N, device=DEVICE)[:, None]
    state = WorldState(z[:, -1:].view(1, 1, spatial, d).repeat_interleave(N, 0),
                       _repeat_memory(memory, 1, N), t,
                       features[:, -1:].repeat_interleave(N, dim=0))
    _, agent = advance(world, state, actions, rng, config)
    generated = heads(agent)["continuation"][:, 0, 0]
    _, agent_obs = initial(world, true_z.to(DEVICE).view(N, 1, spatial, d), actions,
                           rng, config, _repeat_memory(memory, 1, N), t)
    observed = heads(agent_obs)["continuation"][:, 0, 0]
    return (-generated).cpu().numpy(), (-observed).cpu().numpy()


def main() -> None:
    base = replace(Config(), n_latents=64, d_bottleneck=16)
    cached = torch.load(HERE / "death_transfer_true.pt", weights_only=False)
    true_z, histories, death = cached["true_z"].float(), cached["histories"], cached["death"]
    keys = set()
    for path in sorted(glob.glob(str(HERE / "fork_successors" / "shard-*.pt"))):
        for row in torch.load(path, weights_only=False):
            keys.add((int(row["seed"]), int(row["step"])))
    rows = [r for r in torch.load(HERE / "fork_histories" / "branched_965.pt", weights_only=False)
            if (int(r["seed"]), int(r["step"])) in keys]
    splits = np.array([seed_split(int(r["seed"])) for r in rows])
    lethal = death.sum(1)
    varies = death.max(1) > death.min(1)
    print(f"{len(rows)} roots | fit {int((splits=='fit').sum())} "
          f"tune {int((splits=='tune').sum())} test {int((splits=='test').sum())}", flush=True)

    for arm in ("attention", "mamba"):
        folder = HERE / f"v2_phase2_{arm}"
        saved = replace(base, transition="direct", time_mixer=arm)
        world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
        load(folder / "phase2_final.pt", saved, part0=world, part1=heads)
        world, heads = world.eval(), heads.eval()

        gen, obs = [], []
        for i, row in enumerate(rows):
            g, o = readouts(world, heads, saved, histories[i],
                            row["led_to_action"].long(), true_z[i])
            gen.append(g); obs.append(o)
            if (i + 1) % 400 == 0:
                print(f"  {arm}: {i+1}/{len(rows)}", flush=True)
        gen, obs = np.stack(gen), np.stack(obs)

        fitmask = np.isin(splits, ("fit", "tune"))
        test = splits == "test"
        grid = np.linspace(-10, 10, 4001)
        shift = float(grid[np.argmin([bce(gen[fitmask].ravel() + b, death[fitmask].ravel())
                                      for b in grid])])
        print(f"\n{arm}: intercept {shift:+.3f} fitted on {int(fitmask.sum())} fit+tune roots")
        print(f"{'':<22}{'n':>4}{'BCE gen':>9}{'BCE +b':>9}{'BCE obs':>9}"
              f"{'AUCgen':>8}{'AUCobs':>8}{'AUCfloor':>9}")
        marginal = death[fitmask].mean(0)
        for name, mask in (("all test", test & varies),
                           ("  escape-rich", test & varies & (lethal <= 2)),
                           ("  trap-heavy", test & varies & (lethal >= 14))):
            if mask.sum() < 3:
                print(f"  {name:<20}{int(mask.sum()):>4}  too few"); continue
            floor = np.tile(marginal, (int(mask.sum()), 1))
            print(f"  {name:<20}{int(mask.sum()):>4}{bce(gen[mask], death[mask]):>9.3f}"
                  f"{bce(gen[mask]+shift, death[mask]):>9.3f}{bce(obs[mask], death[mask]):>9.3f}"
                  f"{within_auc(gen[mask], death[mask]):>8.3f}"
                  f"{within_auc(obs[mask], death[mask]):>8.3f}"
                  f"{within_auc(floor, death[mask]):>9.3f}")
        np.savez(folder / "matched_continuation.npz", generated=gen, observed=obs,
                 death=death.numpy() if torch.is_tensor(death) else death,
                 splits=splits, shift=shift)


if __name__ == "__main__":
    main()
