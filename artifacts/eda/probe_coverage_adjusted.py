"""Does coverage distance explain residual error independently of transition difficulty?

The atlas showed residual energy correlating +0.60 and +0.64 with distance to the
nearest fit root. But distance also travels with effect magnitude, action identity,
timestep and light level, so the marginal correlation cannot tell coverage apart from
"large transitions are simply harder". This fits

    log(E + eps) ~ nearest_fit_distance + log(true_effect + eps) + action
                   + timestep + light + successor_class_count

and asks what is left of the distance coefficient once the difficulty controls are in.

Uncertainty is a root-clustered bootstrap: observations within a root share a state and
are not independent, so whole roots are resampled rather than (root, action) pairs.

Read-only. Everything here is recomputed from the existing checkpoint and caches.
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
from evaluate_phase1b_fork import predict_branches
from probe_action_matching import classes_of
from probe_decoder_ceiling import cache as cache_backbone
from probe_hidden_ceiling import cache_hidden
from train_phase1b_fork import (MixerWorld, fork_actions, load_forkset,
                                load_promoted, seed_split)

from d4mj.config import Config

DEVICE = "cuda"
SUFFIX = "abm0"
EPS = 1e-6
LIGHT, TIMESTEP = 5, 1134          # visible[5] is light_level, hidden[4] is timestep


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def ols(design, y):
    return np.linalg.lstsq(design, y, rcond=None)[0]


def main() -> None:
    rows = load_forkset(HERE / f"forkset_{SUFFIX}_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    keys = [(int(r["seed"]), int(r["step"])) for r in rows]

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = MixerWorld(config).to(DEVICE)
    load_promoted(HERE / f"phase1b_{SUFFIX}_n64" / "world_020000.pt", world)
    world.eval()
    for p in world.parameters():
        p.requires_grad_(False)
    led = fork_actions(rows)

    predicted = predict_branches(world, config, history, led).float()
    residual = ((branch - branch.mean(1, keepdim=True))
                - (predicted - predicted.mean(1, keepdim=True)))
    effect = (branch - branch.mean(1, keepdim=True))

    fit_idx = np.where(splits == "fit")[0]
    test_idx = np.where(splits == "test")[0]
    spaces = {"pooled": cache_backbone(world, config, history, led, "pooled").flatten(1).float()}
    raw = cache_hidden(world, config, history, led)
    with torch.no_grad():
        spaces["hidden"] = torch.cat(
            [world.mix_norm(raw[lo:lo+64].to(DEVICE).float()).half().cpu()
             for lo in range(0, len(raw), 64)]).float().mean(1).flatten(1)
    del raw

    state = {(int(r["seed"]), int(r["step"])): r for r in
             torch.load(HERE / "state_features" / "features.pt", weights_only=False)}
    both = torch.stack([torch.cat([state[k]["visible"], state[k]["hidden"]]) for k in keys]).float()

    eff_test = effect[test_idx]
    gram = torch.cdist(eff_test, eff_test).pow(2).numpy()
    n_classes = np.array([classes_of(gram[i]).max() + 1 for i in range(len(gram))], dtype=float)

    energy = residual[test_idx].pow(2).sum(-1).numpy()            # (n, 17)
    magnitude = eff_test.pow(2).sum(-1).numpy()
    n_roots, n_actions = energy.shape
    root_of = np.repeat(np.arange(n_roots), n_actions)

    y = np.log(energy.reshape(-1) + EPS)
    action_dummies = np.eye(n_actions)[np.tile(np.arange(n_actions), n_roots)][:, 1:]
    controls = np.column_stack([
        zscore(np.log(magnitude.reshape(-1) + EPS)),
        action_dummies,
        zscore(np.repeat(both[test_idx][:, TIMESTEP].numpy(), n_actions)),
        zscore(np.repeat(both[test_idx][:, LIGHT].numpy(), n_actions)),
        zscore(np.repeat(n_classes, n_actions)),
    ])

    result = {"arm": SUFFIX, "observations": len(y), "roots": n_roots}
    generator = np.random.default_rng(20260824)
    for name, space in spaces.items():
        f = torch.nn.functional.normalize(space[fit_idx], dim=1)
        t = torch.nn.functional.normalize(space[test_idx], dim=1)
        near = torch.cat([(1 - t[lo:lo+256] @ f.T).min(1).values
                          for lo in range(0, len(t), 256)]).numpy()
        distance = zscore(np.repeat(near, n_actions))

        bare = np.column_stack([np.ones_like(y), distance])
        full = np.column_stack([np.ones_like(y), distance, controls])
        beta_bare, beta_full = ols(bare, y)[1], ols(full, y)[1]

        draws = []
        for _ in range(400):
            pick = generator.integers(0, n_roots, n_roots)
            take = (pick[:, None] * n_actions + np.arange(n_actions)).reshape(-1)
            draws.append([ols(bare[take], y[take])[1], ols(full[take], y[take])[1]])
        draws = np.array(draws)
        band = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]

        resid_full = y - full @ ols(full, y)
        without = np.column_stack([np.ones_like(y), controls])
        resid_without = y - without @ ols(without, y)
        partial = 1 - resid_full.var() / resid_without.var()

        result[name] = {
            "beta_unadjusted": float(beta_bare), "beta_unadjusted_ci": band(draws[:, 0]),
            "beta_adjusted": float(beta_full), "beta_adjusted_ci": band(draws[:, 1]),
            "attenuation": float(1 - beta_full / beta_bare) if beta_bare else float("nan"),
            "partial_r2_of_distance": float(partial),
        }
        r = result[name]
        print(f"{name:<8} distance coefficient on log residual energy, per SD")
        print(f"         unadjusted {r['beta_unadjusted']:+.4f} "
              f"[{r['beta_unadjusted_ci'][0]:+.4f}, {r['beta_unadjusted_ci'][1]:+.4f}]")
        print(f"         adjusted   {r['beta_adjusted']:+.4f} "
              f"[{r['beta_adjusted_ci'][0]:+.4f}, {r['beta_adjusted_ci'][1]:+.4f}]"
              f"   attenuation {r['attenuation']:.1%}"
              f"   partial R2 {r['partial_r2_of_distance']:.4f}", flush=True)

    (HERE / f"coverage_adjusted_{SUFFIX}.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
