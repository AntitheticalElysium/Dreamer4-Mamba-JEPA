"""Two fit subsets of equal size, differing only in coverage.

The adjusted regression showed distance from fit coverage keeps an independent
association with residual error. That is observational: distance may still proxy for a
kind of transition difficulty we did not measure. This builds the causal test -- vary
coverage while holding the amount of data fixed.

  random    1,825 fit roots drawn uniformly
  spread    1,825 fit roots by k-center greedy in pooled space, which maximises the
            minimum distance between selected roots

Selection uses fit roots only and never looks at tune or test. Both subsets are the
same size, so any difference in the trained model is attributable to which states are
covered rather than how many.

Note on the selection space: pooled features come from the promoted mixer checkpoint,
which was itself trained on all 3,651 fit roots. That is not test leakage -- selection
stays inside fit -- but it does mean `spread` is chosen with a representation that has
seen the whole fit set. The frozen root latent z_t is an alternative selection space
that depends on no dynamics training at all; it is written out here too so the same
experiment can be re-run without that caveat.
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
from probe_decoder_ceiling import cache as cache_backbone
from legacy import open_checkpoint
from train_phase1b_fork import fork_actions, load_forkset, seed_split

from d4mj.config import Config

DEVICE = "cuda"
BUDGET = 1825


def k_center(features, budget, seed):
    """Greedy farthest-point selection: each step adds the least-covered root."""
    x = torch.nn.functional.normalize(features, dim=1)
    generator = np.random.default_rng(seed)
    picked = [int(generator.integers(0, len(x)))]
    nearest = (1 - x @ x[picked[0]])
    for _ in range(budget - 1):
        nxt = int(nearest.argmax())
        picked.append(nxt)
        nearest = torch.minimum(nearest, 1 - x @ x[nxt])
    return np.array(picked)


def main() -> None:
    rows = load_forkset(HERE / "forkset_abm0_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    fit = np.where(splits == "fit")[0]
    history = torch.stack([r["z_history"] for r in rows])

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)
    world = open_checkpoint(HERE / "phase1b_abm0_n64" / "world_020000.pt",
                            config, "promoted")
    pooled = cache_backbone(world, config, history, fork_actions(rows), "pooled").flatten(1).float()

    spaces = {"pooled": pooled[fit], "root_latent": history[fit][:, -1].float()}
    out = {}
    for name, space in spaces.items():
        chosen = fit[k_center(space, BUDGET, seed=20260824)]
        out[f"spread_{name}"] = chosen.tolist()
        x = torch.nn.functional.normalize(space, dim=1)
        sel = torch.nn.functional.normalize(space[np.searchsorted(fit, chosen)], dim=1)
        spread = float((1 - x @ sel.T).min(1).values.mean())
        out[f"spread_{name}_mean_distance_to_selection"] = spread
        print(f"{name:<12} spread subset: mean distance from all fit roots to the "
              f"selection {spread:.5f}", flush=True)

    generator = np.random.default_rng(20260824)
    random = np.sort(generator.permutation(fit)[:BUDGET])
    out["random"] = random.tolist()
    x = torch.nn.functional.normalize(pooled[fit], dim=1)
    sel = torch.nn.functional.normalize(pooled[random], dim=1)
    out["random_mean_distance_to_selection"] = float((1 - x @ sel.T).min(1).values.mean())
    print(f"{'random':<12} subset: mean distance {out['random_mean_distance_to_selection']:.5f} "
          f"(lower means the fit set is better covered)", flush=True)
    out["budget"] = BUDGET
    out["fit_total"] = len(fit)

    (HERE / "coverage_subsets.json").write_text(json.dumps(out))
    print(f"wrote coverage_subsets.json: {BUDGET} of {len(fit)} fit roots per arm")


if __name__ == "__main__":
    main()
