"""Paired comparisons between evaluated per-tile worlds, with episode-seed-clustered bootstrap intervals.

Reads evals/<tag>_per_root.pt (teval.py). All worlds are scored on the same 1,002 futures roots, so differences are
paired. Resampling unit: the walk seed (143 seeds; roots within a seed are correlated), 2,000 draws, seed 20260927.
Statistics (lower is better for all):
  onestep_<class>   mean squared one-step error / mean copy error, over (root, action) in that transition class
  gen_k             mean imagined error / V at depth k (alive roots), k in 1, 4, 8, 16
Each comparison: A, B, B - A, 95% percentile interval, and whether it excludes 0.
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
CLASSES = ("moved", "blocked", "interact", "sleep", "idle")


def stats(d, idx):
    """idx: root indices (with repetition) -> dict of statistics."""
    out = {}
    err, cp, cls = d["onestep_err"][idx], d["onestep_copy"][idx], d["class"][idx]
    for i, c in enumerate(CLASSES):
        m = cls == i
        out[f"onestep_{c}"] = float(err[m].sum() / cp[m].sum())
    out["onestep_all"] = float(err.sum() / cp.sum())
    g, al = d["gen_err"][idx], d["alive"][idx]
    for k in (1, 4, 8, 16):
        m = al[:, k - 1]
        out[f"gen_{k}"] = float(g[m, k - 1].mean() / d["V"])
    return out


def compare(a, b, draws=2000, seed=20260927):
    da, db = (torch.load(HERE / "evals" / f"{t}_per_root.pt", weights_only=False) for t in (a, b))
    seeds = da["seed"]
    assert torch.equal(seeds, db["seed"]), "different root sets"
    groups = [torch.where(seeds == s)[0] for s in seeds.unique()]
    full = torch.arange(len(seeds))
    sa, sb = stats(da, full), stats(db, full)
    g = torch.Generator().manual_seed(seed)
    diffs = {k: [] for k in sa}
    for _ in range(draws):
        pick = torch.randint(len(groups), (len(groups),), generator=g)
        idx = torch.cat([groups[i] for i in pick])
        xa, xb = stats(da, idx), stats(db, idx)
        for k in sa:
            diffs[k].append(xb[k] - xa[k])
    res = {}
    for k in sa:
        v = torch.tensor(diffs[k])
        lo, hi = float(v.quantile(0.025)), float(v.quantile(0.975))
        res[k] = {"A": sa[k], "B": sb[k], "B_minus_A": sb[k] - sa[k], "ci95": [lo, hi], "resolved": lo > 0 or hi < 0}
    return res


if __name__ == "__main__":
    pairs = [tuple(p.split(":")) for p in sys.argv[1:]]
    out = {}
    for a, b in pairs:
        out[f"{a} -> {b}"] = r = compare(a, b)
        print(f"\n{a} -> {b}")
        for k, v in r.items():
            print(f"  {k:18s} A {v['A']:.3f}  B {v['B']:.3f}  B-A {v['B_minus_A']:+.3f} [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]{' *' if v['resolved'] else ''}")
    path = HERE / "compare.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    path.write_text(json.dumps(old | out, indent=2) + "\n")
