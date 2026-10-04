"""E3a. Are the TC encoder's patch tokens equivalent to the Raw encoder's for a per-tile world? Representation only.

Both token sets are the layer-normed patch tokens the per-tile world consumes (teval caches, 1,002 futures roots;
the diagnosis 70/30 seed split). Measured:
  linear     ridge maps raw -> tc and tc -> raw (one shared 192x192 map for all 81 positions, fitted on 400k train
             tokens), R^2 on test tokens, overall and for map / HUD / player tokens
  cka        linear CKA between the two token matrices (50k test tokens)
  spectrum   per-token covariance: eigenvalue spread (max/min), effective rank (exp entropy), top-10 share
  temporal   within-16-frame-window share of variance of the flattened state (factual futures)
  change     one-step change learnability: per-action local-linear map on [self, 4 neighbours] (learnable.py's
             `local`) fitted on train roots, captured = 1 - err / err(copy) on test roots, by transition class
  facts      teval.Probes on true tokens (identical probe family, fitted per space)
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis"))
import teval as T  # noqa: E402


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def lin(x, y, lam=1e-3):
    x1 = torch.cat([x, torch.ones(len(x), 1)], 1).double()
    w = torch.linalg.solve(x1.T @ x1 + lam * len(x1) * torch.eye(x1.shape[1], dtype=x1.dtype), x1.T @ y.double())
    return lambda q: (torch.cat([q, torch.ones(len(q), 1)], 1).double() @ w).float()


def cka(x, y):
    x, y = x - x.mean(0), y - y.mean(0)
    xy = (x.T.double() @ y.double()).norm() ** 2
    return float(xy / ((x.T.double() @ x.double()).norm() * (y.T.double() @ y.double()).norm()))


def spectrum(x):
    ev = torch.linalg.eigvalsh(torch.cov((x - x.mean(0)).T.double())).clamp_min(1e-12)
    p = ev / ev.sum()
    return {"eigen_spread": float(ev.max() / ev.min()), "effective_rank": float(torch.exp(-(p * p.log()).sum())),
            "top10_share": float(p.flip(0)[:10].sum())}


def main():
    device = torch.device("cuda")
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    caches = {k: T.build_cache(k, device) for k in ("raw", "tc")}
    g = torch.Generator().manual_seed(0)
    frames = lambda c, m: torch.cat([c["ctx"][m], c["fut"][m]], 1).float()           # [n,20,81,192]
    res = {}
    tr = {k: frames(c, train_roots) for k, c in caches.items()}
    te = {k: frames(c, test) for k, c in caches.items()}
    flat_tr = {k: v.flatten(0, 2) for k, v in tr.items()}
    flat_te = {k: v.flatten(0, 2) for k, v in te.items()}
    sub = torch.randperm(len(flat_tr["raw"]), generator=g)[:400_000]
    regions = {"all": slice(None), "map": T.MAP, "hud": list(range(63, 81)), "player": [31]}
    for a, b in (("raw", "tc"), ("tc", "raw")):
        f = lin(flat_tr[a][sub], flat_tr[b][sub])
        out = {}
        for name, idx in regions.items():
            xa, xb = te[a][:, :, idx].flatten(0, 2), te[b][:, :, idx].flatten(0, 2)
            s = torch.randperm(len(xa), generator=g)[:200_000]
            out[name] = r2(f(xa[s]), xb[s])
        res[f"linear_{a}_to_{b}_r2"] = out
    s = torch.randperm(len(flat_te["raw"]), generator=g)[:50_000]
    res["cka"] = cka(flat_te["raw"][s], flat_te["tc"][s])
    res["spectrum"] = {k: spectrum(v[s]) for k, v in flat_te.items()}
    res["temporal_within_window_share"] = {}
    for k, c in caches.items():
        x = c["fut"][test].float().flatten(2)
        res["temporal_within_window_share"][k] = float(((x - x.mean(1, keepdim=True)) ** 2).sum(-1).mean()
                                                       / ((x - x.flatten(0, 1).mean(0)) ** 2).sum(-1).mean())
    # one-step change learnability (local-linear per action), by class
    from onestep import CLASSES, classify
    from learnable import neighbours, ridge, apply
    cls, _ = classify(meta)
    res["change_captured"] = {}
    for k, c in caches.items():
        root = c["ctx"][:, -1].float()
        one = c["one"].float()
        nb = neighbours(root)
        pred = torch.empty_like(one)
        for a in range(17):
            x = nb[train_roots].flatten(0, 1)
            y = (one[train_roots, a] - root[train_roots]).flatten(0, 1)
            idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(a))[:200_000]
            w = ridge(x[idx], y[idx], 1e-3)
            pred[:, a] = apply(nb.flatten(0, 1), w).view(len(root), 81, -1) + root
        err = ((pred - one) ** 2).sum((-1, -2))
        cp = ((root[:, None] - one) ** 2).sum((-1, -2))
        res["change_captured"][k] = {cn: float(1 - err[test[:, None] & (cls == i)].mean() / cp[test[:, None] & (cls == i)].mean())
                                     for i, cn in enumerate(CLASSES)}
    res["facts_true_tokens_k8"] = {}
    for k, c in caches.items():
        p = T.Probes(c, meta, train_roots, train_seeds)
        res["facts_true_tokens_k8"][k] = p.read(c["fut"][test, 7].float(), meta["future_visible"][test, 0, 7])
    (HERE / "tc_equiv.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
