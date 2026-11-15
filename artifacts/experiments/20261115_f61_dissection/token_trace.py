"""F6.1 dissection, check 11 (PLAN.md): can F6.1's own tokens feed a generic slow trace that times the awake clocks?
Codex's analytic leaky recurrence (`../20261115_mamba_timescales/`, time constants 8 and 32), with the per-cell scalar
taken as the cell's own squared token change (check 8: 7-11 at a digit change, ~0.002 otherwise) in place of a fitted
luminance projection. A shared label-free predictor (LightGBM, Codex's settings) of "this cell's token changes at the
next frame" (squared change > 1) reads the current change, the two traces and the cell index; trained on 400 DEV
episodes of folds 0-3, read on fold 4 at the clock events of the after_reset cohorts (the digit's cell), full-prefix
traces against traces started 8 frames back.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_f61_dissection.token_trace
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
HERE = Path(__file__).resolve().parent
CELL = {"food_down": 64, "drink_down": 65, "energy_down_awake": 66}


def traces(x, start=0):
    """EMAs with time constants 8 and 32 of x [T, 81], from frame `start` (zero before)."""
    out = np.zeros((len(x), 81, 2), np.float32)
    e = np.zeros((81, 2), np.float32)
    for t in range(len(x)):
        if t < start:
            continue
        e = e + (x[t][:, None] - e) * np.array([1 / 8, 1 / 32], np.float32)
        out[t] = e
    return out


@torch.no_grad()
def main():
    import lightgbm as lgb
    torch.set_num_threads(6)
    model, cfg = G.CR.NP.IF.load((G.EXP / "20261114_p1f61/run/ckpt_10.pt").resolve())
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, ok = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = G.S.episodes(L)
    train_eps = [e for e in range(len(eps)) if e % 5 != 4][:400]
    test_eps = [e for e in range(len(eps)) if e % 5 == 4]
    rng = np.random.default_rng(0)
    X, Y, rows = [], [], {"full": [], "short": []}
    for e in train_eps + test_eps:
        fr = torch.as_tensor(G.D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy())
        p = torch.cat([G.C.coarse(model.perception, z)[0] for z in fr.split(512)]).float().numpy()
        x = np.r_[np.zeros((1, 81), np.float32), np.square(p[1:] - p[:-1]).sum(-1)]   # x[t]: change into frame t
        tr = traces(x)
        T = len(p)
        if e in train_eps:                                              # generic targets: every cell, sampled frames
            ts = rng.choice(np.arange(1, T - 1), min(T - 2, 120), replace=False)
            for t in ts:
                X.append(np.c_[x[t], tr[t], tr[t, :, 0] - tr[t, :, 1], np.arange(81)])
                Y.append(x[t + 1] > 1.0)
        else:                                                           # the clock events, digit cell, both variants
            for t in range(9, T):                                       # state t: transition t-1 -> t, features through t-1
                g = off[e] + t
                for j, (n, c) in enumerate(CELL.items()):
                    if not after[g, j]:
                        continue
                    for v in ("full", "short"):
                        tv = tr[t - 1] if v == "full" else traces(x[t - 9:t], start=1)[-1]
                        rows[v].append((j, int(y[g, j]), e, *x[t - 1, c:c + 1], *tv[c], tv[c, 0] - tv[c, 1], c))
    X, Y = np.concatenate(X), np.concatenate(Y)
    par = dict(objective="binary", learning_rate=.06, num_leaves=31, min_data_in_leaf=50, num_threads=6, verbosity=-1, seed=0)
    m = lgb.train(par, lgb.Dataset(X, Y.astype(float)), num_boost_round=200)
    res = {"train_cells": int(len(Y)), "train_change_rate": float(Y.mean())}
    for v in ("full", "short"):
        R = np.array(rows[v], dtype=np.float64)
        s = m.predict(R[:, 3:])
        for j, n in enumerate(CELL):
            k = R[:, 0] == j
            res.setdefault(n, {})[v] = {"rows": int(k.sum()), "events": int(R[k, 1].sum()), "ap": G.AP(R[k, 1].astype(bool), s[k])}
    for n in CELL:
        print(n, json.dumps(res[n]), flush=True)
    (HERE / "token_trace.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
