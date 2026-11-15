"""Temporal memory pass, measurement 2 (PLAN.md): readability of fixed SSM bases (HiPPO-LegT / LMU, S4D-Lin, S4D-Real)
of each digit cell's event train, with measurement 1's rows, readers and references.

    GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261116_temporal_memory.ssm_bases
"""
import importlib
import json

import numpy as np
from scipy.signal import cont2discrete, lfilter

R = importlib.import_module("artifacts.experiments.20261116_temporal_memory.readability")
G, HERE = R.G, R.HERE
THETA, DT = 64.0, 1.0 / 32


def legt(N):
    """HiPPO-LegT, LMU form (HiPPO eq. translated-legendre-dynamics), ZOH at dt = 1: c_t = Ad c_{t-1} + Bd f_t."""
    n = np.arange(N)[:, None]; k = np.arange(N)[None, :]
    A = (2 * n + 1) * np.where(n >= k, (-1.0) ** (n - k), 1.0) / THETA
    B = ((2 * np.arange(N) + 1) * (-1.0) ** np.arange(N) / THETA)[:, None]
    Ad, Bd, *_ = cont2discrete((-A, B, np.eye(N), np.zeros((N, 1))), 1.0, method="zoh")
    return Ad, Bd[:, 0]


def s4d(A):
    """The S4D listing's bilinear discretisation of a diagonal A with B = 1, at dt = DT."""
    return (1 + DT * A / 2) / (1 - DT * A / 2), DT / (1 - DT * A / 2)


BASES = {"legt_16": legt(16), "legt_64": legt(64), "s4d_lin": s4d(-0.5 + 1j * np.pi * np.arange(32)),
         "s4d_real": s4d(-(np.arange(64) + 1.0) + 0j)}


def code(ev, name):
    """ev [T] -> state after each frame [T, d] (real features)."""
    a, b = BASES[name]
    if a.ndim == 2:                                                     # dense LegT
        c, out = np.zeros(len(b)), np.zeros((len(ev), len(b)))
        for t, f in enumerate(ev):
            c = a @ c + b * f
            out[t] = c
        return out
    s = np.stack([lfilter([bi], [1.0, -ai], ev.astype(complex)) for ai, bi in zip(a, b)], 1)
    return np.concatenate([s.real, s.imag], 1) if np.abs(a.imag).max() > 0 else s.real


def codes(x):
    ev = (x > 1.0).astype(np.float64)
    return {name: np.stack([code(ev[:, c], name) for c in range(4)], 1) for name in BASES}


def main():
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, _ = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    tr, te, rng = {}, {}, np.random.default_rng(0)                      # measurement 1's row draws, line for line
    for e, x in R.X.items():
        e, C = int(e), codes(x)
        T = len(x)
        ts_tr = rng.choice(np.arange(1, T - 1), min(T - 2, 120), replace=False)
        for name, c in C.items():
            if e in R.TRAIN:
                for j in range(4):
                    tr.setdefault((name, j), []).append((c[ts_tr, j].astype(np.float32), x[ts_tr + 1, j] > 1.0))
            else:
                for n, (col, cell) in R.EVENTS.items():
                    j = R.CELLS.index(cell)
                    ts = np.array([t for t in range(9, T) if after[off[e] + t, col]], int)
                    if len(ts):
                        te.setdefault((name, n), []).append((c[ts - 1, j], y[off[e] + ts, col]))
    ref = json.loads((HERE / "readability.json").read_text())["aps"]
    res = {}
    for name in BASES:
        res[name] = {}
        for n, (col, cell) in R.EVENTS.items():
            j = R.CELLS.index(cell)
            Xtr = np.concatenate([a[0] for a in tr[(name, j)]]); ytr = np.concatenate([a[1] for a in tr[(name, j)]])
            Xte = np.concatenate([a[0] for a in te[(name, n)]]); yte = np.concatenate([a[1] for a in te[(name, n)]])
            lin = G.AP(yte, R.logistic(Xtr, ytr, Xte))
            res[name][n] = {"linear": lin, "nonlinear": G.AP(yte, R.gbm(Xtr, ytr, Xte)),
                            "linear_fraction_of_recency": lin / ref["csc"][n]["linear"], "rows": int(len(yte))}
            print(name, n, json.dumps(res[name][n]), flush=True)
    (HERE / "ssm_bases.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
