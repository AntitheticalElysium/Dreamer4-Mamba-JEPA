"""Temporal memory pass, measurement 4 (PLAN.md): fixed readable bases (S4D-Lin, HiPPO-LegT) on the graded change and on
the raw 16-channel token of each digit cell, with measurement 1's rows, readers and references.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.ssm_tokens
"""
import importlib
import json

import numpy as np
import torch

R = importlib.import_module("artifacts.experiments.20261116_temporal_memory.readability")
SB = importlib.import_module("artifacts.experiments.20261116_temporal_memory.ssm_bases")
G, HERE, RUN = R.G, R.HERE, R.RUN


@torch.no_grad()
def cell_tokens():
    """F6.1's coarse tokens at the digit cells, per DEV episode of measurement 1 (cached)."""
    f = RUN / "tok_cells.npz"
    if f.exists():
        return dict(np.load(f))
    model, _ = G.CR.NP.IF.load(R.TF.CKPT.resolve())
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    out = {}
    for e, ep in enumerate(G.S.episodes(L)):
        if str(e) in R.X:
            fr = G.D.RP.episode_fields(ep.shard, ep.slot)["observations"]
            dev = next(model.parameters()).device
            p = torch.cat([G.C.coarse(model.perception, z.to(dev))[0] for z in fr.split(512)]).float().cpu().numpy()
            out[str(e)] = p[:, R.CELLS]
            assert np.allclose(np.r_[np.zeros((1, 4)), np.square(p[1:] - p[:-1]).sum(-1)][:, R.CELLS], R.X[str(e)], atol=1e-3)
    np.savez(f, **out)
    return out


def record(u, name):
    """u [T, C] -> fixed basis state per channel, concatenated [T, C * d] (dense LegT: all channels per step)."""
    a, b = SB.BASES[name]
    if a.ndim == 2:
        s, out = np.zeros((len(b), u.shape[1])), np.zeros((len(u), u.shape[1], len(b)))
        for t in range(len(u)):
            s = a @ s + b[:, None] * u[t][None]
            out[t] = s.T
        return out.reshape(len(u), -1)
    return np.concatenate([SB.code(u[:, c], name) for c in range(u.shape[1])], 1)


def main():
    TOK = cell_tokens()
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, _ = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    names = {"lin_x": ("s4d_lin", "x"), "legt_x": ("legt_64", "x"), "lin_tok": ("s4d_lin", "tok"),
             "legt_tok": ("legt_64", "tok")}
    tr, te, rng = {}, {}, np.random.default_rng(0)                      # measurement 1's row draws, line for line
    for e, x in R.X.items():
        T = len(x)
        ts_tr = rng.choice(np.arange(1, T - 1), min(T - 2, 120), replace=False)
        for name, (basis, inp) in names.items():
            for col_name, (col, cell) in R.EVENTS.items():
                j = R.CELLS.index(cell)
                u = x[:, j:j + 1].astype(np.float64) if inp == "x" else TOK[e][:, j].astype(np.float64)
                c = record(u, basis)
                if int(e) in R.TRAIN:
                    if col_name != "energy_up_asleep":                  # cell 66's reader is shared by both energy events
                        tr.setdefault((name, j), []).append((c[ts_tr].astype(np.float32), x[ts_tr + 1, j] > 1.0))
                else:
                    ts = np.array([t for t in range(9, T) if after[off[int(e)] + t, col]], int)
                    if len(ts):
                        te.setdefault((name, col_name), []).append((c[ts - 1], y[off[int(e)] + ts, col]))
    ref = json.loads((HERE / "readability.json").read_text())["aps"]
    res = {}
    for name in names:
        res[name] = {}
        for n, (col, cell) in R.EVENTS.items():
            j = R.CELLS.index(cell)
            Xtr = np.concatenate([a[0] for a in tr[(name, j)]]); ytr = np.concatenate([a[1] for a in tr[(name, j)]])
            Xte = np.concatenate([a[0] for a in te[(name, n)]]); yte = np.concatenate([a[1] for a in te[(name, n)]])
            lin = G.AP(yte, R.logistic(Xtr, ytr, Xte))
            res[name][n] = {"linear": lin, "nonlinear": G.AP(yte, R.gbm(Xtr, ytr, Xte)),
                            "linear_fraction_of_recency": lin / ref["csc"][n]["linear"], "rows": int(len(yte))}
            print(name, n, json.dumps(res[name][n]), flush=True)
    (HERE / "ssm_tokens.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
