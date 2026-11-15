"""Temporal memory pass, measurement 1 (PLAN.md): how readable is each per-cell time code for the awake clocks?

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261116_temporal_memory.readability
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.signal import lfilter

HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
UC = importlib.import_module("artifacts.experiments.20261115_f61_dissection.use_check")
CK = importlib.import_module("artifacts.experiments.20261116_temporal_memory.cogrnn_kabir2025")
G, TT, TF = UC.G, UC.TT, UC.TF
CELLS = [63, 64, 65, 66]
CAP = 63
EVENTS = {"food_down": (0, 64), "drink_down": (1, 65), "energy_down_awake": (2, 66), "energy_up_asleep": (3, 66)}
LAPLACE = {"paper_8_k8": (1, 1000, 8, 8), "repo_80_k4": (1, 1000, 80, 4), "repo_80_k8": (1, 1000, 80, 8),
           "repo_80_k16": (1, 1000, 80, 16)}                              # (tau*_min, tau*_max, time cells, k)
LAGS = {"energy_30_22": (30, 22), "food_25_17": (25, 17), "drink_20_12": (20, 12), "energy_30_29": (30, 29)}


@torch.no_grad()
def changes():
    """Squared token change into each frame at the digit cells, per DEV episode (cached)."""
    f = RUN / "x_cells.npz"
    if f.exists():
        return dict(np.load(f))
    model, _ = G.CR.NP.IF.load(TF.CKPT.resolve())
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    out = {}
    for e, ep in enumerate(G.S.episodes(L)):
        if e % 5 == 4 or e in TRAIN:
            p = UC.coarse_np(model, G.D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy())
            out[str(e)] = UC.change(p)[:, CELLS]
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez(f, **out)
    return out


def laplace(spec):
    """CogRNN's own decay rates and Post matrix (float64), inner-node slices as its forward returns them."""
    torch.set_default_dtype(torch.float64)
    m = CK.CogRNN(tstr_min=spec[0], tstr_max=spec[1], n_taus=spec[2], k=spec[3], dt=1.0, g=1)
    torch.set_default_dtype(torch.float32)
    return m, m.s_full.double().numpy(), m.post.double().numpy(), spec[3]


def scan(ev, s):
    """F_t = exp(-s) F_{t-1} + f_t per node (CogRNN's recurrence, dt = 1, alpha = 1), ev [T] -> [T, nodes]."""
    return np.stack([lfilter([1.0], [1.0, -np.exp(-si)], ev) for si in s], 1)


def codes(x):
    """Every code at frames 0..T-1 from the changes into frames <= t, x [T, 4] -> {name: [T, 4, d]}."""
    ev = (x > 1.0).astype(np.float64)
    T = len(x)
    since = np.full((T, 4), 10 ** 6)
    last = np.full(4, -10 ** 6)
    for t in range(T):
        last = np.where(ev[t] > 0, t, last)
        since[t] = t - last
    u = np.where(since <= CAP, CAP - since, 0)
    out = {"csc": np.eye(CAP + 1)[u], "traces": np.log1p(TT.traces(np.pad(x, ((0, 0), (0, 77))))[:, :4])}
    for name, (m, s, post, k) in LAP.items():
        F = np.stack([scan(ev[:, c], s) for c in range(4)], 1)          # [T, 4, N]
        out[name + "_F"] = F[..., k:-k]
        out[name] = (F @ post)[..., k:-k]
    return out


def check_scan():
    """Our lfilter scan equals CogRNN's own log-domain scan on a real event train (both settings)."""
    x = next(iter(X.values()))
    ev = (x[:, 3] > 1.0).astype(np.float64)
    for name, (m, s, post, k) in LAP.items():
        torch.set_default_dtype(torch.float64)
        til, _, F = m(torch.as_tensor(ev)[:, None, None])
        torch.set_default_dtype(torch.float32)
        F_ours = scan(ev, s)
        assert np.allclose(F[:, 0, 0].numpy(), F_ours[:, k:-k], rtol=1e-6, atol=1e-9), name
        assert np.allclose(til[:, 0, 0].numpy(), (F_ours @ post)[:, k:-k], rtol=1e-5, atol=1e-8), name


def logistic(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-3 * sd.max(), sd, 1.0)
    A, B = (torch.as_tensor((z - mu) / sd) for z in (Xtr, Xte))
    y = torch.as_tensor(ytr, dtype=torch.float64)
    w = torch.zeros(A.shape[1] + 1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w], max_iter=500, tolerance_grad=1e-9, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        z = A @ w[:-1] + w[-1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 1e-4 * w[:-1].square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return (B @ w[:-1] + w[-1]).detach().sigmoid().numpy()


def gbm(Xtr, ytr, Xte):
    import lightgbm as lgb
    par = dict(objective="binary", learning_rate=.06, num_leaves=31, min_data_in_leaf=50, num_threads=6, verbosity=-1, seed=0)
    return lgb.train(par, lgb.Dataset(Xtr, ytr.astype(float)), num_boost_round=200).predict(Xte)


def main():
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, _ = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    check_scan()
    tr, te = {}, {}                                                     # name -> list of rows per digit cell
    rng = np.random.default_rng(0)
    for e, x in X.items():
        e, C = int(e), codes(x)
        T = len(x)
        ts_tr = rng.choice(np.arange(1, T - 1), min(T - 2, 120), replace=False)   # token_trace's 120 frames per episode
        for name, c in C.items():
            if e in TRAIN:
                for j in range(4):
                    tr.setdefault((name, j), []).append((c[ts_tr, j].reshape(len(ts_tr), -1).astype(np.float32),
                                                         x[ts_tr + 1, j] > 1.0))
            else:
                for n, (col, cell) in EVENTS.items():
                    j = CELLS.index(cell)
                    ts = np.array([t for t in range(9, T) if after[off[e] + t, col]], int)
                    if len(ts):
                        te.setdefault((name, n), []).append((c[ts - 1, j].reshape(len(ts), -1), y[off[e] + ts, col]))
    res = {"train_frames_per_cell": int(sum(len(a[1]) for a in tr[("csc", 0)])), "aps": {}, "separation": {}}
    for name in sorted({k[0] for k in tr}):
        res["aps"][name] = {}
        for n, (col, cell) in EVENTS.items():
            j = CELLS.index(cell)
            Xtr = np.concatenate([a[0] for a in tr[(name, j)]]); ytr = np.concatenate([a[1] for a in tr[(name, j)]])
            Xte = np.concatenate([a[0] for a in te[(name, n)]]); yte = np.concatenate([a[1] for a in te[(name, n)]])
            res["aps"][name][n] = {"linear": G.AP(yte, logistic(Xtr, ytr, Xte)), "nonlinear": G.AP(yte, gbm(Xtr, ytr, Xte)),
                                   "base_rate": float(yte.mean()), "rows": int(len(yte)), "events": int(yte.sum())}
            print(name, n, json.dumps(res["aps"][name][n]), flush=True)
        Xtr = np.concatenate([a[0] for a in tr[(name, 3)]])             # cell 66 (energy), the spread of the code
        spread = float(np.sqrt(Xtr.var(0).sum()))
        res["separation"][name] = {"spread": spread}
        for lag, (a, b) in LAGS.items():                                # one isolated event, read `a` / `b` frames later
            xx = np.zeros((70, 4)); xx[0] = 9.0                         # 9: a digit change's typical size (check 8: 7-11)
            c = codes(xx)[name][:, 3].reshape(70, -1)
            res["separation"][name][lag] = float(np.linalg.norm(c[a] - c[b]) / spread)
        print(name, json.dumps(res["separation"][name]), flush=True)
    (HERE / "readability.json").write_text(json.dumps(res, indent=1) + "\n")


L0 = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
TRAIN = set([e for e in range(len(G.S.episodes(L0))) if e % 5 != 4][:400])
LAP = {n: laplace(s) for n, s in LAPLACE.items()}
X = changes()

if __name__ == "__main__":
    main()
