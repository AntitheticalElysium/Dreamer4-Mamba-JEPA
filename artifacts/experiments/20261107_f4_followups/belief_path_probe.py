"""Belief-path probe (PLAN.md 11): where on the belief's path are the timers lost? For the step that produces frame t:
the HUD tokens of frame t - 1 (control), percept_embed(frame t - 1) (the belief's input), b_{t-1} (the belief after it)
and bp_t (after the action); targets at state t - 1: the HUD digits, the exact counters (counters.py) and, asleep, the
steps asleep. Ridge (linear) and an MLP (nonlinear), 5 episode folds, out-of-fold R^2.
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.belief_path_probe CKPT OUT"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M
from saccade.eval import s1 as S

DC = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_checks")
CV, RT = DC.CV, DC.RT
HERE = Path(__file__).resolve().parent
CAP, HUD = 40000, slice(63, 67)                                           # health, food, drink, energy cells


def ridge(Xtr, ytr, Xte, lam):
    m, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
    w = torch.linalg.solve(Xtr.T @ Xtr + lam * torch.eye(Xtr.shape[1], dtype=Xtr.dtype), Xtr.T @ (ytr - ytr.mean()))
    return (Xte @ w + ytr.mean()).numpy()


def mlp(Xtr, ytr, Xte, ep_tr, seed=2116):
    """One hidden layer of 256 (GELU), Adam 1e-3, batch 512, at most 60 epochs, the best epoch on a held-out tenth of
    the fitting episodes."""
    g = torch.Generator().manual_seed(seed)
    m, sd, ym, ys = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6), ytr.mean(), ytr.std().clamp(min=1e-6)
    f = lambda X: ((X - m) / sd).float().to(CV.DEV)
    val = np.isin(ep_tr, np.unique(ep_tr)[torch.randperm(len(np.unique(ep_tr)), generator=g)[:max(1, len(np.unique(ep_tr)) // 10)].numpy()])
    Xa, ya, Xv, yv = f(Xtr[~val]), ((ytr[~val] - ym) / ys).float().to(CV.DEV), f(Xtr[val]), ((ytr[val] - ym) / ys).float().to(CV.DEV)
    torch.manual_seed(seed)
    net = torch.nn.Sequential(torch.nn.Linear(Xa.shape[1], 256), torch.nn.GELU(), torch.nn.Linear(256, 1)).to(CV.DEV)
    opt, best, state = torch.optim.Adam(net.parameters(), 1e-3), float("inf"), None
    for _ in range(60):
        perm = torch.randperm(len(Xa), generator=g).to(CV.DEV)
        for a in range(0, len(Xa), 512):
            i = perm[a:a + 512]
            loss = (net(Xa[i])[:, 0] - ya[i]).square().mean()
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            v = float((net(Xv)[:, 0] - yv).square().mean())
        if v < best:
            best, state = v, {k: x.clone() for k, x in net.state_dict().items()}
    net.load_state_dict(state)
    with torch.no_grad():
        return (net(f(Xte))[:, 0].double().cpu() * ys + ym).numpy()


def main(ckpt, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(ckpt)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    C = dict(np.load(HERE / "readings/counters_dev.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    rows = np.nonzero(T["t"] >= CV.P)[0]
    j = np.sort(np.random.default_rng(2115).choice(rows, CAP, replace=False))
    at = lambda key, dt: torch.stack([enc[int(T["ep"][x])][key][int(T["t"][x]) + dt] for x in j])
    P = M.Percept(*(at(k, -1).float().to(CV.DEV) for k in ("p", "g", "v")), at("fix", -1).to(CV.DEV))
    with torch.no_grad():
        embed = torch.cat([model.belief.percept_embed(M.Percept(*(z[a:a + 4096] for z in P.parts()), P.fix[a:a + 4096])).float().cpu()
                           for a in range(0, len(j), 4096)])
    feats = {"hud_tokens": at("p", -1)[:, HUD].flatten(1), "percept_embed": embed, "b_prev": at("b", -1), "bp": at("bp", 0)}
    feats = {k: v.double() for k, v in feats.items()}
    g, ep, asleep = T["g"][j], T["ep"][j], T["asleep"][j]
    body = L["s_body"][g]
    targets = {"health": (body[:, 0], None), "food": (body[:, 1], None), "drink": (body[:, 2], None), "energy": (body[:, 3], None),
               "hunger_awake": (C["hunger"][g], ~asleep), "thirst_awake": (C["thirst"][g], ~asleep),
               "fatigue_awake": (C["fatigue"][g], ~asleep), "fatigue_asleep": (C["fatigue"][g], asleep),
               "recover_awake": (C["recover"][g], ~asleep), "steps_asleep": (T["asleep_for"][j].astype(float), asleep)}
    res = {"checkpoint": ckpt, "rows": int(len(j)), "episodes": int(len(np.unique(ep))), "targets": {}}
    for name, (y_all, m) in targets.items():
        m = np.ones(len(j), bool) if m is None else m
        y, e = torch.as_tensor(y_all[m], dtype=torch.float64), ep[m]
        folds, r = RT.folds_of(e), {}
        for fk, X_all in feats.items():
            X = X_all[m]
            for reader in ("ridge10", "ridge1000", "mlp"):
                pred = np.zeros(len(y))
                for f in range(5):
                    tr, te = folds != f, folds == f
                    pred[te] = (mlp(X[tr], y[tr], X[te], e[tr]) if reader == "mlp" else ridge(X[tr], y[tr], X[te], 10.0 if reader == "ridge10" else 1000.0))
                yn = y.numpy()
                r[f"{fk}/{reader}"] = {"r2": float(1 - ((pred - yn) ** 2).sum() / ((yn - yn.mean()) ** 2).sum())}
                if name in ("health", "food", "drink", "energy"):
                    r[f"{fk}/{reader}"]["accuracy"] = float((np.clip(np.round(pred), 0, 9) == yn).mean())
            print(name, fk, {k.split("/")[1]: round(v["r2"], 3) for k, v in r.items() if k.startswith(fk + "/")}, flush=True)
        res["targets"][name] = {"rows": int(m.sum()), "mean": float(y.mean()), "sd": float(y.std()), "readers": r}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
