"""F6.1 dissection, check 1 (PLAN.md): does the predicted belief carry the asleep energy-rise phase on sleeps where
WAKE_UP has already paid (where no reward forecast asks for it), and does it beat elapsed time and lighting? F6.1 and
F6 checkpoint 10, read by the gaze-learnability instrument's own probes (`../20261115_gaze_learnability/gaze.py`:
learnable.probe, the cohorts, Codex's time-and-light reader as a fixed offset). CPU, Mamba's reference scan.

    MAMBA_DETERMINISTIC=1 GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_f61_dissection.sleep_phase
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

assert os.environ.get("GAZE_CPU") == "1"
G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
HERE = Path(__file__).resolve().parent
RUN = HERE / "run"
CKPTS = {"F6": G.EXP / "20261113_p1f6/run/ckpt_10.pt", "F6.1": G.EXP / "20261114_p1f61/run/ckpt_10.pt"}


def encode(name, L):
    path = RUN / f"X_{name}.npy"
    if path.exists():
        return torch.as_tensor(np.load(path)).float()
    model, cfg = G.CR.NP.IF.load(CKPTS[name].resolve())
    for m in model.belief.mixers:
        m.backend = "reference"
    X = G.LB.encode_dev(model, model.belief, cfg, L)
    np.save(path, X.half().numpy())
    return X.float()


def main():
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "6")))
    RUN.mkdir(exist_ok=True)
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, ok = G.LB.events(L)
    fold = np.repeat(np.arange(len(L["ep_T"])) % 5, L["ep_T"] + 1)
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"] + 1)
    first = np.zeros(len(fold), bool); first[np.r_[0, np.cumsum(L["ep_T"] + 1)][:-1]] = True
    paid = np.r_[False, L["s_ach"][:-1, 15].astype(bool)] & ~first             # WAKE_UP earned by state t - 1
    m = G.cohorts(L)
    masks = {"natural": ok, **m}
    for tag, cond in (("asleep_unpaid", ~paid), ("asleep_paid", paid)):       # column 3 only is read
        x = ok.copy(); x[:, 3] &= cond; masks[tag] = x
    x = m["steady8"].copy(); x[:, 3] &= paid; masks["asleep_paid_steady8"] = x   # = after_reset column 3
    X = {n: encode(n, L) for n in CKPTS}
    nll = lambda z, t: F.binary_cross_entropy_with_logits(torch.as_tensor(z), torch.as_tensor(t, dtype=torch.float32), reduction="none").numpy()
    rng = np.random.default_rng(215)
    res = {}
    for name, mk in masks.items():
        cols = [3] if name.startswith("asleep") else range(5)
        tl = G.time_light(L, y, mk, fold)
        plain = {n: G.probe(X[n], y, mk, fold) for n in CKPTS}
        off = {n: G.probe(X[n], y, mk, fold, offset=tl) for n in CKPTS}
        res[name] = {}
        for j in cols:
            ev = G.LB.EVENTS[j]
            te = (fold == 4) & mk[:, j]
            base = nll(tl[te, j], y[te, j])
            loss = {n: nll(off[n][ev]["logit"], y[te, j]) for n in CKPTS}
            d = loss["F6"] - loss["F6.1"]
            ids, inv = np.unique(ep[te], return_inverse=True)
            boot = rng.integers(len(ids), size=(10000, len(ids)))
            ns = np.bincount(inv, minlength=len(ids))[boot].sum(1)
            bs = np.bincount(inv, weights=d, minlength=len(ids))[boot].sum(1)
            res[name][ev] = {"rows": int(te.sum()), "events": int(y[te, j].sum()), "episodes": len(ids),
                             "prevalence": float(y[te, j].mean()),
                             "time_light_ap": G.AP(y[te, j], tl[te, j]),
                             "ap": {n: G.AP(y[te, j], plain[n][ev]["logit"]) for n in CKPTS},
                             "ap_with_time_light": {n: G.AP(y[te, j], off[n][ev]["logit"]) for n in CKPTS},
                             "gain_over_time_light_nats": {n: float((base - loss[n]).mean()) for n in CKPTS},
                             "f61_minus_f6_nats": [float(d.mean()), *np.quantile(bs[ns > 0] / ns[ns > 0], [.025, .975]).tolist()]}
            print(name, ev, json.dumps(res[name][ev]), flush=True)
    (HERE / "sleep_phase.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main()
