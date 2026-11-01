"""Dissection of M0.5 arm G: what the glimpse token of a HUD counter encodes (GPU, dev, guided-route encodings).
1. Variance decomposition. The real token of the counter's cell at the rule's window centre (drink 65 at centre (7, 2),
   wood 68 at centre (7, 5)), least squares on one-hot factors: the counter's own value, each other cell of the
   window (HUD values, map block ids), and daylight; R^2 of each factor alone and the joint R^2. The periphery token
   of the same cell (16-d, the cell's own pixels at 3 x 3 acuity) beside it; the own value split into icon presence
   (value >= 1) and the digit among frames showing the icon (R^2 of the value within those frames).
2. The pixel decoder (shared by TSP and the fovea reconstruction): its log-scale depends on lighting and position only;
   mean log-scale on HUD cells against map cells, at the encodings' own lighting latents.
Usage: python -m artifacts.experiments.20261030_m05_guided_fovea.dissect.glimpse_variance [LABEL ...]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR

E = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.evaluate")
GF = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.dissect.glimpse_field")
EV, PM, IN = E.EV, E.PM, E.IN
DEV = E.DEV
HERE = Path(__file__).resolve().parent
TARGETS = {"drink": (65, (7, 2)), "wood": (68, (7, 5))}
COUNTERS = {63: ("s_body", 0), 64: ("s_body", 1), 65: ("s_body", 2), 66: ("s_body", 3), 67: ("s_inv", 5),
            68: ("s_inv", 0), 69: ("s_inv", 1)}


def onehot(x):
    u, inv = np.unique(x, return_inverse=True)
    return np.eye(len(u))[inv]


def r2(Y, X):
    """Share of the tokens' total variance explained by least squares on X (with intercept)."""
    X = np.c_[np.ones(len(X)), X]
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    res = Y - X @ beta
    return float(1 - res.var(0).sum() / Y.var(0).sum())


def factor(L, s, c):
    """Content of cell c at states s: counter value for HUD counters, block id for map cells, else -1."""
    if c in COUNTERS:
        arr, col = COUNTERS[c]
        return np.clip(L[arr][s, col], 0, None)
    if c < 63:
        return L["s_view"][s, c // 9, c % 9]
    return np.full(len(s), -1)


def main():
    labels = sys.argv[1:] or ["base", "guided_1500"]
    L = dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    res = {}
    for label in labels:
        model, cfg, _ = E.load(GF.CKPTS[label])
        enc = EV.encode_with(model, L, cfg, PM.window_of_state(L, rule))
        s = IN.states(enc, L)
        fix, g, p = torch.cat([r["fix"] for r in enc]), torch.cat([r["g"] for r in enc]), torch.cat([r["p"] for r in enc])
        v = torch.cat([r["v"] for r in enc])
        res[label] = {"variance": {}, "decoder_log_scale": {}}
        for name, (cell, centre) in TARGETS.items():
            m = ((fix[:, 0] == centre[0]) & (fix[:, 1] == centre[1])).numpy()
            win = M.window(torch.as_tensor(centre))
            slot = int((win == cell).nonzero()[0, 0])
            Yg, Yp, st = g[m][:, slot].float().numpy(), p[m][:, cell].float().numpy(), s[m]
            own = factor(L, st, cell)
            fac = {f"own ({cell})": onehot(own), "own icon (value >= 1)": np.c_[own >= 1]}
            for c in win.tolist():
                if c != cell:
                    fac[f"cell {c}"] = onehot(factor(L, st, c))
            fac["daylight"] = np.c_[L["s_light"][st]] if "s_light" in L else np.zeros((len(st), 1))
            joint = np.concatenate([x for k, x in fac.items() if k != "own icon (value >= 1)"], 1)
            icon = own >= 1                                              # the digit among frames showing the icon
            digit = {"glimpse": r2(Yg[icon], onehot(own[icon])), "periphery": r2(Yp[icon], onehot(own[icon]))}
            res[label]["variance"][name] = {"frames": int(m.sum()),
                                            "glimpse": {k: r2(Yg, x) for k, x in fac.items()} | {"joint": r2(Yg, joint)},
                                            "periphery": {k: r2(Yp, x) for k, x in fac.items()} | {"joint": r2(Yp, joint)},
                                            "digit_given_icon": digit}
        with torch.no_grad(), TR.autocast(DEV, cfg):
            vv = v[torch.randperm(len(v), generator=torch.Generator().manual_seed(0))[:4096]].float().to(DEV)
            cells = torch.arange(81, device=DEV)
            ls = model.pixels.scale(torch.cat([vv[:, None].expand(-1, 81, -1), model.pixels.cell(cells)[None].expand(len(vv), -1, -1)], -1))
        ls = ls.float().mean(0).cpu()                                    # [81, 147]
        hud = [63, 64, 65, 66, 67, 68, 69]
        res[label]["decoder_log_scale"] = {"hud_counters": float(ls[hud].mean()), "map_cells": float(ls[:63].mean()),
                                           "per_cell": {c: float(ls[c].mean()) for c in hud + [30, 31, 39, 40]}}
        print(label, json.dumps(res[label], indent=1), flush=True)
        del model, enc
        torch.cuda.empty_cache()
    (HERE / "glimpse_variance.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
