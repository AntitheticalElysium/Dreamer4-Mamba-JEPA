"""Do clock resets enter Mamba's state, and is their timestamp kept? (RESULTS.md, uptake.) Frozen readout, no training.

DEV episodes, S1 encode (fixed fixation seed). For a frame t and a lag L: did event X happen exactly L steps earlier,
i.e. at the transition into frame t - L? Read from:
  b_t      Mamba's corrected belief after frame t (it has seen frames <= t);
  embed_t  the belief's input at frame t, the current percept only (control: shows the current digit values).
L = 0 is uptake: the event is in the frame Mamba has just read. L >= 1 is retention of its timestamp. A clock needs
the timestamp for up to 21-31 steps.
Events: energy, food and drink drops while awake (the clocks' resets; awake energy changes are all decrements),
health +1 (recovery's reset, learned in F5), health drops. L2 logistic (routes.logistic), 5 physical-episode folds,
AP with an episode bootstrap. 40,000 frames with t >= 32, the same rows for every lag and feature set.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.uptake
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection"
DS = importlib.import_module("artifacts.experiments.20261111_p1f5.dissect")
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
LAGS = (0, 1, 2, 4, 8, 12, 16, 20, 24, 28)
EVENTS = {"energy_drop_awake": (3, -1, False), "food_drop_awake": (1, -1, False), "drink_drop_awake": (2, -1, False),
          "health_rise": (0, +1, None), "health_drop": (0, -1, None)}


@torch.no_grad()
def features(arm, L, rows_e, rows_t):
    model, cfg = CV.IF.load(str(DS.CKPTS[arm]))
    enc = S.encode(model, L, cfg)
    b = torch.stack([enc[e]["b"][t] for e, t in zip(rows_e, rows_t)]).float()
    emb = []
    for a in range(0, len(rows_e), 2048):
        at = lambda key: torch.stack([enc[e][key][t] for e, t in zip(rows_e[a:a + 2048], rows_t[a:a + 2048])])
        p, g, v, fx = (at(k).to(DS.DEV) for k in ("p", "g", "v", "fix"))
        emb.append(model.belief.percept_embed(M.Percept(p.float(), g.float(), v.float(), fx)).float().cpu())
    del model, enc
    torch.cuda.empty_cache()
    return {"b": b.double(), "embed": torch.cat(emb).double()}


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    body, sleep = L["s_body"].astype(int), L["s_sleep"].astype(bool)
    cand = [(e, t) for e in range(len(L["ep_T"])) for t in range(32, int(L["ep_T"][e]) + 1)]
    pick = np.sort(np.random.default_rng(2125).choice(len(cand), 40000, replace=False))
    rows_e = np.array([cand[i][0] for i in pick]); rows_t = np.array([cand[i][1] for i in pick])
    g = off[rows_e] + rows_t                                                # global state index of frame t
    folds = RT.folds_of(rows_e)
    res = {"rows": 40000, "episodes": int(len(np.unique(rows_e))), "lags": list(LAGS)}
    feats = {arm: features(arm, L, rows_e, rows_t) for arm in ("f5", "hud")}
    for name, (col, sign, need_sleep) in EVENTS.items():
        res[name] = {}
        for lag in LAGS:
            j = g - lag                                                         # the frame the event leads into
            y = np.sign(body[j, col] - body[j - 1, col]) == sign
            if need_sleep is not None:
                y &= sleep[j - 1] == need_sleep
            y = y.astype(float)
            entry = {"prevalence": float(y.mean())}
            for arm, F in feats.items():
                for fk, X in F.items():
                    s = np.zeros(len(y))
                    for f in range(5):
                        m = folds == f
                        s[m] = RT.logistic(X[~m], y[~m], X[m])
                    entry[f"{arm}/{fk}"] = RT.ap_boot(y, s, rows_e, n=300)
            res[name][str(lag)] = entry
            print(name, lag, {k: (round(v["ap"], 3) if isinstance(v, dict) else round(v, 4)) for k, v in entry.items()}, flush=True)
    (OUT / "uptake.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
