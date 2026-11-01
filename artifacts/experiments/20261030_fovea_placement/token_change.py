"""Size of each HUD counter transition in the base checkpoint's real periphery tokens (dissection, 10-30: drink and wood
are generated from 0 -> 1 but almost never from v -> v + 1, at matched training frequency; the renderer draws an empty
cell at 0 and the icon plus a digit glyph at v >= 1). Per counter (drink cell 65, wood 68) and value v: mean token,
within-value RMS spread, and |mean(v + 1) - mean(v)| (raw, and over the spread). Table: faced-cell token before vs
after a placement. Dev split, first EPISODES episodes, S1-seeded fixations (S.encode).
Usage (project venv, GPU): python -m artifacts.experiments.20261030_fovea_placement.token_change [EPISODES]
"""
import json, sys

import numpy as np
import torch

from saccade import model as M
from saccade.eval import s1 as S

EV = __import__("importlib").import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
CAT = EV.CAT


def main():
    n_ep = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ck = torch.load(EV.BASE, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(EV.DEV); model.load_state_dict(ck["model"]); model.eval()
    L = S.first_episodes(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    enc = S.encode(model, L, cfg)
    p = torch.cat([e["p"].float() for e in enc]).cpu().numpy()            # [states, 81, 16]
    out = {"episodes": n_ep, "states": int(len(p))}
    for name, (cell, arr, col) in CAT.HUD.items():
        v = L["s_body" if arr == 0 else "s_inv"][:, col].astype(int)
        x = p[:, cell]
        means = {k: x[v == k].mean(0) for k in np.unique(v) if (v == k).sum() >= 20}
        spread = {k: float(np.sqrt(((x[v == k] - means[k]) ** 2).sum(-1).mean())) for k in means}
        out[name] = {int(k): {"states": int((v == k).sum()), "spread": spread[k],
                              **({"step_to_next": float(np.linalg.norm(means[k + 1] - means[k])),
                                  "step_over_spread": float(np.linalg.norm(means[k + 1] - means[k]) /
                                                            np.sqrt((spread[k] ** 2 + spread[k + 1] ** 2) / 2))}
                                 if k + 1 in means else {})} for k in means}
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    first = np.zeros(len(p), bool); first[s_off[:-1]] = True
    prev = np.maximum(np.arange(len(p)) - 1, 0)
    d = CAT.FC.DIRS[L["s_dir"][prev]]
    fr, fc = 3 + d[:, 0], 4 + d[:, 1]
    ar = np.arange(len(p))
    placed = (L["s_view"][ar, fr, fc] == CAT.TABLE) & (L["s_view"][prev, fr, fc] != CAT.TABLE) & ~first
    cells = fr * 9 + fc
    step = np.linalg.norm(p[ar, cells] - p[prev, cells], axis=-1)
    out["table_placed"] = {"events": int(placed.sum()), "mean_step": float(step[placed].mean()) if placed.any() else None}
    print(json.dumps(out, indent=1))
    (EV.HERE / "eval_stage1" / "token_change.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
