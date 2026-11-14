"""Reading 3 (PLAN.md): the share of a code-likelihood loss that the pixel-clock events would carry.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_currency_eda.share_codes
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
CD = importlib.import_module("artifacts.experiments.20261114_currency_eda.codes")
DZ, DEV = CD.DZ, CD.DEV
CLASSES = ("moved", "still_day", "still_dusk_dawn", "still_night", "asleep")


@torch.no_grad()
def main():
    _, model, cfg, _ = DZ.load(10)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    enc = [r["p"] for r in S.encode(model, L, cfg)]
    fit = [e for e in range(len(enc)) if e % 5 <= 2]
    test = [e for e in range(len(enc)) if e % 5 == 4]
    g = torch.Generator().manual_seed(0)
    pool = torch.cat([enc[e].reshape(-1, 16) for e in fit])
    pool = pool[torch.randperm(len(pool), generator=g)[:3_000_000]].float().to(DEV)
    res = {}
    for K in (64, 256):
        c = CD.kmeans(pool, K)                                            # the same seed and data as codes.py
        rows = {}
        for split, eps in (("fit", fit), ("test", test)):
            out = []
            for e in eps:
                code = CD.assign(enc[e].to(DEV), c).cpu().numpy()
                a, b = off[e], off[e + 1]
                pos, sleep, lt, body = L["s_pos"][a:b].astype(int), L["s_sleep"][a:b].astype(bool), L["s_light"][a:b], L["s_body"][a:b].astype(int)
                moved = (pos[1:] != pos[:-1]).any(-1)
                cls = np.where(sleep[:-1], 4, np.where(moved, 0, np.where(lt[1:] >= .99, 1, np.where(lt[1:] >= .5, 2, 3))))
                out.append((code[:-1], code[1:], cls, body[1:] != body[:-1]))
            rows[split] = [np.concatenate(z) for z in zip(*out)]
        c0, c1, cls, dch = rows["fit"]
        ch = c0 != c1
        rate = np.full((5, 81), .5)
        new = np.ones((81, K))                                            # add-one counts of each cell's new code
        for k in range(5):
            m = cls == k
            rate[k] = np.clip((ch[m].sum(0) + 1) / (m.sum() + 2), 1e-6, 1 - 1e-6)
        for cell in range(81):
            np.add.at(new[cell], c1[ch[:, cell], cell], 1)
        pnew = new / new.sum(1, keepdims=True)
        c0, c1, cls, dch = rows["test"]
        ch = c0 != c1
        r = rate[cls]                                                     # [N, 81]
        loss = np.where(ch, -np.log(r) - np.log(pnew[np.arange(81)[None], c1]), -np.log(1 - r))
        total = loss.sum()
        out = {"frames": int(len(cls)), "loss_per_frame": float(total / len(cls)),
               "share_by_class": {n: float(loss[cls == k].sum() / total) for k, n in enumerate(CLASSES)},
               "share_map_cells": float(loss[:, :63].sum() / total), "share_hud_digit_cells": float(loss[:, 63:67].sum() / total),
               "event_share": {}, "event_nats_per_event": {}}
        for j, n in enumerate(("health", "food", "drink", "energy")):
            m = dch[:, j]
            out["event_share"][n] = float(loss[m, 63 + j].sum() / total)
            out["event_nats_per_event"][n] = float(loss[m, 63 + j].mean())
        out["event_share"]["food+drink+energy"] = sum(out["event_share"][n] for n in ("food", "drink", "energy"))
        res[K] = out
        print(K, json.dumps(out), flush=True)
    (HERE / "share_codes.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main()
