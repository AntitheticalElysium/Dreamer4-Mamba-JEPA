"""F6 dissection, reading 3 (PLAN.md): what the token forecast's loss pays for, by frame class and cell group.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.lossmap [checkpoints...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings"
F6 = importlib.import_module("artifacts.experiments.20261113_p1f6.train")
IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
DIGIT = {"health": 63, "food": 64, "drink": 65, "energy": 66}
CELLS = {"map": np.arange(63), "hud_digits": np.array([63, 64, 65, 66]), "hud_other": np.arange(67, 81)}


@torch.no_grad()
def checkpoint(k):
    path = OUT / f"lossmap_{k:02d}.json"
    if path.exists():
        return
    ck = (F6.HERE / f"run/ckpt_{k:02d}.pt").resolve()
    model, cfg = IF.load(ck)
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    head, copy, cls, change = [], [], [], {n: [] for n in DIGIT}
    for e, r in enumerate(S.encode(model, L, cfg)):
        prev, tgt = r["p"][:-1].float().to(IF.DEV), r["p"][1:].float().to(IF.DEV)
        pred = prev + model.token_forecast(r["bp"][1:].float().to(IF.DEV))
        head.append((pred - tgt).square().mean(-1).cpu().numpy()); copy.append((prev - tgt).square().mean(-1).cpu().numpy())
        a, b = off[e], off[e + 1]
        pos, sleep, body = L["s_pos"][a:b].astype(int), L["s_sleep"][a:b].astype(bool), L["s_body"][a:b].astype(int)
        moved = (pos[1:] != pos[:-1]).any(-1)
        cls.append(np.where(sleep[:-1], 2, np.where(moved, 0, 1)))        # 0 moved, 1 still awake, 2 asleep (before the step)
        for j, n in enumerate(DIGIT):
            change[n].append(np.sign(body[1:, j] - body[:-1, j]))
    head, copy, cls = np.concatenate(head), np.concatenate(copy), np.concatenate(cls)
    change = {n: np.concatenate(v) for n, v in change.items()}
    total = float(head.sum())
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "frames": int(len(cls)),
           "token_mse": {"head": float(head.mean()), "copy_previous": float(copy.mean())}, "by_class_and_cells": {},
           "frame_share": {n: float((cls == c).mean()) for c, n in enumerate(("moved", "still_awake", "asleep"))}, "digits": {}}
    for c, cn in enumerate(("moved", "still_awake", "asleep")):
        for gn, cells in CELLS.items():
            h, q = head[cls == c][:, cells], copy[cls == c][:, cells]
            res["by_class_and_cells"][f"{cn}/{gn}"] = {"share_of_term": float(h.sum() / total), "head_per_cell": float(h.mean()),
                                                       "copy_per_cell": float(q.mean()), "lift_over_copy": float(1 - h.sum() / max(q.sum(), 1e-30))}
    for n, cell in DIGIT.items():
        row = {}
        for tag, m in (("down_awake", (change[n] < 0) & (cls != 2)), ("up_awake", (change[n] > 0) & (cls != 2)),
                       ("down_asleep", (change[n] < 0) & (cls == 2)), ("up_asleep", (change[n] > 0) & (cls == 2)),
                       ("unchanged", change[n] == 0)):
            if m.sum() >= 20:
                h, q = head[m, cell], copy[m, cell]
                row[tag] = {"frames": int(m.sum()), "head": float(h.mean()), "copy_previous": float(q.mean()),
                            "predicted_share": float(1 - h.mean() / max(q.mean(), 1e-30)) if tag != "unchanged" else None,
                            "share_of_term": float(h.sum() / total)}
        res["digits"][n] = row
    print(k, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    for k in ([int(x) for x in sys.argv[1:]] or (2, 10)):
        checkpoint(k)
