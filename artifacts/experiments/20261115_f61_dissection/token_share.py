"""F6.1 dissection, check 8 (PLAN.md): each clock event's size in the denoiser's own currency. On held-out DEV (fold 4),
F6.1's periphery tokens (`chooser.coarse`: the world's own periphery path, G1's token target): per transition the
squared token change at the digit's cell and over all 81 cells, at each event and at eligible frames without one.

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_f61_dissection.token_share
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
HERE = Path(__file__).resolve().parent
CELL = {"food_down": 64, "drink_down": 65, "energy_down_awake": 66, "energy_up_asleep": 66, "recovery": 63}


@torch.no_grad()
def main():
    torch.set_num_threads(6)
    model, cfg = G.CR.NP.IF.load((G.EXP / "20261114_p1f61/run/ckpt_10.pt").resolve())
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, ok = G.LB.events(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    cell, total, rows = [], [], []
    for e, ep in enumerate(G.S.episodes(L)):
        if e % 5 != 4:
            continue
        fr = torch.as_tensor(G.D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy())
        p = torch.cat([G.C.coarse(model.perception, z)[0] for z in fr.split(512)]).float()
        d = (p[1:] - p[:-1]).square().sum(-1)                           # [T, 81]: transition t-1 -> t
        cell.append(d.numpy()); total.append(d.sum(-1).numpy()); rows.append(off[e] + np.arange(1, len(p)))
    cell, total, rows = np.concatenate(cell), np.concatenate(total), np.concatenate(rows)
    res = {}
    for j, n in enumerate(G.LB.EVENTS):
        c = CELL[n]
        ev, el = y[rows, j], ok[rows, j]
        sel = {"event": el & ev, "eligible_no_event": el & ~ev}
        res[n] = {k: {"frames": int(m.sum()), "cell_change": float(cell[m, c].mean()), "frame_change": float(total[m].mean()),
                      "cell_share_of_frame": float((cell[m, c] / np.maximum(total[m], 1e-12)).mean())} for k, m in sel.items()}
        print(n, json.dumps(res[n]), flush=True)
    (HERE / "token_share.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
