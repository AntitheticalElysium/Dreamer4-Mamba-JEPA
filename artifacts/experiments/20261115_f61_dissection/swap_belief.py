"""Check 10b (PLAN.md): does F6.1's belief tell the swap pairs' due branch from the delayed one? The world's own asleep-
rise probe (learnable.probe on its DEV beliefs, check 1) on each branch's belief before the target frame, as built by
swap.cell_rises.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.swap_belief      (GPU)
"""
import importlib
import json
from pathlib import Path

import numpy as np
import torch

G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
SW = importlib.import_module("artifacts.experiments.20261115_f61_dissection.swap")
HERE = Path(__file__).resolve().parent


@torch.no_grad()
def main():
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    y, ok = G.LB.events(L)
    fold = np.repeat(np.arange(len(L["ep_T"])) % 5, L["ep_T"] + 1)
    pr = G.probe(torch.as_tensor(np.load(HERE / "run/X_F6.1.npy")).float(), y, ok, fold)["energy_up_asleep"]
    model, cfg = SW.DS.NP.IF.load(SW.CKPT)
    dev = SW.DS.DEV
    sep = []
    for i, row in enumerate(torch.load(SW.BANK, map_location="cpu", weights_only=False)):
        q = row["end"]
        f = np.concatenate([np.repeat(row["prefix_frames"][None], 2, 0), row["frames"][:, :q + 1]], 1)
        a = np.concatenate([np.repeat(row["prefix_actions"][None], 2, 0), row["actions"][:, :q + 1]], 1)
        rw = np.column_stack([np.repeat(np.r_[0., row["prefix_reward"]][None], 2, 0), row["reward"][:, :q]])
        td = np.column_stack([np.repeat(np.r_[False, row["prefix_term"]][None], 2, 0), row["term"][:, :q]])
        fix = torch.as_tensor(np.repeat(G.D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
        act = torch.as_tensor(a, device=dev)
        with G.TR.autocast(dev, cfg):
            p = model.perception(torch.as_tensor(f, device=dev), fix)
            yy, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                     torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
        b = yy[:, -1].float().cpu()
        z = ((b - pr["mu"]) / pr["sd"]) @ pr["w"][:-1] + pr["w"][-1]
        sep.append([float(z[0]), float(z[1])])
    s = np.array(sep)
    d = s[:, 0] - s[:, 1]
    bs = d[np.random.default_rng(1111).integers(len(d), size=(10000, len(d)))].mean(1)
    res = {"pairs": len(d), "logit_due": float(s[:, 0].mean()), "logit_delayed": float(s[:, 1].mean()),
           "p_due": float(torch.sigmoid(torch.as_tensor(s[:, 0])).mean()), "p_delayed": float(torch.sigmoid(torch.as_tensor(s[:, 1])).mean()),
           "due_minus_delayed_logit": [float(d.mean()), *np.quantile(bs, [.025, .975]).tolist()], "pairs_due_higher": int((d > 0).sum())}
    print(json.dumps(res), flush=True)
    (HERE / "swap_belief_onset.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
