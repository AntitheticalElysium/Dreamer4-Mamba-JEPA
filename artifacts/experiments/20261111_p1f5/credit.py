"""Does a forecast loss ask the model to tell a due branch from a not-due one? (RESULTS.md, credit.) Frozen, no training.

On each exact-history pair, at its due transition, for each forecast head h at an F5 checkpoint:
  loss_h  that head's training term on both members: reward forecast -log p(actual reward); TSP forecast at a 3x3 window
          containing the digit's cell, the mean squared error over the window's 1,323 numbers (as trained).
  sep_h   how much loss_h would rise if the two members' beliefs were swapped, in the same units:
          reward: lp_A(r_A) + lp_B(r_B) - lp_B(r_A) - lp_A(r_B); TSP: [|f_B - y_A|^2 + |f_A - y_B|^2 - |f_A - y_A|^2
          - |f_B - y_B|^2] / 1323. sep_h > 0 means the head currently tells the branches apart.
  rate_h  d sep_h / d eta under one SGD step theta <- theta - eta grad(loss_h): -grad(sep_h) . grad(loss_h), split by
          parameter group: belief (Mamba, its percept projection and input tokens) and head (the head's own network,
          plus the shared pixel decoder for TSP). This is the first-order effect of training on these two examples on
          their separation. The two branches differ only between their earlier events, so any belief-side separation
          has to come through Mamba carrying that difference forward. The full history is one sequence here, so
          there is no chunk truncation; data_counts.json gives the fraction of training events whose reset fell
          inside their chunk. The perception encoder (0.3x learning rate in training) is held fixed.
Both heads are in G4 with one harmoniser weight, so their losses and rates are in comparable units. A TSP event is
seen only when its window covers the cell; data_counts.json gives those counts per frame.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.credit [checkpoint numbers...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection/credit"
TJ = importlib.import_module("artifacts.experiments.20261111_p1f5.trajectory")
DS, NP = TJ.DS, TJ.NP
DEV = DS.DEV
SETS = {"recovery": (63, ("reward", "tsp")), "cooldown": (63, ("reward", "tsp")), "energy": (66, ("tsp",)),
        "hunger": (64, ("tsp",))}


def belief_inputs(model, cfg, row):
    i, e, prefix, frames, rewards, terms, q, truth = row
    old_f, old_a, old_r, old_d = prefix
    f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
    a = np.r_[old_a, np.zeros(q + 1, dtype=np.int64)]
    rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
    td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
    fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=DEV)
    with torch.no_grad(), TR.autocast(DEV, cfg):
        p = model.perception(torch.as_tensor(f, device=DEV), fix)
    p = M.Percept(*(z.detach().float() for z in p.parts()), p.fix)
    return p, torch.as_tensor(rw, device=DEV, dtype=torch.float32), torch.as_tensor(td, device=DEV), \
        torch.as_tensor(np.repeat(a[None], 2, 0), device=DEV)


def head_terms(model, b, head, row, slot):
    """(loss, separation) for one head on the due transition; b = post-action beliefs [2, 1, W]."""
    i, e, prefix, frames, rewards, terms, q, truth = row
    if head == "reward":
        h = model.outcome.features(b, None)
        r = torch.as_tensor(rewards[:, q], device=DEV, dtype=torch.float32)
        lp = [[model.outcome.reward_logp(h[m:m + 1], r[k].view(1, 1)).sum() for k in (0, 1)] for m in (0, 1)]
        return -(lp[0][0] + lp[1][1]), lp[0][0] + lp[1][1] - lp[1][0] - lp[0][1]
    centre, _ = TJ.WINDOW[slot]
    fx = torch.tensor(centre, device=DEV).expand(2, 1, 2)
    fc = model.pixels.mean(model.tsp.predict(b, None, None, M.window(fx))).float()[:, 0]           # [2, 9, 147]
    y = M.cell_pixels(M.crop(torch.as_tensor(frames[:, q + 1:q + 2], device=DEV), fx)).float()[:, 0] / 127.5 - 1
    n = fc[0].numel()
    se = lambda u, v: (u - v).square().sum() / n
    return se(fc[0], y[0]) + se(fc[1], y[1]), se(fc[1], y[0]) + se(fc[0], y[1]) - se(fc[0], y[0]) - se(fc[1], y[1])


def groups(model, head):
    belief = [p for p in model.belief.parameters() if p.requires_grad]
    own = list(model.outcome.parameters()) if head == "reward" else list(model.tsp.parameters()) + list(model.pixels.parameters())
    return {"belief": belief, "head": [p for p in own if p.requires_grad]}


def one(model, cfg, row, slot, head):
    p, rw, td, act = belief_inputs(model, cfg, row)
    with TR.autocast(DEV, cfg):
        inputs = model.belief.tokens(model.belief.percept_embed(p), rw, td, act)
        y, _ = model.belief(inputs, model.belief.initial(2, DEV))
        b = y[:, -1:]
        loss, sep = head_terms(model, b, head, row, slot)
    out = {"loss": float(loss), "sep": float(sep)}
    G = groups(model, head)
    params = G["belief"] + G["head"]
    gl = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(sep, params, allow_unused=True)
    k = len(G["belief"])
    dot = lambda a, c: sum(float((x.float() * z.float()).sum()) for x, z in zip(a, c) if x is not None and z is not None)
    norm = lambda a: sum(float(x.float().square().sum()) for x in a if x is not None) ** .5
    out["rate_belief"] = -dot(gs[:k], gl[:k])
    out["rate_head"] = -dot(gs[k:], gl[k:])
    out["grad_loss_norm_belief"], out["grad_sep_norm_belief"] = norm(gl[:k]), norm(gs[:k])
    out["grad_loss_norm_head"], out["grad_sep_norm_head"] = norm(gl[k:]), norm(gs[k:])
    return out


def checkpoint(k):
    path = OUT / f"ckpt_{k:02d}.json"
    if path.exists():
        return
    ck = HERE / f"run/ckpt_{k:02d}.pt"
    model, cfg = NP.IF.load(ck.resolve())
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "sigma": model.harmoniser.log_sigma.detach().exp().tolist()}
    for name, (slot, hs) in SETS.items():
        rows = TJ.pair_rows(name)
        for head in hs:
            per = [one(model, cfg, r, slot, head) for r in rows]
            summ = {key: float(np.mean([x[key] for x in per])) for key in per[0]}
            summ["pairs_rate_belief_positive"] = int(sum(x["rate_belief"] > 0 for x in per))
            summ["pairs_sep_positive"] = int(sum(x["sep"] > 0 for x in per))
            res[f"{name}/{head}"] = {"summary": summ, "pairs": per}
            print(k, name, head, json.dumps({kk: round(v, 6) if isinstance(v, float) else v for kk, v in summ.items()}), flush=True)
    del model
    torch.cuda.empty_cache()
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else range(1, 11)):
        checkpoint(k)
