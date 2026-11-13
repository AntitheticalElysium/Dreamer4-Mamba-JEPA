"""Split credit.py's TSP gap into readout and currency (RESULTS.md, credit). Frozen, no training.

credit.py's separation rate for the TSP forecast loss, with F5's trained forecast head replaced by the fresh readouts from
readout_fit.py (concat: the current form; attend: cell queries over Mamba's memory tokens), both fitted on checkpoint
10's frozen beliefs. Same pairs, same window-mean squared-error loss on both members, same swap separation; rate over
belief parameters at checkpoint 10. If a readout that renders belief content well still sends ~0 request for energy and
hunger, the pixel-MSE currency is the bulk of the gap; if its request rises toward the reward forecast's, the readout is.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.credit_readout
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection/credit"
CR = importlib.import_module("artifacts.experiments.20261111_p1f5.credit")
RF = importlib.import_module("artifacts.experiments.20261111_p1f5.readout_fit")
TJ, DS, NP = CR.TJ, CR.DS, CR.NP
DEV = DS.DEV


def one(model, cfg, net, row, slot):
    i, e, prefix, frames, rewards, terms, q, truth = row
    p, rw, td, act = CR.belief_inputs(model, cfg, row)
    with TR.autocast(DEV, cfg):
        y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), rw, td, act), model.belief.initial(2, DEV))
    b = y[:, -1].float()
    centre, _ = TJ.WINDOW[slot]
    fx = torch.tensor(centre, device=DEV).expand(2, 2)
    cells = M.window(fx)
    fc = net(b, model.belief.memory(b).float(), cells).float()                           # [2, 9, 147]
    tgt = M.cell_pixels(M.crop(torch.as_tensor(frames[:, q + 1], device=DEV), fx)).float() / 127.5 - 1
    n = fc[0].numel()
    se = lambda u, v: (u - v).square().sum() / n
    loss = se(fc[0], tgt[0]) + se(fc[1], tgt[1])
    sep = se(fc[1], tgt[0]) + se(fc[0], tgt[1]) - loss
    params = [x for x in model.belief.parameters() if x.requires_grad]
    gl = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(sep, params, allow_unused=True)
    dot = sum(float((u.float() * z.float()).sum()) for u, z in zip(gs, gl) if u is not None and z is not None)
    return {"loss": float(loss.detach()), "sep": float(sep.detach()), "rate_belief": -dot}


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = NP.IF.load(DS.CKPTS["f5"].resolve())
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)
    saved = torch.load(HERE / "readings/dissection/readout_fit_nets.pt", map_location=DEV, weights_only=False)
    W = model.belief.memory_out.in_features
    Wm = model.belief.memory(torch.zeros(1, W, device=DEV)).shape[-1]
    nets = {"concat": RF.Concat(W).to(DEV), "attend": RF.Attend(W, Wm).to(DEV)}
    for k, net in nets.items():
        net.load_state_dict(saved[k]); net.eval()
    res = {"checkpoint": 10}
    for name, slot in (("recovery", 63), ("energy", 66), ("hunger", 64)):
        rows = TJ.pair_rows(name)
        for k, net in nets.items():
            per = [one(model, cfg, net, r, slot) for r in rows]
            summ = {key: float(np.mean([x[key] for x in per])) for key in per[0]}
            summ["pairs_rate_positive"] = int(sum(x["rate_belief"] > 0 for x in per))
            res[f"{name}/{k}"] = {"summary": summ, "pairs": per}
            print(name, k, json.dumps(summ), flush=True)
    (OUT / "readout_credit.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
