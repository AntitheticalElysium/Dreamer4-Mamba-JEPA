"""How much of an earlier reset survives in Mamba's state until the due step? (RESULTS.md, trace.) Frozen, no training.

On each exact-history pair the two branches differ only between their earlier events: steps 6 vs 14, a reward
input for recovery, a digit change for energy and hunger. Per step s of the branch, d(s) = |b_A(s) - b_B(s)| /
rms|b(s)|: the distance between the two branches' corrected beliefs, relative to the belief's own size. It is read
from step 6 to the due step. A gradient can only separate the branches at the due step through what of this
difference survives there; credit.py's rate vanishes when the states coincide. Checkpoints 1, 3 and 10.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.trace
"""
import importlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from saccade import train as TR

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection"
CR = importlib.import_module("artifacts.experiments.20261111_p1f5.credit")
TJ, NP, DEV = CR.TJ, CR.NP, CR.DEV


@torch.no_grad()
def curve(model, cfg, row):
    i, e, prefix, frames, rewards, terms, q, truth = row
    p, rw, td, act = CR.belief_inputs(model, cfg, row)
    n = len(prefix[0])
    with TR.autocast(DEV, cfg):
        y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), rw, td, act), model.belief.initial(2, DEV))
    b = y[:, 0::2].float()                                                   # corrected belief after each frame
    seg = b[:, n:n + q + 1]                                                  # branch steps 0..q
    d = (seg[0] - seg[1]).norm(dim=-1) / seg.square().mean(-1).mean(0).sqrt().clamp(min=1e-9) / seg.shape[-1] ** .5
    return d.cpu().numpy()


def main():
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    res = {}
    for k in (1, 3, 10):
        model, cfg = NP.IF.load((HERE / f"run/ckpt_{k:02d}.pt").resolve())
        model.eval()
        res[k] = {}
        for name in ("recovery", "energy", "hunger"):
            cs = [curve(model, cfg, r) for r in TJ.pair_rows(name)]
            L = min(len(c) for c in cs)
            m = np.mean([c[:L] for c in cs], 0)
            res[k][name] = {"q": L - 1, "mean_by_step": m.tolist(),
                            "at_6": float(m[6]), "at_15": float(m[15]), "at_due": float(m[L - 1]),
                            "due_over_15": float(m[L - 1] / m[15]) if m[15] > 0 else None}
            print(k, name, {kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in res[k][name].items() if kk != "mean_by_step"}, flush=True)
        del model
        torch.cuda.empty_cache()
    (OUT / "trace.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
