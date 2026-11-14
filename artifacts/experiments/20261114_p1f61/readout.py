"""Checkpoint readings of P1-F6.1 (PLAN.md, readings): F6's clock instruments without the token-forecast one (gate3),
unchanged and pointed at this run, so every number has F5's and F6's at the same checkpoint beside it.

  trajectory  ../20261111_p1f5/trajectory.py: reward-forecast separation (distance 1), native generation due / not due,
              belief probe (event APs from the predicted belief).
  d1          ../20261113_token_forecast_gate/d1.py: each reset's registration at the tokens, Mamba's input and state.
  trace       ../20261111_p1f5/trace.py: how much of the earlier reset survives in the state to the due step.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.readout CHECKPOINT [stages...]
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
G2 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate2")
D1 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.d1")
TC = importlib.import_module("artifacts.experiments.20261111_p1f5.trace")
TJ, NP = G2.TJ, G2.NP
STAGES = ("trajectory", "d1", "trace")


def main(k, stages):
    ck = (HERE / f"run/ckpt_{k:02d}.pt").resolve()
    sha = TR.file_sha256(ck)
    if "trajectory" in stages:
        TJ.HERE, TJ.OUT = HERE, OUT / "trajectory"
        TJ.OUT.mkdir(parents=True, exist_ok=True)
        TJ.checkpoint(k)
    if "d1" in stages and not (OUT / f"d1/d1_ckpt_{k:02d}.json").exists():
        G2.OUT = D1.OUT = OUT / "d1"
        model, cfg = NP.IF.load(ck)
        L = dict(np.load(G2.CV.LABELS / "s0_dev_labels.npz"))
        G2.d1_features(model, cfg, S.encode(model, L, cfg), L, k)
        del model
        torch.cuda.empty_cache()
        D1.checkpoint(k)
    if "trace" in stages and not (OUT / f"trace_{k:02d}.json").exists():
        model, cfg = NP.IF.load(ck)
        res = {"checkpoint": k, "sha256": sha}
        for name in ("recovery", "energy", "hunger"):
            cs = [TC.curve(model, cfg, r) for r in TJ.pair_rows(name)]
            n = min(len(c) for c in cs)
            m = np.mean([c[:n] for c in cs], 0)
            res[name] = {"q": n - 1, "mean_by_step": m.tolist(), "at_6": float(m[6]), "at_15": float(m[15]), "at_due": float(m[n - 1])}
            print(k, "trace", name, round(res[name]["at_due"], 5), flush=True)
        (OUT / f"trace_{k:02d}.json").write_text(json.dumps(res, indent=1) + "\n")
    assert TR.file_sha256(ck) == sha


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main(int(sys.argv[1]), sys.argv[2:] or STAGES)
