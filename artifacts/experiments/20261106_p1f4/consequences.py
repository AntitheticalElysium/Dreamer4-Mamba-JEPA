"""P1-F4 consequence readout (PLAN.md "Readout"): exp 2's determinant curves (../20261106_determinant_curves/curves.py)
and crafting at the frame level (../20261106_decision_levels/crafting.py), same populations, strata, caps, seeds, K = 4
futures and reader protocol, on an F4 checkpoint through its own sampler (shortcut forcing, 4 evaluations from pure
noise: no start recipe, the sampler starts at the trained endpoint) and its own S1 reader. F3's references: curves at 16
EDM steps (exp 2) and with the prefix-20 start (decision levels T2), crafting with prefix-20.
Usage (GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_p1f4.consequences CKPT OUT_DIR [--smoke]
"""
import importlib, json, os, sys
from pathlib import Path

import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
CR = importlib.import_module("artifacts.experiments.20261106_decision_levels.crafting")


def main(ckpt, out, smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(ckpt)
    assert model.denoiser.shortcut and model.denoiser.dense and cfg["edm"]["sampler"]["steps"] == 4
    s1 = S.load_reader(Path(TR.ROOT / ckpt).parent / "s1/results.reader.pt", TR.ROOT / ckpt, CV.DEV)
    L = S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 60 if smoke else 0)
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    head = {"checkpoint": ckpt, "sha256": TR.file_sha256(TR.ROOT / ckpt), "sampler": "shortcut, K = 4, from pure noise", "K": CV.K}
    # determinant curves (curves.main without its F3 reproduction check and step override)
    oc = out / ("smoke" if smoke else "curves"); oc.mkdir(parents=True, exist_ok=True)
    R, valid = CV.fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    lab = CV.strata(T)
    rows = CV.select(lab, np.random.default_rng(2106), 0.02 if smoke else 1.0)
    raw = {}
    for i, pop in enumerate(CV.POPS):
        o = CV.generate(model, cfg, enc, L, T, rows[pop], 4100 + i, R, s1)
        o["stratum"] = lab[pop][o["j"]].astype(str)
        raw.update({f"{pop}/{k}": x for k, x in o.items()})
        print(pop, "done", len(o["j"]), flush=True)
    np.savez_compressed(oc / "raw.npz", **raw)
    (oc / "run.json").write_text(json.dumps({**head, "readers": valid, "populations": {
        p: {s: int((lab[p][rows[p]] == s).sum()) for s in sorted(set(lab[p][rows[p]]))} for p in CV.POPS}}, indent=1, default=str))
    CV.summarise(oc)
    # crafting (crafting.main's selection and generation, this sampler); the tool slots join the readers
    ok = out / ("crafting_smoke" if smoke else "crafting"); ok.mkdir(parents=True, exist_ok=True)
    CV.HUD = CR.HUD
    R2, valid2 = CV.fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    lab2 = CR.strata(T, L)
    rng, keep = np.random.default_rng(2108), []
    for s in sorted(set(lab2) - {""}):
        idx, cap = np.nonzero(lab2 == s)[0], CR.CAPS[s.split("_")[1]]
        keep.append(idx if cap is None or len(idx) <= cap else rng.choice(idx, cap, replace=False))
    rows2 = np.sort(np.concatenate(keep))
    o = CV.generate(model, cfg, enc, L, T, rows2, 4200, R2, s1)
    o["stratum"] = lab2[o["j"]].astype(str)
    np.savez_compressed(ok / "raw.npz", **o)
    (ok / "run.json").write_text(json.dumps({**head, "readers": valid2, "populations": {
        s: int((lab2[rows2] == s).sum()) for s in sorted(set(lab2[rows2]))}}, indent=1, default=str))
    CR.summarise(ok, smoke)


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]), "--smoke" in sys.argv)
