"""Follow-up 1 (PLAN.md): P1-F4's determinant curves and actor interface at K = 8, the same populations, seeds, reader
protocol and classes as F4's readout (../20261106_p1f4/consequences.py's curves half, interface.py), only the sampler's
step count changed. Usage (GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.steps8"""
import copy, importlib, json, os, sys
from pathlib import Path

import numpy as np

from saccade import train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
I = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
HERE = Path(__file__).resolve().parent
CKPT = "artifacts/experiments/20261106_p1f4/run/ckpt_10.pt"
STEPS = 8
_load = I.load


def load(path):
    model, cfg = _load(path)
    cfg = copy.deepcopy(cfg)
    cfg["edm"]["sampler"]["steps"] = STEPS
    return model, cfg


def curves(out):
    model, cfg = load(CKPT)
    assert model.denoiser.shortcut and model.denoiser.dense
    s1 = S.load_reader(TR.ROOT / Path(CKPT).parent / "s1/results.reader.pt", TR.ROOT / CKPT, CV.DEV)
    L = S.first_episodes(dict(np.load(CV.LABELS / "s0_dev_labels.npz")), 0)
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    lab = CV.strata(T)
    rows = CV.select(lab, np.random.default_rng(2106), 1.0)
    raw = {}
    for i, pop in enumerate(CV.POPS):
        o = CV.generate(model, cfg, enc, L, T, rows[pop], 4100 + i, R, s1)
        o["stratum"] = lab[pop][o["j"]].astype(str)
        raw.update({f"{pop}/{k}": x for k, x in o.items()})
        print(pop, "done", len(o["j"]), flush=True)
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "run.json").write_text(json.dumps({"checkpoint": CKPT, "sha256": TR.file_sha256(TR.ROOT / CKPT),
                                              "sampler": f"shortcut, K = {STEPS}, from pure noise", "K": CV.K, "readers": valid,
                                              "populations": {p: {s: int((lab[p][rows[p]] == s).sum()) for s in sorted(set(lab[p][rows[p]]))}
                                                              for p in CV.POPS}}, indent=1, default=str))
    CV.summarise(out)


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out = HERE / "readings/k8"; (out / "curves").mkdir(parents=True, exist_ok=True)
    curves(out / "curves")
    I.load = load
    sys.argv = ["interface", str(out / "interface.json"), "--policy=random", f"f4_{STEPS}steps={CKPT}"]
    I.main()
