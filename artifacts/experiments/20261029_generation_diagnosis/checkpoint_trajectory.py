"""PLAN v2 addendum G (user, 10-29): how did the field's handling of consequences evolve over P1 world seed 1?

QUESTION: across checkpoints 1 -> 10 (2.56 M frames apart), did the field's conditional-mean projection of small
consequences (wood, drink, sapling, table, damage) approach the recorded values, stall, or regress, while the common
consequence (movement) stayed carried?
COMPETING READINGS: still improving at checkpoint 10 / plateaued early / regressed late.
CONTROL: matched rows (the same (episode, t) per class at every checkpoint, chosen from labels only: tests F and A's
first 1,000 rows), matched noise (one generator, same seed and call order per checkpoint), references re-estimated
with each checkpoint's own encoder (value means, table means, the context frame's own tokens for movement), so every
checkpoint is read on its own real-target scale. Checks per checkpoint: |bias| at sigma 0.05 < 0.01; SAFE within
0.05 at sigma <= 6.67; checkpoint 10 reproduces tests A / F within their SE.
Reading: this run's trajectory only. It does not show what longer training would do, nor separate capacity, loss
weighting or noise-level coverage (Codex review, 10-29). Projections, not branch probabilities.
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.checkpoint_trajectory OUT_JSON
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
CAT = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.catalogue")
ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / "artifacts/saccade_m0/p1_world1"
N = 1000
READ_SIGMAS = ("0.05", "0.7", "2.0", "6.67", "20.0")


def main():
    out = Path(sys.argv[1]); out.parent.mkdir(parents=True, exist_ok=True)
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    res, cat_ctx, fc_ctx = {"rows_per_class": N, "checkpoints": {}}, None, None
    for i in range(1, 11):
        path = RUN / f"ckpt_{i:02d}.pt"
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(FC.DEV); model.load_state_dict(ck["model"]); model.eval()
        enc = S.encode(model, L, cfg)
        hm, tm, means = CAT.hud_means(L, enc), CAT.table_means(L, enc), FC.value_means(L, enc)
        if cat_ctx is None:                                              # rows depend on labels only
            cat_ctx = CAT.contexts(L, enc, np.random.default_rng(2031))[0]
            for kind, (cell, arr, col) in CAT.HUD.items():
                x = L["s_body" if arr == 0 else "s_inv"]
                cat_ctx[kind] = [r for r in cat_ctx[kind]
                                 if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= set(hm[kind])]
            fc_ctx = FC.contexts(L, enc, np.random.default_rng(2029))
        gen = torch.Generator(device=FC.DEV).manual_seed(2032)
        row = {"sha256": TR.file_sha256(path), "frames": ck.get("frames")}
        for kind in ("move", "blocked", "wood", "sapling", "table", "drink"):
            rows = cat_ctx[kind][:N]
            o = CAT.run(model, enc, L, rows, kind, hm, tm, cfg, gen, heun=False)
            ep = np.array([rows[j][0] for j in o["row_ctx"]])
            est = lambda v: G.estimate(np.asarray(v, float), o["valid"], ep)
            row[kind] = {"f_recorded": est(o["f_real"]), "f_production": est(o["f_gen"]),
                         "field": {s: est(o["f_den"][:, FC.SIGMAS.index(float(s))]) for s in READ_SIGMAS}}
        for c in ("risky", "adjacent", "safe"):
            rows = fc_ctx[c][:N]
            o = FC.run(model, enc, rows, means, cfg, gen)
            ep = np.array([rows[j][0] for j in o["row_ctx"]])
            est = lambda v: G.estimate(np.asarray(v, float), np.ones(len(ep), bool), ep)
            row[c] = {"f_recorded": est(o["f_real"]), "f_production": est(o["f_gen"]),
                      "death_head_production": est(o["death_gen"]),
                      "field": {s: est(o["f_den"][:, FC.SIGMAS.index(float(s))]) for s in READ_SIGMAS}}
        res["checkpoints"][str(i)] = row
        print(i, {k: (round(v["f_recorded"]["est"], 3), round(v["field"]["6.67"]["est"], 3), round(v["f_production"]["est"], 3))
                  for k, v in row.items() if isinstance(v, dict) and "field" in v}, flush=True)
        out.write_text(json.dumps(res, indent=1))
        del model, enc
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
