"""T2 raw-pixel probe reference, memory-lean: every 2nd DEV frame, float16 patches; same probes as t2_eval.Probes.
Usage: t2_raw_ref.py --data DIR --out FILE"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from t2_eval import load, Probes
from t2_joint import Encoder

p = argparse.ArgumentParser(); p.add_argument("--data", required=True); p.add_argument("--out", required=True)
a = p.parse_args()
fr, m, lab, fit, win = load(a.data)
keep = np.arange(0, len(fr), 2)
P = np.concatenate([Encoder.patches(torch.from_numpy(np.ascontiguousarray(fr[keep[i:i + 4096]]))).numpy().astype(np.float16)
                    for i in range(0, len(keep), 4096)]).astype(np.float32)
L = {k: v[keep] for k, v in lab.items()}; f = fit[keep]; s = ~f
pr = Probes(P, L, f)
res = {"raw_pixels_every2": pr.score(P[s], L["mob"][s], L["view"][s], L["vit"][s])}
json.dump(res, open(a.out, "w"), indent=1); print(json.dumps(res))
