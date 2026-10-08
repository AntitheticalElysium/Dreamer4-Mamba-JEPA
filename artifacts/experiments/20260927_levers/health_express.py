"""Why do L1 worlds express a small share of the hits their own h63 supports, and D 72%? (2026-10-09 09:20.) Row 9 (window 10)
extended states of each world; the calibrated drawable probe of health_evidence (balanced MLP on h63, weighted Platt to the
TRAIN mix, P > 0.5) fitted per world; on held hits that are drawable: emitted share, the corrt mixture's generate weight at
token 63, whether the generate candidate beats copy (per-dim L1 vs the true next token), and the emitted token's L1. Split of the
drawable-but-not-emitted hits into: generator good (beats copy) but weight < 0.5 (router), generator not better than copy
(generator), both. Usage: health_express.py <name=__w10__ext.pt> ...  -> artifacts/eda/health_chain_v1/express.json
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'

if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    import teval as T
    import health_evidence as HE
    sub = torch.load(OUT / 'subset.pt', weights_only=False); st = torch.load(OUT / 'strata.pt', weights_only=False)
    mm = np.memmap(OUT / 'subset.f16', dtype=np.float16, mode='r', shape=tuple(sub['shape']))
    meta, tr, ts = T.split(); P = T.Probes(T.build_cache('raw', torch.device('cpu')), meta, tr, ts)
    read = lambda hud: (P.hud(hud.float().flatten(-2))[..., 0] * 9).float()
    N = len(sub['classes']); hit, unch = sub['classes'] == 1, sub['classes'] == 2; keep = hit | unch
    fit = sub['fit']; held = ~fit
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    cur = torch.cat([read(torch.from_numpy(np.array(mm[i:i + 2048, 14, 63:81]))) for i in range(0, N, 2048)])
    t14 = torch.from_numpy(np.array(mm[:, 14, 63])).float(); t15 = torch.from_numpy(np.array(mm[:, 15, 63])).float()
    copy_l1 = (t14 - t15).abs().mean(-1)
    res = {}
    for name, path in (a.split('=', 1) for a in sys.argv[1:]):
        r = torch.load(path, weights_only=False)
        Pr = HE.weighted_probe(r['h_ext'][:, 0].float(), hit, st['weights'], keep & fit & ~inner, keep & inner)
        draw = (Pr > 0.5) & hit & held
        em = read(r['hud']) < cur - 1.5
        gw = r['weights63'][:, 5].float()
        g = F.layer_norm(r['gen63'].float(), (192,)); gl = (g - t15).abs().mean(-1)
        out_l1 = (r['hud'][:, 0].float() - t15).abs().mean(-1)
        good = gl < copy_l1
        miss = draw & ~em
        res[name] = {'drawable_hits': int(draw.sum()), 'emitted_of_drawable': round(float(em[draw].float().mean()), 3),
                     'gen_weight_on_drawable': round(float(gw[draw].mean()), 3),
                     'gen_weight_on_drawable_emitted': round(float(gw[draw & em].mean()), 3) if (draw & em).any() else None,
                     'gen_weight_on_drawable_missed': round(float(gw[miss].mean()), 3) if miss.any() else None,
                     'gen_beats_copy_on_drawable': round(float(good[draw].float().mean()), 3),
                     'missed_split': {'router (gen good, weight < 0.5)': int((miss & good & (gw < 0.5)).sum()),
                                      'generator (gen not better than copy)': int((miss & ~good).sum()),
                                      'gen good and weight >= 0.5': int((miss & good & (gw >= 0.5)).sum())},
                     'gen_weight_on_held_unchanged': round(float(gw[unch & held].mean()), 3),
                     'output_l1_drawable_hits': round(float(out_l1[draw].mean()), 3), 'copy_l1_drawable_hits': round(float(copy_l1[draw].mean()), 3)}
        print(json.dumps({name: res[name]}), flush=True)
    (OUT / 'express.json').write_text(json.dumps(res, indent=1))
