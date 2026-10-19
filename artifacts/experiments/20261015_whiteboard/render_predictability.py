"""Track B (2026-10-15 evening, CPU, frozen Raw encoder, no training): is the rendering change of persistent content
predictable from history?

Same data as rep_layers (597 futures x 16 steps; exact visible state used only to select persistent cells). For each frame
pair (t, t+1) with >= 10 persistent cells (aligned tile and mob channels unchanged), day and camera-still unless stated:
  c_t        the common-mode change (mean dz over persistent cells, 192-d)
  per-cell   dz_j - c_t (content-dependent part)
Predictors, all label-free given the history: zero (no change); previous common mode c_{t-1}; linear extrapolation over the
last two steps; and for the per-cell part, the same cell's previous residual. Reported: share of drift energy explained
(1 - |error|^2 / |dz|^2), overall and for the common / per-cell parts, day vs night.
Usage: render_predictability.py -> artifacts/eda/whiteboard/render_predictability.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rep_layers as RL  # noqa: E402


def main():
    torch.set_num_threads(8)
    import rollouts as Rr
    roots = [r for f in sorted(Rr.STORE.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261015))[:120].tolist()
    enc = RL.encoder()
    seqs = []
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if not bool(r['future_dead'][k].any()):
                seqs.append((torch.cat([r['context'][-1:], r['future_frames'][k]]), torch.cat([r['root_visible'][None], r['future_visible'][k]])))
    frames = torch.cat([s[0] for s in seqs]); vis = torch.cat([s[1] for s in seqs])
    n = len(seqs)
    RL.LAYERS = (12,)
    z = RL.layer_tokens(enc, frames)[12].view(n, 17, 81, 192).float()
    tiles, mobs, light = RL.facts(vis)
    tiles, mobs, light = tiles.view(n, 17, 63), mobs.view(n, 17, 63, 7), light.view(n, 17)
    acc = {key: {'tot': 0.0, 'common_zero': 0.0, 'common_prev': 0.0, 'common_lin': 0.0, 'cell_zero': 0.0, 'cell_prev': 0.0,
                 'full_zero': 0.0, 'full_prev': 0.0, 'pairs': 0} for key in ('day', 'night')}
    for s in range(n):
        prev_c, prev2_c, prev_res = None, None, None
        for t in range(16):
            if RL.camera(tiles[s, t], tiles[s, t + 1]) != (0, 0):
                prev_c = prev2_c = prev_res = None
                continue
            pers = [j for j in range(63) if j != 31 and tiles[s, t + 1, j] == tiles[s, t, j] and bool((mobs[s, t + 1, j] == mobs[s, t, j]).all())]
            if len(pers) < 10:
                prev_c = prev2_c = prev_res = None
                continue
            dz = z[s, t + 1, pers] - z[s, t, pers]                          # [m,192]
            c = dz.mean(0)
            res = dz - c
            if prev_c is not None:
                key = 'night' if light[s, t] < .5 else 'day'
                a = acc[key]
                e_tot = float(dz.square().sum())
                a['tot'] += e_tot; a['pairs'] += 1
                m = len(pers)
                a['common_zero'] += m * float(c.square().sum())
                a['common_prev'] += m * float((c - prev_c).square().sum())
                lin = prev_c + (prev_c - prev2_c) if prev2_c is not None else prev_c
                a['common_lin'] += m * float((c - lin).square().sum())
                pr = torch.stack([prev_res.get(j, torch.zeros(192)) for j in pers])
                a['cell_zero'] += float(res.square().sum())
                a['cell_prev'] += float((res - pr).square().sum())
                a['full_zero'] += e_tot
                a['full_prev'] += float((dz - (prev_c + pr)).square().sum())
            prev2_c, prev_c = prev_c, c
            prev_res = {j: res[i] for i, j in enumerate(pers)}
    out = {}
    for key, a in acc.items():
        if not a['pairs']:
            continue
        out[key] = {'pairs': a['pairs'],
                    'common_share_of_drift': round(a['common_zero'] / a['tot'], 4),
                    'common_explained_by_previous': round(1 - a['common_prev'] / a['common_zero'], 4),
                    'common_explained_by_linear_extrapolation': round(1 - a['common_lin'] / a['common_zero'], 4),
                    'per_cell_explained_by_previous': round(1 - a['cell_prev'] / a['cell_zero'], 4),
                    'total_drift_explained_by_previous_common_plus_cell': round(1 - a['full_prev'] / a['full_zero'], 4)}
    (RL.OUT / 'render_predictability.json').write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
