"""Where does the token drift of persistent content come from? (2026-10-15, CPU, frozen Raw encoder, no training)

rep_layers found the drift of unchanged cells is as large at layer 0 (own 7x7 pixels + position) as at the final layer, so it
is not contextual mixing. Candidates: global illumination (Craftax's daylight cycle darkens every pixel), the position code
(content changes grid position when the camera moves), night static. Same data as rep_layers (persistent cells: own aligned
tile class and mob channels unchanged between t and t+1). Layers 0 and 12. For each frame pair, decompose the change of every
persistent cell, dz = z_{t+1}[j] - z_t[j'], into
  common     the mean dz over the frame's persistent cells (one 192-vector per frame pair: a global shift)
  residual   dz - common
and report the share of drift energy that is common, by stratum (day still / day moved / night still).
Position: the mean token per (grid position, tile class) over persistent no-mob cells; content code = token minus its
position's mean over all classes (removes the position offset); drift of that code for camera-moved cells vs still.
Light: correlation of |common| with the change of the bank light reading between the two frames.
Usage: rep_drift.py -> artifacts/eda/whiteboard/rep_drift.json
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
    files = sorted(Rr.STORE.glob('seed-*.pt'))
    roots = [r for f in files for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(roots), generator=torch.Generator().manual_seed(20261015))[:120].tolist()
    enc = RL.encoder()
    seqs = []
    for ri in pick:
        r = roots[ri]
        for k in range(5):
            if bool(r['future_dead'][k].any()):
                continue
            seqs.append((torch.cat([r['context'][-1:], r['future_frames'][k]]), torch.cat([r['root_visible'][None], r['future_visible'][k]])))
    frames = torch.cat([s[0] for s in seqs]); vis = torch.cat([s[1] for s in seqs])
    n = len(seqs)
    RL.LAYERS = (0, 12)
    toks = {l: v.view(n, 17, 81, 192).float() for l, v in RL.layer_tokens(enc, frames).items()}
    tiles, mobs, light = RL.facts(vis)
    tiles, mobs, light = tiles.view(n, 17, 63), mobs.view(n, 17, 63, 7), light.view(n, 17)
    res = {'sequences': n}
    for l in (0, 12):
        z = toks[l]
        # position-class means for the position-free content code
        pos_mean = z[..., :63, :][~mobs[..., :3].any(-1)].mean(0)            # overall mean (no-mob cells)
        per_pos = torch.zeros(63, 192)
        for j in range(63):
            m = ~mobs[:, :, j, :3].any(-1)
            per_pos[j] = z[:, :, j][m].mean(0) - pos_mean
        rows = {'day_still': [], 'day_moved': [], 'night_still': []}
        rows_pf = {'day_still': [], 'day_moved': [], 'night_still': []}
        light_pairs = []
        for s in range(n):
            for t in range(16):
                dr, dc = RL.camera(tiles[s, t], tiles[s, t + 1])
                pairs = []
                for r in range(7):
                    for c in range(9):
                        if 0 <= r + dr < 7 and 0 <= c + dc < 9:
                            j, jp = r * 9 + c, (r + dr) * 9 + c + dc
                            if 31 in (j, jp):
                                continue
                            if bool(tiles[s, t + 1, j] == tiles[s, t, jp]) and bool((mobs[s, t + 1, j] == mobs[s, t, jp]).all()):
                                pairs.append((j, jp))
                if len(pairs) < 10:
                    continue
                j = torch.tensor([p[0] for p in pairs]); jp = torch.tensor([p[1] for p in pairs])
                dz = z[s, t + 1, j] - z[s, t, jp]                                # [m,192]
                common = dz.mean(0, keepdim=True)
                key = 'night_still' if light[s, t] < .5 and (dr, dc) == (0, 0) else ('day_still' if (dr, dc) == (0, 0) else ('day_moved' if light[s, t] >= .5 else None))
                if key is None:
                    continue
                rows[key].append((float(dz.square().sum()), float((dz - common).square().sum()), len(pairs)))
                dpf = (z[s, t + 1, j] - per_pos[j]) - (z[s, t, jp] - per_pos[jp])    # position-free content code change
                rows_pf[key].append((float(dpf.square().sum()), float((dpf - dpf.mean(0, keepdim=True)).square().sum()), len(pairs)))
                if key == 'day_still':
                    light_pairs.append((float(common.norm()), float(abs(light[s, t + 1] - light[s, t]))))
        out = {}
        for key in rows:
            a = np.array(rows[key]); b = np.array(rows_pf[key])
            if not len(a):
                continue
            out[key] = {'frame_pairs': len(a), 'rms_drift': round(float(np.sqrt(a[:, 0].sum() / a[:, 2].sum() / 192)), 4),
                        'common_mode_share': round(float(1 - a[:, 1].sum() / a[:, 0].sum()), 4),
                        'rms_residual_after_common': round(float(np.sqrt(a[:, 1].sum() / a[:, 2].sum() / 192)), 4),
                        'position_free_rms_drift': round(float(np.sqrt(b[:, 0].sum() / b[:, 2].sum() / 192)), 4),
                        'position_free_rms_after_common': round(float(np.sqrt(b[:, 1].sum() / b[:, 2].sum() / 192)), 4)}
        lp = np.array(light_pairs)
        out['corr_common_norm_vs_light_change'] = round(float(np.corrcoef(lp[:, 0], lp[:, 1])[0, 1]), 4) if len(lp) > 2 else None
        out['position_offset_rms'] = round(float(per_pos.square().mean().sqrt()), 4)
        res[f'layer{l}'] = out
        print(json.dumps({f'layer{l}': out}), flush=True)
    (RL.OUT / 'rep_drift.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
