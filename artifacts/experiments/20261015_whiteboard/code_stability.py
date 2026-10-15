"""Would categorical targets on our frozen Raw tokens be stable? Code flips on persisting content vs changes on real content
(2026-10-16; representation question; GPU inference; exact simulator state labels cells for this audit only)

Data: diagnosis_futures_v1, the same 240 seeded roots as objective_audit.py; their 5 futures x 16 steps (consecutive frames,
alive). Codes: nearest centroid of the Raw K1024 / K4096 k-means codebooks (levers_codebooks_v1, built on Raw tokens). Per
transition t -> t+1, exact camera shift (tile-grid match) and labels (objective_audit.layout):
  persist map cells (tile + mob layer unchanged): code FLIP rate = noise a categorical target would carry
  mob_change / tile_change map cells: code CHANGE rate = signal
  HUD cells: change rate when a displayed value changed vs when none did
Persist flips split by lighting: day steady (light >= 0.99 both frames), day changing (light in [0.5, 0.99) or changing), night
(light < 0.5: random static), and by sleep.
Usage: code_stability.py -> artifacts/eda/whiteboard/code_stability.json
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('TRITON_F32_DEFAULT', 'ieee')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / '20261012_oracle'))
sys.path.insert(0, str(HERE))

import torch

import recontext_stale as RS
import objective_audit as OA

ROOT = HERE.parents[2]


@torch.no_grad()
def main():
    books = {K: torch.load(ROOT / f'artifacts/eda/levers_codebooks_v1/raw_K{K}.pt', weights_only=False, map_location='cpu')['codes'].float().cuda()
             for K in (1024, 4096)}
    rows = [r for f in sorted(OA.DIAG.glob('seed-*.pt')) for r in torch.load(f, weights_only=False)]
    pick = torch.randperm(len(rows), generator=torch.Generator().manual_seed(20261060))[:240].tolist()
    enc = RS.Encoder()
    keys = ['persist_day_steady', 'persist_day_changing', 'persist_night', 'persist_asleep', 'mob_change', 'tile_change', 'hud_value_changed', 'hud_unchanged']
    acc = {K: {k: [0, 0] for k in keys} for K in books}
    for ri in pick:
        r = rows[ri]
        for f in range(5):
            dead = r['future_dead'][f].bool()
            frames = torch.cat([r['context'][-1:], r['future_frames'][f]])                        # [17,63,63,3]
            vis = torch.cat([r['root_visible'][None], r['future_visible'][f]])                    # [17,1534]
            tok = enc.recontext(enc.content(frames)).float()                                       # [17,81,192]
            codes = {K: torch.cdist(tok.flatten(0, 1), B).argmin(-1).view(17, 81).cpu() for K, B in books.items()}
            tiles, lay, shown = OA.layout(vis)
            light, sleep = vis[:, 1521].float(), vis[:, 1520].float()
            for t in range(16):
                if t > 0 and bool(dead[t - 1]) or bool(dead[t]):
                    break
                dr, dc = OA.shift_of(tiles[t], tiles[t + 1])
                src = OA.sources(dr, dc)
                hud_changed = bool((shown[t + 1] != shown[t]).any())
                l0, l1 = float(light[t]), float(light[t + 1])
                if sleep[t] > 0 or sleep[t + 1] > 0:
                    pk = 'persist_asleep'
                elif min(l0, l1) < 0.5:
                    pk = 'persist_night'
                elif min(l0, l1) >= 0.99 and abs(l1 - l0) < 1e-6:
                    pk = 'persist_day_steady'
                else:
                    pk = 'persist_day_changing'
                for K in books:
                    c0, c1 = codes[K][t], codes[K][t + 1]
                    for c in range(81):
                        if c >= 63:
                            k = 'hud_value_changed' if hud_changed else 'hud_unchanged'
                            flip = c1[c] != c0[c]
                        elif c == 31 or src[c] < 0:
                            continue
                        else:
                            s = src[c]
                            if tiles[t + 1, c] != tiles[t, s]:
                                k = 'tile_change'
                            elif lay[t + 1, c] != lay[t, s]:
                                k = 'mob_change'
                            else:
                                k = pk
                            flip = c1[c] != c0[s]
                        acc[K][k][0] += int(flip); acc[K][k][1] += 1
    res = {'roots': len(pick)}
    for K, A in acc.items():
        res[f'K{K}'] = {k: {'n': v[1], 'code_change_rate': round(v[0] / max(v[1], 1), 4)} for k, v in A.items()}
    print(json.dumps(res, indent=1))
    (ROOT / 'artifacts/eda/whiteboard/code_stability.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
