"""Rows 2 / 4: is imagined entering terrain wrong because it is unknowable (epistemic) or because memory fails to recover it?
(2026-10-15, CPU, saved recursion samples; no model runs)

transport_recursion.py: every model decodes 25-40% of entering cells with the wrong tile class. For each imagined depth k on
camera-consistent trajectories, each cell entering the view (its world position was outside the previous view) is classed by
the real camera track:
  REVISIT   its world position was inside the view in at least one of the 15 context frames (panel frames 1..15): the
            history contains it, so a sufficient memory could recover it
  NEW       never inside any context frame: genuinely unknown (epistemic), only a prior can fill it
Tile error rate per class (frozen bank tile reader, real vs imagined frame; cells with a decoded mob in either are excluded),
and the error of the trivial 'most common tile class in the context frames' guess on NEW cells (a prior baseline).
REVISIT cells are also split by the number of camera scrolls between the last context frame that showed them and the
root (0-2 / 3-5 / 6+), crossed with age, separating addressing (content moved across screen slots) from decay.
REVISIT cells are also split by recency: frames since the world position was last inside a context frame (1-3 / 4-8 / 9-15
frames before the root), separating memory decay (error grows with age) from addressing failure (flat).
Usage: entering_memory.py <model> ... -> artifacts/eda/whiteboard/entering_<model>.json
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'artifacts/experiments/20261012_oracle'))
import audit_s5_direct as A5  # noqa: E402

OUT = ROOT / 'artifacts/eda/whiteboard'


def main():
    import direct_coherence as C
    torch.set_num_threads(8)
    panel = torch.load(A5.BASE / 'panel.pt', map_location='cpu', weights_only=False, mmap=True)
    bank = torch.load(A5.BASE / 'snap_bank_readers.pt', map_location='cpu', weights_only=False)
    head = torch.load(A5.BASE / 'learned_innovation/eval_u3000_h01/mob_heads.pt', map_location='cpu', weights_only=False)['heads']['real']
    selected = set(json.loads((A5.BASE / 'direct_generator/data/existing_eval_selection.json').read_text())['selected_indices'])
    scroll = A5.load_scroll()
    shifts = torch.as_tensor(scroll.SHIFTS)
    tok = panel['tokens'].float()                                          # [R,32,81,192] frames 0..31; context 1..15, root 15
    R = len(tok)
    sh = shifts[scroll.estimate(tok[:, :-1], tok[:, 1:])]                  # [R,31,2] shift from frame f to f+1
    dec_real = C.decode(tok, bank, head)
    rtile, rmob = dec_real['tile'], dec_real['mob']['real'][..., :3].any(-1)
    # world coordinates: origin = frame 1's view; position of view f = cumulative shifts (successor[r,c] = prev[r+dr,c+dc])
    origin = torch.zeros(R, 32, 2, dtype=torch.long)
    origin[:, 1:] = sh.cumsum(1)
    origin = origin - origin[:, 1:2]                                       # frame 1 at (0, 0)
    for name in sys.argv[1:]:
        imag = torch.load(A5.BASE / f'transport/recursion_samples_{name}.pt', weights_only=False)   # [R,K,16,81,192] fp16
        K = imag.shape[1]
        acc = {p: {c: [0, 0] for c in ('REVISIT', 'NEW')} for p in ('bright', 'dark')}
        prior = {p: [0, 0] for p in ('bright', 'dark')}
        for i in range(R):
            part = 'bright' if i in selected else 'dark'
            seen, last = set(), {}                                          # world cells inside any context frame 1..15
            for f in range(1, 16):
                o = origin[i, f]
                for r in range(7):
                    for c in range(9):
                        seen.add((r + int(o[0]), c + int(o[1]))); last[(r + int(o[0]), c + int(o[1]))] = f
            ctx_tiles = rtile[i, 1:16].flatten()
            mode_tile = int(ctx_tiles[ctx_tiles >= 0].mode().values)
            img = imag[i].float()
            idec = C.decode(img, bank, head)
            itile, imob = idec['tile'], idec['mob']['real'][..., :3].any(-1)    # [K,16,63]
            prev_s = shifts[scroll.estimate(torch.cat([tok[i, 15][None, None].expand(K, 1, -1, -1), img[:, :-1]], 1), img)]  # [K,16,2]
            cam_ok = (prev_s == sh[i, 15:31][None]).all(-1).cumprod(1).bool()
            for k in range(16):
                f = 16 + k
                o, op = origin[i, f], origin[i, f - 1]
                for r in range(7):
                    for c in range(9):
                        x = r * 9 + c
                        if x == 31:
                            continue
                        w = (r + int(o[0]), c + int(o[1]))
                        if 0 <= w[0] - int(op[0]) < 7 and 0 <= w[1] - int(op[1]) < 9:
                            continue                                       # visible in the previous frame: not entering
                        cls = 'REVISIT' if w in seen else 'NEW'
                        if cls == 'REVISIT':
                            age = 15 - last[w]
                            sub = 'REVISIT_age0-2' if age <= 2 else ('REVISIT_age3-7' if age <= 7 else 'REVISIT_age8-14')
                            nscroll = int((sh[i, last[w]:15] != 0).any(-1).sum())
                            sub2 = f"{sub}|scrolls{'0-2' if nscroll <= 2 else ('3-5' if nscroll <= 5 else '6+')}"
                        real_t = int(rtile[i, f, x])
                        if bool(rmob[i, f, x]):
                            continue
                        ok = cam_ok[:, k] & ~imob[:, k, x]
                        if ok.any():
                            acc[part][cls][0] += int((itile[ok, k, x] != real_t).sum()); acc[part][cls][1] += int(ok.sum())
                            if cls == 'REVISIT':
                                for key in (sub, sub2):
                                    acc[part].setdefault(key, [0, 0])
                                    acc[part][key][0] += int((itile[ok, k, x] != real_t).sum()); acc[part][key][1] += int(ok.sum())
                        if cls == 'NEW':
                            prior[part][0] += int(mode_tile != real_t); prior[part][1] += 1
        res = {'model': name}
        for part in ('bright', 'dark'):
            res[part] = {c: {'cells': v[1], 'tile_error': round(v[0] / max(v[1], 1), 4)} for c, v in acc[part].items()}
            res[part]['NEW_context_mode_prior_error'] = round(prior[part][0] / max(prior[part][1], 1), 4)
        (OUT / f'entering_{name}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
