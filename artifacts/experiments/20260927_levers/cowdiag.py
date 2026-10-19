"""10-10 (E22MS self-fed cow inflation: 3.65 cows at depth 16 vs 0.46 real, 2.98 even with the posterior's TRUE codes). CPU.
Where do the extra cows come from? For each variant's self-fed 16-step rollout (gen.npy; the factual future 0 as the real
reference), every decoded cow at step k (scoreboard.AllProbes cow score > its calibrated threshold, TEST roots, alive) is
classified against the SAME sequence's previous frame (the root at k = 1). The cells are first aligned by that sequence's own
scroll (scroll.estimate between its consecutive frames; content at (r, c) at k was at (r + dr, c + dc) at k - 1):
  persisted   a cow in the same (aligned) cell at k - 1
  moved       a cow in an aligned 4-neighbour at k - 1, and that neighbour has NO cow at k
  duplicated  a cow in an aligned 4-neighbour at k - 1 that STILL has a cow at k (the source did not vacate)
  created     no cow within the aligned 3 x 3 at k - 1 (or only a diagonal)
  entering    the cell has no source after the scroll (it entered the view)
Reported per variant and depth group (1, 2-4, 5-8, 9-16): cows per frame, the share of each class, and the mean cow score per
class (the threshold is calibrated on true tokens; a faint superposed cow scores just above it).
Variants: real (future 0's true tokens), E22MS_postcodes (the posterior's codes of the true transitions), E22MS_sample0..3
(prior samples), E22MS_zeroed (Δ = 0), DS (D scratch, deterministic).
Usage: cowdiag.py  -> artifacts/eda/health_chain_v1/cowdiag.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/health_chain_v1'
SBV = ROOT / 'artifacts/eda/scoreboard_v1'
H = 16
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
VARIANTS = ('real', 'E22MS_postcodes', 'E22MS_sample0', 'E22MS_sample1', 'E22MS_sample2', 'E22MS_sample3', 'E22MS_zeroed', 'DS')
CLASSES = ('persisted', 'moved', 'duplicated', 'created', 'entering')
COW = 2                                                                                              # AllProbes mob order


def shifted(grid, dr, dc, fill):
    """[n,7,9] -> out[r, c] = grid[r + dr, c + dc], `fill` off-view"""
    out = torch.full_like(grid, fill)
    rs, re = max(0, -dr), 7 - max(0, dr)
    cs, ce = max(0, -dc), 9 - max(0, dc)
    out[:, rs:re, cs:ce] = grid[:, rs + dr:re + dr, cs + dc:ce + dc]
    return out


def main():
    import teval as T
    import scoreboard as SB
    from scroll import estimate
    torch.set_num_threads(6)
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    thr = float(probes.thr[COW])
    alive0 = ~meta['future_dead'][:, 0].cumsum(1).bool()
    ii = torch.where(~train_roots)[0]
    root = cache['ctx'][ii, -1].float()
    mapc = torch.tensor(SB.MAPC)

    def cow_grid(tok):                                                                               # [n,81,192] -> score [n,7,9]
        s = torch.zeros(len(tok), 63)
        s[:, mapc] = probes.mob(tok[:, mapc].float().flatten(0, 1))[:, COW].view(len(tok), 62)
        return s.view(-1, 7, 9)
    res = {}
    for v in VARIANTS:
        seq = cache['fut'][ii] if v == 'real' else torch.from_numpy(np.ascontiguousarray(np.load(SBV / v / 'gen.npy', mmap_mode='r')[ii.numpy()]))
        prev_tok, prev_s = root, cow_grid(root)
        acc = {}
        for k in range(1, H + 1):
            cur_tok = seq[:, k - 1].float()
            cur_s = cow_grid(cur_tok)
            sh = estimate(prev_tok, cur_tok)
            p_prev, p_cur = prev_s > thr, cur_s > thr
            cls = torch.full(p_cur.shape, -1, dtype=torch.long)
            for si, (dr, dc) in enumerate(SHIFTS):
                m = sh == si
                if not m.any():
                    continue
                src = shifted(p_prev[m].long(), dr, dc, -1)                                          # aligned previous cows; -1 = entering
                c = torch.full(src.shape, CLASSES.index('created'), dtype=torch.long)
                cur_m = p_cur[m]
                for ndr, ndc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nb_src = shifted(src, ndr, ndc, 0) == 1                                          # a cow at that aligned neighbour at k-1
                    nb_now = shifted(cur_m.long(), ndr, ndc, 0) == 1                                 # ... and that neighbour has a cow at k
                    c = torch.where(nb_src & ~nb_now & (c == CLASSES.index('created')), torch.full_like(c, CLASSES.index('moved')), c)
                    c = torch.where(nb_src & nb_now & (c != CLASSES.index('moved')), torch.full_like(c, CLASSES.index('duplicated')), c)
                c = torch.where(src == 1, torch.full_like(c, CLASSES.index('persisted')), c)
                c = torch.where(src == -1, torch.full_like(c, CLASSES.index('entering')), c)
                cls[m] = c
            al = alive0[ii, k - 1][:, None, None].expand_as(p_cur)
            grp = '1' if k == 1 else ('2-4' if k <= 4 else ('5-8' if k <= 8 else '9-16'))
            a = acc.setdefault(grp, {'frames': 0, **{c: [0, 0.0] for c in CLASSES}})
            a['frames'] += int(alive0[ii, k - 1].sum())
            for ci, c in enumerate(CLASSES):
                m = p_cur & al & (cls == ci)
                a[c][0] += int(m.sum()); a[c][1] += float(cur_s[m].sum())
            prev_tok, prev_s = cur_tok, cur_s
        row = {}
        for g, a in acc.items():
            n = sum(a[c][0] for c in CLASSES)
            row[g] = {'cows_per_frame': round(n / a['frames'], 4),
                      'share': {c: round(a[c][0] / n, 4) if n else None for c in CLASSES},
                      'mean_score': {c: round(a[c][1] / a[c][0], 3) if a[c][0] else None for c in CLASSES}}
        res[v] = row
        print(json.dumps({v: row}), flush=True)
    res['threshold'] = thr
    (OUT / 'cowdiag.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
