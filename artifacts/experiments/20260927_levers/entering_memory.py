"""10-10 PROGRAM C4 (P-B's epistemic part, and memory): terrain entering the view is 30-49% of a world's one-step L2 error and is not
random (the map is the same under every env key). Is it terrain the world SAW earlier in its input window (a memory failure), or
terrain it never saw (epistemic: only a generative model can produce it)? CPU, on rollout_export.py's teacher-forced predictions.
World coordinates: the view's offset o_t is accumulated over the sequence [context 0..3, factual future 1..16]: context shifts from
scroll.estimate on the tokens (no simulator state for context frames), future shifts from the TRUE tile maps (ties by the token
estimate), with scroll.estimate's convention (content at (r, c) in frame t+1 was at (r + dr, c + dc) in frame t: o_{t+1} = o_t +
(dr, dc), world = o + (r, c)).
For the teacher-forced prediction of future frame k (k = 1..16; the world's window = the last min(4 + k, cap) true frames, cap = world
frames - 1), each ENTERING cell (no source after the shift from frame k-1) is `seen` if its world position was inside the view of any
window frame, else `never_seen`. Per world: count, mean L2 error of the prediction, tile-class accuracy of the predicted token
(scoreboard.AllProbes, fitted on true TRAIN tokens; judged on TEST roots), and for `seen` cells the oracle that copies the cell's
token from the most recent window frame showing it (what perfect position-keyed memory gives, despite re-encoding).
Usage: entering_memory.py <variant> ...  -> artifacts/eda/scoreboard_v1/<variant>/entering_memory.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
SHIFTS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
H = 16


def main():
    import teval as T
    import encoder_diag as ED
    import scoreboard as SB
    from scroll import estimate
    torch.set_num_threads(6)
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    cache = T.build_cache('raw', torch.device('cpu'))
    R = len(meta['seed'])
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    seq = torch.cat([cache['ctx'], cache['fut']], 1)                                                 # [R,20,81,192] fp16
    vis = torch.cat([meta['root_visible'][:, None], meta['future_visible'][:, 0]], 1)               # frames 3..19
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()                                              # [R,16]
    # shifts between consecutive frames of the sequence (index t -> t+1), t = 0..18
    sh = torch.zeros(R, 19, dtype=torch.long)
    for t in range(3):
        sh[:, t] = torch.cat([estimate(seq[i:i + 64, t].float(), seq[i:i + 64, t + 1].float()) for i in range(0, R, 64)])
    for t in range(3, 19):
        ta = ED.frame_state(vis[:, t - 3])[0]; tb = ED.frame_state(vis[:, t - 2])[0]
        frac = []
        for dr, dc in SHIFTS:
            src = ED.neighbour(ta, dr, dc, -1)
            ov = src >= 0
            frac.append(((src != tb) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
        est = torch.cat([estimate(seq[i:i + 64, t].float(), seq[i:i + 64, t + 1].float()) for i in range(0, R, 64)])
        sh[:, t] = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)
    d = torch.tensor(SHIFTS)[sh]                                                                      # [R,19,2]
    off = torch.cat([torch.zeros(R, 1, 2, dtype=torch.long), d.cumsum(1)], 1)                        # [R,20,2] offset of each frame
    tiles_true = ED.frame_state(vis.flatten(0, 1))[0].view(R, 17, 7, 9)                               # frames 3..19
    for name in sys.argv[1:]:
        dd = OUT / name
        meta_v = json.loads((dd / 'meta.json').read_text())
        cap = int(meta_v['window_cap'])
        tf = torch.from_numpy(np.load(dd / 'tf.npy', mmap_mode='r'))
        acc = {}
        for k in range(1, H + 1):
            f = 3 + k                                                                                  # absolute index of future frame k
            lo = max(0, f - (4 if k == 1 else min(4 + k - 1, cap)))
            dr, dc = d[:, f - 1, 0], d[:, f - 1, 1]                                                    # shift from frame f-1 to f
            m_root = alive[:, k - 1]
            for r_ in range(7):
                for c_ in range(9):
                    ent = m_root & ~((r_ + dr >= 0) & (r_ + dr < 7) & (c_ + dc >= 0) & (c_ + dc < 9))   # no source: entering
                    if not ent.any():
                        continue
                    ii = torch.where(ent)[0]
                    w = off[ii, f] + torch.tensor([r_, c_])                                            # world position [n,2]
                    rel = w[:, None, :] - off[ii, lo:f]                                                # position in each window frame
                    inside = (rel[..., 0] >= 0) & (rel[..., 0] < 7) & (rel[..., 1] >= 0) & (rel[..., 1] < 9)
                    seen = inside.any(1)
                    pred = tf[ii, k - 1, r_ * 9 + c_].float(); truth = seq[ii, f, r_ * 9 + c_].float()
                    e2 = ((pred - truth) ** 2).sum(-1)
                    ok = (probes.tile(pred).argmax(-1) == tiles_true[ii, k, r_, c_]).float()
                    # oracle: the most recent window frame showing it
                    last = torch.where(inside, torch.arange(lo, f)[None].expand_as(inside), torch.full_like(inside, -1, dtype=torch.long)).max(1).values
                    orc = torch.full_like(e2, float('nan'))
                    if seen.any():
                        js = torch.where(seen)[0]
                        tt = last[js]; rr = rel[js, (tt - lo)]
                        mem = seq[ii[js], tt, rr[:, 0] * 9 + rr[:, 1]].float()
                        orc[js] = ((mem - truth[js]) ** 2).sum(-1)
                    grp = 'k1' if k == 1 else ('k2-4' if k <= 4 else ('k5-8' if k <= 8 else 'k9-16'))
                    for sv, sn in ((True, 'seen'), (False, 'never_seen')):
                        mk = seen == sv
                        tm = mk & test[ii]
                        a = acc.setdefault(f'{grp}/{sn}', [0, 0.0, 0.0, 0, 0.0, 0])
                        a[0] += int(mk.sum()); a[1] += float(e2[mk].sum()); a[2] += float(ok[tm].sum()); a[3] += int(tm.sum())
                        if sv:
                            a[4] += float(orc[mk].nansum()); a[5] += int((~orc[mk].isnan()).sum())
        res = {'variant': name, 'window_cap': cap}
        for key, a in sorted(acc.items()):
            res[key] = {'cells': a[0], 'mean_L2': round(a[1] / max(a[0], 1), 3), 'tile_acc_test': round(a[2] / max(a[3], 1), 4),
                        'oracle_recall_mean_L2': round(a[4] / a[5], 3) if a[5] else None}
        tot_seen = sum(a[0] for k_, a in acc.items() if k_.endswith('/seen')); tot = sum(a[0] for a in acc.values())
        res['seen_share_of_entering'] = round(tot_seen / max(tot, 1), 4)
        (dd / 'entering_memory.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
