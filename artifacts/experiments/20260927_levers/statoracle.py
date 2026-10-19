"""10-10 PROGRAM (why does the objective fight us? the loss vs the network, by substitution). CPU.
A deterministic world trained with a per-element loss can at best output that loss's optimal STATISTIC of the next-token distribution:
the per-element mean (squared error) or the per-element median (L1). This replaces the network by the statistic, computed EXACTLY from
the five real futures of each root (same actions, different env keys; stochdiag's tokens). Given the full simulator state at the root,
that is better than any world can do. The statistic is read with the facts every world is judged on (scoreboard.AllProbes, fitted on
true TRAIN tokens, read on TEST roots). If the statistics show the worlds' failures, those failures belong to the objective, not to
the network.
Per depth k (1..16; TEST roots whose five futures are all alive at k), for mean5, median5, the root frame (copy) and, as the
reference, each real future:
  census     decoded mob count per type (zombie, skeleton, cow, arrow) over the 62 map cells; the simulator's mean count
  mob_cells  on cells where >= 1 future has the mob (per type), the share where the statistic decodes it present, by how many of the
             five futures have it there (1..5)
  tiles      decoded tile accuracy vs future 0's simulator tile, on cells where the five futures' tiles agree / disagree
  hud_events rate of decoded health / food / drink / energy decrements (readout falls by > 0.5 from step k-1; the root at k = 1),
             as scoreboard's hud_events
  error      mean over the five futures of sum |s - y_j|^2 (L2) and sum |s - y_j| (L1) per frame; the floor (unbiased variance)
World vs statistic (depth 1, teacher-forced, where all five futures share the world's window). For each named variant's export
(scoreboard_v1/<variant>/tf.npy, depth 1), on the elements where mean5 and median5 differ by > 0.1: the share where the world's
prediction is closer to the median than to the mean; and |p - mean5|^2, |p - median5|^2 per frame.
Usage: statoracle.py [<variant> ...]  -> artifacts/eda/health_chain_v1/statoracle.json
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
STATS = ('mean5', 'median5', 'root', 'real')
MOBS = ('zombie', 'skeleton', 'cow', 'arrow')
HUDN = ('health', 'food', 'drink', 'energy')


def main():
    import teval as T
    import stochdiag as SD
    import scoreboard as SB
    torch.set_num_threads(4)
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    thr = torch.tensor(probes.thr)
    fut5 = SD.token_cache(torch.device('cpu'))[0]                                                    # [R,5,16,81,192] fp16
    alive5 = ~meta['future_dead'].cumsum(2).bool().any(1)                                             # [R,16]
    root = cache['ctx'][:, -1]

    def dec(tok):
        """[n,81,192] -> tile [n,62], mobs present [n,62,4], hud readout [n,4]"""
        cells = tok[:, SB.MAPC].float().flatten(0, 1)
        return (probes.tile(cells).view(len(tok), 62, 17).argmax(-1), (probes.mob(cells) > thr).view(len(tok), 62, 4),
                probes.hud(tok[:, 63:81].float().flatten(1))[:, :4])

    res = {'depths': {}}
    prev = {}
    for k in range(1, H + 1):
        ii = torch.where(test & alive5[:, k - 1])[0]
        Y = fut5[ii, :, k - 1].float()                 # [n,5,81,192]
        f_ = SB.AllProbes.facts(meta['future_visible'][ii, :, k - 1])
        n = len(ii)
        stat_tok = {'mean5': Y.mean(1), 'median5': Y.median(1).values, 'root': root[ii].float()}
        d = {s: dec(t) for s, t in stat_tok.items()}
        real = [dec(Y[:, j]) for j in range(5)]
        # previous-step HUD readouts (statistics at k-1 on the SAME roots; root readout at k = 1)
        if k == 1:
            hprev = {s: dec(root[ii].float())[2] for s in STATS[:3]}
            hprev_real = [hprev['root']] * 5
        else:
            Yp = fut5[ii, :, k - 2].float()
            hprev = {'mean5': dec(Yp.mean(1))[2], 'median5': dec(Yp.median(1).values)[2], 'root': dec(root[ii].float())[2]}
            hprev_real = [dec(Yp[:, j])[2] for j in range(5)]
        row = {'roots': n, 'census_simulator': {m: round(float(f_['mobs'][..., j].float().sum(-1).mean()), 4) for j, m in enumerate(MOBS)}}
        cnt = f_['mobs'].sum(1)                                                                       # [n,62,4] futures with the mob
        agree = (f_['tile'] == f_['tile'][:, :1]).all(1)                                              # [n,62]
        floor = float(((Y - Y.mean(1, keepdim=True)) ** 2).sum((1, 2, 3)).mean() / 4)
        row['floor_L2'] = round(floor, 3)
        for s in STATS:
            outs = real if s == 'real' else [d[s]]
            hp = hprev_real if s == 'real' else [hprev[s]]
            r = {}
            r['census'] = {m: round(float(np.mean([float(o[1][..., j].float().sum(-1).mean()) for o in outs])), 4) for j, m in enumerate(MOBS)}
            mc = {}
            for j, m in enumerate(MOBS):
                mc[m] = {str(c): round(float(np.mean([float(o[1][..., j][cnt[..., j] == c].float().mean()) for o in outs])), 4)
                         for c in range(1, 6) if (cnt[..., j] == c).any()}
            r['mob_cells_decoded_present_by_futures_with_mob'] = mc
            r['tile_acc_agree'] = round(float(np.mean([float((o[0] == f_['tile'][:, 0])[agree].float().mean()) for o in outs])), 4)
            r['tile_acc_disagree'] = round(float(np.mean([float((o[0] == f_['tile'][:, 0])[~agree].float().mean()) for o in outs])), 4) if (~agree).any() else None
            r['hud_event_rate'] = {h: round(float(np.mean([float((o[2][:, j] < p_[:, j] - 0.5).float().mean()) for o, p_ in zip(outs, hp)])), 5)
                                   for j, h in enumerate(HUDN)}
            if s != 'real':
                t = stat_tok[s]
                r['L2_to_futures'] = round(float(((t[:, None] - Y) ** 2).sum((2, 3)).mean()), 3)
                r['L1_to_futures'] = round(float((t[:, None] - Y).abs().sum((2, 3)).mean()), 3)
            row[s] = r
        row['cells_with_mob_by_futures'] = {m: {str(c): int((cnt[..., j] == c).sum()) for c in range(1, 6)} for j, m in enumerate(MOBS)}
        row['tile_disagree_share'] = round(float((~agree).float().mean()), 4)
        res['depths'][str(k)] = row
        print(json.dumps({k: {s: {'census': row[s]['census'], 'hud': row[s]['hud_event_rate']} for s in STATS} | {'sim': row['census_simulator']}}), flush=True)
    # world vs statistic at depth 1
    ii = torch.where(test & alive5[:, 0])[0]
    Y = fut5[ii, :, 0].float()
    mean5, med5 = Y.mean(1), Y.median(1).values
    sep = (mean5 - med5).abs() > 0.1
    res['world_vs_statistic_depth1'] = {'elements_mean_median_differ_share': round(float(sep.float().mean()), 4)}
    for v in sys.argv[1:]:
        p = torch.from_numpy(np.ascontiguousarray(np.load(SBV / v / 'tf.npy', mmap_mode='r')[ii.numpy(), 0])).float()
        closer = ((p - med5).abs() < (p - mean5).abs())[sep]
        res['world_vs_statistic_depth1'][v] = {'closer_to_median_share': round(float(closer.float().mean()), 4),
                                               'L2_to_mean5': round(float(((p - mean5) ** 2).sum((1, 2)).mean()), 3),
                                               'L2_to_median5': round(float(((p - med5) ** 2).sum((1, 2)).mean()), 3),
                                               'L2_to_future0': round(float(((p - Y[:, 0]) ** 2).sum((1, 2)).mean()), 3)}
    print(json.dumps(res['world_vs_statistic_depth1']), flush=True)
    (OUT / 'statoracle.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
