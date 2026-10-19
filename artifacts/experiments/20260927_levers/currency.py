"""10-10 PROGRAM (P-A / loss currency, prediction side): does the per-element token loss measure what matters? CPU, on the exports.
For each world's teacher-forced predictions (tf, every depth; TEST roots; alive sample 0), every predicted map cell (62, player cell
excluded) is decoded by scoreboard.AllProbes (fitted on true TRAIN tokens): tile class (argmax) and mob presence per type (calibrated
thresholds). The cell's FACTS are correct when the tile class and all four mob flags equal the simulator's.
Reported per world:
  shares of the L2 (sum e^2) and L1 (sum |e|) error on fact-correct vs fact-wrong cells, and per-cell mean error of each
  AUC of the per-cell error (L2, L1) for separating fact-wrong from fact-correct cells (1 = the loss's currency ranks every wrong
  fact above every right one; 0.5 = it does not distinguish them)
  the same split for the true tokens' own frame-to-frame change (what a copy predictor pays) as a reference
  HUD: the 18 HUD tokens' error on steps where the decoded health / food / drink / energy (rounded) are all correct vs any wrong
Usage: currency.py <variant> ...  -> artifacts/eda/scoreboard_v1/<variant>/currency.json
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
H = 16


def main():
    import teval as T
    import scoreboard as SB
    from compound import auc
    torch.set_num_threads(6)
    meta, train_roots, train_seeds = T.split()
    test = ~train_roots
    cache = T.build_cache('raw', torch.device('cpu'))
    probes = SB.AllProbes(T, cache, meta, train_roots, train_seeds)
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()
    fut = cache['fut']
    idx = torch.where(test)[0]
    facts = SB.AllProbes.facts(meta['future_visible'][idx, 0].float())                               # [n,16,...]
    thr = torch.tensor(probes.thr)

    def decode(tok):                                                                                  # [m,81,192] -> tile [m,62], mobs [m,62,4]
        cells = tok[:, SB.MAPC].float().flatten(0, 1)
        return probes.tile(cells).view(len(tok), 62, 17).argmax(-1), (probes.mob(cells) > thr).view(len(tok), 62, 4)

    def hud_vals(tok):
        return probes.hud(tok[:, 63:81].float().flatten(1))[:, :4].round()

    for name in sys.argv[1:]:
        d = OUT / name
        tf = torch.from_numpy(np.load(d / 'tf.npy', mmap_mode='r'))
        e2s, e1s, wrongs, he, hw = [], [], [], [], []
        for k in range(H):
            m = alive[idx, k]
            ii = idx[m]
            p = tf[ii, k].float(); y = fut[ii, k].float()
            tile, mobs = decode(p)
            wrong = (tile != facts['tile'][m, k]) | (mobs != facts['mobs'][m, k]).any(-1)              # [n,62]
            e = p[:, SB.MAPC] - y[:, SB.MAPC]
            e2s.append((e ** 2).sum(-1).flatten()); e1s.append(e.abs().sum(-1).flatten()); wrongs.append(wrong.flatten())
            hv = hud_vals(p); ht = facts['hud'][m, k][:, :4].round()
            he.append(((p[:, 63:81] - y[:, 63:81]) ** 2).sum((-1, -2))); hw.append((hv != ht).any(-1))
        e2, e1, wr = torch.cat(e2s), torch.cat(e1s), torch.cat(wrongs)
        he, hw = torch.cat(he), torch.cat(hw)
        res = {'variant': name, 'cells': int(len(wr)), 'fact_wrong_share': round(float(wr.float().mean()), 4),
               'L2_share_on_fact_correct': round(float(e2[~wr].sum() / e2.sum()), 4),
               'L1_share_on_fact_correct': round(float(e1[~wr].sum() / e1.sum()), 4),
               'mean_L2_fact_correct': round(float(e2[~wr].mean()), 3), 'mean_L2_fact_wrong': round(float(e2[wr].mean()), 3),
               'mean_L1_fact_correct': round(float(e1[~wr].mean()), 3), 'mean_L1_fact_wrong': round(float(e1[wr].mean()), 3),
               'AUC_L2_separates_wrong_facts': round(auc(e2, wr), 4), 'AUC_L1_separates_wrong_facts': round(auc(e1, wr), 4),
               'hud_steps': int(len(hw)), 'hud_value_wrong_share': round(float(hw.float().mean()), 4),
               'hud_L2_share_on_correct_values': round(float(he[~hw].sum() / he.sum()), 4),
               'hud_AUC_L2_separates_wrong_values': round(auc(he, hw), 4)}
        (d / 'currency.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
