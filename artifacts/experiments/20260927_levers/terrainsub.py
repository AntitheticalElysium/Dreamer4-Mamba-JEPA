"""10-10 (does a generator's noise need to reach entering terrain?). CPU, on the exports.
properscore C: at depth 1, 61-71% of every world's gap to its ideal is on the edge ring, where terrain enters the view. The ideal
knows that terrain (the five futures share the map); no world does. How much of the gap could a world close WITHOUT knowing the
map, by predicting (a point) or sampling (a sampler) plausible terrain from what it sees?
Targets: roots whose five futures are alive at depth 1 and whose factual future (0) scrolls there. The scroll comes from the true
tile maps, with ties broken by the token estimate, as budget.categories. The entering line is row 0 / 6 or column 0 / 8 of the
successor. The world's prediction on that line is replaced by:
  ideal   the other futures' tokens on that line (leave-one-out; for a point forecast, their mean): knows the map
  nn      terrain from DONOR transitions of other episodes with the same scroll direction, chosen by the root's edge line (the
          visible line the terrain enters next to). Taken: the K = 4 donors whose edge line is nearest in token space (L2) to
          the target root's (one per sample; for a point forecast, their mean): plausible terrain given what the world sees
  rand    4 random donors of the same direction from other episodes: the terrain prior, ignoring the view
  tile    CONTROL (added 18:5x, after nn and rand both scored worse than the worlds): the 4 donors whose entering line's
          TRUE tiles agree best with the target's true entering tiles (an oracle on the facts; ties by the nn distance). If even
          these do not close the gap, a donor token carries more than its terrain (its own frame's context, light and static;
          encoder_why), and the substitution cannot measure terrain sampling.
Donors: the 17 one-step successors (env key 0) of every root (1,002 x 17) that scroll (token estimate) and are alive.
Scored: whole-frame fair energy score (properscore.es) vs the five futures, scored on the held-out future j and averaged over j.
Predictors: M16, D continued, D scratch, E22MS Δ zeroed and E22MS (4 samples), at depth 1 (gen.npy). Reported:
  - ES as is / ideal / nn / rand on the target roots;
  - the share of the entering-line gap (as is - ideal) closed by nn and by rand (root night / day too);
  - donor quality: tile agreement of the nn / rand donors' entering line with the target's true entering tiles (simulator),
    and between real futures 0 and j on that line.
Usage: terrainsub.py  -> artifacts/eda/health_chain_v1/terrainsub.json
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
LINE = {1: list(range(9)), 2: [54 + c for c in range(9)], 3: [r * 9 for r in range(7)], 4: [r * 9 + 8 for r in range(7)]}
DET = ('M16', 'DC', 'DS', 'E22MS_zeroed')
SAMPLES = tuple(f'E22MS_sample{s}' for s in range(4))
K = 4


def main():
    import teval as T
    import stochdiag as SD
    import encoder_diag as ED
    import budget as BU
    import properscore as PS
    from scroll import estimate
    torch.set_num_threads(8)
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5, one4 = SD.token_cache(torch.device('cpu'))
    R = len(meta['seed'])
    seeds = meta['seed']
    root = cache['ctx'][:, -1]
    alive5 = ~meta['future_dead'].cumsum(2).bool().any(1)
    night = meta['root_visible'][:, 1521].float() < 0.5
    gens = {v: np.load(SBV / v / 'gen.npy', mmap_mode='r') for v in DET + SAMPLES}
    # donors
    don = {s: {'E': [], 'A': [], 'seed': [], 'tiles': []} for s in LINE}
    for i in range(0, R, 32):
        b = min(32, R - i)
        x0, x1 = root[i:i + b].float(), one4[i:i + b, 0].float()                                     # [b,81,192], [b,17,81,192]
        sh = estimate(x0[:, None].expand_as(x1), x1)                                                  # [b,17]
        tl = ED.frame_state(meta['onestep_visible'][i:i + b, 0].flatten(0, 1))[0].view(b, 17, 63)
        for r_, a_ in zip(*torch.where((sh != 0) & ~meta['onestep_dead'][i:i + b, 0])):
            s = int(sh[r_, a_]); ln = LINE[s]
            don[s]['E'].append(x1[r_, a_, ln].half()); don[s]['A'].append(x0[r_, ln].half())
            don[s]['seed'].append(int(seeds[i + r_])); don[s]['tiles'].append(tl[r_, a_, ln])
    for s in don:
        don[s] = {k: (torch.stack(v) if k != 'seed' else torch.tensor(v)) for k, v in don[s].items()}
    res = {'donors': {str(s): int(len(don[s]['seed'])) for s in don}}
    # targets: factual scroll at depth 1 from the true tile maps (budget.categories' rule)
    t0, t1 = ED.frame_state(meta['root_visible'])[0], ED.frame_state(meta['future_visible'][:, 0, 0])[0]
    frac = []
    for dr, dc in BU.SHIFTS:
        src = ED.neighbour(t0, dr, dc, -1); ov = src >= 0
        frac.append(((src != t1) & ov).flatten(1).sum(1).float() / ov.flatten(1).sum(1).float())
    est = torch.cat([estimate(root[i:i + 64].float(), fut5[i:i + 64, 0, 0].float()) for i in range(0, R, 64)])
    sh0 = (torch.stack(frac, 1) + 1e-3 * (torch.arange(5)[None] != est[:, None]).float()).argmin(1)
    tl_true = ED.frame_state(meta['future_visible'][:, :, 0].flatten(0, 1))[0].view(R, 5, 63)
    gen_ = torch.Generator().manual_seed(20261010)
    acc = {}
    qual = {'nn': [], 'rand': [], 'real_pair': []}
    for s, ln in LINE.items():
        tg = torch.where(alive5[:, 0] & (sh0 == s))[0]
        if not len(tg):
            continue
        D = don[s]
        dmat = torch.cdist(root[tg][:, ln].float().flatten(1), D['A'].float().flatten(1))            # [n_t, n_d]
        dmat[seeds[tg][:, None] == D['seed'][None]] = float('inf')
        nn = dmat.topk(K, largest=False).indices                                                     # [n_t, K]
        agree = (D['tiles'][None] == tl_true[tg, 0][:, ln][:, None]).float().mean(-1)                # [n_t, n_d]
        agree[seeds[tg][:, None] == D['seed'][None]] = -1.0
        tm = (agree - 1e-6 * dmat.clamp(max=1e6) / dmat[dmat < float('inf')].max()).topk(K).indices
        rd = torch.stack([torch.where(D['seed'] != seeds[t])[0][torch.randperm(int((D['seed'] != seeds[t]).sum()), generator=gen_)[:K]] for t in tg])
        truth_tiles = tl_true[tg, 0][:, ln]
        qual['nn'].append((D['tiles'][nn] == truth_tiles[:, None]).float().mean((1, 2)))
        qual['rand'].append((D['tiles'][rd] == truth_tiles[:, None]).float().mean((1, 2)))
        qual.setdefault('tile', []).append((D['tiles'][tm] == truth_tiles[:, None]).float().mean((1, 2)))
        qual['real_pair'].append((tl_true[tg, 1:][:, :, ln] == truth_tiles[:, None]).float().mean((1, 2)))
        for c0 in range(0, len(tg), 16):
            ii = tg[c0:c0 + 16]
            Y = fut5[ii, :, 0].float()
            subs = {'nn': D['E'][nn[c0:c0 + 16]].float(), 'rand': D['E'][rd[c0:c0 + 16]].float(), 'tile': D['E'][tm[c0:c0 + 16]].float()}
            P = {v: torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), 0])).float()[:, None] for v in DET}
            P['E22MS'] = torch.stack([torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), 0])).float() for v in SAMPLES], 1)
            for p_, X in P.items():
                M = X.shape[1]
                out = {q: [] for q in ('as_is', 'ideal', 'nn', 'rand', 'tile')}
                for j in range(5):
                    yj = Y[:, j:j + 1]
                    oth = Y[:, [q for q in range(5) if q != j]][:, :, ln]
                    out['as_is'].append(PS.es(X, yj))
                    for q, rep in (('ideal', oth), ('nn', subs['nn']), ('rand', subs['rand']), ('tile', subs['tile'])):
                        Xs = X.clone()
                        Xs[:, :, ln] = rep.mean(1, keepdim=True) if M == 1 else rep[:, :M]
                        out[q].append(PS.es(Xs, yj))
                for q, v in out.items():
                    a = acc.setdefault(p_, {}).setdefault(q, [])
                    a.append(torch.stack(v, 1).mean(1))
        print(json.dumps({'direction': s, 'targets': int(len(tg)), 'donors': int(len(D['seed']))}), flush=True)
    tg_all = torch.cat([torch.where(alive5[:, 0] & (sh0 == s))[0] for s in LINE])
    nt = night[tg_all]
    res['targets'] = int(len(tg_all)); res['targets_night'] = int(nt.sum())
    res['tile_agreement_entering_line'] = {k: round(float(torch.cat(v).mean()), 4) for k, v in qual.items()}
    res['predictors'] = {}
    for p_, d in acc.items():
        v = {q: torch.cat(x) for q, x in d.items()}
        row = {}
        for sub, m in (('all', torch.ones_like(nt)), ('root_night', nt), ('root_day', ~nt)):
            e = {q: float(v[q][m].mean()) for q in v}
            gap = e['as_is'] - e['ideal']
            row[sub] = {**{q: round(x, 3) for q, x in e.items()},
                        'nn_closes': round((e['as_is'] - e['nn']) / gap, 4) if gap else None,
                        'rand_closes': round((e['as_is'] - e['rand']) / gap, 4) if gap else None,
                        'tile_closes': round((e['as_is'] - e['tile']) / gap, 4) if gap else None}
        res['predictors'][p_] = row
    (OUT / 'terrainsub.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res), flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
