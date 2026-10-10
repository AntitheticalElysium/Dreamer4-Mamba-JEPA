"""10-10 (Codex's proposal: a noise-conditioned generator trained by an energy score on completed successors). CPU, on the exports.
Two premises of that proposal, measured in OUR token space before anything is trained:
A. PROPER SCORES. The scoreboard's per-class energy scores use outcome-dependent masks (mob / entering / static are selected from
   the realized futures; stochdiag.classes) and so are not proper (Codex, 10-10; the forecaster's dilemma). Here, only fixed
   regions are used: the whole frame (81 tokens), the map (63) and the HUD (18). The fair energy score ES(F, y) =
   mean_i |X_i - y| - 1 / (2 M (M - 1)) sum_{i != l} |X_i - X_l| (unbiased for any M; a point forecast: |x - y|) is used, against
   each of the five real futures (same actions, different env keys) of a root, at self-fed depths 1, 2, 4, 8, 16 (all five
   futures alive). Predictors:
     real_LOO     the IDEAL sampler: 4 real futures as samples, scored on the held-out fifth, averaged over the five
                  (cross-validated)
     mean_LOO / median_LOO   the per-element mean / median of those 4 (the L2 / L1 optima of a point forecast)
     root         copy the root frame
     M16, DC, DS, E22MS_zeroed   deterministic worlds (gen.npy)
     E22MS        the prior's 4 samples (E22MS_sample0..3, gen.npy), M = 4 as real_LOO
   Reported: mean ES per region x depth (all roots; root-night / root-day), and paired episode-clustered bootstrap intervals (2,000
   draws, seed 20261010) for the key differences.
B. THE CURRENCY OF EACH LOSS AT THE OUTPUT (depth 1, one observed successor per context, as in training). Taken at the same
   predictions: the gradient each loss sends to the predicted tokens --
     l1 sign(x - y); mse (x - y); es_whole: u(X_i - y) - 1 / (M - 1) sum_{l != i} u(X_i - X_l), u(v) = v / |v| over the whole
     frame (a point forecast: u(x - y)); es_regions: the same with u taken within the map and the HUD separately (a sum of
     proper scores on fixed regions, Pic et al. 2025)
   -- split by budget.CATS (root -> factual successor; entering split into seen / never seen in the 4 context frames, as budget.py).
   Per category: share of the gradient energy (sum over samples, tokens and dims of g^2; mean over the five truths), and the
   SYSTEMATIC fraction |mean_j g|^2 / mean_j |g|^2 over the five truths j (the part that does not depend on which future
   happened; pure outcome noise gives 1/5 = 0.2). real_LOO's samples depend on j, so it gets energy shares only: the allocation of
   the score's gradient AT its optimum, which is pure outcome noise.
   This is model-free (output space); P-C (gradanat) measures the backbone side.
C. WHERE THE GAP IS, by substitution (`--subst`, added 18:0x; runs only this part). The fixed regions are edge (the 30 map cells on
   rows 0 / 6 or columns 0 / 8: where entering terrain appears), interior (the other 32 map cells, not the player's), player
   (token 31) and hud (18 tokens). For each predictor, each region in turn is replaced by the ideal content of its class: the
   ideal point (mean of the 4 other futures) for a point forecast; the ideal sampler (the 4 other futures, sample i <- future
   i) for E22MS. The whole-frame ES is then rescored on the held-out future, averaged over the five. Reported: ES as is, ES with
   each region replaced, and ES with everything replaced (= the ideal of its class, a check). The drop per region is the part
   of the predictor's gap to its ideal that lives in that region (not additive across regions: ES is a norm).
Usage: properscore.py [--subst]  -> artifacts/eda/health_chain_v1/properscore[_subst].json
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
DEPTHS = (1, 2, 4, 8, 16)
REGIONS = {'whole': list(range(81)), 'map': list(range(63)), 'hud': list(range(63, 81))}
DET = ('M16', 'DC', 'DS', 'E22MS_zeroed')
SAMPLES = tuple(f'E22MS_sample{s}' for s in range(4))
PRED = ('real_LOO', 'mean_LOO', 'median_LOO', 'root') + DET + ('E22MS',)
LOSSES = ('l1', 'mse', 'es_whole', 'es_regions')


def dist(A, B):
    """A [b,m,T,D], B [b,n,T,D] -> Euclidean distances [b,m,n]"""
    return torch.stack([((A - B[:, j:j + 1]) ** 2).sum((-1, -2)).sqrt() for j in range(B.shape[1])], 2)


def es(X, Y):
    """fair energy score, mean over the truths: X [b,M,T,D] samples (M = 1: a point), Y [b,S,T,D] -> [b]"""
    s = dist(X, Y).mean((1, 2))
    M = X.shape[1]
    if M > 1:
        dxx = dist(X, X)
        s = s - 0.5 * dxx.sum((1, 2)) / (M * (M - 1))
    return s


def unit(v, blocks):
    """v [..., 81, 192] -> v / |v| within each token block"""
    out = torch.empty_like(v)
    for idx in blocks:
        out[..., idx, :] = v[..., idx, :] / (v[..., idx, :] ** 2).sum((-1, -2), keepdim=True).sqrt().clamp(min=1e-8)
    return out


def grad(loss, X, y):
    """output gradient of `loss` for samples X [b,M,81,192] given ONE truth y [b,81,192] -> [b,M,81,192]"""
    d = X - y[:, None]
    if loss == 'l1':
        return d.sign()
    if loss == 'mse':
        return d
    blocks = [REGIONS['whole']] if loss == 'es_whole' else [REGIONS['map'], REGIONS['hud']]
    g = unit(d, blocks)
    M = X.shape[1]
    if M > 1:
        rep = torch.stack([sum(unit(X[:, i] - X[:, l], blocks) for l in range(M) if l != i) for i in range(M)], 1)
        g = g - rep / (M - 1)
    return g


def main():
    import teval as T
    import stochdiag as SD
    import budget as BU
    from scroll import estimate
    torch.set_num_threads(8)
    meta, _, _ = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    fut5 = SD.token_cache(torch.device('cpu'))[0]                                                    # [R,5,16,81,192]
    R = len(meta['seed'])
    alive5 = ~meta['future_dead'].cumsum(2).bool().any(1)                                             # [R,16]
    night = meta['root_visible'][:, 1521].float() < 0.5
    root = cache['ctx'][:, -1]
    gens = {v: np.load(SBV / v / 'gen.npy', mmap_mode='r') for v in DET + SAMPLES}
    if '--subst' in sys.argv:
        edge = [r * 9 + c for r in range(7) for c in range(9) if r in (0, 6) or c in (0, 8)]
        regs = {'edge': edge, 'interior': [q for q in range(63) if q not in edge and q != 31], 'player': [31], 'hud': list(range(63, 81)),
                'all': list(range(81))}
        acc = {}
        for k in (1, 4, 16):
            for i in range(0, R, 16):
                ii = torch.arange(i, min(i + 16, R))
                ii = ii[alive5[ii, k - 1]]
                if not len(ii):
                    continue
                Y = fut5[ii, :, k - 1].float()
                P = {v: torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), k - 1])).float()[:, None] for v in DET}
                P['E22MS'] = torch.stack([torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), k - 1])).float() for v in SAMPLES], 1)
                for p_, X in P.items():
                    rows_ = {'as_is': [], **{r: [] for r in regs}}
                    for j in range(5):
                        oth = Y[:, [q for q in range(5) if q != j]]
                        ideal = oth.mean(1, keepdim=True) if X.shape[1] == 1 else oth
                        yj = Y[:, j:j + 1]
                        rows_['as_is'].append(es(X, yj))
                        for rn, idx in regs.items():
                            Xs = X.clone(); Xs[:, :, idx] = ideal[:, :, idx]
                            rows_[rn].append(es(Xs, yj))
                    for rn, v in rows_.items():
                        a = acc.setdefault((p_, k, rn), [0.0, 0])
                        a[0] += float(torch.stack(v, 1).mean(1).sum()); a[1] += len(ii)
            print(json.dumps({k: {p_: round(acc[(p_, k, 'as_is')][0] / acc[(p_, k, 'as_is')][1], 3) for p_ in DET + ('E22MS',)}}), flush=True)
        res = {}
        for (p_, k, rn), (sm, n) in acc.items():
            res.setdefault(p_, {}).setdefault(str(k), {})[rn] = round(sm / n, 3)
        for p_ in res:
            for k in res[p_]:
                d = res[p_][k]
                gap = d['as_is'] - d['all']
                d['gap_to_ideal'] = round(gap, 3)
                d['share_of_gap_removed'] = {rn: round((d['as_is'] - d[rn]) / gap, 4) for rn in regs if rn != 'all'}
        (OUT / 'properscore_subst.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res), flush=True)
        return

    # A. proper scores
    rows = {p: {r: torch.full((R, len(DEPTHS)), float('nan')) for r in REGIONS} for p in PRED}
    for di, k in enumerate(DEPTHS):
        for i in range(0, R, 16):
            ii = torch.arange(i, min(i + 16, R))
            m = alive5[ii, k - 1]
            if not m.any():
                continue
            ii = ii[m]
            Y = fut5[ii, :, k - 1].float()
            P = {'root': root[ii].float()[:, None]}
            for v in DET:
                P[v] = torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), k - 1])).float()[:, None]
            P['E22MS'] = torch.stack([torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), k - 1])).float() for v in SAMPLES], 1)
            for rn, idx in REGIONS.items():
                Yr = Y[:, :, idx]
                loo = {'real_LOO': [], 'mean_LOO': [], 'median_LOO': []}
                for j in range(5):
                    oth = Yr[:, [q for q in range(5) if q != j]]
                    yj = Yr[:, j:j + 1]
                    loo['real_LOO'].append(es(oth, yj))
                    loo['mean_LOO'].append(es(oth.mean(1, keepdim=True), yj))
                    loo['median_LOO'].append(es(oth.sort(1).values[:, 1:3].mean(1, keepdim=True), yj))
                for p_, v in loo.items():
                    rows[p_][rn][ii, di] = torch.stack(v, 1).mean(1)
                for p_, X in P.items():
                    rows[p_][rn][ii, di] = es(X[:, :, idx], Yr)
        print(json.dumps({k: {p: round(float(rows[p]['whole'][:, di].nanmean()), 3) for p in PRED}}), flush=True)
    res = {'A_proper_scores': {}, 'A_intervals': {}}
    for rn in REGIONS:
        res['A_proper_scores'][rn] = {str(k): {p: {'all': round(float(rows[p][rn][:, di].nanmean()), 3),
                                                   'root_night': round(float(rows[p][rn][night, di].nanmean()), 3),
                                                   'root_day': round(float(rows[p][rn][~night, di].nanmean()), 3)} for p in PRED}
                                      for di, k in enumerate(DEPTHS)}
    seeds = meta['seed']
    groups = [torch.where(seeds == s)[0] for s in seeds.unique()]
    gen_ = torch.Generator().manual_seed(20261010)
    draws = [torch.cat([groups[q] for q in torch.randint(len(groups), (len(groups),), generator=gen_).tolist()]) for _ in range(2000)]
    pairs = [('E22MS', 'M16'), ('E22MS', 'DS'), ('E22MS', 'E22MS_zeroed'), ('E22MS', 'real_LOO'), ('DS', 'M16'), ('M16', 'median_LOO'),
             ('DS', 'mean_LOO'), ('mean_LOO', 'real_LOO'), ('median_LOO', 'real_LOO')]
    for rn in REGIONS:
        for di, k in enumerate(DEPTHS):
            for a, b in pairs:
                d = rows[a][rn][:, di] - rows[b][rn][:, di]
                ok = ~d.isnan()
                d0 = torch.where(ok, d, torch.zeros(()))
                boot = torch.stack([d0[q].sum() / ok[q].float().sum().clamp(min=1) for q in draws])
                lo, hi = float(boot.quantile(0.025)), float(boot.quantile(0.975))
                res['A_intervals'].setdefault(rn, {}).setdefault(str(k), {})[f'{a}-{b}'] = \
                    [round(float(d0.sum() / ok.float().sum()), 4), round(lo, 4), round(hi, 4), lo > 0 or hi < 0]
    print(json.dumps({'intervals_whole': res['A_intervals']['whole']}), flush=True)

    # B. the currency of each loss at the output (depth 1)
    rootv = meta['root_visible']
    ctx = cache['ctx']
    dctx = torch.stack([torch.cat([estimate(ctx[i:i + 64, t].float(), ctx[i:i + 64, t + 1].float()) for i in range(0, R, 64)]) for t in range(3)], 1)
    dvec = torch.tensor(BU.SHIFTS)[dctx]
    offs = torch.zeros(R, 4, 2, dtype=torch.long)
    for t in (2, 1, 0):
        offs[:, t] = offs[:, t + 1] - dvec[:, t]
    nc = len(BU.CATS)
    tot = {(p, l): torch.zeros(nc, dtype=torch.float64) for p in PRED for l in LOSSES}
    sysm = {(p, l): torch.zeros(nc, dtype=torch.float64) for p in PRED for l in LOSSES}
    for i in range(0, R, 8):
        ii = torch.arange(i, min(i + 8, R))
        ii = ii[alive5[ii, 0]]
        if not len(ii):
            continue
        Y = fut5[ii, :, 0].float()                                                                    # [b,5,81,192]
        cat = BU.categories(rootv[ii], meta['future_visible'][ii, 0, 0], root[ii].float(), Y[:, 0], offs[ii])[0]   # [b,81]
        oh = torch.nn.functional.one_hot(cat, nc).double()                                            # [b,81,nc]
        P = {'root': root[ii].float()[:, None]}
        for v in DET:
            P[v] = torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), 0])).float()[:, None]
        P['E22MS'] = torch.stack([torch.from_numpy(np.ascontiguousarray(gens[v][ii.numpy(), 0])).float() for v in SAMPLES], 1)
        for l in LOSSES:
            for p_ in PRED:
                gs = []
                for j in range(5):
                    if p_ == 'real_LOO':
                        X = Y[:, [q for q in range(5) if q != j]]
                    elif p_ in ('mean_LOO', 'median_LOO'):
                        oth = Y[:, [q for q in range(5) if q != j]]
                        X = oth.mean(1, keepdim=True) if p_ == 'mean_LOO' else oth.sort(1).values[:, 1:3].mean(1, keepdim=True)
                    else:
                        X = P[p_]
                    gs.append(grad(l, X, Y[:, j]))
                G = torch.stack(gs, 1)                                                                # [b,5,M,81,192]
                e_tok = (G.double() ** 2).sum((2, -1)).mean(1)                                       # [b,81]
                tot[(p_, l)] += (e_tok[..., None] * oh).sum((0, 1))
                if p_ not in ('real_LOO', 'mean_LOO', 'median_LOO'):
                    s_tok = (G.double().mean(1) ** 2).sum((1, -1))                                   # [b,81]
                    sysm[(p_, l)] += (s_tok[..., None] * oh).sum((0, 1))
        if i % 160 == 0:
            print(f'B roots {i + len(ii)}/{R}', flush=True)
    res['B_output_currency_depth1'] = {}
    for l in LOSSES:
        for p_ in PRED:
            t_, s_ = tot[(p_, l)], sysm[(p_, l)]
            row = {'energy_share': {c: round(float(t_[ci] / t_.sum()), 4) for ci, c in enumerate(BU.CATS) if t_[ci] > 0}}
            if s_.sum() > 0:
                row['systematic_fraction'] = {c: round(float(s_[ci] / t_[ci]), 4) for ci, c in enumerate(BU.CATS) if t_[ci] > 0}
                row['systematic_fraction_total'] = round(float(s_.sum() / t_.sum()), 4)
                row['systematic_share'] = {c: round(float(s_[ci] / s_.sum()), 4) for ci, c in enumerate(BU.CATS) if t_[ci] > 0}
            res['B_output_currency_depth1'].setdefault(l, {})[p_] = row
    (OUT / 'properscore.json').write_text(json.dumps(res, indent=1))
    torch.save(rows, OUT / 'properscore_rows.pt')
    print(json.dumps(res['B_output_currency_depth1']['es_whole']['E22MS']), flush=True)
    print('wrote', OUT / 'properscore.json', flush=True)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
