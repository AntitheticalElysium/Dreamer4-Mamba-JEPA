"""Uncertainty of the h63 ingredient AUCs (2026-10-08 night): how big is a resolved difference between worlds?
For each world (health_chain.py --ext file) and ingredient (since = 6 [windows >= 7], since 1..4, approach adjacency on scroll steps
with a zombie within 2): health_evidence's probe (balanced MLP, early stop on the inner split) trained with 5 seeds on h63; reported
  seed_mean / seed_std   held AUC across the 5 probe seeds (probe-training noise)
  ci95                   episode-clustered bootstrap (2,000 draws over held episodes, seed 20261008) of the seed-averaged probe's AUC
and for each requested pair A:B the paired bootstrap of AUC(B) - AUC(A) on the same resampled held episodes (95% interval; resolved
when it excludes 0). -> artifacts/eda/health_chain_v1/ci_<tag>.json
Usage: health_ci.py <tag> <name=ext.pt> ... [--pairs A:B ...]
"""
import hashlib
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'


def main():
    import health_evidence as HE
    args = sys.argv[1:]
    tag = args[0]
    pairs = [p.split(':') for p in args[args.index('--pairs') + 1:]] if '--pairs' in args else []
    worlds = dict(a.split('=', 1) for a in args[1:args.index('--pairs') if '--pairs' in args else len(args)])
    sub = torch.load(OUT / 'subset.pt', weights_only=False)
    st = torch.load(OUT / 'strata.pt', weights_only=False)
    N = len(sub['classes']); allm = torch.ones(N, dtype=torch.bool)
    fit = sub['fit']
    inner = fit & torch.tensor([int(hashlib.sha256((e + '/inner').encode()).hexdigest(), 16) % 4 == 0 for e, _ in sub['ids']])
    held = ~fit
    since, scroll, near2 = st['since'], sub['scroll'] != 0, sub['zombie_distance'] <= 2
    episodes = [e for e, _ in sub['ids']]
    held_idx = torch.where(held)[0]
    eps = sorted({episodes[i] for i in held_idx.tolist()})
    ep_of = torch.tensor([eps.index(episodes[i]) for i in held_idx.tolist()])
    members = [held_idx[ep_of == k] for k in range(len(eps))]
    g = torch.Generator().manual_seed(20261008)
    draws = [torch.cat([members[k] for k in torch.randint(len(eps), (len(eps),), generator=g).tolist()]) for _ in range(2000)]
    targets = {'since_eq6': (since == 6, allm), 'since_1to4': ((since >= 1) & (since <= 4), allm),
               'adj_scroll_zombie2': (st['adj_post'], scroll & near2)}
    scores, res = {}, {'worlds': worlds, 'held_episodes': len(eps)}
    for name, path in worlds.items():
        r = torch.load(path, weights_only=False)
        h63 = r['h_ext'][:, 0].float()
        res[name] = {'window': r['window']}
        for k, (y, scope) in targets.items():
            if k == 'since_eq6' and r['window'] < 7:
                continue
            preds = [HE.weighted_probe(h63, y, torch.ones(N), allm & fit & ~inner, allm & inner, seed=s, steps=2000) for s in range(5)]
            aucs = torch.tensor([HE.auc(p[held & scope], y[held & scope]) for p in preds])
            avg = torch.stack(preds).mean(0)
            scores[(name, k)] = avg
            boot = torch.tensor([HE.auc(avg[d][scope[d]], y[d][scope[d]]) for d in draws])
            res[name][k] = {'seed_mean': round(float(aucs.mean()), 4), 'seed_std': round(float(aucs.std()), 4),
                            'ensemble_auc': round(HE.auc(avg[held & scope], y[held & scope]), 4),
                            'ci95': [round(float(boot.quantile(0.025)), 4), round(float(boot.quantile(0.975)), 4)]}
        print(json.dumps({name: res[name]}), flush=True)
    res['pairs'] = {}
    for a, b in pairs:
        for k, (y, scope) in targets.items():
            if (a, k) not in scores or (b, k) not in scores:
                continue
            sa, sb = scores[(a, k)], scores[(b, k)]
            diff = torch.tensor([HE.auc(sb[d][scope[d]], y[d][scope[d]]) - HE.auc(sa[d][scope[d]], y[d][scope[d]]) for d in draws])
            lo, hi = float(diff.quantile(0.025)), float(diff.quantile(0.975))
            res['pairs'][f'{a}->{b}:{k}'] = {'diff': round(float(diff.mean()), 4), 'ci95': [round(lo, 4), round(hi, 4)],
                                            'resolved': lo > 0 or hi < 0}
            print(json.dumps({f'{a}->{b}:{k}': res['pairs'][f'{a}->{b}:{k}']}), flush=True)
    (OUT / f'ci_{tag}.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
