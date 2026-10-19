"""10-10 PROGRAM Part I: one table across scoreboard variants, with paired episode-clustered bootstrap intervals (143 walk seeds, 2,000
draws, seed 20261010) for the per-root quantities saved in rows.pt: self-fed error / V at depths 1, 4, 8, 16 (alive sample 0), energy
score per class at depths 1, 4, 8, 16 (all five futures alive; for a group the group's own score), aligned share at depth 16.
Usage: scoreboard_compare.py <A> <B> [<C> ...]   (intervals for every variant vs the first)  -> stdout + scoreboard_v1/compare_<A>.json
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
OUT = ROOT / 'artifacts/eda/scoreboard_v1'
CLASSES = ('hud', 'player', 'mob', 'entering', 'static')
DEPTHS = (1, 2, 4, 8, 16)


def main():
    import teval as T
    names = sys.argv[1:]
    meta, _, _ = T.split()
    seeds = meta['seed']
    alive0 = ~meta['future_dead'][:, 0].cumsum(1).bool()
    alive5 = ~meta['future_dead'].cumsum(2).bool().any(1)
    groups = [torch.where(seeds == s)[0] for s in seeds.unique()]
    g = torch.Generator().manual_seed(20261010)
    draws = [torch.cat([groups[k] for k in torch.randint(len(groups), (len(groups),), generator=g).tolist()]) for _ in range(2000)]
    sb = {n: json.loads((OUT / n / 'scoreboard.json').read_text()) for n in names}
    rows = {n: torch.load(OUT / n / 'rows.pt', weights_only=False) for n in names}
    V = sb[names[0]]['V']

    def per_root(n):
        r = rows[n]
        out = {}
        for k in (1, 4, 8, 16):
            out[f'gen{k}'] = (r['gen_err'].float().mean(0)[:, k - 1] / V, alive0[:, k - 1])
        for di, k in enumerate(DEPTHS):
            if k in (1, 4, 8, 16):
                for ci, c in enumerate(CLASSES):
                    out[f'es{k}_{c}'] = (r['es'][:, di, ci], alive5[:, k - 1])
        out['aligned16'] = (r['aligned'][:, 15], alive0[:, 15])
        return out
    pr = {n: per_root(n) for n in names}
    res = {'variants': names, 'intervals_vs': names[0], 'metrics': {}}
    for key in pr[names[0]]:
        row = {}
        for n in names:
            v, m = pr[n][key]
            m = m & ~v.isnan()
            row[n] = round(float(v[m].mean()), 5)
        for n in names[1:]:
            (va, ma), (vb, mb) = pr[names[0]][key], pr[n][key]
            m = ma & mb & ~va.isnan() & ~vb.isnan()
            diff = torch.where(m, vb - va, torch.zeros(()))
            cnt = m.float()
            boot = torch.stack([diff[d].sum() / cnt[d].sum().clamp(min=1) for d in draws])
            lo, hi = float(boot.quantile(0.025)), float(boot.quantile(0.975))
            row[f'{n}-{names[0]}'] = [round(float(diff.sum() / cnt.sum()), 5), round(lo, 5), round(hi, 5), lo > 0 or hi < 0]
        res['metrics'][key] = row
    # non-per-root summaries side by side
    res['facts16'] = {n: sb[n]['facts']['16']['imagined'] for n in names}
    res['facts1'] = {n: sb[n]['facts']['1']['imagined'] for n in names}
    res['facts16_true_tokens'] = sb[names[0]]['facts']['16']['true_tokens']
    res['scroll'] = {n: sb[n]['scroll'] for n in names}
    res['hud_events'] = {n: sb[n]['hud_events'] for n in names}
    res['damage'] = {n: sb[n]['damage']['mean'] for n in names}
    res['decomposition16'] = {n: sb[n]['decomposition']['16'] for n in names}
    res['onestep'] = {n: sb[n]['errors']['onestep_over_copy'] for n in names}
    res['manifold16'] = {n: sb[n]['manifold_imagined_vs_true']['16'] for n in names}
    (OUT / f'compare_{names[0]}.json').write_text(json.dumps(res, indent=1))
    w = max(len(n) for n in names) + 2
    print('metric'.ljust(16) + ''.join(n.rjust(w) for n in names) + '   intervals vs ' + names[0])
    for key, row in res['metrics'].items():
        ints = '  '.join(f"{n}:{row[f'{n}-{names[0]}'][0]:+.4f}[{row[f'{n}-{names[0]}'][1]:+.4f},{row[f'{n}-{names[0]}'][2]:+.4f}]{'*' if row[f'{n}-{names[0]}'][3] else ''}" for n in names[1:])
        print(key.ljust(16) + ''.join(f'{row[n]:.4f}'.rjust(w) for n in names) + '   ' + ints)


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
