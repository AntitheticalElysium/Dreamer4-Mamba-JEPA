"""World cost in L1, beside teval's MSE / copy (2026-10-09; E21 D predeclared: an L2-trained world may be favoured by teval's
squared-error currency, so L1 is reported beside it). Exactly teval's held data and teacher-forced path: the Raw diagnosis cache
(1,002 roots never trained on: 4 context frames, 16 true future frames with their actions), teval.evaluate's `tf` rollout with
`window` frames (L16 worlds: 15, as teval_export --window 15): frame k is predicted from the true frames before it. Per root, mean
|pred - true| over the 16 steps (alive steps only) x 81 tokens x 192, and over the 63 map tokens. Paired seed-clustered bootstrap
(143 walk seeds, 2,000 draws, seed 20261009) of B - A for every pair given. -> artifacts/eda/health_chain_v1/l1cost.json
Usage: health_l1cost.py <name=world.pt> ... --pairs A:B ... [--window 15]
"""
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
OUT = ROOT / 'artifacts/eda/health_chain_v1'


@torch.no_grad()
def main():
    import teval as T
    from d4mj.config import config_from_dict
    import spatial as S
    args = sys.argv[1:]
    window = int(args[args.index('--window') + 1]) if '--window' in args else 15
    if '--window' in args:
        i = args.index('--window'); args = args[:i] + args[i + 2:]
    pairs = [p.split(':') for p in args[args.index('--pairs') + 1:]]
    worlds = dict(a.split('=', 1) for a in args[:args.index('--pairs')])
    dev = torch.device('cuda')
    config = config_from_dict(torch.load(S.CHECKPOINT, map_location='cpu', weights_only=False)['config'])
    meta, train_roots, train_seeds = T.split()
    cache = T.build_cache('raw', torch.device('cpu'))
    R, H = len(meta['seed']), cache['fut'].shape[1]
    alive = ~meta['future_dead'][:, 0].cumsum(1).bool()                             # [R, 16]
    per = {}
    for name, path in worlds.items():
        w, _ = T.load_world(Path(path), dev)
        tot_all, tot_map = torch.zeros(R), torch.zeros(R)
        for i in range(0, R, 16):
            ctx, ca = cache['ctx'][i:i + 16].float(), cache['ctx_a'][i:i + 16]
            fa, fut = cache['fut_a'][i:i + 16], cache['fut'][i:i + 16].float()
            frames, a_hist = [ctx[:, j] for j in range(4)], [ca[:, j] for j in range(3)]
            for k in range(H):                                                      # teval.evaluate's tf path
                wk = 4 if k == 0 else window
                a = torch.stack(a_hist[-(wk - 1):] + [fa[:, k]], 1)
                pred = T.step(w, torch.stack(frames[-wk:], 1), a, dev, config)
                err = (pred - fut[:, k]).abs()
                m = alive[i:i + 16, k].float()
                tot_all[i:i + 16] += err.mean((1, 2)) * m
                tot_map[i:i + 16] += err[:, :63].mean((1, 2)) * m
                frames.append(fut[:, k]); a_hist.append(fa[:, k])
        n = alive.sum(1).clamp(min=1).float()
        per[name] = torch.stack([tot_all / n, tot_map / n], 1)
        print(json.dumps({name: [round(float(per[name][:, 0].mean()), 5), round(float(per[name][:, 1].mean()), 5)]}), flush=True)
    seeds = meta['seed']
    groups = [torch.where(seeds == s)[0] for s in seeds.unique()]
    g = torch.Generator().manual_seed(20261009)
    draws = [torch.cat([groups[k] for k in torch.randint(len(groups), (len(groups),), generator=g).tolist()]) for _ in range(2000)]
    res = {'window': window, 'means': {n: {'all': float(v[:, 0].mean()), 'map': float(v[:, 1].mean())} for n, v in per.items()}, 'pairs': {}}
    for a, b in pairs:
        for j, part in enumerate(('all', 'map')):
            diff = per[b][:, j] - per[a][:, j]
            boot = torch.stack([diff[d].mean() for d in draws])
            lo, hi = float(boot.quantile(0.025)), float(boot.quantile(0.975))
            res['pairs'][f'{a}->{b}:{part}'] = {'diff': round(float(diff.mean()), 5), 'ci95': [round(lo, 5), round(hi, 5)],
                                                'resolved': lo > 0 or hi < 0}
            print(json.dumps({f'{a}->{b}:{part}': res['pairs'][f'{a}->{b}:{part}']}), flush=True)
    (OUT / 'l1cost.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260926_diagnosis'))
    sys.path.insert(0, str(ROOT / 'artifacts/experiments/20260921_readout_ladder'))
    main()
