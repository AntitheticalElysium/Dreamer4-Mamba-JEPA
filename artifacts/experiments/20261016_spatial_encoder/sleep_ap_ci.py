"""R62, POST HOC and DESCRIPTIVE (not a preregistered reading): episode-bootstrap (2,000) intervals of sleeping-frame AP
differences (full - MoP, full - Raw, MoP - Raw; deployed and backbone_ln; zombie / cow; populations A / B), from
sleep_strat_eval.py's units. The quantity was named in R61 before any full-arm number; the intervals were added after.
Population A's sleeping frames come from only 8 episodes: its intervals are fragile.
Usage: sleep_ap_ci.py -> prints JSON"""
import torch, json
OUT = '/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/artifacts/eda/spatial_encoder_v1/'
res = {}
for s in (1, 2):
    U = {a: torch.load(f'{OUT}sleep2_units_{a}_s{s}.pt', weights_only=False) for a in ('raw', 'mop', 'full')}
    for pop in ('A', 'B'):
        lab = U['mop']['labels'][pop]
        for a in U: assert all(torch.equal(U[a]['labels'][pop][k], lab[k]) for k in ('zl', 'cl', 'sleep', 'episode'))
        eps = sorted(set(lab['episode'].tolist())); eo = {e: i for i, e in enumerate(eps)}
        m = lab['sleep'].repeat_interleave(63)
        ep = torch.tensor([eo[e] for e in lab['episode'].tolist()]).repeat_interleave(63)[m]
        draws = torch.randint(len(eps), (2000, len(eps)), generator=torch.Generator().manual_seed(7 + s))
        mult = torch.zeros(2000, len(eps)).scatter_add_(1, draws, torch.ones(2000, len(eps))).long()
        for mob, key, lk in (('zombie', 'z', 'zl'), ('cow', 'c', 'cl')):
            l = lab[lk][m]
            for st in ('deployed', 'backbone_ln'):
                ap = {}
                for a in U:
                    sc = U[a][pop][st][key][m]; o = sc.argsort(descending=True); ls, es = l[o], ep[o]
                    def f(w):
                        x = ls.repeat_interleave(w[es]).double()
                        return float((x.cumsum(0) / torch.arange(1, len(x) + 1) * x).sum() / x.sum()) if x.sum() > 0 else float('nan')
                    ap[a] = (f(torch.ones(len(eps), dtype=torch.long)), torch.tensor([f(mult[b]) for b in range(2000)]))
                for x, y in (('full', 'mop'), ('full', 'raw'), ('mop', 'raw')):
                    d = ap[x][1] - ap[y][1]; d = d[~torch.isnan(d)]
                    res[f's{s}_{pop}_{mob}_{st}_{x}-{y}'] = [round(ap[x][0], 3), round(ap[y][0], 3), round(float(d.quantile(.025)), 3), round(float(d.quantile(.975)), 3)]
        res[f's{s}_{pop}_counts'] = {'sleep_frames': int(lab['sleep'].sum()), 'zombie_pos': int((lab['zl'] & m).sum()),
                                     'cow_pos': int((lab['cl'] & m).sum()), 'episodes_with_sleep': len(set(lab['episode'][lab['sleep']].tolist()))}
print(json.dumps(res, indent=0))
