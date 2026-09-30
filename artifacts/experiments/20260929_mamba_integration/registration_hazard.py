"""Exploratory near-player scope of terrain registration on prior inspected factual futures."""
import hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),str(HERE)]
import teval as E
from registration_scope import terrain_decision, zombie, shifted
from scroll import SHIFTS


def main():
    torch.set_num_threads(8)
    meta,_,_=E.split()
    cache=torch.load(Path(str(E.CACHE).format('raw')),map_location='cpu',weights_only=False,mmap=True)
    alive=~meta['future_dead'][:,0].cumsum(1).bool()
    totals=dict(steps=0,near_cells=0,near_zombie_changed=0,near_zombie_present=0,
                near_changed_steps=0,near_fixed_mse_sum=0.,near_aligned_mse_sum=0.,
                near_changed_aligned_mse_sum=0.,near_stable_aligned_mse_sum=0.,
                near_changed_pairs=0,near_stable_pairs=0)
    by_seed={}
    for i in range(len(cache['ctx'])):
        v0=meta['root_visible'][i];t0=cache['ctx'][i,-1].float()
        seed=int(meta['seed'][i]);sb=by_seed.setdefault(seed,[0.,0.])
        for k in range(cache['fut'].shape[1]):
            if not bool(alive[i,k]):break
            v1=meta['future_visible'][i,0,k];t1=cache['fut'][i,k].float();act=int(meta['future_actions'][i,k])
            if 1<=act<=4:
                shift,_,_=terrain_decision(v0,v1,act)
                if shift>0:
                    dr,dc=SHIFTS[shift]
                    a,b,fixed=shifted(t0[:63].reshape(7,9,192),t1[:63].reshape(7,9,192),shift)
                    za,zb,_=shifted(zombie(v0),zombie(v1),shift)
                    rr=torch.arange(max(0,-dr),7-max(0,dr))[:,None]
                    cc=torch.arange(max(0,-dc),9-max(0,dc))[None,:]
                    near=((rr>=2)&(rr<=4)&(cc>=3)&(cc<=5))
                    assert int(near.sum())==9
                    changed=za!=zb
                    f=(fixed-b).square().mean(-1)[near]
                    e=(a-b).square().mean(-1)[near]
                    ne=changed[near]
                    totals['steps']+=1;totals['near_cells']+=9
                    totals['near_zombie_changed']+=int(ne.sum())
                    totals['near_zombie_present']+=int((za|zb)[near].sum())
                    totals['near_changed_steps']+=int(bool(ne.any()))
                    totals['near_fixed_mse_sum']+=float(f.mean())
                    totals['near_aligned_mse_sum']+=float(e.mean())
                    sb[0]+=float(f.mean());sb[1]+=float(e.mean())
                    if bool(ne.any()):
                        totals['near_changed_pairs']+=1
                        totals['near_changed_aligned_mse_sum']+=float(e[ne].mean())
                    if bool((~ne).any()):
                        totals['near_stable_pairs']+=1
                        totals['near_stable_aligned_mse_sum']+=float(e[~ne].mean())
            v0,t0=v1,t1
    n=totals['steps'];assert n==5690
    out={**totals,
         'near_zombie_changed_fraction':totals['near_zombie_changed']/totals['near_cells'],
         'near_changed_step_fraction':totals['near_changed_steps']/n,
         'near_mean_fixed_mse':totals['near_fixed_mse_sum']/n,
         'near_mean_aligned_mse':totals['near_aligned_mse_sum']/n,
         'near_relative_mse_reduction':1-totals['near_aligned_mse_sum']/totals['near_fixed_mse_sum'],
         'near_changed_cell_error':totals['near_changed_aligned_mse_sum']/totals['near_changed_pairs'],
         'near_stable_cell_error':totals['near_stable_aligned_mse_sum']/totals['near_stable_pairs']}
    rows=list(by_seed.values());g=torch.Generator().manual_seed(20260929);bs=[]
    for _ in range(1000):
        ix=torch.randint(len(rows),(len(rows),),generator=g)
        f=sum(rows[int(j)][0] for j in ix);a=sum(rows[int(j)][1] for j in ix)
        if f:bs.append(1-a/f)
    out['near_relative_mse_reduction_seed_ci95']=torch.tensor(bs).quantile(torch.tensor([.025,.975])).tolist()
    out['raw_cache_sha256']=hashlib.sha256(Path(str(E.CACHE).format('raw')).read_bytes()).hexdigest()
    out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    HERE.joinpath('registration_hazard.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
