"""Freeze U Mamba h; refit only h -> next-U to localize output failure.

Ridge refits use independent 900k future roots, split by episode seed. One fit
uses only the logged factual action per root, the other all 17 forks. They are
diagnostics, not LeWM training proposals. Native projector is untouched.
"""
import json
from pathlib import Path

import torch

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
DATA=ROOT/'artifacts/eda/diagnosis_rollouts_v1'

def fit(x,y,lam):
    mu=x.mean(0);sd=x.std(0).clamp_min(1e-6);ym=y.mean(0)
    z=((x-mu)/sd).double();target=(y-ym).double();D=z.shape[1]
    weights=torch.linalg.solve(z.T@z+lam*len(z)*torch.eye(D,dtype=torch.float64),z.T@target)
    return lambda q: (((q-mu)/sd).double()@weights+ym).float()

def main():
    torch.set_num_threads(6)
    d=torch.load(DATA/'U.pt',weights_only=False,mmap=True)
    m=torch.load(DATA/'meta.pt',weights_only=False,mmap=True)
    h=d['one_h'].float();true=d['one_true'].float()[:,0]
    mean=d['one_true'].float().mean(1);native=d['one'].float();root=d['root'].float()
    seeds=m['seed'];u=seeds.unique();order=u[torch.randperm(len(u),generator=torch.Generator().manual_seed(20260926))]
    n=int(.7*len(u));dv=int(.15*len(u));fit_seed,dev_seed,test_seed=order[:n-dv],order[n-dv:n],order[n:]
    tr=torch.isin(seeds,fit_seed);dev=torch.isin(seeds,dev_seed);test=torch.isin(seeds,test_seed)
    logged=m['future_actions'][:,0].long();idx=torch.arange(len(logged))
    V=float(true.flatten(0,1).var(0).sum())
    out={'roots':len(root),'fit_roots':int(tr.sum()),'dev_roots':int(dev.sum()),'test_roots':int(test.sum()),'V':V,'arms':{}}
    norm=lambda x:x.square().sum(-1)
    for arm in ('factual','all_actions'):
        def rows(sel):
            if arm=='factual':return h[sel,logged[sel]],mean[sel,logged[sel]]
            return h[sel].flatten(0,1),mean[sel].flatten(0,1)
        x,y=rows(tr);xd,yd=rows(dev)
        choices=[]
        for lam in (1e-5,1e-4,1e-3,1e-2,1e-1,1):
            pred=fit(x,y,lam)(xd)
            choices.append((float(norm(pred-yd).mean()),lam))
        _,lam=min(choices)
        model=fit(*rows(tr|dev),lam)
        got=model(h[test].flatten(0,1)).reshape(int(test.sum()),17,192)
        target=true[test]; target_mean=mean[test];rr=root[test,None];nn=native[test]
        noise=float(norm(d['one_true'][test].float()-target_mean[:,None]).sum(1).mean()/3/V)
        out['arms'][arm]={'selected_lambda':lam,'dev_mse':min(choices)[0]/V,
            'all_actions':{'refit_mse':float(norm(got-target).mean()/V),'native_mse':float(norm(nn-target).mean()/V),
                           'persist_mse':float(norm(rr-target).mean()/V),
                           'refit_vs_mean_mse':float(norm(got-target_mean).mean()/V),
                           'native_vs_mean_mse':float(norm(nn-target_mean).mean()/V),'key_noise':noise},
            'factual_action':{}}
        a=logged[test];j=torch.arange(len(a));g=got[j,a];na=nn[j,a];t=target[j,a];r=rr[:,0]
        out['arms'][arm]['factual_action']={
            'refit_mse':float(norm(g-t).mean()/V),'native_mse':float(norm(na-t).mean()/V),
            'persist_mse':float(norm(r-t).mean()/V)}
        print(arm,json.dumps(out['arms'][arm]),flush=True)
    (HERE/'projector_refit.json').write_text(json.dumps(out,indent=2)+'\n')

if __name__=='__main__':main()
