"""CPU frozen trace geometry and distribution-matched local readability.

Ridge classifiers train on diagnosis FIT seeds, lambda selected on their fixed
validation seeds, evaluated on TEST seeds. Binary ordinary >=2 damage versus
unchanged; recovery excluded. h63 is a local hidden feature, HUD-only input is a
limited control, not the world's full input. Failure is not an information ceiling.
Geometry separates the true health-token change from orthogonal generator error.
No GPU, no new training world, atomic source/input-bound completed-arm resume.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import e19_diagnose as D
T,R=D.T,D.R


def auc(score,y,weights=None):
    order=np.argsort(score,kind='stable');s=score[order];y=y[order]
    ww=np.ones(len(y))if weights is None else weights[order]
    ends=np.r_[np.flatnonzero(s[1:]!=s[:-1])+1,len(s)];starts=np.r_[0,ends[:-1]]
    p=np.add.reduceat(ww*y,starts);n=np.add.reduceat(ww*(1-y),starts)
    den=p.sum()*n.sum()
    return float((p*(n.cumsum()-.5*n)).sum()/den)if den else np.nan


def interval(score,y,seeds):
    score,y,seeds=[x.cpu().numpy()for x in (score,y,seeds)]
    u,inverse=np.unique(seeds,return_inverse=True);rng=np.random.default_rng(20261005)
    vals=[]
    for _ in range(1000):
        count=np.bincount(rng.integers(len(u),size=len(u)),minlength=len(u))
        vals.append(auc(score,y,count[inverse]))
    return {'auc':auc(score,y),'interval95':np.nanquantile(vals,[.025,.975]).tolist(),
            'positive':int(y.sum()),'negative':int((1-y).sum()),'episode_seeds':len(u)}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,choices=[7,8],default=7)
    p.add_argument('--backbone',choices=['fmamba','full'],default='fmamba')
    args=p.parse_args()
    torch.set_num_threads(3)
    meta,fit,seeds=T.split();mask={k:v[:,0]for k,v in D.D.DR.masks(meta).items()}
    shift,ok=D.true_shifts(meta);valid=mask['valid']&mask['k3'];assert(ok|~valid).all()
    eligible=valid&(mask['drop2']|(mask['dh']==0));rr,kk=eligible.nonzero().T
    labels=mask['drop2'][rr,kk].float();rootseed=meta['seed'][rr]
    va=torch.isin(rootseed,seeds[:len(seeds)//5]);tr=fit[rr]&~va;test=~fit[rr]
    fut,_=D.D.SD.token_cache(torch.device('cpu'));target=fut[:,0,:,63].float()
    cache=T.build_cache('raw',torch.device('cpu'));actions=cache['fut_a'][rr,kk]
    a=torch.nn.functional.one_hot(actions,17).float()
    for arm in ('C','B'):
        name=f'e19_{arm}_s{args.seed}_{args.backbone}_from36000'
        path=HERE/'evals'/f'{name}__e19_router_trace.pt'
        # Keep the historical s7 result files intact; new replicas have explicit lineage names.
        result_name=f'e19_{arm}_trace_readout' if args.seed==7 and args.backbone=='fmamba' else name+'__e19_trace_readout'
        spec={'trace':R.file_hash(path),'script':R.file_hash(__file__),'teval':R.file_hash(T.__file__),
              'metadata':R.file_hash(T.META),'cut':'drop2 vs unchanged; original FIT/TEST seed split'}
        store=R.Store(HERE/'evals/resume'/result_name,spec)
        with store.lock():
            result=store.load('result')
            if result is None:
                z=torch.load(path,map_location='cpu',weights_only=False)['features']
                truth=target-z['input63'];generated=z['gen63']-z['input63']
                power=truth.square().sum(-1).clamp_min(1e-9)
                alpha=(generated*truth).sum(-1)/power
                orthogonal=(generated-alpha[...,None]*truth).square().sum(-1)/power
                groups={'fresh':valid&mask['drop2']&mask['adjacent']&~mask['win']&~mask['adjwin'],
                        'scroll':valid&mask['drop2']&(shift!=0),'stationary':valid&mask['drop2']&(shift==0)}
                geometry={name:{'n':int(m.sum()),'parallel_coefficient_median':float(alpha[m].median()),
                               'orthogonal_error_over_true_change_median':float(orthogonal[m].median())}
                          for name,m in groups.items()}
                x=z['input63'][rr,kk]
                inputs={'HUD_input':torch.cat([x,a],-1),
                        'hidden63':torch.cat([x,z['h63'][rr,kk],a],-1),
                        'output63':torch.cat([x,z['output63'][rr,kk]-x,a],-1),
                        'generator63':torch.cat([x,z['gen63'][rr,kk]-x,a],-1)}
                readings={}
                fresh=mask['adjacent'][rr,kk]&~mask['win'][rr,kk]&~mask['adjwin'][rr,kk]
                subsets={'overall':test,'scroll':test&(shift[rr,kk]!=0),
                         'fresh_vs_scroll_unchanged':test&((labels.bool()&fresh)|(~labels.bool()&(shift[rr,kk]!=0)))}
                for name,xx in inputs.items():
                    head=T.ridge(xx,labels[:,None],tr,va)
                    score=head(xx)[:,0]
                    readings[name]={s:interval(score[m],labels[m],rootseed[m])for s,m in subsets.items()}
                result={'arm':arm,'scope':__doc__,'geometry':geometry,'readability':readings,'contract':spec}
                store.save('result',result,1)
            R.atomic_json(HERE/'evals'/f'{result_name}.json',result)
            print(json.dumps(result,indent=2),flush=True)
