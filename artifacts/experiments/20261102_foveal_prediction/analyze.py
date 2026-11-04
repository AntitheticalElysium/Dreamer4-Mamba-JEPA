"""Paired episode-bootstrap predictive-value readings; no model fitting."""
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
MODES=('base','fine','input','predicted','memory')


def stats(a,b,episode):
    """Positive reduction means b predicts better than a on identical target rows."""
    a,b,episode=np.asarray(a,float),np.asarray(b,float),np.asarray(episode)
    ids,inv=np.unique(episode,return_inverse=True)
    sa=np.bincount(inv,weights=a);sb=np.bincount(inv,weights=b);counts=np.bincount(inv)
    rng=np.random.default_rng(611730);ix=rng.integers(len(ids),size=(2000,len(ids)))
    av=sa[ix].sum(1)/counts[ix].sum(1);bv=sb[ix].sum(1)/counts[ix].sum(1)
    delta=av-bv;relative=100*delta/av
    return dict(base=float(a.mean()),candidate=float(b.mean()),reduction=float(a.mean()-b.mean()),
        reduction_CI=np.quantile(delta,[.025,.975]).tolist(),
        reduction_percent=float(100*(a.mean()-b.mean())/a.mean()),
        reduction_percent_CI=np.quantile(relative,[.025,.975]).tolist(),rows=len(a),episodes=len(ids))


def reduce(q,meta):
    p=q['p'].float().numpy();g=q['g'].float().numpy()
    mask=meta['mask'].float().numpy();dil=meta['dilated'].float().numpy()
    out=dict(coarse_crop=(p*mask).sum(-1)/mask.sum(-1),coarse_dilated=(p*dil).sum(-1)/dil.sum(-1),
             coarse_global=p.mean(-1),fine=g.mean(-1))
    for name,field,err in (('coarse_source','source',p),('coarse_source_neighborhood','source_neighborhood',p),
                           ('fine_source','fine_source',g),('fine_source_neighborhood','fine_source_neighborhood',g)):
        m=meta[field].float().numpy();out[name]=(err*m).sum(-1)/m.sum(-1).clip(min=1)
    return out


def main():
    torch.set_num_threads(4)
    meta=torch.load(HERE/'dev_metadata.pt',weights_only=False,map_location='cpu')
    z=torch.load(HERE/'dev_400.pt',weights_only=False,map_location='cpu')
    dr,dc=torch.meshgrid(torch.arange(-1,2),torch.arange(-1,2),indexing='ij')
    cells=(z['fix'][:,0,None]*9+z['fix'][:,1,None]+dr.flatten()[None]*9+dc.flatten()[None]).long()
    source=torch.zeros(len(z['p']),81);source[:,:63]=(z['truth']>0).float()
    source*=meta['mask'].float()
    around=F.max_pool2d(source.reshape(-1,1,9,9),3,stride=1,padding=1).reshape(-1,81)
    meta.update(source=source,source_neighborhood=around,fine_source=source.gather(1,cells),
                fine_source_neighborhood=around.gather(1,cells))
    del z
    ep=meta['episode'].numpy();night=meta['night'].numpy();ent=meta['entity'].numpy();scroll=meta['scroll'].numpy()
    strata=dict(primary=night&ent&~scroll,all=np.ones(len(ep),bool),night_entity=night&ent,
                day_entity=~night&ent,night_empty=night&~ent,no_scroll=~scroll,scroll=scroll)
    copy=reduce({'p':meta['copy_p'],'g':meta['copy_g']},meta)
    scopes={'diagnosis':ep<250,'confirmation':ep>=250}
    out={};counts={}
    for scope,s in scopes.items():
        counts[scope]={name:dict(rows=int((s&m).sum()),episodes=len(np.unique(ep[s&m]))) for name,m in strata.items()}
    for seed in (611701,611702):
        errors={mode:reduce(torch.load(HERE/f'errors_{mode}_{seed}.pt',weights_only=False,map_location='cpu'),meta) for mode in MODES}
        r={}
        for scope,s in scopes.items():
            r[scope]={}
            for name,mm in strata.items():
                sel=s&mm
                if not sel.any():continue
                r[scope][name]={}
                for target in ('coarse_crop','coarse_dilated','coarse_global','fine','coarse_source',
                               'coarse_source_neighborhood','fine_source','fine_source_neighborhood'):
                    if 'source' in target and name not in ('primary','night_entity','day_entity'):continue
                    base=errors['base'][target][sel]
                    comp={mode:stats(base,errors[mode][target][sel],ep[sel]) for mode in MODES if mode!='base'}
                    comp['copy_current']=stats(base,copy[target][sel],ep[sel])
                    comp['fine_vs_memory']=stats(errors['memory'][target][sel],errors['fine'][target][sel],ep[sel])
                    r[scope][name][target]=comp
        out[str(seed)]=r
    out['counts']=counts
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    for seed in ('611701','611702'):
        for scope in scopes:
            print(seed,scope,'primary',counts[scope]['primary'])
            for target in ('coarse_crop','coarse_dilated','fine'):
                print(target,[(m,round(q['reduction_percent'],3),[round(v,3) for v in q['reduction_percent_CI']])
                    for m,q in out[seed][scope]['primary'][target].items()])


if __name__=='__main__':main()
