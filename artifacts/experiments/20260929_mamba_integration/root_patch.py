"""Same-fit patch-root facts for Raw, LDAD lambda 1 and 10 on inspected diagnosis roots.

This is an exploratory encoder-interface check, not a world or actor result. Identical seed split,
labels, feature selectors, ridge grid, and paired seed-cluster bootstrap for every encoder.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260926_diagnosis')]
import teval as E  # noqa: E402
from compound import auc  # noqa: E402

LAMS=(1e-3,1e-2,1e-1,1,10)


def fit(x,y,tr,va):
    mu=x[tr].mean(0);sd=x[tr].std(0).clamp_min(1e-6)
    def trans(z):return torch.cat([((z-mu)/sd).double(),torch.ones(len(z),1,dtype=torch.float64)],1)
    X,Y=trans(x[tr]),y[tr].double();Xv,Yv=trans(x[va]),y[va].double()
    mat=X.T@X;rhs=X.T@Y;eye=torch.eye(mat.shape[0],dtype=torch.float64)
    best=None
    for lam in LAMS:
        w=torch.linalg.solve(mat+lam*len(X)*eye,rhs)
        err=float(((Xv@w-Yv)**2).mean())
        if best is None or err<best[0]:best=(err,lam,w)
    return lambda z:(trans(z)@best[2]).float(),best[1]


def r2(p,y):
    return float(1-(p-y).square().sum()/(y-y.mean()).square().sum().clamp_min(1e-9))


def main():
    meta,train,train_seeds=E.split()
    seeds=meta['seed'];test=~train
    val=torch.isin(seeds,train_seeds[:len(train_seeds)//5]);tr=train&~val
    vis=E.facts_of(meta['root_visible'])
    near=torch.tensor(E.NEAR)
    labels={'zombie_near':vis['zombie'][test][:,near],
            'health':vis['hud'][test,0],'food':vis['hud'][test,1],
            'energy':vis['hud'][test,3],'tile_near':vis['tile'][test][:,near],
            'facing':vis['facing'][test]}
    result={}
    saved={'seed':seeds[test],'labels':labels,'arms':{}}
    for name in ('raw','ldad1','ldad10'):
        cache=E.build_cache(name,torch.device('cuda'))
        tok=cache['ctx'][:,-1].float()
        assert tok.shape==(len(seeds),81,192)
        cell=tok[:,E.MAP].flatten(0,1)
        cell_tr=tr[:,None].expand(-1,63).flatten()
        cell_va=val[:,None].expand(-1,63).flatten()
        zfn,zlam=fit(cell,vis['zombie'].flatten()[:,None],cell_tr,cell_va)
        tfn,tlam=fit(cell,F.one_hot(vis['tile'].flatten(),17).float(),cell_tr,cell_va)
        hfn,hlam=fit(tok[:,63:81].flatten(1),vis['hud'],tr,val)
        ffn,flam=fit(tok[:,31],F.one_hot(vis['facing'],4).float(),tr,val)
        xt=tok[test]
        zs=zfn(xt[:,E.MAP].flatten(0,1)).view(len(xt),63)[:,near]
        ts=tfn(xt[:,E.MAP].flatten(0,1)).view(len(xt),63,17).argmax(-1)[:,near]
        hs=hfn(xt[:,63:81].flatten(1))
        fs=ffn(xt[:,31]).argmax(-1)
        scores={'zombie_near':zs,'health':hs[:,0],'food':hs[:,1],
                'energy':hs[:,3],'tile_near':ts,'facing':fs}
        saved['arms'][name]=scores
        result[name]={'zombie_near_auc':auc(zs,labels['zombie_near']),
                      'health_r2':r2(scores['health'],labels['health']),
                      'food_r2':r2(scores['food'],labels['food']),
                      'energy_r2':r2(scores['energy'],labels['energy']),
                      'tile_near_acc':float((ts==labels['tile_near']).float().mean()),
                      'facing_acc':float((fs==labels['facing']).float().mean()),
                      'selected_ridge_lambda':{'zombie':zlam,'tile':tlam,'hud':hlam,'facing':flam}}
        print(json.dumps({'stage':'arm','name':name,'metrics':result[name]}),flush=True)
        del cache,tok
    unique=seeds[test].unique()
    groups=[torch.where(seeds[test]==s)[0] for s in unique]
    gen=torch.Generator().manual_seed(20260929)
    def metric(arm,key,idx):
        p=saved['arms'][arm][key][idx];y=labels[key][idx]
        if key=='zombie_near':return auc(p,y)
        if key in ('health','food','energy'):return r2(p,y)
        return float((p==y).float().mean())
    intervals={}
    for left,right in (('ldad1','raw'),('ldad10','ldad1')):
        pair=f'{left}_minus_{right}';intervals[pair]={}
        for key in ('zombie_near','health','food','energy','tile_near','facing'):
            vals=[]
            for _ in range(1000):
                chosen=torch.randint(len(unique),(len(unique),),generator=gen)
                idx=torch.cat([groups[int(i)] for i in chosen])
                x=metric(left,key,idx)-metric(right,key,idx)
                if x==x:vals.append(x)
            assert len(vals)>=950,(pair,key,len(vals))
            q=torch.tensor(vals).quantile(torch.tensor([.025,.975])).tolist()
            intervals[pair][key]={'difference':result[left][key+'_auc' if key=='zombie_near' else key+'_r2' if key in ('health','food','energy') else key+'_acc']-result[right][key+'_auc' if key=='zombie_near' else key+'_r2' if key in ('health','food','energy') else key+'_acc'],
                                  'ci95':q,'bootstrap':len(vals)}
    report={'status':'complete','scope':'exploratory diagnosis seeds, root patch tokens only',
            'roots_test':int(test.sum()),'test_seed_count':len(unique),'fit_roots':int(tr.sum()),
            'validation_roots':int(val.sum()),'metrics':result,'paired_seed_bootstrap':intervals,
            'cache_sha256':{name:hashlib.sha256(Path(str(E.CACHE).format(name)).read_bytes()).hexdigest()
                            for name in ('raw','ldad1','ldad10')}}
    HERE.joinpath('root_patch.json').write_text(json.dumps(report,indent=2)+'\n')
    torch.save(saved,HERE/'root_patch_rows.pt')
    print(json.dumps({'stage':'complete','report':str(HERE/'root_patch.json')}),flush=True)

if __name__=='__main__':main()
