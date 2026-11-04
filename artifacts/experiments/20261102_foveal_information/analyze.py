"""Read saved conditional and native paired arrays; no model fitting."""
import json,importlib
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')

def bootstrap_metric(diff,y,ep,mask):
    good=mask.flatten();y=y.flatten()[good];diff=diff.flatten()[good];ep=np.repeat(ep,63)[good]
    E=int(ep.max())+1
    totals=np.bincount(ep*5+y,weights=diff,minlength=E*5).reshape(E,5)
    count=np.bincount(ep*5+y,minlength=E*5).reshape(E,5)
    rng=np.random.default_rng(611550);i=rng.integers(E,size=(2000,E));ns=count[i].sum(1);valid=(ns>0).all(1)
    values=(totals[i].sum(1)/np.maximum(ns,1)).mean(1)
    return {'mean':float((totals.sum(0)/count.sum(0)).mean()),'by_kind':(totals.sum(0)/count.sum(0)).tolist(),
            'ci95':np.quantile(values[valid],[.025,.975]).tolist(),'kind_rows':count.sum(0).tolist(),'episodes':int(np.unique(ep).size)}

def ordinary():
    p=HERE/'conditional/dev_logp.pt'
    if not p.exists():return
    logs=torch.load(p,weights_only=False);z=torch.load(HERE/'dev_250.pt',weights_only=False)
    y=z['truth'].numpy();ep=z['episode'].numpy();inside=z['localg'].numpy()>=0;night=z['night'].numpy()
    ans={};flat={}
    for k,lp in logs.items():
        a=lp.float().numpy();flat[k]=(-np.take_along_axis(a,y[...,None],-1)[...,0],a.argmax(-1)==y)
    for family in ('flat','gated'):
      for seed in (0,1):
        base=flat[f'{family}_{seed}_base']
        for mode in ('fine','input','corrected','predicted','memory'):
            key=f'{family}_{seed}_{mode}';ce,acc=flat[key];res={}
            for part,mask in [('in_all',inside),('in_night',inside&night[:,None]),('in_day',inside&~night[:,None]),('out_all',~inside)]:
                res[part]={'ce_delta':bootstrap_metric(ce-base[0],y,ep,mask),'balanced_recall_delta':bootstrap_metric(acc.astype(float)-base[1],y,ep,mask)}
            ans[key]=res
    (HERE/'conditional_comparisons.json').write_text(json.dumps(ans,indent=1))
    for k,v in ans.items():
        print(k,'night CE',v['in_night']['ce_delta']['mean'],'night recall',v['in_night']['balanced_recall_delta']['mean'],flush=True)

def paired():
    if not (HERE/'paired_logp.pt').exists():return
    logs={k:v.float().numpy() for k,v in torch.load(HERE/'paired_logp.pt',weights_only=False).items()}
    meta=dict(np.load(HERE/'paired_meta.npz'));raw=dict(np.load(HERE.parent/'20261102_p1f15_readiness/uptake_local/raw.npz'))
    truth=meta['truth'];N=len(truth);pred={k:v.argmax(-1) for k,v in logs.items()}
    lp={k:np.take_along_axis(v,truth[:,None,None].repeat(49,axis=1),axis=-1)[...,0] for k,v in logs.items()}
    # Qualification is SOURCE-only and never uses any next-frame outcome/loss.
    allbase=np.stack([pred[f'{f}_{s}_base']!=truth[:,None] for f in ('flat','gated') for s in (0,1)]).all(0)
    allfine=np.stack([pred[f'{f}_{s}_fine']==truth[:,None] for f in ('flat','gated') for s in (0,1)]).all(0)
    conditional=allbase&allfine&meta['available']
    finegood=allfine&meta['available']
    qualified={}
    for cohort,infoall in [('conditional_useful',conditional),('fine_readable',finegood)]:
      vals={k:[] for k in logs};dd=[];bb=[];episodes=[];ids=[];rms=[]
      for q in range(N):
        ri=meta['raw_i'][q];w=A.weights(infoall[q],~raw['held_mob'][ri])
        if w is None:continue
        a,b=w;ids.append(q);episodes.append(meta['episode'][q]);loss=raw['loss'][ri]
        dd.append(np.einsum('snwg,w->sng',loss,a-b));bb.append(np.einsum('snwg,w->sng',loss,b))
        rms.append(float(np.sqrt((np.einsum('swcd,w->scd',raw['prediction'][ri],a-b)**2).mean())))
        for k in logs:
            # Subtract gaze-only baseline effects in the independently fitted readout.
            f,s,mode=k.split('_');base=lp[f'{f}_{s}_base'];v=lp[k]-base
            vals[k].append(float(np.dot(v[q],a-b)))
      rr={'queries':len(ids),'scenes':len(np.unique(meta['raw_i'][ids])) if ids else 0,'episodes':len(np.unique(episodes)),
          'queries_by_kind':np.bincount(truth[ids],minlength=5).tolist() if ids else [0]*5,'query_ids':ids}
      if ids:
        rr['conditional_log_score_gain']={k:A.estimate(v,episodes) for k,v in vals.items()}
        dd=np.stack(dd);bb=np.stack(bb)
        rr['native_denoise_fractional_change']={reg:{band:A.estimate(dd[:,:,sl,j].mean((1,2))/bb[:,:,sl,j].mean((1,2)),episodes)
            for band,sl in [('all',slice(None)),('high',slice(-2,None))]} for reg,j in [('map',0)]}
        rr['native_first_call_map_rms']=A.estimate(rms,episodes)
        rr['local_scope_warning']='Local loss intentionally omitted: saved source neighborhoods do not match newly qualified entities. Native per-cell losses were not saved.'
      qualified[cohort]=rr
    np.savez_compressed(HERE/'paired_qualification.npz',conditional=conditional,finegood=finegood)
    (HERE/'paired_summary.json').write_text(json.dumps(qualified,indent=1))
    for k,v in qualified.items():print(k,v['queries'],v['scenes'],v['queries_by_kind'],flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    ordinary();paired()
