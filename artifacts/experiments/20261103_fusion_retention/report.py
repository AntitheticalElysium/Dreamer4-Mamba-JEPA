"""CPU-only paired reporting; no world/model imports or fitting."""
import importlib
import json
from pathlib import Path
import numpy as np
import torch

F=importlib.import_module('artifacts.experiments.20261102_foveal_prediction.analyze')
A=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')
HERE=Path(__file__).resolve().parent
MODES=('base','fine','joint','mixed','predicted','memory')
SEEDS=(611701,611702)


def paired_contrast(base_a,a,base_b,b,ep):
    """Difference of within-world percentage gains, paired resampling of episodes."""
    es,inv=np.unique(ep,return_inverse=True)
    sums=[np.bincount(inv,weights=x) for x in (base_a,a,base_b,b)]
    draws=np.random.default_rng(611730).integers(len(es),size=(2000,len(es)))
    ba,aa,bb,ab=[x[draws].sum(1) for x in sums]
    delta=100*((1-ab/bb)-(1-aa/ba))
    point=100*((1-b.mean()/base_b.mean())-(1-a.mean()/base_a.mean()))
    return dict(gain_difference_pp=float(point),ci95=np.quantile(delta,[.025,.975]).tolist(),rows=len(ep),episodes=len(es))


def main():
    torch.set_num_threads(4);result={};error={};metas={}
    for world in ('p1f15','p1f16'):
        out=HERE/world;m=torch.load(out/'metadata.pt',map_location='cpu',weights_only=False)
        m={k:v.numpy() if isinstance(v,torch.Tensor) else v for k,v in m.items()};metas[world]=m
        ep=m['episode'];strata=dict(primary=m['night']&m['entity']&~m['scroll'],all=np.ones(len(ep),bool),
            day_entity=~m['night']&m['entity'],night_entity=m['night']&m['entity'],night_empty=m['night']&~m['entity'],scroll=m['scroll'])
        scopes=dict(diagnosis=ep<250,confirmation=ep>=250);errors={}
        for seed in SEEDS:
            errors[seed]={}
            for mode in (*MODES,'copy'):
                actual='fine' if world=='p1f15' and mode=='joint' else mode
                path=out/'copy_errors.pt' if mode=='copy' else out/f'errors_{actual}_{seed}.pt'
                x=torch.load(path,map_location='cpu',weights_only=False).numpy()
                errors[seed][mode]=dict(crop=x.mean(-1),source=(x*m['source']).sum(-1)/m['source'].sum(-1).clip(min=1),
                    neighborhood=(x*m['neighborhood']).sum(-1)/m['neighborhood'].sum(-1).clip(min=1))
        error[world]=errors;rr={}
        for seed in SEEDS:
            rr[str(seed)]={}
            for scope,ss in scopes.items():
                rr[str(seed)][scope]={}
                for cohort,sm in strata.items():
                    sel=ss&sm
                    if not sel.any():continue
                    row=dict(rows=int(sel.sum()),episodes=len(np.unique(ep[sel])),targets={})
                    for target in ('crop','source','neighborhood'):
                        if target!='crop' and cohort not in ('primary','day_entity','night_entity'):continue
                        ee={k:errors[seed][k][target][sel] for k in errors[seed]}
                        row['targets'][target]={mode:F.stats(ee['base'],ee[mode],ep[sel]) for mode in (*MODES[1:],'copy')}
                    rr[str(seed)][scope][cohort]=row
        result[world]=rr
        print(world,'primary confirmation')
        for seed in SEEDS:
            for target in ('crop','source'):
                print(seed,target,{k:round(v['reduction_percent'],3) for k,v in rr[str(seed)]['confirmation']['primary']['targets'][target].items()})
    for key in metas['p1f15']:assert np.array_equal(metas['p1f15'][key],metas['p1f16'][key]),key
    ep=metas['p1f15']['episode'];m=metas['p1f15'];sel=(ep>=250)&m['night']&m['entity']&~m['scroll']
    result['paired_world_gain_difference']={}
    for seed in SEEDS:
        result['paired_world_gain_difference'][str(seed)]={}
        for target in ('crop','source','neighborhood'):
            aa=error['p1f15'][seed];bb=error['p1f16'][seed]
            result['paired_world_gain_difference'][str(seed)][target]={mode:paired_contrast(aa['base'][target][sel],aa[mode][target][sel],bb['base'][target][sel],bb[mode][target][sel],ep[sel]) for mode in MODES[1:]}
    result['paired_semantic']={}
    for world in ('p1f15','p1f16'):
        out=HERE/world;meta=dict(np.load(out/'paired_meta.npz'));lp=torch.load(out/'paired_logp.pt',map_location='cpu',weights_only=False)
        ix=np.arange(len(meta['truth']));rr={}
        for seed in SEEDS:
            baseline=lp[f'base_{seed}'].numpy()[ix,:,meta['truth']]
            base_effect=((meta['a']-meta['b'])*baseline).sum(1)
            rr[str(seed)]={}
            for mode in MODES[1:]:
                actual='fine' if world=='p1f15' and mode=='joint' else mode
                score=lp[f'{actual}_{seed}'].numpy()[ix,:,meta['truth']]
                effect=((meta['a']-meta['b'])*score).sum(1)-base_effect
                rr[str(seed)][mode]=A.estimate(effect,meta['episode'])
        result['paired_semantic'][world]=rr
        print('PAIRED SEMANTIC',world,json.dumps(rr))
    (HERE/'RESULTS.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
