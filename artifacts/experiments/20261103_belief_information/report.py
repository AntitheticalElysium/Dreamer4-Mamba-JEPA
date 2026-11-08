"""Predeclared broad prediction/paired-look report; CPU only, no selection."""
import importlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
F=importlib.import_module('artifacts.experiments.20261102_foveal_prediction.analyze')
E=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.analyze_uptake')
MODES=('base','fine','joint','mixed','recovered_bf16','predicted')
SEEDS=(612101,612102)


def main():
    torch.set_num_threads(4)
    m=torch.load(HERE/'metadata.pt',map_location='cpu',weights_only=False);m={k:v.numpy() for k,v in m.items()}
    ep=m['episode'];out=dict(predictors={},paired={})
    strata=dict(all=np.ones(len(ep),bool),day=~m['night'],night=m['night'],
                query_map=m['query'][:,0]<6,query_hud=m['query'][:,0]>=6,
                no_scroll=~m['scroll'],scroll=m['scroll'])
    masks={'physical_all':('p',np.ones((len(ep),81))),
           'physical_map':('p',np.tile(np.r_[np.ones(63),np.zeros(18)],(len(ep),1))),
           'physical_hud':('p',np.tile(np.r_[np.zeros(63),np.ones(18)],(len(ep),1))),
           'physical_entity':('p',m['source']),
           'fine_all':('g',np.ones((len(ep),9))),
           'fine_map':('g',~m['fine_hud']),'fine_hud':('g',m['fine_hud']),
           'fine_entity':('g',m['fine_source'])}
    for seed in SEEDS:
        err={mode:torch.load(HERE/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False) for mode in MODES}
        result={}
        for scope,ss in dict(diagnosis=ep<250,confirmation=ep>=250).items():
            result[scope]={}
            for cohort,cs in strata.items():
                result[scope][cohort]={}
                for target,(key,mask) in masks.items():
                    sel=ss&cs&(mask.sum(1)>0)
                    if not sel.any():continue
                    es={mode:(err[mode][key].numpy()*mask).sum(1)/mask.sum(1).clip(min=1) for mode in MODES}
                    result[scope][cohort][target]={mode:F.stats(es['base'][sel],es[mode][sel],ep[sel]) for mode in MODES[1:]}
        out['predictors'][str(seed)]=result
        print('CONFIRMATION',seed,{target:{mode:round(v['reduction_percent'],3) for mode,v in r.items()}
              for target,r in result['confirmation']['all'].items()},flush=True)
    paired=torch.load(HERE/'paired.pt',map_location='cpu',weights_only=False)
    frames=np.array([r['frame'] for r in paired['meta']]);pe=np.array([r['episode'] for r in paired['meta']])
    a=np.stack([r['a'] for r in paired['meta']]);b=np.stack([r['b'] for r in paired['meta']])
    pmasks={name:(key,mask[frames]) for name,(key,mask) in masks.items()}
    pstrata=dict(all=np.ones(len(pe),bool),day=~m['night'][frames],night=m['night'][frames],
                 query_map=m['query'][frames,0]<6,query_hud=m['query'][frames,0]>=6)
    for seed in SEEDS:
        out['paired'][str(seed)]={}
        for target,(key,mask) in pmasks.items():
            vals={}
            for mode in MODES:
                v=paired['readers'][f'{mode}_{seed}'][key].numpy()
                v=(v*mask[:,None,:]).sum(-1)/mask.sum(-1).clip(min=1)[:,None]
                vals[mode]=((b-a)*v).sum(1) # positive means informative look helps
            out['paired'][str(seed)][target]={}
            for cohort,cs in pstrata.items():
                sel=cs&(mask.sum(1)>0)
                if not sel.any():continue
                out['paired'][str(seed)][target][cohort]={mode:E.estimate((vals[mode]-vals['base'])[sel],pe[sel]) for mode in MODES[1:]}
    out['native']={}
    for target,(key,mask) in pmasks.items():
        v=paired['native'][key].numpy().mean((1,2))
        v=(v*mask[:,None,:]).sum(-1)/mask.sum(-1).clip(min=1)[:,None]
        diff=((b-a)*v).sum(1);control=(b*v).sum(1)
        out['native'][target]={}
        for cohort,cs in pstrata.items():
            sel=cs&(mask.sum(1)>0)
            if not sel.any():continue
            out['native'][target][cohort]=dict(gain=E.estimate(diff[sel],pe[sel]),
                fractional_gain=E.estimate(diff[sel]/control[sel].clip(min=1e-12),pe[sel]))
    # Instrument qualifications are explicit; a null is never relabeled erasure.
    qualify={}
    for target in masks:
        qualify[target]={}
        for cohort in ('all','day','night','query_map','query_hud'):
            prediction=[];look=[]
            for seed in SEEDS:
                p=out['predictors'][str(seed)]['confirmation'][cohort].get(target)
                prediction.append(bool(p and p['fine']['reduction_percent_CI'][0]>0))
                q=out['paired'][str(seed)].get(target,{}).get(cohort,{}).get('fine')
                look.append(bool(q and q['ci95'][0]>0))
            qualify[target][cohort]=dict(predictive_control_both_seeds=all(prediction),
                                        genuine_look_control_both_seeds=all(look))
    out['qualification']=qualify
    (HERE/'RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('QUALIFICATION',json.dumps(qualify),flush=True)
    print('NATIVE',json.dumps(out['native']),flush=True)


if __name__=='__main__':main()
