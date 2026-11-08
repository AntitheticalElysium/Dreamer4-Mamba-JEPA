"""Paired frozen audit readings. No fitting, target selection or budget changes."""
import json
import importlib
from pathlib import Path
import numpy as np
import torch
F1A = importlib.import_module('artifacts.experiments.20261102_foveal_prediction.analyze')
HERE = Path(__file__).resolve().parent
MODES = ('base','fine','native','retained','null','random_retained','mixed','recovered_bf16')


def stats(a,b,ep):
    return F1A.stats(a,b,ep)


def main():
    torch.set_num_threads(4)
    meta = torch.load(HERE/'metadata.pt',map_location='cpu',weights_only=False)
    ep = meta['episode'].numpy(); night=meta['night'].numpy(); ent=meta['entity'].numpy(); scroll=meta['scroll'].numpy()
    strata = dict(primary=night&ent&~scroll,all=np.ones(len(ep),bool),night_entity=night&ent,
                  day_entity=~night&ent,night_empty=night&~ent,scroll=scroll)
    scopes = dict(diagnosis=ep<250,confirmation=ep>=250)
    out = {}; counts={}; errors={}
    for seed in (611701,611702):
        errors[seed]={}
        for mode in MODES:
            x=torch.load(HERE/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False).numpy()
            errors[seed][mode]=dict(fine=x.mean(-1))
            for target,mask in (('source',meta['source']),('neighborhood',meta['neighborhood'])):
                m=mask.numpy()
                errors[seed][mode][target]=(x*m).sum(-1)/m.sum(-1).clip(min=1)
        r={}
        for scope,ss in scopes.items():
            r[scope]={}
            counts[scope]={}
            for name,sm in strata.items():
                sel=ss&sm
                if not sel.any():continue
                counts[scope][name]=dict(rows=int(sel.sum()),episodes=len(np.unique(ep[sel])))
                r[scope][name]={}
                for target in ('fine','source','neighborhood'):
                    if target!='fine' and name not in ('primary','night_entity','day_entity'):continue
                    ee={mode:errors[seed][mode][target][sel] for mode in MODES}
                    q={mode:stats(ee['base'],ee[mode],ep[sel]) for mode in MODES if mode!='base'}
                    q['retained_vs_native']=stats(ee['native'],ee['retained'],ep[sel])
                    q['fine_vs_retained']=stats(ee['retained'],ee['fine'],ep[sel])
                    q['recovered_vs_retained']=stats(ee['retained'],ee['recovered_bf16'],ep[sel])
                    q['random_vs_retained']=stats(ee['retained'],ee['random_retained'],ep[sel])
                    r[scope][name][target]=q
        out[str(seed)]=r
    out['counts']=counts
    numeric=torch.load(HERE/'numeric_dev.pt',map_location='cpu',weights_only=False)
    nr={}
    for scope,ss in scopes.items():
        nr[scope]={}
        for name,sm in strata.items():
            sel=ss&sm
            means={k:float(v[sel].double().mean()) for k,v in numeric.items()}
            ratios=dict(retained_energy_fraction=means['retained_energy']/means['raw_energy'],
                null_energy_fraction=means['null_energy']/means['raw_energy'],
                fp32_subtraction_rms_over_fine_rms=(means['full_fp32_subtraction_error']/means['fine_energy'])**.5,
                bf16_error_rms_over_fine_rms=(means['bf16_total_error']/means['fine_energy'])**.5,
                bf16_accum_output_rms_over_fine_rms=(means['bf16_accum_output_error']/means['fine_energy'])**.5,
                bf16_recovered_rms_over_retained_rms=(means['bf16_recovered_error']/means['retained_energy'])**.5,
                cache_error_rms_over_fine_rms=(means['cached_fine_error']/means['fine_energy'])**.5)
            nr[scope][name]=dict(means=means,ratios=ratios)
    out['numeric']=nr
    replication={}
    for seed in (611701,611702):
        for mode in ('base','fine'):
            old=torch.load(HERE.parent/'20261102_foveal_prediction'/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False)['g']
            new=torch.load(HERE/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False)
            replication[f'{mode}_{seed}']=dict(max_error_difference=float((old-new).abs().max()),
                old_mean=float(old.mean()),new_mean=float(new.mean()))
    out['replication']=replication
    (HERE/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False))
    for seed in ('611701','611702'):
        for scope in scopes:
            print(seed,scope,counts[scope]['primary'])
            for target in ('fine','source','neighborhood'):
                print(target,[(m,round(q['reduction_percent'],3),[round(v,3) for v in q['reduction_percent_CI']])
                    for m,q in out[seed][scope]['primary'][target].items()])
    print('NUMERIC PRIMARY',nr['diagnosis']['primary']['ratios'],nr['confirmation']['primary']['ratios'])
    print('REPLICATION',replication)


if __name__=='__main__':main()
