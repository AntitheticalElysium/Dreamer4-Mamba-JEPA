"""CPU paired reporting; reads every predeclared arm, population and seed."""
import importlib
import json
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / '20261103_fusion_retention/p1f16'
F = importlib.import_module('artifacts.experiments.20261102_foveal_prediction.analyze')
MODES = ('base', 'fine', 'joint', 'native', 'retained', 'null', 'random_retained', 'mixed', 'recovered_bf16')


def main():
    torch.set_num_threads(4)
    m = torch.load(HERE/'metadata.pt', map_location='cpu', weights_only=False)
    m = {k:v.numpy() for k,v in m.items()}; ep=m['episode']
    cohorts = dict(primary=m['night']&m['entity']&~m['scroll'], all=np.ones(len(ep),bool),
                   day_entity=~m['night']&m['entity'], night_entity=m['night']&m['entity'],
                   night_empty=m['night']&~m['entity'], scroll=m['scroll'])
    scopes = dict(diagnosis=ep<250, confirmation=ep>=250)
    out = {}
    for seed in (611701,611702):
        errors={}
        for mode in MODES:
            directory=OLD if mode in ('base','fine') else HERE
            e=torch.load(directory/f'errors_{mode}_{seed}.pt',map_location='cpu',weights_only=False).numpy()
            errors[mode]=dict(crop=e.mean(-1), **{
                target:(e*m[target]).sum(-1)/m[target].sum(-1).clip(min=1)
                for target in ('source','neighborhood')})
        out[str(seed)]={}
        for scope, sm in scopes.items():
            out[str(seed)][scope]={}
            for cohort, cm in cohorts.items():
                sel=sm&cm
                row=dict(rows=int(sel.sum()),episodes=len(np.unique(ep[sel])),targets={})
                for target in ('crop','source','neighborhood'):
                    if target!='crop' and cohort not in ('primary','night_entity','day_entity'):continue
                    e={k:errors[k][target][sel] for k in MODES}
                    rr={k:F.stats(e['base'],e[k],ep[sel]) for k in MODES[1:]}
                    for name,a,b in (('retained_vs_native','native','retained'),
                        ('joint_vs_retained','retained','joint'),
                        ('recovered_vs_retained','retained','recovered_bf16'),
                        ('random_vs_retained','retained','random_retained')):
                        rr[name]=F.stats(e[a],e[b],ep[sel])
                    row['targets'][target]=rr
                out[str(seed)][scope][cohort]=row
        for scope in scopes:
            print(seed,scope)
            for target,r in out[str(seed)][scope]['primary']['targets'].items():
                print(target,{k:(round(v['reduction_percent'],3),[round(x,3) for x in v['reduction_percent_CI']])
                              for k,v in r.items()},flush=True)
    n=torch.load(HERE/'numeric_dev.pt',map_location='cpu',weights_only=False)
    out['numeric']={}
    for scope, sm in scopes.items():
        sel=sm&cohorts['primary']; mean={k:float(v[sel].double().mean()) for k,v in n.items()}
        ratios=dict(retained_energy_fraction=mean['retained_energy']/mean['raw_energy'],
                    null_energy_fraction=mean['null_energy']/mean['raw_energy'],
                    fp32_subtraction_relative_rms=(mean['full_fp32_subtraction_error']/mean['fine_energy'])**.5,
                    bf16_mixed_relative_rms=(mean['bf16_total_error']/mean['fine_energy'])**.5,
                    bf16_decoded_relative_rms=(mean['bf16_recovered_error']/mean['retained_energy'])**.5)
        out['numeric'][scope]=dict(means=mean,ratios=ratios)
    (HERE/'RESULTS.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    print('NUMERIC',out['numeric'],flush=True)


if __name__=='__main__':main()
