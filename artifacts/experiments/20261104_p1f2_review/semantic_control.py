"""Read existing cheap forecasts and native samples on EXACT overlapping frames.

No inference, fitting, new generated futures or cohort search. Drink/wood,
both existing routes, both saved head cohorts, all overlapping eligible rows.
These are different predictors, not calibrated predictive distributions with
matched architectures. Cheap head uses clean past coarse; native temporal
context is noise-conditioned and additionally uses its learned belief.
"""
import importlib
import json
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;RUN=HERE.parent/'20261103_p1f2'
E=importlib.import_module('artifacts.experiments.20261030_m05_guided_fovea.evaluate')
A=importlib.import_module('artifacts.experiments.20261103_gaze_belief.analyze')

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    rd=torch.load(RUN/'readings/eval_m05/p1f2_1500_readers.pt',map_location='cpu',weights_only=False)
    assert rd['checkpoint_sha256']=='e0f6a24ec51d2bf021ca88ba8080f804595800dc9295aecf0e37c7da7d7e1234'
    rd=rd['readers'];ctx=json.loads((HERE/'M05_CONTEXTS.json').read_text())
    cache=torch.load(RUN/'readings/head/dev_596_coarse.pt',map_location='cpu',weights_only=False)
    z={co:dict(np.load(HERE/f'ck10/{co}.npz')) for co in ['saved','confirmation']}
    result={};arrays={}
    for co,f in z.items():
        result[co]={}
        for kind in ['drink','wood']:
            result[co][kind]={}
            look={(int(e),int(t)):i for i,(e,t,c) in enumerate(zip(f['ep'],f['t'],f['cls'])) if c==kind}
            for route in ['guided','periphery']:
                native=dict(np.load(RUN/f'readings/eval_m05/p1f2_1500_{route}_{kind}.npz'))
                ix=np.array([look.get(tuple(ctx[kind][c][:2]),-1) for c in native['row_ctx']])
                use=native['eligible']&(ix>=0)
                assert use.any()
                rows=np.flatnonzero(use);fi=ix[rows]
                ep=f['ep'][fi];tt=f['t'][fi];cell=f['counter'][fi]
                assert len(np.unique(cell))==1
                pred=torch.as_tensor(f['prediction'][fi,cell].astype(np.float32))
                real=cache['p'][cache['offsets'][ep]+tt,cell].float()
                before=cache['p'][cache['offsets'][ep]+tt-1,cell].float()
                nullv=torch.zeros((len(pred),8)) # Counter reader does not consume lighting.
                eligible=E.eligible_periphery(rd,kind,native['start'][rows],native['end'][rows],np.ones(len(rows),bool))
                assert eligible.all()
                hp,ho=E.read_periphery(rd,kind,pred,nullv,eligible)
                rp,ro=E.read_periphery(rd,kind,real,nullv,eligible)
                cp,co_ok=E.read_periphery(rd,kind,before,nullv,eligible)
                assert (cp==native['start'][rows]).all()
                assert (rp==native['observed'][rows]).all()
                assert (native['observed'][rows]==native['end'][rows]).all()
                success=ho&(hp==native['end'][rows]);copy=ho&(hp==native['start'][rows]);unknown=~ho
                other=ho&~success&~copy;assert (success.astype(int)+copy+unknown+other==1).all()
                metrics=dict(head_correct=success,head_unknown=unknown,head_copy=copy,head_other=other,
                    # Descriptive point-forecast diagnostic, added after seeing
                    # support abstention. Never substitutes for the qualified
                    # generation endpoint above or changes its support radius.
                    head_nearest_value_correct=hp==native['end'][rows],
                    head_nearest_value_copy=hp==native['start'][rows],
                    native_correct=native['success'][rows],native_unknown=native['unknown'][rows],
                    native_copy=native['copy'][rows],real_correct=ro&(rp==native['end'][rows]),
                    head_minus_native=success.astype(float)-native['success'][rows])
                result[co][kind][route]={}
                for group,mask in [('all',np.ones(len(rows),bool)),('first',native['start'][rows]==0),('later',native['start'][rows]>0)]:
                    if not mask.any():continue
                    estimates=A.estimates(np.stack([x[mask] for x in metrics.values()],-1),np.ones(mask.sum()),ep[mask])
                    result[co][kind][route][group]=dict(zip(metrics,estimates))
                    result[co][kind][route][group]['unique_contexts']=len(set(zip(ep[mask].tolist(),tt[mask].tolist())))
                arrays.update({f'{co}_{kind}_{route}_{k}':v for k,v in dict(ep=ep,t=tt,start=native['start'][rows],end=native['end'][rows],head_read=hp,**metrics).items()})
    note='Saved-data semantic readability check, NOT a capacity theorem or fine-route localization. One cheap conditional-mean forecast vs sampled native futures, with different training objectives/conditioning noise. Real-support UNKNOWN retained. No output-based case filtering. Post-result nearest-value point accuracy is additionally described; it does not qualify these means as real-manifold generative samples or calibrated probabilities.'
    (HERE/'SEMANTIC_CONTROL.json').write_text(json.dumps(dict(results=result,note=note),indent=2)+'\n')
    np.savez_compressed(HERE/'semantic_control.npz',**arrays)
    for co,rr in result.items():
        for kind,rs in rr.items():
            for route,groups in rs.items():
                r=groups['all']
                print(co,kind,route,'contexts',r['unique_contexts'],'head',r['head_correct'],'native',r['native_correct'],'head unknown',r['head_unknown']['mean'],'native unknown',r['native_unknown']['mean'],flush=True)

if __name__=='__main__':main()
