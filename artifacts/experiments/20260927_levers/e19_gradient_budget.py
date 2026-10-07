"""CPU postprocess of existing TRAIN gradients: the actual objective direction.

Previously measured gradient cosines do not determine whether the total objective
opposes ordinary damage. Sum the exact saved class and teacher gradients using
the actual B/C loss coefficients. Report dot products, param-family splits and
per-batch signs. Positive dot means a small negative-gradient step lowers class
loss; negative means it raises it to first order. Frozen endpoint proj/choose
weights only, no bias/backbone gradients, Adam moments, or historical claim.
No new model forward or training. Evidence is hash-bound and atomic.
"""
import argparse
import json
from pathlib import Path
import torch
import h16_resume as R

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CLASSES=('death','ordinary_ge2','unchanged','other')


def summarize(records,arm,slice_=slice(None)):
    gradients={c:torch.stack([r['gradients'][c][slice_]for r in records]).mean(0)
               for c in (*CLASSES,'teacher')}
    total=gradients['teacher']+(sum(gradients[c]for c in CLASSES)if arm=='C'else 0)
    ordinary=gradients['ordinary_ge2'];dots={k:float(ordinary@v)for k,v in gradients.items()}
    per=[]
    for r in records:
        if not r['selected']['ordinary_ge2']:continue
        g=r['gradients']['ordinary_ge2'][slice_]
        gt=r['gradients']['teacher'][slice_]+(sum(r['gradients'][c][slice_]for c in CLASSES)if arm=='C'else 0)
        per.append({'update':r['update'],'selected_damage':r['selected']['ordinary_ge2'],'dot':float(g@gt)})
    d=float(ordinary@total)
    ret={'ordinary_dot_objective':d,'objective_norm':float(total.norm()),
         'ordinary_self_dot':float(ordinary@ordinary),'ordinary_dot_each_term':dots,
         'damage_batches':len(per),'uphill_damage_batches':sum(r['dot']<0 for r in per),
         'downhill_damage_batches':sum(r['dot']>0 for r in per),'per_batch':per}
    if arm=='C':
        without=d-dots['unchanged'];ret['ordinary_dot_without_unchanged']=without
        ret['unchanged_weight_for_zero_dot']=-without/dots['unchanged']if dots['unchanged']else None
    return ret


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=7)
    p.add_argument('--backbone',choices=['fmamba','full'],default='fmamba');a=p.parse_args()
    torch.set_num_threads(3)
    dirs={}
    for arm in ('B','C'):
        matches=list((ROOT/'artifacts/eda/frozen_eval_resume_v1').glob(
            f'e19_{arm}_s{a.seed}_{a.backbone}_from36000__e19_train_mechanism__*/contract.json'))
        assert len(matches)==1,matches;dirs[arm]=matches[0].parent
    files=[Path(__file__).resolve(),HERE/'h16_resume.py']
    for d in dirs.values():
        files.extend([d/'contract.json',d/'result.json'])
        for update in range(0,6000,200):
            j=d/f'batch_{update}.json';files.append(j);files.append(d/json.loads(j.read_text())['file'])
    spec={'scope':__doc__,'inputs':{str(p):R.file_hash(p)for p in files},'seed':a.seed,'backbone':a.backbone,
          'coefficients':{'B':{'teacher':1},'C':{c:1 for c in (*CLASSES,'teacher')}},
          'gradient_order':'proj.weight flattened, then choose.weight flattened','projection_size':192*256}
    name=f'e19_s{a.seed}_{a.backbone}__gradient_budget'
    store=R.Store(HERE/'evals/resume'/name,spec)
    with store.lock():
        result=store.load('result')
        if result is None:
            out={}
            for arm,d in dirs.items():
                original=R.Store(d,json.loads((d/'contract.json').read_text()))
                records=[original.load('batch_'+str(i))for i in range(0,6000,200)]
                # Check aggregation against the already sealed report.
                old=original.load('result')
                for c in (*CLASSES,'teacher'):
                    norm=float(torch.stack([r['gradients'][c]for r in records]).mean(0).norm())
                    assert abs(norm-old['mean_head_gradient_norm'][c])<=1e-7,(c,norm,old['mean_head_gradient_norm'][c])
                assert records[0]['gradients']['teacher'].numel()==(192+6)*256
                out[arm]={'all_measured_weights':summarize(records,arm),
                          'proj_weight':summarize(records,arm,slice(0,192*256)),
                          'choose_weight':summarize(records,arm,slice(192*256,None))}
            result={'scope':__doc__,'contract':spec,'arms':out}
            store.save('result',result,1)
        R.atomic_json(HERE/'evals'/(name+'.json'),result)
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
