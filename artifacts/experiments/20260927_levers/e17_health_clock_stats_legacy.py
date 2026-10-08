"""Completed E17 health interventions: paired episode-cluster statistics.

No fitting/GPU. Quantify actual early/late teacher catches and the isolated final
time-embedding effect, including repeated-current and true-history controls.
These are internal/OOD interventions on two frozen worlds, not a repaired recipe.
"""
import json
from pathlib import Path
import numpy as np
import torch
import h16_resume as R
import e17_h16_diagnose as D

HERE=Path(__file__).resolve().parent


def main():
    stores={};rows={};contracts={}
    for stem in ('e17_health_clock_b16','e17_health_time_swap','e17_health_time_true'):
        p=HERE/'evals/resume'/stem;c=json.loads((p/'contract.json').read_text());s=R.Store(p,c)
        assert s.load('result') is not None
        stores[stem]=s;rows[stem]=s.load('rows');contracts[stem]=c
    spec={'scope':__doc__,'references':contracts,'sources':{f:R.file_hash(f) for f in (__file__,R.__file__,D.__file__)}}
    store=R.Store(HERE/'evals/resume/e17_health_clock_stats',spec)
    with store.lock():
        done=store.load('result')
        if done is not None:print(json.dumps(done),flush=True);return
        b=rows['e17_health_clock_b16'];sw=rows['e17_health_time_swap'];tr=rows['e17_health_time_true'];out={}
        for seed in (7,8):
            orig=b['worlds'][seed]['true_history']['drawn_health_change']
            same=b['worlds'][seed]['repeat_same_length']['drawn_health_change']
            counts={}
            for name,mask in b['masks'].items():
                per={}
                for label,part in (('context_below15',b['lengths']<15),('context15',b['lengths']==15)):
                    m=mask&part;per[label]={'n':int(m.sum()),'caught_or_false':int((orig[m]<-.5).sum())}
                per['overlap_true_repeated_same_length']=int(((orig<-.5)&(same<-.5)&mask).sum())
                counts[name]=per
            timeeffects={}
            for reference,changed,data,left,right in (
                ('intact15','len15_last4',sw,sw['worlds'][seed]['intact15'],sw['worlds'][seed]['len15_last4']),
                ('intact5','len5_last14',sw,sw['worlds'][seed]['intact5'],sw['worlds'][seed]['len5_last14']),
                ('true_intact','true_last4',tr,tr['worlds'][seed]['intact'],tr['worlds'][seed]['last4'])):
                per={}
                for name,mask in data['masks'].items():
                    ids=data['seed'][mask].numpy()
                    per[name]={'health_change_effect':D.paired(right[mask].numpy(),left[mask].numpy(),ids),
                               'drop_rate_effect':D.paired((right[mask]<-.5).double().numpy(),(left[mask]<-.5).double().numpy(),ids)}
                timeeffects[changed+'_minus_'+reference]=per
            out[str(seed)]={'teacher_catches_by_context_length':counts,'time_embedding_effects':timeeffects}
        result={'scope':__doc__,'contract':spec,'worlds':out};store.save('result',result,1)
        R.atomic_json(HERE/'evals/e17_health_clock_stats.json',result)
        print(json.dumps(out,indent=2),flush=True)


if __name__=='__main__':main()
