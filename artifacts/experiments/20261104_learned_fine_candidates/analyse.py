"""Case-level paired reporting; generation draws are not independent episodes."""
import json,sys
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent

def write(name,x):
    (HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def interval(a):
    a=np.asarray(a,float)
    if len(a)==0:return None
    rng=np.random.default_rng(615800)
    boots=a[rng.integers(len(a),size=(10000,len(a)))].mean(1)
    return dict(mean=float(a.mean()),case_bootstrap_95=np.quantile(boots,[.025,.975]).tolist(),cases=len(a))

def arrays(r,key='event'):
    return np.stack([torch.cat([d[key] for d in r['draws'] if d['condition']==c]).numpy() for c in sorted({d['condition'] for d in r['draws']})])

def gates():
    rows=torch.load(HERE/'gates.pt',map_location='cpu',weights_only=False);out={}
    for family in sorted({r['meta']['family']+'/'+r['meta']['kind'] for r in rows}):
        a=[r for r in rows if r['meta']['family']+'/'+r['meta']['kind']==family]
        ev=np.stack([arrays(r).mean(1) for r in a])
        pos,neg=interval(ev[:,0]),interval(1-ev[:,1])
        out[family]=dict(present_correct=pos,absent_correct=neg,
            **{'pass':pos['mean']>=.75 and neg['mean']>=.75},
            cases=[dict(index=r['index'],episode=r['meta']['e'],previous8_visible=r['meta']['previous8_visible'],
                present_correct=float(p[0]),absent_correct=float(1-p[1])) for r,p in zip(a,ev)])
    write('gate_decisions.json',out)
    for k,v in out.items():print(k,'pass',v['pass'],'present',v['present_correct'],'absent',v['absent_correct'])

def traces():
    rows=torch.load(HERE/'traces.pt',map_location='cpu',weights_only=False);out={};cases=[]
    for family in sorted({r['meta']['family']+'/'+r['meta']['kind'] for r in rows}):
        a=[r for r in rows if r['meta']['family']+'/'+r['meta']['kind']==family]
        d={}
        for mode in ['gaze','content']:
            b=[r for r in a if r['mode']==mode];events=np.stack([arrays(r).mean(1) for r in b]);scores=np.stack([arrays(r,'score').mean(1) for r in b])
            if mode=='gaze':
                effects=np.stack([events[:,0]-events[:,1],events[:,3]-events[:,2]],1)
                correctness=np.stack([events[:,0],events[:,1],1-events[:,2],1-events[:,3]],1)
                score_effects=np.stack([scores[:,0]-scores[:,1],scores[:,3]-scores[:,2]],1)
                names=['present_focus','present_control','absent_focus','absent_control']
            else:
                effects=np.stack([events[:,0]-events[:,1],events[:,2]-events[:,3]],1)
                score_effects=np.stack([scores[:,0]-scores[:,1],scores[:,2]-scores[:,3]],1)
                correctness=np.stack([events[:,0],events[:,1],1-events[:,2],1-events[:,3]],1)
                names=['coarse_present_fine_present','coarse_present_fine_absent','coarse_absent_fine_present','coarse_absent_fine_absent']
            d[mode]=dict(correctness={k:interval(v) for k,v in zip(names,correctness.T)},
                event_effect_context_A=interval(effects[:,0]),event_effect_context_B=interval(effects[:,1]),
                balanced_event_effect=interval(effects.mean(1)),reader_score_effect_A=interval(score_effects[:,0]),
                reader_score_effect_B=interval(score_effects[:,1]),
                paired_label_flips=[int((arrays(r)[[0,2]]!=arrays(r)[[1,3]]).sum()) for r in b])
            for r,ev,ef in zip(b,events,effects):
                cases.append(dict(index=r['index'],family=family,mode=mode,episode=r['meta']['e'],time=r['meta']['t'],
                    previous8_visible=r['meta']['previous8_visible'],condition_event_rate=ev.tolist(),signed_effect=ef.tolist(),stages=r['stages']))
        out[family]=d
    write('trace_summary.json',out);write('trace_cases.json',cases)
    print(json.dumps(out,indent=2))

if __name__=='__main__':
    torch.set_num_threads(2)
    gates() if sys.argv[1]=='gates' else traces()
