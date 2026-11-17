"""Bounded CPU-only falsification checks on saved F3/F4 outputs; see PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import importlib, json, hashlib
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, train as TR, data as D

HERE=Path(__file__).resolve().parent
FOLLOW=HERE.parent/'20261107_f4_followups'
CV=importlib.import_module('artifacts.experiments.20261106_determinant_curves.curves')
CKPT=HERE.parent/'20261106_p1f4/run/ckpt_10.pt'

def boot(x,ep,w=None):
    w=np.ones(len(x)) if w is None else np.asarray(w)
    ids,iv=np.unique(ep,return_inverse=True)
    n=np.bincount(iv,w);s=np.bincount(iv,w*x)
    rng=np.random.default_rng(110711)
    z=[]
    for _ in range(2000):
        pick=rng.integers(0,len(ids),len(ids));z.append(s[pick].sum()/n[pick].sum())
    return dict(mean=float((w*x).sum()/w.sum()),ci95=np.percentile(z,[2.5,97.5]).tolist())

def proper_scores(cfg):
    ld=dict(np.load(CV.LABELS/'s0_dev_labels.npz'));td=CV.transitions(ld)
    tr=dict(np.load(CV.LABELS/'s0_train_labels.npz'));tt=CV.transitions(tr)
    keep={(ep.shard,ep.slot) for ep in D.load_split(cfg)['train']}
    keep_ep=np.array([(int(s),int(o)) in keep for s,o in zip(tr['ep_shard'],tr['ep_slot'])])
    eligible=keep_ep[tt['ep']] & (tt['t']>=CV.P) & tt['adjacent']
    yt=(tt['true_body'][:,0]<tt['body'][:,0]).astype(float)
    base={str(s):float(yt[eligible & (tt['asleep']==s)].mean()) for s in (False,True)}
    files={'F3':CV.HERE/'readings/raw.npz','F4':HERE.parent/'20261106_p1f4/readings/consequences/curves/raw.npz'}
    z={k:np.load(p) for k,p in files.items()}
    j=z['F3']['zombie/j'];assert np.array_equal(j,z['F4']['zombie/j'])
    u,iv=np.unique(j,return_inverse=True);n=np.bincount(iv)
    assert np.all(n==4)
    y=(td['true_body'][u,0]<td['body'][u,0]).astype(float)
    ep=td['ep'][u];sl=td['asleep'][u]
    strata=CV.strata(td)['zombie'];panel=strata[u]
    weights=np.zeros(len(u))
    for name in np.unique(panel):
        m=panel==name;weights[m]=(strata==name).sum()/m.sum()
    score={};forecasts={}
    for name,r in z.items():
        k=np.bincount(iv,(r['zombie/gen_hud63']<td['body'][j,0]).astype(float))
        q=k/n;forecasts[name]=q
        score[name]=k*(k-1)/(n*(n-1))-2*q*y+y
    qb=np.where(sl,base['True'],base['False'])
    score['train_sleep_adjacency_base_rate']=(qb-y)**2
    out={'histories':len(u),'draws_per_history':4,'training_only_base_rates':base,
         'score':'unbiased ensemble estimate k(k-1)/(m(m-1)) - 2(k/m)y + y',
         'note':'Panel and DEV-stratum weighted results are separate; this is not a learned deployment policy.'}
    for label,ww in [('panel',np.ones(len(u))),('dev_population_ipw',weights)]:
        out[label]={}
        for state,m in [('all',np.ones(len(u),bool)),('awake',~sl),('asleep',sl)]:
            out[label][state]={'rows':int(m.sum()),'true_rate':float(np.average(y[m],weights=ww[m])),
                'scores':{a:boot(s[m],ep[m],ww[m]) for a,s in score.items()},
                'F4_minus_F3':boot((score['F4']-score['F3'])[m],ep[m],ww[m]),
                'F4_minus_base':boot((score['F4']-score['train_sleep_adjacency_base_rate'])[m],ep[m],ww[m]),
                'F3_minus_base':boot((score['F3']-score['train_sleep_adjacency_base_rate'])[m],ep[m],ww[m])}
    return out

def cache_dependency(ck):
    den=M.Denoiser(ck['cfg']).eval()
    den.load_state_dict({k[len('denoiser.'):]:v for k,v in ck['model'].items() if k.startswith('denoiser.')})
    cfg=ck['cfg'];k=den.k_max
    dims=[(81,16),(9,64),(8,)]
    torch.manual_seed(110711)
    x=tuple(torch.randn(1,1,*d) for d in dims)
    shape=(1,1,cfg['denoiser']['memory_tokens'],cfg['denoiser']['width'])
    m0=torch.zeros(shape);m1=torch.randn(shape)*.1
    a=torch.zeros(1,1,dtype=torch.long);fix=torch.full((1,1,2),4,dtype=torch.long)
    caches=[]
    with torch.no_grad():
        for mem in (m0,m1):
            cache=den.new_cache()
            for _ in range(3):
                cache=M.context_pass(den,x,a,mem,fix,cache,cfg,torch.Generator().manual_seed(18))
            caches.append(cache)
        code=torch.full((1,1),M.shortcut_code(0,2,k),dtype=torch.long)
        outs=[den.step(x,code,a,m0,fix,c)[0] for c in caches]
    delta=float((outs[0][0]-outs[1][0]).abs().max())
    assert delta>0
    return dict(physical_inputs_and_current_memory_fixed=True,only_historical_memory_changed=True,
        physical_output_max_abs_difference=delta,
        interpretation='A structural dependency test with synthetic inputs; not a benefit/necessity ablation.')

def phase_curves():
    out={}
    for name in ('r1_f4','r1_f3_16','r1_f3_3'):
        z=json.loads((FOLLOW/f'readings/danger/{name}.json').read_text())
        out[name]={s:{b:{q:r[q]['est'] for q in ('drawn','true')} for b,r in v.items()} for s,v in z['strata'].items()}
    return out

def first_guess_counterexample():
    # Exact posterior-mean flow field for Y in {0,2}, independent N(0,1)
    # initial noise. Its tau=0 clean estimate projects to p for EVERY sample.
    # Euler integration still generates both modes; that first mean is not a
    # discrete commitment. This is an analytic-model control, NOT F4's trained
    # shortcut field or a proposed F4 sampler change.
    initial=np.random.default_rng(110711).standard_normal(100000)
    result={}
    for p in (.03,.1,.5):
        rates={}
        for steps in (4,8,128):
            x=initial.copy()
            for j in range(steps):
                t=j/steps;s=1-t
                logit=np.log(p/(1-p))+(2*t*x-2*t*t)/(s*s)
                mean=2*np.exp(-np.logaddexp(0,-logit))
                x=x+(mean-x)/(s*steps)
            rates[str(steps)]=float((x>1).mean())
        result[str(p)]={'first_estimate_projection':p,'final_event_rates':rates}
    return {'model':'exact two-atom posterior-mean flow, scalar means 0 and 2',
            'F4_first_Euler_move_fraction':.25,'not_a_test_of_F4_shortcut_accuracy':True,'results':result}

def main():
    torch.set_num_threads(1)
    ck=torch.load(CKPT,map_location='cpu',weights_only=False)
    result={'proper_brier':proper_scores(ck['cfg']),'cache_dependency':cache_dependency(ck),'phase_curves':phase_curves(),
            'first_guess_counterexample':first_guess_counterexample()}
    inputs=[CKPT,FOLLOW/'danger_readings.py',FOLLOW/'readings/danger/zombie_brier.json',
            FOLLOW/'readings/danger/r1_f4.json',FOLLOW/'readings/danger/r1_f3_16.json',TR.ROOT/'saccade/model.py']
    result['input_sha256']={str(p.relative_to(TR.ROOT)):TR.file_sha256(p) for p in inputs}
    (HERE/'CPU.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='phase_curves'},indent=2),flush=True)

if __name__=='__main__':main()
