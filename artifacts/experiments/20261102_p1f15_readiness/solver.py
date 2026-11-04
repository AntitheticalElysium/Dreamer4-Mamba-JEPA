"""Frozen native F1.5 sampler test. PLAN.md; no model/reader swaps or training."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import copy, importlib, json, time
from pathlib import Path
import numpy as np
import torch
from saccade import train as TR
from saccade.eval import s1 as S

T = importlib.import_module('artifacts.experiments.20261101_regression_localization.trace')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

def main():
    torch.set_num_threads(4)
    dest = HERE / 'solver'; dest.mkdir(exist_ok=True)
    path = ROOT / 'artifacts/experiments/20261101_p1f15/run/ckpt_10.pt'
    model, cfg = T.AI.load(path)
    model.requires_grad_(False)
    L = S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')), 200)
    old = json.loads((ROOT/'artifacts/experiments/20261101_regression_localization/timeline/contexts.json').read_text())
    full = T.AI.classes(L, np.random.default_rng(2031))
    names = ('random','risky','ach_WAKE_UP','ach_COLLECT_COAL','ach_COLLECT_IRON')
    rng = np.random.default_rng(611202)
    classes = {}
    for c in names:
        rows = old.get(c, full[c])
        classes[c] = [tuple(rows[i]) for i in sorted(rng.choice(len(rows), min(256,len(rows)), replace=False))]
    T.CLASSES = names
    (dest/'contexts.json').write_text(json.dumps(classes, indent=1))
    result = {'checkpoint':str(path), 'sha256':TR.file_sha256(path), 'draws':4,
              'script_sha256':TR.file_sha256(__file__), 'trace_sha256':TR.file_sha256(T.__file__),
              'plan_sha256':TR.file_sha256(HERE/'PLAN.md'), 'algorithms':{}}
    start=time.time()
    enc=S.encode(model,L,cfg)
    result['encode_seconds']=time.time()-start
    print('encoded',round(result['encode_seconds'],1),flush=True)
    for name, steps, churn in (('e3c1',3,1),('e8c0',8,0),('e32c1',32,1)):
        c=copy.deepcopy(cfg); c['edm']['sampler'].update(steps=steps,s_churn=churn)
        start=time.time()
        raw, checks=T.evaluate(model,c,enc,L,classes,4)
        np.savez_compressed(dest/f'{name}.npz', **{f'{cl}__{k}':v for cl,z in raw.items() for k,v in z.items()})
        result['algorithms'][name]={'seconds':time.time()-start,'checks':checks,
                                   'classes':T.summarise(raw,4)}
        # Brier below corrects the finite-K variance of the predictive probability estimate.
        for cl,z in raw.items():
            n=len(z['episode'])//4
            rr=result['algorithms'][name]['classes'][cl]
            for key,target in (('reward_pos',z['reward']>.5),('death_joint',z['death'])):
                p=z['gen_'+key].reshape(n,4)
                y=target.reshape(n,4)[:,0]
                fair=(p.mean(1)-y)**2-p.var(1,ddof=1)/4
                rr['gen_'+key+'_fair_brier']=T.G.estimate(fair,np.ones(n,bool),z['episode'][::4])
            print(name,cl,{k:round(rr[k]['est'],4) for k in
                  ('recorded_positive_reward','real_reward_pos','gen_reward_pos','recorded_death','gen_death_joint')},flush=True)
        (dest/'summary.json').write_text(json.dumps(result,indent=1))
        print(name,'seconds',round(time.time()-start,1),flush=True)

if __name__=='__main__':main()
