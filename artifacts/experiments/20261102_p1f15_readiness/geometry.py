"""Probe-independent token-constraint diagnosis; selected follow-up in PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import copy, importlib, json
from pathlib import Path
import numpy as np
import torch
from saccade import train as TR
from saccade.eval import s1 as S
T=importlib.import_module('artifacts.experiments.20261101_regression_localization.trace')
HERE=Path(__file__).resolve().parent
BASE_READ=T.read

@torch.no_grad()
def read(model,cfg,bp,parts,fix,truth):
    ret,features,h,p=BASE_READ(model,cfg,bp,parts,fix,truth)
    for name,x in zip(('p','g','v'),parts):
        xx=x.float().reshape(len(x),-1,x.shape[-1])
        inv=3*torch.atanh((xx/3).clamp(-.99999,.99999))
        for key,v in [('inverse_mean_abs',inv.mean(-1).abs().mean(-1)),
                      ('inverse_second_moment_error',(inv.square().mean(-1)-1).abs().mean(-1)),
                      ('inverse_rms',inv.square().mean(-1).sqrt().mean(-1)),
                      ('token_rms',xx.square().mean(-1).sqrt().mean(-1)),
                      ('off_constraint_fraction',((inv.mean(-1).abs()>.01)|((inv.square().mean(-1)-1).abs()>.05)).float().mean(-1))]:
            ret[f'geometry_{name}_{key}']=v.cpu().numpy()
    return ret,features,h,p

def main():
    torch.set_num_threads(4);dest=HERE/'geometry';dest.mkdir(exist_ok=True)
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),200)
    model,cfg=T.AI.load(HERE.parent/'20261101_p1f15/run/ckpt_10.pt');model.requires_grad_(False)
    enc=S.encode(model,L,cfg)
    classes={k:[tuple(r) for r in v] for k,v in json.loads((HERE/'solver/contexts.json').read_text()).items()}
    T.CLASSES=tuple(classes);T.read=read
    result={'script_sha256':TR.file_sha256(__file__),'plan_sha256':TR.file_sha256(HERE/'PLAN.md'),'algorithms':{}}
    for alg,n,churn in (('e3c1',3,1),('e8c0',8,0),('e32c1',32,1)):
        c=copy.deepcopy(cfg);c['edm']['sampler'].update(steps=n,s_churn=churn)
        raw,checks=T.evaluate(model,c,enc,L,classes,4)
        np.savez_compressed(dest/f'{alg}.npz',**{f'{cl}__{k}':v for cl,z in raw.items() for k,v in z.items()
                                               if 'geometry' in k or k in ('episode','t')})
        result['algorithms'][alg]={cl:{k:T.G.estimate(v.reshape(-1,4).mean(1),np.ones(len(v)//4,bool),z['episode'][::4])
                                     for k,v in z.items() if 'geometry' in k} for cl,z in raw.items()}
        (dest/'summary.json').write_text(json.dumps(result,indent=1))
        print(alg,'done',flush=True)

if __name__=='__main__':main()
