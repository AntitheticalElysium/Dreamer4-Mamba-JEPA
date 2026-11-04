"""Bounded actual-path checks; no efficacy selection and no overnight launch."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
from saccade import train as TR, model as M
L=importlib.import_module('artifacts.experiments.20261103_p1f2.train_p1f2')
C,T=L.C,L.T
HERE=Path(__file__).resolve().parent
OUT=HERE/'checks'


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(v,b[k]) for k,v in a.items())
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    if hasattr(a,'conv') and hasattr(a,'ssm'):return equal(a.conv,b.conv) and equal(a.ssm,b.ssm)
    return a==b


def optimizer_error(a,b):
    errors=[float((a['state'][i][k]-b['state'][i][k]).abs().max()) for i in a['state'] for k in ('exp_avg','exp_avg_sq')]
    assert equal(a['param_groups'],b['param_groups'])
    assert all(equal(a['state'][i]['step'],b['state'][i]['step']) for i in a['state'])
    return max(errors)


def load(name):return torch.load(OUT/name/'ckpt_02.pt',map_location='cpu',weights_only=False)


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    continuation='--continue' in sys.argv
    if OUT.exists() and not continuation:raise RuntimeError('Preserve previous checks; explicit --continue required')
    OUT.mkdir(exist_ok=continuation);torch.set_num_threads(4);start=time.time();result={'passed':False}
    cfg=L.check_recipe(L.recipe(TR.load_cfg()));small=copy.deepcopy(cfg)
    small['p1']['frames']=20480;small['p1']['monitor']['windows']=4
    # Constructor must preserve both default generators, including torch.manual_seed's CUDA side effect.
    torch.manual_seed(19);world=M.M0(cfg).cuda();a=torch.get_rng_state().clone();b=torch.cuda.get_rng_state().clone()
    con=C.Controller(world,cfg,C.config(),OUT/'init','cuda')
    assert equal(a,torch.get_rng_state()) and equal(b,torch.cuda.get_rng_state());del world,con
    result['isolated_initialization_rng']=True
    # Canonical versus identical gaze, with auxiliary training genuinely active.
    if not continuation:
        model=TR.p1(small,1,OUT/'canonical',updates=4,log_every=1);del model;torch.cuda.empty_cache()
    control=C.config();control['policy']='uniform_control'
    if not continuation:
        model=T.p1(small,1,OUT/'uniform_control',updates=4,log_every=1,controller_cfg=control);del model;torch.cuda.empty_cache()
    a,b=load('canonical'),load('uniform_control')
    keys=('model','streams','carry','cache','state','rng')
    result['canonical_uniform_parity']={k:bool(equal(a[k],b[k])) for k in keys}
    assert all(result['canonical_uniform_parity'].values())
    result['canonical_uniform_optimizer_max_abs']=optimizer_error(a['opt'],b['opt'])
    assert result['canonical_uniform_optimizer_max_abs']<1e-11
    if (OUT/'canonical_repeat/ckpt_02.pt').exists():
        r=load('canonical_repeat');result['canonical_repetition_optimizer_max_abs']=optimizer_error(a['opt'],r['opt']);del r
    assert b['fixation_controller']['updates']==4
    result['uniform_control_head_updates']=b['fixation_controller']['updates'];del a,b
    # Genuine learned gaze, exact resumed state, optimizer and emitted gaze trajectory.
    for name,updates in (('straight',(4,)),('resumed',(2,4))):
        for u in updates:
            model=T.p1(small,1,OUT/name,updates=u,log_every=1);del model;torch.cuda.empty_cache()
    a,b=load('straight'),load('resumed')
    result['learned_resume']={k:bool(equal(a[k],b[k])) for k in (*keys,'fixation_controller')}
    result['learned_resume_optimizer_max_abs']=optimizer_error(a['opt'],b['opt'])
    assert result['learned_resume_optimizer_max_abs']<1e-11
    result['learned_resume']['gaze_bytes']=(OUT/'straight/gaze.bin').read_bytes()==(OUT/'resumed/gaze.bin').read_bytes()
    result['learned_resume']['chooser_log_bytes']=(OUT/'straight/chooser.jsonl').read_bytes()==(OUT/'resumed/chooser.jsonl').read_bytes()
    assert all(result['learned_resume'].values());del a,b
    # Real launcher, exact recipe, original full chunk size. No checkpoints / science gate here.
    subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_p1f2.train_p1f2',str(OUT/'launcher'),
                    '--updates','2','--no-checkpoints'],cwd=TR.ROOT,check=True)
    for name,u,steps in (('sustained64',200,64),('phase256',3,256)):
        model=T.p1(cfg,1,OUT/name,updates=u,steps=steps,checkpoints=False,log_every=25 if steps==64 else 1)
        rows=[json.loads(x) for x in (OUT/name/'log.jsonl').read_text().splitlines()]
        aux=[json.loads(x) for x in (OUT/name/'chooser.jsonl').read_text().splitlines()]
        assert len(aux)==u and all(np.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int)))
        assert all(np.isfinite(r['head_loss']) for r in aux if r['head_loss'] is not None)
        result[name]=dict(updates=u,last=rows[-1],head_updates=aux[-1]['head_updates'],
            head_loss=aux[-1]['head_loss'],gaze_changed_fraction=float(np.mean([r['changed_fraction'] for r in aux])),
            mean_final_half_sec_per_update=float(np.mean([r['sec_per_update'] for r in rows[len(rows)//2:]])))
        if name=='sustained64':
            torch.save(dict(model=model.state_dict(),cfg=cfg,world_seed=1,state=dict(frames=u*1024)),OUT/'smoke_model.pt')
            result['monitor_uniform']=TR.monitor(model,cfg,__import__('saccade.data',fromlist=['load_split']).load_split(cfg)['monitor'])
        del model;torch.cuda.empty_cache()
    result.update(passed=True,elapsed_seconds=time.time()-start,peak_gib=torch.cuda.max_memory_allocated()/2**30)
    (HERE/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('ALL CHECKS PASSED',json.dumps(result),flush=True)


if __name__=='__main__':main()
