"""Bounded pre-launch trainer, both chunk phases, resume, source/config and evaluation checks."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import copy
import importlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import torch
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

L=importlib.import_module('artifacts.experiments.20261102_p1f16.train_p1f16')
HERE=Path(__file__).resolve().parent
SCRATCH=HERE/'checks'


def save(result):
    (HERE/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


def finite(records):
    return all(math.isfinite(v) for r in records for k,v in r.items() if isinstance(v,(int,float)))


def maximum(a,b):
    return max(float((a[k].float()-b[k].float()).abs().max()) for k in a)


@torch.no_grad()
def scales(model,cfg):
    split=D.load_split(cfg);ep=split['train'][0]
    frames=D.RP.episode_fields(ep.shard,ep.slot)['observations'][:64].to('cuda')
    fix=torch.ones(len(frames),2,dtype=torch.long,device='cuda')*4
    with TR.autocast(torch.device('cuda'),cfg):
        P=model.perception(frames,fix);joint=model.belief.spatial_fusion(P,M.window(fix))
    return dict(raw_fine_rms=float(P.g.float().square().mean().sqrt()),
        joint_rms=float(joint.float().square().mean().sqrt()),joint_max=float(joint.abs().max()),
        embedding_rms=float(model.belief.percept_embed(P).float().square().mean().sqrt()))


def main():
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    torch.set_num_threads(4);SCRATCH.mkdir(exist_ok=True)
    result={};cfg=L.check_recipe(L.recipe(TR.load_cfg()));start=time.time()
    # Real launcher identity/config path, separate from sustained in-process trainer smoke.
    out=SCRATCH/'launcher'
    if out.exists():raise RuntimeError('Use a fresh checks directory; preserve previous evidence')
    subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261102_p1f16.train_p1f16',str(out),
                    '--updates','2','--no-checkpoints'],check=True,cwd=TR.ROOT)
    result['launcher']=dict(identity=json.loads((out/'identity.json').read_text()),
        log=[json.loads(x) for x in (out/'log.jsonl').read_text().splitlines()])
    torch.manual_seed(1);initial=M.M0(cfg).cuda();result['initial_scale']=scales(initial,cfg);del initial
    torch.cuda.empty_cache()
    model=TR.p1(cfg,1,SCRATCH/'sustained64',updates=200,checkpoints=False,log_every=25)
    rows=[json.loads(x) for x in (SCRATCH/'sustained64/log.jsonl').read_text().splitlines()]
    assert finite(rows)
    result['sustained64']=dict(records=rows,all_finite=True,updates=200,final_scale=scales(model,cfg))
    result['sustained64']['full_monitor']=TR.monitor(model,cfg,D.load_split(cfg)['monitor'])
    # Probe real future-G1 gradients after the untouched production objective has trained the denoiser.
    streams=D.Streams(D.load_split(cfg)['train'][:16],2,8,seed=1)
    terms,_,_=TR.p1_losses(model,streams.next(8),model.belief.initial(2,'cuda'),torch.zeros(2,9,256,device='cuda'),None,cfg)
    model.zero_grad(set_to_none=True);terms['g1'].backward()
    grads={k:float(p.grad.norm()) for k,p in model.belief.spatial_fusion.named_parameters()}
    assert all(math.isfinite(v) and v>0 for v in grads.values())
    result['future_g1_fusion_gradient_norms']=grads
    torch.save(dict(model=model.state_dict(),cfg=cfg,world_seed=1,state=dict(frames=204800)),SCRATCH/'smoke_model.pt')
    # S1 encoder and actual cascade on the trained smoke checkpoint, no stub sampling.
    model.eval();labels=S.first_episodes(dict(__import__('numpy').load(S.LABELS/'s0_dev_labels.npz')),2)
    enc=S.encode(model,labels,cfg)
    e=0;t=9;take=lambda k,dt:enc[e][k][t+dt].to('cuda')
    cache=model.denoiser.new_cache();gen=torch.Generator(device='cuda').manual_seed(611950)
    with torch.no_grad():
        for dt in range(-8,0):
            cache=M.context_pass(model.denoiser,tuple(take(k,dt).float()[None,None] for k in ('p','g','v')),
                take('prev_action',dt).reshape(1,1),model.belief.memory(take('bp',dt).float()[None])[:,None],
                take('fix',dt)[None,None],cache,cfg,gen)
        out=M.sample(model.denoiser,[tuple(enc[e][k].shape[1:]) for k in ('p','g','v')],
            take('prev_action',0).reshape(1,1),model.belief.memory(take('bp',0).float()[None])[:,None],
            take('fix',0)[None,None],cache,cfg,gen)
        assert all(torch.isfinite(x).all() for x in out)
    result['evaluation_model_smoke']=dict(encoded_episodes=len(enc),cascade_shapes=[list(x.shape) for x in out],all_finite=True)
    del model,enc,terms;torch.cuda.empty_cache();save(result)
    # Small frame/monitor counts only for checkpoint mechanics; no scientific readings.
    small=copy.deepcopy(cfg);small['p1']['frames']=20480;small['p1']['monitor']['windows']=8
    ck={}
    for name,plan in (('straight',(4,)),('resumed',(2,4))):
        for u in plan:
            m=TR.p1(small,1,SCRATCH/name,updates=u,checkpoints=True,log_every=1);del m;torch.cuda.empty_cache()
        ck[name]=torch.load(SCRATCH/name/'ckpt_02.pt',map_location='cpu',weights_only=False)
    a,b=ck['straight'],ck['resumed']
    carry=max(float((getattr(x,f)-getattr(y,f)).abs().max()) for x,y in zip(a['carry'],b['carry']) for f in ('conv','ssm'))
    result['resume']=dict(model_max_abs=maximum(a['model'],b['model']),carry_max_abs=carry,
        cache_max_abs=float((a['cache']-b['cache']).abs().max()),state_equal=a['state']==b['state'],
        cpu_rng_equal=torch.equal(a['rng']['cpu'],b['rng']['cpu']),cuda_rng_equal=torch.equal(a['rng']['cuda'],b['rng']['cuda']))
    assert result['resume']['model_max_abs']==0 and carry==0 and result['resume']['cache_max_abs']==0
    assert all(result['resume'][k] for k in ('state_equal','cpu_rng_equal','cuda_rng_equal'))
    wrong=copy.deepcopy(small);wrong['belief'].pop('spatial_fusion')
    try:TR.validate_resume(a,wrong,1,TR.run_provenance(wrong));raise AssertionError('wrong architecture accepted')
    except ValueError:result['resume']['rejects_missing_fusion']=True
    del a,b,ck
    m=TR.p1(small,1,SCRATCH/'phase256',updates=3,steps=256,checkpoints=True,log_every=1)
    phase=[json.loads(x) for x in (SCRATCH/'phase256/log.jsonl').read_text().splitlines()]
    assert finite(phase)
    result['phase256']=dict(records=phase,all_finite=True)
    del m;torch.cuda.empty_cache()
    # Check actual full old checkpoint loading and new CPU reload.
    old=torch.load(L.CONTROL,map_location='cpu',weights_only=False);legacy=M.M0(old['cfg']);legacy.load_state_dict(old['model']);del legacy,old
    smoke=torch.load(SCRATCH/'smoke_model.pt',map_location='cpu',weights_only=False);reload=M.M0(smoke['cfg']);reload.load_state_dict(smoke['model'])
    with torch.no_grad():
        P=reload.perception(torch.randint(0,256,(2,63,63,3),dtype=torch.uint8),torch.tensor([[4,4],[2,3]]))
        assert torch.isfinite(reload.belief.percept_embed(P)).all()
    result['reload']=dict(old_control_strict=True,new_cpu_strict=True)
    result['elapsed_seconds']=time.time()-start
    result['max_allocated_GiB']=torch.cuda.max_memory_allocated()/2**30
    result['max_reserved_GiB']=torch.cuda.max_memory_reserved()/2**30
    save(result);print('CHECKS COMPLETE',json.dumps({k:v for k,v in result.items() if k not in ('sustained64','phase256','launcher')}),flush=True)


if __name__=='__main__':main()
