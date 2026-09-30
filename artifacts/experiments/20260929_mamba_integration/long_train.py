"""Source-bound, optimizer-resumed length-64 fmamba+corrg continuation.

All arms use the same TRAIN-only 64-frame ledger and long-phase sampler seed.
The LDAD1 reset6 control differs only in resetting Mamba temporal state every six frames.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
import short_train as SHORT
import spatial as S
from long_world import LongTWorld, rollout_loss
from d4mj.config import config_from_dict
from d4mj.train import _phase_lr, autocast_context, optimizer_step, phase_optimizer

SHORT_OUT=ROOT/'artifacts/eda/levers_mamba_integration_v1'
POOLS=ROOT/'artifacts/eda/levers_mamba_long_pools_v1'
OUT=ROOT/'artifacts/eda/levers_mamba_long_worlds_v1'
SOURCE_FILES=[HERE/'long_train.py',HERE/'long_world.py',
              ROOT/'artifacts/experiments/20260927_levers/tworld.py',
              ROOT/'artifacts/experiments/20260921_readout_ladder/spatial.py',
              ROOT/'d4mj/train.py',ROOT/'d4mj/mamba_recurrence.py']


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def atomic_save(payload,path):
    temp=path.with_suffix(path.suffix+'.tmp')
    torch.save(payload,temp)
    os.replace(temp,path)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--arm',required=True,choices=('raw','ldad1','ldad10'))
    p.add_argument('--mode',required=True,choices=('full','reset6'))
    p.add_argument('--batch',required=True,type=int)
    p.add_argument('--total',required=True,type=int)
    p.add_argument('--until',type=int,default=None)
    p.add_argument('--out',type=Path,default=OUT)
    a=p.parse_args()
    if a.mode=='reset6' and a.arm!='ldad1':
        p.error('reset6 is predeclared only for the LDAD1 temporal control')
    assert a.batch>0 and a.total>0 and 0<(a.until or a.total)<=a.total
    torch.backends.cuda.matmul.allow_tf32=False

    short_name=f'int_corrg_{a.arm}_suffix_s7_fmamba_u18000'
    short_resume=SHORT_OUT/f'{short_name}.resume.pt'
    short_final=SHORT_OUT/f'{short_name}.pt'
    short= torch.load(short_resume,map_location='cpu',weights_only=False)
    final= torch.load(short_final,map_location='cpu',weights_only=False)
    assert short['update']==18000 and short['contract']['total_updates']==18000
    assert short['contract']==final['contract']
    for source,expected in short['contract']['source_sha256'].items():
        assert digest(ROOT/source)==expected,f'short-lineage source drift: {source}'
    assert short['world'].keys()==final['world'].keys()
    assert all(torch.equal(short['world'][k],final['world'][k]) for k in short['world'])

    pool_dir=POOLS/a.arm
    manifest_path=pool_dir/'manifest.json'
    manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='complete'
    assert manifest['row_count']==19789 and manifest['terminal_count']==7501
    assert digest(pool_dir/'tokens.f16')==manifest['tokens_sha256']
    assert digest(pool_dir/'labels.pt')==manifest['labels_sha256']
    assert manifest['contract']['ledger_sha256']==json.loads((HERE/'long_ledger.json').read_text())['ledger_sha256']
    shape=tuple(manifest['contract']['shape'])
    assert shape==(19789,64,81,192)
    assert manifest['contract']['dtype']=='float16'
    labels=torch.load(pool_dir/'labels.pt',map_location='cpu',weights_only=False)
    assert labels['actions'].shape==(shape[0],63)
    assert bool(((labels['actions']>=0)&(labels['actions']<17)).all())
    assert int(labels['terminal'].sum())==7501
    tokens=np.memmap(pool_dir/'tokens.f16',mode='r',dtype=np.float16,shape=shape)

    config=config_from_dict(torch.load(S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    contract={
        'name':f'long_corrg_{a.arm}_{a.mode}_b{a.batch}_u{a.total}_s7',
        'arm':a.arm,'mode':a.mode,'batch':a.batch,'total':a.total,
        'short_resume_sha256':digest(short_resume),'short_final_sha256':digest(short_final),
        'pool_manifest_sha256':digest(manifest_path),'pool_tokens_sha256':manifest['tokens_sha256'],
        'pool_labels_sha256':manifest['labels_sha256'],'ledger_sha256':manifest['contract']['ledger_sha256'],
        'config_checkpoint_sha256':digest(S.CHECKPOINT),
        'source_sha256':{str(q.relative_to(ROOT)):digest(q) for q in SOURCE_FILES},
        'sampler_seed':111,'objective':'teacher_63_plus_generated_final_2',
        'lr_schedule':'canonical_phase_absolute_update_18000_plus_long_update',
        'data_split':'train_only',
    }
    a.out.mkdir(parents=True,exist_ok=True)
    resume=a.out/f"{contract['name']}.resume.pt"
    final_path=a.out/f"{contract['name']}.pt"
    if final_path.exists():
        raise SystemExit(f'completed final exists: {final_path}; refusing overwrite')

    device=torch.device('cuda')
    world=LongTWorld(reset_every=(6 if a.mode=='reset6' else None)).to(device)
    world.load_state_dict(short['world'])
    opt=phase_optimizer([world],config)
    opt.load_state_dict(short['optimizer'])
    params=[q for g in opt.param_groups for q in g['params']]
    order=torch.Generator().manual_seed(111)
    history=[]
    start=0
    if resume.exists():
        saved=torch.load(resume,map_location='cpu',weights_only=False)
        assert saved['contract']==contract,'long continuation contract drift'
        world.load_state_dict(saved['world'])
        opt.load_state_dict(saved['optimizer'])
        order.set_state(saved['order_rng'])
        torch.set_rng_state(saved['cpu_rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
        history,start=saved['history'],saved['update']
        print(json.dumps({'stage':'resume','name':contract['name'],'update':start}),flush=True)
    else:
        torch.set_rng_state(short['cpu_rng'])
        torch.cuda.set_rng_state_all(short['cuda_rng'])
        print(json.dumps({'stage':'init','contract':contract,
                          'parameters':sum(q.numel() for q in world.parameters())}),flush=True)
    stop=a.until or a.total
    assert start<=stop
    began=time.time()
    world.train()
    for update in range(start,stop):
        ids=torch.randint(shape[0],(a.batch,),generator=order)
        batch=torch.from_numpy(np.asarray(tokens[ids.numpy()]).copy()).to(device)
        actions=labels['actions'][ids].to(device)
        with autocast_context(config):
            objective=rollout_loss(world,batch,actions)
        norm=optimizer_step(opt,objective,params,learning_rate=_phase_lr(config,18000+update),
                            grad_clip=config.agent.grad_clip,strict=True,zero_grad=True)
        step=update+1
        if step%100==0 or step==stop:
            row={'update':step,'objective':float(objective.detach()),
                 'gradient_norm':float(norm),'seconds_since_resume':round(time.time()-began,1),
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated()}
            history.append(row)
            print(json.dumps({'stage':'train','name':contract['name'],**row}),flush=True)
            atomic_save({'world':world.state_dict(),'optimizer':opt.state_dict(),
                         'order_rng':order.get_state(),'cpu_rng':torch.get_rng_state(),
                         'cuda_rng':torch.cuda.get_rng_state_all(),
                         'history':history,'update':step,'contract':contract},resume)
        if step in {round(a.total/4),round(a.total/2),a.total}:
            snapshot=final_path if step==a.total else a.out/f"{contract['name']}_at{step}.pt"
            assert not snapshot.exists(),f'snapshot exists: {snapshot}'
            atomic_save({'world':world.state_dict(),'history':history,'contract':contract},snapshot)
            print(json.dumps({'stage':'snapshot','name':contract['name'],'update':step,
                              'sha256':digest(snapshot)}),flush=True)
    print(json.dumps({'stage':'complete' if stop==a.total else 'paused',
                      'name':contract['name'],'update':stop,
                      'seconds_since_resume':round(time.time()-began,1)}),flush=True)


if __name__=='__main__':
    main()
