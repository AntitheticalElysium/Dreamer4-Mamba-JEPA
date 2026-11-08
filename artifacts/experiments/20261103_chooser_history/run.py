"""Approved one/eight-frame x coarse/belief frozen-head comparison."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch

E=importlib.import_module('artifacts.experiments.20261103_chooser_development.campaign')
C,Q=E.C,E.Q
HERE=Path(__file__).resolve().parent
ARMS={'1C':(1,'coarse'),'1B':(1,'rich'),'8C':(8,'coarse'),'8B':(8,'rich')}


def mask_history(inputs,history):
    p,ctx,act,aux=inputs
    if history==1:
        p=p.clone();ctx=ctx.clone()
        p[:,:-1]=0;ctx.view(-1,8,29)[:,:-1]=0
    return p,ctx,act,aux


class Data(E.AuxData):
    def __init__(self,*args,history=8):
        super().__init__(*args);self.history=history

    def inputs(self,ix,action=None):
        return mask_history(super().inputs(ix,action),self.history)


def compact(head):
    """Exact information/weights, omit multiplication of seven zero slots."""
    h=deepcopy(head)
    for old,parent,key,width in [(h.local,h,'local',16),(h.context[0],h.context,0,29)]:
        new=torch.nn.Linear(width,old.out_features,bias=old.bias is not None).to(old.weight.device)
        with torch.no_grad():
            new.weight.copy_(old.weight[:,-width:])
            if old.bias is not None:new.bias.copy_(old.bias)
        if isinstance(key,str):setattr(parent,key,new)
        else:parent[key]=new
    return h.eval().requires_grad_(False)


def tensor_sha(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def weight_sha(h):
    return hashlib.sha256(''.join(k+tensor_sha(v) for k,v in h.state_dict().items()).encode()).hexdigest()


def prepare(kind,device):
    tr=E.load_cache(10,'train',device);dv=E.load_cache(10,'dev',device)
    if kind=='rich':
        aux=[torch.load(C.HERE/f'{s}_preaction_belief.pt',map_location='cpu',weights_only=False)['aux'] for s in ('train','dev')]
    else:aux=[E.sketch(z) for z in (tr,dv)]
    mean,std=E.aux_stats(aux[0],tr)
    return tr,dv,aux,mean,std


@torch.no_grad()
def snapshot(h,data,stem):
    for cohort,meta in C.cohorts().items():
        ids=data.row_ids(meta['ep'],meta['t'])
        pred,taken,no=E.predict(h,data,ids)
        actual=data.z['p'][ids].float();previous=data.z['p'][ids-1].float()
        ar=torch.arange(len(ids),device=data.device)
        cell=torch.tensor(np.maximum(meta['counter'],0),device=data.device)
        error=(taken-actual).square().mean(-1)
        cc=(taken-no).square().sum(-1)
        errors=dict(counter_mse=error[ar,cell].cpu().numpy(),
                    counter_copy_mse=(previous[ar,cell]-actual[ar,cell]).square().mean(-1).cpu().numpy(),
                    counter_C=cc[ar,cell].cpu().numpy(),
                    counter_target_step=(previous[ar,cell]-actual[ar,cell]).square().sum(-1).cpu().numpy(),
                    cell_mse=error.cpu().numpy(),cell_C=cc.cpu().numpy())
        assert (cc[data.z['prev_action'][ids]==0]==0).all()
        np.savez_compressed(HERE/f'{stem}_{cohort}.npz',**pred,**meta,**errors)


def smoke(device):
    torch.manual_seed(611850)
    h=E.AuxHead().to(device)
    # Random nonzero output layer makes input-exclusion checks non-vacuous.
    torch.nn.init.normal_(h.output.weight,std=.02)
    inp=(torch.randn(3,8,81,16,device=device),torch.randn(3,8*29,device=device),
         torch.tensor([0,1,5],device=device),torch.randn(3,256,device=device))
    one=mask_history(inp,1)
    changed=tuple(x.clone() for x in inp)
    changed[0][:,:-1]+=99;changed[1].view(3,8,29)[:,:-1]-=17
    assert torch.equal(h(one),h(mask_history(changed,1)))
    assert one[0][:,:-1].count_nonzero()==0 and one[1].view(3,8,29)[:,:-1].count_nonzero()==0
    assert torch.equal(one[0][:,-1],inp[0][:,-1])
    folded=compact(h)
    short=(one[0][:,-1:],one[1].view(3,8,29)[:,-1],one[2],one[3])
    err_tf32=float((h(one)-folded(short)).abs().max())
    # Changing GEMM shape also changes TF32's rounding. Verify the algebra with
    # IEEE arithmetic, and retain the TF32 discrepancy instead of widening a gate.
    saved_tf32=torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32=False
    err=float((h(one)-folded(short)).abs().max())
    torch.backends.cuda.matmul.allow_tf32=saved_tf32
    assert err<2e-6,err
    # Four arms use precisely the same initialized weights and sampling streams.
    starts=[];streams=[]
    for arm in ARMS:
        hh,oo,rr=E.init_fit(611601,device,aux=True)
        starts.append(weight_sha(hh));streams.append(tensor_sha(rr.get_state()))
    assert len(set(starts))==len(set(streams))==1
    # Actual cache: verify future exclusion and reset/padding under both histories.
    tr,dv,aux,mean,std=prepare('coarse',device)
    data=Data(tr,device,aux[0],mean,std,history=1)
    ids=torch.where(data.z['t']>8)[0][::32][:16]
    before=tuple(x.clone() for x in data.inputs(ids))
    target=data.z['p'][ids].clone();data.z['p'][ids]+=1
    after=data.inputs(ids)
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    data.z['p'][ids]=target
    assert all(not x.requires_grad for x in before)
    return dict(masked_history_excluded=True,all_context_channels_masked=True,
                nonzero_head_compact_max_abs_ieee=err,compact_tf32_shape_rounding_max_abs=err_tf32,
                identical_initialization_and_sampling=True,
                current_target_excluded=True,parameter_count=sum(p.numel() for p in h.parameters()))


def fit(arm,seed,device):
    start=time.time();history,kind=ARMS[arm]
    tr,dv,aux,mean,std=prepare(kind,device)
    train=Data(tr,device,aux[0],mean,std,history=history)
    dev=Data(dv,device,aux[1],mean,std,history=history)
    h,opt,rng=E.init_fit(seed,device,aux=True)
    init_hash=weight_sha(h);rng0=tensor_sha(rng.get_state())
    stem=f'{arm}_{seed}'
    def save(step):
        rs=rng.get_state().clone();gs=torch.cuda.get_rng_state().clone()
        snapshot(h,dev,f'{stem}_u{step:05d}')
        assert torch.equal(rs,rng.get_state()) and torch.equal(gs,torch.cuda.get_rng_state())
        torch.save(dict(state={k:v.detach().cpu() for k,v in h.state_dict().items()},
                        optimizer=opt.state_dict(),rng=rs,mean=mean.cpu(),std=std.cpu(),arm=arm,seed=seed,
                        updates=step,total_updates=12000,checkpoint_sha256=tr['checkpoint_sha256']),
                   HERE/f'{stem}_u{step:05d}.pt')
        print('snapshot',arm,seed,step,'elapsed',round(time.time()-start,1),flush=True)
    save(0)
    curve=E.advance(h,opt,rng,train,0,12000,12000,
                    lambda step:save(step) if step in (1000,4000,8000,12000) else None)
    identity=dict(arm=arm,seed=seed,history=history,auxiliary=kind,updates=12000,
                  initial_weight_sha256=init_hash,initial_batch_rng_sha256=rng0,
                  final_batch_rng_sha256=tensor_sha(rng.get_state()),
                  train_rows=len(E.split_rows(train)[0]),validation_rows=len(E.split_rows(train)[1]),
                  checkpoint_sha256=tr['checkpoint_sha256'],curve=curve,seconds=time.time()-start,
                  parameters=sum(p.numel() for p in h.parameters()),peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    C.write(HERE/f'{stem}_identity.json',identity)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','fit']);p.add_argument('arm',nargs='?',choices=ARMS)
    p.add_argument('seed',nargs='?',type=int);args=p.parse_args()
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    if args.phase=='smoke':C.write(HERE/'SMOKE.json',smoke(device))
    else:fit(args.arm,args.seed,device)


if __name__=='__main__':main()
