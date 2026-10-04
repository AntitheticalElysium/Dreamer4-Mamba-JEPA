"""Exploratory physical-label probe: is visible zombie adjacency decodable from z4?

Fixed before run. Encode FIT-train roots and the previously inspected 54k roots with
the frozen canonical Raw H2 encoder. Label zombie adjacency from the visible
simulator state rendered at the root, not from outcomes. Split FIT-train seeds
85/15 for fit/selection; judge 54k once. Fit three 2-layer MLP seeds on z1 and
z4, and a position-aware patch-token attention positive control, each with
balanced BCE and fixed 1500 steps. Select by inner-FIT AUC every 100 steps.
Primary: held-out 54k AUC of z4 for visible zombie adjacency. This only tests
physical presence; it does not bound action-conditioned death or policy choice.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).parent
OLD=ROOT/'artifacts/experiments/20260921_readout_ladder'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(OLD))
from boundary import judge_store
from confirm import seeds_for
from frozen_ladder import strata
from observability import load
from d4mj.data import _sha256
from d4mj.experiments import _load_bridge_parent
from d4mj.lewm_diagnostics import FORK_STORE

CHECKPOINT=ROOT/'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt'
STORE=ROOT/'artifacts/eda/observe_fresh_v5'
SEEDS=(7,11,19)
STEPS=1500


@torch.no_grad()
def encode(encoder,frames,device):
    out={'z1':[],'z4':[],'tokens':[]}
    for i in range(0,len(frames),64):
        f=frames[i:i+64,-4:].to(device)
        z,_,tokens,_,_=encoder._hidden(f)
        n=len(f)
        z=z.reshape(n,4,-1).float().cpu()
        t=tokens.reshape(n,4,81,-1)[:,-1].float().cpu()
        out['z1'].append(z[:,-1]);out['z4'].append(z.flatten(1));out['tokens'].append(t)
    return {k:torch.cat(v) for k,v in out.items()}


def auc(score,label):
    pos=score[label.bool()].float().cpu()
    neg=score[~label.bool()].float().cpu().sort().values
    less=torch.searchsorted(neg,pos,right=False).float()
    leq=torch.searchsorted(neg,pos,right=True).float()
    return float((less+.5*(leq-less)).mean()/len(neg))


class VectorHead(nn.Module):
    def __init__(self,dim):
        super().__init__();self.net=nn.Sequential(nn.Linear(dim,512),nn.ReLU(),nn.Linear(512,1))
    def forward(self,x):return self.net(x).squeeze(-1)


class TokenHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.position=nn.Parameter(torch.zeros(81,192))
        self.value=nn.Linear(192,128);self.score=nn.Linear(192,1)
        self.net=nn.Sequential(nn.GELU(),nn.Linear(128,512),nn.ReLU(),nn.Linear(512,1))
    def forward(self,x):
        x=x+self.position
        pooled=(self.value(x)*self.score(x).softmax(1)).sum(1)
        return self.net(pooled).squeeze(-1)


def fit(name,x,y,train_rows,hold_rows,xj,yj,seed,device):
    torch.manual_seed(seed)
    train=x[train_rows]
    mean=train.reshape(-1,train.shape[-1]).mean(0)
    std=train.reshape(-1,train.shape[-1]).std(0).clamp_min(1e-5)
    x=((x-mean)/std).float();xj=((xj-mean)/std).float()
    model=(TokenHead() if name=='tokens' else VectorHead(x.shape[-1])).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=torch.Generator().manual_seed(seed+100)
    pos=float(y[train_rows].mean())
    weight=(1-pos)/max(pos,1e-6)
    best=-1.0;state=None;step_best=None
    for step in range(1,STEPS+1):
        ids=train_rows[torch.randint(len(train_rows),(256,),generator=rng)]
        pred=model(x[ids].to(device))
        loss=F.binary_cross_entropy_with_logits(pred,y[ids].to(device),pos_weight=torch.tensor(weight,device=device))
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step%100==0:
            model.eval()
            with torch.no_grad():
                h=torch.cat([model(x[b].to(device)).float().cpu() for b in hold_rows.split(512)])
            value=auc(h,y[hold_rows])
            if value>best:best=value;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};step_best=step
            model.train()
    model.load_state_dict(state);model.eval()
    with torch.no_grad():
        score=torch.cat([model(xj[b].to(device)).float().cpu() for b in torch.arange(len(xj)).split(512)])
    return {'seed':seed,'selected_step':step_best,'fit_inner_auc':best,'judge_auc':auc(score,yj),
            'parameters':sum(p.numel() for p in model.parameters())}


def main():
    start=time.time();os.chdir(ROOT)
    assert _sha256(CHECKPOINT)==json.loads((OLD/'evidence/confirm.json').read_text())['checkpoint_sha256']
    partition=json.loads((OLD/'evidence/root_partition.json').read_text())
    fit_seeds,_=seeds_for(partition,FORK_STORE)
    fit_data=load(fit_seeds)['fit']
    judge,manifest,files=judge_store(STORE)
    bundle,_,_=_load_bridge_parent(CHECKPOINT)
    bundle.encoder.freeze()
    fit_x=encode(bundle.encoder,fit_data['frames'],bundle.device)
    judge_x=encode(bundle.encoder,judge['frames'],bundle.device)
    del bundle
    y=strata(fit_data['visible'])['zombie_adjacent'].float()
    yj=strata(judge['visible'])['zombie_adjacent'].float()
    seeds=fit_data['seed']
    unique=seeds.unique().sort().values
    perm=torch.randperm(len(unique),generator=torch.Generator().manual_seed(20261009))
    hold=set(unique[perm[:round(.15*len(unique))]].tolist())
    hold_mask=torch.tensor([int(s) in hold for s in seeds.tolist()])
    tr=torch.where(~hold_mask)[0];va=torch.where(hold_mask)[0]
    print(json.dumps({'stage':'features','fit':len(y),'judge':len(yj),'fit_positive':float(y.mean()),
                      'judge_positive':float(yj.mean()),'seconds':round(time.time()-start,1)}),flush=True)
    result={'status':'exploratory_preinspected_54k','script_sha256':_sha256(Path(__file__)),
            'checkpoint_sha256':_sha256(CHECKPOINT),'judge_manifest':manifest,
            'rows':{'fit':len(y),'inner':len(va),'judge':len(yj)},
            'prevalence':{'fit':float(y.mean()),'judge':float(yj.mean())},'arms':{}}
    device=torch.device('cuda')
    for name in ('z1','z4','tokens'):
        result['arms'][name]=[]
        for seed in SEEDS:
            row=fit(name,fit_x[name],y,tr,va,judge_x[name],yj,seed,device)
            result['arms'][name].append(row)
            print(json.dumps({'stage':'arm','arm':name,**row,'seconds':round(time.time()-start,1)}),flush=True)
    result['seconds']=round(time.time()-start,1)
    (HERE/'hazard_presence.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'stage':'done','seconds':result['seconds']}),flush=True)


if __name__=='__main__':main()
