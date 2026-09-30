"""Equal-capacity within-root damage ranking from U-world h versus generated U.

Fit 55k, choose step on 56k, refit 55k+56k, judge 57k+58k. All-action
fork labels, previously inspected roots; diagnostic readout only, not model
training or a fresh gate. This guards against mistaking a linear-probe drop
for lost decision information.
"""
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
from frozen_ladder import strata

def data(block,key):
    d=ROOT/'artifacts/eda/diagnosis_dump_v1'
    meta=torch.load(d/f'{block}_meta.pt',weights_only=False)
    world=torch.load(d/f'{block}_U.pt',weights_only=False)
    z=strata(meta['visible'])['zombie_adjacent']
    if key=='root_action':
        root=world['root'][z].float()
        act=F.one_hot(torch.arange(17),17).float()[None].expand(len(root),-1,-1)
        x=torch.cat([root[:,None].expand(-1,17,-1),act],-1)
    else:
        x=world[key][z].float()
    y=(meta['health_delta'][z]<0).float()
    return x,y

def model(dim,seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(dim,256),nn.LayerNorm(256),nn.GELU(),
                         nn.Linear(256,128),nn.GELU(),nn.Linear(128,1))

def metric(logits,y):
    diff=logits[:,:,None]-logits[:,None,:]
    valid=(y[:,:,None]>y[:,None,:])
    auc=float((diff[valid]>0).float().mean()) if valid.any() else float('nan')
    choice=logits.argmin(1)
    safe=float((1-y.gather(1,choice[:,None]).squeeze(1)).mean())
    return auc,safe,int(valid.sum())

def train(x,y,seed,steps,dev=None):
    net=model(x.shape[-1],seed)
    opt=torch.optim.AdamW(net.parameters(),lr=3e-4,weight_decay=.01)
    order=torch.Generator().manual_seed(100+seed)
    best=(-1.,0)
    for step in range(1,steps+1):
        rows=torch.randint(len(x),(128,),generator=order)
        out=net(x[rows]).squeeze(-1);lab=y[rows]
        dif=out[:,:,None]-out[:,None,:]
        valid=(lab[:,:,None]>lab[:,None,:])
        loss=F.softplus(-dif[valid]).mean()
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if dev is not None and step%100==0:
            with torch.no_grad():auc,_,_=metric(net(dev[0]).squeeze(-1),dev[1])
            if auc>best[0]:best=(auc,step)
    return net.eval(),best

def main():
    torch.set_num_threads(6)
    out={}
    for key in ('root_action','h','gen'):
        tr=data('55k',key);dv=data('56k',key)
        te=[data(b,key) for b in ('57k','58k')]
        xf=torch.cat([tr[0],dv[0]]);yf=torch.cat([tr[1],dv[1]])
        xt=torch.cat([v[0] for v in te]);yt=torch.cat([v[1] for v in te])
        mu,sd=tr[0].flatten(0,1).mean(0),tr[0].flatten(0,1).std(0).clamp_min(1e-6)
        ztr=(tr[0]-mu)/sd;zdv=(dv[0]-mu)/sd
        mu,sd=xf.flatten(0,1).mean(0),xf.flatten(0,1).std(0).clamp_min(1e-6)
        zf,zt=(xf-mu)/sd,(xt-mu)/sd
        runs=[]
        for seed in (1,2,3):
            _,best=train(ztr,tr[1],seed,1200,dev=(zdv,dv[1]))
            net,_=train(zf,yf,seed,best[1])
            with torch.no_grad():test=metric(net(zt).squeeze(-1),yt)
            runs.append({'seed':seed,'chosen_step':best[1],'dev_auc':best[0],
                         'test_within_root_auc':test[0],'test_safe_choice':test[1]})
            print(key,runs[-1],flush=True)
        out[key]={'fit_roots':len(xf),'test_roots':len(xt),'test_pairs':metric(torch.zeros(len(yt),17),yt)[2],
                  'runs':runs,'mean_auc':sum(r['test_within_root_auc'] for r in runs)/3,
                  'mean_safe_choice':sum(r['test_safe_choice'] for r in runs)/3}
    (HERE/'damage_rank_h_vs_g.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:(v['mean_auc'],v['mean_safe_choice']) for k,v in out.items()}),flush=True)

if __name__=='__main__':main()
