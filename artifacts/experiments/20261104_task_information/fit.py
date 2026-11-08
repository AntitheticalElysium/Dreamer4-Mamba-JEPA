"""Matched conditional semantic readers; full raw coarse context and native TSP in every baseline."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import json,sys,time,gc
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];DEV=torch.device('cuda')
MODES=['coarse','fine','pixels'];SHA='34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045'

def load(split):
    a=torch.load(ROOT/f'artifacts/experiments/20261103_fusion_retention/p1f16/{split}.pt',map_location='cpu',weights_only=False,mmap=True)
    assert a['checkpoint_sha256']==SHA
    keys=['p','g','v','prior','action','prev_action','reward','terminated','fix','truth','episode','t','hist','localg','night']
    z={k:a['current'][k] for k in keys}
    raw=torch.load(HERE/f'{split}_raw.pt',map_location='cpu',weights_only=False,mmap=True)
    tsp=torch.load(HERE/f'{split}_tsp.pt',map_location='cpu',weights_only=False,mmap=True)
    assert raw['checkpoint_sha256']==tsp['checkpoint_sha256']==SHA
    z.update(raw=raw['raw'],fine_pixels=raw['fine_pixels'],heading=raw['heading'],tsp=tsp['mean'])
    return z

class Features:
    def __init__(self,z):
        self.z={k:v.to(DEV) for k,v in z.items()};z=self.z;self.n=len(z['p'])
        self.identity=torch.eye(81,device=DEV);self.eye9=torch.eye(9,device=DEV)
        self.history=torch.cat([z['p'].flatten(1).float(),z['v'].float(),z['prior'].float(),F.one_hot(z['prev_action'].long()+1,18).float(),
            torch.sign(z['reward'].float())[:,None]*torch.log1p(z['reward'].float().abs())[:,None],z['terminated'].float()[:,None],torch.ones(self.n,1,device=DEV)],-1).half()
        self.history=torch.cat([self.history,torch.zeros_like(self.history[:1])])
        self.raw=torch.cat([z['raw'].flatten(1),torch.full_like(z['raw'][:1].flatten(1),128)])
        self.hist=z['hist'].clone();self.hist[self.hist<0]=self.n
        self.queries=torch.where((z['localg']>=0).flatten())[0]
        self.dim=self.make(self.queries[:2],'coarse')[0].shape[-1]
    def make(self,ix,mode):
        z=self.z;f=ix//63;c=ix%63;slot=z['localg'][f,c].long();assert (slot>=0).all()
        base=torch.cat([z['p'][f].flatten(1).float(),z['v'][f].float(),z['prior'][f].float(),self.history[self.hist[f]].flatten(1).float(),
            z['raw'][f].flatten(1).float()/127.5-1,self.raw[self.hist[f]].flatten(1).float()/127.5-1,
            z['p'][f,c].float(),z['raw'][f,c].float()/127.5-1,z['tsp'][f,slot].float(),self.identity[c],self.eye9[slot],
            z['fix'][f].float()/8,F.one_hot(z['action'][f].long(),17).float(),F.one_hot(z['prev_action'][f].long()+1,18).float(),
            torch.sign(z['reward'][f].float())[:,None]*torch.log1p(z['reward'][f].float().abs())[:,None],z['terminated'][f].float()[:,None]],-1)
        extra=torch.zeros(len(ix),1323,device=DEV)
        if mode=='fine':extra[:,:576]=z['g'][f].flatten(1).float()
        elif mode=='pixels':extra=z['fine_pixels'][f].flatten(1).float()/127.5-1
        elif mode=='coarse_tsp':extra=z['tsp'][f].flatten(1).float()
        local=torch.cat([z['p'][f,c].float(),z['raw'][f,c].float()/127.5-1,z['tsp'][f,slot].float()],-1)
        return base,extra,local,c
    def pops(self):
        z=self.z;q=self.queries;f=q//63;c=q%63;train=z['episode'][f]<360;y=z['truth'][f,c].long();h=z['heading'][f,c].long()
        return [q[train&(y==k)] for k in range(5)],[q[train&(h==k)] for k in range(4)]

class Reader(nn.Module):
    def __init__(self,dim,family):
        super().__init__();self.family=family
        if family=='flat':self.net=nn.Sequential(nn.Linear(dim+1323,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,9))
        else:
            self.base=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
            self.local=nn.Sequential(nn.Linear(190,128),nn.GELU())
            self.extra=nn.Sequential(nn.Linear(1323,128),nn.GELU())
            self.gate=nn.Embedding(81,128)
            self.net=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,9))
    def forward(self,x):
        b,s,l,c=x
        if self.family=='flat':return self.net(torch.cat([b,s],-1))
        return self.net(torch.cat([self.base(b),self.local(l),self.extra(s)*torch.sigmoid(self.gate(c))],-1))

@torch.no_grad()
def evaluate(net,feat,mode,queries):
    out=[]
    for ix in queries.split(512):
        with torch.autocast('cuda',dtype=torch.bfloat16):pred=net(feat.make(ix,mode))
        out.append(pred.float().cpu())
    return torch.cat(out)

def calibrate(logits,labels):
    logits=logits.double();labels=labels.long();bias=torch.zeros(logits.shape[1],dtype=torch.float64,requires_grad=True);temp=torch.zeros((),dtype=torch.float64,requires_grad=True)
    opt=torch.optim.LBFGS([bias,temp],max_iter=70,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad();score=logits/torch.exp(temp.clamp(-3,3))+(bias-bias.mean());loss=F.cross_entropy(score,labels);loss.backward();return loss
    opt.step(closure)
    return dict(log_temperature=float(temp.detach().clamp(-3,3)),bias=(bias.detach()-bias.detach().mean()).tolist())

def tests(feat):
    q=feat.queries[100:104];b=feat.make(q,'coarse');f=q//63
    old=feat.z['g'][f].clone();oldx=feat.z['fine_pixels'][f].clone()
    feat.z['g'][f]=100;feat.z['fine_pixels'][f]=0
    after=feat.make(q,'coarse');assert all(torch.equal(x,y) for x,y in zip(b,after))
    feat.z['g'][f]=old;feat.z['fine_pixels'][f]=oldx
    assert (feat.hist[feat.z['t']==0]==feat.n).all()
    for family in ['flat','split']:
        net=Reader(feat.dim,family).to(DEV);x=list(feat.make(q,'fine'));x[1]=x[1].detach().requires_grad_();v=net(x).square().mean();gr=torch.autograd.grad(v,x[1])[0];assert gr.abs().sum()>0
    return dict(current_fine_excluded_from_coarse=True,episode_padding=True,extra_gradient_connected=True,dimension=feat.dim)

def main(smoke=False):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    feat=Features(load('train'));kp,hp=feat.pops();print('dimension',feat.dim,'pops',[len(p) for p in kp],[len(p) for p in hp],flush=True)
    assert min(map(len,kp+hp))>0
    (HERE/'checks.json').write_text(json.dumps(tests(feat),indent=2)+'\n')
    saved={};fit={};val=feat.queries[feat.z['episode'][feat.queries//63]>=360];z=feat.z
    for family in ['flat','split']:
        for seed in [0,1]:
            for mode in MODES:
                key=f'{family}_{seed}_{mode}';torch.manual_seed(617001+seed);net=Reader(feat.dim,family).to(DEV)
                opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);gen=torch.Generator(device=DEV).manual_seed(617101+seed)
                start=time.time();curve=[]
                for step in range(3 if smoke else 1200):
                    q=torch.cat([p[torch.randint(len(p),(100,),device=DEV,generator=gen)] for p in kp]+[p[torch.randint(len(p),(32,),device=DEV,generator=gen)] for p in hp])
                    f,c=q//63,q%63;kind=z['truth'][f,c].long();heading=z['heading'][f,c].long();has=heading>=0
                    with torch.autocast('cuda',dtype=torch.bfloat16):y=net(feat.make(q,mode));loss=F.cross_entropy(y[:,:5].float(),kind)+F.cross_entropy(y[has,5:].float(),heading[has])
                    opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
                    if step%300==0 or step==1199:curve.append(dict(step=step+1,loss=float(loss.detach()),gradient_norm=float(norm)));print(key,curve[-1],flush=True)
                net.eval();vv=val[:1024] if smoke else val;pred=evaluate(net,feat,mode,vv);ff,cc=vv//63,vv%63;yk=z['truth'][ff,cc].cpu();yh=z['heading'][ff,cc].cpu();cal={}
                for task,labs,logit in [('kind',yk,pred[:,:5]),('heading',yh,pred[:,5:])]:
                    ok=labs>=0;cal[task]=calibrate(logit[ok],labs[ok]) if ok.any() else None
                saved[key]=dict(state={k:v.cpu() for k,v in net.state_dict().items()},calibration=cal,family=family,seed=seed,mode=mode,dimension=feat.dim)
                fit[key]=dict(curve=curve,seconds=time.time()-start,parameters=sum(p.numel() for p in net.parameters()),calibration=cal)
                torch.save(saved,HERE/('smoke_readers.pt' if smoke else 'readers.pt'));(HERE/('smoke_fit.json' if smoke else 'fit.json')).write_text(json.dumps(fit,indent=2)+'\n')
                print('finished',key,round(time.time()-start,1),'seconds',flush=True);del net,opt;gc.collect();torch.cuda.empty_cache()
    if smoke:return
    del feat;gc.collect();torch.cuda.empty_cache();dev=Features(load('dev'));queries=dev.queries
    torch.save(dict(queries=queries.cpu(),episode=dev.z['episode'][queries//63].cpu(),t=dev.z['t'][queries//63].cpu(),
        kind=dev.z['truth'][queries//63,queries%63].cpu(),heading=dev.z['heading'][queries//63,queries%63].cpu(),
        night=dev.z['night'][queries//63].cpu(),fix=dev.z['fix'][queries//63].cpu()),HERE/'dev_queries.pt')
    for key,s in saved.items():
        net=Reader(dev.dim,s['family']).to(DEV);net.load_state_dict(s['state']);net.eval();pred=evaluate(net,dev,s['mode'],queries)
        torch.save(pred.half(),HERE/f'dev_logits_{key}.pt');print('dev evaluated',key,len(queries),flush=True);del net;torch.cuda.empty_cache()

if __name__=='__main__':main('--smoke' in sys.argv)
