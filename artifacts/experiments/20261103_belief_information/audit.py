"""One bounded frozen-world diagnostic; target/query contracts in PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC', '1')
import argparse
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

R=importlib.import_module('artifacts.experiments.20261103_fusion_retention.audit')
J=importlib.import_module('artifacts.experiments.20261103_fusion_projection.audit')
P,A,M,TR,D,S,U=R.P,R.A,R.M,R.TR,R.D,R.S,R.U
HERE=Path(__file__).resolve().parent
DEVICE=torch.device('cuda')
MODES=('base','fine','joint','mixed','recovered_bf16','predicted')
SEEDS=(612101,612102)


def write(name,x):
    p=HERE/name;tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');tmp.replace(p)


def pins():
    for p,h in json.loads((HERE/'SOURCE_PINS.json').read_text()).items():
        assert TR.file_sha256(TR.ROOT/p)==h,p


@torch.no_grad()
def targets(m,cfg,z,t,split):
    path=HERE/f'{split}_query.pt'
    if path.exists():
        x=torch.load(path,map_location='cpu',weights_only=False)
        assert x['checkpoint_sha256']==R.SHAS['p1f16']
        assert torch.equal(x['observed_fix'],z['fix'])
        return x
    gen=torch.Generator().manual_seed(612111+(split=='dev'))
    query=torch.randint(1,8,z['fix'].shape,generator=gen)
    repeat=torch.rand(len(query),generator=gen)<.5;query[repeat]=z['fix'][repeat]
    L=S.first_episodes(dict(np.load(S.LABELS/f's0_{split}_labels.npz')),400)
    gs,vs=[],[];offset=0
    for e,ep in enumerate(S.episodes(L)):
        raw=D.RP.episode_fields(ep.shard,ep.slot)['observations'];T=ep.T
        assert torch.equal(z['episode'][offset:offset+T],torch.full((T,),e))
        # Match S1's source-frame batch boundaries, including frame0. Moving
        # the batching origin by one frame can change low-precision kernels.
        fq=torch.cat([z['fix'][offset:offset+1],query[offset:offset+T]])
        for st in range(0,T+1,512):
            end=min(T+1,st+512);fr=raw[st:end].to(DEVICE)
            with TR.autocast(DEVICE,cfg):pp=m.perception(fr,fq[st:end].to(DEVICE))
            skip=int(st==0);sl=slice(offset+st+skip-1,offset+end-1)
            gs.append(pp.g[skip:].half().cpu());vs.append(pp.v[skip:].half().cpu())
            delta=float((pp.p[skip:].half().cpu()-t['p'][sl]).abs().max())
            assert delta==0.,('future coarse cache parity',e,st,delta)
        offset+=T
        if e%100==0:print('query targets',split,e,flush=True)
    v=torch.cat(vs)
    ng=torch.Generator().manual_seed(612131+(split=='dev'))
    pc=(t['p'].float()+.1*torch.randn(t['p'].shape,generator=ng)).half()
    vc=(v.float()+.1*torch.randn(v.shape,generator=ng)).half()
    x=dict(g=torch.cat(gs),v=v,p_condition=pc,v_condition=vc,query=query,repeat=repeat,observed_fix=z['fix'],
           checkpoint_sha256=R.SHAS['p1f16'])
    torch.save(x,path);return x


class Features:
    def __init__(self,z,t,q,m):
        self.f=P.Features(z,{**t,'g':q['g']},m,DEVICE);self.z=self.f.z
        self.q={k:v.to(DEVICE) if isinstance(v,torch.Tensor) else v for k,v in q.items()}
        self.n=self.f.n;self.ids=torch.where((self.z['episode']<360)&~self.f.target['monitor'])[0]
        self.val=torch.where((self.z['episode']>=360)&~self.f.target['monitor'])[0]
        self.fine_history=torch.cat([self.z['g'].flatten(1).float(),self.z['fix'].float()/8],-1).half()
        self.fine_history=torch.cat([self.fine_history,torch.zeros_like(self.fine_history[:1])])
        self.norm=None
    def raw(self,ix,override=None):
        b,_,_=self.f.make(ix,'base')
        if override is not None:
            fix=override
            mask=torch.zeros(len(ix),81,device=DEVICE).scatter(1,M.window(fix),1.)
            b=b.clone();b[:,-83:-81]=fix.float()/8;b[:,-81:]=mask
        else:fix=self.z['fix'][ix]
        old=self.fine_history[self.f.hidx[ix]].flatten(1).float()
        query=self.q['query'][ix]
        mask=torch.zeros(len(ix),81,device=DEVICE).scatter(1,M.window(query),1.)
        fine=torch.cat([b,old,self.q['p_condition'][ix].flatten(1).float(),self.q['v_condition'][ix].float(),query.float()/8,mask],-1)
        obsid=(fix[:,0]-1)*7+fix[:,1]-1;queryid=(query[:,0]-1)*7+query[:,1]-1
        gatep=F.one_hot(obsid,49).float();gateg=torch.cat([gatep,F.one_hot(queryid,49).float()],-1)
        return b,fine,gatep,gateg
    def make(self,ix,override=None):
        b,f,gp,gg=self.raw(ix,override)
        if self.norm is not None:
            b=(b-self.norm['b_mean'])/self.norm['b_std'];f=(f-self.norm['f_mean'])/self.norm['f_std']
        return b,f,gp,gg
    def truth(self,ix):return self.f.target['p'][ix].float(),self.f.target['g'][ix].float()


@torch.no_grad()
def moments(feat):
    sums={};squares={};count=0
    for ix in feat.ids.split(1024):
        b,f,_,_=feat.raw(ix)
        for k,v in (('b',b),('f',f)):
            v=v.double();sums[k]=sums.get(k,0)+v.sum(0);squares[k]=squares.get(k,0)+v.square().sum(0)
        count+=len(ix)
    out={}
    for k in sums:
        mean=sums[k]/count;std=(squares[k]/count-mean.square()).clamp(min=0).sqrt().clamp(min=.01)
        out[k+'_mean']=mean.float();out[k+'_std']=std.float()
    for k,t in (('p',feat.f.target['p']),('g',feat.f.target['g'])):
        x=t[feat.ids].float();out[k+'_scale']=x.std((0,1)).clamp(min=.05)
    return out


class Head(nn.Module):
    def __init__(self,dim,gate,out,zero=False):
        super().__init__()
        self.context=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.side=nn.Sequential(nn.Linear(1024,256),nn.GELU())
        self.gate=nn.Linear(gate,256)
        self.out=nn.Linear(512,out)
        if zero:nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
    def forward(self,b,s,g):
        return self.out(torch.cat([self.context(b),self.side(s)*torch.sigmoid(self.gate(g))],-1))


class Reader(nn.Module):
    def __init__(self,bd,fd):
        super().__init__();self.physical=Head(bd,49,81*16,True);self.fine=Head(fd,98,9*64)
    def forward(self,inputs,s,current):
        b,f,gp,gg=inputs
        return current+self.physical(b,s,gp).reshape(-1,81,16),self.fine(f,s,gg).reshape(-1,9,64)


def side(feat,extra,mode):
    if mode=='base':x=torch.zeros(feat.n,1,device=DEVICE)
    elif mode=='fine':x=feat.z['g'].flatten(1).float()
    elif mode=='predicted':x=feat.z['predicted'].float()
    else:x=extra[mode].to(DEVICE)
    mean=x[feat.ids].mean(0);std=x[feat.ids].std(0).clamp(min=.01)
    return x,mean,std


def padded(x,mean,std):
    out=torch.zeros(len(x),1024,device=DEVICE);out[:,:x.shape[-1]]=(x-mean)/std;return out


@torch.no_grad()
def errors(net,feat,s,ids):
    outp,outg=[],[]
    for ix in ids.split(256):
        p,g=net(feat.make(ix),padded(s[0][ix],s[1],s[2]),feat.z['p'][ix].float())
        yp,yg=feat.truth(ix)
        outp.append(((p-yp)/feat.norm['p_scale']).square().mean(-1).cpu())
        outg.append(((g-yg)/feat.norm['g_scale']).square().mean(-1).cpu())
    return dict(p=torch.cat(outp),g=torch.cat(outg))


def fit(feat,extra,mode,seed,steps):
    torch.manual_seed(seed);b,f,*_=feat.make(feat.ids[:2]);net=Reader(b.shape[-1],f.shape[-1]).to(DEVICE)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    s=side(feat,extra,mode);gen=torch.Generator(device=DEVICE).manual_seed(seed+10);curve=[]
    for step in range(steps):
        ix=feat.ids[torch.randint(len(feat.ids),(256,),device=DEVICE,generator=gen)]
        p,g=net(feat.make(ix),padded(s[0][ix],s[1],s[2]),feat.z['p'][ix].float());yp,yg=feat.truth(ix)
        lp=((p-yp)/feat.norm['p_scale']).square().mean();lg=((g-yg)/feat.norm['g_scale']).square().mean()
        opt.zero_grad(set_to_none=True);(lp+lg).backward()
        gp=torch.nn.utils.clip_grad_norm_(net.physical.parameters(),1.)
        gg=torch.nn.utils.clip_grad_norm_(net.fine.parameters(),1.);opt.step()
        if step%600==0 or step==steps-1:
            val=errors(net,feat,s,feat.val[:2048]);row=dict(step=step+1,physical=float(lp.detach()),fine=float(lg.detach()),
                physical_grad=float(gp),fine_grad=float(gg),val_p=float(val['p'].mean()),val_g=float(val['g'].mean()))
            curve.append(row);print('fit',mode,seed,row,flush=True)
    return net,dict(curve=curve,side_mean=s[1].cpu(),side_std=s[2].cpu())


def extra_transforms(m,cfg,z):
    # The exact preceding projection audit's frozen algebra and native transforms.
    linear=torch.load(J.HERE/'maps.pt',map_location='cpu',weights_only=False)
    old=torch.backends.cuda.matmul.allow_tf32;torch.backends.cuda.matmul.allow_tf32=False
    try:all_extra,_=J.prepare(m,cfg,z,linear)
    finally:torch.backends.cuda.matmul.allow_tf32=old
    return {k:all_extra[k] for k in ('joint','mixed','recovered_bf16')}


def mechanics(feat,extra,m):
    ix=feat.ids[:4];before=[x.clone() for x in feat.make(ix)]
    old=feat.f.target['g'][ix].clone();feat.f.target['g'][ix]=999
    assert all(torch.equal(a,b) for a,b in zip(before,feat.make(ix)));feat.f.target['g'][ix]=old
    # Future physical evidence may influence the fine context only.
    old=feat.q['p_condition'][ix].clone();feat.q['p_condition'][ix]+=1
    after=feat.make(ix);assert torch.equal(before[0],after[0]);assert not torch.equal(before[1],after[1])
    feat.q['p_condition'][ix]=old
    x=torch.ones(4,576,device=DEVICE,requires_grad=True);s=padded(x,torch.zeros(576,device=DEVICE),torch.ones(576,device=DEVICE))
    net=Reader(before[0].shape[-1],before[1].shape[-1]).to(DEVICE)
    nn.init.normal_(net.physical.out.weight,std=.01)
    p,g=net(before,s,feat.z['p'][ix].float());grad=torch.autograd.grad(p.square().mean()+g.square().mean(),x)[0]
    assert grad.abs().sum()>0
    assert all(not v.requires_grad for v in m.parameters());del net
    assert (feat.q['query']!=feat.z['fix']).any()
    return dict(future_fine_excluded=True,future_coarse_excluded_from_physical=True,
        future_coarse_supplied_to_fine=True,independent_query_support=True,
        world_frozen=True,side_gradient_nonzero=True)


def metadata(feat,t):
    z=feat.z;query=feat.q['query'];cells=M.window(query);source=torch.zeros(feat.n,81,device=DEVICE)
    source[:,:63]=(z['truth']>0).float()
    return {k:v.cpu() for k,v in dict(episode=z['episode'],t=z['t'],night=z['night'],scroll=t['scroll'].to(DEVICE),
        query=query,observed_fix=z['fix'],query_cells=cells,source=source,
        fine_source=source.gather(1,cells),fine_hud=(cells>=63),repeat=feat.q['repeat']).items()}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('smoke','run'));args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;start=time.time();pins()
    m,cfg=R.load_world('p1f16');z,t=J.load('train');q=targets(m,cfg,z,t,'train')
    feat=Features(z,t,q,m);extra=extra_transforms(m,cfg,z);feat.norm=moments(feat)
    write('checks.json',mechanics(feat,extra,m))
    if args.stage=='smoke':
        net,stat=fit(feat,extra,'fine',612199,3);write('smoke.json',dict(curve=stat['curve'],seconds=time.time()-start))
        pins();return
    states={};stats={};saved_norm={k:v.cpu() for k,v in feat.norm.items()}
    for seed in SEEDS:
        for mode in MODES:
            key=f'{mode}_{seed}';net,stat=fit(feat,extra,mode,seed,2400)
            states[key]=dict(weights={k:v.detach().cpu() for k,v in net.state_dict().items()},**stat)
            stats[key]=stat['curve'];torch.save(dict(states=states,norm=saved_norm),HERE/'readers.pt')
            write('fit.json',stats);del net
    del feat,extra,z,t,q;torch.cuda.empty_cache()
    z,t=J.load('dev');q=targets(m,cfg,z,t,'dev');feat=Features(z,t,q,m)
    feat.norm={k:v.to(DEVICE) for k,v in saved_norm.items()};extra=extra_transforms(m,cfg,z)
    torch.save(metadata(feat,t),HERE/'metadata.pt');ids=torch.arange(feat.n,device=DEVICE)
    b,f,*_=feat.make(ids[:2])
    for key,state in states.items():
        mode=key.rsplit('_',1)[0];net=Reader(b.shape[-1],f.shape[-1]).to(DEVICE)
        net.load_state_dict(state['weights']);net.eval()
        if mode=='base':x=torch.zeros(feat.n,1,device=DEVICE)
        elif mode=='fine':x=feat.z['g'].flatten(1).float()
        elif mode=='predicted':x=feat.z['predicted'].float()
        else:x=extra[mode].to(DEVICE)
        err=errors(net,feat,(x,state['side_mean'].to(DEVICE),state['side_std'].to(DEVICE)),ids)
        torch.save(err,HERE/f'errors_{key}.pt');del net,x
    pins();assert all(not p.requires_grad for p in m.parameters())
    write('STATUS.json',dict(phase='predictors_complete',seconds=time.time()-start,
          peak_GiB=torch.cuda.max_memory_allocated()/2**30,checkpoint_sha256=R.SHAS['p1f16']))


if __name__=='__main__':main()
