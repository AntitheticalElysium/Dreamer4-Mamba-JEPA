"""F1 frozen predictive instruments, PLAN.md. No architecture/world updates."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import argparse
import importlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S
C=importlib.import_module('artifacts.experiments.20261102_foveal_information.conditional')
Q=importlib.import_module('artifacts.experiments.20261102_direct_chooser.direct')
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'20261102_foveal_information'
MODES=('base','fine','input','predicted','memory')


def load_current(m,cfg,split,n):
    old=OLD/f'{split}_{n}.pt'
    if old.exists():return torch.load(old,map_location='cpu',weights_only=False)
    # Reuse the exact previous collection function, writing new cache only here.
    original=C.HERE;C.HERE=HERE
    try:return C.dataset(m,cfg,split,n)
    finally:C.HERE=original


@torch.no_grad()
def future(m,cfg,z,split,n,device):
    path=HERE/f'{split}_{n}_future.pt'
    if path.exists():
        q=torch.load(path,map_location='cpu',weights_only=False)
        assert q['checkpoint_sha256']==TR.file_sha256(Q.CK)
        return q
    L=S.first_episodes(dict(np.load(S.LABELS/f's0_{split}_labels.npz')),n)
    coarse=torch.load(Q.HERE/f'{split}_{400 if split=="train" else 600}_coarse.pt',weights_only=False,map_location='cpu')
    ids=coarse['offsets'][z['episode'].numpy()]+z['t'].numpy()+1
    assert np.all(z['t'].numpy()+1<=L['ep_T'][z['episode'].numpy()])
    current_ids=ids-1
    assert torch.equal(coarse['p'][current_ids],z['p'])
    out=[];start=0;offsets=np.r_[0,np.cumsum(L['ep_T']+1)]
    pos=L['s_pos'].astype(int)
    si=offsets[z['episode'].numpy()]+z['t'].numpy()
    for e,ep in enumerate(S.episodes(L)):
        count=ep.T
        assert bool((z['episode'][start:start+count]==e).all())
        assert torch.equal(z['t'][start:start+count],torch.arange(count))
        frames=D.RP.episode_fields(ep.shard,ep.slot)['observations']
        # Observed future frame, held CURRENT fixation; current crop does not receive
        # the future frame, so this is a target only, never a predictor input.
        for a in range(0,count,512):
            f=frames[a+1:min(a+512,count)+1].to(device)
            fix=z['fix'][start+a:start+min(a+512,count)].to(device)
            with TR.autocast(device,cfg):P=m.perception(f,fix)
            out.append(P.g.half().cpu())
        start+=count
        if e%100==0:print('future targets',split,e,n,flush=True)
    q=dict(p=coarse['p'][ids],g=torch.cat(out),scroll=torch.tensor(np.any(pos[si+1]!=pos[si],axis=1)),
           monitor=coarse['monitor'][current_ids],eps=coarse['eps'][current_ids],
           checkpoint_sha256=TR.file_sha256(Q.CK))
    assert len(q['g'])==len(z['p'])
    torch.save(q,path);return q


class Features:
    def __init__(self,z,target,m,device):
        self.device=device
        self.z={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in z.items()}
        self.target={k:v.to(device) if isinstance(v,torch.Tensor) else v for k,v in target.items()}
        z=self.z;self.n=len(z['p'])
        hist=torch.cat([z['p'].flatten(1).float(),z['v'].float(),z['prior'].float(),
                        F.one_hot((z['prev_action']+1).clamp(0,17),18).float(),
                        M.SYMLOG(z['reward'].float())[:,None],z['terminated'].float()[:,None],
                        torch.ones(self.n,1,device=device)],-1).half()
        self.history=torch.cat([hist,torch.zeros_like(hist[:1])])
        self.hidx=z['hist'].clone();self.hidx[self.hidx<0]=self.n
        self.cells=M.window(z['fix'])
        self.mask=torch.zeros(self.n,81,device=device).scatter(1,self.cells,1).half()
        self.dilated=F.max_pool2d(self.mask.float().reshape(-1,1,9,9),3,stride=1,padding=1).reshape(-1,81).half()
        self.memory=m.belief.memory(z['predicted'].float()).flatten(1).detach().half()
        self.dim=self.make(torch.arange(2,device=device),'base')[0].shape[-1]
        self.entity=((z['truth']>0)&(z['localg']>=0)).any(-1)

    def make(self,ix,mode):
        z=self.z
        b=torch.cat([z['p'][ix].flatten(1).float(),z['v'][ix].float(),z['prior'][ix].float(),
                     self.history[self.hidx[ix]].flatten(1).float(),
                     F.one_hot(z['action'][ix].long(),17).float(),M.SYMLOG(z['reward'][ix].float())[:,None],
                     z['terminated'][ix].float()[:,None],z['fix'][ix].float()/8,self.mask[ix].float()],-1)
        s=torch.zeros(len(ix),1024,device=self.device)
        if mode=='fine':s[:,:576]=z['g'][ix].flatten(1).float()
        elif mode=='memory':s=self.memory[ix].float()
        elif mode!='base':s[:,:256]=z[mode][ix].float()
        return b,s,z['p'][ix].float()


class Predictor(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.physical_body=nn.Sequential(nn.Linear(dim+1024,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.fine_body=nn.Sequential(nn.Linear(dim+1024,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
        self.physical=nn.Linear(256,81*16)
        self.fine=nn.Linear(256,9*64)
        nn.init.zeros_(self.physical.weight);nn.init.zeros_(self.physical.bias)
    def forward(self,inputs):
        b,s,p=inputs;x=torch.cat([b,s],-1)
        return p+self.physical(self.physical_body(x)).reshape(-1,81,16),self.fine(self.fine_body(x)).reshape(-1,9,64)


def error(pred,feat,ix):
    p,g=pred
    pe=(p-feat.target['p'][ix].float()).square().mean(-1)
    ge=(g-feat.target['g'][ix].float()).square().mean(-1)
    return pe,ge


@torch.no_grad()
def evaluate(net,feat,mode,ids):
    pp,gg=[],[]
    for ix in ids.split(512):
        p,g=error(net(feat.make(ix,mode)),feat,ix)
        pp.append(p.cpu());gg.append(g.cpu())
    return dict(p=torch.cat(pp),g=torch.cat(gg))


def fit(feat,mode,seed,steps):
    torch.manual_seed(seed);net=Predictor(feat.dim).to(feat.device)
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
    fitids=torch.where((feat.z['episode']<360)&~feat.target['monitor'])[0]
    valid=torch.where((feat.z['episode']>=360)&~feat.target['monitor'])[0]
    gen=torch.Generator(device=feat.device).manual_seed(seed+10);start=time.time();curve=[]
    for step in range(steps):
        ix=fitids[torch.randint(len(fitids),(512,),device=feat.device,generator=gen)]
        p,g=error(net(feat.make(ix,mode)),feat,ix)
        loss=p.mean()+g.mean()
        opt.zero_grad(set_to_none=True);loss.backward()
        gn=torch.nn.utils.clip_grad_norm_(list(net.physical_body.parameters())+list(net.physical.parameters()),1.)
        gg=torch.nn.utils.clip_grad_norm_(list(net.fine_body.parameters())+list(net.fine.parameters()),1.);opt.step()
        if step%600==0 or step==steps-1:
            val=evaluate(net,feat,mode,valid[:2048])
            r=dict(step=step+1,train_coarse=float(p.mean().detach()),train_fine=float(g.mean().detach()),
                   val_coarse=float(val['p'].float().mean()),val_fine=float(val['g'].float().mean()),
                   gradient_norm_physical=float(gn),gradient_norm_fine=float(gg),seconds=time.time()-start)
            curve.append(r);print('fit',mode,seed,r,flush=True)
    return net,dict(seed=seed,mode=mode,steps=steps,fit_rows=len(fitids),validation_rows=len(valid),
                    parameters=sum(p.numel() for p in net.parameters()),curve=curve,seconds=time.time()-start)


def output_metadata(feat):
    return {k:v.detach().cpu() for k,v in dict(episode=feat.z['episode'],t=feat.z['t'],night=feat.z['night'],
          entity=feat.entity,scroll=feat.target['scroll'],mask=feat.mask,dilated=feat.dilated,
          copy_p=(feat.z['p'].float()-feat.target['p'].float()).square().mean(-1),
          copy_g=(feat.z['g'].float()-feat.target['g'].float()).square().mean(-1)).items()}


def checks(feat):
    ix=torch.tensor([2,20,40],device=feat.device)
    net=Predictor(feat.dim).to(feat.device)
    x=feat.make(ix,'fine');p,g=net(x)
    assert torch.equal(p,feat.z['p'][ix].float())
    assert x[1].shape==(3,1024) and torch.equal(x[1][:,:576],feat.z['g'][ix].flatten(1).float())
    assert torch.count_nonzero(x[1][:,576:])==0
    before=feat.make(ix,'fine')
    oldp=feat.target['p'][ix].clone();oldg=feat.target['g'][ix].clone()
    feat.target['p'][ix]=900;feat.target['g'][ix]=900
    after=feat.make(ix,'fine')
    assert all(torch.equal(a,b) for a,b in zip(before,after))
    feat.target['p'][ix]=oldp;feat.target['g'][ix]=oldg
    # Initialize output randomly for input-gradient checks; avoid zero-output tautology.
    nn.init.normal_(net.physical.weight,std=.01)
    b,s,p=[a.detach().requires_grad_(True) for a in x]
    yp,yg=net((b,s,p));(yp.square().mean()+yg.square().mean()).backward()
    assert bool((s.grad[:,:576].abs().sum()>0))
    return dict(copy_coarse_expressible=True,whole_crop_present=True,targets_not_inputs=True,
                fine_gradient_connected=True,baseline_dimension=feat.dim)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=('collect','smoke','fit'));args=ap.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;device=torch.device('cuda')
    start=time.time();ck=torch.load(Q.CK,weights_only=False,map_location='cpu')
    m=M.M0(ck['cfg']).to(device);m.load_state_dict(ck['model']);m.eval();m.requires_grad_(False);cfg=ck['cfg'];del ck
    train=load_current(m,cfg,'train',400);tt=future(m,cfg,train,'train',400,device)
    if args.mode=='collect':
        dev=load_current(m,cfg,'dev',400);future(m,cfg,dev,'dev',400,device)
        (HERE/'collection_identity.json').write_text(json.dumps(dict(checkpoint_sha256=TR.file_sha256(Q.CK),
             script_sha256=TR.file_sha256(__file__),seconds=time.time()-start),indent=2));return
    feat=Features(train,tt,m,device)
    r=checks(feat);(HERE/'checks.json').write_text(json.dumps(r,indent=2))
    if args.mode=='smoke':
        _,r=fit(feat,'fine',611799,3);(HERE/'smoke.json').write_text(json.dumps(r,indent=2));return
    nets={};stats={}
    for seed in (611701,611702):
        for mode in MODES:
            key=f'{mode}_{seed}';net,r=fit(feat,mode,seed,2400)
            nets[key]={k:v.detach().cpu() for k,v in net.state_dict().items()};stats[key]=r
            torch.save(nets,HERE/'predictors.pt');(HERE/'fit.json').write_text(json.dumps(stats,indent=2))
    del feat;torch.cuda.empty_cache()
    dev=load_current(m,cfg,'dev',400);dt=future(m,cfg,dev,'dev',400,device);feat=Features(dev,dt,m,device)
    ids=torch.arange(feat.n,device=device);torch.save(output_metadata(feat),HERE/'dev_metadata.pt')
    for key,state in nets.items():
        mode=key.rsplit('_',1)[0];net=Predictor(feat.dim).to(device);net.load_state_dict(state);net.eval()
        out=evaluate(net,feat,mode,ids);torch.save(out,HERE/f'errors_{key}.pt')
        print('read',key,float(out['p'].float().mean()),float(out['g'].float().mean()),flush=True)
    (HERE/'identity.json').write_text(json.dumps(dict(checkpoint_sha256=TR.file_sha256(Q.CK),
        script_sha256=TR.file_sha256(__file__),plan_sha256=TR.file_sha256(HERE/'PLAN.md'),seconds=time.time()-start,
        peak_GiB=torch.cuda.max_memory_allocated()/2**30),indent=2))


if __name__=='__main__':main()
