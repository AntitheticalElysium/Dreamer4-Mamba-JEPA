"""Frozen native conditional information/readout study; PLAN.md. No world training."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,time,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from saccade import model as M,train as TR
from saccade.eval import s1 as S,gates as G
U=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.uptake_local')
HERE=Path(__file__).resolve().parent
DEV=torch.device('cuda')
MODES=('base','fine','input','corrected','predicted','memory')

@torch.no_grad()
def dataset(m,cfg,split,n):
    path=HERE/f'{split}_{n}.pt'
    if path.exists():return torch.load(path,map_location='cpu',weights_only=False)
    L=S.first_episodes(dict(np.load(S.LABELS/f's0_{split}_labels.npz')),n)
    enc=S.encode(m,L,cfg);grids=G.episode_grids(L)
    fields={k:[] for k in ('p','g','v','prior','corrected','predicted','input','action','prev_action','reward','terminated','fix','truth','episode','t','hist','localg','night')}
    off=0;so=np.r_[0,np.cumsum(L['ep_T']+1)]
    for e,q in enumerate(enc):
        T=len(q['p'])-1
        cells=M.window(q['fix'][:T]);localg=torch.full((T,63),-1,dtype=torch.int16)
        for j in range(9):
            ok=cells[:,j]<63
            localg[torch.arange(T)[ok],cells[ok,j]]=j
        hist=torch.arange(T)[:,None]-torch.arange(8,0,-1)[None]
        hist=torch.where(hist>=0,hist+off,-1)
        P=M.Percept(*(q[k][:T].float().to(DEV) for k in ('p','g','v')),q['fix'][:T].to(DEV))
        canvas=torch.zeros(T,81,64,device=DEV).scatter(1,cells.to(DEV)[:,:,None].expand_as(P.g),P.g)
        W=m.belief.percept_in.weight.float();eg=F.linear(canvas.flatten(1),W[:,1296:1296+5184]).half().cpu()
        vals=dict(p=q['p'][:T],g=q['g'][:T],v=q['v'][:T],prior=q['bp'][:T],corrected=q['b'][:T],predicted=q['bp'][1:T+1],input=eg,
                  action=q['action'][:T],prev_action=q['prev_action'][:T],reward=q['reward'][:T],terminated=q['terminated'][:T],fix=q['fix'][:T],
                  truth=torch.as_tensor(grids[e]['grid'][:T].reshape(T,63)).long(),episode=torch.full((T,),e),t=torch.arange(T),hist=hist,localg=localg,
                  night=torch.as_tensor(L['s_light'][so[e]:so[e]+T]<.5))
        for k,v in vals.items():fields[k].append(v)
        off+=T
    z={k:torch.cat(v) for k,v in fields.items()}
    z['episode_keys']=np.stack([L['ep_shard'],L['ep_slot']],1)
    torch.save(z,path)
    print('dataset',split,n,'frames',off,'saved',flush=True)
    return z

class Features:
    def __init__(self,z,m):
        self.z={k:v.to(DEV) if isinstance(v,torch.Tensor) else v for k,v in z.items()}
        z=self.z;self.n=len(z['p'])
        self.identity=torch.eye(81,device=DEV)
        # Physical denoiser context: previous action, predicted belief and physical percept.
        self.history=torch.cat([z['p'].flatten(1).float(),z['v'].float(),z['prior'].float(),F.one_hot((z['prev_action']+1).clamp(0,17),18).float(),
                                M.SYMLOG(z['reward'].float())[:,None],z['terminated'].float()[:,None],torch.ones(self.n,1,device=DEV)],-1).half()
        self.history=torch.cat([self.history,torch.zeros_like(self.history[:1])])
        self.hist=z['hist'].clone();self.hist[self.hist<0]=self.n
        self.memory=m.belief.memory(z['predicted'].float()).flatten(1).half()
        self.dim=self.make(torch.arange(2,device=DEV),'base')[0].shape[-1]
    def make(self,ix,mode):
        z=self.z;f=ix//63;c=ix%63;lg=z['localg'][f,c].long();avail=lg>=0
        base=torch.cat([z['p'][f].flatten(1).float(),z['v'][f].float(),z['prior'][f].float(),self.history[self.hist[f]].flatten(1).float(),
            z['p'][f,c].float(),self.identity[c],F.one_hot(z['action'][f].long(),17).float(),
            M.SYMLOG(z['reward'][f].float())[:,None],z['terminated'][f].float()[:,None],z['fix'][f].float()/8,avail[:,None].float()],-1)
        extra=torch.zeros(len(ix),1024,device=DEV)
        if mode=='fine':extra[:,:64]=z['g'][f,lg.clamp(min=0)].float()*avail[:,None]
        elif mode=='memory':extra=self.memory[f].float()
        elif mode!='base':extra[:,:256]=z[mode][f].float()
        return base,extra,c

class Reader(nn.Module):
    def __init__(self,dim,family):
        super().__init__();self.family=family
        if family=='flat':self.net=nn.Sequential(nn.Linear(dim+1024,256),nn.GELU(),nn.Linear(256,256),nn.GELU(),nn.Linear(256,5))
        else:
            self.base=nn.Sequential(nn.Linear(dim,256),nn.GELU(),nn.Linear(256,256),nn.GELU())
            self.side=nn.Sequential(nn.Linear(1024,256),nn.GELU())
            self.gate=nn.Embedding(81,256)
            self.net=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,5))
    def forward(self,x):
        b,s,c=x
        if self.family=='flat':return self.net(torch.cat([b,s],-1))
        return self.net(torch.cat([self.base(b),self.side(s)*torch.sigmoid(self.gate(c))],-1))

def populations(feat,split):
    z=feat.z;ok=(z['episode']<360 if split=='train' else z['episode']>=360)[:,None].expand(-1,63)
    inside=z['localg']>=0
    return [torch.where((ok & (z['truth']==k)&(inside==a)).flatten())[0] for k in range(5) for a in (True,False)]

@torch.no_grad()
def evaluate(reader,feat,mode,frames):
    ids=torch.where(frames)[0];out=torch.empty(len(ids),63,5,device='cpu',dtype=torch.float16)
    flat=(ids[:,None]*63+torch.arange(63,device=DEV)[None]).flatten()
    for st in range(0,len(flat),1024):
        ix=flat[st:st+1024];out.view(-1,5)[st:st+len(ix)]=reader(feat.make(ix,mode)).float().log_softmax(-1).half().cpu()
    return out

def metrics(logp,z,frames):
    y=z['truth'][frames].cpu();inside=z['localg'][frames].cpu()>=0;night=z['night'][frames].cpu();ans={}
    for a in (True,False):
        for n in (None,False,True):
            ok=(inside==a)&(torch.ones_like(night) if n is None else night==n)[:,None]
            yh=logp.argmax(-1);conf=torch.bincount((y[ok]*5+yh[ok]).long(),minlength=25).reshape(5,5)
            ce=-logp.float().gather(-1,y[...,None]).squeeze(-1)
            means=[float(ce[ok&(y==k)].mean()) for k in range(5)]
            ans[f'{"in" if a else "out"}_{"all" if n is None else "night" if n else "day"}']={
                'confusion':conf.tolist(),'recall':(conf.diag()/conf.sum(1).clamp(min=1)).tolist(),'macro_ce':float(np.mean(means)),
                'ce_by_kind':means,'rows':int(ok.sum())}
    return ans

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    out=HERE/('smoke' if '--smoke' in sys.argv else 'conditional');out.mkdir(exist_ok=True)
    start=time.time();m,cfg=U.load();train=dataset(m,cfg,'train',400)
    feat=Features(train,m);pops=populations(feat,'train')
    print('features',feat.dim,'populations',[len(a) for a in pops],flush=True)
    assert min(map(len,pops))>0
    val=feat.z['episode']>=360
    steps=3 if '--smoke' in sys.argv else 2400
    stats={};saved={};vallog={}
    # Restore any completed readouts; same protocol, no partial-fit resumption.
    if (out/'readers.pt').exists():saved=torch.load(out/'readers.pt',map_location='cpu',weights_only=False)
    if (out/'metrics.json').exists():stats=json.loads((out/'metrics.json').read_text())
    for family in ('flat','gated'):
        for seed in (0,1):
            for mode in MODES:
                key=f'{family}_{seed}_{mode}'
                if key in saved:continue
                torch.manual_seed(611510+seed)
                reader=Reader(feat.dim,family).to(DEV);opt=torch.optim.AdamW(reader.parameters(),lr=.001,weight_decay=.0001)
                gen=torch.Generator(device=DEV).manual_seed(611520+seed)
                for step in range(steps):
                    ix=torch.cat([a[torch.randint(len(a),(128,),device=DEV,generator=gen)] for a in pops])
                    lp=reader(feat.make(ix,mode));lo=F.cross_entropy(lp,feat.z['truth'].flatten()[ix])
                    opt.zero_grad(set_to_none=True);lo.backward();opt.step()
                reader.eval();logp=evaluate(reader,feat,mode,val)
                stats[key]=metrics(logp,feat.z,val);stats[key]['last_train_loss']=float(lo)
                saved[key]={'state':{k:v.cpu() for k,v in reader.state_dict().items()},'family':family,'dim':feat.dim,'mode':mode,'seed':seed}
                vallog[key]=logp
                torch.save(saved,out/'readers.pt');torch.save(vallog,out/'validation_logp.pt')
                (out/'metrics.json').write_text(json.dumps(stats,indent=1,allow_nan=False))
                print(key,'inside CE',stats[key]['in_all']['macro_ce'],'recall',stats[key]['in_all']['recall'],'seconds',time.time()-start,flush=True)
    del feat;torch.cuda.empty_cache()
    dev=dataset(m,cfg,'dev',250);feat=Features(dev,m);allframes=torch.ones(feat.n,dtype=torch.bool,device=DEV)
    devlog={};devstats={}
    for key,s in saved.items():
        reader=Reader(s['dim'],s['family']).to(DEV);reader.load_state_dict(s['state']);reader.eval()
        logp=evaluate(reader,feat,s['mode'],allframes);devlog[key]=logp
        devstats[key]=metrics(logp,feat.z,allframes)
        print('dev',key,devstats[key]['in_all']['macro_ce'],flush=True)
    torch.save(devlog,out/'dev_logp.pt');(out/'dev_metrics.json').write_text(json.dumps(devstats,indent=1,allow_nan=False))
    (out/'identity.json').write_text(json.dumps({'checkpoint':str(U.CK),'checkpoint_sha256':TR.file_sha256(U.CK),
        'script_sha256':TR.file_sha256(__file__),'plan_sha256':TR.file_sha256(HERE/'PLAN.md'),'steps':steps,'seconds':time.time()-start,
        'cuda_peak_GiB':torch.cuda.max_memory_allocated()/2**30},indent=1))

if __name__=='__main__':main()
