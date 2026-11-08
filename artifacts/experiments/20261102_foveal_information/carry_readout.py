"""One predeclared carry-query instrument, frozen native Mamba, not a repair."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from saccade import model as M,train as TR,data as D
from saccade.eval import s1 as S
C=importlib.import_module('artifacts.experiments.20261102_foveal_information.conditional')
HERE=Path(__file__).resolve().parent;DEV=torch.device('cuda')

def choose(z):
    gen=np.random.default_rng(611570);ep=z['episode'].numpy()[:,None];y=z['truth'].numpy();av=z['localg'].numpy()>=0
    ids=[];valid=[]
    for val,n in ((False,600),(True,100)):
      for k in range(5):
       for a in (False,True):
        pop=np.flatnonzero(((ep>=360)==val)&(y==k)&(av==a));take=gen.choice(pop,min(n,len(pop)),replace=False)
        ids.extend(take.tolist());valid.extend([val]*len(take))
    return torch.tensor(ids),torch.tensor(valid)

@torch.no_grad()
def collect(m,cfg,z):
    if (HERE/'carry_pool.pt').exists():return torch.load(HERE/'carry_pool.pt',weights_only=False)
    ids,val=choose(z);frames,inv=torch.unique(ids//63,return_inverse=True);ep=z['episode'][frames];times=z['t'][frames]
    data={};checks=[]
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_train_labels.npz')),400);eps=S.episodes(L)
    for e in ep.unique().tolist():
        rows=torch.where(ep==e)[0];ts=times[rows];episode=eps[e]
        rng=np.random.default_rng([cfg['p1']['monitor']['fixed_fixation_seed'],episode.shard,episode.slot])
        fix=[D.fixation_prior(rng,()) for _ in range(episode.T+1)]
        chunk=TR.tensors(D.gather([episode],[[(0,1,t,*fix[t]) for t in range(episode.T+1)]]),DEV)
        with TR.autocast(DEV,cfg):
            P=[m.perception(chunk['frames'][:,a:a+512],chunk['fixation'][:,a:a+512]) for a in range(0,episode.T+1,512)]
            P=M.Percept(*(torch.cat(a,1) for a in zip(*[p.parts() for p in P])),chunk['fixation'])
            tok=m.belief.tokens(m.belief.percept_embed(P),chunk['reward'],chunk['terminated'],chunk['action'])
            carry=m.belief.initial(1,DEV);at=0
            for row,t in zip(rows.tolist(),ts.tolist()):
                out,carry=m.belief(tok[:,at:2*t+2],carry);at=2*t+2
                err=float((out[:,-1].float()-z['predicted'][frames[row]].to(DEV).float()).square().mean().sqrt());checks.append(err)
                assert err<.03,('native carry replay',e,t,err)
                data[row]=tuple(M.MambaCarry(c.conv.cpu(),c.ssm.cpu()) for c in carry)
        if e%40==0:print('carry collected episode',e,'frames',len(data),'error max',max(checks),flush=True)
    carries=tuple(M.MambaCarry(torch.cat([data[i][l].conv for i in range(len(frames))]),torch.cat([data[i][l].ssm for i in range(len(frames))])) for l in range(len(m.belief.mixers)))
    result={'ids':ids,'validation':val,'frame_inverse':inv,'frames':frames,'carry':carries,'replay_rms_max':max(checks)}
    torch.save(result,HERE/'carry_pool.pt');return result

class Instrument(nn.Module):
    def __init__(self,dim):
        super().__init__();self.reader=C.Reader(dim,'flat');self.query=nn.Embedding(81,256)
    def forward(self,inputs,mode,m,carry=None,cfg=None):
        b,s,cell=inputs
        if mode=='carry':
            with TR.autocast(DEV,cfg):q,_=m.belief(self.query(cell)[:,None],carry,backend='reference')
            s=F.pad(q[:,0].float(),(0,768))
        return self.reader((b,s,cell))

def carry_batch(pool,ix):
    k=pool['frame_inverse'][ix.cpu()]
    return tuple(M.MambaCarry(c.conv[k].to(DEV),c.ssm[k].to(DEV)) for c in pool['carry'])

def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.time();m,cfg=C.U.load();z=torch.load(HERE/'train_400.pt',weights_only=False)
    pool=collect(m,cfg,z);feat=C.Features(z,m);ids=pool['ids'].to(DEV);valid=pool['validation'].to(DEV);y=feat.z['truth'].flatten()[ids]
    pops=[torch.where(~valid&(y==k))[0] for k in range(5)];vix=torch.where(valid)[0]
    stats={};saved={};logs={}
    for seed in (0,1):
      for mode in ('base','fine','predicted','carry'):
        torch.manual_seed(611580+seed);net=Instrument(feat.dim).to(DEV);opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001)
        gen=torch.Generator(device=DEV).manual_seed(611590+seed)
        for step in range(2400):
            ix=torch.cat([a[torch.randint(len(a),(50,),device=DEV,generator=gen)] for a in pops]);actual=ids[ix]
            cc=carry_batch(pool,ix) if mode=='carry' else None
            logits=net(feat.make(actual,'base' if mode=='carry' else mode),mode,m,cc,cfg)
            loss=F.cross_entropy(logits,y[ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        net.eval();vv=[]
        with torch.no_grad():
            for ix in vix.split(250):
                cc=carry_batch(pool,ix) if mode=='carry' else None
                vv.append(net(feat.make(ids[ix],'base' if mode=='carry' else mode),mode,m,cc,cfg).log_softmax(-1).half().cpu())
        lp=torch.cat(vv);yy=y[vix].cpu();pred=lp.argmax(-1);av=feat.z['localg'].flatten()[ids[vix]]>=0
        conf=torch.bincount(yy*5+pred,minlength=25).reshape(5,5)
        ce=-lp.float().gather(-1,yy[:,None])[:,0]
        key=f'{seed}_{mode}';stats[key]={'confusion':conf.tolist(),'ce_by_kind':[float(ce[yy==k].mean()) for k in range(5)],'last_train_loss':float(loss),
                'in_confusion':torch.bincount(yy[av.cpu()]*5+pred[av.cpu()],minlength=25).reshape(5,5).tolist()}
        saved[key]={k:v.cpu() for k,v in net.state_dict().items()};logs[key]=lp
        torch.save(saved,HERE/'carry_readers.pt');torch.save(logs,HERE/'carry_logp.pt')
        (HERE/'carry_metrics.json').write_text(json.dumps(stats,indent=1));print(key,stats[key]['ce_by_kind'],'seconds',time.time()-start,flush=True)
    (HERE/'carry_identity.json').write_text(json.dumps({'replay_rms_max':pool['replay_rms_max'],'pool_queries':len(ids),'unique_frames':len(pool['frames']),
        'script_sha256':TR.file_sha256(__file__),'checkpoint_sha256':TR.file_sha256(C.U.CK),'seconds':time.time()-start},indent=1))

if __name__=='__main__':main()
