"""Frozen native reliability and sequential chooser behavior. UPTAKE_PROTOCOL.md.
Working policies are diagnostic, not a newly adopted training curriculum.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,math,time
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
from saccade.eval import s1 as S,gates as G

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEV=torch.device('cuda')
LC=importlib.import_module('artifacts.experiments.20261031_learned_fixation_sweep.locus')
CENTRES=LC.CENTRES.to(DEV);WIN=M.window(CENTRES)


def load(n):
    path=ROOT/f'artifacts/experiments/20261101_p1f15/run/ckpt_{n:02d}.pt'
    c=torch.load(path,map_location='cpu',weights_only=False);m=M.M0(c['cfg']).to(DEV)
    m.load_state_dict(c['model']);m.eval();m.requires_grad_(False)
    return m,c['cfg'],path


def twice(cache):
    return {'t':cache['t'],'kv':[(None,None) if k is None else (k.repeat_interleave(2,0),v.repeat_interleave(2,0)) for k,v in cache['kv']]}


@torch.no_grad()
def signals(m,cfg,p,bp,bnoop,prev,fix,cache,gen):
    n=len(p);hat=cfg['edm']['sampler']['sigma_max']*(1+min(cfg['edm']['sampler']['s_churn']/cfg['edm']['sampler']['steps'],math.sqrt(2)-1))
    pa=torch.stack([prev,torch.where(prev<0,prev,torch.zeros_like(prev))],1).flatten()[:,None]
    mem=m.belief.memory(torch.stack([bp,bnoop],1).flatten(0,1))[:,None]
    shape=[(81,p.shape[-1]),(9,cfg['perception']['fovea']['token_dim']),(cfg['perception']['lighting']['dim'],)]
    x=tuple(hat*torch.randn((n,1,*sh),device=DEV,generator=gen).repeat_interleave(2,0) for sh in shape)
    pred,_=M.denoised(m.denoiser,x,torch.full((n*2,1),hat,device=DEV),pa,mem,fix.repeat_interleave(2,0)[:,None],twice(cache),3.,cfg['edm']['sigma_data'])
    pp=pred[0][:,0].float().view(n,2,81,-1)
    c=(pp[:,0]-pp[:,1]).square().sum(-1);s=(p.float()-pp[:,0]).square().sum(-1)
    if (prev==0).any():assert c[prev==0].max()==0,('NOOP common-noise mismatch',float(c[prev==0].max()))
    return c[:,WIN].sum(-1),s[:,WIN].sum(-1)


def priorities(c,s,trace=None):
    def norm(x):
        mu=x.mean(-1,keepdim=True)
        return torch.where(mu>1e-12,x/mu.clamp(min=1e-12),torch.zeros_like(x))
    logits=.5*norm(c)+.5*norm(s)
    if trace is not None:logits=logits-trace[:,WIN].mean(-1)
    prob=.8*logits.softmax(-1)+.2/49
    assert torch.isfinite(prob).all() and (prob>=.2/49-1e-7).all()
    return prob


@torch.no_grad()
def reliability(out):
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),12)
    rows=[(e,int(t)) for e,T in enumerate(L['ep_T']) for t in np.unique(np.linspace(8,min(int(T),255),4,dtype=int)) if T>=8]
    if os.environ.get('GAZE_SMOKE'):rows=rows[:2]
    records={}
    for ck in ((10,) if os.environ.get('GAZE_SMOKE') else (1,5,10)):
        m,cfg,path=load(ck);enc=S.encode(m,L,cfg)
        static=np.load(HERE/'static_repeats.npz') if (HERE/'static_repeats.npz').exists() else None
        extra=[] if static is None else [tuple(map(int,r)) for r in static['rows']]
        bps,err=LC.beliefs(m,cfg,enc,sorted(set(rows+extra)));assert err<.03,err
        vals=[]
        for e,t in rows:
            en=enc[e];take=lambda k,dt:en[k][t+dt:t+dt+1].to(DEV)
            cg=torch.Generator(device=DEV).manual_seed(611401+100*e+t)
            with TR.autocast(DEV,cfg):
                cache=LC.context_cache(m,cfg,take,1,cg)
                prev=take('prev_action',0);bp=bps[(e,t)][int(prev)].to(DEV)[None];bn=bps[(e,t)][0].to(DEV)[None]
                draws=[]
                for _ in range(2):
                    c,s=signals(m,cfg,take('p',0),bp,bn,prev,take('fix',0),cache,cg)
                    draws.append(torch.stack([c,s,priorities(c,s)],0)[:,0].cpu().numpy())
            vals.append(np.stack(draws))
        z=np.stack(vals)
        np.savez_compressed(out/f'reliability_{ck:02d}.npz',scores=z,rows=np.array(rows))
        if static is not None:
            sr=[]
            for (e,t),frames in zip(extra,static['frames']):
                en=enc[e];take=lambda k,dt:en[k][t+dt:t+dt+1].to(DEV)
                with TR.autocast(DEV,cfg):
                    cache=LC.context_cache(m,cfg,take,1,torch.Generator(device=DEV).manual_seed(611470+e))
                    prev=take('prev_action',0);bp=bps[(e,t)][int(prev)].to(DEV)[None];bn=bps[(e,t)][0].to(DEV)[None]
                    p=m.perception(torch.as_tensor(frames,device=DEV),CENTRES[:1].expand(4,-1)).p
                    vv=[]
                    for j in range(4):
                        c,s=signals(m,cfg,p[j:j+1],bp,bn,prev,take('fix',0),cache,torch.Generator(device=DEV).manual_seed(611471))
                        vv.append(torch.stack([c,s,priorities(c,s)])[:,0].cpu().numpy())
                sr.append(np.stack(vv))
            np.savez_compressed(out/f'static_{ck:02d}.npz',scores=np.stack(sr),rows=np.array(extra))
        records[ck]={'sha256':TR.file_sha256(path),'bp_recorded_rel_error_max':err,'rows':len(rows)}
        print('reliability checkpoint',ck,'rows',len(rows),flush=True)
        del m
    (out/'reliability_identity.json').write_text(json.dumps(records,indent=1))


@torch.no_grad()
def sequential(out):
    m,cfg,path=load(10)
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),32 if os.environ.get('GAZE_BENCH16') else 12)
    if os.environ.get('GAZE_BENCH16'):
        U=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.uptake_local')
        L=U.subset(L,np.flatnonzero(L['ep_T']>=64)[:16]);assert len(L['ep_T'])==16
    eps=S.episodes(L);grids=G.episode_grids(L);so=np.r_[0,np.cumsum(L['ep_T']+1)]
    T=min(max(e.T+1 for e in eps),256);n=len(eps)
    if os.environ.get('GAZE_BENCH16'):T=64
    if os.environ.get('GAZE_SMOKE'):T=4
    records=[];timings={};replaycheck=[]
    for policy in ('uniform','mix','mix_recency'):
        start=time.time();carry=m.belief.initial(n,DEV);bp=torch.zeros(n,cfg['belief']['width'],device=DEV);bn=bp.clone()
        cache=m.denoiser.new_cache();trace=torch.zeros(n,81,device=DEV)
        ng=torch.Generator(device=DEV).manual_seed(611440)
        fg=torch.Generator(device=DEV).manual_seed(611441)
        for t in range(T):
            live=np.array([t<=e.T for e in eps]);tt=[min(t,e.T) for e in eps]
            obs=torch.stack([D.RP.episode_fields(e.shard,e.slot)['observations'][i] for e,i in zip(eps,tt)]).to(DEV)
            take=lambda k:torch.tensor([float(D.RP.episode_fields(e.shard,e.slot)[k][i-1]) if i else 0 for e,i in zip(eps,tt)],device=DEV)
            action=torch.tensor([int(D.RP.episode_fields(e.shard,e.slot)['actions_taken'][i]) if i<e.T else 0 for e,i in zip(eps,tt)],device=DEV)
            prev=take('actions_taken').long();prev[:]=-1 if t==0 else prev
            with TR.autocast(DEV,cfg):
                # No realized fine crop is read by the scoring function.
                p0=m.perception(obs,CENTRES[:1].expand(n,-1))
                if policy=='uniform':
                    c=s=torch.zeros(n,49,device=DEV);prob=torch.full((n,49),1/49,device=DEV)
                else:
                    c,s=signals(m,cfg,p0.p,bp,bn,prev,CENTRES[:1].expand(n,-1),cache,ng)
                    prob=priorities(c,s,trace if policy=='mix_recency' else None)
                chosen=torch.multinomial(prob,1,generator=fg).squeeze(-1);fix=CENTRES[chosen]
                P=m.perception(obs,fix)
                # Cache of this real frame uses incoming belief, not the just-corrected one.
                cache=M.context_pass(m.denoiser,tuple(x[:,None] for x in P.parts()),prev[:,None],m.belief.memory(bp)[:,None],fix[:,None],cache,cfg,ng)
                embed=m.belief.percept_embed(P)
                oldcarry=carry
                pt=embed+m.belief.outcome_in(torch.stack([M.SYMLOG(take('rewards')),take('terminated')],-1).to(embed.dtype))
                y,cpost=m.belief(pt[:,None],carry,backend='reference')
                forks=tuple(M.MambaCarry(z.conv.repeat_interleave(2,0),z.ssm.repeat_interleave(2,0)) for z in cpost)
                aa=torch.stack([action,torch.zeros_like(action)],1).flatten()
                yy,cc=m.belief(m.belief.action(aa)[:,None],forks,backend='reference')
                yy=yy[:,0].view(n,2,-1);bp=yy[:,0].float();bn=yy[:,1].float()
                carry=tuple(M.MambaCarry(z.conv[::2].contiguous(),z.ssm[::2].contiguous()) for z in cc)
                if t<8:
                    # Auxiliary NOOP fork must not change the actual two-token update.
                    tk=m.belief.tokens(embed[:,None],take('rewards')[:,None],take('terminated')[:,None],action[:,None])
                    yr,cr=m.belief(tk,oldcarry,backend='reference')
                    err=float((yr[:,-1].float()-bp).square().mean(-1).sqrt().max())
                    assert err<.03,('two-token actual versus NOOP fork',err)
                    replaycheck.append({'policy':policy,'t':t,'actual_fork_rms_error':err})
                trace*=.5**.25
                trace.scatter_add_(1,WIN[chosen],torch.ones(n,9,device=DEV))
            for e in np.flatnonzero(live):
                cells=WIN[chosen[e]].cpu().numpy();grid=grids[e]['grid'][tt[e]].reshape(63)
                mob=(grid[np.minimum(cells,62)]>0)&(cells<63)
                records.append({'policy':policy,'episode':int(e),'t':t,'chosen':int(chosen[e]),
                    'prob':prob[e].float().cpu().numpy(),'C':c[e].float().cpu().numpy(),'S':s[e].float().cpu().numpy(),
                    'night':bool(L['s_light'][so[e]+tt[e]]<.5),'mob_covered':bool(mob.any()),
                    'hud':bool((cells>=63).any()),'heldout':bool(e>=8)})
            if t%64==0:print('sequential',policy,t,flush=True)
        timings[policy]=time.time()-start
    np.savez_compressed(out/'sequential.npz',**{k:np.stack([r[k] for r in records]) for k in records[0]})
    (out/'sequential_identity.json').write_text(json.dumps({'sha256':TR.file_sha256(path),'timings':timings,
        'policy':'equal mean-normalized C/S; softmax T1; uniform .2; recency window-mean trace half-life4',
        'rows':len(records),'finite_checks':replaycheck},indent=1))


def main():
    torch.set_num_threads(4)
    out=HERE/('gaze_bench16' if os.environ.get('GAZE_BENCH16') else 'gaze_checks_smoke' if os.environ.get('GAZE_SMOKE') else 'gaze_checks');out.mkdir(exist_ok=True)
    if not os.environ.get('GAZE_BENCH16'):reliability(out)
    sequential(out)

if __name__=='__main__':main()
