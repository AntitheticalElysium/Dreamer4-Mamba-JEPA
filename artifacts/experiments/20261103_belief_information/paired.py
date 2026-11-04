"""Genuine-look prediction effects, fixed query/target and full real prefixes."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

B=importlib.import_module('artifacts.experiments.20261103_belief_information.audit')
M,TR,D,S,U=B.M,B.TR,B.D,B.S,B.U
HERE=B.HERE;DEVICE=B.DEVICE


def cohort(feat,t):
    z=feat.z;q=feat.q['query'];hud=q[:,0]>=6
    valid=(z['episode']<250)&(z['t']>=8)&~t['scroll'].to(DEVICE)
    gen=np.random.default_rng(612141);chosen=[];used=set()
    for night,is_hud in ((False,False),(True,False),(False,True),(True,True)):
        ids=torch.where(valid&(z['night']==night)&(hud==is_hud))[0].cpu().numpy();gen.shuffle(ids)
        take=[]
        for f in ids:
            ep=int(z['episode'][f])
            if ep not in used:take.append(int(f));used.add(ep)
            if len(take)==16:break
        assert len(take)==16,(night,is_hud,len(take))
        chosen.extend(take)
    return np.array(sorted(chosen))


def weights(query):
    cells=M.window(query[None])[0].cpu().numpy()
    overlap=np.isin(U.WIN.cpu().numpy(),cells).any(1);control=~overlap
    aa=np.zeros(49);bb=np.zeros(49)
    rows=U.CENTRES.cpu().numpy()[:,0]
    common=[r for r in range(1,8) if np.any(overlap&(rows==r)) and np.any(control&(rows==r))]
    assert common
    for r in common:
        a=overlap&(rows==r);b=control&(rows==r);aa[a]=1/len(common)/a.sum();bb[b]=1/len(common)/b.sum()
    assert abs(aa.sum()-1)<1e-8 and abs(bb.sum()-1)<1e-8
    return aa,bb


@torch.no_grad()
def supplements(m,cfg,pp,embedding,predicted):
    linear=torch.load(B.J.HERE/'maps.pt',map_location=DEVICE,weights_only=False)
    cells=M.window(pp.fix);fid=(pp.fix[:,0]-1)*7+pp.fix[:,1]-1
    joint=m.belief.spatial_fusion(pp,cells).flatten(1).float()
    w=m.belief.percept_in.weight.float().bfloat16().float();bias=m.belief.percept_in.bias.float().bfloat16().float()
    mask=torch.zeros(49,81,device=DEVICE).scatter(1,cells,1.)
    xc=torch.cat([pp.p.flatten(1),torch.zeros(49,5184,device=DEVICE),mask,pp.v],-1).bfloat16().float()
    with torch.autocast('cuda',enabled=False):
        offset=torch.nn.functional.linear(xc,w,bias)
        recovered=torch.bmm(linear['pinvq'][fid],(embedding.float()-offset)[...,None]).squeeze(-1)
    return dict(base=torch.zeros(49,1,device=DEVICE),fine=pp.g.flatten(1).float(),joint=joint,
                mixed=embedding.float(),recovered_bf16=recovered,predicted=predicted.float())


@torch.no_grad()
def native(m,cfg,cache,memory,pa,query,target,conditioning,scales,gen,smoke):
    levels=U.FP.levels(cfg)[:1] if smoke else U.FP.levels(cfg)
    ph,fg=[],[]
    for draw in range(1 if smoke else 2):
        rp,rg=[],[]
        for sig in levels:
            noisy=tuple(x+sig*torch.randn((1,*x.shape[1:]),device=DEVICE,generator=gen) for x in target)
            cs,co,ci,cn=M.precondition(torch.full((49,1),sig,device=DEVICE),cfg['edm']['sigma_data'])
            fo,_=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in noisy),(cn,cn),pa,memory,query,cache)
            raw=(target[0]-M._bcast(cs,target[0])*noisy[0])/M._bcast(co,target[0])
            rp.append((fo[0].float()-raw).square()[:,0].mean(-1).cpu())
            sp=torch.full((49,1),.1,device=DEVICE);_,_,cip,cnp=M.precondition(sp,cfg['edm']['sigma_data'])
            xp=(M._bcast(cip,conditioning[0])*conditioning[0],M._bcast(ci,noisy[1])*noisy[1],
                M._bcast(cip,conditioning[1])*conditioning[1])
            fo,_=m.denoiser.step(xp,(cnp,cn),pa,memory,query,cache)
            raw=(target[1]-M._bcast(cs,target[1])*noisy[1])/M._bcast(co,target[1])
            rg.append((fo[1].float()-raw).square()[:,0].mean(-1).cpu())
        ph.append(torch.stack(rp));fg.append(torch.stack(rg))
    return dict(p=torch.stack(ph),g=torch.stack(fg))


@torch.no_grad()
def run(smoke=False):
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True;B.pins()
    m,cfg=B.R.load_world('p1f16');z,t=B.J.load('dev');q=B.targets(m,cfg,z,t,'dev')
    feat=B.Features(z,t,q,m)
    saved=None
    if not smoke:
        saved=torch.load(HERE/'readers.pt',map_location='cpu',weights_only=False)
        feat.norm={k:v.to(DEVICE) for k,v in saved['norm'].items()}
    selected=cohort(feat,t);selected=selected[:2] if smoke else selected
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),400);eps=S.episodes(L)
    data=[];side_records=[];max_replay=0.;native_records=[];raw_context=[]
    for e in sorted(set(feat.z['episode'][torch.as_tensor(selected,device=DEVICE)].tolist())):
        ids=torch.where(feat.z['episode']==e)[0];offset=int(ids[0]);carry=m.belief.initial(1,DEVICE);at=0
        with TR.autocast(DEVICE,cfg):
            pp=M.Percept(*(feat.z[k][ids].float()[None] for k in ('p','g','v')),feat.z['fix'][ids][None])
            tok=m.belief.tokens(m.belief.percept_embed(pp),feat.z['reward'][ids][None],feat.z['terminated'][ids][None],feat.z['action'][ids][None])
            for f in selected:
                if int(feat.z['episode'][f])!=e:continue
                tt=int(feat.z['t'][f]);_,carry=m.belief(tok[:,at:2*tt],carry);at=2*tt
                raw=D.RP.episode_fields(eps[e].shard,eps[e].slot)['observations']
                pc=m.perception(raw[tt].to(DEVICE)[None].expand(49,-1,-1,-1),U.CENTRES)
                emb=m.belief.percept_embed(pc)
                tokens=m.belief.tokens(emb[:,None],feat.z['reward'][f].expand(49,1),feat.z['terminated'][f].expand(49,1),feat.z['action'][f].expand(49,1))
                yy,_=m.belief(tokens,U.expanded(carry,49));bp=yy[:,1].float()
                recorded=int(((U.CENTRES==feat.z['fix'][f]).all(-1)).nonzero()[0])
                diff=float((bp[recorded]-feat.z['predicted'][f].float()).square().mean().sqrt())
                max_replay=max(max_replay,diff);assert diff<.03,('prefix replay',e,tt,diff)
                ss=supplements(m,cfg,pc,emb,bp);side_records.append({k:v.cpu() for k,v in ss.items()})
                ix=torch.full((49,),int(f),device=DEVICE)
                raw_context.append(tuple(x.cpu() for x in feat.raw(ix,U.CENTRES)))
                assert torch.equal(pc.p,pc.p[:1].expand_as(pc.p))
                a,b=weights(feat.q['query'][f]);data.append(dict(frame=int(f),episode=e,t=tt,a=a,b=b))
                gen=torch.Generator(device=DEVICE).manual_seed(612151+int(f));cache=m.denoiser.new_cache()
                for s in range(f-8,f):
                    cache=U.enter(m,cfg,tuple(feat.z[k][s:s+1].float()[None] for k in ('p','g','v')),
                        feat.z['prev_action'][s:s+1][None],feat.z['prior'][s:s+1].float()[None],feat.z['fix'][s:s+1][None],cache,gen)
                cache=U.expandcache(cache,49)
                cache=U.enter(m,cfg,tuple(x[:,None] for x in pc.parts()),feat.z['prev_action'][f].expand(49,1),
                    feat.z['prior'][f].float().expand(49,1,-1),U.CENTRES[:,None],cache,gen)
                target=(feat.f.target['p'][f:f+1].float()[None].expand(49,-1,-1,-1),
                        feat.f.target['g'][f:f+1].float()[None].expand(49,-1,-1,-1),
                        feat.q['v'][f:f+1].float()[None].expand(49,-1,-1))
                condition=(feat.q['p_condition'][f:f+1].float()[None].expand(49,-1,-1,-1),
                           feat.q['v_condition'][f:f+1].float()[None].expand(49,-1,-1))
                query=feat.q['query'][f].expand(49,1,2);pa=feat.z['action'][f].expand(49,1)
                native_records.append(native(m,cfg,cache,m.belief.memory(bp)[:,None],pa,query,target,condition,None,gen,smoke))
                print('paired scene',len(data),'/',len(selected),'prefix',diff,flush=True)
    if smoke:
        B.write('paired_smoke.json',dict(scenes=len(data),prefix_rms_max=max_replay,
            consistent_native_context=True,independent_fixed_future_query=True,genuine49_crops=True));B.pins();return
    torch.cuda.empty_cache()
    results={};bd=raw_context[0][0].shape[-1];fd=raw_context[0][1].shape[-1]
    for key,state in saved['states'].items():
        mode=key.rsplit('_',1)[0];net=B.Reader(bd,fd).to(DEVICE);net.load_state_dict(state['weights']);net.eval();pp,gg=[],[]
        for row,context,ss in zip(data,raw_context,side_records):
            f=row['frame'];b,g,gp,gq=(x.to(DEVICE).float() for x in context)
            b=(b-feat.norm['b_mean'])/feat.norm['b_std'];g=(g-feat.norm['f_mean'])/feat.norm['f_std']
            x=B.padded(ss[mode].to(DEVICE),state['side_mean'].to(DEVICE),state['side_std'].to(DEVICE))
            yp,yg=net((b,g,gp,gq),x,feat.z['p'][f:f+1].float().expand(49,-1,-1))
            ptarget,gtarget=feat.truth(torch.tensor([f],device=DEVICE))
            pp.append(((yp-ptarget)/feat.norm['p_scale']).square().mean(-1).cpu())
            gg.append(((yg-gtarget)/feat.norm['g_scale']).square().mean(-1).cpu())
        results[key]=dict(p=torch.stack(pp),g=torch.stack(gg));del net
    torch.save(dict(meta=data,readers=results,native={k:torch.stack([r[k] for r in native_records]) for k in ('p','g')}),HERE/'paired.pt')
    B.write('paired_identity.json',dict(scenes=len(data),prefix_rms_max=max_replay,queries_held_fixed=True,
        native_noise_levels=U.FP.levels(cfg),checkpoint_sha256=B.R.SHAS['p1f16']))
    B.pins()


if __name__=='__main__':run('--smoke' in sys.argv)
