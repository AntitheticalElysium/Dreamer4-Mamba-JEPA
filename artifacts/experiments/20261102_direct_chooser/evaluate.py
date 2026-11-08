"""Chooser retention readings and independent native-reference confirmation; PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from saccade import model as M,train as TR
from saccade.eval import s1 as S,gates as G
Q=importlib.import_module('artifacts.experiments.20261102_direct_chooser.direct')
LC=importlib.import_module('artifacts.experiments.20261031_learned_fixation_sweep.locus')
HERE=Q.HERE


def subset(L, rows):
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    return {k:v[rows] if k.startswith('ep_') else np.concatenate([
        v[(so if k.startswith('s_') else ao)[e]:(so if k.startswith('s_') else ao)[e+1]] for e in rows])
        for k,v in L.items()}


def metadata():
    saved=dict(np.load(Q.REF))
    L=S.first_episodes(dict(np.load(S.LABELS/'s0_dev_labels.npz')),600)
    fresh=subset(L,np.arange(400,600));rng=np.random.default_rng(611640)
    ctx,face,d=LC.contexts(fresh,rng)
    off=np.r_[0,np.cumsum(fresh['ep_T']+1)]
    cls=[];rows=[]
    for c,rs in ctx.items():
        take=np.sort(rng.choice(len(rs),min(128,len(rs)),replace=False))
        rows.extend(rs[i] for i in take);cls.extend([c]*len(take))
    cls=np.array(cls);ep=np.array([e for e,t in rows]);tt=np.array([t for e,t in rows])
    state=off[ep]+tt-1
    counter=np.array([LC.ATTEMPTS[c][1] if c in LC.ATTEMPTS else LC.ATTEMPTS['stone'][1] if c=='stone_no_tool' else -1 for c in cls])
    out=dict(cls=cls,ep=ep+400,t=tt,counter=counter,faced=(3+d[state,0])*9+4+d[state,1],light=fresh['s_light'][state])
    # Metadata-only counterpart to native source contexts: no semantic target is used
    # in the head fitting or inputs. Archive it before reading head results.
    np.savez_compressed(HERE/'confirmation_metadata.npz',**out)
    choose=[]
    for c in np.unique(saved['cls']):
        ix=np.flatnonzero(saved['cls']==c)
        choose.extend(rng.choice(ix,min(32,len(ix)),replace=False))
    old_noise={k:saved[k][np.array(choose)] for k in ('cls','ep','t','counter','faced','light')}
    old_noise['original_index']=np.array(choose)
    return saved,out,old_noise,L


@torch.no_grad()
def native(meta,L,device):
    ck=torch.load(Q.CK,map_location='cpu',weights_only=False);cfg=ck['cfg']
    m=M.M0(cfg).to(device);m.load_state_dict(ck['model']);m.eval();m.requires_grad_(False)
    epids=np.unique(meta['ep']);local={int(e):i for i,e in enumerate(epids)}
    labels=subset(L,epids);enc=S.encode(m,labels,cfg)
    rows=[(local[int(e)],int(t)) for e,t in zip(meta['ep'],meta['t'])]
    bps,worst=LC.beliefs(m,cfg,enc,rows)
    assert worst<.02,worst
    print('native replay',len(epids),'episodes',len(rows),'contexts; relative error',worst,flush=True)
    gen=torch.Generator(device=device).manual_seed(611642)
    hat=cfg['edm']['sampler']['sigma_max']*(1+min(cfg['edm']['sampler']['s_churn']/cfg['edm']['sampler']['steps'],math.sqrt(2)-1))
    cc,ss=[],[]
    for st in range(0,len(rows),8):
        rs=rows[st:st+8];n=len(rs)
        take=lambda k,dt:torch.stack([enc[e][k][t+dt] for e,t in rs]).to(device)
        prev=take('prev_action',0).long()
        bp=torch.stack([torch.stack([bps[r][int(a)],bps[r][0]]) for r,a in zip(rs,prev)]).to(device)
        pa=torch.stack([prev,torch.zeros_like(prev)],1).flatten()[:,None]
        with TR.autocast(device,cfg):
            cache=LC.context_cache(m,cfg,take,2,gen)
            mem=m.belief.memory(bp.flatten(0,1))[:,None]
            fx=take('fix',0).repeat_interleave(2,0)[:,None]
            draw=[]
            for _ in range(2):
                shapes=[(81,16),(9,64),(8,)]
                x=tuple(hat*torch.randn((n,1,*sh),device=device,generator=gen).repeat_interleave(2,0) for sh in shapes)
                pred,_=M.denoised(m.denoiser,x,torch.full((2*n,1),hat,device=device),pa,mem,fx,cache,3.,cfg['edm']['sigma_data'])
                draw.append(pred[0][:,0].float().reshape(n,2,81,16))
        c=sum((x[:,0]-x[:,1]).square().sum(-1) for x in draw)/2
        s=sum((take('p',0).float()-x[:,0]).square().sum(-1) for x in draw)/2
        assert not (prev==0).any() or bool((c[prev==0]==0).all())
        win=Q.WIN.to(device)
        cc.append(c[:,win].sum(-1).cpu());ss.append(s[:,win].sum(-1).cpu())
        if st%256==0:print('native scores',st,len(rows),flush=True)
    c,s=torch.cat(cc),torch.cat(ss)
    return dict(C=c.numpy(),S=s.numpy(),prob=Q.policy(c,s).numpy()),dict(belief_replay_worst_relative_l2=worst,
        contexts=len(rows),episodes=len(epids),checkpoint_sha256=TR.file_sha256(Q.CK))


def comparisons(meta,cand,ref,eps):
    win=Q.WIN.numpy();report={}
    cs_ref={'C':ref['C'],'S':ref['S'],'mixed':ref['prob']}
    cs_new={'C':cand['C'],'S':cand['S'],'mixed':cand['prob']}
    rng=np.random.default_rng(611655);u=rng.random((len(meta['ep']),32))
    draws=lambda p:np.minimum((p.cumsum(-1)[:,None,:]<u[...,None]).sum(-1),48)
    rd,nd=draws(ref['prob']),draws(cand['prob'])
    for cls in np.unique(meta['cls']):
        ix=meta['cls']==cls;ep=meta['ep'][ix];n=int(ix.sum());ar=np.arange(n)
        remote=(win[None]==meta['counter'][ix,None,None]).any(-1)
        contact=(win[None]==meta['faced'][ix,None,None]).any(-1)
        report[str(cls)]={};r=report[str(cls)]
        add=lambda name,a,b:r.update({name:{'reference':Q.bootstrap(a,ep),'candidate':Q.bootstrap(b,ep),'difference':Q.bootstrap(np.asarray(b,float)-np.asarray(a,float),ep)}})
        for sig in ('C','S','mixed'):
            a,b=cs_ref[sig][ix].argmax(-1),cs_new[sig][ix].argmax(-1)
            if cls!='random':add(sig+'_remote_argmax',remote[ar,a],remote[ar,b])
            add(sig+'_contact_argmax',contact[ar,a],contact[ar,b])
            add(sig+'_union_argmax',(remote|contact)[ar,a],(remote|contact)[ar,b])
        for label,mask in (('remote',remote),('contact',contact)):
            if cls=='random' and label=='remote':continue
            add('mixed_'+label+'_mass',(ref['prob'][ix]*mask).sum(-1),(cand['prob'][ix]*mask).sum(-1))
            add('mixed_'+label+'_sampled',mask[ar[:,None],rd[ix]].mean(-1),mask[ar[:,None],nd[ix]].mean(-1))
            r['uniform_'+label+'_mass']=float(mask.mean())
        r['policy_TV']=Q.bootstrap(.5*np.abs(ref['prob'][ix]-cand['prob'][ix]).sum(-1),ep)
        r['factual_mse']=Q.bootstrap(cand['mse'][ix],ep)
        r['copy_mse']=Q.bootstrap(cand['copy_mse'][ix],ep)
        r['factual_mse_vs_copy']=float(cand['mse'][ix].mean()/cand['copy_mse'][ix].mean())
        r['exogenous_rows']=int((eps[ix]==1.).sum())
        for tag,sel in (('exogenous',eps[ix]==1.),('nonexogenous',eps[ix]!=1.)):
            if not sel.any() or cls=='random':continue
            r[tag+'_remote_mass']={'n':int(sel.sum()),'reference':float((ref['prob'][ix][sel]*remote[sel]).sum(-1).mean()),
                                   'candidate':float((cand['prob'][ix][sel]*remote[sel]).sum(-1).mean())}
    gates={}
    for cls,sig in (('drink','C'),('stone','C'),('coal','S'),('iron','S')):
        for metric in (sig+'_remote_argmax','mixed_remote_mass'):
            q=report[cls][metric]['difference']
            gates[cls+'_'+metric]='pass' if q['low']>=-.05 else 'fail' if q['high']<-.05 else 'inconclusive'
    return dict(classes=report,gates=gates,all_primary_pass=all(v=='pass' for v in gates.values()))


def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    start=time.time();device=torch.device('cuda')
    saved,fresh,old_noise,L=metadata()
    z=torch.load(HERE/'dev_600_coarse.pt',map_location='cpu',weights_only=False);data=Q.Data(z,device)
    olds={k:saved[k] for k in ('cls','ep','t','counter','faced','light')}
    ref={'C':saved['contingency'],'S':saved['surprise']}
    ref['prob']=Q.policy(torch.tensor(ref['C']),torch.tensor(ref['S'])).numpy()
    cohorts={'saved':olds,'confirmation':fresh}
    result={}; predictions={}
    for seed in (611601,611602):
        ck=torch.load(HERE/f'head_{seed}.pt',weights_only=False,map_location='cpu')
        assert ck['checkpoint_sha256']==TR.file_sha256(Q.CK)
        h=Q.Head().to(device);h.load_state_dict(ck['state']);h.eval()
        for name,meta in cohorts.items():
            ids=data.row_ids(meta['ep'],meta['t']);pr=Q.predict(h,data,ids)
            predictions[(seed,name)]=pr
            np.savez_compressed(HERE/f'predictions_{seed}_{name}.npz',**pr,**meta,eps=z['eps'][ids.cpu()].numpy())
        ids=data.row_ids(olds['ep'],olds['t'])
        r=comparisons(olds,predictions[(seed,'saved')],ref,z['eps'][ids.cpu()].numpy())
        result[str(seed)]={'saved':r}
        print('primary',seed,r['gates'],flush=True)
    # Predeclared independent native reference on fresh episodes + a small old-bank
    # noise-replication set; no changes to the candidate after seeing primary results.
    combined={k:np.r_[fresh[k],old_noise[k]] for k in fresh}
    nr,ident=native(combined,L,device)
    np.savez_compressed(HERE/'native_confirmation_and_noise.npz',**nr,**combined)
    nf=len(fresh['ep']);freshref={k:v[:nf] for k,v in nr.items()};oldref={k:v[nf:] for k,v in nr.items()}
    for seed in (611601,611602):
        ids=data.row_ids(fresh['ep'],fresh['t'])
        result[str(seed)]['confirmation']=comparisons(fresh,predictions[(seed,'confirmation')],freshref,z['eps'][ids.cpu()].numpy())
    # Reference repeat: compare independent current draw to saved native scores.
    oi=old_noise['original_index'];rr={k:v[oi] for k,v in ref.items()}
    dummy=dict(oldref,mse=np.zeros(len(oi)),copy_mse=np.ones(len(oi)))
    ids=data.row_ids(old_noise['ep'],old_noise['t'])
    result['native_noise_repeat']=comparisons(old_noise,dummy,rr,z['eps'][ids.cpu()].numpy())
    result['identity']=dict(ident,seconds=time.time()-start,script_sha256=TR.file_sha256(__file__),
                            peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    (HERE/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('done',time.time()-start,'seconds',flush=True)


if __name__=='__main__':main()
