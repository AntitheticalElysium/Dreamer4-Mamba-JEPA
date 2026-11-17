"""One bounded, native sleep-duration falsification test; PLAN.md. build=CPU, run=GPU."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import dataclasses,importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, data as D, train as TR
from saccade.eval import s1 as S

CV=importlib.import_module('artifacts.experiments.20261106_determinant_curves.curves')
HERE=Path(__file__).resolve().parent
CKPT='artifacts/experiments/20261106_p1f4/run/ckpt_10.pt'
H,K,N=48,8,32

def choose(L):
    T=CV.transitions(L);g=T['g'];pos=L['s_pos'][g].astype(int)
    good=T['asleep']&(T['body'][:,3]==8)&(T['body'][:,0]>=7)&(T['body'][:,1]>=3)&(T['body'][:,2]>=3)&(T['t']>=17)
    good&=~L['s_arrows_m'][g].any(1)
    for k in ('zombies','skeletons'):
        near=L[f's_{k}_m'][g]&(abs(L[f's_{k}_p'][g].astype(int)-pos[:,None]).sum(-1)<=6)
        good&=~near.any(1)
    rng=np.random.default_rng(110711);eps=np.unique(T['ep'][good]);rows=[]
    for e in rng.permutation(eps)[:N]:
        j=int(rng.choice(np.flatnonzero(good&(T['ep']==e))))
        rows.append(dict(e=int(e),t0=int(T['t'][j]-1),j=j,shard=int(L['ep_shard'][e]),slot=int(L['ep_slot'][e])))
    return rows,len(eps)

def build():
    import jax,jax.numpy as jnp
    assert jax.default_backend()=='cpu'
    L=dict(np.load(CV.LABELS/'s0_dev_labels.npz'));rows,n=choose(L)
    env,params,_,step,render=D.RP.env_and_render()
    def rollout(s,keys):
        def one(st,key):
            _,q,r,d,_=step(key,st,jnp.asarray(0,jnp.int32))
            obs=jnp.array([q.player_energy,q.is_sleeping,q.player_health,d],dtype=jnp.float32)
            return q,(obs,r)
        return jax.lax.scan(one,s,keys)[1]
    sample=jax.jit(jax.vmap(rollout,in_axes=(None,0)))
    truths=[]
    for i,r in enumerate(rows):
        st=D.RP.advance_to(r['shard'],r['slot'],r['t0'])
        actual=np.asarray(render(st));saved=D.RP.episode_fields(r['shard'],r['slot'])['observations'][r['t0']].numpy()
        assert np.array_equal(actual,saved),('source replay mismatch',r)
        assert st.is_sleeping and int(st.player_energy)==8
        r['source_fatigue']=float(st.player_fatigue)
        keys=jax.random.split(jax.random.PRNGKey(110711+i),K*H).reshape(K,H,2)
        tr,_=sample(st,keys);truths.append(np.asarray(tr))
        D.RP._shard.cache_clear()
        print('oracle',i+1,len(rows),flush=True)
    torch.save(dict(rows=rows,eligible_episodes=n,truth=np.stack(truths),H=H,K=K),HERE/'bank.pt')

def subset(L,eps):
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    return {k:(v[eps] if k.startswith('ep_') else np.concatenate([v[(so if k.startswith('s_') else ao)[e]:(so if k.startswith('s_') else ao)[e+1]] for e in eps]))
            for k,v in L.items()}

@torch.no_grad()
def run(smoke=False):
    assert os.environ.get('MAMBA_DETERMINISTIC')=='1'
    torch.set_num_threads(4)
    bank=torch.load(HERE/'bank.pt',map_location='cpu',weights_only=False)
    rows=bank['rows'][:2] if smoke else bank['rows'];h=4 if smoke else H;k=2 if smoke else K
    L=dict(np.load(CV.LABELS/'s0_dev_labels.npz'));eps=[r['e'] for r in rows];sub=subset(L,eps)
    model,cfg=CV.IF.load(CKPT);model.requires_grad_(False);dev=next(model.parameters()).device
    enc=S.encode(model,sub,cfg)
    original_hud=CV.HUD
    CV.HUD={66:('s_body',3),63:('s_body',0)}
    with torch.enable_grad(): readers,valid=CV.fit_readers(model,cfg,sub,enc,0)
    CV.HUD=original_hud
    assert all(q['valid'] for q in valid.values()),valid
    energy=readers[66];sleep=readers['asleep']
    with TR.autocast(dev,cfg):
        pp=torch.stack([en['p'][r['t0']] for en,r in zip(enc,rows)]).to(dev).float()
        vv=torch.stack([en['v'][r['t0']] for en,r in zip(enc,rows)]).to(dev).float()
        source_e=energy(torch.cat([pp[:,66],vv],-1))['y'].argmax(-1)
        source_s=sleep(torch.cat([pp[:,31],vv,pp[:,:63].mean(1)],-1))['y'].argmax(-1)
    assert (source_e==8).all() and (source_s==1).all(), 'source readers do not recover the declared cohort'
    # Confidence stress: these have no asserted health label. This is NOT evidence
    # of actual generated-token error, only a test of confidence as a certificate.
    stress_gen=torch.Generator(device=dev).manual_seed(110711)
    actual=torch.cat([en['p'][:,63] for en in enc]).to(dev).float()[:4096]
    lighting=torch.cat([en['v'] for en in enc]).to(dev).float()[:len(actual)]
    perm=torch.stack([torch.randperm(16,device=dev,generator=stress_gen) for _ in range(len(actual))])
    invalid=actual.gather(1,perm)
    with TR.autocast(dev,cfg):
        confidence=readers[63](torch.cat([invalid,lighting],-1))['y'].softmax(-1).max(-1).values.float()
    reader_stress={'channel_permuted_tokens_no_semantic_label':True,'median_confidence':float(confidence.median()),
        'fraction_confidence_above_0p9':float((confidence>.9).float().mean()),
        'interpretation':'High confidence on invalid controls cannot establish correctness on generated tokens.'}
    dims=[tuple(enc[0][p].shape[1:]) for p in ('p','g','v')]
    results=[];shams=[];part_records=[]
    for start in range(0,len(rows),8):
        batch=rows[start:start+8];eidx=list(range(start,start+len(batch)));cs=[];bs=[];rl=[]
        for ei,r in zip(eidx,batch):
            en=enc[ei];t=r['t0'];n=t+1
            take=lambda key:en[key][:n].to(dev)[None]
            with TR.autocast(dev,cfg):
                pp=M.Percept(*(take(p).float() for p in ('p','g','v')),take('fix'))
                act=take('action').clone();act[:,-1]=0
                reads,_,last=S.history(model,pp.p)   # P1.5: the record from the episode start, carried with Mamba's
                yy,cc=model.belief(model.belief.tokens(S.embed(model,pp,reads),take('reward'),take('terminated'),act),model.belief.initial(1,dev))
            err=float((yy[0,-2].float()-en['b'][t].to(dev).float()).square().mean().sqrt())
            assert err<.03,('corrected prefix sham',err)
            if int(en['action'][t])==0:
                err2=float((yy[0,-1].float()-en['bp'][t+1].to(dev).float()).square().mean().sqrt())
                assert err2<.03,('predicted prefix sham',err2)
            shams.append(err);cs.append(cc);bs.append(yy[:,-1].float());rl.append(last)
        carry=tuple(type(cs[0][i])(*(torch.cat([getattr(c[i],f.name) for c in cs]).repeat_interleave(k,0) for f in dataclasses.fields(cs[0][i]))) for i in range(len(cs[0])))   # Mamba-2 or -3
        rs=None if rl[0] is None else torch.cat(rl).repeat_interleave(k,0)
        rin=S.window(model,enc,[(ei,r['t0']) for ei,r in zip(eidx,batch)],7)
        ri=lambda dt:None if rin is None else rin[:,7+dt].repeat_interleave(k,0)[:,None]
        bp=torch.cat(bs).repeat_interleave(k,0);B=len(bp)
        prior=lambda key,dt:torch.stack([enc[ei][key][r['t0']+dt] for ei,r in zip(eidx,batch)]).repeat_interleave(k,0).to(dev)
        cache=model.denoiser.new_cache();gen=torch.Generator(device=dev).manual_seed(110711+start)
        with TR.autocast(dev,cfg):
            for dt in range(-7,1):
                cache=M.context_pass(model.denoiser,tuple(prior(p,dt).float()[:,None] for p in ('p','g','v')),prior('prev_action',dt)[:,None],
                    S.cond(model.belief.memory(prior('bp',dt).float())[:,None],ri(dt)),prior('fix',dt)[:,None],cache,cfg,gen)
        fixations=D.fixation_prior(np.random.default_rng(110711+start),(h,B))
        alive=torch.ones(B,device=dev,dtype=torch.bool);out=[];raw=[]
        action=torch.zeros(B,1,device=dev,dtype=torch.long)
        for step in range(h):
            fix=torch.as_tensor(fixations[step],device=dev)
            with TR.autocast(dev,cfg):
                p,g,v=tuple(t[:,0].float() for t in M.sample(model.denoiser,dims,action,S.cond(model.belief.memory(bp)[:,None],S.at(rs,np.s_[:,None])),fix[:,None],cache,cfg,gen))
                ee=energy(torch.cat([p[:,66],v],-1))['y'].softmax(-1)
                ss=sleep(torch.cat([p[:,31],v,p[:,:63].mean(1)],-1))['y'].softmax(-1)[:,1]
                reads,_,rs_next=S.history(model,p,state=rs)
                percept=M.Percept(p,g,v,fix);emb=S.embed(model,percept,reads)
                feat=model.outcome.features(bp,emb).float();reward=model.outcome.sample_reward(feat,gen)
                done=torch.rand(B,device=dev,generator=gen)>torch.sigmoid(model.outcome.cont_logit(feat,reward).float())
                out.append(torch.stack([ee.argmax(-1).float(),(ss>.5).float(),done.float(),alive.float(),ee.max(-1).values,ss],-1).cpu())
                raw.append(tuple(t.half().cpu() for t in (p,g,v)))
                cache=M.context_pass(model.denoiser,tuple(t[:,None] for t in (p,g,v)),action,S.cond(model.belief.memory(bp)[:,None],S.at(rs,np.s_[:,None])),fix[:,None],cache,cfg,gen)
                y,carry=model.belief(model.belief.tokens(emb[:,None],reward[:,None],done[:,None],action),carry)
                bp=y[:,-1].float();alive &= ~done;rs=rs_next
        results.append(torch.stack(out,1).reshape(len(batch),k,h,6))
        part_records.append(raw)
        print('native scenes',start+len(batch),'/',len(rows),flush=True)
    out=dict(rows=rows,truth=bank['truth'][:len(rows),:k,:h],generated=torch.cat(results).numpy(),
        generated_columns=['energy','asleep','done','active','energy_reader_confidence','asleep_probability'],
        readers_valid=valid,prefix_sham_max=max(shams),parts=part_records,H=h,K=k,checkpoint_sha256=TR.file_sha256(TR.ROOT/CKPT))
    out['reader_stress']=reader_stress
    out['reader_states']={str(q):r.state_dict() for q,r in readers.items()}
    torch.save(out,HERE/('smoke.pt' if smoke else 'run.pt'))

if __name__=='__main__':
    if sys.argv[1]=='build':build()
    else:run('--smoke' in sys.argv)
