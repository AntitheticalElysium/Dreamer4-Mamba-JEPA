"""Copy of Codex's 20261102_p1f15_readiness/recursion.py (11-02; logic, window selection, seeds and modes unchanged),
parameterised 11-05 (Claude): --ckpt, --out, --steps (edm.sampler.steps), --windows. Question: how much of the
free-running entity collapse (F1.5: 1.1 -> 0.05 by step 16) is the 3-step sampler's per-step majority rounding
(RESULTS.md 5, 7) compounding through the history? Run on one world at 3 and 16 steps with identical seeds.

Bounded native free-running F1.5 diagnosis on exogenous action windows.

Teacher forcing is a local-history comparator, not a second coherent joint forecast.
No pointwise token error between independently sampled futures and the recording.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
import argparse, importlib, json, time
from pathlib import Path
import numpy as np
import torch
from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

T=importlib.import_module('artifacts.experiments.20261101_regression_localization.trace')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEV=torch.device('cuda')

@torch.no_grad()
def counts(reader,parts):
    p,g,v=parts
    x=torch.cat([p[:,:63].reshape(-1,p.shape[-1]),v.repeat_interleave(63,0)],-1)
    k=reader(x.half())['kind'].argmax(1).reshape(len(p),63)
    return torch.stack([(k[:,S.KEEP]==j).sum(1) for j in range(1,5)],-1).float()

def pick(L,n):
    ao=np.r_[0,np.cumsum(L['ep_T'])]
    c=[(e,t) for e,N in enumerate(L['ep_T']) for t in range(16,int(N)-15,16)
       if L['t_exogenous'][ao[e]+t:ao[e]+t+16].all()]
    rng=np.random.default_rng(20261028)
    return [c[i] for i in sorted(rng.choice(len(c),min(n,len(c)),replace=False))],len(c)

@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--windows',type=int,default=128);ap.add_argument('--ckpt',required=True)
    ap.add_argument('--out',required=True);ap.add_argument('--steps',type=int,required=True);args=ap.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    path=ROOT/args.ckpt
    model,cfg=T.AI.load(path);model.requires_grad_(False)
    import copy;cfg=copy.deepcopy(cfg);cfg['edm']['sampler']['steps']=args.steps
    L=dict(np.load(S.LABELS/'s0_dev_labels.npz'))
    rows,ncan=pick(L,args.windows)
    # Encode only complete episodes required by the selected windows; preserve original label indexing.
    original_eps=sorted({e for e,t in rows});new_of={e:i for i,e in enumerate(original_eps)}
    so=np.r_[0,np.cumsum(L['ep_T']+1)];ao=np.r_[0,np.cumsum(L['ep_T'])]
    sub={}
    for k,v in L.items():
        if k.startswith('ep_'):sub[k]=v[original_eps]
        else:
            off=so if k.startswith('s_') else ao
            sub[k]=np.concatenate([v[off[e]:off[e+1]] for e in original_eps])
    enc=S.encode(model,sub,cfg)
    reader=S.load_reader(path.parent/'s1/results.reader.pt',path,DEV)
    grids=G.episode_grids(L)
    K=4;allrows=[];prefix_errors=[];replay_errors=[]
    for st in range(0,len(rows),16):
        rr=rows[st:st+16];br=[r for r in rr for _ in range(K)]
        take=lambda k,dt:torch.stack([enc[new_of[e]][k][t+dt] for e,t in br]).to(DEV)
        # Rebuild every full real prefix once, retaining the native Mamba carry after action at t0.
        cs=[];bs=[]
        for e,t in rr:
            en=enc[new_of[e]]
            with TR.autocast(DEV,cfg):
                pp=M.Percept(*(en[k][:t+1].to(DEV).float()[None] for k in ('p','g','v')),en['fix'][:t+1].to(DEV)[None])
                emb=model.belief.percept_embed(pp)
                x=model.belief.tokens(emb, en['reward'][:t+1].to(DEV)[None],
                                      en['terminated'][:t+1].to(DEV)[None],en['action'][:t+1].to(DEV)[None])
                y,c=model.belief(x,model.belief.initial(1,DEV))
            cs.append(c);bs.append(y[:,-1].float())
            ref=en['bp'][t+1].to(DEV).float()
            prefix_errors.append(float((y[0,-1].float()-ref).square().mean().sqrt()))
        carry=tuple(M.MambaCarry(torch.cat([c[l].conv for c in cs]).repeat_interleave(K,0),
                                  torch.cat([c[l].ssm for c in cs]).repeat_interleave(K,0))
                    for l in range(len(cs[0])))
        initial_bp=torch.cat(bs).repeat_interleave(K,0)
        cache=model.denoiser.new_cache()
        with TR.autocast(DEV,cfg):
            g=torch.Generator(device=DEV).manual_seed(611240+st)
            for dt in range(-7,1):
                cache=M.context_pass(model.denoiser,tuple(take(k,dt).float()[:,None] for k in ('p','g','v')),
                    take('prev_action',dt)[:,None],model.belief.memory(take('bp',dt).float())[:,None],
                    take('fix',dt)[:,None],cache,cfg,g)
        dims=[tuple(enc[0][k].shape[1:]) for k in ('p','g','v')]
        results={}
        for mode in ('teacher','free','replay'):
            g=torch.Generator(device=DEV).manual_seed(611250+st)
            cc=cache;bp=initial_bp.clone();state=carry
            alive=torch.ones(len(br),device=DEV,dtype=torch.bool);records=[]
            for j in range(1,17):
                fix=take('fix',j);pa=take('prev_action',j)
                with TR.autocast(DEV,cfg):
                    parts=tuple(x[:,0].float() for x in M.sample(model.denoiser,dims,pa[:,None],
                        model.belief.memory(bp)[:,None],fix[:,None],cc,cfg,g))
                    actual=tuple(take(k,j).float() for k in ('p','g','v'))
                    h=model.outcome.features(bp,model.belief.percept_embed(M.Percept(*parts,fix))).float()
                    prob=model.outcome.bin_logp(h).exp()
                    d=T.marginal_death(model.outcome,h,prob)
                    r=model.outcome.sample_reward(h,g)
                    done=torch.rand(len(br),device=DEV,generator=g)>torch.sigmoid(model.outcome.cont_logit(h,r).float())
                    cnt=counts(reader,parts);rcnt=counts(reader,actual)
                    # Absorb only free-running samples. Teacher one-step diagnostics use recorded survival.
                    active=alive.clone() if mode=='free' else torch.stack([
                        ~enc[new_of[e]]['terminated'][t:t+j].any() for e,t in br]).to(DEV)
                    records.append({'count':cnt.cpu().numpy(),'real_reader_count':rcnt.cpu().numpy(),
                        'positive':prob[:,model.outcome.bin_value>.5].sum(-1).cpu().numpy(),
                        'death_probability':d.cpu().numpy(),'reward_sample':r.cpu().numpy(),
                        'death_sample':done.cpu().numpy(),'active':active.cpu().numpy()})
                    if mode=='teacher':
                        context_parts=actual
                        next_bp=take('bp',j+1).float() if j<16 else bp
                    else:
                        evidence=actual if mode=='replay' else parts
                        r_in=take('reward',j).float() if mode=='replay' else r
                        d_in=take('terminated',j) if mode=='replay' else done
                        embed=model.belief.percept_embed(M.Percept(*evidence,fix))[:,None]
                        yy,state=model.belief(model.belief.tokens(embed,r_in[:,None],d_in[:,None],take('action',j)[:,None]),state)
                        next_bp=yy[:,-1].float();context_parts=evidence
                        if mode=='replay' and j<16:
                            err=(next_bp-take('bp',j+1).float()).square().mean(-1).sqrt()
                            replay_errors.extend(err.cpu().tolist())
                            assert err.max()<.03,('real_replay',float(err.max()))
                        alive=alive & ~d_in
                    cc=M.context_pass(model.denoiser,tuple(a[:,None] for a in context_parts),pa[:,None],
                        model.belief.memory(bp)[:,None],fix[:,None],cc,cfg,g)
                    bp=next_bp
            results[mode]={k:np.stack([r[k] for r in records],1) for k in records[0]}
        # With identical real start/cache/randomness, the first forecast in both modes must agree exactly.
        for k in results['teacher']:
            for mode in ('free','replay'):
                assert np.array_equal(results['teacher'][k][:,0],results[mode][k][:,0]),('first_step',mode,k)
        z={f'{m}__{k}':v for m,q in results.items() for k,v in q.items()}
        z['episode']=np.array([e for e,t in br]);z['t0']=np.array([t for e,t in br])
        z['truth_count']=np.stack([np.stack([(grids[e]['grid'][t+1:t+17].reshape(16,-1)[:,S.KEEP]==k).sum(-1)
                                           for k in range(1,5)],-1) for e,t in br])
        z['truth_reward']=np.stack([L['t_reward'][ao[e]+t:ao[e]+t+16] for e,t in br])
        z['truth_death']=np.stack([L['t_terminated'][ao[e]+t:ao[e]+t+16] for e,t in br])
        allrows.append(z)
        np.savez_compressed(out/'raw.npz',**{k:np.concatenate([v[k] for v in allrows]) for k in z})
        print('windows',min(st+16,len(rows)),flush=True)
    (out/'identity.json').write_text(json.dumps({'checkpoint':str(args.ckpt),'sampler_steps':args.steps,'checkpoint_sha256':TR.file_sha256(path),'script_sha256':TR.file_sha256(__file__),
        'plan_sha256':TR.file_sha256(HERE/'PLAN.md'),'windows':rows,'candidate_windows':ncan,'draws':K,
        'first_step_equal':True,'prefix_belief_rms_error_max':max(prefix_errors),
        'prefix_belief_rms_error_mean':float(np.mean(prefix_errors)),
        'real_replay_belief_rms_error_max':max(replay_errors),
        'real_replay_belief_rms_error_mean':float(np.mean(replay_errors))},indent=1))

if __name__=='__main__':main()
