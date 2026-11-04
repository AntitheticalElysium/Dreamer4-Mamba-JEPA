"""Same-gaze semantic content intervention; pre-result amendment in PLAN.md."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import copy,importlib,json
import numpy as np
import torch
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
M,TR,DEV,HERE=N.M,N.TR,N.DEV,N.HERE

@torch.no_grad()
def main():
    smoke=bool(os.environ.get('SACCADE_CONTENT_SMOKE'))
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;m,cfg=N.load()
    z=torch.load(N.ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',weights_only=False,mmap=True)['current']
    reader=N.S.load_reader(N.CK.parent/'s1/results.reader.pt',N.CK,DEV)
    raw=torch.load(HERE/'native_results.pt',map_location='cpu',weights_only=False)
    rows=[]
    for case,row in enumerate(raw['results']):
        r=row['meta']
        if row['kind']!='natural' or r['cohort']!='distant_arrows' or not r['coarse_missed']:continue
        a=torch.load(HERE/'activations'/f'case_{case:03d}.pt',map_location=DEV,weights_only=False)
        P=M.Percept(a['p'][[0,0,2,2]],a['g'][[0,2,0,2]],a['v'][[0,0,2,2]],a['fix'][:1].expand(4,-1))
        e,t=r['e'],r['t'];off=int(torch.where((z['episode']==e)&(z['t']==t))[0]);carry=N.prefix(m,cfg,z,e,t)
        with TR.autocast(DEV,cfg):
            emb=m.belief.percept_embed(P);tok=m.belief.tokens(emb[:,None],z['reward'][off].to(DEV).expand(4,1),
                z['terminated'][off].to(DEV).expand(4,1),torch.full((4,1),r['action'],device=DEV))
            yy,cc=m.belief(tok,N.expandcarry(carry,4));memory=m.belief.memory(yy[:,1].float())
            cache=m.denoiser.new_cache();gen=torch.Generator(device=DEV).manual_seed(614100+case)
            for f in range(off-7,off):
                cache=M.context_pass(m.denoiser,tuple(z[k][f:f+1].to(DEV).float()[None] for k in ['p','g','v']),
                    z['prev_action'][f:f+1].to(DEV)[None],m.belief.memory(z['prior'][f:f+1].to(DEV).float())[None],
                    z['fix'][f:f+1].to(DEV)[None],cache,cfg,gen)
            cache=N.expandcache(cache,4)
            # Shared context noise; the physical keys are invariant to fine
            # content within each coarse-world pair.
            sc=cfg['edm']['inference_context_sigma'];parts=tuple(x[:,None] for x in P.parts())
            xs=tuple(x+sc*torch.randn((1,*x.shape[1:]),device=DEV,generator=gen) for x in parts)
            _,_,ci,cn=M.precondition(torch.full((4,1),sc,device=DEV),cfg['edm']['sigma_data'])
            _,new=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in xs),cn,z['prev_action'][off].to(DEV).expand(4,1),
                m.belief.memory(z['prior'][off].to(DEV).float()).expand(4,1,-1,-1),P.fix[:,None],cache)
            for k,v in new:
                for b in [0,2]:
                    assert torch.equal(k[b,m.denoiser.phys],k[b+1,m.denoiser.phys])
                    assert torch.equal(v[b,m.denoiser.phys],v[b+1,m.denoiser.phys])
            cache=m.denoiser.commit(cache,new)
        draws=[]
        for steps,churn in [(3,1.),(8,0.)]:
            c=copy.deepcopy(cfg);c['edm']['sampler'].update(steps=steps,s_churn=churn);values=[]
            for b in range(0,4 if smoke else 64,16):
                ns=4 if smoke else 16
                for condition in range(4):
                    seed=614200+case*1000+steps*100+b;gen=torch.Generator(device=DEV).manual_seed(seed)
                    one=dict(t=cache['t'],kv=[(k[condition:condition+1],v[condition:condition+1]) for k,v in cache['kv']])
                    with TR.autocast(DEV,cfg):
                        x=M.sample(m.denoiser,[(81,16),(9,64),(8,)],torch.full((ns,1),r['action'],device=DEV),
                            memory[condition:condition+1,None].expand(ns,-1,-1,-1),torch.tensor([7,1],device=DEV).expand(ns,1,2),
                            N.expandcache(one,ns),c,gen)
                        logits=reader(torch.cat([x[0][:,0,r['target_cell']].float(),x[2][:,0].float()],-1))['kind'].float()
                    values.append(dict(condition=condition,batch=b,arrow=(logits.argmax(-1)==4).cpu(),score=logits.softmax(-1)[:,4].cpu()))
            draws.append(dict(steps=steps,values=values))
        rows.append(dict(case=case,meta=r,embedding_delta=[N.delta(emb[b],emb[b+1]) for b in [0,2]],
            predicted_delta=[N.delta(yy[b,1],yy[b+1,1]) for b in [0,2]],
            carry_ssm=[[N.delta(q.ssm[b],q.ssm[b+1]) for q in cc] for b in [0,2]],draws=draws))
        torch.save(dict(rows=rows,checkpoint_sha256=N.SHA),HERE/('content_smoke.pt' if smoke else 'content_control.pt'))
        print('content',len(rows),'/12',e,t,flush=True)
        if smoke:break

if __name__=='__main__':main()
