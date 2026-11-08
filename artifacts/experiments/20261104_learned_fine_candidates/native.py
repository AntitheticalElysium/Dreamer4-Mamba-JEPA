"""Qualify known semantic persistence before attempting any route interpretation."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu');os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import importlib,json,sys
from pathlib import Path
import numpy as np
import torch
from craftax.craftax_classic.constants import BlockType
N=importlib.import_module('artifacts.experiments.20261104_fine_causal_design.test_native')
M,TR,D,S,DEV=N.M,N.TR,N.D,N.S,N.DEV
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]

def write(name,x): (HERE/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def load():
    m,cfg=N.load();cached=torch.load(ROOT/'artifacts/experiments/20261103_fusion_retention/p1f16/dev.pt',map_location='cpu',weights_only=False,mmap=True)
    assert cached['checkpoint_sha256']==N.SHA
    z=cached['current']
    reader=S.load_reader(N.CK.parent/'s1/results.reader.pt',N.CK,DEV)
    saved=torch.load(ROOT/'artifacts/experiments/20261103_p1f16_review/p1f16/readers.pt',map_location=DEV,weights_only=False)
    fine=[]
    for seed in [0,1]:
        q=saved[f'g_{seed}'];r=S.MLP(q['dim'],{'kind':5}).to(DEV);r.load_state_dict(q['state']);r.eval();fine.append(r)
    return m,cfg,z,reader,fine

@torch.no_grad()
def event(reader,parts,r):
    cells=r['target_cells'];p=parts[0][:,0,cells].float();v=parts[2][:,0,None].expand(-1,len(cells),-1).float()
    logits=reader(torch.cat([p,v],-1))['kind' if r['family']=='mob' else 'terr'].float()
    return dict(event=(logits.argmax(-1)==r['ky']).any(-1).cpu(),score=logits.softmax(-1)[...,r['ky']].max(-1).values.cpu())

@torch.no_grad()
def qualify():
    m,cfg,z,reader,fine=load();source=torch.load(HERE/'sources.pt',map_location='cpu',weights_only=False);out=[]
    for i,row in enumerate(source['rows']):
        r=row['meta'];fx=N.gazes(r['cell'])[:1].expand(2,-1);frames=torch.tensor(row['current'],device=DEV)
        with TR.autocast(DEV,cfg):
            P=m.perception(frames,fx);F=m.perception(torch.tensor(row['future'],device=DEV),torch.tensor([7,1],device=DEV).expand(2,-1))
        # Both current alternatives must be read, not merely the factual scene.
        pv=torch.cat([P.p[:,r['cell']].float(),P.v.float()],-1);coarse=reader(pv)
        cp=coarse['kind' if r['family']=='mob' else 'terr'].argmax(-1).tolist()
        slot=int((M.window(fx[:1])[0]==r['cell']).nonzero()[0]);fp=[]
        if r['family']=='mob':
            gv=torch.cat([P.g[:,slot].float(),P.v.float()],-1);fp=[rd(gv)['kind'].argmax(-1).tolist() for rd in fine]
            fine_ok=all(p[0]==r['ky'] and p[1]!=r['ky'] for p in fp);coarse_ok=cp[0]==r['ky'] and cp[1]!=r['ky'];absent_ok=cp[1]==0
        else:
            with TR.autocast(DEV,cfg):mean,_=m.pixels(m.tsp.reconstruct(P.g,P.v,M.window(fx)),P.v,M.window(fx))
            target=M.cell_pixels(M.crop(frames,fx)).float()[:,slot];mask=target[0]!=target[1]
            assert mask.any(), 'terrain alternatives have identical source pixels'
            decoded=(mean[:,slot].float()+1)*127.5
            dd=((decoded[:,None,mask]-target[None,:,mask])/255).square().mean(-1);fp=dd.argmin(-1).tolist()
            fine_ok=fp==[0,1];coarse_ok=cp==[r['ky'],BlockType.GRASS.value];absent_ok=cp[1]==BlockType.GRASS.value
        target=event(reader,tuple(x[:,None] for x in F.parts()),r)['event'].tolist();target_ok=target==[True,False]
        ff=int(torch.where((z['episode']==r['e'])&(z['t']==r['t']))[0])
        pd=float((P.p[0].half().cpu()-z['p'][ff]).abs().max());assert pd<.03,('cached source parity',pd,r)
        source_missed_verified=cp[0]==0 if r['family']=='mob' else cp[0]!=r['ky']
        rr=dict(index=i,meta=r,current_coarse=cp,fine_readings=fp,fine_ok=fine_ok,coarse_ok=coarse_ok,absent_coarse_ok=absent_ok,source_missed_verified=source_missed_verified,
            target_readings=target,target_ok=target_ok,cached_periphery_max_difference=pd)
        out.append(rr)
    write('qualification.json',out)
    for family in sorted({q['meta']['family']+'/'+q['meta']['kind'] for q in out}):
        a=[q for q in out if q['meta']['family']+'/'+q['meta']['kind']==family]
        print(family,'clear',sum(not q['meta']['missed'] and q['coarse_ok'] and q['target_ok'] for q in a),
            'missed',sum(q['meta']['missed'] and q['fine_ok'] and q['absent_coarse_ok'] and q['target_ok'] for q in a),flush=True)

@torch.no_grad()
def prepare(m,cfg,z,row,source_id,mode):
    r=row['meta'];e,t=r['e'],r['t'];off=int(torch.where((z['episode']==e)&(z['t']==t))[0]);carry=N.prefix(m,cfg,z,e,t)
    frames=torch.tensor(row['current'],device=DEV);gazes=N.gazes(r['cell'])
    if mode=='gate':fs=gazes[:1].expand(2,-1);frames=frames
    else:fs=gazes.repeat(2,1) if mode=='gaze' else gazes[:1].expand(4,-1);frames=frames.repeat_interleave(2,0)
    B=len(frames)
    with TR.autocast(DEV,cfg):
        reference=M.Percept(*(z[k][off:off+1].to(DEV).float() for k in ['p','g','v']),z['fix'][off:off+1].to(DEV))
        reftok=m.belief.tokens(m.belief.percept_embed(reference)[:,None],z['reward'][off:off+1].to(DEV)[None],
            z['terminated'][off:off+1].to(DEV)[None],z['action'][off:off+1].to(DEV)[None])
        refy,_=m.belief(reftok,carry)
        prefix_rms=float((refy[0,0].float()-z['corrected'][off].to(DEV).float()).square().mean().sqrt())
        assert prefix_rms<.03,('full-prefix replay check',prefix_rms,e,t)
        P=m.perception(frames,fs)
        if mode=='content':P=M.Percept(P.p,P.g[[0,2,0,2]],P.v,fs)
        emb=m.belief.percept_embed(P);tokens=m.belief.tokens(emb[:,None],z['reward'][off].to(DEV).expand(B,1),z['terminated'][off].to(DEV).expand(B,1),
            torch.full((B,1),r['action'],device=DEV))
        yy,cc=m.belief(tokens,N.expandcarry(carry,B));memory=m.belief.memory(yy[:,1].float())
        cache=m.denoiser.new_cache();gen=torch.Generator(device=DEV).manual_seed(615200+source_id)
        for f in range(off-7,off):
            cache=M.context_pass(m.denoiser,tuple(z[k][f:f+1].to(DEV).float()[None] for k in ['p','g','v']),z['prev_action'][f:f+1].to(DEV)[None],
                m.belief.memory(z['prior'][f:f+1].to(DEV).float())[None],z['fix'][f:f+1].to(DEV)[None],cache,cfg,gen)
        cache=N.expandcache(cache,B);sc=cfg['edm']['inference_context_sigma'];xs=tuple(x[:,None]+sc*torch.randn((1,1,*x.shape[1:]),device=DEV,generator=gen) for x in P.parts())
        _,_,ci,cn=M.precondition(torch.full((B,1),sc,device=DEV),cfg['edm']['sigma_data'])
        _,new=m.denoiser.step(tuple(M._bcast(ci,x)*x for x in xs),cn,z['prev_action'][off].to(DEV).expand(B,1),m.belief.memory(z['prior'][off].to(DEV).float()).expand(B,1,-1,-1),fs[:,None],cache)
        if mode!='gate':
            for k,v in new:
                for b in [0,2]:
                    assert torch.equal(k[b,m.denoiser.phys],k[b+1,m.denoiser.phys])
                    assert torch.equal(v[b,m.denoiser.phys],v[b+1,m.denoiser.phys])
        cache=m.denoiser.commit(cache,new)
    pair=[(0,1)] if B==2 else [(0,1),(2,3)]
    stages=dict(prefix_rms=prefix_rms,embedding=[N.delta(emb[a],emb[b]) for a,b in pair],predicted=[N.delta(yy[a,1],yy[b,1]) for a,b in pair],
        full_ssm=[all(torch.equal(c.ssm[a],c.ssm[b]) for c in cc) for a,b in pair],
        carry_ssm=[[N.delta(c.ssm[a],c.ssm[b]) for c in cc] for a,b in pair],
        carry_conv=[[N.delta(c.conv[a],c.conv[b]) for c in cc] for a,b in pair])
    return cache,memory,stages

@torch.no_grad()
def predict(m,cfg,reader,row,cache,memory,index,K):
    r=row['meta'];out=[]
    for b in range(0,K,16):
        n=min(16,K-b)
        for c in range(len(memory)):
            gen=torch.Generator(device=DEV).manual_seed(615300+1000*index+b)
            one=dict(t=cache['t'],kv=[(k[c:c+1],v[c:c+1]) for k,v in cache['kv']])
            with TR.autocast(DEV,cfg):
                parts=M.sample(m.denoiser,[(81,16),(9,64),(8,)],torch.full((n,1),r['action'],device=DEV),memory[c:c+1,None].expand(n,-1,-1,-1),
                    torch.tensor([7,1],device=DEV).expand(n,1,2),N.expandcache(one,n),cfg,gen)
            out.append(dict(condition=c,batch=b,**event(reader,parts,r)))
    return out

def gate(smoke=False):
    m,cfg,z,reader,_=load();source=torch.load(HERE/'sources.pt',weights_only=False,map_location='cpu')['rows'];q=json.loads((HERE/'qualification.json').read_text())
    useful={r['meta']['family']+'/'+r['meta']['kind'] for r in q if r['meta']['missed'] and r['source_missed_verified'] and r['fine_ok'] and r['absent_coarse_ok'] and r['target_ok']}
    accepted=[r for r in q if r['meta']['family']+'/'+r['meta']['kind'] in useful and not r['meta']['missed'] and r['coarse_ok'] and r['target_ok']]
    if smoke:accepted=accepted[:1]
    result=[]
    for a in accepted:
        i=a['index'];row=source[i];cache,memory,stages=prepare(m,cfg,z,row,i,'gate');v=predict(m,cfg,reader,row,cache,memory,i,4 if smoke else 32)
        result.append(dict(index=i,meta=row['meta'],draws=v,stages=stages));torch.save(result,HERE/('gate_smoke.pt' if smoke else 'gates.pt'))
        print('gate',len(result),'/',len(accepted),row['meta']['family'],row['meta']['kind'],flush=True)

def trace():
    decision=json.loads((HERE/'gate_decisions.json').read_text());passed={k for k,v in decision.items() if v['pass']}
    m,cfg,z,reader,_=load();source=torch.load(HERE/'sources.pt',weights_only=False,map_location='cpu')['rows'];q=json.loads((HERE/'qualification.json').read_text());out=[]
    selected=[r for r in q if r['meta']['family']+'/'+r['meta']['kind'] in passed and r['meta']['missed'] and r['source_missed_verified'] and r['fine_ok'] and r['absent_coarse_ok'] and r['target_ok']]
    for a in selected:
        i=a['index'];row=source[i]
        for mode in ['gaze','content']:
            cache,memory,stages=prepare(m,cfg,z,row,i,mode);v=predict(m,cfg,reader,row,cache,memory,i,64)
            out.append(dict(index=i,mode=mode,meta=row['meta'],draws=v,stages=stages));torch.save(out,HERE/'traces.pt')
        print('trace',len(out)//2,'/',len(selected),row['meta']['family'],row['meta']['kind'],flush=True)
    if not selected:torch.save([],HERE/'traces.pt')

if __name__=='__main__':
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True
    if sys.argv[1]=='qualify':qualify()
    elif sys.argv[1]=='gate':gate('--smoke' in sys.argv)
    else:trace()
