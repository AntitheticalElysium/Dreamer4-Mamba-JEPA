"""Correctness-first adaptive collector prototype, not an overnight launcher.
Per-stream caches support asynchronous resets correctly; throughput is a measured lower
performance bound, not a claim about an optimized batched implementation.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
import copy,importlib,json,time
from pathlib import Path
import numpy as np
import torch
from saccade import data as D,model as M,train as TR
GC=importlib.import_module('artifacts.experiments.20261102_p1f15_readiness.gaze_checks')
HERE=Path(__file__).resolve().parent;DEV=torch.device('cuda')


class Collector:
    def __init__(self):
        self.ng=torch.Generator(device=DEV).manual_seed(611601)
        self.fg=torch.Generator(device=DEV).manual_seed(611602)
        self.pre_action={}

    @torch.no_grad()
    def collect(self,m,cfg,chunk,carry,bcache,streams,enabled=True):
        if not enabled:return copy.deepcopy(chunk),None
        initial_cpu=torch.get_rng_state();initial_gpu=torch.cuda.get_rng_state()
        c=copy.deepcopy(chunk);P=cfg['data']['context_prefix'];outcarry=[];bchecks=[];resets=0;boundary_errors=[]
        start=time.time()
        for i in range(len(c['valid'])):
            cc=m.denoiser.new_cache();state=M.carry_rows(carry,[i]);bp=bcache[i:i+1,P].clone()
            bn=bp.clone()
            with TR.autocast(DEV,cfg):
                # Parameter-dependent K/V rebuilt every collection from raw prefix and cached predicted beliefs.
                for j in range(P):
                    if not c['valid'][i,j]:continue
                    if c['first'][i,j]:cc=m.denoiser.new_cache()
                    fx=torch.as_tensor(c['fixation'][i:i+1,j],device=DEV)
                    pp=m.perception(torch.as_tensor(c['frames'][i:i+1,j],device=DEV),fx)
                    cc=M.context_pass(m.denoiser,tuple(x[:,None] for x in pp.parts()),
                        torch.as_tensor(c['prev_action'][i:i+1,j:j+1],device=DEV),m.belief.memory(bcache[i:i+1,j:j+1]),fx[:,None],cc,cfg,self.ng)
                for j in range(P,c['valid'].shape[1]):
                    if c['first'][i,j]:
                        state=m.belief.initial(1,DEV);bp=torch.zeros_like(bp);bn=bp.clone();cc=m.denoiser.new_cache();resets+=1
                    fr=torch.as_tensor(c['frames'][i:i+1,j],device=DEV)
                    dummy=GC.CENTRES[:1];pp=m.perception(fr,dummy)
                    pa=torch.as_tensor(c['prev_action'][i:i+1,j],device=DEV)
                    a=torch.as_tensor(c['action'][i:i+1,j],device=DEV)
                    r=torch.as_tensor(c['reward'][i:i+1,j],device=DEV)
                    d=torch.as_tensor(c['terminated'][i:i+1,j],device=DEV)
                    score_bp=bp
                    if j==P and not c['first'][i,j] and i in self.pre_action:
                        # Recompute BOTH counterfactuals from the same saved pre-action carry under
                        # current parameters. Mixing native cached actual output with a separately
                        # computed NOOP output invents contingency at a chunk boundary, even for NOOP.
                        post=self.pre_action[i]
                        rep=tuple(M.MambaCarry(z.conv.repeat_interleave(2,0),z.ssm.repeat_interleave(2,0)) for z in post)
                        ab=torch.stack([pa,torch.zeros_like(pa)],1).flatten()
                        ypair,_=m.belief(m.belief.action(ab)[:,None],rep,backend='reference')
                        score_bp=ypair[0:1,0].float();bn=ypair[1:2,0].float()
                        boundary_errors.append(float((score_bp-bp).square().mean().sqrt()))
                    C,S=GC.signals(m,cfg,pp.p,score_bp,bn,pa,dummy,cc,self.ng)
                    prob=GC.priorities(C,S);ix=torch.multinomial(prob,1,generator=self.fg)[:,0];fx=GC.CENTRES[ix]
                    pp=m.perception(fr,fx);c['fixation'][i,j]=fx[0].cpu().numpy()
                    cc=M.context_pass(m.denoiser,tuple(x[:,None] for x in pp.parts()),pa[:,None],m.belief.memory(bp)[:,None],fx[:,None],cc,cfg,self.ng)
                    embed=m.belief.percept_embed(pp)
                    pt=embed+m.belief.outcome_in(torch.stack([M.SYMLOG(r),d.float()],-1).to(embed.dtype))
                    _,post=m.belief(pt[:,None],state,backend='reference')
                    self.pre_action[i]=post
                    rep=tuple(M.MambaCarry(z.conv.repeat_interleave(2,0),z.ssm.repeat_interleave(2,0)) for z in post)
                    yy,states=m.belief(m.belief.action(torch.stack([a,torch.zeros_like(a)],1).flatten())[:,None],rep,backend='reference')
                    bp=yy[0:1,0].float();bn=yy[1:2,0].float()
                    state=tuple(M.MambaCarry(z.conv[0:1].contiguous(),z.ssm[0:1].contiguous()) for z in states)
                outcarry.append(state);bchecks.append(bp)
        # Prefix records must contain the glimpses that were actually read, never their discarded random draws.
        for i,h in enumerate(streams.hist):
            # hist is precisely the last P emitted new records; gather order is identical.
            for k,r in enumerate(h):
                j=c['valid'].shape[1]-len(h)+k
                assert r[1]==int(c['episode'][i,j])
                h[k]=(*r[:3],*map(int,c['fixation'][i,j]))
        assert torch.equal(initial_cpu,torch.get_rng_state()) and torch.equal(initial_gpu,torch.cuda.get_rng_state()),'chooser consumed global training RNG'
        return c,{'seconds':time.time()-start,'resets':resets,'bp':torch.cat(bchecks),'carry':outcarry,
                  'boundary_score_bp_rms':boundary_errors}


def main():
    torch.set_num_threads(4);m,cfg,path=GC.load(10);m.requires_grad_(True);m.train();m.denoiser.checkpoint=True
    eps=D.load_split(cfg)['train'];offsets=D.frame_offsets(eps)
    targets=np.load(TR.ROOT/cfg['p1']['out_dir']/'targets/change_targets.npy',mmap_mode='r')
    opt=torch.optim.AdamW(m.param_groups(),lr=cfg['optimizer']['lr'],betas=tuple(cfg['optimizer']['betas']),eps=cfg['optimizer']['eps'])
    result={'checkpoint_sha256':TR.file_sha256(path),'prototype':True,'streams':2,'phases':[]}
    for steps in (64,256):
        # Two real training streams, two consecutive native updates; no model changes are persisted.
        streams=D.Streams(eps,2,8,1,offsets);control=D.Streams(eps,2,8,1,offsets);collector=Collector()
        carry=m.belief.initial(2,DEV);cache=torch.zeros(2,9,cfg['belief']['width'],device=DEV)
        for update in range(2):
            chunk=streams.next(steps);base=control.next(steps)
            # Choices must not alter episode order or any physical data / outcome field.
            for k in chunk:
                if update and k=='fixation':continue
                assert np.array_equal(chunk[k],base[k],equal_nan=True),(k,update)
            disabled,_=collector.collect(m,cfg,chunk,carry,cache,streams,False)
            assert all(np.array_equal(chunk[k],disabled[k],equal_nan=True) for k in chunk)
            guided,stats=collector.collect(m,cfg,chunk,carry,cache,streams)
            assert all(np.array_equal(chunk[k],guided[k],equal_nan=True) for k in chunk if k!='fixation')
            before=torch.cuda.Event(enable_timing=True);after=torch.cuda.Event(enable_timing=True)
            before.record();opt.zero_grad(set_to_none=True)
            terms,newcarry,newcache=TR.p1_losses(m,guided,carry,cache,targets,cfg)
            loss=m.harmoniser(TR.groups(terms))+terms['noise'];loss.backward()
            grad={}
            for name,mod in (('fovea',m.perception.fovea),('periphery',m.perception.p_layers),('belief',m.belief),
                             ('denoiser',m.denoiser),('TSP',m.tsp),('pixels',m.pixels),('outcome',m.outcome)):
                norm=sum(float(p.grad.float().square().sum()) for p in mod.parameters() if p.grad is not None)**.5
                assert norm>0 and np.isfinite(norm),(name,norm);grad[name]=norm
            err=float((newcache[:,-1]-stats['bp']).square().mean(-1).sqrt().max());assert err<.03,err
            torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['optimizer']['grad_clip']);opt.step();after.record();torch.cuda.synchronize()
            carry=tuple(M.MambaCarry(z.conv.detach(),z.ssm.detach()) for z in newcarry);cache=newcache
            result['phases'].append({'steps':steps,'update':update,'collection_seconds':stats['seconds'],'resets':stats['resets'],
                'loss':float(loss.detach()),'gradients':grad,'collection_vs_training_final_bp_rms':err,
                'boundary_score_bp_rms':stats['boundary_score_bp_rms'],'native_train_seconds':before.elapsed_time(after)/1000})
            print(result['phases'][-1],flush=True)
    # Callback sees exactly the clean generated p/v returned by the cascade, before glimpse-conditioning noise.
    seen={};m.eval()
    def choose(p,v):seen['p']=p.clone();seen['v']=v.clone();return torch.tensor([[[4,4]]],device=DEV)
    with torch.no_grad(),TR.autocast(DEV,cfg):
        parts,fx=M.sample(m.denoiser,[(81,16),(9,64),(8,)],torch.zeros(1,1,device=DEV,dtype=torch.long),
            m.belief.memory(torch.zeros(1,1,256,device=DEV)),torch.ones(1,1,2,device=DEV,dtype=torch.long),m.denoiser.new_cache(),cfg,
            torch.Generator(device=DEV).manual_seed(611610),choose)
    assert torch.equal(parts[0],seen['p']) and torch.equal(parts[2],seen['v'])
    result['clean_generated_callback_identity']=True
    result['limitation']='per-stream correctness prototype, not optimized batched collector; no production launcher qualification'
    (HERE/'collector_preflight.json').write_text(json.dumps(result,indent=1))

if __name__=='__main__':main()
