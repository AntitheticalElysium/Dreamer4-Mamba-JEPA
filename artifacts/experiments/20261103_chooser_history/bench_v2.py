"""Actual per-frame fine-belief feedback cost; no world optimizer updates."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MAMBA_DETERMINISTIC','1')
os.environ.setdefault('TORCHINDUCTOR_CACHE_DIR','/tmp/saccade-history-inductor')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS','2')
import argparse
import copy
import importlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F

R=importlib.import_module('artifacts.experiments.20261103_chooser_history.run')
C,Q,E,HERE=R.C,R.Q,R.E,R.HERE
from saccade import model as M,train as TR,data as D


def sync():torch.cuda.synchronize()


def detach(carry):
    return tuple(M.MambaCarry(x.conv.detach(),x.ssm.detach()) for x in carry)


def initial(m,batch):
    # Native convolution stores bf16 projected inputs; keep that lossless cache
    # dtype from startup, while the SSM accumulator remains float32.
    return tuple(M.MambaCarry(c.conv.bfloat16(),c.ssm) for c in m.belief.initial(batch,torch.device('cuda')))


@torch.no_grad()
def coarse(m,frames,cfg):
    # Same native periphery, cached once for every frame. Dummy fine output is
    # unnecessary but included equally in both collectors' measured cache cost.
    with TR.autocast(torch.device('cuda'),cfg):
        p=m.perception(frames,torch.ones(*frames.shape[:2],2,device=frames.device,dtype=torch.long))
    return p.p.detach(),p.v.detach()


def inputs(p,v,t,h,aux,mean,std):
    # p/v are the PREVIOUS coarse frame, not the target being scored.
    ctx=torch.cat([v.float(),F.one_hot((t['previous_action']+1).clamp(0,17),18).float(),
                   M.SYMLOG(t['previous_reward'].float())[...,None],t['previous_term'].float()[...,None],
                   t['previous_valid'].float()[...,None]],-1)
    valid=t['previous_valid'].float()
    return (p.float()*valid[...,None,None],ctx*valid[...,None],t['query'].clamp(min=0),
            (aux.half().float()-mean)/std)


@torch.no_grad()
def choose(h,x,target,first,gen):
    # The two action queries share information and can be evaluated together.
    pair=tuple(a.repeat_interleave(2,0) for a in x)
    action=torch.stack([x[2],torch.zeros_like(x[2])],1).flatten()
    both=h((pair[0],pair[1],action,pair[3])).view(len(target),2,81,16)
    out,no=both[:,0],both[:,1]
    c=(out-no).square().sum(-1)[:,Q.WIN.to(target.device)].sum(-1)
    s=(out-target.float()).square().sum(-1)[:,Q.WIN.to(target.device)].sum(-1)
    prob=Q.policy(c,s)
    prob=torch.where(first[:,None],torch.full_like(prob,1/49),prob)
    ix=torch.multinomial(prob,1,generator=gen)[:,0]
    return Q.CENTRES.to(target.device)[ix],prob


class Collector:
    def __init__(self,m,cfg,h,mean,std,kind,step):
        self.m,self.cfg,self.h,self.mean,self.std=m,cfg,h,mean,std
        self.kind,self.step=kind,step
        gg=torch.Generator().manual_seed(611730)
        self.w=(torch.randn(81*16,256,generator=gg)/(81*16)**.5).cuda()

    @torch.no_grad()
    def collect(self,chunk,carry,before,gen):
        m,cfg=self.m,self.cfg;P=cfg['data']['context_prefix']
        t=TR.tensors(chunk,torch.device('cuda'));frames=t['frames'];B,L=frames.shape[:2]
        p,v=coarse(m,frames,cfg);fix=t['fixation'].clone()
        oldcarry=carry
        if self.kind=='coarse':
            # All choices can be made simultaneously. Only the previous coarse
            # frame and its observed context are supplied to the one-slot head.
            pv=p[:,P-1:L-1].reshape(-1,81,16);vv=v[:,P-1:L-1].reshape(-1,8)
            first=t['first'][:,P:].flatten();valid=~first
            meta=dict(previous_action=t['prev_action'][:,P-1:L-1].flatten(),
                      previous_reward=t['reward'][:,P-1:L-1].flatten(),
                      previous_term=t['terminated'][:,P-1:L-1].flatten(),previous_valid=valid,
                      query=t['prev_action'][:,P:].flatten())
            aux=(pv.float().flatten(1)@self.w).half();aux[~valid]=0
            x=inputs(pv,vv,meta,self.h,aux,self.mean,self.std)
            x=(x[0][:,None],x[1],x[2],x[3])
            fx,_=choose(self.h,x,p[:,P:].flatten(0,1),first,gen)
            fix[:,P:]=fx.view(B,L-P,2)
        else:
            for j in range(P,L):
                first=t['first'][:,j];valid=~first
                meta=dict(previous_action=t['prev_action'][:,j-1],previous_reward=t['reward'][:,j-1],
                          previous_term=t['terminated'][:,j-1],previous_valid=valid,query=t['prev_action'][:,j])
                aux=before*valid[:,None]
                x=inputs(p[:,j-1],v[:,j-1],meta,self.h,aux,self.mean,self.std)
                x=(x[0][:,None],x[1],x[2],x[3])
                fx,_=choose(self.h,x,p[:,j],first,gen);fix[:,j]=fx
                with TR.autocast(torch.device('cuda'),cfg):
                    crop=M.crop(frames[:,j],fx).permute(0,3,1,2).float()/127.5-1
                    g=M.out_map(m.perception.fovea(crop).flatten(2).transpose(1,2),m.perception.s)
                    pp=M.Percept(p[:,j:j+1],g[:,None],v[:,j:j+1],fx[:,None])
                    tok=m.belief.tokens(m.belief.percept_embed(pp),t['reward'][:,j:j+1],
                                       t['terminated'][:,j:j+1],t['action'][:,j:j+1])
                    seq=first.int()[:,None].expand(-1,2).contiguous()
                    y,carry=self.step(tok,carry,seq)
                    before=y[:,0].float()
        out={**chunk,'fixation':fix.cpu().numpy()}
        return out,detach(carry),before,oldcarry


def metric(a,b):return float((a.float()-b.float()).norm()/b.float().norm().clamp(min=1e-8))


@torch.no_grad()
def validate(m,cfg,chunk,incoming,final,before):
    P=cfg['data']['context_prefix'];t=TR.tensors(chunk,torch.device('cuda'))
    with TR.autocast(torch.device('cuda'),cfg):
        pc=m.perception(t['frames'][:,P:],t['fixation'][:,P:])
        tok=m.belief.tokens(m.belief.percept_embed(pc),t['reward'][:,P:],t['terminated'][:,P:],t['action'][:,P:])
        seq=t['first'][:,P:].int().cumsum(1).repeat_interleave(2,1)
        y,native=m.belief(tok,incoming,seq)
    r=dict(final_corrected_belief_relative_error=metric(before,y[:,-2]),
           ssm_relative_error=max(metric(a.ssm,b.ssm) for a,b in zip(final,native)))
    assert r['final_corrected_belief_relative_error']<.03,r
    assert r['ssm_relative_error']<.03,r
    return r


def batch_inputs(m,cfg,streams,steps):
    eps=D.load_split(cfg)['train'];off=D.frame_offsets(eps)
    stream=D.Streams(eps,streams,8,611861,off)
    return [stream.next(steps) for _ in range(3)]


def stat(x):return dict(median=float(np.median(x)),min=float(min(x)),max=float(max(x)),values=[float(v) for v in x])


def core(compiled):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    ck=torch.load(C.CKDIR/'ckpt_10.pt',map_location='cpu',weights_only=False);cfg=ck['cfg']
    m=M.M0(cfg).cuda();m.load_state_dict(ck['model']);m.eval();m.denoiser.checkpoint=True
    initial_sha=R.weight_sha(m)
    states={k:torch.load(HERE/f'1{k}_611601_u12000.pt',map_location='cpu',weights_only=False) for k in ('C','B')}
    heads={};checks={}
    for kind,s in states.items():
        h=E.AuxHead().cuda();h.load_state_dict(s['state']);h.eval().requires_grad_(False)
        heads[kind]=R.compact(h)
        gen=torch.Generator(device='cuda').manual_seed(611862)
        x=(torch.randn(16,8,81,16,device='cuda',generator=gen),torch.randn(16,8*29,device='cuda',generator=gen),
           torch.randint(0,17,(16,),device='cuda',generator=gen),torch.randn(16,256,device='cuda',generator=gen))
        masked=R.mask_history(x,1);short=(masked[0][:,-1:],masked[1].view(16,8,29)[:,-1],x[2],x[3])
        with torch.no_grad():
            err=float((h(masked)-heads[kind](short)).abs().max())
        checks['compact_'+kind+'_IEEE_error']=err
        # Preserve the failed3e-6 attempt. Separately establish mathematical
        # equality in float64, and use actual policy distortion as the safety
        # check, rather than widening a raw-coordinate threshold after a failure.
        hd=copy.deepcopy(h).double();cd=copy.deepcopy(heads[kind]).double()
        with torch.no_grad():
            xd=tuple(a.double() if a.is_floating_point() else a for a in masked)
            sd=tuple(a.double() if a.is_floating_point() else a for a in short)
            de=float((hd(xd)-cd(sd)).abs().max())
        assert de<1e-10,de
        del hd,cd
        dv=E.load_cache(10,'dev',torch.device('cpu'))
        aux=(E.sketch(dv) if kind=='C' else torch.load(C.HERE/'dev_preaction_belief.pt',map_location='cpu',weights_only=False)['aux'])
        dd=R.Data(dv,torch.device('cpu'),aux,s['mean'],s['std'],history=1)
        meta=C.cohorts()['confirmation'];allids=dd.row_ids(meta['ep'],meta['t'])
        ii=allids[torch.linspace(0,len(allids)-1,256).long()]
        xx=tuple(a.cuda() for a in dd.inputs(ii));ss=(xx[0][:,-1:],xx[1].view(len(ii),8,29)[:,-1],xx[2],xx[3])
        actual=dv['p'][ii].float().cuda()
        def policy_of(hh,zz):
            with torch.no_grad():
                y=hh(zz);nn=hh((*zz[:2],torch.zeros_like(zz[2]),zz[3]))
                c=(y-nn).square().sum(-1)[:,Q.WIN.cuda()].sum(-1)
                se=(y-actual).square().sum(-1)[:,Q.WIN.cuda()].sum(-1)
                return Q.policy(c,se)
        tv=.5*(policy_of(h,xx)-policy_of(heads[kind],ss)).abs().sum(-1)
        assert float(tv.max())<1e-4,float(tv.max())
        checks['compact_'+kind+'_float64_error']=de
        checks['compact_'+kind+'_real_policy_max_TV']=float(tv.max())
        del dd,dv,aux,xx,ss,actual
    def reference(x,c,s):return m.belief(x,c,s,backend='reference')
    def triton(x,c,s):return m.belief(x,c,s,backend='triton')
    step=torch.compile(reference,fullgraph=True,dynamic=False) if compiled else reference
    targets=np.load(TR.ROOT/cfg['p1']['out_dir']/'targets/change_targets.npy',mmap_mode='r')
    result=dict(compiled=compiled,heads=checks,shapes={},checkpoint_sha256=C.sha(C.CKDIR/'ckpt_10.pt'))
    for B,T in ((16,64),(4,256)):
        chunks=batch_inputs(m,cfg,B,T);shape=f'{B}x{T}'
        result['shapes'][shape]={}
        x=torch.randn(B,2,256,device='cuda',dtype=torch.bfloat16);c=initial(m,B);s=torch.zeros(B,2,device='cuda',dtype=torch.int32)
        # Compilation/warmup excluded from steady timing and reported independently.
        sync();start=time.perf_counter()
        with torch.no_grad(),TR.autocast(torch.device('cuda'),cfg):
            yy,cc=step(x,c,s)
            ref,rc=reference(x,c,s)
        sync();compile_seconds=time.perf_counter()-start
        error=metric(yy,ref)
        assert error<.03,error
        result['shapes'][shape]['compile_or_warmup_seconds']=compile_seconds
        result['shapes'][shape]['step_vs_reference_relative_error']=error
        # Head-only timing is separated from collection/state maintenance.
        for kind in ('C','B'):
            for count in (B,B*T):
                inp=(torch.randn(count,1,81,16,device='cuda'),torch.randn(count,29,device='cuda'),
                     torch.ones(count,device='cuda',dtype=torch.long),torch.randn(count,256,device='cuda'))
                target=torch.randn(count,81,16,device='cuda');first=torch.zeros(count,device='cuda',dtype=torch.bool)
                fg=torch.Generator(device='cuda').manual_seed(611867);times=[]
                for repeat in range(6):
                    sync();tt=time.perf_counter();choose(heads[kind],inp,target,first,fg);sync()
                    if repeat:times.append(time.perf_counter()-tt)
                result['shapes'][shape][f'{kind}_head_{count}_seconds']=stat(times)
        # Eager scan implementations are compared at the native two-token work size.
        for name,fn in [('reference',reference),('triton',triton)]+([('compiled',step)] if compiled else []):
            times=[]
            for repeat in range(5):
                sync();t0=time.perf_counter()
                with torch.no_grad(),TR.autocast(torch.device('cuda'),cfg):out=fn(x,c,s)
                sync()
                if repeat:times.append(time.perf_counter()-t0)
            result['shapes'][shape][name+'_two_token_seconds']=stat(times)
        for kind,tag in [('C','coarse'),('B','rich')]:
            state=states[kind];collector=Collector(m,cfg,heads[kind],state['mean'].cuda(),state['std'].cuda(),tag,step)
            coltimes=[];traintimes=[];resets=[];validation=[]
            # First complete run warms all kernels. Three subsequent chunks give
            # distributions over depth and asynchronous resets, with own gaze history.
            carry=initial(m,B);before=torch.zeros(B,256,device='cuda')
            traincarry=m.belief.initial(B,torch.device('cuda'));cache=torch.zeros(B,9,256,device='cuda')
            previous_fix=None;fg=torch.Generator(device='cuda').manual_seed(611863)
            for i,raw in enumerate(chunks):
                chunk=copy.deepcopy(raw)
                if previous_fix is not None:chunk['fixation'][:,:8]=previous_fix
                m.eval();sync();st=time.perf_counter()
                chosen,carry,before,incoming=collector.collect(chunk,carry,before,fg)
                sync();coltimes.append(time.perf_counter()-st)
                if kind=='B':validation.append(validate(m,cfg,chosen,incoming,carry,before))
                previous_fix=chosen['fixation'][:,-8:].copy()
                resets.append(int(chosen['first'][:,8:].sum()))
                m.train();m.zero_grad(set_to_none=True)
                torch.manual_seed(611864+i);torch.cuda.manual_seed(611864+i)
                torch.backends.cuda.matmul.allow_tf32=True
                sync();st=time.perf_counter()
                terms,newcarry,newcache=TR.p1_losses(m,chosen,traincarry,cache,targets,cfg)
                loss=m.harmoniser(TR.groups(terms))+terms.get('noise',0)
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['optimizer']['grad_clip'])
                sync();traintimes.append(time.perf_counter()-st)
                torch.backends.cuda.matmul.allow_tf32=False
                M.carry_put(traincarry,np.arange(B),newcarry);cache=newcache.detach()
                m.zero_grad(set_to_none=True)
            # Initial chunk includes cold training kernels; keep it but use last2
            # as steady-state. Each shape covers3072 newly emitted frames per arm.
            result['shapes'][shape][kind]=dict(collection=stat(coltimes[1:]),training=stat(traintimes[1:]),
                pipeline=stat([a+b for a,b in zip(coltimes[1:],traintimes[1:])]),
                first_collection_seconds=coltimes[0],first_training_seconds=traintimes[0],
                resets_by_chunk=resets,frames_per_chunk=B*T,validation=validation)
        print('cost shape',shape,'compiled',compiled,result['shapes'][shape]['B']['collection']['median'],flush=True)
        C.write(HERE/('cost_v2_compiled.json' if compiled else 'cost_v2_eager.json'),{**result,'complete':False})
    assert R.weight_sha(m)==initial_sha,'World weights changed during cost measurement'
    result.update(complete=True,world_weights_unchanged=True,peak_GiB=torch.cuda.max_memory_allocated()/2**30)
    C.write(HERE/('cost_v2_compiled.json' if compiled else 'cost_v2_eager.json'),result)


def main():
    p=argparse.ArgumentParser();p.add_argument('--core',action='store_true');p.add_argument('--compiled',action='store_true');args=p.parse_args()
    if args.core or args.compiled:return core(args.compiled)
    report={};start=time.time()
    for name,arg in [('eager','--core'),('compiled','--compiled')]:
        try:
            with open(HERE/f'cost_v2_{name}_detail.log','a') as log:
                subprocess.run([sys.executable,'-u','-m','artifacts.experiments.20261103_chooser_history.bench_v2',arg],
                               stdout=log,stderr=subprocess.STDOUT,check=True,timeout=170)
            report[name]=json.loads((HERE/f'cost_v2_{name}.json').read_text())
        except (subprocess.SubprocessError,OSError) as e:
            path=HERE/f'cost_v2_{name}.json'
            report[name]=(json.loads(path.read_text()) if path.exists() else {})
            report[name].update(error=repr(e),limitation='Failure is not evidence that optimization is impossible.')
    report['seconds']=time.time()-start
    C.write(HERE/'COST_V2.json',report)


if __name__=='__main__':main()
