"""Long-history sampler parity, with an actual full-history reference."""
import copy
import io
import json
import math
from pathlib import Path

import torch

from saccade import model as M, train as TR


def main():
    torch.set_num_threads(6)
    torch.manual_seed(0)
    cfg=TR.load_cfg();den=M.Denoiser(cfg).eval()
    for b in den.blocks: torch.nn.init.normal_(b.mod.weight,std=.02)
    for m in den.out: torch.nn.init.normal_(m.weight,std=.02)
    T=15;K=cfg['denoiser']['memory_tokens'];W=cfg['denoiser']['width']
    mem=torch.randn(1,T,K,W);act=torch.randint(0,17,(1,T));fix=torch.randint(1,8,(1,T,2))
    dims=[(81,16),(9,64),(8,)]
    s=cfg['edm']['sampler'];ctxsigma=cfg['edm']['inference_context_sigma'];sd=cfg['edm']['sigma_data']
    sig=[(s['sigma_max']**(1/s['rho'])+i/(s['steps']-1)*(s['sigma_min']**(1/s['rho'])-s['sigma_max']**(1/s['rho'])))**s['rho'] for i in range(s['steps'])]+[0.]
    def rec(hist,t,reps,seed):
        g=torch.Generator().manual_seed(seed)
        sl=lambda a:a[:,t-len(hist):t+1].repeat_interleave(reps,0)
        x=tuple(sig[0]*torch.randn(reps,1,*d,generator=g) for d in dims)
        for k in range(s['steps']):
            gamma=min(s['s_churn']/s['steps'],math.sqrt(2)-1)
            hat=sig[k]*(1+gamma)
            x=tuple(a+math.sqrt(hat*hat-sig[k]*sig[k])*torch.randn(a.shape,generator=g) for a in x)
            seq=tuple(torch.cat([*(h[j] for h in hist),a],1) if hist else a for j,a in enumerate(x))
            sg=torch.full((reps,len(hist)+1),ctxsigma);sg[:,-1]=hat
            cs,co,ci,cn=M.precondition(sg,sd)
            f=den(tuple(M._bcast(ci,a)*a for a in seq),cn,sl(act),sl(mem),sl(fix),torch.zeros_like(sg,dtype=torch.long),torch.ones_like(sg,dtype=torch.bool))
            d=tuple(M.out_map(M._bcast(cs[:,-1:],a)*a+M._bcast(co[:,-1:],a)*ff[:,-1:],3) for a,ff in zip(x,f))
            x=tuple(a+(sig[k+1]-hat)*(a-dd)/hat for a,dd in zip(x,d))
        ctx=tuple(a+ctxsigma*torch.randn(a.shape,generator=g) for a in x)
        return x,ctx
    cache=den.new_cache();histfull=[];hist4=[];hist8=[];reps=1;rows=[]
    with torch.no_grad():
        for t in range(T):
            if t==12:cache=den.new_cache();histfull=[];hist4=[];hist8=[]
            if t==7:
                cache={'t':cache['t'],'kv':[(k.repeat_interleave(2,0),v.repeat_interleave(2,0)) for k,v in cache['kv']]}
                histfull=[tuple(a.repeat_interleave(2,0) for a in h) for h in histfull]
                hist4=[tuple(a.repeat_interleave(2,0) for a in h) for h in hist4]
                hist8=[tuple(a.repeat_interleave(2,0) for a in h) for h in hist8];reps=2
            if t==9:
                buf=io.BytesIO();torch.save(cache,buf);buf.seek(0);cache=torch.load(buf,weights_only=False)
            sl=lambda a:a[:,t:t+1].repeat_interleave(reps,0)
            gen=torch.Generator().manual_seed(719+t)
            xc=M.sample(den,dims,sl(act),sl(mem),sl(fix),cache,cfg,gen)
            cache=M.context_pass(den,xc,sl(act),sl(mem),sl(fix),cache,cfg,gen)
            xf,cf=rec(histfull,t,reps,719+t);x4,c4=rec(hist4,t,reps,719+t);x8,c8=rec(hist8,t,reps,719+t)
            delta=lambda a,b:max(float((u-v).abs().max()) for u,v in zip(a,b))
            rows.append({'frame':t,'cached_vs_full':delta(xc,xf),'cached_vs_recompute4':delta(xc,x4),'cached_vs_recompute8':delta(xc,x8)})
            histfull.append(cf);hist4=(hist4+[c4])[-4:];hist8=(hist8+[c8])[-8:]
    result={'per_frame':rows,'max_cached_vs_full':max(x['cached_vs_full'] for x in rows),
            'max_cached_vs_recompute4':max(x['cached_vs_recompute4'] for x in rows),
            'max_cached_vs_recompute8':max(x['cached_vs_recompute8'] for x in rows)}
    assert result['max_cached_vs_full']<1e-4
    assert result['max_cached_vs_recompute8']<1e-4
    Path(__file__).with_name('long_sampler_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_frame'},indent=2))


if __name__=='__main__':main()
