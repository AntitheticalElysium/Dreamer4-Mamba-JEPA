"""Final bounded arithmetic optimization: compile scorer as well as recurrent step.

No new fit, changed information, policy weights, random-choice law or refresh rate.
Existing compiled-Mamba timings remain in COST_V2.json. This file writes separately.
"""
import importlib
import json
import time
import torch

B=importlib.import_module('artifacts.experiments.20261103_chooser_history.bench_v2')
HERE=B.HERE


def probabilities(h,x,target,first):
    pair=tuple(a.repeat_interleave(2,0) for a in x)
    act=torch.stack([x[2],torch.zeros_like(x[2])],1).flatten()
    pred=h((pair[0],pair[1],act,pair[3])).view(len(target),2,81,16)
    out,no=pred[:,0],pred[:,1]
    win=B.Q.WIN.to(target.device)
    c=(out-no).square().sum(-1)[:,win].sum(-1)
    s=(out-target.float()).square().sum(-1)[:,win].sum(-1)
    p=B.Q.policy(c,s)
    return torch.where(first[:,None],torch.full_like(p,1/49),p)


def main():
    compiled=torch.compile(probabilities,fullgraph=True,dynamic=False)
    checked={};start=time.time()
    @torch.no_grad()
    def choose(h,x,target,first,gen):
        p=compiled(h,x,target,first)
        key=(id(h),len(target))
        if key not in checked and bool((~first).any()):
            ref=probabilities(h,x,target,first)
            tv=float((.5*(p-ref).abs().sum(-1)).max())
            assert tv<1e-4,tv
            checked[key]=tv
        ix=torch.multinomial(p,1,generator=gen)[:,0]
        return B.Q.CENTRES.to(target.device)[ix],p
    B.choose=choose
    original_write=B.C.write
    def write(path,obj):
        if path.name=='cost_v2_compiled.json':
            path=HERE/'cost_full_compile.json'
            obj={**obj,'scorer_compilation':True,'scorer_policy_max_TV_checks':list(checked.values()),
                 'full_benchmark_seconds':time.time()-start}
        return original_write(path,obj)
    B.C.write=write
    B.core(True)


if __name__=='__main__':main()
