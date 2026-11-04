"""History-length shape cost only, never a policy-quality ablation."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import copy
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch

E=importlib.import_module('artifacts.experiments.20261103_chooser_development.campaign')
R=importlib.import_module('artifacts.experiments.20261103_chooser_history.run')
B=importlib.import_module('artifacts.experiments.20261103_chooser_history.bench_full_compile')
HERE=Path(__file__).resolve().parent


def compact(h,n):
    h=copy.deepcopy(h)
    for old,parent,key,width in ((h.local,h,'local',16*n),(h.context[0],h.context,0,29*n)):
        q=torch.nn.Linear(width,old.out_features,bias=old.bias is not None).to('cuda')
        with torch.no_grad():
            q.weight.copy_(old.weight[:,-width:]);q.bias.copy_(old.bias)
        if isinstance(key,str):setattr(parent,key,q)
        else:parent[key]=q
    return h.eval().requires_grad_(False)


@torch.no_grad()
def main():
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    state=torch.load(R.HERE/'8C_611601_u12000.pt',map_location='cpu',weights_only=False)
    h=E.AuxHead().cuda();h.load_state_dict(state['state']);h.eval().requires_grad_(False)
    heads={n:compact(h,n) for n in (1,2,4,8)}
    gen=torch.Generator(device='cuda').manual_seed(612100);result={}
    for batch in (1024,4096):
        x=(torch.randn(batch,8,81,16,generator=gen,device='cuda'),torch.randn(batch,8*29,generator=gen,device='cuda'),
           torch.randint(17,(batch,),generator=gen,device='cuda'),torch.randn(batch,256,generator=gen,device='cuda'))
        y=torch.randn(batch,81,16,generator=gen,device='cuda');first=torch.zeros(batch,dtype=torch.bool,device='cuda')
        inputs={n:(x[0][:,-n:].contiguous(),x[1].view(batch,8,29)[:,-n:].reshape(batch,-1).contiguous(),x[2],x[3]) for n in heads}
        checks={}
        for n,hh in heads.items():
            p=x[0].clone();ctx=x[1].view(batch,8,29).clone();p[:,:8-n]=0;ctx[:,:8-n]=0
            full=(p,ctx.flatten(1),x[2],x[3]);out=h(full);short=hh(inputs[n]);ref=B.probabilities(h,full,y,first);prob=B.probabilities(hh,inputs[n],y,first)
            err=float((out-short).abs().max());tv=float((.5*(ref-prob).abs().sum(-1)).max())
            # Changing GEMM shapes changes FP32 accumulation order. Establish
            # the exact folding separately in FP64; retain native discrepancies.
            hh64=copy.deepcopy(hh).double();h64=copy.deepcopy(h).double()
            cv=lambda xx:tuple(t[:4].double() if t.is_floating_point() else t[:4] for t in xx)
            exact=float((h64(cv(full))-hh64(cv(inputs[n]))).abs().max())
            del hh64,h64
            assert exact<1e-10 and tv<1e-4,(n,err,tv,exact)
            checks[n]=dict(output_max_abs=err,policy_TV_max=tv,fp64_folding_max_abs=exact)
        times={n:[] for n in heads}
        for rep in range(15):
            for n in np.random.default_rng(612110+rep).permutation(list(heads)):
                st,en=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                st.record();B.probabilities(heads[n],inputs[n],y,first);en.record();en.synchronize()
                if rep>=3:times[n].append(st.elapsed_time(en)/1000)
        result[batch]=dict(checks=checks,timing={n:dict(median=float(np.median(t)),min=float(min(t)),max=float(max(t)),seconds=t) for n,t in times.items()})
        print(batch,{n:np.median(t) for n,t in times.items()},flush=True)
    delta=sum((result[b]['timing'][8]['median']-result[b]['timing'][1]['median'])*12_800_000/b for b in result)
    result['full_budget_8_minus_1_seconds']=delta
    result['scope']='CUDA-event time of paired actual/NOOP forecasts and probability score; excludes common encoding, sampling, data gathering, backward and world optimizer. Eager shape benchmark; no new fit or changed policy quality claim.'
    (HERE/'history_cost.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
