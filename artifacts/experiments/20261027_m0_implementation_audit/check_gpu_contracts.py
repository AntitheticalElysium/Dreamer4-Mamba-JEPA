"""Checkpoint-gradient parity and long, segmented Mamba kernel checks."""
import copy
import json
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR


def checkpoint_parity(cfg):
    cfg=copy.deepcopy(cfg);cfg['p1']['precision']='fp32'
    torch.manual_seed(18)
    a=M.M0(cfg).cuda().train()
    for blk in a.denoiser.blocks:torch.nn.init.normal_(blk.mod.weight,std=.02)
    for out in a.denoiser.out:torch.nn.init.normal_(out.weight,std=.02)
    b=copy.deepcopy(a);a.denoiser.checkpoint=False;b.denoiser.checkpoint=True
    eps=D.load_split(cfg)['train'][:8]
    streams=D.Streams(eps,2,8,seed=71,targets_offsets=D.frame_offsets(eps))
    prefix=streams.next(8)
    chunk=streams.next(24)
    targets=np.load('artifacts/saccade_m0/targets/change_targets.npy',mmap_mode='r')
    with torch.no_grad():
        _,carry,cache=TR.p1_losses(a,prefix,a.belief.initial(2,'cuda'),torch.zeros(2,9,256,device='cuda'),targets,cfg)
    vals=[];grads=[]
    for model in (a,b):
        torch.manual_seed(914)
        terms,_,_=TR.p1_losses(model,chunk,copy.deepcopy(carry),cache.clone(),targets,cfg)
        loss=model.harmoniser(TR.groups(terms));loss.backward()
        vals.append({k:float(v.detach()) for k,v in terms.items()})
        grads.append({k:v.grad.detach().clone() for k,v in model.named_parameters() if v.grad is not None})
    diffs={k:float((grads[0][k]-grads[1][k]).abs().max()) for k in grads[0]}
    rel={k:diffs[k]/max(float(grads[0][k].abs().max()),1e-9) for k in diffs}
    return {'terms_max_abs_delta':max(abs(vals[0][k]-vals[1][k]) for k in vals[0]),
            'grad_max_abs_delta':max(diffs.values()),'worst_parameter':max(diffs,key=diffs.get),
            'grad_max_relative_to_parameter_max':max(rel.values()),
            'allclose':all(torch.allclose(grads[0][k],grads[1][k],atol=3e-6,rtol=2e-4) for k in grads[0])}


def segmented_kernel(cfg):
    torch.manual_seed(5)
    mixer=M.FunctionalMamba2(cfg['belief']).cuda()
    x=torch.randn(2,513,256,device='cuda',requires_grad=True)
    base=mixer.initial(2,'cuda')
    carry=M.MambaCarry(torch.randn_like(base.conv,requires_grad=True),torch.randn_like(base.ssm,requires_grad=True))
    seq=torch.zeros(2,513,dtype=torch.long,device='cuda')
    for t in (7,255,256,300,511):seq[0,t:]+=1
    seq[1]+=1
    for t in (1,129,257,500):seq[1,t:]+=1
    out,c=mixer(x,carry,seq)
    loss=out[:,:3].square().mean()+out[:,-1:].square().mean()
    gs=torch.autograd.grad(loss,(x,carry.conv,carry.ssm))
    xr=x.detach().requires_grad_();cr=M.MambaCarry(carry.conv.detach().requires_grad_(),carry.ssm.detach().requires_grad_())
    ref,rc=mixer(xr,cr,seq,'reference')
    rl=ref[:,:3].square().mean()+ref[:,-1:].square().mean()
    gr=torch.autograd.grad(rl,(xr,cr.conv,cr.ssm))
    return {'tokens':513,'reset_tokens_stream0':[7,255,256,300,511],'reset_tokens_stream1':[0,1,129,257,500],
            'output_max_abs_delta':float((out-ref).abs().max()),
            'conv_carry_max_abs_delta':float((c.conv-rc.conv).abs().max()),
            'ssm_carry_max_abs_delta':float((c.ssm-rc.ssm).abs().max()),
            'gradient_max_abs_delta':[float((g-r).abs().max()) for g,r in zip(gs,gr)],
            'reset_stream_incoming_conv_grad_max':float(gs[1][1].abs().max()),
            'reset_stream_incoming_ssm_grad_max':float(gs[2][1].abs().max())}


def main():
    torch.set_num_threads(6)
    cfg=TR.load_cfg()
    r={'checkpoint_parity':checkpoint_parity(cfg)}
    torch.cuda.empty_cache()
    r['segmented_kernel']=segmented_kernel(cfg)
    assert r['checkpoint_parity']['allclose']
    assert r['segmented_kernel']['output_max_abs_delta']<5e-3
    assert r['segmented_kernel']['reset_stream_incoming_conv_grad_max']==0
    assert r['segmented_kernel']['reset_stream_incoming_ssm_grad_max']==0
    Path(__file__).with_name('gpu_contract_results.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
