"""Actual-world CUDA chronology, original-loss and final-gradient checks.

Separate from the trainer source contract; does not change model weights or
launch training. Run before the full experiment and preserve this numerical
proof alongside its source/input hashes.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import prefix_intervention as P
import oracle_eval as OE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',default='generated_prefix_real_checks.json')
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(7);torch.cuda.manual_seed_all(7)
    config=P.config_from_dict(torch.load(P.TW.S.CHECKPOINT,weights_only=False,map_location='cpu')['config'])
    pool=torch.load(P.TW.POOLS['rawlong']/'labels.pt',weights_only=False,map_location='cpu',mmap=True)
    valid=torch.from_numpy(np.load(P.OUT/'layouts_valid.npy'))
    rows=torch.tensor([int(torch.where(~pool['terminal']&valid)[0][0]),int(torch.where(pool['terminal']&valid)[0][0])])
    n=len(pool['terminal'])
    tokens=np.memmap(P.TW.POOLS['rawlong']/'tokens.f16',dtype=np.float16,mode='r',shape=(n,64,81,192))
    all_layout=np.load(P.OUT/'layouts.npy',mmap_mode='r')
    stream=torch.from_numpy(np.array(tokens[rows.numpy()])).to('cuda').float()
    actions=pool['actions'][rows].to('cuda')
    layouts=torch.from_numpy(np.array(all_layout[rows.numpy()])).to('cuda')
    targets=torch.tensor([30,63],device='cuda')
    parent=torch.load(P.INIT,weights_only=False,map_location='cpu')
    world=P.TW.TWorld('corrt',None,'fmamba','all',0,False,16)
    world.load_state_dict(parent['world']); world=world.to('cuda').eval()
    cond=P.OW.OracleCond();cond.load_state_dict(parent['oracle']);cond=cond.to('cuda').eval()
    proof={}
    with torch.no_grad():
        teacher,t1,i1=P.endpoint(world,cond,stream,actions,layouts,targets,0,False,config)
        generated,t2,i2=P.endpoint(world,cond,stream,actions,layouts,targets,0,True,config)
        maxdiff=float((teacher-generated).abs().max())
        assert torch.equal(t1,t2) and torch.equal(i1,i2) and maxdiff<=1e-5
        proof['zero_prefix']={'outputs_equal':torch.equal(teacher,generated),'max_abs':maxdiff,
                              'inputs_targets_equal':True,'tolerance':1e-5}
        start=targets-15-15
        initial=P.window(stream,start,15)
        ac=P.window(actions,start,15);nl=P.window(layouts,start+1,15)
        actual=P.conditioned(world,cond,initial,ac,nl,config).cpu()
        evaluator=OE.step(world,cond,initial,ac,nl,torch.device('cuda'),config)
        maxdiff=float((actual-evaluator).abs().max());assert maxdiff<=1e-5
        proof['first_prefix_matches_evaluator']={'outputs_equal':torch.equal(actual,evaluator),
                                                 'max_abs':maxdiff,'tolerance':1e-5}
        starts=torch.tensor([0,48],device='cuda')
        main=P.window(stream,starts,16); main_a=P.window(actions,starts,15); main_nl=P.window(layouts,starts+1,15)
        ours=P.LI.objective(P.main_teacher(world,cond,main,main_a,main_nl,config),'D')
        with P.autocast_context(config):
            world.delta=cond(main_nl,16)
            try: reference=P.TW.rollout_losses(world,main,main_a,'teacher',recon='deltairis')
            finally:world.delta=None
        difference=float((ours-reference).abs());assert difference<=1e-6
        proof['main_teacher_original_loss']={'ours':float(ours),'reference':float(reference),'abs_diff':difference,'tolerance':1e-6}
    calls=[]
    handle=world.register_forward_pre_hook(lambda module,args:calls.append(torch.is_grad_enabled()))
    world.train();cond.train();world.zero_grad(set_to_none=True);cond.zero_grad(set_to_none=True)
    out,target,inputs=P.endpoint(world,cond,stream,actions,layouts,targets,15,True,config)
    handle.remove()
    assert not inputs.requires_grad and calls==[False]*15+[True]
    loss=P.LI.objective(P.LI.components(out[:,None],target[:,None]),'D');loss.backward()
    def norm(module):
        tensors=[v.grad.detach().float().square().sum() for v in module.parameters() if v.grad is not None]
        assert tensors and all(torch.isfinite(v) for v in tensors)
        result=float(torch.stack(tensors).sum().sqrt());assert result>0
        return result
    proof['full_generated_prefix_final_gradient']={'prefix_nograd_calls':15,'final_gradient_calls':1,
        'inputs_detached':True,'world_gradient_norm':norm(world),'oracle_gradient_norm':norm(cond),'loss':float(loss.detach())}
    assert world.delta is None
    result={'status':'pass','rows':rows.tolist(),'targets':targets.tolist(),'checks':proof,
            'source_inputs_sha256':{str(f.relative_to(P.ROOT)):P.LI.sha(f) for f in
            (Path(__file__),Path(P.__file__),Path(OE.__file__),P.INIT,P.TW.POOLS['rawlong']/'labels.pt',P.OUT/'layouts_valid.npy')},
            'precision':'BF16/IEEE/TF32disabled','seconds':time.time()-started,
            'gpu_peak_gb':torch.cuda.max_memory_allocated()/2**30}
    P.LI.atomic(result,P.OUT/args.out,True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
