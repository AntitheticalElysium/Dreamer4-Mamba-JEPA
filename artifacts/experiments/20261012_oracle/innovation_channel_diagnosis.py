"""Numerical diagnosis of early channel collapse, before changing the recipe."""
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import loss_intervention as L
from d4mj.config import config_from_dict
from d4mj.train import autocast_context


def main():
    torch.set_num_threads(4);torch.manual_seed(171)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    result={}
    for update in (0,500):
        for arm in ('capacity','rate'):
            saved=torch.load(T.OUT/f'{arm}_u{update:04d}.pt',map_location='cpu',weights_only=False)
            world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16).cuda().eval();world.load_state_dict(saved['world'])
            channel=I.Innovation().cuda().eval();channel.load_state_dict(saved['channel'])
            codes,swapped=[],[]
            for start in range(0,200,8):
                frames=panel['tokens'][start:start+8,1:16].cuda().float()
                actions=panel['actions'][start:start+8,1:16].cuda()
                belief=I.native_belief(world,frames,actions,config)
                with torch.no_grad():
                    true=panel['tokens'][start:start+8,16].cuda().float()
                    altered=panel['tokens'][torch.arange(start,start+len(frames)).roll(1),16].cuda().float()
                    codes.append(channel.posterior(belief[:,-1],frames[:,-1],true).argmax(-1).cpu())
                    swapped.append(channel.posterior(belief[:,-1],frames[:,-1],altered).argmax(-1).cpu())
            cc=torch.cat(codes);ss=torch.cat(swapped)
            counts=torch.stack([torch.bincount(cc[:,j],minlength=8)for j in range(6)])
            frames=panel['tokens'][:2,:16].cuda().float()
            actions=F.pad(panel['actions'][:2,:15].cuda(),(0,1))
            belief=I.native_belief(world,frames,actions,config)
            q=channel.posterior(belief[:,:15],frames[:,:15],frames[:,1:])
            hard,code=channel.hard(q)
            delta=F.pad(channel.condition(belief[:,:15],hard),(0,0,0,0,0,1))
            with I.conditioned(world,delta),autocast_context(config):
                prediction=world(frames,actions)[0][:,:15].float()
            completion=L.objective(L.components(prediction,frames[:,1:]),'D')
            qgrad=torch.autograd.grad(completion,q,retain_graph=True)[0]
            pl,_=channel.prior(belief[:,:15],code.detach())
            rate=.001*I.frame_rate(pl.detach(),hard)
            rgrad=torch.autograd.grad(rate,q)[0]
            with torch.no_grad():
                random=F.one_hot(torch.randint(8,code.shape,device='cuda'),8).float()
                rd=F.pad(channel.condition(belief[:,:15],random),(0,0,0,0,0,1))
                with I.conditioned(world,rd),autocast_context(config):
                    changed=world(frames,actions)[0][:,:15].float()
                random_D=L.objective(L.components(changed,frames[:,1:]),'D')
            key=f'{arm}_u{update}'
            result[key]={'checkpoint_sha256':L.sha(T.OUT/f'{arm}_u{update:04d}.pt'),
                         'DEV_root_code_counts':counts.tolist(),
                         'DEV_distinct_joint_codes':int(cc.unique(dim=0).shape[0]),
                         'DEV_future_swap_code_fraction':float((cc!=ss).float().mean()),
                         'completion_gradient_q_logits_norm':float(qgrad.norm()),
                         'beta001_rate_gradient_q_logits_norm':float(rgrad.norm()),
                         'rate_to_completion_gradient_norm':float(rgrad.norm()/qgrad.norm()) if qgrad.norm()>0 else None,
                         'adapter_output_weight_norm':float(channel.output.weight.norm()),
                         'D_on_two_roots':float(completion.detach()),'D_random_codes_same_world':float(random_D),
                         'random_code_prediction_maxabs':float((prediction.detach()-changed).abs().max())}
            print(json.dumps({key:result[key]}),flush=True)
            del world,channel;torch.cuda.empty_cache()
    L.atomic({'sources':{str(Path(x).relative_to(L.ROOT)):L.sha(x)for x in (__file__,I.__file__,T.__file__)},
              'scope':'exploratory monitored500-step checkpoint; 200DEVroot code-use; 2-root gradient control',
              'readings':result},T.OUT/'channel_diagnosis_u0500.json',True)


if __name__=='__main__':main()
