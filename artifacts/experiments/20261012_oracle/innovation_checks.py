"""Numerical and information-boundary checks for learned innovations."""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
import innovation_model as I
import innovation_train as T
import loss_intervention as L
from d4mj.config import config_from_dict
from d4mj.train import autocast_context


def differences(a, b):
    if isinstance(a, torch.Tensor):
        return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max()) if a.numel() else 0.
    if isinstance(a, dict):
        assert a.keys()==b.keys()
        return max((differences(a[k], b[k]) for k in a), default=0.)
    if isinstance(a, (list, tuple)):
        assert len(a)==len(b)
        return max((differences(x,y) for x,y in zip(a,b)), default=0.)
    assert a==b
    return 0.


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(77)
    root=T.OUT
    resumed=torch.load(root/'smoke/state.pt',map_location='cpu',weights_only=False)
    reference=torch.load(root/'smoke_reference/state.pt',map_location='cpu',weights_only=False)
    assert resumed['update']==reference['update']==4
    assert resumed['contract_id']==reference['contract_id']
    exact={key:L.restored_equal(resumed[key],reference[key])
           for key in ('order','torch_rng','cuda_rng','ledger_chain')}
    assert all(exact.values())
    maxima={key:differences(resumed[key],reference[key])
            for key in ('worlds','channels','optimizers','prior_optimizers')}
    assert max(maxima.values())<1e-4, maxima
    config=config_from_dict(torch.load(L.TW.S.CHECKPOINT,map_location='cpu',weights_only=False)['config'])
    world=L.TW.TWorld('corrt',None,'fmamba','all',0,False,16).cuda().eval()
    parent=torch.load(T.INIT,map_location='cpu',weights_only=False)
    world.load_state_dict(parent['world'])
    panel=torch.load(L.OUT/'panel.pt',map_location='cpu',weights_only=False)
    frames=panel['tokens'][:2,:16].cuda().float()
    actions=F.pad(panel['actions'][:2,:15].cuda(),(0,1))
    channel=I.Innovation().cuda().eval()
    belief=I.native_belief(world,frames,actions,config)
    changed=frames.clone(); changed[:,-1]=torch.randn_like(changed[:,-1])
    other=I.native_belief(world,changed,actions,config)
    causality=float((belief[:,:-1]-other[:,:-1]).abs().max())
    assert causality==0., causality
    q=channel.posterior(belief[:,:-1],frames[:,:-1],frames[:,1:])
    hard,codes=channel.hard(q)
    assert torch.equal(hard,F.one_hot(codes,I.CHOICES).float())
    with torch.no_grad(),autocast_context(config):
        baseline=world(frames,actions)[0].float()
        delta=F.pad(channel.condition(belief[:,:-1],hard),(0,0,0,0,0,1))
        with I.conditioned(world,delta):
            initial=world(frames,actions)[0].float()
    parity=float((baseline-initial).abs().max())
    assert parity==0.
    # Once the adapter is active, fixed hard codes make output independent of
    # posterior confidence: continuous future features never reach completion.
    with torch.no_grad():
        channel.output.weight.normal_(0,.01)
    arbitrary=q.detach().clone()+torch.randn_like(q)*.001
    # Preserve each index explicitly while changing all non-winning logits.
    arbitrary.scatter_(-1,codes[...,None],arbitrary.max(-1,keepdim=True).values+10)
    hard2,codes2=channel.hard(arbitrary)
    assert torch.equal(codes,codes2) and torch.equal(hard,hard2)
    d1=channel.condition(belief[:,:-1],hard)
    d2=channel.condition(belief[:,:-1],hard2)
    assert torch.equal(d1,d2)
    d1.square().mean().backward()
    qgrad=sum(float(p.grad.abs().sum()) for p in channel.posterior.parameters() if p.grad is not None)
    assert qgrad>0., 'posterior cannot learn from completion'
    generator=torch.Generator(device='cuda').manual_seed(123)
    rng=generator.get_state()
    with torch.no_grad():
        _,first=channel.prior(belief[:,-1],sample=True,generator=generator)
        generator.set_state(rng)
        _,second=channel.prior(belief[:,-1],sample=True,generator=generator)
    assert torch.equal(first,second)
    result={'resumed_vs_uninterrupted_updates':4,'exact_streams':exact,'endpoint_max_abs':maxima,
            'native_belief_future_mutation_max_abs':causality,'initial_output_max_abs':parity,
            'hard_codes_exact':True,'fixed_code_continuous_bypass':False,
            'posterior_gradient_sum':qgrad,'ancestral_sampling_rng_restore_exact':True,
            'sources':{str(Path(p).relative_to(L.ROOT)):L.sha(p)
                       for p in (I.__file__,T.__file__,__file__)}}
    L.atomic(result,root/'validation.json',True)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
