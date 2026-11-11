"""Bounded CPU adversarial checks on claims in Claude's belief-objective proposal."""
import json,math
from pathlib import Path
import torch
from saccade import model as M
HERE=Path(__file__).resolve().parent

def main():
    torch.set_num_threads(2);torch.manual_seed(110801)
    ck=torch.load('artifacts/experiments/20261106_p1f4/run/ckpt_10.pt',map_location='cpu',weights_only=False)
    den=M.Denoiser(ck['cfg'])
    den.load_state_dict({k.removeprefix('denoiser.'):v for k,v in ck['model'].items() if k.startswith('denoiser.')});den.eval()
    B,T=2,2
    x=(torch.randn(B,T,81,16),torch.randn(B,T,9,64),torch.randn(B,T,8))
    mem=torch.randn(B,T,4,den.W,requires_grad=True)
    action=torch.zeros(B,T,dtype=torch.long);fix=torch.full((B,T,2),4,dtype=torch.long)
    valid=torch.ones(B,T,dtype=torch.bool);episode=torch.zeros(B,T,dtype=torch.long);boot=torch.tensor([False,True])
    levels=[(torch.zeros(B,T,dtype=torch.long),torch.tensor([[3,3],[2,2]])) for _ in range(2)]
    loss=M.shortcut_loss(den,x,action,mem,fix,episode,valid,ck['cfg']['edm']['shortcut'],boot,True,levels=levels)
    loss[1].sum().backward()
    assert mem.grad[1].norm()>0 and mem.grad[0].norm()==0
    r={'actual_f4_denoiser_bootstrap_test':{'loss':float(loss[1].sum().detach()),
        'memory_grad_bootstrap_row_norm':float(mem.grad[1].norm()),'memory_grad_unselected_flow_row_norm':float(mem.grad[0].norm()),
        'claim':'Only teacher memory is detached; student bootstrap loss trains memory.'},
       'contrastive_counterexample':{'candidates':128,'correct_scene_logit':30.,'other_scene_logits':0.,
        'infonce_nats':math.log1p(127*math.exp(-30.)),'future_event_bit_retained':False},
       'innovation_counterexample':{'history_state_A':[0,1],'predicted_mean':[0,0],'conditional_variance':[1,9],
        'E_error_squared_given_history':[1,9],'state_sufficient':True,
        'point':'Predictable error magnitude does not imply missing state; predictable signed residual can reflect decoder error.'}}
    (HERE/'CHECKS.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))

if __name__=='__main__':main()
