"""Candidate E19 boundary randomization: preserve every factual target/action instead of dropping terminal prefixes.

No research training is launched. CPU source/control and target-ledger checks only.
"""
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(4)
sys.path.insert(0,'artifacts/experiments/20260927_levers')
import tworld as T

HERE=Path(__file__).parent
parent=T.OUT/'corrt_raw_teacher_s7_u36000.pt'
st=torch.load(parent,map_location='cpu',weights_only=False)
w=T.TWorld('corrt',backbone='full')
w.load_state_dict(st['world'])
w.eval()
pool=torch.load(T.POOLS['raw']/'pool.pt',weights_only=False,mmap=True)
idx=torch.cat([torch.where(~pool['terminal'])[0][:2],torch.where(pool['terminal'])[0][:2]])
s=pool['tokens'][idx].float()
a=pool['actions'][idx]

def segments(r):
    # Last chunk's death input sits at row r. Earlier targets are predicted in a separate original prefix.
    start=4-r
    return ([(0,start)] if start else [])+[(start,5)]

out={'scope':'CPU mechanisms/ledgers for a proposed target-preserving alternative; no retrain or causal treatment result.'}
for r in range(5):
    edges=[(i,i+1,i) for lo,hi in segments(r) for i in range(lo,hi)]
    assert edges==[(i,i+1,i) for i in range(5)]
out['all_offsets_preserve_exact_five_transition_action_target_pairs']=True

old=T.rollout_losses(w,s,a,'teacher')
old.backward()
g0={k:p.grad.clone() for k,p in w.named_parameters() if p.grad is not None}
w.zero_grad(set_to_none=True)
same=sum((w(s[:,lo:hi+1],F.pad(a[:,lo:hi],(0,1)))[0][:,:hi-lo]-s[:,lo+1:hi+1]).abs().sum()
         for lo,hi in segments(4))/s[:,1:].numel()
same.backward()
out['no_split_loss_delta']=float(abs(old.detach()-same.detach()))
out['no_split_gradient_max_abs']=max(float((p.grad-g0[k]).abs().max()) for k,p in w.named_parameters() if k in g0)
assert out['no_split_loss_delta']==0 and out['no_split_gradient_max_abs']==0
w.zero_grad(set_to_none=True)

# Right-padding perturbation cannot change pre-padding predictions in the actual attention/corrt forward.
with torch.no_grad():
    cut=s[:,:3]
    aa=F.pad(a[:,:2],(0,1))
    short=w(cut,aa)[0][:,:2]
    padded=torch.cat([cut,torch.randn_like(s[:,3:])],1)
    ap=torch.cat([aa,torch.randint(0,17,(len(s),3))],1)
    long=w(padded,ap)[0][:,:2]
    out['future_padding_prediction_max_abs']=float((short-long).abs().max())
assert out['future_padding_prediction_max_abs']<=1e-5
out['original_targets_per_window']=5
out['candidate_targets_per_window']=5
out['candidate_terminal_target_rows']=[r for r in range(5)]
out['remaining_treatment_difference']='Some contexts become shorter and recurrent boundaries reset; this isolates a boundary/layout intervention, not position embeddings alone.'
out['Mamba_validation']='Pending: same pair ledger applies, but actual Mamba padding/reset CUDA check is not run while trainer owns GPU.'
out['parent_sha256']=hashlib.sha256(parent.read_bytes()).hexdigest()
out['trainer_sha256']=hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest()
tmp=HERE/'layout_proof.tmp'
tmp.write_text(json.dumps(out,indent=2)+'\n')
tmp.replace(HERE/'layout_proof.json')
print(json.dumps(out,indent=2))
