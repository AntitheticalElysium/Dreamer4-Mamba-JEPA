"""Exact mapping, gradients, and zero-shift Mamba-2 equivalence for carry transport."""
import copy
import hashlib
import json
import sys
from pathlib import Path
import torch
ROOT=Path('/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(HERE),str(ROOT/'artifacts/experiments/20260927_levers'),
              str(ROOT/'artifacts/experiments/20260921_readout_ladder')]
from carry_transport import transport,SHIFTS,SLOTS  # noqa: E402
from d4mj.mamba_recurrence import MambaCarry,FunctionalMamba2  # noqa: E402
from d4mj.config import config_from_dict  # noqa: E402
from dataclasses import replace
import tworld as T
from transport_world import TransportFactored

def main():
    torch.set_num_threads(8)
    B=5
    conv=torch.arange(B*SLOTS*2*3,dtype=torch.float32).reshape(B*SLOTS,2,3).requires_grad_()
    ssm=torch.arange(B*SLOTS*2*3*4,dtype=torch.float32).reshape(B*SLOTS,2,3,4).requires_grad_()
    codes=torch.arange(5,dtype=torch.long)
    moved=transport(MambaCarry(conv,ssm),codes,B)
    cv=moved.conv.reshape(B,SLOTS,2,3)
    ss=moved.ssm.reshape(B,SLOTS,2,3,4)
    initial_conv=conv.reshape(B,SLOTS,2,3)
    initial_ssm=ssm.reshape(B,SLOTS,2,3,4)
    for b,(dr,dc) in enumerate(SHIFTS):
        for slot in (0,*range(1,64),*range(64,82)):
            if slot==0 or slot>=64:old=slot
            else:
                r,c=divmod(slot-1,9)
                old=1+(r+dr)*9+(c+dc) if 0<=r+dr<7 and 0<=c+dc<9 else None
            if old is None:
                assert not bool(cv[b,slot].any()) and not bool(ss[b,slot].any())
            else:
                assert torch.equal(cv[b,slot],initial_conv[b,old])
                assert torch.equal(ss[b,slot],initial_ssm[b,old])
    (moved.conv.sum()+moved.ssm.sum()).backward()
    assert bool(conv.grad.isfinite().all()) and bool(ssm.grad.isfinite().all())
    cfg=torch.load(ROOT/'artifacts/lewm_m4_canonical/raw/bridge/step-002000.pt',map_location='cpu',weights_only=False)
    settings=replace(config_from_dict(cfg['config']).dynamics,backend='reference',chunk_size=64)
    torch.manual_seed(7)
    model=FunctionalMamba2(settings).eval()
    x=torch.randn(1*SLOTS,4,settings.width)
    with torch.no_grad():
        together,carry=model.scan(x,backend='reference')
        state=model.initial(SLOTS,device='cpu',dtype=x.dtype)
        pieces=[]
        for t in range(4):
            state=transport(state,torch.tensor([0]),1)
            y,state=model.step_reference(x[:,t:t+1],state)
            pieces.append(y)
        stepping=torch.cat(pieces,1)
    diff=float((together-stepping).abs().max())
    convdiff=float((carry.conv-state.conv).abs().max())
    ssmdiff=float((carry.ssm-state.ssm).abs().max())
    assert diff<2e-5 and convdiff==0 and ssmdiff<2e-4,(diff,convdiff,ssmdiff)
    base=T.Factored('fmamba',settings).eval()
    variant=copy.deepcopy(base);variant.__class__=TransportFactored
    z=torch.randn(1,4,SLOTS,settings.width)
    shift=torch.zeros(1,4,dtype=torch.long)
    with torch.no_grad():
        layer_base=base(z,None);layer_same=variant(z,shift)
        shift[0,2]=3
        layer_shift=variant(z,shift)
    layer_diff=float((layer_base-layer_same).abs().max())
    shift_rmse=float((layer_base-layer_shift).square().mean().sqrt())
    assert layer_diff<2e-5
    out={'mapping_pass':True,'gradient_pass':True,'zero_shift_mamba_maxabs':diff,
         'full_layer_no_shift_maxabs':layer_diff,'full_layer_with_shift_delta_rmse':shift_rmse,
         'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         'zero_shift_final_conv_maxabs':convdiff,'zero_shift_final_ssm_maxabs':ssmdiff,
         'carry_bytes_per_token_per_layer':(settings.width*settings.expand+2*settings.d_state)*settings.d_conv*2+
                                        (settings.width*settings.expand//settings.headdim)*settings.headdim*settings.d_state*4,
         'num_slots':SLOTS,'layers':settings.depth}
    HERE.joinpath('carry_transport_smoke.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
