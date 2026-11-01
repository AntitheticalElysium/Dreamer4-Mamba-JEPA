"""Check installed Triton kernel's incoming-carry sequence-id convention."""
import json
from pathlib import Path

import torch
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def main():
    torch.manual_seed(0)
    device="cuda"
    b,l,h,p,n=1,16,4,64,64
    x=torch.randn(b,l,h,p,device=device)
    dt=torch.full((b,l,h),.01,device=device)
    a=-torch.ones(h,device=device)
    bb=torch.randn(b,l,1,n,device=device)
    cc=torch.randn(b,l,1,n,device=device)
    state=torch.randn(b,h,p,n,device=device)
    def run(s,idx):
        return mamba_chunk_scan_combined(x,dt,a,bb,cc,chunk_size=256,
            initial_states=s,seq_idx=idx,return_final_states=True,state_dtype=torch.float32)
    with torch.no_grad():
        ref=run(state,None)
        zero_id=run(state,torch.zeros(b,l,dtype=torch.int32,device=device))
        nonzero_id=run(state,torch.full((b,l),7,dtype=torch.int32,device=device))
        no_carry=run(torch.zeros_like(state),None)
    delta=lambda x,y:[float((u-v).abs().max()) for u,v in zip(x,y)]
    result={"max_abs_output_and_final_state_difference":{
        "id0_vs_no_ids":delta(zero_id,ref),
        "id7_vs_no_ids":delta(nonzero_id,ref),
        "id7_vs_zero_incoming_state":delta(nonzero_id,no_carry)},
        "interpretation":"seq_idx must use the kernel's local carry convention, not arbitrary global episode ids."}
    assert max(delta(zero_id,ref))<1e-4
    assert max(delta(nonzero_id,no_carry))<1e-4
    assert min(delta(nonzero_id,ref))>1e-3
    Path(__file__).with_name("seq_idx_results.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
