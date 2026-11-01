"""Packing/noise checks for the 2026-10-27 proposal, not an M0 prototype."""
import json
from pathlib import Path
import torch


def main():
    torch.manual_seed(20261027)
    torch.set_num_threads(4)
    b,dp,dg,dv=3,16,64,8
    p=torch.randn(b,9,9,dp); g=torch.randn(b,3,3,dg); v=torch.randn(b,dv)
    top_left=((0,0),(3,3),(6,6))
    gm=torch.zeros(b,9,9,dg); mask=torch.zeros(b,9,9,1)
    for i,(r,c) in enumerate(top_left):
        gm[i,r:r+3,c:c+3]=g[i]; mask[i,r:r+3,c:c+3]=1
    vm=v[:,None,None].expand(-1,9,9,-1)
    packed=torch.cat([p,gm,vm,mask],-1)
    pg=packed[...,:dp]; gg=packed[...,dp:dp+dg]
    vg=packed[...,dp+dg:dp+dg+dv].mean((1,2))
    recovered_g=torch.stack([gg[i,r:r+3,c:c+3] for i,(r,c) in enumerate(top_left)])
    assert torch.equal(pg,p) and torch.equal(recovered_g,g) and torch.allclose(vg,v)
    n=20000
    independent_grid_noise=torch.randn(n,81)
    shared_vector_noise=torch.randn(n,1).expand(-1,81)
    x=torch.arange(81,dtype=torch.float64).reshape(1,1,9,9).requires_grad_()
    y=torch.nn.functional.group_norm(x,1)
    grad=torch.autograd.grad(y[0,0,8,8],x)[0][0,0,0,0]
    result={
        'scope':'representation and normalisation only; no model training or throughput claim',
        'compact_percept_dimensions':81*dp+9*dg+dv,
        'packing_round_trip_passed':True,
        'glimpse_padding_zero':bool((gm[mask[...,0]==0]==0).all()),
        'wrong_independent_grid_noise_after_average_variance':float(independent_grid_noise.mean(1).var()),
        'correct_shared_vector_noise_after_average_variance':float(shared_vector_noise.mean(1).var()),
        'analytical_independent_to_shared_variance_ratio':1/81,
        'broadcast_lighting_summed_squared_error_multiplier':81,
        'group_norm_far_corner_input_gradient':float(grad),
    }
    Path(__file__).with_suffix('.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
