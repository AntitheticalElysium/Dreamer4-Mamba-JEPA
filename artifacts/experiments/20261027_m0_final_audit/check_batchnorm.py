"""Causal normalization counterexample for the concurrent e1b4cb40 revision."""
import json
from pathlib import Path

import torch


def main():
    torch.set_num_threads(4)
    x=(torch.arange(2*4*3*2,dtype=torch.float64).reshape(2,4,3,2)/10-2).requires_grad_()
    altered=x.detach().clone()
    altered[:,2:]+=4
    def clamp(v): return 3*torch.tanh(v/3)
    def batch(v):
        bn=torch.nn.BatchNorm1d(2,affine=False,dtype=torch.float64)
        return bn(clamp(v).reshape(-1,2)).reshape_as(v)
    y=batch(x)
    z=batch(altered)
    prefix_delta=float((y[:,:2]-z[:,:2]).abs().max().detach())
    grad=torch.autograd.grad(y[0,0].sum(),x)[0]
    future_grad=float(grad[:,2:].abs().max())
    bn=torch.nn.BatchNorm1d(2,affine=False,dtype=torch.float64).eval()
    fixed=lambda v:bn(clamp(v).reshape(-1,2)).reshape_as(v)
    eval_delta=float((fixed(x)[:,:2]-fixed(altered)[:,:2]).abs().max().detach())
    v=torch.full((100,1),3.,dtype=torch.float64);v[0]=-3
    bound=float(torch.nn.BatchNorm1d(1,affine=False,dtype=torch.float64)(v).abs().max())
    result={"past_output_max_delta_when_only_future_changes_train_mode":prefix_delta,
            "past_output_future_input_max_gradient_train_mode":future_grad,
            "past_output_max_delta_fixed_stats_eval_mode":eval_delta,
            "post_bn_max_abs_from_inputs_bounded_by3":bound,
            "scope":"Ordinary BatchNorm over chunk tokens, as newly specified; no saccade implementation exists."}
    assert prefix_delta>0.1 and future_grad>0 and eval_delta==0 and bound>3
    Path(__file__).with_name("batchnorm_results.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
