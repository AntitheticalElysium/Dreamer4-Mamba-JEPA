"""Exact reduction scheduling comparison, isolated from M0 implementation."""
import gc
import json
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

from d4mj.lewm import SIGReg


def main():
    torch.set_num_threads(6)
    torch.manual_seed(0)
    full = SIGReg().cuda()
    chunk = SIGReg(num_proj=128).cuda()
    p = torch.randn(1,4096,16,device="cuda",requires_grad=True)
    g = torch.randn(1,4096,64,device="cuda",requires_grad=True)
    dirs=[]
    for d in (16,64):
        a=torch.randn(d,1024,device="cuda")
        dirs.append(a/a.norm(dim=0,keepdim=True))
    loss=.0028*(full(p,directions=dirs[0])+full(g,directions=dirs[1]))
    loss.backward()
    ref_loss=float(loss.detach())
    ref_grad=[p.grad.clone(),g.grad.clone()]
    p.grad=g.grad=None
    del loss
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start=time.perf_counter()
    def piece(v,d):
        return chunk(v,directions=d)
    total=0
    for values,a in zip((p,g),dirs):
        total=total+sum(checkpoint(piece,values,a[:,i:i+128],use_reentrant=False) for i in range(0,1024,128))/8
    loss=.0028*total
    loss.backward()
    torch.cuda.synchronize()
    result={"loss_reference":ref_loss,"loss_chunked":float(loss.detach()),
            "gradient_max_abs_error":[float((x.grad-y).abs().max()) for x,y in zip((p,g),ref_grad)],
            "value_pass":abs(float(loss.detach())-ref_loss)<=1e-6+1e-5*abs(ref_loss),
            "gradient_pass":[bool(torch.allclose(x.grad,y,rtol=1e-5,atol=1e-6)) for x,y in zip((p,g),ref_grad)],
            "peak_allocated_MiB":torch.cuda.max_memory_allocated()/2**20,
            "peak_reserved_MiB":torch.cuda.max_memory_reserved()/2**20,
            "elapsed_forward_backward_s":time.perf_counter()-start,
            "limitation":"Isolated regularizer; timings are one measurement, not steady-state M0 throughput."}
    assert result["value_pass"] and all(result["gradient_pass"])
    Path(__file__).with_name("sigreg_chunking.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
