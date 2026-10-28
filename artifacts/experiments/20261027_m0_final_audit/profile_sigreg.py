"""Profile only the exact canonical regularizer at the declared M0 sample sizes."""
import json
import time
from pathlib import Path

import torch

from d4mj.lewm import SIGReg


def main():
    torch.set_num_threads(6)
    torch.manual_seed(0)
    torch.cuda.init()
    mod = SIGReg().cuda()
    # Warm CUDA before measuring; no large allocator cache to contaminate peaks.
    t = torch.ones(16, device="cuda", requires_grad=True)
    t.square().sum().backward()
    del t
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    p = torch.randn(1, 4096, 16, device="cuda", requires_grad=True)
    g = torch.randn(1, 4096, 64, device="cuda", requires_grad=True)
    rng = torch.Generator(device="cuda").manual_seed(17)
    lp = mod(p, rng)
    lg = mod(g, rng)
    (.0028 * (lp+lg)).backward()
    torch.cuda.synchronize()
    result = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(), "dtype": "float32",
              "tokens_each_family": 4096, "projections": 1024, "knots": 17,
              "elapsed_forward_backward_s": time.perf_counter()-start,
              "base_allocated_MiB": base/2**20,
              "peak_allocated_MiB": torch.cuda.max_memory_allocated()/2**20,
              "peak_reserved_MiB": torch.cuda.max_memory_reserved()/2**20,
              "values": [float(lp.detach()),float(lg.detach())],
              "grad_finite": [bool(p.grad.isfinite().all()),bool(g.grad.isfinite().all())],
              "interpretation": "Isolated regularizer cost; not M0 peak or throughput."}
    Path(__file__).with_name("sigreg_profile.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
