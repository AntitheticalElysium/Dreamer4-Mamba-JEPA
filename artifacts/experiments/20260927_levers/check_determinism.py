"""Check (2026-10-02): are tworld trainings reproducible on the GPU? (The 36k runs' snapshots differ from the original 18k runs'
at 6k already, median relative weight difference 19% at both seeds, with identical args; s8's consequence transition moved from
12-18k to 24-30k.) Two identical GPU trainings of the current trainer (corrt, teacher, seed 7, batch 40) for N updates, weights
compared; then the same pair under torch.use_deterministic_algorithms(True) with CUBLAS_WORKSPACE_CONFIG=:4096:8 (set by the
caller). Reported: bit-identical or not, max |diff|, median relative difference, at N = 20 and 300.
Reading, declared before running:
  gpu_nondeterministic   the plain pair differs at N = 20 while the deterministic pair is bit-identical (the divergence is
                         kernel nondeterminism, amplified over training)
  other_cause            the plain pair is bit-identical (the runs' divergence then comes from elsewhere: code / data / init)
Usage: [CUBLAS_WORKSPACE_CONFIG=:4096:8] check_determinism.py plain|deterministic
"""
import sys, json, torch
sys.path[:0] = ["artifacts/experiments/20260927_levers", "artifacts/experiments/20260926_diagnosis", "artifacts/experiments/20260921_readout_ladder"]
import tworld as TW

mode = sys.argv[1]
if mode == "deterministic":
    torch.use_deterministic_algorithms(True)
dev = torch.device("cuda")
quiet = lambda **kw: None
out = {}
for n in (20, 300):
    ws = []
    for rep in range(2):
        torch.manual_seed(0)
        w = TW.train("corrt", "raw", "teacher", 7, n, dev, quiet)[0]
        ws.append({k: v.detach().float().cpu() for k, v in w.state_dict().items()})
        del w; torch.cuda.empty_cache()
    a, b = ws
    md = max(float((a[k] - b[k]).abs().max()) for k in a)
    rel = sorted(float((a[k] - b[k]).norm() / a[k].norm().clamp(min=1e-12)) for k in a if a[k].numel() > 1)
    out[n] = {"bit_identical": md == 0.0, "max_abs_diff": md, "median_rel_diff": rel[len(rel) // 2]}
    print(json.dumps({"mode": mode, "updates": n, **out[n]}), flush=True)
print(json.dumps({"mode": mode, "results": out}))
