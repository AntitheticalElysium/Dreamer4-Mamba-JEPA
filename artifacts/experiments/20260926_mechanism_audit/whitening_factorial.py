"""Cross coordinate system with effective prediction metric, holding the recipe fixed.

Historical U uses u and inv-std weights in u coordinates. Historical W uses
w=u/std and nearly uniform weights in w coordinates. These two NEW cells cross
the coordinate system and the relative prediction-error metric. The old U/W
checkpoints are immutable references. No new judging data is collected here.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path[:0] = [str(ROOT), str(LADDER)]
import interface as I
from whiten import whitened_pool


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=("U_Wmetric", "W_Umetric"))
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    out = HERE / f"{args.arm}.pt"
    if out.exists() and not args.smoke:
        raise SystemExit(f"Refusing to overwrite {out}")
    pool, std = whitened_pool()
    lam_u, lam_w = pool["weights"]["U"], pool["weights"]["W"]
    if args.arm == "U_Wmetric":
        key, metric = "u", lam_w / std.square()
    else:
        key, metric = "w", lam_u * std.square()
    metric = metric / metric.mean()
    I.KEY[args.arm] = key
    pool["weights"][args.arm] = metric
    if args.smoke:
        I.PHASE1_UPDATES, I.PHASE2_UPDATES = 10, 10
    started = time.time()
    def log(**row):
        print(json.dumps({**row, "seconds": round(time.time() - started, 1)}), flush=True)
    log(stage="begin", arm=args.arm, state=key,
        metric_ratio=float(metric.max() / metric.min()),
        pool_sha256=sha(I.POOL / "pool.pt"))
    world, heads, history, counts = I.train(args.arm, pool, torch.device("cuda"), log)
    if not args.smoke:
        payload = {"arm": args.arm, "world": world.state_dict(), "heads": heads.state_dict(),
                   "history": history, "depth_counts": counts, "std": std, "metric": metric,
                   "pool_sha256": sha(I.POOL / "pool.pt"),
                   "interface_sha256": sha(LADDER / "interface.py"),
                   "whitening_factorial_sha256": sha(__file__)}
        tmp = out.with_suffix(".tmp")
        torch.save(payload, tmp)
        tmp.replace(out)
    log(stage="complete", arm=args.arm)


if __name__ == "__main__":
    main()
