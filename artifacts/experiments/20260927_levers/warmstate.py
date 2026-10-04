"""Warm-restart state from a weights-only world (2026-10-02, option A): the world's weights, NO optimizer state (tworld then
starts a fresh AdamW), the batch-order generator replayed to update U exactly as tworld.train draws it (seed 11, one randint of
BATCH rows per update over the same training rows), the current CPU RNG. `tworld.py --resume <out> --updates N` continues it.
Usage: warmstate.py <world.pt> <update U> <out.state.pt>
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import tworld as TW

src, U, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
st = torch.load(src, map_location="cpu", weights_only=False)
pool = torch.load(TW.POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
order = torch.Generator().manual_seed(11)
for _ in range(U):
    torch.randint(len(rows), (TW.BATCH,), generator=order)
out.parent.mkdir(parents=True, exist_ok=True)
torch.save({"update": U, "world": st["world"], "optimizer": None, "order": order.get_state(), "rng_cpu": torch.get_rng_state(),
            "rng_cuda": None, "history": list(st.get("history", []))}, out)
print({"from": str(src), "update": U, "out": str(out)})
