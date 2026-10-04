"""E13 check (2026-10-01): action / consequence alignment in spatial_pool_v1. Faced-tile change rate (tile probe + token-change > 120) by the action at t and whether the action at t-1 was DO. Result: NOOP 0.0016, DO 0.217 (0.000 after a DO), place_stone 0.217-0.585: the consequence of action t is in frame t+1."""
import sys, torch
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS
meta, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, tr, ts)
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
FACED = {0: 30, 1: 32, 2: 22, 3: 40}
rows = torch.randperm(len(pool["actions"]), generator=torch.Generator().manual_seed(0))[:6000]
stats = {}
for i in range(0, len(rows), 200):
    r = rows[i:i + 200]
    s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
    face = probes.facing(s[:, :, 31].flatten(0, 1)).argmax(-1).view(len(r), 6)
    for t in range(1, 5):
        cell = torch.tensor([FACED[int(f)] for f in face[:, t]])
        x0 = s[torch.arange(len(r)), t, cell]; x1 = s[torch.arange(len(r)), t + 1, cell]
        d = ((x1 - x0) ** 2).sum(-1)
        c0 = probes.tile(x0).argmax(-1); c1 = probes.tile(x1).argmax(-1)
        changed = (c0 != c1) & (d > 120) & alive[:, t + 1]
        for j in range(len(r)):
            if not alive[j, t + 1]: continue
            key = (int(a[j, t]), int(a[j, t - 1]) == 5)
            n, c = stats.get(key, (0, 0)); stats[key] = (n + 1, c + int(changed[j]))
names = {0: "noop", 1: "left", 2: "right", 3: "up", 4: "down", 5: "DO", 6: "sleep", 7: "pl_stone", 8: "pl_table", 9: "pl_furnace", 10: "pl_plant"}
print("action_t   prev_was_DO   n    faced-tile change rate")
for k in sorted(stats):
    n, c = stats[k]
    if n >= 50: print(f"{names.get(k[0], k[0]):10s} {str(k[1]):6s} {n:6d} {c / n:.4f}")
