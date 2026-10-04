"""E13 check (2026-10-01): predictability of entering terrain. Held-out entering cells (scroll steps of pool windows): MLP on the 3 visible edge tokens 0.756, corrt teacher s7 teacher-forced 0.760, copy the adjacent tile 0.714, majority 0.476."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS
from scroll import SHIFTS, estimate
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as Sp
torch.manual_seed(0)
dev = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
meta, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, tr, ts)
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
world, st = T.load_world("artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_u18000.pt", dev)
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
train = main_rows[~torch.isin(main_rows, held)][torch.randperm(len(main_rows) - 2048, generator=torch.Generator().manual_seed(2))[:8000]]
def collect(rows):
    F_, Y, W, C = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            r = rows[i:i + 64]; s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
            with autocast_context(config):
                pred = world(s.to(dev), F.pad(a, (0, 1)).to(dev))[0].float().cpu()
            for t in range(5):
                sh = estimate(s[:, t], s[:, t + 1])
                for j in torch.nonzero((sh != 0) & alive[:, t + 1])[:, 0].tolist():
                    dr, dc = SHIFTS[int(sh[j])]
                    g0, g1, gp = s[j, t, :63].view(7, 9, -1), s[j, t + 1, :63].view(7, 9, -1), pred[j, t, :63].view(7, 9, -1)
                    if dr: cells = [((6 if dr == 1 else 0), c) for c in range(9)]
                    else: cells = [(rr, (8 if dc == 1 else 0)) for rr in range(7)]
                    for (rr, cc) in cells:
                        # the cell that entered at (rr,cc) in frame t+1; its inside neighbour in frame t+1 was at (rr+dr, cc+dc)... use frame t's edge
                        er, ec = rr + dr, cc + dc                          # same world cell's neighbour position in frame t: the edge cell of frame t
                        er, ec = min(max(er, 0), 6), min(max(ec, 0), 8)
                        feat = torch.cat([g0[er, ec], g0[max(er - 1, 0), ec] if dc else g0[er, max(ec - 1, 0)],
                                          g0[min(er + 1, 6), ec] if dc else g0[er, min(ec + 1, 8)]])
                        F_.append(feat); Y.append(g1[rr, cc]); W.append(gp[rr, cc]); C.append(g0[er, ec])
    return torch.stack(F_), torch.stack(Y), torch.stack(W), torch.stack(C)
Ftr, Ytr, _, _ = collect(train); Fte, Yte, Wte, Cte = collect(held)
ytr, yte = probes.tile(Ytr).argmax(-1), probes.tile(Yte).argmax(-1)
world_cls, copy_cls = probes.tile(Wte).argmax(-1), probes.tile(Cte).argmax(-1)
net = nn.Sequential(nn.Linear(Ftr.shape[1], 256), nn.ReLU(), nn.Linear(256, 17)); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
mu, sd = Ftr.mean(0), Ftr.std(0).clamp(min=1e-6)
for ep in range(40):
    perm = torch.randperm(len(ytr))
    for j in range(0, len(perm), 512):
        b = perm[j:j + 512]; loss = F.cross_entropy(net((Ftr[b] - mu) / sd), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
with torch.no_grad(): p = net((Fte - mu) / sd).argmax(-1)
maj = torch.bincount(ytr, minlength=17).argmax()
print(f"entering cells: train {len(ytr)} held {len(yte)}")
print(f"held class accuracy: probe(3 edge tokens) {float((p == yte).float().mean()):.3f} | world teacher-forced {float((world_cls == yte).float().mean()):.3f} | copy inside-edge tile {float((copy_cls == yte).float().mean()):.3f} | majority class {float((yte == maj).float().mean()):.3f}")
