"""E13 check (2026-10-01): information ceiling. MLP on the INPUT faced token (+ HUD) + action -> next tile class of DO / place transitions (changed x5 weight), pool training windows -> held-out. Result: faced+action changed 0.977 / unchanged 0.928; +HUD 0.984 / 0.945; copy 0.000."""
import sys, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS
torch.manual_seed(0)
meta, tr, ts = T.split()
probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, tr, ts)
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
FACED = {0: 30, 1: 32, 2: 22, 3: 40}; ACT = (5, 7, 8, 9, 10)
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
train = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
train = train[torch.randperm(len(train), generator=torch.Generator().manual_seed(2))[:12000]]
def collect(rows):
    X, Y, B, CH = [], [], [], []
    for i in range(0, len(rows), 256):
        r = rows[i:i + 256]; s = pool["tokens"][r].float(); a = pool["actions"][r]; alive = pool["alive"][r]
        face = probes.facing(s[:, :5, 31].flatten(0, 1)).argmax(-1).view(len(r), 5)
        for t in range(5):
            m = torch.isin(a[:, t], torch.tensor(ACT)) & alive[:, t + 1]
            idx = torch.nonzero(m)[:, 0]
            if not len(idx): continue
            cell = torch.tensor([FACED[int(f)] for f in face[idx, t]])
            x0 = s[idx, t, cell]; x1 = s[idx, t + 1, cell]
            hud = s[idx, t, 63:81].flatten(1)
            act = F.one_hot(torch.tensor([ACT.index(int(v)) for v in a[idx, t]]), 5).float()
            c0 = probes.tile(x0).argmax(-1); c1 = probes.tile(x1).argmax(-1)
            ch = (c0 != c1) & (((x1 - x0) ** 2).sum(-1) > 120)
            X.append(torch.cat([x0, hud, act], 1)); Y.append(c1); B.append(c0); CH.append(ch)
    return torch.cat(X), torch.cat(Y), torch.cat(B), torch.cat(CH)
Xtr, Ytr, Btr, CHtr = collect(train); Xte, Yte, Bte, CHte = collect(held)
print("train n", len(Ytr), "changed", int(CHtr.sum()), "| held n", len(Yte), "changed", int(CHte.sum()))
for name, cols in (("faced+action", list(range(192)) + list(range(192 + 18 * 192, 192 + 18 * 192 + 5))), ("faced+HUD+action", list(range(Xtr.shape[1])))):
    net = nn.Sequential(nn.Linear(len(cols), 256), nn.ReLU(), nn.Linear(256, 17))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    w = torch.where(CHtr, 5.0, 1.0)            # balance changed transitions so the probe cannot ignore them
    for ep in range(60):
        perm = torch.randperm(len(Ytr))
        for j in range(0, len(perm), 256):
            b = perm[j:j + 256]
            loss = (F.cross_entropy(net(Xtr[b][:, cols]), Ytr[b], reduction="none") * w[b]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        p = net(Xte[:, cols]).argmax(-1)
    print(f"{name:18s} held: changed caught {float((p == Yte)[CHte].float().mean()):.3f}  unchanged right {float((p == Yte)[~CHte].float().mean()):.3f}  copy-baseline on changed {float((Bte == Yte)[CHte].float().mean()):.3f}")
