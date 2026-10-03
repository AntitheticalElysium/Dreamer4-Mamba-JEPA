"""Queue check (2026-10-03): is "~24% of entering terrain cannot be inferred from what is visible" a property of the data, or of
the weak predictor that measured it? (check_enterpred: held-out entering cells, class accuracy: MLP on the 3 visible edge tokens
0.756, corrt teacher s7 teacher-forced 0.760, copy the adjacent tile 0.714, majority 0.476. The E13 review: an MLP baseline is not
a ceiling.)
Same windows (spatial_pool_v1: 8,000 TRAIN main windows, seed 2; the 2,048 held-out main windows, seed 1), same labels (teval's
tile probe on the true tokens), same entering cells (scroll steps by scroll.estimate, alive next frame). Stronger input: the tile
classes (probe argmax, one-hot 17) of ALL 63 map cells of the current frame and of up to 3 previous frames of the window (zeros
where the window has none), the scroll direction (4) and the entering cell's position along the edge (9). MLP 2 x 512, CE,
AdamW 1e-3, wd 1e-4, 20% of the training cells for early stopping (best epoch of 60).
Reading, declared before running: terrain_bound_holds = held-out accuracy <= 0.78 (within ~0.02 of the 3-edge MLP and the world);
otherwise the "~24% unknowable" statement is retracted and replaced by the measured figure.
Usage: check_enterbound.py   (CPU)
Result v1 (2026-10-03): held accuracy 0.645 (val 0.72; 60 epochs), majority 0.476. The reading is formally TRUE but the check is
INVALID as a ceiling test: this "stronger" predictor is below copying the adjacent tile (0.714) and the 3-edge MLP (0.756). The
frame-wide class one-hots make the MLP learn the entering-cell / edge-cell correspondence per direction, and the probe argmax
discards token detail the baseline used. Replaced by v2 (`--aligned`): tokens of a patch aligned to the entering cell.
v2 (`--aligned`, declared before its run): per entering cell, the TOKENS of frame t's cells at depth 1-4 into the view along the
scroll axis and lateral offset -3..+3 (canonical orientation: depth axis = scroll axis), zeros + a validity flag off-view, the
scroll direction (4); a strict superset of the 3-edge input (depth 1, lateral -1..+1). Same MLP / training / reading.
"""
import json
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis")
sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T  # noqa: E402
from scroll import SHIFTS, estimate  # noqa: E402
from tworld import POOLS  # noqa: E402

HIST = 3


def collect(pool, rows, probes):
    X, Y = [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]
        s = pool["tokens"][r].float(); alive = pool["alive"][r]
        cls = probes.tile(s[:, :, :63].flatten(0, 2)).argmax(-1).view(len(r), 6, 63)                # [b,6,63]
        onehot = F.one_hot(cls, 17).float()                                                        # [b,6,63,17]
        for t in range(5):
            sh = estimate(s[:, t], s[:, t + 1])
            for j in torch.nonzero((sh != 0) & alive[:, t + 1])[:, 0].tolist():
                dr, dc = SHIFTS[int(sh[j])]
                cells = [((6 if dr == 1 else 0), c) for c in range(9)] if dr else [(rr, (8 if dc == 1 else 0)) for rr in range(7)]
                past = [onehot[j, t - h].flatten() if t - h >= 0 else torch.zeros(63 * 17) for h in range(HIST + 1)]
                base = torch.cat(past + [F.one_hot(torch.tensor(int(sh[j]) - 1), 4).float()])
                for pos, (rr, cc) in enumerate(cells):
                    X.append(torch.cat([base, F.one_hot(torch.tensor(pos), 9).float()]))
                    Y.append(int(cls[j, t + 1, rr * 9 + cc]))
    return torch.stack(X), torch.tensor(Y)


def collect_aligned(pool, rows, probes, depth=4, lateral=3):
    X, Y = [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]
        s = pool["tokens"][r].float(); alive = pool["alive"][r]
        for t in range(5):
            sh = estimate(s[:, t], s[:, t + 1])
            for j in torch.nonzero((sh != 0) & alive[:, t + 1])[:, 0].tolist():
                dr, dc = SHIFTS[int(sh[j])]
                g0 = s[j, t, :63].view(7, 9, -1)
                lab = probes.tile(s[j, t + 1, :63]).argmax(-1).view(7, 9)
                cells = [((6 if dr == 1 else 0), c) for c in range(9)] if dr else [(rr, (8 if dc == 1 else 0)) for rr in range(7)]
                for rr, cc in cells:
                    feats, valid = [], []
                    for d in range(1, depth + 1):                      # entering cell = (rr + dr, cc + dc) in frame t coordinates;
                        for l in range(-lateral, lateral + 1):         # d cells back into the view (d = 1: the edge), l across
                            fr = rr + dr - d * dr + (l if dc else 0)
                            fc = cc + dc - d * dc + (l if dr else 0)
                            ok = 0 <= fr < 7 and 0 <= fc < 9
                            feats.append(g0[fr, fc] if ok else torch.zeros(192)); valid.append(float(ok))
                    X.append(torch.cat(feats + [torch.tensor(valid), F.one_hot(torch.tensor(int(sh[j]) - 1), 4).float()]))
                    Y.append(int(lab[rr, cc]))
    return torch.stack(X), torch.tensor(Y)


def main():
    torch.manual_seed(0)
    global collect
    if "--aligned" in sys.argv:
        collect = collect_aligned
    meta, tr, ts = T.split()
    probes = T.Probes(T.build_cache("raw", torch.device("cpu")), meta, tr, ts)
    pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
    main_rows = torch.where(~pool["terminal"])[0]
    held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
    train = main_rows[~torch.isin(main_rows, held)][torch.randperm(len(main_rows) - 2048, generator=torch.Generator().manual_seed(2))[:8000]]
    Xtr, ytr = collect(pool, train, probes)
    Xte, yte = collect(pool, held, probes)
    n_val = len(ytr) // 5
    perm = torch.randperm(len(ytr))
    va, trn = perm[:n_val], perm[n_val:]
    net = nn.Sequential(nn.Linear(Xtr.shape[1], 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 17))
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    best, state, curve = -1.0, None, []
    for ep in range(60):
        net.train()
        p = trn[torch.randperm(len(trn))]
        for j in range(0, len(p), 512):
            b = p[j:j + 512]
            loss = F.cross_entropy(net(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            acc = float((net(Xtr[va]).argmax(-1) == ytr[va]).float().mean())
        curve.append(round(acc, 4))
        if acc > best:
            best, state = acc, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    with torch.no_grad():
        pred = net(Xte).argmax(-1)
    held_acc = float((pred == yte).float().mean())
    maj = int(torch.bincount(ytr, minlength=17).argmax())
    out = {"train_cells": len(ytr), "held_cells": len(yte), "val_best": best, "val_curve": curve,
           "held_accuracy": held_acc, "majority": float((yte == maj).float().mean()),
           "per_class_held_recall": {int(c): round(float((pred[yte == c] == c).float().mean()), 3)
                                     for c in yte.unique() if int((yte == c).sum()) >= 50}}
    out["readings"] = {"terrain_bound_holds": held_acc <= 0.78}
    print(json.dumps(out))


if __name__ == "__main__":
    main()
