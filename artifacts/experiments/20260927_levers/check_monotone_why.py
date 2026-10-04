"""E14m check (2026-10-02): why is revealed terrain monotone, and do the categorical / direct worlds ever scroll?
(1) Scroll rate: share of imagined steps (diagnosis futures, monotone.py's rollout, test roots, sample 0 alive) whose drawn frame
    scrolls (scroll.estimate against the previous drawn frame), vs the true share.
(2) Entering cells, one step, teacher-forced on held pool windows (check_enterpred's cells): class histogram of the world's
    prediction, of an MLP on the 3 visible edge tokens (the local deterministic predictor), and of the truth; per-class recall;
    and the squared distance of the predicted / true entering token to its nearest of the 4,096 k-means codes of true training
    tokens (levers_codebooks_v1): off-manifold (blurred) tokens read as the majority class would show as a large distance.
Result (2026-10-02, log levers_logs/check_monotone_why.log). True scroll rate 0.369.
  corrt teacher s7 / s8 18k: imagined scroll rate 0.276 / 0.323; 27,976 held entering cells; accuracy 0.760 / 0.758 (MLP edge 0.756);
    class shares true / world s7 / world s8 / MLP: grass 0.476 / 0.528 / 0.527 / 0.527, tree 0.037 / 0.0012 / 0.0007 / 0.0002,
    stone 0.172 / 0.186 / 0.190 / 0.186, water 0.116 / 0.109 / 0.108 / 0.117; tree recall 0.003 / 0.0 / 0.0 (MLP).
    Nearest-code distance (median): true 2.69, world 2.75 / 2.72; at true-tree cells true 1.98, world 0.92 / 0.90 (a crisp grass token).
    -> the world's entering content IS the local deterministic predictor's: the mode under unremovable uncertainty, not blur.
  categorical s7 / direct s7: imagined scroll rate 0.0002 / 0.0: they never draw a scroll; their entering-cell "predictions" are the
    old frame left in place (accuracy 0.715 / 0.724 = copying the same screen cell, 0.714 in check_enterpred)."""
import sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
import monotone as M
from tworld import POOLS
from scroll import SHIFTS, estimate
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as Sp
torch.manual_seed(0)
dev = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
meta, tr, ts = T.split()
cache = T.build_cache("raw", torch.device("cpu"))
probes = T.Probes(cache, meta, tr, ts)
codes = torch.load("artifacts/eda/levers_codebooks_v1/raw_K4096.pt", weights_only=False)
codes = (codes["codes"] if isinstance(codes, dict) else codes).float().to(dev)
near = lambda x: torch.cat([torch.cdist(x[i:i + 8192].to(dev), codes).min(-1).values ** 2 for i in range(0, len(x), 8192)]).cpu()
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
train = main_rows[~torch.isin(main_rows, held)][torch.randperm(len(main_rows) - 2048, generator=torch.Generator().manual_seed(2))[:8000]]
ctx, ca, fa, fut = cache["ctx"], cache["ctx_a"], cache["fut_a"], cache["fut"]
R, root = len(ctx), ctx[:, -1]
test = ~tr
alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
prev = torch.cat([root[:, None], fut[:, :-1]], 1)
true_sh = torch.cat([estimate(prev[i:i + 32].float(), fut[i:i + 32].float()) for i in range(0, R, 32)])
m = test[:, None] & alive
out = {"true_scroll_rate": float((true_sh[m] != 0).float().mean())}


def collect(world, rows):
    F_, Y, W = [], [], []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            r = rows[i:i + 64]; s = pool["tokens"][r].float(); a = pool["actions"][r]; al = pool["alive"][r]
            if world is not None:
                x = s.to(dev)
                if world.head == "categorical":
                    from tworld import quantize
                    x = world.codes[quantize(x, world.codes)]
                with autocast_context(config):
                    pred = world(x, F.pad(a, (0, 1)).to(dev))[0].float().cpu()
            for t in range(5):
                sh = estimate(s[:, t], s[:, t + 1])
                for j in torch.nonzero((sh != 0) & al[:, t + 1])[:, 0].tolist():
                    dr, dc = SHIFTS[int(sh[j])]
                    g0, g1 = s[j, t, :63].view(7, 9, -1), s[j, t + 1, :63].view(7, 9, -1)
                    cells = [((6 if dr == 1 else 0), c) for c in range(9)] if dr else [(rr, (8 if dc == 1 else 0)) for rr in range(7)]
                    for (rr, cc) in cells:
                        er, ec = min(max(rr + dr, 0), 6), min(max(cc + dc, 0), 8)
                        F_.append(torch.cat([g0[er, ec], g0[max(er - 1, 0), ec] if dc else g0[er, max(ec - 1, 0)],
                                             g0[min(er + 1, 6), ec] if dc else g0[er, min(ec + 1, 8)]]))
                        Y.append(g1[rr, cc])
                        if world is not None:
                            W.append(pred[j, t, :63].view(7, 9, -1)[rr, cc])
    return torch.stack(F_), torch.stack(Y), (torch.stack(W) if W else None)


Ftr, Ytr, _ = collect(None, train)
ytr = probes.tile(Ytr).argmax(-1)
net = nn.Sequential(nn.Linear(Ftr.shape[1], 256), nn.ReLU(), nn.Linear(256, 17)); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
mu, sd = Ftr.mean(0), Ftr.std(0).clamp(min=1e-6)
for ep in range(40):
    perm = torch.randperm(len(ytr))
    for j in range(0, len(perm), 512):
        b = perm[j:j + 512]; loss = F.cross_entropy(net((Ftr[b] - mu) / sd), ytr[b]); opt.zero_grad(); loss.backward(); opt.step()
hist = lambda c: torch.bincount(c, minlength=17).float() / len(c)
for w in sys.argv[1:]:
    world, st = T.load_world(w, dev)
    name = st["name"]
    gen = M.rollout(world, ctx, ca, fa, dev, config, False)
    gprev = torch.cat([root[:, None], gen[:, :-1]], 1)
    img_sh = torch.cat([estimate(gprev[i:i + 32].float(), gen[i:i + 32].float()) for i in range(0, R, 32)])
    Fte, Yte, Wte = collect(world, held)
    yte, wc = probes.tile(Yte).argmax(-1), probes.tile(Wte).argmax(-1)
    with torch.no_grad():
        pc = net((Fte - mu) / sd).argmax(-1)
    ht, hw, hp = hist(yte), hist(wc), hist(pc)
    res = {"imagined_scroll_rate": float((img_sh[m] != 0).float().mean()),
           "entering_n": len(yte), "accuracy": {"world": float((wc == yte).float().mean()), "mlp_edge": float((pc == yte).float().mean())},
           "share": {M.NAMES[c]: {"true": round(float(ht[c]), 4), "world": round(float(hw[c]), 4), "mlp_edge": round(float(hp[c]), 4)}
                     for c in range(17) if ht[c] >= 0.005},
           "recall": {M.NAMES[c]: {"world": round(float((wc[yte == c] == c).float().mean()), 3), "mlp_edge": round(float((pc[yte == c] == c).float().mean()), 3)}
                      for c in range(17) if ht[c] >= 0.005},
           "nearest_code_sqdist": {"true_entering": float(near(Yte).median()), "world_entering": float(near(Wte).median())},
           "nearest_code_sqdist_by_true_class": {M.NAMES[c]: {"true": round(float(near(Yte[yte == c]).median()), 2), "world": round(float(near(Wte[yte == c]).median()), 2)}
                                                 for c in range(17) if ht[c] >= 0.02}}
    out[name] = res
    print(json.dumps({name: res}), flush=True)
    del world; torch.cuda.empty_cache()
print(json.dumps(out))
