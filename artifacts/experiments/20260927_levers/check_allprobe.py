"""E14a check (2026-10-02): can the consequence tokens be SINGLED OUT from h among ALL tokens? (E13d/e probed only the faced
cells of attempts: next-class accuracy 0.86 (s7) / 0.83 (s8), change AUC 0.945 / 0.950, so they do not explain why a fresh head
catches 0.002 on the s7 backbone and 0.257 on s8's. The output head must fire "generate" at 0.057% of all tokens and nowhere
else; with a rare positive the relevant quantity is precision at the natural base rate, not AUC.)
Per world: h (backbone output read by the head; position t predicts frame t+1) on teacher-forced pool windows. Label per token:
strict consequence (headfit_labels_v1). Probes fitted on training windows (every positive of 6,000 consequence-rich windows +
2% of their negatives, negatives re-weighted x50 to the natural rate), scored on tworld's 2,048 held-out windows at the
natural rate (all tokens): AUC, average precision (AP), precision at recall 0.5, recall at precision 0.5. Probes: logistic
(linear) and a 2-layer MLP (512, GELU), both on h; the same on the input (the token itself + its 4 neighbours + the action, the
information available before the backbone) as the reference.
Reading, declared before running:
  linear_bottleneck   MLP AP >= 2 x linear AP on h (an expressive readout singles them out, a linear one cannot)
  representation      MLP AP on h < 0.3 (h does not single them out at the natural rate, whatever the readout)
  seed_difference     s8's linear AP on h >= 2 x s7's (the backbone property that lets a fresh head catch 0.257)
"""
import sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers"); sys.path.insert(0, "artifacts/experiments/20260926_diagnosis"); sys.path.insert(0, "artifacts/experiments/20260921_readout_ladder")
import teval as T
from tworld import POOLS
from d4mj.config import config_from_dict
from d4mj.train import autocast_context
import spatial as Sp
dev = torch.device("cuda")
config = config_from_dict(torch.load(Sp.CHECKPOINT, map_location="cpu", weights_only=False)["config"])
pool = torch.load(POOLS["raw"] / "pool.pt", weights_only=False, mmap=True)
lab = torch.load("artifacts/eda/headfit_labels_v1.pt")
n = len(lab["cons"])
cons = torch.zeros(n, 5, 81, dtype=torch.bool); cons.scatter_(2, lab["faced"][..., None], lab["cons"][..., None])
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
train_rows = torch.cat([main_rows[~torch.isin(main_rows, held)], torch.where(pool["terminal"])[0]])
rich = train_rows[cons[train_rows].flatten(1).any(1)]
rich = rich[torch.randperm(len(rich), generator=torch.Generator().manual_seed(3))[:6000]]
G = torch.Generator().manual_seed(4)
NEG_KEEP = 0.02


def inputs(s, a):
    """[b,5,81, 192*5 + 17]: the token, its 4 neighbours (zero off-grid) and the action, per position t (predicting t+1)."""
    g = s[:, :5].view(-1, 5, 9, 9, 192); p = F.pad(g, (0, 0, 1, 1, 1, 1))
    nb = [g] + [p[:, :, 1 + dr:10 + dr, 1 + dc:10 + dc] for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))]
    act = F.one_hot(a[:, :5], 17).float()[:, :, None, None].expand(-1, -1, 9, 9, -1)
    return torch.cat(nb + [act], -1).view(len(s), 5, 81, -1)


@torch.no_grad()
def collect(world, rows, keep_all):
    H, X, Y, W = [], [], [], []
    for i in range(0, len(rows), 32):
        r = rows[i:i + 32]
        s = pool["tokens"][r].float().to(dev); a = pool["actions"][r].to(dev)
        x = world.codes[__import__("tworld").quantize(s, world.codes)] if world.head == "categorical" else s
        with autocast_context(config):
            h = world(x, F.pad(a, (0, 1)))[1][:, :5].float()
        y = cons[r].to(dev)
        inp = inputs(s, a)
        if keep_all:
            m = torch.ones_like(y)
        else:
            m = y | (torch.rand(y.shape, generator=G).to(dev) < NEG_KEEP)
        H.append(h[m].cpu().half()); X.append(inp[m].cpu().half()); Y.append(y[m].cpu())
        W.append(torch.where(y[m], 1.0, 1.0 / (1.0 if keep_all else NEG_KEEP)).cpu())
    return torch.cat(H), torch.cat(X), torch.cat(Y), torch.cat(W)


def ap_metrics(score, y):
    order = score.argsort(descending=True); ys = y[order].float()
    tp = ys.cumsum(0); prec = tp / torch.arange(1, len(ys) + 1); rec = tp / ys.sum()
    ap = float((prec * ys).sum() / ys.sum())
    p_at_r5 = float(prec[(rec >= 0.5).nonzero()[0, 0]]) if (rec >= 0.5).any() else 0.0
    ok = prec >= 0.5
    r_at_p5 = float(rec[ok].max()) if ok.any() else 0.0
    pos, neg = score[y], score[~y]
    ranks = torch.cat([pos, neg]).argsort().argsort().float() + 1
    auc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))
    return {"auc": auc, "ap": ap, "precision_at_recall_0.5": p_at_r5, "recall_at_precision_0.5": r_at_p5}


def fit(Xtr, ytr, wtr, Xte, mlp, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.float().mean(0), Xtr.float().std(0).clamp(min=1e-6)
    d = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(d, 512), nn.GELU(), nn.Linear(512, 1)) if mlp else nn.Linear(d, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    Xg, yg, wg = ((Xtr.float() - mu) / sd).to(dev), ytr.float().to(dev), wtr.to(dev)
    for ep in range(30):
        perm = torch.randperm(len(yg), device=dev)
        for j in range(0, len(perm), 1024):
            b = perm[j:j + 1024]
            loss = (F.binary_cross_entropy_with_logits(net(Xg[b])[:, 0], yg[b], reduction="none") * wg[b]).sum() / wg[b].sum()
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.cat([net(((Xte[i:i + 65536].float() - mu) / sd).to(dev))[:, 0].cpu() for i in range(0, len(Xte), 65536)])


out = {}
ref = None
for w in sys.argv[1:]:
    world, st = T.load_world(w, dev)
    name = st["name"]
    Htr, Xtr, ytr, wtr = collect(world, rich, False)
    Hte, Xte, yte, _ = collect(world, held, True)
    res = {"train_tokens": len(ytr), "train_pos": int(ytr.sum()), "held_tokens": len(yte), "held_pos": int(yte.sum()),
           "base_rate": float(yte.float().mean())}
    for feat, (A, B) in (("h", (Htr, Hte)),) + ((("input", (Xtr, Xte)),) if ref is None else ()):
        for kind in ("linear", "mlp"):
            res[f"{feat}_{kind}"] = ap_metrics(fit(A, ytr, wtr, B, kind == "mlp"), yte)
    if ref is None:
        ref = {k: res[k] for k in ("input_linear", "input_mlp")}
    res.update(ref)
    res["readings"] = {"linear_bottleneck": res["h_mlp"]["ap"] >= 2 * res["h_linear"]["ap"], "representation": res["h_mlp"]["ap"] < 0.3}
    out[name] = res
    print(json.dumps({name: res}), flush=True)
    del world, Htr, Hte; torch.cuda.empty_cache()
names = list(out)
s7 = [n for n in names if "s7" in n and "corrt" in n]; s8 = [n for n in names if "s8" in n and "corrt" in n]
if s7 and s8:
    out["seed_difference"] = out[s8[0]]["h_linear"]["ap"] >= 2 * out[s7[0]]["h_linear"]["ap"]
print(json.dumps(out))
