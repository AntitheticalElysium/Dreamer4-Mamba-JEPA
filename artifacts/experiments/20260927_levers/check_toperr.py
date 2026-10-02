"""E14a check (2026-10-02): can the dose come from the model's own error instead of Craftax labels? (Head-only: consequences
need a dose >= x304 (mask1); x31 does nothing. A generic hard-token term, mean + lambda * mean over the top-q fraction of tokens
by the current per-token error (OHEM, Shrivastava et al. 2016; SimPLe's clipping in spirit), doses whatever sits in the error
tail by ~lambda / q.) Held pool windows, teacher-forced, per-token L1 (the training objective's token term). For q in
{0.1, 0.3, 1, 3}%: the share of consequence tokens inside the top-q set (recall), the class composition of the set
(check_costwhere's classes), and the dose on a consequence inside the set at lambda = 1 (1 + 1 / q).
Worlds: the trained 18k worlds and their 6k snapshots (the error tail during training, before / after copying is learned).
"""
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C

QS = (0.001, 0.003, 0.01, 0.03)
out = {}
for w in sys.argv[1:]:
    world, st = C.T.load_world(w, C.dev)
    E, K = [], []
    with torch.no_grad():
        for i in range(0, len(C.held), 64):
            r = C.held[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]
            with C.autocast_context(C.config):
                pred = world(s.to(C.dev), F.pad(a, (0, 1)).to(C.dev))[0][:, :5].float().cpu()
            E.append((pred - s[:, 1:]).abs().mean(-1).flatten()); K.append(C.classes(r, s).flatten())
    E, K = torch.cat(E), torch.cat(K)
    cons = K == C.CLASSES.index("consequence")
    res = {"tokens": len(E), "consequences": int(cons.sum()), "mean_err": {c: round(float(E[K == k].mean()), 4) for k, c in enumerate(C.CLASSES)}}
    order = E.argsort(descending=True)
    rank = torch.empty_like(order); rank[order] = torch.arange(len(E))
    res["consequence_rank_quantiles"] = {f"q{int(p * 100)}": round(float(rank[cons].float().quantile(p) / len(E)), 5) for p in (0.25, 0.5, 0.75)}
    for q in QS:
        top = rank < int(q * len(E))
        res[f"top{q * 100:g}%"] = {"recall_consequence": round(float(top[cons].float().mean()), 4),
                                    "dose_at_lambda1": round(1 + 1 / q, 1),
                                    "composition": {c: round(float((K[top] == k).float().mean()), 4) for k, c in enumerate(C.CLASSES)}}
    out[st["name"]] = res
    print(json.dumps({st["name"]: res}), flush=True)
print(json.dumps(out))
