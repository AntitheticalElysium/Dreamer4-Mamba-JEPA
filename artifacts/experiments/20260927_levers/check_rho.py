"""E14 check (2026-10-02): can RHO-LOSS select the consequences without labels? (Mindermann et al., ICML 2022, eq. 3: select by
training loss - irreducible holdout loss (IL), the IL from a small model trained on a holdout set; "points selected by high loss
... are overwhelmingly those with noise-corrupted labels". check_toperr: our top-1% loss tail is 93% non-consequence, mostly
entering cells, which no model can predict from what it sees: the "noisy" points.)
IL model: an MLP on each token's raw local input (the token, its 4 grid neighbours, the action; headfit's skip features;
977 -> 512 -> 512 -> 192, GELU, output layer-normed), trained with the uniform per-token L1 on 8,000 training windows (seeded,
disjoint from the held windows), 12 epochs, AdamW 1e-3. Then on tworld's 2,048 held windows, per token: world L1 (trained
world, teacher-forced), IL, reducible = world - IL. Reported per class (check_costwhere's classes): mean world loss, mean IL;
for q in {0.1, 0.3, 1, 3}%: consequence recall and class composition of the top-q set by reducible loss, next to the same for
the top-q set by world loss (check_toperr's selection).
Reading, declared before running:
  rho_selective   consequences are >= 30% of the top-0.3% set by reducible loss with recall >= 0.5 (vs 10% / 0.47 by world
                  loss on s7): a label-free selection that targets them
  il_copies       the IL model's mean L1 on consequence tokens >= 0.8 x the world's (it does not learn them either; RHO's
                  selection then cannot favour them)
Result (2026-10-02, levers_logs/check_rho.log): IL model train L1 0.151 (the world's held all-token L1 is 0.058). IL on consequence
tokens 0.900 vs the world's 0.919 (s7) / 0.738 (s8): il_copies TRUE -- a local MLP trained with the uniform L1 copies them too.
Top-0.3% by reducible loss: 71% static, 10-12% entering, 9-11% near-player, consequence recall 0.000 (s7) / 0.002 (s8);
rho_selective FALSE at both seeds. RHO's IL model faces the same rarity, so its selection cannot favour what it cannot learn.
"""
import sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C
from tworld import local_features

QS = (0.001, 0.003, 0.01, 0.03)
dev = C.dev
torch.manual_seed(0)
main_rows = torch.where(~C.pool["terminal"])[0]
train_rows = torch.cat([main_rows[~torch.isin(main_rows, C.held)], torch.where(C.pool["terminal"])[0]])
il_rows = train_rows[torch.randperm(len(train_rows), generator=torch.Generator().manual_seed(9))[:8000]]
il = nn.Sequential(nn.Linear(977, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 192)).to(dev)
opt = torch.optim.AdamW(il.parameters(), 1e-3, weight_decay=0.01)
for ep in range(12):
    perm = il_rows[torch.randperm(len(il_rows))]
    tot = 0.0
    for i in range(0, len(perm), 32):
        r = perm[i:i + 32]
        s = C.pool["tokens"][r].float().to(dev); a = F.pad(C.pool["actions"][r], (0, 1)).to(dev)
        x = local_features(s, a)[:, :5]
        pred = F.layer_norm(il(x), (192,))
        loss = (pred - s[:, 1:]).abs().mean()
        opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
    print(json.dumps({"il_epoch": ep, "train_l1": round(tot / (len(perm) / 32), 5)}), flush=True)
out = {}
for w in sys.argv[1:]:
    world, st = C.T.load_world(w, dev)
    E, I, K = [], [], []
    with torch.no_grad():
        for i in range(0, len(C.held), 64):
            r = C.held[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]
            ap = F.pad(a, (0, 1)).to(dev)
            with C.autocast_context(C.config):
                pred = world(s.to(dev), ap)[0][:, :5].float().cpu()
            ilp = F.layer_norm(il(local_features(s.to(dev), ap)[:, :5]), (192,)).float().cpu()
            E.append((pred - s[:, 1:]).abs().mean(-1).flatten()); I.append((ilp - s[:, 1:]).abs().mean(-1).flatten())
            K.append(C.classes(r, s).flatten())
    E, I, K = torch.cat(E), torch.cat(I), torch.cat(K)
    cons = K == C.CLASSES.index("consequence")
    res = {"mean_world": {c: round(float(E[K == k].mean()), 4) for k, c in enumerate(C.CLASSES)},
           "mean_il": {c: round(float(I[K == k].mean()), 4) for k, c in enumerate(C.CLASSES)}}
    for sel, score in (("world_loss", E), ("reducible", E - I)):
        rank = torch.empty(len(score), dtype=torch.long); rank[score.argsort(descending=True)] = torch.arange(len(score))
        for q in QS:
            top = rank < int(q * len(score))
            res[f"{sel}_top{q * 100:g}%"] = {"recall_consequence": round(float(top[cons].float().mean()), 4),
                                             "composition": {c: round(float((K[top] == k).float().mean()), 4) for k, c in enumerate(C.CLASSES)}}
    r3 = res["reducible_top0.3%"]
    res["readings"] = {"rho_selective": r3["composition"]["consequence"] >= 0.3 and r3["recall_consequence"] >= 0.5,
                       "il_copies": res["mean_il"]["consequence"] >= 0.8 * res["mean_world"]["consequence"]}
    out[st["name"]] = res
    print(json.dumps({st["name"]: res}), flush=True)
print(json.dumps(out))
