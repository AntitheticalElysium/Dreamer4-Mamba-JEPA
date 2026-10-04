"""Item 4 check (2026-10-02): where is the agent's turn lost in the dosed worlds? (check_facing s7: turns drawn right 0.985 / 0.950
(moved / blocked) at baseline, 0.014 / 0.002 with mask1 + skip, 0.16 / 0.12 with mask0.1 + skip, 0.998 / 0.956 at 36k; the refit
put the damage in the backbone; E14d's mask gradient is NOT dominant (0.13x), so dominance is not the mechanism.)
On move transitions (actions 1-4) of pool windows, h at the player token 31 (the backbone output the head reads; position t
predicts t+1): linear probes (multinomial logistic, fitted on 6,000 training windows, scored on tworld's 2,048 held windows) for
(a) the NEXT facing (teval facing probe on the true t+1 player token), (b) the CURRENT facing (on the t token), (c) the action
(1-4); the same probes on the raw input token 31 + the action one-hot as the reference. And the corr head's mixture weights at
token 31 on move steps (self / generate), held windows.
Reading, declared before running:
  backbone_lost_turn   (a) on the dosed world's h is >= 0.2 below the baseline's while (b) is within 0.05: h still knows where
                       the agent faces but no longer where it will face (the action -> facing computation is lost in h)
  head_copies          (a) within 0.05 of the baseline's but the self weight at token 31 on moves >= 0.9: h knows, the head copies
Usage: check_facing_h.py <world.pt> ...
"""
import sys, json, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_costwhere as C

probes = C.T.Probes(C.T.build_cache("raw", torch.device("cpu")), *C.T.split())
main_rows = torch.where(~C.pool["terminal"])[0]
train_rows = main_rows[~torch.isin(main_rows, C.held)][torch.randperm(len(main_rows) - 2048, generator=torch.Generator().manual_seed(2))[:6000]]


@torch.no_grad()
def collect(world, rows):
    H, X, Y1, Y0, A, W = [], [], [], [], [], []
    for i in range(0, len(rows), 64):
        r = rows[i:i + 64]; s = C.pool["tokens"][r].float(); a = C.pool["actions"][r]; alive = C.pool["alive"][r][:, 1:]
        with C.autocast_context(C.config):
            out = world(s.to(C.dev), F.pad(a, (0, 1)).to(C.dev))
        h = out[1][:, :5, 31].float().cpu() if world is not None else None
        lw = world.last_weights[:, :5, 31].float().cpu() if hasattr(world, "last_weights") else None
        for t in range(5):
            m = (a[:, t] >= 1) & (a[:, t] <= 4) & alive[:, t]
            if not m.any():
                continue
            H.append(h[m, t]); X.append(torch.cat([s[m, t, 31], F.one_hot(a[m, t], 17).float()], 1))
            Y1.append(probes.facing(s[m, t + 1, 31]).argmax(-1)); Y0.append(probes.facing(s[m, t, 31]).argmax(-1)); A.append(a[m, t] - 1)
            if lw is not None:
                W.append(lw[m, t])
    return torch.cat(H), torch.cat(X), torch.cat(Y1), torch.cat(Y0), torch.cat(A), (torch.cat(W) if W else None)


def probe(Xtr, ytr, Xte, yte, k):
    torch.manual_seed(0)
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    net = nn.Linear(Xtr.shape[1], k); opt = torch.optim.AdamW(net.parameters(), 1e-2, weight_decay=1e-4)
    Xn, Xt = (Xtr - mu) / sd, (Xte - mu) / sd
    for _ in range(300):
        loss = F.cross_entropy(net(Xn), ytr); opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float((net(Xt).argmax(-1) == yte).float().mean())


out = {}
for p in sys.argv[1:]:
    w, st = C.T.load_world(p, C.dev)
    Htr, Xtr, Y1tr, Y0tr, Atr, _ = collect(w, train_rows)
    Hte, Xte, Y1te, Y0te, Ate, Wte = collect(w, C.held)
    r = {"n_train": len(Y1tr), "n_held": len(Y1te), "turn_share_held": float((Y1te != Y0te).float().mean()),
         "h_next_facing": probe(Htr, Y1tr, Hte, Y1te, 4), "h_current_facing": probe(Htr, Y0tr, Hte, Y0te, 4),
         "h_action": probe(Htr, Atr, Hte, Ate, 4), "input_next_facing": probe(Xtr, Y1tr, Xte, Y1te, 4)}
    if Wte is not None:
        r["mix_self_on_moves"] = float(Wte[:, 0].mean()); r["mix_generate_on_moves"] = float(Wte[:, 5].mean())
        turn = Y1te != Y0te
        r["mix_self_on_turns"] = float(Wte[turn, 0].mean()); r["mix_generate_on_turns"] = float(Wte[turn, 5].mean())
    out[st["name"]] = r
    print(json.dumps({st["name"]: r}), flush=True)
base = [k for k in out if k.endswith("teacher_s7_u18000")]
if base:
    b = out[base[0]]
    out["readings"] = {k: {"backbone_lost_turn": v["h_next_facing"] <= b["h_next_facing"] - 0.2 and abs(v["h_current_facing"] - b["h_current_facing"]) <= 0.05,
                           "head_copies": abs(v["h_next_facing"] - b["h_next_facing"]) <= 0.05 and v.get("mix_self_on_moves", 0) >= 0.9}
                       for k, v in out.items() if k != base[0]}
print(json.dumps(out))
