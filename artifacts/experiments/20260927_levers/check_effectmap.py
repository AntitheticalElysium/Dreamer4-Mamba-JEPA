"""E14 check (2026-10-02): a label-free dose WITH matching negatives -- an action-effect map counted from the data.
(Addendum 5: dosing only the changes shifts the change prior (hallucination 24-30%); the attempt mask doses changed AND
unchanged attempts (2-3%). An unconditioned (action, position) count map covers only 42% of consequences at lift >= 0.05 while
dosing 6.4% of tokens, because the agent's facing splits the effect over 4 positions.)
Generic construction (no facing probe, no action names): the agent sits at the centre token of the egocentric view (31);
k-means (K clusters, seeded) on that token gives an agent state. For each (action, agent state) with >= 200 calm frames
(training rows; < 10% of the frame's tokens with copy residual > 60), the change rate per position (copy residual > 60) minus
the no-op rate of that position = lift. The dose set of a frame = the positions with lift >= tau for its (action, agent state),
whether or not they change (matching negatives). Reported for K in {4, 8, 16} and tau in {0.05, 0.1, 0.2}, on all training
frames: consequence recall, dosed share of tokens, consequences among dosed tokens, attempt-mask overlap.
Reading, declared before running:
  effectmap_selective  some (K, tau) reaches consequence recall >= 0.6 with dosed share <= 1% of tokens
"""
import json
import torch

lab = torch.load("artifacts/eda/headfit_labels_v1.pt")
pool = torch.load("artifacts/eda/spatial_pool_v1/pool.pt", weights_only=False, mmap=True)
main_rows = torch.where(~pool["terminal"])[0]
held = main_rows[torch.randperm(len(main_rows), generator=torch.Generator().manual_seed(1))[:2048]]
train = torch.ones(len(pool["actions"]), dtype=torch.bool); train[held] = False
A = pool["actions"]; alive = pool["alive"][:, 1:]
ch = lab["copyres"].float() > 60
calm = (ch.float().mean(-1) < 0.1) & alive
cons = torch.zeros(ch.shape, dtype=torch.bool); cons.scatter_(2, lab["faced"][..., None], lab["cons"][..., None])
att = torch.zeros(ch.shape, dtype=torch.bool); att.scatter_(2, lab["faced"][..., None], lab["attempt"][..., None])
agent = torch.cat([pool["tokens"][i:i + 2048, :5, 31].float() for i in range(0, len(A), 2048)])     # [N,5,192]
out = {}
for K in (4, 8, 16):
    g = torch.Generator().manual_seed(0)
    X = agent[train].flatten(0, 1)
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(25):
        lbl = torch.cdist(X, C).argmin(-1)
        C = torch.stack([X[lbl == k].mean(0) if (lbl == k).any() else C[k] for k in range(K)])
    state = torch.cat([torch.cdist(agent[i:i + 4096].flatten(0, 1), C).argmin(-1).view(-1, 5) for i in range(0, len(A), 4096)])
    tr = train[:, None].expand_as(A) & calm
    base = ch[tr & (A == 0)].float().mean(0)
    for tau in (0.05, 0.1, 0.2):
        dose = torch.zeros(ch.shape, dtype=torch.bool)
        for a in range(17):
            for k in range(K):
                m = tr & (A == a) & (state == k)
                if m.sum() < 200:
                    continue
                pos = ((ch[m].float().mean(0) - base) >= tau)
                if pos.any():
                    sel = (A == a) & (state == k) & alive
                    dose[sel] = dose[sel] | pos
        d = dose[train]; c = cons[train]; t = att[train]
        r = {"recall_consequence": round(float(d[c].float().mean()), 4), "dosed_share": round(float(d.float().mean()), 5),
             "consequences_among_dosed": round(float(c[d].float().mean()), 4), "attempt_tiles_dosed": round(float(d[t].float().mean()), 4),
             "dose_per_token_at_lambda1": round(1 + 1 / max(float(d.float().mean()), 1e-9), 1)}
        out[f"K{K}_tau{tau}"] = r
        print(json.dumps({f"K{K}_tau{tau}": r}), flush=True)
out["readings"] = {"effectmap_selective": any(v["recall_consequence"] >= 0.6 and v["dosed_share"] <= 0.01 for v in out.values())}
print(json.dumps(out["readings"]))
