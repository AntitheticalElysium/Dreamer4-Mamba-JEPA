"""Panel check (2026-10-03): how much of the H16 decision can ONE future per action carry, and how many sampled futures does it
take? (check_h16_signal: 98.5% of the across-action variation of P(dead by 16) is a real first-action effect; sealed deepeval:
oracle 0.711, prior 0.586, real16 one draw 0.652, every world's gen16 0.593-0.602. A deterministic world draws one future per
action; P(dead by 16) is an average over futures.)
deeppanel FIT + DEV rows (judge not read), roots whose P(dead by 16) varies over the 17 first actions ("opp16"); expected safe at
k = 16 per root, averaged over roots:
  uniform              a random first action
  prior                the FIT action prior (lowest mean P16 over FIT opp16 roots), DEV only
  by_h1, by_h4         uniform among the actions minimizing P(dead by 1) / P(dead by 4) (perfect short-horizon knowledge)
  one_real_future      uniform among the actions whose key-0 future survives to 16 (all if none): ONE real future per action
                       with hindsight, shared keys; judged on the other 31 keys (P_rest), so selection and judging are independent
  sampled_M            argmin over M simulated Bernoulli(P16) draws per action (random ties), 20 simulations: M futures per
                       action from a PERFECT stochastic world, M in 1, 2, 4, 8, 16, 32
  oracle31             argmin of P_rest judged on key 0 (unbiased value of knowing P from 31 keys)
Also on "late" roots: opp16 with P(dead by 4) constant over actions (no short-horizon opportunity).
Readings, declared before running: one_sample_insufficient = (one_real_future - uniform) < 0.5 x (oracle31 - uniform) on opp16;
M90 = the smallest M with sampled_M - uniform >= 0.9 x (oracle31 - uniform) (reported).
Usage: check_h16_value.py   (CPU)
"""
import json
import sys

import torch

sys.path.insert(0, "artifacts/experiments/20260927_levers")
import check_h16_signal as HS  # noqa: E402


def arms(P, D, prior_action, g):
    p1, p4, p16 = P[..., 0], P[..., 1], P[..., 2]
    rest16 = (32 * p16 - D[..., 2]) / 31
    tie = lambda m: m.float() / m.float().sum(1, keepdim=True)                           # uniform over a mask
    out = {"roots": len(P), "uniform": float(1 - p16.mean())}
    if prior_action is not None:
        out["prior"] = float(1 - p16[:, prior_action].mean())
    for name, p in (("by_h1", p1), ("by_h4", p4)):
        out[name] = float(1 - (tie(p == p.amin(1, keepdim=True)) * p16).sum(1).mean())
    alive0 = D[..., 2] == 0
    sel = torch.where(alive0.any(1, keepdim=True), alive0, torch.ones_like(alive0))
    out["one_real_future"] = float(1 - (tie(sel) * rest16).sum(1).mean())
    pick = rest16.argmin(1)
    out["oracle31"] = float(1 - D[torch.arange(len(P)), pick, 2].mean())
    out["oracle32_optimistic"] = float(1 - p16.amin(1).mean())
    for M in (1, 2, 4, 8, 16, 32):
        vals = []
        for _ in range(20):
            est = torch.binomial(torch.full_like(p16, float(M)), p16, generator=g) / M
            vals.append(float(1 - (tie(est == est.amin(1, keepdim=True)) * p16).sum(1).mean()))
        out[f"sampled_{M}"] = sum(vals) / len(vals)
    return out


def main():
    P, D, S = HS.load()
    opp16 = P[..., 2].amax(1) > P[..., 2].amin(1)
    late = opp16 & ~(P[..., 1].amax(1) > P[..., 1].amin(1))
    fit = opp16 & (S == 0)
    prior_action = int(P[fit][..., 2].mean(0).argmin())
    g = torch.Generator().manual_seed(20261003)
    out = {"prior_action": prior_action, "opp16": arms(P[opp16], D[opp16], None, g),
           "opp16_dev": arms(P[opp16 & (S == 1)], D[opp16 & (S == 1)], prior_action, g),
           "late": arms(P[late], D[late], None, g)}
    o = out["opp16"]
    gap = o["oracle31"] - o["uniform"]
    out["readings"] = {"one_sample_insufficient": (o["one_real_future"] - o["uniform"]) < 0.5 * gap,
                       "M90": next((M for M in (1, 2, 4, 8, 16, 32) if o[f"sampled_{M}"] - o["uniform"] >= 0.9 * gap), None)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
