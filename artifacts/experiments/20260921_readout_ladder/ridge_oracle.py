"""Challenge to the spectral mechanism: optimization/allocation, or predictability?

WHY.md link 2 claims the u->u transition learns the high-variance scroll first and the low-variance
consequence (the health cell) last, because gradient descent is greedy in variance. The alternative:
the consequence is simply not predictable from the world's input. Closed-form least squares separates
them -- it solves every output dimension independently (no training dynamics, no competition between
outputs), so its fit to the tail depends only on what the inputs make predictable.

Model: ridge regression from phi = [the 4 context u (768), the 3 past actions one-hot, the candidate
action one-hot, (last context u) x (candidate action one-hot) (3,264 interaction terms)] to the next u
(192), fitted on the interface pool's TRAIN windows (frame 4 from 0-3 and frame 5 from 1-4, 65,294
transitions), standardized features, penalty chosen on a held-out 10% of windows (seeded). Plain
unweighted squared error. POST HOC, exploratory: the fatal direction is fitted on FIT real successors
(transition_diag); judged on the 56k block (read three times) against the U worlds' numbers there.

Reading (committed before the run):
  ridge generated AUC along the fatal direction (all opportunity roots) >= 0.80
      -> consequence_predictable: the Mamba worlds' shortfall (0.61-0.64) is optimization/allocation
  <= 0.66 -> not_linearly_predictable: the limit is predictability / function class, not allocation
  otherwise -> mixed
Reported: zombie roots, effect correlation along the direction, top-10 and tail (ranks 60-192) effect R^2.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from d4mj.data import _sha256

N = 17
JUDGE = ROOT / "artifacts/eda/observe_fresh_v7"


def features(ctx, past, action):
    """ctx [n,4,192], past [n,3] long, action [n] long -> phi [n, 768 + 51 + 17 + 3264]."""
    a = F.one_hot(action, N).float()
    return torch.cat([ctx.flatten(1), F.one_hot(past, N).float().flatten(1), a,
                      (ctx[:, -1, :, None] * a[:, None, :]).flatten(1)], 1)


def main():
    started = time.time()
    log = lambda **kw: print(json.dumps({**kw, "seconds": round(time.time() - started, 1)}), flush=True)
    device = torch.device("cuda")
    os.chdir(ROOT)
    from d4mj.lewm_diagnostics import FORK_STORE
    from boundary import judge_store
    from confirm import seeds_for
    from diagnose import within_auc
    from frozen_ladder import strata
    from interface import POOL, encode, load_bridge, project
    from observability import load
    from transition_diag import centre, direction
    from u_world import successors

    pool = torch.load(POOL / "pool.pt", weights_only=False, mmap=True)
    u, acts = pool["u"], pool["actions"]
    X, Y = [], []
    for s in (0, 1):
        X.append(features(u[:, s:s + 4], acts[:, s:s + 3], acts[:, s + 3]))
        Y.append(u[:, s + 4])
    X, Y = torch.cat(X), torch.cat(Y)
    n = len(u)
    hold = torch.zeros(n, dtype=torch.bool)
    hold[torch.randperm(n, generator=torch.Generator().manual_seed(20261017))[: n // 10]] = True
    hold = torch.cat([hold, hold])
    mean, scale = X[~hold].mean(0), X[~hold].std(0).clamp_min(1e-6)
    ym = Y[~hold].mean(0)

    def gram(rows):
        G = torch.zeros(X.shape[1], X.shape[1], dtype=torch.float64, device=device)
        B = torch.zeros(X.shape[1], 192, dtype=torch.float64, device=device)
        idx = torch.where(rows)[0]
        for j in range(0, len(idx), 8192):
            A = ((X[idx[j:j + 8192]] - mean) / scale).to(device).double()
            G += A.T @ A
            B += A.T @ (Y[idx[j:j + 8192]] - ym).to(device).double()
        return G, B

    def solve(G, B, lam):
        eye = torch.eye(G.shape[0], device=device, dtype=torch.float64)
        return torch.linalg.solve(G + lam * eye, B).float().cpu()

    predict = lambda W, x: ((x - mean) / scale) @ W + ym
    G_tr, B_tr = gram(~hold)
    best = None
    for lam in (1e1, 1e2, 1e3, 1e4, 1e5):
        W = solve(G_tr, B_tr, lam)
        err = float((predict(W, X[hold]) - Y[hold]).square().sum(-1).mean())
        log(stage="ridge", lam=lam, held_out_sse=round(err, 3))
        if best is None or err < best[0]:
            best = (err, lam)
    G_h, B_h = gram(hold)
    W = solve(G_tr + G_h, B_tr + B_h, best[1])
    del G_tr, B_tr, G_h, B_h
    del X, Y

    encoder, _ = load_bridge()
    fit_seeds, _ = seeds_for(json.loads((HERE / "evidence/root_partition.json").read_text()), FORK_STORE)
    fit = load(fit_seeds)["fit"]
    fit["successors"] = successors(fit_seeds)["fit"][0]
    judge, manifest, _ = judge_store(JUDGE)
    judge["successors"] = torch.stack([r["successors"] for f in sorted(JUDGE.glob("seed-*.pt"))
                                       for r in torch.load(f, weights_only=False)])

    def states(frames):
        out = []
        for i in range(0, len(frames), 64):
            _, grid = encode(encoder, frames[i:i + 64], device)
            out.append(project(pool["pca"], grid))
        return torch.cat(out)

    def fan(d):
        ctx = states(d["frames"][:, -4:])
        past = d["actions"][:, -3:].argmax(-1)
        gen = torch.stack([predict(W, features(ctx, past, torch.full((len(ctx),), a))) for a in range(N)], 1)
        real = states(d["successors"].flatten(0, 1)[:, None])[:, 0].view(-1, N, 192)
        return gen, real
    gen_f, real_f = fan(fit)
    gen_j, real_j = fan(judge)
    log(stage="fanned")

    fatal_f, fatal_j = fit["p_death1"] > 0.5, judge["p_death1"] > 0.5
    opp_f, opp_j = fatal_f.any(1) & (~fatal_f).any(1), fatal_j.any(1) & (~fatal_j).any(1)
    w = direction(centre(real_f, torch.ones_like(fatal_f))[opp_f].reshape(-1, 192), fatal_f[opp_f].reshape(-1).float())
    var_u = pool["u"][~pool["terminal"]][:, 1:].reshape(-1, 192).var(0)
    order = var_u.argsort(descending=True)
    strat = strata(judge["visible"])
    rc, gc = centre(real_j, torch.ones_like(fatal_j)), centre(gen_j, torch.ones_like(fatal_j))
    result = {}
    for sname, mask in (("all", opp_j), ("zombie", opp_j & strat["zombie_adjacent"])):
        r, g = rc[mask].reshape(-1, 192), gc[mask].reshape(-1, 192)
        r2 = lambda idx: float(1 - (g[:, idx] - r[:, idx]).square().sum() / r[:, idx].square().sum())
        result[sname] = {"roots": int(mask.sum()),
                         "real_auc": within_auc(list((real_j @ w)[mask]), list(fatal_j[mask])),
                         "generated_auc": within_auc(list((gen_j @ w)[mask]), list(fatal_j[mask])),
                         "effect_corr": float(torch.corrcoef(torch.stack((g @ w, r @ w)))[0, 1]),
                         "effect_ratio": float(((g @ w).square().mean() / (r @ w).square().mean()).sqrt()),
                         "effect_r2_all": r2(order), "top10_effect_r2": r2(order[:10]), "tail_effect_r2": r2(order[60:])}
        log(stage="judge", subset=sname, **{k: round(v, 4) if isinstance(v, float) else v for k, v in result[sname].items()})
    a = result["all"]["generated_auc"]
    reading = "consequence_predictable" if a >= 0.80 else "not_linearly_predictable" if a <= 0.66 else "mixed"
    evidence = {"schema": "d4mj_ridge_oracle_v1", "status": "POST HOC, exploratory (56k read three times)",
                "script_sha256": _sha256(Path(__file__)), "judge_manifest": manifest, "ridge_lambda": best[1],
                "held_out_sse": best[0], "reading": reading, "result": result,
                "u_worlds_on_56k": {"U_s1": 0.6357, "U_s2": 0.6136}}
    (HERE / "evidence/ridge_oracle.json").write_text(json.dumps(evidence, indent=2) + "\n")
    log(status="ridge_oracle_complete", reading=reading)


if __name__ == "__main__":
    main()
