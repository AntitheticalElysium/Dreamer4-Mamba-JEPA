"""D11. Why imagination loses information, and why recursion compounds it -- from the D10 rollouts, nothing trained.

All errors are divided by V = total variance of the world's true states (sample 0, all depths), so worlds
in different state spaces are comparable ("fraction of natural variance"). Per world and depth k = 1..16:

  gen     E||g_k - s_k||^2 / V            imagined rollout vs the factual future (sample 0)
  tf      E||t_k - s_k||^2 / V            teacher-forced one-step at depth k (true frames fed back)
  persist E||s_0 - s_k||^2 / V            copying the root
  noise   E tr Var_keys(s_k) / V          aleatoric spread given the FULL root state (5 samples, unbiased)
  bias    E||g_k - mean_k||^2 / V - noise/5   error against the conditional mean: the only reducible part
  signal  E||mean_k - s_0||^2 / V - noise/5   predictable change
  captured 1 - bias / signal
  prop    E||g_k - t_k||^2 / V            divergence propagated from earlier imagined steps
  gain    rms(g_k - t_k) / rms(g_{k-1} - s_{k-1})   per-step amplification of the inherited error
  maha    E ||Sigma^-1/2 (x - mu)||^2 / D for x = g_k (true states give ~1): off-manifold if >> 1
  knn     median distance of g_k to its 10th nearest TRUE state of other roots / same for s_k
  spread  tr Cov_roots(g_k) / tr Cov_roots(s_k), and tr Cov_roots(mean_k) / tr Cov_roots(s_k)
  tail    share of the error g_k - s_k in the 150 lowest-variance eigen-directions of Sigma (they hold
          `tail_var` of the natural variance)
Facts (visible simulator state of the factual future): multi-output ridge probes, split by seed 70/30.
  on_true   fitted on true states (all depths), read on true states at depth k   -> what the state can hold
  transfer  the same probe read on imagined states at depth k                    -> is the fact where it should be
  on_gen    fitted on imagined states at depth k, read on imagined states        -> is the fact in there at all
  keys      agreement of the fact between samples 1-4 and sample 0 (determinism given the full state)
Groups: tiles near (player's 4 neighbours), interior, edge (view border: new content after a move);
zombie / cow presence (per-position AUC pooled); HUD (health, food, drink, energy: R^2); inventory R^2;
facing accuracy; sleeping AUC; light R^2.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")
H = 16


def auc(score, label):
    score, label = score.double().flatten(), label.bool().flatten()
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = torch.cat([pos, neg]).argsort().argsort().double() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def facts(vis):
    """[..., 1534] visible -> dict of targets."""
    v = vis.float()
    tiles = v[..., :1071].reshape(*v.shape[:-1], 7, 9, 17)
    mobs = v[..., 1071:1512].reshape(*v.shape[:-1], 7, 9, 7)
    return {"tiles": tiles, "zombie": (mobs[..., 0] > 0).float(), "cow": (mobs[..., 1] > 0).float(),
            "hud": v[..., 1512:1516], "facing": v[..., 1516:1520], "sleep": v[..., 1520:1521],
            "light": v[..., 1521:1522], "inventory": v[..., 1522:1534]}


NEAR = [(2, 4), (4, 4), (3, 3), (3, 5)]
EDGE = [(r, c) for r in range(7) for c in range(9) if r in (0, 6) or c in (0, 8)]
INTERIOR = [(r, c) for r in range(1, 6) for c in range(1, 8) if (r, c) not in NEAR and (r, c) != (3, 4)]


def targets_matrix(f):
    return torch.cat([f["tiles"].flatten(-3), f["zombie"].flatten(-2), f["cow"].flatten(-2), f["hud"], f["facing"],
                      f["sleep"], f["light"], f["inventory"]], -1)


def score_facts(pred, f, train_tiles_majority):
    """pred: [n, T] ridge outputs; f: fact dict for n rows."""
    o = 0
    tiles = pred[:, o:o + 1071].view(-1, 7, 9, 17); o += 1071
    zom = pred[:, o:o + 63].view(-1, 7, 9); o += 63
    cow = pred[:, o:o + 63].view(-1, 7, 9); o += 63
    hud = pred[:, o:o + 4]; o += 4
    facing = pred[:, o:o + 4]; o += 4
    sleep = pred[:, o]; o += 1
    light = pred[:, o]; o += 1
    inv = pred[:, o:o + 12]
    truth_tiles = f["tiles"].argmax(-1)
    correct = (tiles.argmax(-1) == truth_tiles).float()
    maj = (train_tiles_majority[None] == truth_tiles).float()
    r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum().clamp_min(1e-9))
    out = {}
    for name, cells in (("near", NEAR), ("interior", INTERIOR), ("edge", EDGE)):
        idx = torch.tensor(cells)
        out[f"tiles_{name}_acc"] = float(correct[:, idx[:, 0], idx[:, 1]].mean())
        out[f"tiles_{name}_majority"] = float(maj[:, idx[:, 0], idx[:, 1]].mean())
    out["zombie_auc"] = auc(zom, f["zombie"]); out["cow_auc"] = auc(cow, f["cow"])
    out["zombie_near_auc"] = auc(zom[:, [2, 4, 3, 3], [4, 4, 3, 5]], f["zombie"][:, [2, 4, 3, 3], [4, 4, 3, 5]])
    for i, name in enumerate(("health", "food", "drink", "energy")):
        out[f"{name}_r2"] = r2(hud[:, i], f["hud"][:, i])
    out["facing_acc"] = float((facing.argmax(-1) == f["facing"].argmax(-1)).float().mean())
    out["sleep_auc"] = auc(sleep, f["sleep"][:, 0])
    out["light_r2"] = r2(light, f["light"][:, 0])
    out["inventory_r2"] = r2(inv, f["inventory"])
    return out


def ridge_fit(x, y, xval, yval, lams=(1e-2, 1e-1, 1, 10, 100, 1000)):
    mu, sd = x.mean(0), x.std(0).clamp_min(1e-6)
    xs, xv = ((x - mu) / sd).double(), ((xval - mu) / sd).double()
    ym = y.mean(0)
    best = None
    for lam in lams:
        w = torch.linalg.solve(xs.T @ xs + lam * len(xs) * torch.eye(xs.shape[1], dtype=xs.dtype), xs.T @ (y - ym).double())
        err = float(((xv @ w + ym - yval) ** 2).mean())
        if best is None or err < best[0]:
            best = (err, lam, w)
    return lambda q: (((q - mu) / sd).double() @ best[2] + ym).float()


def main():
    meta = torch.load(DATA / "meta.pt")
    seeds = meta["seed"]
    useed = seeds.unique()
    gen_split = torch.Generator().manual_seed(0)
    train_seeds = useed[torch.randperm(len(useed), generator=gen_split)[: int(0.7 * len(useed))]]
    tr = torch.isin(seeds, train_seeds)
    te = ~tr
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()                      # factual still alive at depth k
    fut = facts(meta["future_visible"][:, 0])                                 # sample 0, [R,16,...]
    keysf = facts(meta["future_visible"])                                     # [R,5,16,...]
    Y = targets_matrix(fut)                                                   # [R,16,T]
    majority = fut["tiles"][tr].flatten(0, 1).sum(0).argmax(-1)               # [7,9]
    # determinism of each fact across keys (samples 1-4 vs sample 0), at each depth
    det = {}
    t0 = keysf["tiles"][:, 0].argmax(-1)
    for k in (1, 2, 4, 8, 16):
        same = lambda name: float(torch.stack([(keysf[name][:, j, k - 1] == keysf[name][:, 0, k - 1]).float().mean()
                                               for j in range(1, 5)]).mean())
        tiles_same = float(torch.stack([(keysf["tiles"][:, j, k - 1].argmax(-1) == t0[:, k - 1]).float().mean()
                                        for j in range(1, 5)]).mean())
        zpos = keysf["zombie"][:, 0, k - 1] > 0
        zsame = float(torch.stack([(keysf["zombie"][:, j, k - 1][zpos] > 0).float().mean() for j in range(1, 5)]).mean())
        det[k] = {"tiles_same": tiles_same, "hud_same": same("hud"), "zombie_at_same_cell": zsame}
    result = {"n_roots": len(seeds), "determinism_across_keys": det, "worlds": {}}
    tfull = torch.load(DATA / "T_fullspace.pt")
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        g, t, s, root = d["gen"], d["tf"], d["true"], d["root"]            # [R,16,D], [R,16,D], [R,5,16,D], [R,D]
        s0, mean = s[:, 0], s.mean(1)
        X = s0.flatten(0, 1)
        mu = X.mean(0)
        cov = torch.cov((X - mu).T.double())
        ev, evec = torch.linalg.eigh(cov)
        V = float(ev.sum())
        inv_sqrt = evec @ torch.diag(ev.clamp_min(ev.max() * 1e-6).rsqrt()) @ evec.T
        tail = evec[:, :max(0, len(ev) - 42)]                                 # lowest-variance directions
        tail_var = float(ev[:max(0, len(ev) - 42)].sum() / ev.sum())
        per = []
        bank = X[torch.isin(seeds.repeat_interleave(H), train_seeds)]
        for k in range(H):
            m = te & alive[:, k]
            gk, tk, sk, mk = g[m, k], t[m, k], s0[m, k], mean[m, k]
            noise = float(((s[m, :, k] - mk[:, None]) ** 2).sum(-1).sum(1).mean() / 4)
            e = lambda a, b: float(((a - b) ** 2).sum(-1).mean())
            prev = e(g[m, k - 1], s0[m, k - 1]) if k else float("nan")
            row = {"k": k + 1, "gen": e(gk, sk) / V, "tf": e(tk, sk) / V, "persist": e(root[m], sk) / V,
                   "noise": noise / V, "bias": (e(gk, mk) - noise / 5) / V, "tf_bias": (e(tk, mk) - noise / 5) / V,
                   "signal": (e(mk, root[m]) - noise / 5) / V, "prop": e(gk, tk) / V,
                   "gain": (e(gk, tk) / prev) ** 0.5 if k else float("nan")}
            row["captured"] = 1 - row["bias"] / row["signal"]
            wg = (gk.double() - mu) @ inv_sqrt
            ws = (sk.double() - mu) @ inv_sqrt
            row["maha_gen"] = float((wg ** 2).sum(-1).mean() / X.shape[1])
            row["maha_true"] = float((ws ** 2).sum(-1).mean() / X.shape[1])
            dk = lambda q: torch.cdist(q.float(), bank.float()).topk(10, largest=False).values[:, -1].median()
            row["knn_ratio"] = float(dk(gk) / dk(sk))
            cr = lambda q: float(torch.cov(q.T.double()).trace())
            row["spread_gen"] = cr(gk) / cr(sk)
            row["spread_condmean"] = cr(mk) / cr(sk)
            err = (gk - sk).double()
            row["error_tail_share"] = float(((err @ tail) ** 2).sum() / (err ** 2).sum()) if tail.shape[1] else float("nan")
            row["tail_var"] = tail_var
            row["step_gen"] = e(g[m, k], g[m, k - 1] if k else root[m]) / V
            row["step_true"] = e(s0[m, k], s0[m, k - 1] if k else root[m]) / V
            per.append(row)
        # facts
        Xtr = s0[tr].flatten(0, 1); Ytr = Y[tr].flatten(0, 1)
        vsplit = torch.isin(seeds[tr].repeat_interleave(H), train_seeds[: len(train_seeds) // 5])
        probe = ridge_fit(Xtr[~vsplit], Ytr[~vsplit], Xtr[vsplit], Ytr[vsplit])
        fact_rows = {}
        for k in (1, 2, 4, 8, 16):
            m = te & alive[:, k - 1]
            fk = {n: v[m, k - 1] for n, v in fut.items()}
            row = {"on_true": score_facts(probe(s0[m, k - 1]), fk, majority),
                   "transfer": score_facts(probe(g[m, k - 1]), fk, majority),
                   "persist": score_facts(probe(root[m]), fk, majority)}
            mt = tr & alive[:, k - 1]
            half = torch.isin(seeds[mt], train_seeds[: len(train_seeds) // 5])
            gp = ridge_fit(g[mt, k - 1][~half], Y[mt, k - 1][~half], g[mt, k - 1][half], Y[mt, k - 1][half])
            row["on_gen"] = score_facts(gp(g[m, k - 1]), fk, majority)
            fact_rows[k] = row
        result["worlds"][w] = {"V": V, "depth": per, "facts": fact_rows}
        print(w, json.dumps({f"k{r['k']}": {kk: round(r[kk], 3) for kk in ("gen", "tf", "persist", "noise", "bias", "captured",
                                                                          "prop", "gain", "maha_gen", "knn_ratio", "spread_gen")}
                             for r in per if r["k"] in (1, 2, 4, 8, 16)}), flush=True)
    (HERE / "compound.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
