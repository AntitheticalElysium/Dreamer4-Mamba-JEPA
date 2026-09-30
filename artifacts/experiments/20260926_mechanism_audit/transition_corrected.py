"""D5. What the imagined successor gets wrong, given the state and head are fine (U, W on true successors: 0.98-0.99).

On a zombie root the true successor decides the outcome through two facts:
  moved     did the move succeed (target tile walkable, no mob)? -- visible in the true successor as a scroll
  damaged   did health drop (key-1 outcome, `health_delta` < 0)? -- visible as the HUD health tile
For each world (U, W, Z, H2-z) and each fact: is it in the imagined successor?
  scroll    ||g(a) - root|| vs ||true(a) - root||: AUC for moved-vs-blocked among move actions (distance only)
  probes    ridge probes (+-1 labels, standardised inputs, lambda chosen on 55k vs 56k), fitted on 55k+56k,
            AUC on 57k+58k, from: the root state (per move direction, so the input's own information is the
            reference), the Mamba output h(a), the generated latent g(a), the true successor state true(a).
            For `moved` also from the root's patch tokens (81x192, kernel ridge), the encoder's full output.
Rows: opportunity roots of 55k-58k (all of them for `moved`; zombie roots for `damaged`).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER)); sys.path.insert(0, str(ROOT / "artifacts/experiments/20260926_diagnosis")); sys.path.insert(0, str(HERE))
import interface as I  # noqa: E402
from choices import move_table  # noqa: E402
from decision import unpack, C, MOVES, LAVA
from frozen_ladder import strata  # noqa: E402

DUMP = ROOT / "artifacts/eda/diagnosis_dump_v1"
BLOCKS = {"55k": "observe_fresh_v6", "56k": "observe_fresh_v7", "57k": "observe_fresh_v8", "58k": "observe_fresh_v9"}
ARMS = {"H2": "Z", "Z": "Z", "U": "U", "W": "W"}
N = 17


def auc(score, label):
    score, label = score.double(), label.bool()
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return None
    ranks = torch.cat([pos, neg]).argsort().argsort().double() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def ridge_auc(xtr, ytr, xte, yte, dual=False):
    mu, sd = xtr.mean(0), xtr.std(0).clamp_min(1e-6)
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    t = ytr.double() * 2 - 1
    half = len(xtr) // 2
    best = None
    for lam in (1e-1, 1e0, 1e1, 1e2, 1e3, 1e4):
        w = solve(xtr[:half], t[:half], lam, dual)
        a = auc(xte_score(xtr[half:], w, xtr[:half], dual), ytr[half:])
        if a is not None and (best is None or a > best[0]):
            best = (a, lam)
    w = solve(xtr, t, best[1], dual)
    return auc(xte_score(xte, w, xtr, dual), yte)


def solve(x, t, lam, dual):
    x = x.double()
    if dual:
        k = x @ x.T
        return torch.linalg.solve(k + lam * torch.eye(len(x), dtype=k.dtype), t)
    return torch.linalg.solve(x.T @ x + lam * torch.eye(x.shape[1], dtype=x.dtype), x.T @ t)


def xte_score(x, w, xtr, dual):
    return (x.double() @ xtr.double().T) @ w if dual else x.double() @ w


@torch.no_grad()
def encode_block(store, keep_seeds, encoder, device):
    rows = [r for f in sorted((ROOT / "artifacts/eda" / store).glob("seed-*.pt")) for r in torch.load(f, weights_only=False)]
    rows = [r for r in rows if r["p_death1"].max() > r["p_death1"].min()]
    assert len(rows) == len(keep_seeds)
    roots = torch.stack([r["frames"][-1] for r in rows])
    succ = torch.stack([r["successors"] for r in rows]).flatten(0, 1)
    z, g = I.encode(encoder, roots[:, None], device)
    tokens = []
    for i in range(0, len(roots), 64):
        _, _, t, _, _ = encoder._hidden(roots[i:i + 64, None].to(device))
        tokens.append(t.float().cpu())
    sz, sg = I.encode(encoder, succ[:, None], device)
    return {"z": z[:, 0], "grid": g[:, 0], "tokens": torch.cat(tokens).flatten(1),
            "sz": sz[:, 0].view(len(rows), N, -1), "sgrid": sg[:, 0].view(len(rows), N, -1)}


def main():
    sys.path.insert(0, str(LADDER))
    from zwhite import zwhitened_pool
    pool, _ = zwhitened_pool()
    encoder, _ = I.load_bridge()
    device = torch.device("cuda")
    data = {}
    for b, store in BLOCKS.items():
        meta = torch.load(DUMP / f"{b}_meta.pt")
        enc = encode_block(store, meta["seed"], encoder, device)
        cat, adj = move_table(meta["visible"])
        lava_target = torch.zeros_like(cat,dtype=torch.bool)
        for row,v in enumerate(meta["visible"]):
            tiles,_,_,_=unpack(v)
            for a,(dr,dc) in MOVES.items():
                lava_target[row,a] = int(tiles[C[0]+dr,C[1]+dc]) == LAVA
        data[b] = {"lava_target": lava_target, "meta": meta, "enc": enc, "cat": cat, "zombie": strata(meta["visible"])["zombie_adjacent"],
                   "dump": {w: torch.load(DUMP / f"{b}_{w}.pt") for w in ARMS}}
        print("encoded", b, flush=True)
    fit, test = ("55k", "56k"), ("57k", "58k")
    result = {}
    for w, arm in ARMS.items():
        res = {}
        # states in the arm's coordinates
        for b in BLOCKS:
            e = data[b]["enc"]
            data[b][f"root_{w}"] = I.state_of(arm, pool["pca"], e["z"][:, None], e["grid"][:, None].float())[:, 0]
            data[b][f"true_{w}"] = I.state_of(arm, pool["pca"], e["sz"], e["sgrid"].float())
        # --- moved, among move actions (lava excluded: cat 0 = moved, 1 = blocked)
        def rows_moved(bs, a):
            xs = {k: [] for k in ("root", "h", "g", "true", "tokens")}
            ys = []
            for b in bs:
                d, cat = data[b], data[b]["cat"]
                sel = (cat[:, a] <= 1) & ~d["lava_target"][:,a]
                ys.append((cat[sel, a] == 0).float())
                xs["root"].append(d[f"root_{w}"][sel])
                xs["h"].append(d["dump"][w]["h"][sel, a])
                xs["g"].append(d["dump"][w]["gen"][sel, a])
                xs["true"].append(d[f"true_{w}"][sel, a] - d[f"root_{w}"][sel])
                xs["tokens"].append(d["enc"]["tokens"][sel])
            return {k: torch.cat(v) for k, v in xs.items()}, torch.cat(ys)
        moved = {k: [] for k in ("root", "h", "g", "true", "tokens", "scroll_generated", "scroll_true")}
        for a in (1, 2, 3, 4):
            xtr, ytr = rows_moved(fit, a)
            xte, yte = rows_moved(test, a)
            for k in ("root", "h", "g", "true"):
                moved[k].append(ridge_auc(xtr[k], ytr, xte[k], yte))
            if w == "U":
                moved["tokens"].append(ridge_auc(xtr["tokens"], ytr, xte["tokens"], yte, dual=True))
            gen_dist, true_dist = [], []
            for b in test:
                d, cat = data[b], data[b]["cat"]
                sel = (cat[:, a] <= 1) & ~d["lava_target"][:,a]
                gen_dist.append((d["dump"][w]["gen"][sel, a] - d[f"root_{w}"][sel]).norm(dim=1))
                true_dist.append((d[f"true_{w}"][sel, a] - d[f"root_{w}"][sel]).norm(dim=1))
            moved["scroll_generated"].append(auc(torch.cat(gen_dist), yte))
            moved["scroll_true"].append(auc(torch.cat(true_dist), yte))
        res["moved_auc_mean_over_4_directions"] = {k: float(np.mean(v)) for k, v in moved.items() if v}
        res["moved_blocked_share"] = float(1 - torch.cat([rows_moved(test, a)[1] for a in (1, 2, 3, 4)]).mean())
        # --- damaged (health dropped), zombie roots, all 17 actions
        def rows_damage(bs):
            xs = {k: [] for k in ("h", "g", "true", "head_generated", "head_true")}
            ys = []
            for b in bs:
                d = data[b]
                zr = d["zombie"]
                ys.append((d["meta"]["health_delta"][zr] < 0).float().flatten())
                xs["h"].append(d["dump"][w]["h"][zr].flatten(0, 1))
                xs["g"].append(d["dump"][w]["gen"][zr].flatten(0, 1))
                xs["true"].append((d[f"true_{w}"][zr] - d[f"root_{w}"][zr, None]).flatten(0, 1))
                xs["head_generated"].append(d["dump"][w]["p_dead"][zr].flatten())
                xs["head_true"].append(d["dump"][w]["p_dead_real"][zr].flatten())
            return {k: torch.cat(v) for k, v in xs.items()}, torch.cat(ys)
        xtr, ytr = rows_damage(fit)
        xte, yte = rows_damage(test)
        res["damaged_auc"] = {k: ridge_auc(xtr[k], ytr, xte[k], yte) for k in ("h", "g", "true")}
        res["damaged_auc"]["own_head_generated"] = auc(xte["head_generated"], yte)
        res["damaged_auc"]["own_head_true"] = auc(xte["head_true"], yte)
        res["damaged_rate"] = float(yte.mean())
        result[w] = res
        print(w, json.dumps(res), flush=True)
    (Path(__file__).with_suffix(".json")).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
