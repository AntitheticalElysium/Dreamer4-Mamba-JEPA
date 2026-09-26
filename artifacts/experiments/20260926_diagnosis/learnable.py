"""D15. Is the one-step change LEARNABLE from each world's own state? Closed-form models, nothing iterative.

Diagnostic futures, one step, all 17 actions, key-0 successor; split by seed 70/30 (as D11/D12).
  ridge      per action: s' - s = W_a (s - mu) + b_a, ridge on the state (z, u, w: 192; T: its PCA-1024);
             lambda chosen on a held-out fifth of the training seeds
  local      T only, in full token space: each tile's next token = s_p + A_a [s_p, s_up, s_down, s_left,
             s_right] + b_a, one 960 -> 192 map per action shared by all 81 positions (zero-padded) -- the
             inductive bias of a one-tile translation
Captured = 1 - err / err(copy the root), by transition class (onestep.classify), on test seeds, against
each world's own one-step prediction on the same rows. T errors are compared in its PCA-1024 space.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
LADDER = ROOT / "artifacts/experiments/20260921_readout_ladder"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(LADDER)); sys.path.insert(0, str(HERE))
import interface as I  # noqa: E402
from onestep import CLASSES, classify  # noqa: E402
from rollouts import encode, load_roots, states  # noqa: E402

DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")
N = 17


def ridge(x, y, lam):
    x1 = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1).double()
    return torch.linalg.solve(x1.T @ x1 + lam * len(x1) * torch.eye(x1.shape[1], dtype=x1.dtype), x1.T @ y.double())


def apply(x, w):
    return (torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype)], 1).double() @ w).float()


def neighbours(tok):
    """[n, 81, 192] -> [n, 81, 960]: self, up, down, left, right on the 9x9 grid, zero padded."""
    g = tok.view(-1, 9, 9, tok.shape[-1]).permute(0, 3, 1, 2)
    pad = F.pad(g, (1, 1, 1, 1))
    parts = [g, pad[:, :, :-2, 1:-1], pad[:, :, 2:, 1:-1], pad[:, :, 1:-1, :-2], pad[:, :, 1:-1, 2:]]
    return torch.cat(parts, 1).permute(0, 2, 3, 1).reshape(len(tok), 81, -1)


def main():
    device = torch.device("cuda")
    from whiten import whitened_pool
    pool, _ = whitened_pool()
    encoder, _ = I.load_bridge()
    meta = torch.load(DATA / "meta.pt")
    cls, _ = classify(meta)
    seeds = meta["seed"]
    useed = seeds.unique()
    train_seeds = useed[torch.randperm(len(useed), generator=torch.Generator().manual_seed(0))[: int(0.7 * len(useed))]]
    tr = torch.isin(seeds, train_seeds)
    val = torch.isin(seeds, train_seeds[: len(train_seeds) // 5])
    fit = tr & ~val
    roots = load_roots()
    R = len(roots)
    root_enc = encode(encoder, torch.stack([r["context"][-1] for r in roots]), device)
    succ_enc = encode(encoder, torch.stack([r["onestep_frames"][0] for r in roots]).flatten(0, 1), device)
    rs = states(pool, *[x.cpu() for x in root_enc])
    ss = states(pool, *[x.cpu() for x in succ_enc])
    tf = torch.load(DATA / "T_fullspace.pt")
    proj = lambda x: (x.flatten(-2).float() - tf["t_mean"]) @ tf["t_basis"]
    result = {}
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        V = float(torch.cov(d["true"][:, 0].flatten(0, 1).T.double()).trace())
        if w == "T":
            s0, s1 = proj(rs["T"]), proj(ss["T"]).view(R, N, -1)
        else:
            s0, s1 = rs[w], ss[w].view(R, N, -1)
        pred = torch.empty_like(s1)
        for a in range(N):
            best = None
            for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1):
                wa = ridge(s0[fit], s1[fit, a] - s0[fit], lam)
                e = float(((apply(s0[val], wa) + s0[val] - s1[val, a]) ** 2).sum(-1).mean())
                if best is None or e < best[0]:
                    best = (e, lam)
            wa = ridge(s0[tr], s1[tr, a] - s0[tr], best[1])
            pred[:, a] = apply(s0, wa) + s0
        preds = {"ridge": pred, "world": d["one"]}
        if w == "T":
            full0, full1 = rs["T"], ss["T"].view(R, N, 81, -1)
            local = torch.empty(R, N, 81, full0.shape[-1])
            nb = neighbours(full0)
            for a in range(N):
                x = nb[tr].flatten(0, 1); y = (full1[tr, a] - full0[tr]).flatten(0, 1)
                idx = torch.randperm(len(x), generator=torch.Generator().manual_seed(a))[:200_000]
                wa = ridge(x[idx], y[idx], 1e-3)
                local[:, a] = (apply(nb.flatten(0, 1), wa).view(R, 81, -1) + full0)
            preds["local"] = proj(local)
        err = {k: ((p - s1) ** 2).sum(-1) for k, p in preds.items()}
        err["copy"] = ((s0[:, None] - s1) ** 2).sum(-1)
        te = ~tr
        row = {}
        for i, c in enumerate(CLASSES + ("all",)):
            m = te[:, None] & ((cls == i) if c != "all" else torch.ones_like(cls, dtype=torch.bool))
            base = float(err["copy"][m].mean())
            row[c] = {k: round(1 - float(v[m].mean()) / base, 3) for k, v in err.items() if k != "copy"}
            row[c]["copy_error_over_V"] = round(base / V, 4)
        result[w] = row
        print(w, json.dumps(row), flush=True)
    (HERE / "learnable.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
