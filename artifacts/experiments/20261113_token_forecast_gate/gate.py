"""Option A gate (PLAN.md): an all-cell periphery-token forecast from the belief, plain vs fixed-precision MSE.
Frozen F5 checkpoints; fresh heads fitted on frozen beliefs; no world training.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_token_forecast_gate.gate [checkpoints...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / os.environ.get("GATE_OUT", "readings")
F5 = HERE.parent / "20261111_p1f5"
CR = importlib.import_module("artifacts.experiments.20261111_p1f5.credit")
TJ, NP, DEV = CR.TJ, CR.NP, CR.DEV
CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
STEPS, BATCH = int(os.environ.get("GATE_STEPS", 20000)), 512
SETS = {"energy": 66, "hunger": 64, "recovery": 63, "cooldown": 63}
FOILS = (1, 3, 6)


class TokenHead(nn.Module):
    """readout_fit.Concat's form, emitting the 16-number periphery token of every cell."""
    def __init__(self, W):
        super().__init__()
        self.cell = nn.Embedding(M.N, 32)
        self.belief = nn.Linear(W, 128)
        self.net = nn.Sequential(M.mlp([32 + 128, 256, 256]), nn.GELU(), nn.Linear(256, 16))

    def forward(self, bp):                                                  # [B, W] -> [B, 81, 16]
        c = self.cell.weight.expand(bp.shape[0], -1, -1)
        return self.net(torch.cat([c, self.belief(bp).unsqueeze(1).expand(-1, M.N, -1)], -1))


def boot_mean(x, n=2000, seed=0):
    x = np.asarray(x, float)
    ix = np.random.default_rng(seed).integers(len(x), size=(n, len(x)))
    return [float(x.mean()), *np.quantile(x[ix].mean(1), [.025, .975]).tolist()]


def reference_cosines(k):
    path = F5 / f"readings/dissection/credit/ckpt_{k:02d}.json"
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    out = {}
    for key, v in d.items():
        if isinstance(v, dict) and "pairs" in v:
            cs = [p["rate_belief"] / max(p["grad_sep_norm_belief"] * p["grad_loss_norm_belief"], 1e-30) for p in v["pairs"]]
            out[key] = {"cosine_mean_ci": boot_mean(cs), "positive": int(sum(c > 0 for c in cs))}
    return out


def checkpoint(k):
    path = OUT / f"ckpt_{k:02d}.json"
    if path.exists():
        return
    torch.manual_seed(0)
    model, cfg = NP.IF.load((F5 / f"run/ckpt_{k:02d}.pt").resolve())
    model.eval()
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    tgt, prev, bps, vs, ep_of = [], [], [], [], []
    for e, r in enumerate(enc):
        tgt.append(r["p"][1:].half()); prev.append(r["p"][:-1].half()); bps.append(r["bp"][1:].float())
        vs.append(r["v"][1:].float()); ep_of.append(torch.full((len(r["p"]) - 1,), e))
    tgt, prev, bp, v_all, ep_of = map(torch.cat, (tgt, prev, bps, vs, ep_of))
    keep_enc = enc if k == 10 else None
    del enc
    held = ep_of % 5 == 4
    tr_ix, te_ix = torch.nonzero(~held)[:, 0], torch.nonzero(held)[:, 0]
    with torch.no_grad():
        s2 = torch.zeros(M.N, 16, dtype=torch.float64)
        for a in range(0, len(tr_ix), 65536):
            ix = tr_ix[a:a + 65536]
            s2 += ((tgt[ix].double() - prev[ix].double()) ** 2).sum(0)
        s2 = (s2 / len(tr_ix)).float()
        floor = 0.01 * s2.median()
        w = 1 / s2.clamp(min=floor)
        w = w / w.mean()
    W_ = {"plain": torch.ones(M.N, 16), "precision": w}
    res = {"checkpoint": k, "sha256": TR.file_sha256(F5 / f"run/ckpt_{k:02d}.pt"), "floor": float(floor),
           "weight_by_cell_mean": w.mean(-1).tolist(), "weight_hud_cells": {c: float(w[c].mean()) for c in (63, 64, 65, 66)}}
    heads = {}
    for cur, wc in W_.items():
        wd = wc.to(DEV)
        net = TokenHead(bp.shape[-1]).to(DEV)
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(1)
        for s in range(STEPS):
            ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
            loss = (wd * (net(bp[ix].to(DEV)) - tgt[ix].to(DEV).float()) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 5000 == 0:
                print(k, cur, s, float(loss), flush=True)
        net.eval()
        heads[cur] = net
        with torch.no_grad():
            lh, lc, ls = [], [], []
            shuf = te_ix[torch.randperm(len(te_ix), generator=torch.Generator().manual_seed(5))]
            for a in range(0, len(te_ix), 8192):
                ix, jx = te_ix[a:a + 8192], shuf[a:a + 8192]
                x = tgt[ix].to(DEV).float()
                lh.append(float((wd * (net(bp[ix].to(DEV)) - x) ** 2).mean()) * len(ix))
                ls.append(float((wd * (net(bp[jx].to(DEV)) - x) ** 2).mean()) * len(ix))      # belief-blind control
                lc.append(float((wd * (prev[ix].to(DEV).float() - x) ** 2).mean()) * len(ix))
            res[f"heldout/{cur}"] = {"head": sum(lh) / len(te_ix), "shuffled_belief": sum(ls) / len(te_ix),
                                     "copy_previous": sum(lc) / len(te_ix)}
        print(k, cur, json.dumps(res[f"heldout/{cur}"]), flush=True)
    for prm in model.parameters():
        prm.requires_grad_(True)
    belief_params = [x for x in model.belief.parameters() if x.requires_grad]
    for name, slot in SETS.items():
        rows = TJ.pair_rows(name)
        per = {cur: [] for cur in W_}
        for row in rows:
            i, e, prefix, frames, rewards, terms, q, truth = row
            query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=DEV)
            with torch.no_grad(), TR.autocast(DEV, cfg):
                nxt = torch.as_tensor(frames[:, q + 1:q + 2], device=DEV)
                x = model.perception(nxt, query[None, None].expand(2, 1, -1)).p[:, 0].float()        # [2, 81, 16]
                xb = {a: model.perception(torch.clamp(nxt[1:].int() + a, max=255).to(torch.uint8), query[None, None]).p[0, 0].float()
                      for a in FOILS}
            p, rw, td, act = CR.belief_inputs(model, cfg, row)
            for cur, wc in W_.items():
                wd = wc.to(DEV)
                Lw = lambda u, z: (wd * (u - z) ** 2).mean()
                entry = {"consequence": float(Lw(x[0], x[1])), "foil": {str(a): float(Lw(xb[a], x[1])) for a in FOILS}}
                with TR.autocast(DEV, cfg):
                    y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), rw, td, act), model.belief.initial(2, DEV))
                f = heads[cur](y[:, -1].float())
                loss = Lw(f[0], x[0]) + Lw(f[1], x[1])
                sep = Lw(f[1], x[0]) + Lw(f[0], x[1]) - loss
                gl = torch.autograd.grad(loss, belief_params, retain_graph=True, allow_unused=True)
                gs = torch.autograd.grad(sep, belief_params, allow_unused=True)
                dot = sum(float((u.float() * z.float()).sum()) for u, z in zip(gs, gl) if u is not None and z is not None)
                nl = sum(float(u.float().square().sum()) for u in gl if u is not None) ** .5
                ns = sum(float(u.float().square().sum()) for u in gs if u is not None) ** .5
                dx = (x[0] - x[1])[slot]
                entry.update(rate=-dot, cosine=-dot / max(nl * ns, 1e-30), grad_loss_norm=nl, grad_sep_norm=ns,
                             sep=float(sep.detach()),
                             kappa=float(((f[0] - f[1]).detach()[slot] * dx).sum() / dx.square().sum().clamp(min=1e-12)),
                             share_in_digit_cell=float((wd[slot] * dx ** 2).sum() / (wd * (x[0] - x[1]) ** 2).sum().clamp(min=1e-12)))
                per[cur].append(entry)
        for cur in W_:
            P = per[cur]
            res[f"{name}/{cur}"] = {
                "pairs": len(P),
                "consequence_median": float(np.median([e["consequence"] for e in P])),
                "consequence_over_floor_median": float(np.median([e["consequence"] for e in P]) / res[f"heldout/{cur}"]["head"]),
                "foil_beats_consequence": {a: int(sum(e["foil"][a] >= e["consequence"] for e in P)) for a in map(str, FOILS)},
                "cosine_mean_ci": boot_mean([e["cosine"] for e in P]), "cosine_positive": int(sum(e["cosine"] > 0 for e in P)),
                "rate_mean": float(np.mean([e["rate"] for e in P])), "kappa_mean": float(np.mean([e["kappa"] for e in P])),
                "digit_cell_share_median": float(np.median([e["share_in_digit_cell"] for e in P])),
                "per_pair": P}
            print(k, name, cur, json.dumps({kk: vv for kk, vv in res[f"{name}/{cur}"].items() if kk != "per_pair"}), flush=True)
    res["reward_and_tsp_reference_cosines"] = reference_cosines(k)
    if keep_enc is not None:
        res["reactive_tsp_belief_use"] = tsp_belief_use(model, cfg, keep_enc, L, held)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def tsp_belief_use(model, cfg, enc, L, held):
    """Reactive TSP's held-out error on uniform windows with the true bp vs a bp from a random other held-out frame."""
    eps = S.episodes(L)
    g = torch.Generator().manual_seed(4)
    C49 = torch.cartesian_prod(torch.arange(1, 8), torch.arange(1, 8))
    rows = []
    for e in sorted({int(x) for x in torch.nonzero(torch.as_tensor([i % 5 == 4 for i in range(len(enc))]))[:, 0]}):
        fr = torch.as_tensor(D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy()[1:])
        r = enc[e]
        rows.append((fr, r["p"][1:], r["v"][1:], r["bp"][1:]))
    fr, pt, vt, bp = (torch.cat(z) for z in zip(*rows))
    perm = torch.randperm(len(bp), generator=g)
    err = {"true_belief": 0., "shuffled_belief": 0.}
    for a in range(0, len(bp), 4096):
        ix = torch.arange(a, min(a + 4096, len(bp)))
        c = C49[torch.randint(49, (len(ix),), generator=g)].to(DEV)
        cells = M.window(c)
        y = M.cell_pixels(M.crop(fr[ix].to(DEV), c)).float() / 127.5 - 1
        for tag, b in (("true_belief", bp[ix]), ("shuffled_belief", bp[perm[ix]])):
            mu = model.pixels.mean(model.tsp.predict(b.to(DEV).float(), pt[ix].to(DEV).float(), vt[ix].to(DEV).float(), cells)).float()
            err[tag] += float((mu - y).square().mean()) * len(ix)
    return {k: v / len(bp) for k, v in err.items()} | {"frames": int(len(bp))}


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 10)):
        checkpoint(k)
