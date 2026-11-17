"""credit.py's question for G1, the denoiser's own anticipatory loss (RESULTS.md, credit). Frozen, no training.

G1 sees every frame and reads the belief through the denoiser's attention, the route that draws recovery in F5. On
each exact-history pair, at the due frame:
  loss  G1's flow term as trained on the physical tokens (periphery, lighting): mean over the 8 signal levels
        tau = j/8 of w(tau) * squared error, normalised by the percept's scalar count (periphery + glimpse + lighting),
        w = 0.9 tau + 0.1. The glimpse is held at the sampler's placeholder. The 8-frame cache is built as generation
        does (context_pass) from that member's percepts and its belief's memory tokens; the predicted frame reads
        the post-action memory. The two members share their noise draws.
  sep   the loss with the two members' belief streams swapped (memory tokens in the cache and the generating memory),
        each member keeping its own visual cache and target, minus the actual loss.
  rate  -grad(sep) . grad(loss) over belief parameters (Mamba side) and denoiser parameters (head side).
Rates in the objective's units divide by the group's harmoniser sigma squared: G1 for this term, G4 for credit.py's
forecast heads.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261111_p1f5.credit_g1 [checkpoint numbers...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings/dissection/credit"
CR = importlib.import_module("artifacts.experiments.20261111_p1f5.credit")
TJ, DS, NP = CR.TJ, CR.DS, CR.NP
DEV = DS.DEV
SETS = {"recovery": 63, "cooldown": 63, "energy": 66, "hunger": 64}


def g1_parts(model, cfg, row):
    """Differentiable per-combination G1 losses: {(v, b): loss for member v's target, cache visuals and member b's beliefs}."""
    i, e, prefix, frames, rewards, terms, q, truth = row
    p, rw, td, act = CR.belief_inputs(model, cfg, row)
    n = len(prefix[0])
    prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
    fix = p.fix
    query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=DEV)
    den, k = model.denoiser, model.denoiser.k_max
    with torch.no_grad(), TR.autocast(DEV, cfg):
        x = model.perception(torch.as_tensor(frames[:, q + 1:q + 2], device=DEV), query[None, None].expand(2, 1, -1))
    x = tuple(z.detach().float() for z in x.parts())                                  # (p, g, v) [2, 1, ...]
    denom = sum(z[0, 0].numel() for z in x)
    with TR.autocast(DEV, cfg):
        inputs = model.belief.tokens(model.belief.percept_embed(p), rw, td, act)
        y, _ = model.belief(inputs, model.belief.initial(2, DEV))
        bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
        out = {}
        for v in (0, 1):
            for b in (0, 1):
                rng = torch.Generator(device=DEV).manual_seed(14000 + i)
                cache = den.new_cache()
                for t in range(n + q - 7, n + q + 1):
                    cache = M.context_pass(den, tuple(z[v:v + 1, t:t + 1] for z in p.parts()), prev[v:v + 1, t:t + 1],
                                           model.belief.memory(bp[b:b + 1, t:t + 1]), fix[v:v + 1, t:t + 1], cache, cfg, rng)
                mem = model.belief.memory(y[b:b + 1, -1:])
                g0 = torch.zeros_like(x[1][v:v + 1])
                total = 0.
                for j in range(8):
                    tau = j / 8
                    ng = torch.Generator(device=DEV).manual_seed(17000 + 31 * i + j)
                    zp = (1 - tau) * torch.randn(x[0][v:v + 1].shape, device=DEV, generator=ng) + tau * x[0][v:v + 1]
                    zv = (1 - tau) * torch.randn(x[2][v:v + 1].shape, device=DEV, generator=ng) + tau * x[2][v:v + 1]
                    code = torch.full((1, 1), M.shortcut_code(j * (k // 8), k.bit_length() - 1, k), device=DEV)
                    o, _ = den.step((zp, g0, zv), code, act.new_zeros(1, 1), mem, query[None, None], cache)
                    w = 0.9 * tau + 0.1
                    total = total + w * (((o[0].float() - x[0][v:v + 1]) ** 2).sum() + ((o[2].float() - x[2][v:v + 1]) ** 2).sum()) / denom
                out[(v, b)] = total / 8
    return out


def one(model, cfg, row):
    L = g1_parts(model, cfg, row)
    loss = L[(0, 0)] + L[(1, 1)]
    sep = L[(0, 1)] + L[(1, 0)] - L[(0, 0)] - L[(1, 1)]
    belief = [p for p in model.belief.parameters() if p.requires_grad]
    head = [p for p in model.denoiser.parameters() if p.requires_grad]
    params = belief + head
    gl = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    gs = torch.autograd.grad(sep, params, allow_unused=True)
    kb = len(belief)
    dot = lambda a, c: sum(float((u.float() * z.float()).sum()) for u, z in zip(a, c) if u is not None and z is not None)
    norm = lambda a: sum(float(u.float().square().sum()) for u in a if u is not None) ** .5
    return {"loss": float(loss.detach()), "sep": float(sep.detach()), "rate_belief": -dot(gs[:kb], gl[:kb]),
            "rate_head": -dot(gs[kb:], gl[kb:]), "grad_loss_norm_belief": norm(gl[:kb]), "grad_sep_norm_belief": norm(gs[:kb])}


def checkpoint(k):
    path = OUT / f"g1_ckpt_{k:02d}.json"
    if path.exists():
        return
    ck = HERE / f"run/ckpt_{k:02d}.pt"
    model, cfg = NP.IF.load(ck.resolve())
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(True)
    sig = model.harmoniser.log_sigma.detach().exp().tolist()
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "sigma": sig}
    for name in SETS:
        per = [one(model, cfg, r) for r in TJ.pair_rows(name)]
        summ = {key: float(np.mean([x[key] for x in per])) for key in per[0]}
        summ["rate_belief_objective_units"] = summ["rate_belief"] / sig[0] ** 2
        summ["pairs_rate_belief_positive"] = int(sum(x["rate_belief"] > 0 for x in per))
        summ["pairs_sep_positive"] = int(sum(x["sep"] > 0 for x in per))
        res[f"{name}/g1"] = {"summary": summ, "pairs": per}
        print(k, name, "g1", json.dumps({kk: round(v, 8) if isinstance(v, float) else v for kk, v in summ.items()}), flush=True)
    del model
    torch.cuda.empty_cache()
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 3, 5, 10)):
        checkpoint(k)
