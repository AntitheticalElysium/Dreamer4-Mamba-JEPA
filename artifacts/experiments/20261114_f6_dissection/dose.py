"""F6 dissection, readings 1 and 2 (PLAN.md): the token forecast's request against the reward forecast's, as trained.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.dose pairs [checkpoints...]
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.dose batch [checkpoints...]
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
OUT = HERE / "readings"
F6 = importlib.import_module("artifacts.experiments.20261113_p1f6.train")
G3 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate3")
G, G2, CR, TJ, NP, DEV = G3.G, G3.G2, G3.CR, G3.TJ, G3.NP, G3.DEV
mse = G3.mse


def load(k):
    ck = (F6.HERE / f"run/ckpt_{k:02d}.pt").resolve()
    model, cfg = NP.IF.load(ck)
    for prm in model.parameters():
        prm.requires_grad_(True)
    return ck, model, cfg, [x for x in model.belief.parameters()]


def pairs(k):
    path = OUT / f"pairs_{k:02d}.json"
    if path.exists():
        return
    ck, model, cfg, params = load(k)
    grad = lambda f: G2.flat(torch.autograd.grad(f, params, retain_graph=True, allow_unused=True), params)
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "sigma_G4": float(model.harmoniser.log_sigma.exp()[3])}
    for name, slot in G2.SETS.items():
        per = []
        for row in TJ.pair_rows(name):
            p, rw, td, act = CR.belief_inputs(model, cfg, row)
            with TR.autocast(DEV, cfg):
                y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), rw, td, act), model.belief.initial(2, DEV))
                b = y[:, -1:]
                loss_r, sep_r = CR.head_terms(model, b, "reward", row, slot)
            cur = p.p[:, -1].float().detach()
            out = cur + model.token_forecast(b[:, 0].float())
            x, _ = G2.due_targets(model, row)
            loss_t = mse(out[0], x[0]) + mse(out[1], x[1])
            sep_t = mse(out[1], x[0]) + mse(out[0], x[1]) - loss_t
            e = {"token_loss": float(loss_t), "token_sep": float(sep_t), "reward_loss": float(loss_r), "reward_sep": float(sep_r)}
            gl, gs = grad(loss_t), grad(sep_t)
            e.update(token_gsep=float(gs.norm()), token_gloss=float(gl.norm()), token_rate=float(-(gs @ gl)))
            dx = (x[0] - x[1])[slot]
            e["token_kappa"] = float(((out[0] - out[1])[slot] * dx).sum() / dx.square().sum().clamp(min=1e-12))
            e["consequence_price"] = float(mse(x[0], x[1]))                # the due / not-due target difference, as the term prices it
            if float(sep_r.detach().abs()) > 0 or name in ("recovery", "cooldown"):
                gl, gs = grad(loss_r.float()), grad(sep_r.float())
                e.update(reward_gsep=float(gs.norm()), reward_gloss=float(gl.norm()), reward_rate=float(-(gs @ gl)))
            per.append(e)
        summ = {key: float(np.mean([q[key] for q in per])) for key in per[0]}
        if "reward_gsep" in summ:
            summ["gsep_ratio_token_over_reward_median"] = float(np.median([q["token_gsep"] / max(q["reward_gsep"], 1e-30) for q in per]))
            summ["rate_ratio_token_over_reward"] = summ["token_rate"] / summ["reward_rate"] if summ["reward_rate"] else None
        res[name] = {"summary": summ, "pairs": per}
        print(k, name, json.dumps({kk: (float(f"{v:.4g}") if v is not None else None) for kk, v in summ.items()}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


def batch(k):
    path = OUT / f"batch_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, params = load(k)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=F6.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    chunk = streams.next(64)
    carry = tuple(M.MambaCarry(c.conv.to(DEV), c.ssm.to(DEV)) for c in ck["carry"])
    cache = ck["cache"].to(DEV)
    update = ck["state"]["update"]
    boot = TR.bootstrap_rows(cfg, F6.WORLD_SEED, update)
    sigma = dict(zip(model.harmoniser.groups, model.harmoniser.log_sigma.detach().exp().tolist()))
    group = {"g1": "G1", "tsp": "G2", "rec": "G2", "reward": "G3", "cont": "G3", "token_forecast": "G4",
             "reward_forecast": "G4", "cont_forecast": "G4"}
    torch.manual_seed(0)
    per = []
    for r in range(8):
        rows = np.array([r])
        terms, _, _ = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg,
                                   boot[rows], update, None)
        g = {n: G2.flat(torch.autograd.grad(terms[n], params, retain_graph=True, allow_unused=True), params) for n in group}
        total = sum(g[n] / sigma[group[n]] for n in group)
        cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp(min=1e-30))
        per.append({"loss": {n: float(terms[n]) for n in group}, "norm": {n: float(g[n].norm()) for n in group},
                    "weighted_norm": {n: float(g[n].norm() / sigma[group[n]]) for n in group},
                    "cos_to_total": {n: cos(g[n], total) for n in group}, "total_norm": float(total.norm()),
                    "cos_token_reward_forecast": cos(g["token_forecast"], g["reward_forecast"]),
                    "cos_token_g1": cos(g["token_forecast"], g["g1"])})
        del terms, g, total
        torch.cuda.empty_cache()
    med = lambda f: {n: float(np.median([f(q)[n] for q in per])) for n in group}
    res = {"checkpoint": k, "update": update, "sigma": sigma, "streams": 8, "new_frames_per_stream": 64,
           "loss_median": med(lambda q: q["loss"]), "norm_median": med(lambda q: q["norm"]),
           "weighted_norm_median": med(lambda q: q["weighted_norm"]), "cos_to_total_median": med(lambda q: q["cos_to_total"]),
           "total_norm_median": float(np.median([q["total_norm"] for q in per])),
           "cos_token_reward_forecast_median": float(np.median([q["cos_token_reward_forecast"] for q in per])),
           "cos_token_g1_median": float(np.median([q["cos_token_g1"] for q in per])), "per_stream": per}
    print(k, json.dumps({kk: ({n: float(f"{v:.3g}") for n, v in vv.items()} if isinstance(vv, dict) else vv)
                         for kk, vv in res.items() if kk != "per_stream"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    mode, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
    for k in ks or (range(1, 11) if mode == "pairs" else (1, 2, 5, 10)):
        {"pairs": pairs, "batch": batch}[mode](k)
