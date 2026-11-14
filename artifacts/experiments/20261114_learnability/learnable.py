"""Learnability of the pixel-only clocks under one pressure at a time (PLAN.md).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_learnability.learnable ARM
"""
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from saccade import data as D, model as M, train as TR
from saccade.eval import gates as GT, s1 as S

HERE = Path(__file__).resolve().parent
DZ = importlib.import_module("artifacts.experiments.20261114_f6_dissection.dose")
TH = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate").TokenHead
DEV = DZ.DEV
EVENTS = ("food_down", "drink_down", "energy_down_awake", "energy_up_asleep", "recovery")
W, B, UPDATES, EVALS = 128, 32, 3000, (750, 1500, 3000)
DIGITS = [63, 64, 65, 66]


def events(L):
    """Per state t (episode-local offsets as L's s_ arrays): (target [N, 5], eligible [N, 5]) for transition t-1 -> t."""
    body, sleep = L["s_body"].astype(int), L["s_sleep"].astype(bool)
    so = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pb, ps = np.roll(body, 1, 0), np.roll(sleep, 1, 0)
    first = np.zeros(len(body), bool); first[so[:-1]] = True
    y = np.stack([body[:, 1] < pb[:, 1], body[:, 2] < pb[:, 2], body[:, 3] < pb[:, 3], body[:, 3] > pb[:, 3],
                  body[:, 0] > pb[:, 0]], 1)
    ok = np.stack([~ps & (pb[:, 1] > 0), ~ps & (pb[:, 2] > 0), ~ps & (pb[:, 3] > 0), ps & (pb[:, 3] < 9),
                   ~ps & (pb[:, 0] < 9)], 1) & ~first[:, None]
    return y & ok, ok


class Events(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(w, 256), nn.GELU(), nn.Linear(256, len(EVENTS)))

    def forward(self, b):
        return self.net(b)


def run_belief(belief, P, chunk):
    emb = belief.percept_embed(P)
    x = belief.tokens(emb, chunk["reward"], chunk["terminated"], chunk["action"])
    y, _ = belief(x, belief.initial(x.shape[0], DEV))
    return y[:, 1::2][:, :-1]                                          # bp of frames 1 .. W - 1


@torch.no_grad()
def encode_dev(model, belief, cfg, L):
    out = []
    for e, ep in enumerate(S.episodes(L)):
        rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
        fx = [D.fixation_prior(rng, ()) for _ in range(ep.T + 1)]
        c = TR.tensors(D.gather([ep], [[(0, 1, t, *fx[t]) for t in range(ep.T + 1)]]), DEV)
        with TR.autocast(DEV, cfg):
            P = M.Percept(*(torch.cat(z, 1) for z in zip(*[model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]).parts()
                                                          for a in range(0, ep.T + 1, 512)])), c["fixation"])
            y, _ = belief(belief.tokens(belief.percept_embed(P), c["reward"], c["terminated"], c["action"]), belief.initial(1, DEV))
        bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)[0].float()
        out.append(bp.cpu())
    return torch.cat(out)


def probe(X, y, ok, fold):
    """Linear logistic probe per event, fit on folds 0-3, AP on fold 4."""
    X = (X - X[fold < 4].mean(0)) / X[fold < 4].std(0).clamp(min=1e-6)
    res = {}
    for j, n in enumerate(EVENTS):
        tr, te = (fold < 4) & ok[:, j], (fold == 4) & ok[:, j]
        Xt, yt = X[tr].to(DEV), torch.as_tensor(y[tr, j], dtype=torch.float32, device=DEV)
        w = torch.zeros(X.shape[1] + 1, device=DEV, requires_grad=True)
        opt = torch.optim.LBFGS([w], max_iter=200, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            l = F.binary_cross_entropy_with_logits(Xt @ w[:-1] + w[-1], yt) + 1e-3 * w[:-1].square().sum()
            l.backward()
            return l
        opt.step(closure)
        s = (X[te].to(DEV) @ w[:-1] + w[-1]).detach().cpu().numpy()
        res[n] = {"ap": float(GT.average_precision(y[te, j], s)), "prevalence": float(y[te, j].mean()), "test_rows": int(te.sum())}
    return res


def main(arm):
    out = HERE / f"{arm}.json"
    if out.exists():
        return
    torch.manual_seed(0)
    ck, model, cfg, _ = DZ.load(10)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    Ldev = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    ydev, okdev = events(Ldev)
    fold = np.repeat(np.arange(len(Ldev["ep_T"])) % 5, Ldev["ep_T"] + 1)
    if arm == "g1":
        return g1_arm(model, cfg, Ldev, ydev, okdev, fold, out)
    if arm == "fresh_events":
        torch.manual_seed(1)
        belief = M.M0(cfg).belief.to(DEV)
    else:
        belief = copy.deepcopy(model.belief)
    belief.train()
    for p in belief.parameters():
        p.requires_grad_(arm != "frozen")
    torch.manual_seed(2)
    head = (Events(cfg["belief"]["width"]) if arm in ("frozen", "events", "fresh_events") else TH(cfg["belief"]["width"])).to(DEV)
    groups = [{"params": list(head.parameters()), "lr": 1e-3}]
    if arm != "frozen":
        groups.append({"params": list(belief.parameters()), "lr": 1e-4})
    opt = torch.optim.Adam(groups)
    base_lr = [g["lr"] for g in groups]
    Ltr = dict(np.load(S.LABELS / "s0_train_labels.npz"))
    ytr, oktr = events(Ltr)
    so = np.r_[0, np.cumsum(Ltr["ep_T"] + 1)]
    eps = S.episodes(Ltr)
    long_eps = np.flatnonzero(Ltr["ep_T"] >= W)
    rng = np.random.default_rng(7)                                     # the same draws for every arm
    res = {"arm": arm, "checkpoint_sha256": TR.file_sha256(ck), "window": W, "batch": B, "evals": {}, "log": []}
    t0, losses = time.time(), []
    for u in range(1, UPDATES + 1):
        es = rng.choice(long_eps, B)
        starts = [int(rng.integers(0, Ltr["ep_T"][e] - W + 1)) for e in es]
        recs = [[(int(e), int(e), s + t, *D.fixation_prior(rng, ())) for t in range(W)] for e, s in zip(es, starts)]
        chunk = TR.tensors(D.gather(eps, recs), DEV)
        with torch.no_grad(), TR.autocast(DEV, cfg):
            P = model.perception(chunk["frames"], chunk["fixation"])
        P = M.Percept(*(z.float() for z in P.parts()), P.fix)
        with TR.autocast(DEV, cfg):
            bp = run_belief(belief, P, chunk).float()
        if isinstance(head, Events):
            idx = np.stack([so[e] + s + np.arange(1, W) for e, s in zip(es, starts)])
            y = torch.as_tensor(ytr[idx], dtype=torch.float32, device=DEV)
            ok = torch.as_tensor(oktr[idx], device=DEV)
            loss = F.binary_cross_entropy_with_logits(head(bp)[ok], y[ok])
        else:
            prev, tgt = P.p[:, :-1], P.p[:, 1:]
            err = (prev + head(bp) - tgt).square()
            loss = (err[..., DIGITS, :] if arm == "digit_mse" else err).mean()
        f = min(1.0, u / 100)
        for g, lr in zip(opt.param_groups, base_lr):
            g["lr"] = lr * f
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        opt.step()
        losses.append(float(loss))
        if u % 250 == 0:
            res["log"].append({"update": u, "loss": float(np.mean(losses[-250:])), "sec_per_update": (time.time() - t0) / u})
            print(arm, res["log"][-1], flush=True)
        if u in EVALS:
            belief.eval()
            X = encode_dev(model, belief, cfg, Ldev)
            res["evals"][u] = probe(X, ydev, okdev, fold)
            if isinstance(head, Events):
                with torch.no_grad():
                    s = head(X.to(DEV)).cpu().numpy()
                res["evals"][u]["own_head"] = {n: float(GT.average_precision(ydev[(fold == 4) & okdev[:, j], j], s[(fold == 4) & okdev[:, j], j]))
                                               for j, n in enumerate(EVENTS)}
            print(arm, u, json.dumps(res["evals"][u]), flush=True)
            belief.train()
    out.write_text(json.dumps(res, indent=1) + "\n")


def g1_arm(model, cfg, Ldev, ydev, okdev, fold, out, updates=2000, evals=(1000, 2000)):
    """G1 alone, F6's own stream machinery (16 streams x 64 steps, carried state, 2 microbatches), belief and denoiser
    trained, perception frozen, AdamW at F6's peak rate."""
    ck = torch.load(DZ.F6.HERE / "run/ckpt_10.pt", map_location="cpu", weights_only=False)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    params = list(model.belief.parameters()) + list(model.denoiser.parameters())
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=1e-4, betas=tuple(cfg["optimizer"]["betas"]), eps=cfg["optimizer"]["eps"], weight_decay=0.01)
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=DZ.F6.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    carry = tuple(M.MambaCarry(c.conv.to(DEV), c.ssm.to(DEV)) for c in ck["carry"])
    cache = ck["cache"].to(DEV)
    res = {"arm": "g1", "evals": {}, "log": []}
    t0, losses = time.time(), []
    for u in range(1, updates + 1):
        chunk = streams.next(64)
        boot = TR.bootstrap_rows(cfg, DZ.F6.WORLD_SEED, 15625 + u)
        f = min(1.0, u / 100)
        for g in opt.param_groups:
            g["lr"] = 1e-4 * f
        opt.zero_grad(set_to_none=True)
        for rows in np.array_split(np.arange(d["streams"]), 2):
            terms, c_new, cache_new = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows],
                                                   None, cfg, boot[rows], 15625 + u, None)
            (terms["g1"] / 2).backward()
            M.carry_put(carry, rows, c_new)
            cache[rows] = cache_new
            losses.append(float(terms["g1"]))
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if u % 250 == 0:
            res["log"].append({"update": u, "g1": float(np.mean(losses[-500:])), "sec_per_update": (time.time() - t0) / u})
            print("g1", res["log"][-1], flush=True)
        if u in evals:
            model.eval()
            X = encode_dev(model, model.belief, cfg, Ldev)
            res["evals"][u] = probe(X, ydev, okdev, fold)
            print("g1", u, json.dumps(res["evals"][u]), flush=True)
            model.train()
    out.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main(sys.argv[1])
