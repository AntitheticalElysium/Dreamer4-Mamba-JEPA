"""F6 dissection, second pass (PLAN.md): the factors behind each loss term's gradient into the belief.

    MAMBA_DETERMINISTIC=1 [DOSE_RUN=...train_p1f5] python -m artifacts.experiments.20261114_f6_dissection.share factors [ckpts...]
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_f6_dissection.share align [ckpts...]
"""
import importlib
import json
import os
import sys

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

DZ = importlib.import_module("artifacts.experiments.20261114_f6_dissection.dose")
OUT, F6, TAG, DEV = DZ.OUT, DZ.F6, DZ.TAG, DZ.DEV


def instrument(model, cap, state):
    """Capture the belief tensors the heads and the denoiser read; with state['shift'], hand them another frame's belief."""
    def swap(b):
        return torch.roll(b, b.shape[1] // 2, 1) if state["shift"] else b
    mem, feat = model.belief.memory, model.outcome.features
    model.belief.memory = lambda b: (cap.__setitem__("g1", b), mem(swap(b)))[1]
    model.outcome.features = lambda b, e: (cap.__setitem__("heads", b), feat(swap(b), e))[1]
    tsp = model.tsp.predict
    model.tsp.predict = lambda b, p, v, c: (cap.__setitem__("heads", b), tsp(swap(b), p, v, c))[1]
    if model.token_forecast is not None:
        tok = model.token_forecast.forward
        model.token_forecast.forward = lambda b: (cap.__setitem__("token_forecast", b), tok(swap(b)))[1]


def factors(k):
    path = OUT / f"{TAG}share_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, params = DZ.load(k)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    ck = torch.load(ckp, map_location="cpu", weights_only=False)
    d, P = cfg["data"], cfg["data"]["context_prefix"]
    split = D.load_split(cfg)
    streams = D.Streams(split["train"], d["streams"], P, seed=F6.WORLD_SEED, targets_offsets=D.frame_offsets(split["train"]))
    streams.load_state(ck["streams"])
    chunk = streams.next(64)
    carry = tuple(M.MambaCarry(c.conv.to(DEV), c.ssm.to(DEV)) for c in ck["carry"])
    cache, update = ck["cache"].to(DEV), ck["state"]["update"]
    boot = TR.bootstrap_rows(cfg, F6.WORLD_SEED, update)
    fc = "token_forecast" if model.token_forecast is not None else "tsp_forecast"
    where = {"g1": "g1", "tsp": "heads", "reward": "heads", "reward_forecast": "heads", fc: fc if fc == "token_forecast" else "heads"}
    run = lambda rows: TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg,
                                    boot[rows], update, None)[0]
    torch.manual_seed(0)
    plain = [{n: float(v) for n, v in run(np.array([r])).items()} for r in range(8)]
    cap, state = {}, {"shift": False}
    instrument(model, cap, state)
    per = []
    for r in range(8):
        rows = np.array([r])
        sel = D.select(chunk, rows)
        valid = (~sel["first"][:, P:] & sel["valid"][:, P:])[0]
        rew = sel["reward"][0, P:]
        torch.manual_seed(r)
        terms = run(rows)
        torch.manual_seed(r)
        e = {"frames": int(valid.sum()), "reward_frames": int((np.asarray(rew)[valid] != 0).sum()), "terms": {}}
        for n, key in where.items():
            g = torch.autograd.grad(terms[n], cap[key], retain_graph=True)[0].float()[0]
            g = g[-len(valid):] * float(len(valid) if n in ("g1", "tsp") else valid.sum())   # undo the frame mean: one frame's pull on its belief
            norms = g.norm(dim=-1)[torch.as_tensor(valid)].cpu().numpy()
            top = np.sort(norms)[::-1]
            row = {"loss": float(terms[n]), "pull_mean": float(norms.mean()), "pull_median": float(np.median(norms)),
                   "top5pct_share": float(top[:max(1, len(top) // 20)].sum() / max(norms.sum(), 1e-30)),
                   "addition": float(g[torch.as_tensor(valid)].sum(0).norm() / max(norms.sum(), 1e-30)),
                   "random_addition": float(1 / np.sqrt(len(norms)))}
            if n.startswith("reward"):
                ev = np.asarray(rew)[valid] != 0
                row["reward_frame_share"] = float(norms[ev].sum() / max(norms.sum(), 1e-30))
            e["terms"][n] = row
        state["shift"] = True
        torch.manual_seed(r)
        with torch.no_grad():
            other = run(rows)
        state["shift"] = False
        for n in where:
            e["terms"][n]["other_frames_belief_over_true"] = float(other[n]) / max(float(terms[n]), 1e-30)
        e["known_answer_max_abs"] = max(abs(float(terms[n]) - plain[r][n]) for n in where if n != "g1")
        per.append(e)
        del terms
        torch.cuda.empty_cache()
    sigma = dict(zip(model.harmoniser.groups, model.harmoniser.log_sigma.detach().exp().tolist()))
    med = lambda n, f: float(np.median([q["terms"][n][f] for q in per if f in q["terms"][n]]))
    summ = {n: {f: med(n, f) for f in per[0]["terms"][n]} for n in where}
    gl = {g: float(np.median([sum(p[t] for t in ts if t in p) for p in plain]))
          for g, ts in {"G1": ["g1"], "G2": ["tsp", "rec"], "G3": ["reward", "cont"], "G4": [fc, "reward_forecast", "cont_forecast"]}.items()}
    stationary = {g: float(0.5 * (L + np.sqrt(L * L + 4 * L))) for g, L in gl.items()}   # L = s^2 / (1 + s)
    res = {"checkpoint": k, "update": update, "summary": summ, "sigma": sigma, "group_loss_median": gl,
           "sigma_at_stationary_point": stationary, "known_answer_max_abs": max(q["known_answer_max_abs"] for q in per),
           "reduction": {"g1": "mean over frames x 1,880 token numbers x levels", "tsp": "mean over frames x 9 cells x 147 pixel values",
                         "reward": "mean over frames of one categorical log-loss", "reward_forecast": "mean over frames of one categorical log-loss",
                         "token_forecast": "mean over frames x 1,296 token numbers", "tsp_forecast": "mean over frames x 1,323 pixel values"},
           "per_stream": per}
    print(k, json.dumps({kk: vv for kk, vv in res.items() if kk != "per_stream"}), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def align(k):
    path = OUT / f"align_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, _ = DZ.load(k)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    tot = {s: {"copy": 0., "head": 0., "aligned": 0., "n": 0} for s in ("+", "-")}
    moved = 0
    for e, r in enumerate(S.encode(model, L, cfg)):
        prev, tgt = r["p"][:-1].float().to(DEV), r["p"][1:].float().to(DEV)
        pred = prev + model.token_forecast(r["bp"][1:].float().to(DEV))
        pos, sleep = L["s_pos"][off[e]:off[e + 1]].astype(int), L["s_sleep"][off[e]:off[e + 1]].astype(bool)
        dlt = pos[1:] - pos[:-1]
        for t in np.flatnonzero((np.abs(dlt).sum(1) == 1) & ~sleep[:-1]):
            moved += 1
            a, b, c = (x[t, :63].reshape(7, 9, -1) for x in (prev, tgt, pred))
            for s, (dr, dc) in (("+", dlt[t]), ("-", -dlt[t])):          # both sign conventions; the lower aligned error is the screen's
                rs, cs = slice(max(0, -dr), 7 - max(0, dr)), slice(max(0, -dc), 9 - max(0, dc))
                rq, cq = slice(max(0, dr), 7 + min(0, dr)), slice(max(0, dc), 9 + min(0, dc))
                n = b[rs, cs].numel() // b.shape[-1]
                tot[s]["aligned"] += float((a[rq, cq] - b[rs, cs]).square().mean(-1).sum())
                tot[s]["copy"] += float((a[rs, cs] - b[rs, cs]).square().mean(-1).sum())
                tot[s]["head"] += float((c[rs, cs] - b[rs, cs]).square().mean(-1).sum())
                tot[s]["n"] += n
    res = {"checkpoint": k, "moved_awake_transitions": moved,
           "by_sign": {s: {m: v[m] / v["n"] for m in ("copy", "head", "aligned")} | {"cells": v["n"]} for s, v in tot.items()}}
    print(k, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")


@torch.no_grad()
def light(k):
    """Still-frame map error by daylight: how much of the term is change no history predicts (night noise, lighting)?
    Copy error and head error per map cell on awake frames where the player did not move, split by the frame's light
    level and by whether any entity is in view before or after (entity-free still frames change only by lighting,
    night noise or terrain edits)."""
    path = OUT / f"light_{k:02d}.json"
    if path.exists():
        return
    ckp, model, cfg, _ = DZ.load(k)
    model.eval()
    L = dict(np.load(S.LABELS / "s0_dev_labels.npz"))
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    G = importlib.import_module("saccade.eval.gates")
    grids = G.episode_grids(L)
    acc = {}
    for e, r in enumerate(S.encode(model, L, cfg)):
        prev, tgt = r["p"][:-1, :63].float().to(DEV), r["p"][1:, :63].float().to(DEV)
        pred = prev + model.token_forecast(r["bp"][1:].float().to(DEV))[:, :63]
        cp, hd = (prev - tgt).square().mean((-1, -2)).cpu().numpy(), (pred - tgt).square().mean((-1, -2)).cpu().numpy()
        a, b = off[e], off[e + 1]
        pos, sleep, lt = L["s_pos"][a:b].astype(int), L["s_sleep"][a:b].astype(bool), L["s_light"][a:b]
        still = (pos[1:] == pos[:-1]).all(-1) & ~sleep[:-1] & ~sleep[1:]
        ent = (grids[e]["grid"][:-1] > 0).any((1, 2)) | (grids[e]["grid"][1:] > 0).any((1, 2))
        view_same = (L["s_view"][a:b][1:] == L["s_view"][a:b][:-1]).all((1, 2))
        for name, m in (("day", lt[1:] >= .99), ("dusk_dawn", (lt[1:] < .99) & (lt[1:] >= .5)), ("night", lt[1:] < .5)):
            for en, me in (("entities", ent), ("no_entities_same_terrain", ~ent & view_same)):
                q = still & m & me
                z = acc.setdefault(f"{name}/{en}", [0, 0., 0.])
                z[0] += int(q.sum()); z[1] += float(cp[q].sum()); z[2] += float(hd[q].sum())
    res = {"checkpoint": k, "still_awake_map_error_per_cell": {n: {"frames": v[0], "copy": v[1] / max(v[0], 1), "head": v[2] / max(v[0], 1)}
                                                                 for n, v in acc.items()}}
    print(k, json.dumps(res), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    mode, ks = sys.argv[1], [int(x) for x in sys.argv[2:]]
    for k in ks or ((2, 10) if mode == "factors" else (10,)):
        {"factors": factors, "align": align, "light": light}[mode](k)
