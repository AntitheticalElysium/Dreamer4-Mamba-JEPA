"""Checkpoint readings of P1-F6.1 (PLAN.md, readings): F6's clock instruments without the token-forecast one (gate3),
unchanged and pointed at this run, so every number has F5's and F6's at the same checkpoint beside it.

  trajectory  ../20261111_p1f5/trajectory.py: reward-forecast separation (distance 1), native generation due / not due,
              belief probe (event APs from the predicted belief).
  d1          ../20261113_token_forecast_gate/d1.py: each reset's registration at the tokens, Mamba's input and state.
  trace       ../20261111_p1f5/trace.py: how much of the earlier reset survives in the state to the due step.

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261114_p1f61.readout CHECKPOINT [stages...]
"""
import importlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from saccade import train as TR
from saccade.eval import s1 as S

HERE = Path(__file__).resolve().parent
OUT = HERE / "readings"
G2 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate2")
D1 = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.d1")
TC = importlib.import_module("artifacts.experiments.20261111_p1f5.trace")
TJ, NP = G2.TJ, G2.NP
STAGES = ("trajectory", "d1", "trace", "budget", "ahead")
COUNTERS = HERE.parent / "20261107_f4_followups/readings/counters_dev.npz"


def budget(ck, out):
    """Each loss term's gradient into Mamba's module (belief) and into the encoder (perception), raw and divided by
    its group's sigma, with its cosine to the weighted total, on the checkpoint's own next training chunk (8 streams
    x 64 new frames, one stream at a time; medians), as ../20261114_f6_dissection/dose.py batch but for any recipe.
    The grouped reward forecast is split into distance 1 and the rest ("reward_forecast_ahead")."""
    from saccade import data as D, model as M
    model, cfg = NP.IF.load(ck)
    for prm in model.parameters():
        prm.requires_grad_(True)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    state = torch.load(ck, map_location="cpu", weights_only=False)
    d, P, dev = cfg["data"], cfg["data"]["context_prefix"], next(model.parameters()).device
    train = D.load_split(cfg)["train"]
    streams = D.Streams(train, d["streams"], P, seed=1, targets_offsets=D.frame_offsets(train))
    streams.load_state(state["streams"])
    chunk = streams.next(64)
    carry = tuple(M.MambaCarry(c.conv.to(dev), c.ssm.to(dev)) for c in state["carry"])
    cache, update = state["cache"].to(dev), state["state"]["update"]
    boot = TR.bootstrap_rows(cfg, 1, update)
    sigma = dict(zip(model.harmoniser.groups, model.harmoniser.log_sigma.detach().exp().tolist()))
    group = lambda n: "G1" if n == "g1" else "G2" if n in ("tsp", "rec") else "G3" if n in ("reward", "cont") else "G4"
    sets = {"belief": list(model.belief.parameters()), "encoder": list(model.perception.parameters())}
    torch.manual_seed(0)
    per = []
    for r in range(8):
        rows = np.array([r])
        terms, _, _ = TR.p1_losses(model, D.select(chunk, rows), M.carry_rows(carry, rows), cache[rows], None, cfg, boot[rows], update, None)
        grouped = [n for n in terms if n != "noise" and not n.startswith("reward_forecast_")]   # per-distance terms are parts
        shown = [n for n in grouped if n != "reward_forecast"] + (
            ["reward_forecast_1", "reward_forecast_ahead"] if "reward_forecast_1" in terms else ["reward_forecast"])
        if "reward_forecast_1" in terms:
            terms["reward_forecast_ahead"] = terms["reward_forecast"] - terms["reward_forecast_1"]
        row = {"loss": {n: float(terms[n]) for n in shown}}
        for name, params in sets.items():
            g = {n: G2.flat(torch.autograd.grad(terms[n], params, retain_graph=True, allow_unused=True), params) for n in set(grouped) | set(shown)}
            total = sum(g[n] / sigma[group(n)] for n in grouped)
            cos = lambda a, b: float(a @ b / (a.norm() * b.norm()).clamp(min=1e-30))
            row[name] = {"weighted_norm": {n: float(g[n].norm() / sigma[group(n)]) for n in shown},
                         "cos_to_total": {n: cos(g[n], total) for n in shown}, "total_norm": float(total.norm())}
            del g, total
        per.append(row)
        del terms
        torch.cuda.empty_cache()
    med = lambda f: {n: float(np.median([f(q)[n] for q in per])) for n in per[0]["loss"]}
    res = {"checkpoint": str(ck), "sha256": TR.file_sha256(ck), "update": update, "sigma": sigma,
           "loss_median": med(lambda q: q["loss"])}
    for name in sets:
        res[name] = {"weighted_norm_median": med(lambda q: q[name]["weighted_norm"]),
                     "cos_to_total_median": med(lambda q: q[name]["cos_to_total"]),
                     "total_norm_median": float(np.median([q[name]["total_norm"] for q in per]))}
    res["per_stream"] = per
    out.write_text(json.dumps(res, indent=1) + "\n")
    print("budget", ck.name, {s: {n: float(f"{v:.3g}") for n, v in res[s]["weighted_norm_median"].items()} for s in sets}, flush=True)
    del model
    torch.cuda.empty_cache()


def ahead(ck, out, seed=110711):
    """Codex 11-14: does a reward forecast use the sleep clock? On held-out DEV asleep origins (asleep at the last frame
    the belief has seen), each distance's reward log-loss with two donors' predicted beliefs in place of the origin's
    own: asleep origins of other episodes with the same energy and health digits and the same WAKE_UP status, one at
    the same simulator fatigue phase (control: everything else about a foreign belief) and one at a different phase.
    The phase effect is the mismatched donor's loss minus the matched donor's (labels choose donors only). Also each
    distance's own-belief log-loss on every frame."""
    from craftax.craftax_classic.constants import Achievement
    model, cfg = NP.IF.load(ck)
    n = (cfg["heads"].get("forecast") or {}).get("reward_distances", 1)
    L = dict(np.load(G2.CV.LABELS / "s0_dev_labels.npz"))
    fat = np.load(COUNTERS)["fatigue"]
    enc = S.encode(model, L, cfg)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    rows = []                                                            # (episode, origin, key, fatigue, paid)
    for e, rec in enumerate(enc):
        for o in range(1, int(L["ep_T"][e]) + 1):
            g = off[e] + o - 1
            if L["s_sleep"][g]:
                rows.append((e, o, (int(L["s_body"][g, 3]), int(L["s_body"][g, 0]), bool(L["s_ach"][g, Achievement.WAKE_UP.value])),
                             float(fat[g]), bool(L["s_ach"][g, Achievement.WAKE_UP.value])))
    rng = np.random.default_rng(seed)
    by_key = {}
    for i, r in enumerate(rows):
        by_key.setdefault(r[2], []).append(i)
    by_key = {k: np.array(v) for k, v in by_key.items()}
    def draw(i, same):                                                   # rejection draw: another episode, same / other phase
        e, f = rows[i][0], rows[i][3]
        pool = by_key[rows[i][2]]
        for c in rng.choice(pool, min(256, len(pool)), replace=False):
            if rows[c][0] != e and (rows[c][3] == f) == same:
                return int(c)
        return -1
    donor = [(draw(i, False), draw(i, True)) for i in range(len(rows))]
    dev = next(model.parameters()).device

    def logp(b, r, d):
        outs = []
        for a in range(0, len(b), 32768):
            with torch.no_grad(), TR.autocast(dev, cfg):
                h = model.outcome.features(b[a:a + 32768].to(dev).float(), None)
                rr = r[a:a + 32768].to(dev)
                outs.append((model.outcome.reward_logp(h, rr) if d == 1 else model.outcome.ahead_logp(h, rr, d)).float().cpu())
        return torch.cat(outs)

    res = {"checkpoint": str(ck), "sha256": TR.file_sha256(ck), "distances": n, "asleep_origins": len(rows),
           "with_both_donors": int(sum(a >= 0 and b >= 0 for a, b in donor))}
    bel = lambda j: enc[rows[j][0]]["bp"][rows[j][1]]
    for d in range(1, n + 1):
        keep = [i for i, (e, o, *_) in enumerate(rows) if min(donor[i]) >= 0 and o + d - 1 <= int(L["ep_T"][e])]
        r = torch.stack([enc[rows[i][0]]["reward"][rows[i][1] + d - 1] for i in keep])
        own = -logp(torch.stack([bel(i) for i in keep]), r, d).numpy()
        other = -logp(torch.stack([bel(donor[i][0]) for i in keep]), r, d).numpy()
        same = -logp(torch.stack([bel(donor[i][1]) for i in keep]), r, d).numpy()
        ep = np.array([rows[i][0] for i in keep])
        cohort = {}
        for name, m in (("all", np.ones(len(keep), bool)), ("wake_up_not_yet_paid", np.array([not rows[i][4] for i in keep])),
                        ("wake_up_already_paid", np.array([rows[i][4] for i in keep]))):
            eps, inv = np.unique(ep[m], return_inverse=True)            # bootstrap over episodes, with multiplicity
            idx = rng.integers(0, len(eps), (1000, len(eps)))
            cnts = np.bincount(inv, None, len(eps))
            ci = lambda x: np.percentile(np.bincount(inv, x, len(eps))[idx].sum(1) / cnts[idx].sum(1), [2.5, 97.5]).tolist()
            cohort[name] = {"rows": int(m.sum()), "own_nats": float(own[m].mean()), "same_phase_donor_nats": float(same[m].mean()),
                            "other_phase_donor_nats": float(other[m].mean()),
                            "phase_effect": float((other - same)[m].mean()), "phase_effect_ci95": ci((other - same)[m]),
                            "foreign_belief_effect": float((same - own)[m].mean()), "foreign_belief_ci95": ci((same - own)[m])}
        allb = torch.cat([rec["bp"][1:rec["bp"].shape[0] - d + 1] for rec in enc])
        allr = torch.cat([rec["reward"][d:] for rec in enc])
        res[f"distance_{d}"] = {"asleep": cohort, "every_frame_own_nats": float(-logp(allb, allr, d).mean())}
        print("ahead", ck.name, d, {k: round(v["phase_effect"], 4) for k, v in cohort.items()}, flush=True)
    out.write_text(json.dumps(res, indent=1) + "\n")
    del model, enc
    torch.cuda.empty_cache()


def main(k, stages):
    ck = (HERE / f"run/ckpt_{k:02d}.pt").resolve()
    sha = TR.file_sha256(ck)
    for name, fn in (("budget", budget), ("ahead", ahead)):
        if name in stages and not (OUT / f"{name}_{k:02d}.json").exists():
            OUT.mkdir(parents=True, exist_ok=True)
            fn(ck, OUT / f"{name}_{k:02d}.json")
    if "trajectory" in stages:
        TJ.HERE, TJ.OUT = HERE, OUT / "trajectory"
        TJ.OUT.mkdir(parents=True, exist_ok=True)
        TJ.checkpoint(k)
    if "d1" in stages and not (OUT / f"d1/d1_ckpt_{k:02d}.json").exists():
        G2.OUT = D1.OUT = OUT / "d1"
        model, cfg = NP.IF.load(ck)
        L = dict(np.load(G2.CV.LABELS / "s0_dev_labels.npz"))
        G2.d1_features(model, cfg, S.encode(model, L, cfg), L, k)
        del model
        torch.cuda.empty_cache()
        D1.checkpoint(k)
    if "trace" in stages and not (OUT / f"trace_{k:02d}.json").exists():
        model, cfg = NP.IF.load(ck)
        res = {"checkpoint": k, "sha256": sha}
        for name in ("recovery", "energy", "hunger"):
            cs = [TC.curve(model, cfg, r) for r in TJ.pair_rows(name)]
            n = min(len(c) for c in cs)
            m = np.mean([c[:n] for c in cs], 0)
            res[name] = {"q": n - 1, "mean_by_step": m.tolist(), "at_6": float(m[6]), "at_15": float(m[15]), "at_due": float(m[n - 1])}
            print(k, "trace", name, round(res[name]["at_due"], 5), flush=True)
        (OUT / f"trace_{k:02d}.json").write_text(json.dumps(res, indent=1) + "\n")
    assert TR.file_sha256(ck) == sha


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    main(int(sys.argv[1]), sys.argv[2:] or STAGES)
