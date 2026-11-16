"""Round 2 of the Option A gate (PLAN.md, round 2). Frozen F5 checkpoints; fresh heads on frozen beliefs; no world
training. Part 1: HL-Gauss cross-entropy vs plain MSE token forecast (nuisance as the trained head pays it, request
cosine, request with the loss on the digit's cell only, matched held-out update in the trainer's AdamW geometry, with
the reward forecast on recovery as reference). Part 2: D2 input substitution on the pairs (hunger amplified / sham,
energy attenuated); D1 stage features for d1.py (tokens / Mamba's input / state changes on section 6's transitions).

    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261113_token_forecast_gate.gate2 [checkpoints...]
"""
import importlib
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

G = importlib.import_module("artifacts.experiments.20261113_token_forecast_gate.gate")
CR, TJ, NP, DEV, CV, F5 = G.CR, G.TJ, G.NP, G.DEV, G.CV, G.F5
OUT = G.HERE / os.environ.get("GATE_OUT", "readings/round2")
STEPS, BATCH, FOILS = G.STEPS, G.BATCH, G.FOILS
NB, LO, HI = 51, -3.0, 3.0                                  # HL-Gauss: Farebrother et al. 2024 defaults; support = token range
EDGES = torch.linspace(LO, HI, NB + 1)
CENTRES = (EDGES[:-1] + EDGES[1:]) / 2
SIGMA = 0.75 * (HI - LO) / NB
CURS = ("plain", "categorical")
SETS = {"energy": 66, "hunger": 64, "recovery": 63, "cooldown": 63}
MATCHED = ("energy", "hunger", "recovery")


def labels(x, dtype=torch.float32):
    """HL-Gauss target over the bins for every token number x [...] -> [..., NB], renormalised inside the support."""
    c = torch.special.ndtr((EDGES.to(x.device, dtype) - x[..., None].to(dtype)) / SIGMA)
    return (c[..., 1:] - c[..., :-1]) / (c[..., -1:] - c[..., :1])


def per_number(cur, out, x):
    if cur == "plain":
        return (out.float() - x.float()) ** 2
    return -(labels(x) * out.float().log_softmax(-1)).sum(-1)


def L(cur, out, x, slot=None):
    v = per_number(cur, out, x)
    return (v if slot is None else v[..., slot, :]).mean()


def estimate(cur, out):
    return out.float() if cur == "plain" else (out.float().softmax(-1) * CENTRES.to(out.device)).sum(-1)


def oracle(cur, xa, xb):
    """Exact-forecaster cost of predicting xb when the truth is xa: round 1's MSE, or KL between HL-Gauss labels (no floor)."""
    if cur == "plain":
        return float(((xa - xb) ** 2).mean())
    la, lb = labels(xa, torch.float64), labels(xb, torch.float64)
    return float((la * (la.clamp(min=1e-300).log() - lb.clamp(min=1e-300).log())).sum(-1).mean())


def make_head(cur, W):
    h = G.TokenHead(W)
    if cur == "categorical":
        h.net[-1] = nn.Linear(256, 16 * NB)                                    # the same trunk, NB logits per number
    return h.to(DEV)


def run(h, cur, bp):
    out = h(bp)
    return out if cur == "plain" else out.unflatten(-1, (16, NB))


def flat(gs, params):
    return torch.cat([(g if g is not None else torch.zeros_like(p)).float().flatten() for g, p in zip(gs, params)])


def cosine(a, b):
    return float(-(a @ b) / (a.norm() * b.norm()).clamp(min=1e-30))      # a = grad sep, b = grad loss: -cos = request


def preconditioner(model, cfg, ckpt, params):
    """1 / (sqrt(v_hat) + eps) of F5's AdamW at this checkpoint, flattened in `params` order."""
    st = torch.load(ckpt, map_location="cpu", weights_only=False)["opt"]
    order = [p for g in model.param_groups() for p in g["params"]]
    assert len(order) == len(st["state"])
    where = {id(p): i for i, p in enumerate(order)}
    b2, eps = cfg["optimizer"]["betas"][1], cfg["optimizer"]["eps"]
    out = []
    for p in params:
        s = st["state"][where[id(p)]]
        assert s["exp_avg_sq"].shape == p.shape
        vhat = s["exp_avg_sq"].float() / (1 - b2 ** float(s["step"]))
        out.append((1 / (vhat.sqrt() + eps)).flatten())
    return torch.cat(out).to(DEV)


def history(model, cfg, row, variant=None, scale=1.0, seed=0):
    """Predicted belief at the due step, [2, W], from the pair's inputs, optionally with the members' input-embedding
    difference scaled (amplified / attenuated) or replaced by an equal added norm in a random direction (sham)."""
    p, rw, td, act = CR.belief_inputs(model, cfg, row)
    with TR.autocast(DEV, cfg):
        emb = model.belief.percept_embed(p)
        if variant is not None:
            diff = emb[0] - emb[1]
            if variant == "sham":
                r = torch.randn(diff.shape, generator=torch.Generator().manual_seed(seed)).to(DEV, diff.dtype)
                r = r * (diff.detach().norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True))
                add = (scale - 1) / 2 * r
            else:
                add = (scale - 1) / 2 * diff
            emb = torch.stack([emb[0] + add, emb[1] - add])
        y, _ = model.belief(model.belief.tokens(emb, rw, td, act), model.belief.initial(2, DEV))
    return y[:, -1].float()


def due_targets(model, row):
    i, e, prefix, frames, rewards, terms, q, truth = row
    query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=DEV)
    with torch.no_grad():
        nxt = torch.as_tensor(frames[:, q + 1:q + 2], device=DEV)
        x = model.perception(nxt, query[None, None].expand(2, 1, -1)).p[:, 0].float()
        xb = {a: model.perception(torch.clamp(nxt[1:].int() + a, max=255).to(torch.uint8), query[None, None]).p[0, 0].float()
              for a in FOILS}
    return x, xb


def log_distance(y):
    return ((y[0] - y[1]).square().sum() / ((y[0].square().sum() + y[1].square().sum()) / 2)).log()


def emb_diff_norm(model, cfg, row):
    p, *_ = CR.belief_inputs(model, cfg, row)
    with torch.no_grad(), TR.autocast(DEV, cfg):
        emb = model.belief.percept_embed(p).float()
    return float((emb[0] - emb[1]).norm())


def d1_features(model, cfg, enc, L_, k):
    """|change| at the tokens, Mamba's input and the corrected state on section 6's 40k transitions (seed 2120)."""
    RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
    T = CV.transitions(L_)
    rows = np.sort(np.random.default_rng(2120).choice(np.flatnonzero(T["t"] >= CV.P), 40000, replace=False))
    feats = {"p": [], "x": [], "b": []}
    with torch.no_grad():
        for a in range(0, len(rows), 2048):
            j = rows[a:a + 2048]
            at = {dt: {key: torch.stack([enc[int(T["ep"][x])][key][int(T["t"][x]) + dt] for x in j]).to(DEV)
                       for key in ("p", "g", "v", "fix", "b") + (("reads",) if "reads" in enc[0] else ())} for dt in (-1, 0)}
            x = {dt: S.embed(model, M.Percept(at[dt]["p"].float(), at[dt]["g"].float(), at[dt]["v"].float(), at[dt]["fix"]),
                             at[dt].get("reads")).float() for dt in (-1, 0)}   # P1.5: Mamba's input holds the record's read
            feats["p"].append((at[0]["p"].float() - at[-1]["p"].float()).abs().flatten(1).cpu())
            feats["x"].append((x[0] - x[-1]).abs().cpu())
            feats["b"].append((at[0]["b"].float() - at[-1]["b"].float()).abs().cpu())
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"d1_ckpt_{k:02d}.npz", rows=rows, **{s: torch.cat(v).numpy() for s, v in feats.items()})
    del RT


def checkpoint(k):
    path = OUT / f"ckpt_{k:02d}.json"
    if path.exists():
        return
    torch.manual_seed(0)
    ck = (F5 / f"run/ckpt_{k:02d}.pt").resolve()
    model, cfg = NP.IF.load(ck)
    model.eval()
    L_ = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L_, cfg)
    if os.environ.get("GATE_SKIP_D1") != "1":
        d1_features(model, cfg, enc, L_, k)
    tgt, bps, ep_of = [], [], []
    for e, r in enumerate(enc):
        tgt.append(r["p"][1:].half()); bps.append(r["bp"][1:].float()); ep_of.append(torch.full((len(r["p"]) - 1,), e))
    tgt, bp, ep_of = map(torch.cat, (tgt, bps, ep_of))
    del enc
    held = ep_of % 5 == 4
    tr_ix, te_ix = torch.nonzero(~held)[:, 0], torch.nonzero(held)[:, 0]
    res = {"checkpoint": k, "sha256": TR.file_sha256(ck), "hl_gauss": {"bins": NB, "support": [LO, HI], "sigma": SIGMA}}
    heads = {}
    for cur in CURS:
        net = make_head(cur, bp.shape[-1])
        opt = torch.optim.Adam(net.parameters(), 1e-3)
        g = torch.Generator().manual_seed(1)
        for s in range(STEPS):
            ix = tr_ix[torch.randint(len(tr_ix), (BATCH,), generator=g)]
            out, x = run(net, cur, bp[ix].to(DEV)), tgt[ix].to(DEV).float()
            loss = per_number(cur, out, x).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            if s % 5000 == 0:
                print(k, cur, s, float(loss), flush=True)
        net.eval()
        heads[cur] = net
        with torch.no_grad():
            lh, ls = 0., 0.
            shuf = te_ix[torch.randperm(len(te_ix), generator=torch.Generator().manual_seed(5))]
            step = 8192 if cur == "plain" else 1024                       # plain: round 1's batching, for reproduction
            for a in range(0, len(te_ix), step):
                ix, jx = te_ix[a:a + step], shuf[a:a + step]
                x = tgt[ix].to(DEV).float()
                lh += float(per_number(cur, run(net, cur, bp[ix].to(DEV)), x).mean()) * len(ix)
                ls += float(per_number(cur, run(net, cur, bp[jx].to(DEV)), x).mean()) * len(ix)
            res[f"heldout/{cur}"] = {"head": lh / len(te_ix), "shuffled_belief": ls / len(te_ix)}
        print(k, cur, json.dumps(res[f"heldout/{cur}"]), flush=True)
    r1 = json.loads((G.HERE / f"readings/ckpt_{k:02d}.json").read_text())["heldout/plain"]
    res["round1_plain_reproduction"] = {"round1": r1["head"], "now": res["heldout/plain"]["head"],
                                        "shuffled_round1": r1["shuffled_belief"], "shuffled_now": res["heldout/plain"]["shuffled_belief"]}
    del tgt, bp
    for prm in model.parameters():
        prm.requires_grad_(True)
    params = [x for x in model.belief.parameters() if x.requires_grad]
    pre = preconditioner(model, cfg, ck, params)
    grad = lambda f, keep=True: flat(torch.autograd.grad(f, params, retain_graph=keep, allow_unused=True), params)
    rows = {name: TJ.pair_rows(name) for name in SETS}
    norms = {name: [emb_diff_norm(model, cfg, r) for r in rows[name]] for name in ("energy", "hunger")}
    med = {name: float(np.median(v)) for name, v in norms.items()}
    res["input_difference_norm"] = {"median": med, "per_pair": norms}
    fold_sum = {}
    for name, slot in SETS.items():
        per = {cur: [] for cur in CURS}
        for n_, row in enumerate(rows[name]):
            fold = n_ % 2
            x, xb = due_targets(model, row)
            y = history(model, cfg, row)
            for cur in CURS:
                out = run(heads[cur], cur, y)
                loss = L(cur, out[0], x[0]) + L(cur, out[1], x[1])
                sep = L(cur, out[1], x[0]) + L(cur, out[0], x[1]) - loss
                dig = L(cur, out[0], x[0], slot) + L(cur, out[1], x[1], slot)
                gl, gd, gs = grad(loss), grad(dig), grad(sep)
                if name in MATCHED:
                    fold_sum.setdefault((name, cur), [torch.zeros_like(gl), torch.zeros_like(gl)])[fold] += gl
                with torch.no_grad():
                    cons = sum(L(cur, out[m], x[0]) - L(cur, out[m], x[1]) for m in (0, 1)) / 2
                    foil = {str(a): float(sum(L(cur, out[m], xb[a]) - L(cur, out[m], x[1]) for m in (0, 1)) / 2) for a in FOILS}
                    est = estimate(cur, out)
                    dx = (x[0] - x[1])[slot]
                    kappa = float(((est[0] - est[1])[slot] * dx).sum() / dx.square().sum().clamp(min=1e-12))
                per[cur].append({"consequence": float(cons), "foil": foil, "oracle_consequence": oracle(cur, x[0], x[1]),
                                 "oracle_foil": {str(a): oracle(cur, xb[a], x[1]) for a in FOILS},
                                 "cosine": cosine(gs, gl), "cosine_digit_cell": cosine(gs, gd), "grad_sep_norm": float(gs.norm()),
                                 "grad_loss_norm": float(gl.norm()), "grad_digit_loss_norm": float(gd.norm()),
                                 "sep": float(sep.detach()), "kappa": kappa})
            if name == "recovery":
                loss_r, sep_r = CR.head_terms(model, y[:, None], "reward", row, slot)
                gl_r, gs_r = grad(loss_r), grad(sep_r)
                fold_sum.setdefault((name, "reward"), [torch.zeros_like(gl_r), torch.zeros_like(gl_r)])[fold] += gl_r
                per.setdefault("reward", []).append({"cosine": cosine(gs_r, gl_r), "sep": float(sep_r.detach())})
            del y
        for cur, P in per.items():
            summ = {"pairs": len(P), "cosine_mean_ci": G.boot_mean([e["cosine"] for e in P]),
                    "cosine_positive": int(sum(e["cosine"] > 0 for e in P))}
            if cur != "reward":
                summ.update({
                    "cosine_digit_cell_mean_ci": G.boot_mean([e["cosine_digit_cell"] for e in P]),
                    "foil_beats_consequence": {a: int(sum(e["foil"][a] >= e["consequence"] for e in P)) for a in map(str, FOILS)},
                    "oracle_foil_beats_consequence": {a: int(sum(e["oracle_foil"][a] >= e["oracle_consequence"] for e in P)) for a in map(str, FOILS)},
                    "consequence_median": float(np.median([e["consequence"] for e in P])),
                    "grad_sep_norm_median": float(np.median([e["grad_sep_norm"] for e in P])),
                    "kappa_mean": float(np.mean([e["kappa"] for e in P]))})
            res[f"{name}/{cur}"] = summ | {"per_pair": P}
            print(k, name, cur, json.dumps(summ), flush=True)
    # matched update, unit step, in plain-gradient (_sgd) and AdamW geometry: held-out = the other fold's summed loss gradient, in-pair = the pair's
    # own; read through each loss's own head as d kappa (the fraction of the true due / not-due contrast reproduced) and,
    # for reference, d log D (raw belief distance; failed its reference check in the smoke, PLAN.md round 2)
    unit = {key: [-(s_ * pre) / (s_ * pre).norm().clamp(min=1e-30) for s_ in v] for key, v in fold_sum.items()}
    unit_sgd = {key: [-s_ / s_.norm().clamp(min=1e-30) for s_ in v] for key, v in fold_sum.items()}
    step = lambda g_: -(g_ * pre) / (g_ * pre).norm().clamp(min=1e-30)
    step_sgd = lambda g_: -g_ / g_.norm().clamp(min=1e-30)
    del fold_sum
    for name in MATCHED:
        slot, eff = SETS[name], {}
        for n_, row in enumerate(rows[name]):
            fold = n_ % 2
            x, _ = due_targets(model, row)
            y = history(model, cfg, row)
            gD = grad(log_distance(y))
            terms = {}
            for cur in CURS:
                out = run(heads[cur], cur, y)
                est = estimate(cur, out)
                dx = (x[0] - x[1])[slot]
                terms[cur] = (L(cur, out[0], x[0]) + L(cur, out[1], x[1]),
                              ((est[0] - est[1])[slot] * dx).sum() / dx.square().sum().clamp(min=1e-12))
            if name == "recovery":
                i, e, prefix, frames, rewards, terms_, q, truth = row
                h = model.outcome.features(y[:, None], None)
                r = torch.as_tensor(rewards[:, q], device=DEV, dtype=torch.float32)
                lp = lambda m, j: model.outcome.reward_logp(h[m:m + 1], r[j].view(1, 1)).sum()
                terms["reward"] = (-(lp(0, 0) + lp(1, 1)), lp(0, 0).exp() - lp(1, 0).exp())
            for cur, (loss, kap) in terms.items():
                gk, gl = grad(kap), grad(loss)
                d = eff.setdefault(cur, {"kappa_heldout_sgd": [], "kappa_in_pair_sgd": [], "kappa_heldout": [],
                                         "kappa_in_pair": [], "log_distance_heldout": []})
                d["kappa_heldout_sgd"].append(float(gk @ unit_sgd[(name, cur)][1 - fold]))
                d["kappa_in_pair_sgd"].append(float(gk @ step_sgd(gl)))
                d["kappa_heldout"].append(float(gk @ unit[(name, cur)][1 - fold]))
                d["kappa_in_pair"].append(float(gk @ step(gl)))
                d["log_distance_heldout"].append(float(gD @ unit[(name, cur)][1 - fold]))
            del y
        res[f"matched/{name}"] = {cur: {m: {"mean_ci": G.boot_mean(v), "positive": int(sum(z > 0 for z in v)), "per_pair": v}
                                        for m, v in d.items()} for cur, d in eff.items()}
        res[f"matched/{name}"]["categorical_minus_plain"] = {
            m: G.boot_mean(np.subtract(eff["categorical"][m], eff["plain"][m])) for m in eff["plain"]}
        print(k, "matched", name, json.dumps({c: {m: res[f"matched/{name}"][c][m]["mean_ci"] for m in eff[c]} for c in eff}),
              "cat-plain", json.dumps(res[f"matched/{name}"]["categorical_minus_plain"]), flush=True)
    # D2: input substitution on the pairs
    variants = {"hunger/amplified": ("hunger", "amplified"), "hunger/sham": ("hunger", "sham"),
                "energy/attenuated": ("energy", "attenuated"), "hunger/none": ("hunger", None), "energy/none": ("energy", None)}
    for key, (name, kind) in variants.items():
        slot, P = SETS[name], []
        for n_, row in enumerate(rows[name]):
            x, _ = due_targets(model, row)
            own = norms[name][n_]
            scale = 1.0 if kind is None else (med["energy"] if name == "hunger" else med["hunger"]) / own
            y = history(model, cfg, row, kind, scale, seed=30000 + n_)
            e = {"scale": scale, "log_distance": float(log_distance(y.detach()))}
            for cur in CURS:
                out = run(heads[cur], cur, y)
                loss = L(cur, out[0], x[0]) + L(cur, out[1], x[1])
                sep = L(cur, out[1], x[0]) + L(cur, out[0], x[1]) - loss
                gl, gs = grad(loss), grad(sep)
                e[f"{cur}/cosine"], e[f"{cur}/grad_sep_norm"] = cosine(gs, gl), float(gs.norm())
                if cur == "plain":
                    e["plain/cosine_digit_cell"] = cosine(gs, grad(L(cur, out[0], x[0], slot) + L(cur, out[1], x[1], slot)))
            del y
            P.append(e)
        summ = {"pairs": len(P), "scale_median": float(np.median([e["scale"] for e in P])),
                "distance_median": float(np.exp(np.median([e["log_distance"] for e in P])))}
        for f in ("plain/cosine", "categorical/cosine", "plain/cosine_digit_cell"):
            summ[f + "_mean_ci"] = G.boot_mean([e[f] for e in P])
        for f in ("plain/grad_sep_norm", "categorical/grad_sep_norm"):
            summ[f + "_median"] = float(np.median([e[f] for e in P]))
        res[f"d2/{key}"] = summ | {"per_pair": P}
        print(k, "d2", key, json.dumps(summ), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1) + "\n")
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    OUT.mkdir(parents=True, exist_ok=True)
    for k in (map(int, sys.argv[1:]) if len(sys.argv) > 1 else (1, 2, 10)):
        checkpoint(k)
