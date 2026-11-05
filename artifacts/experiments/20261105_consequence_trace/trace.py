"""Stage B (PLAN.md): where each failing consequence stops in a frozen factorised world. One GPU pass per world.

Classes are functions of the state at t - 1 and a_{t-1} (dev labels, game_logic rules; RESEARCH.md section 1), never of
the outcome: COAL / IRON / WOOD (awake, DO, faced block, tool, count <= 8, no mob on the faced cell), DRINK (awake, DO,
faced water, drink <= 8), and field_calibration's RISKY / ADJACENT / SAFE (10-29 A); RANDOM frames for B4 only.
Functional: catalogue's linear f (10-29 F), per part: the HUD count cell (start value -> consequence value, mean real
tokens) and, for mining, the faced map cell (its token at t - 1 -> the mean real token of the block left behind there).
B0  consequence size |b - a| per part; a declared SNR threshold sigma_d2 = |b - a| / 2 (d' = 2: still ~84% two-mode
    accuracy, not absence of information) and |b - a| / sigma at the first level; draws / surrogate objective above it.
B1  readers (information_probe's protocol, 10-29 E): full conditioning; predicted belief + action + fixation (the memory
    route's source); frame t - 1's raw percept + action (the projection's input) vs its percept embedding + the same
    action (the projection's output); the 8 context frames' COARSE tokens + action (what the physical tokens can read
    directly: never the glimpse); the rule's inputs. One-sided: a weak reader is not proof of absence.
B2  inference: the field across noise levels (real successor + noise; law of total expectation, 10-29 A / F), the field
    at the production decision point (the cascade's first evaluation from pure noise, asserted bitwise equal to the
    sampler's own) beside the exact two-outcome field's value for the same pure-noise input (Codex 11-05: an exact
    denoiser need not reproduce the class rate from pure noise when p < 1), and the production samples from that start.
B3  routes, same draws: beliefs recomputed (control), NOOP at t - 1 in both action channels (in support), and both with
    the 8 context frames entered at sigma 5 (a stress test, not a route identification).
B4  the ACTUAL P1 G1 term (model.denoise_loss's computation, chunked training forward over 8 context frames + the target,
    every frame its own independent physical and glimpse sigma ~ p_train, unclipped raw output vs its preconditioned
    target): the consequence cells' share on their event frames by the target's physical sigma, and on RANDOM frames the
    allocation of the whole term by band, as trained and as an x-space loss would weight the same errors.
Usage (project venv, GPU): python -m artifacts.experiments.20261105_consequence_trace.trace OUT_DIR CKPT [N_EPISODES]
    [KINDS]  (comma list: only those classes, no B1 readers; stone added 11-05, PLAN "Stage B2")
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")                           # no JAX on the GPU, whatever imports craftax
import copy, importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial.hermite import hermgauss
from scipy.special import expit

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
CAT = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.catalogue")
IPB = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.information_probe")
IP = importlib.import_module("artifacts.experiments.20261030_increment_diagnosis.increment_probe")
NP = importlib.import_module("artifacts.experiments.20261105_consequence_trace.noise_profile")
ROOT = Path(__file__).resolve().parents[3]
DEV, P, R, SEED, K_OBJ = FC.DEV, FC.P, 2, 2105, 4                      # K_OBJ: B4 draws of (sigmas, noise) per row
GRASS, WATER, STONE, TREE, PATH, COAL, IRON, DO = 2, 3, 4, 5, 7, 8, 9, 5  # constants.BlockType / Action
HUD = {63: ("s_body", 0), 65: ("s_body", 2), 68: ("s_inv", 0), 69: ("s_inv", 1), 70: ("s_inv", 2), 71: ("s_inv", 3)}  # renderer
MINING = {"coal": (COAL, PATH, 70, 6), "iron": (IRON, PATH, 71, 7), "wood": (TREE, GRASS, 68, None),  # block, left,
          "stone": (STONE, PATH, 69, 6)}                                                           # cell, tool
KINDS = ("coal", "iron", "wood", "drink", "risky", "adjacent", "safe", "random", "stone")
CAP = {"wood": 1500, "drink": 1500, "risky": 2000, "adjacent": 2000, "safe": 500, "random": 1000, "stone": 1500}  # last:
                                                                         # earlier classes' draws unchanged
VARIANTS = ("base", "rerun", "noop", "ctx5", "noop_ctx5")
BANDS = [(0, 0.3), (0.3, 1), (1, 3), (3, 7), (7, np.inf)]               # RESEARCH.md section 4's surrogate bands
CTX_SIGMA, MIN_ROWS = 5.0, 200
FRAME_P, FRAME_G, FRAME_FIX, FRAME_V = 1296, 576, 49, 8                 # s1.frame_features layout (then reward, term,
                                                                         # previous action 17, predicted belief 256, valid)


def mob_at(L, g, cell):
    return any((L[f"s_{k}_m"][g] & (L[f"s_{k}_p"][g].astype(int) == cell).all(-1)).any()
               for k in ("zombies", "cows", "skeletons", "arrows"))


def contexts(L, enc, rng):
    """{kind: [(e, t, hud cell, start, end, ready)]} (mining rows' faced cell comes from s_dir at t - 1) and the
    uncapped count per kind."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    out = {k: [] for k in KINDS}
    for e in range(len(enc)):
        for t in range(P, len(enc[e]["p"])):
            g, a = s_off[e] + t - 1, int(L["t_action"][a_off[e] + t - 1])
            if L["s_sleep"][g] or a != DO or mob_at(L, g, L["s_pos"][g].astype(int) + d[g]):
                continue
            for kind, (block, _, cell, tool) in MINING.items():
                v = int(L["s_inv"][g, HUD[cell][1]])
                if face[g] == block and v <= 8 and (tool is None or L["s_inv"][g, tool] >= 1):
                    out[kind].append((e, t, cell, v, v + 1, False))
            if face[g] == WATER and L["s_body"][g, 2] <= 8:
                out["drink"].append((e, t, 65, int(L["s_body"][g, 2]), int(L["s_body"][g, 2]) + 1, False))
    saved = dict(FC.N_MAX)
    FC.N_MAX.update({k: None for k in saved})                            # uncapped there (true counts), CAP below
    try:
        fc = FC.contexts(L, enc, rng)
    finally:
        FC.N_MAX.update(saved)
    for kind in ("risky", "adjacent", "safe", "random"):
        out[kind] = fc[kind]
    counts = {k: len(v) for k, v in out.items()}
    for kind, n in CAP.items():
        if len(out[kind]) > n:
            out[kind] = [out[kind][i] for i in np.sort(rng.choice(len(out[kind]), n, replace=False))]
    return out, counts


def means(L, enc):
    """Mean real periphery token per (HUD cell, value) (health <= 0 pooled as 0) and per (faced cell, block)."""
    faced = [(3 + dr) * 9 + 4 + dc for dr, dc in FC.DIRS[1:]]
    cells = list(HUD) + faced
    p = torch.cat([e["p"][:, cells].float() for e in enc])
    n = len(p)
    assert n == int((L["ep_T"] + 1).sum())
    hud = {}
    for j, (cell, (arr, col)) in enumerate(HUD.items()):
        v = L[arr][:n, col].clip(min=0)
        hud[cell] = {int(k): p[v == k, j].mean(0) for k in np.unique(v)}
    blocks = {}
    for j, ((dr, dc), cell) in enumerate(zip(FC.DIRS[1:], faced)):
        v = L["s_view"][:n, 3 + dr, 4 + dc]
        blocks.update({(cell, b): p[v == b, len(HUD) + j].mean(0) for b in (GRASS, PATH) if (v == b).any()})
    return hud, blocks


def functional(kind, part, batch, enc, L, hud, blocks):
    """A, W [B, 81, 16] with f(y) = ((y - A) * W).sum() / |W|^2 (catalogue); the separation |b - a|, |a|^2 and |b|^2
    over the part's cells (for the exact two-outcome reference), and validity."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    A, W = torch.zeros(len(batch), 81, 16), torch.zeros(len(batch), 81, 16)
    for i, (e, t, cell, start, end, _) in enumerate(batch):
        if part == "hud":
            if start in hud[cell] and end in hud[cell]:
                A[i, cell], W[i, cell] = hud[cell][start], hud[cell][end] - hud[cell][start]
        else:
            g = s_off[e] + t - 1
            dd = FC.DIRS[L["s_dir"][g]]
            c = (3 + dd[0]) * 9 + 4 + dd[1]
            prev = enc[e]["p"][t - 1, c].float()
            if (c, MINING[kind][1]) in blocks:
                A[i, c], W[i, c] = prev, blocks[(c, MINING[kind][1])] - prev
    sep2 = W.square().sum((-1, -2))
    a2, b2 = A.square().sum((-1, -2)), (A + W).square().sum((-1, -2))
    return A, W / sep2.clamp(min=1e-9)[:, None, None], sep2.sqrt(), a2, b2, sep2 >= CAT.MIN_SEPARATION


def inputs(model, cfg, enc, batch, variant):
    """Stacked denoiser inputs per context: 8 context frames, fixations / previous actions / predicted beliefs t - 8 ..
    t, the real frame t. base: stored beliefs; rerun / noop: increment_probe.row_inputs (belief recomputed from the
    stored tokens; noop: NOOP at t - 1 in the belief AND in the denoiser's own action input)."""
    rows = []
    for e, t, *_ in batch:
        if variant in ("rerun", "noop"):
            rows.append(IP.row_inputs(model, cfg, enc[e], t, {"rerun": True, "noop": variant == "noop"}, None))
        else:
            ep = enc[e]
            rows.append({"p": ep["p"][t - P:t].float(), "g": ep["g"][t - P:t].float(), "v": ep["v"][t - P:t].float(),
                         "fix": ep["fix"][t - P:t + 1], "pa": ep["prev_action"][t - P:t + 1], "bp": ep["bp"][t - P:t + 1].float(),
                         "x0p": ep["p"][t].float(), "x0g": ep["g"][t].float(), "x0v": ep["v"][t].float()})
    return {k: torch.stack([r[k] for r in rows]).repeat_interleave(R, 0).to(DEV) for k in rows[0]}


def schedule(cfg):
    """sample()'s noise levels and churn factors, computed exactly as there."""
    s, n = cfg["edm"]["sampler"], cfg["edm"]["sampler"]["steps"]
    i = torch.arange(n, dtype=torch.float64)
    sig = (s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
    sig = torch.cat([sig, sig.new_zeros(1)]).tolist()
    return sig, [min(s["s_churn"] / n, math.sqrt(2) - 1) if s["s_tmin"] <= sig[k] <= s["s_tmax"] else 0.0 for k in range(n)]


def first_call(den, shapes, a, mem, fx, cache, cfg, gen):
    """The cascade's first evaluation (model._cascade, k = 0): periphery and lighting from sigma_max-scaled noise,
    churned to sigma-hat, the glimpse held at its zero placeholder; same draws in the same order."""
    sig, gam = schedule(cfg)
    s, B = cfg["edm"]["sampler"], mem.shape[0]
    randn = lambda shp: torch.randn(shp, device=DEV, generator=gen)
    p, v = sig[0] * randn((B, 1, *shapes[0])), sig[0] * randn((B, 1, *shapes[2]))
    hat = sig[0] * (1 + gam[0])
    if gam[0] > 0:
        p, v = (t + math.sqrt(hat ** 2 - sig[0] ** 2) * s["s_noise"] * randn(t.shape) for t in (p, v))
    g0 = torch.zeros((B, 1, *shapes[1]), device=DEV)
    D, _ = M.denoised(den, (p, g0, v), torch.full((B, 1), hat, device=DEV), a, mem, fx, cache,
                      cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"])
    return D


@torch.no_grad()
def run(model, cfg, inp, fs, seed, levels=None, check=False):
    """One variant on one batch (inference): context passes, the first evaluation, the production sample from the same
    start (generator state restored) and, with `levels`, the field on the real successor + noise at every level (the
    clean estimate the sampler uses). fs: {part: f(y [B, 81, 16])}."""
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    cache = den.new_cache()
    out = {}
    with TR.autocast(DEV, cfg):
        for k in range(P):
            x = tuple(inp[n][:, k][:, None] for n in ("p", "g", "v"))
            cache = M.context_pass(den, x, inp["pa"][:, k][:, None], model.belief.memory(inp["bp"][:, k])[:, None],
                                   inp["fix"][:, k][:, None], cache, cfg, gen)
        a, mem, fx = inp["pa"][:, P][:, None], model.belief.memory(inp["bp"][:, P])[:, None], inp["fix"][:, P][:, None]
        x0 = (inp["x0p"], inp["x0g"], inp["x0v"])
        shapes = [tuple(t.shape[1:]) for t in x0]
        state = gen.get_state()
        D1 = first_call(den, shapes, a, mem, fx, cache, cfg, gen)
        gen.set_state(state)
        seen, orig = [], M.denoised
        M.denoised = lambda *q, **kw: seen.append(orig(*q, **kw)) or seen[-1]
        try:
            xg = M.sample(den, shapes, a, mem, fx, cache, cfg, gen)
        finally:
            M.denoised = orig
        if check:                                                        # the decision point IS the sampler's first call
            assert all(torch.equal(u, w) for u, w in zip(D1, seen[0][0])), "first_call differs from sample()'s first evaluation"
        for part, f in fs.items():
            out[f"first_{part}"], out[f"prod_{part}"] = f(D1[0][:, 0]), f(xg[0][:, 0])
        if levels is not None:
            g2 = torch.Generator(device=DEV).manual_seed(seed + 1)
            for part, f in fs.items():
                out[f"real_{part}"] = f(x0[0])
            for s_ in levels:
                xs = tuple(t[:, None] + s_ * torch.randn(t[:, None].shape, device=DEV, generator=g2) for t in x0)
                D, _ = M.denoised(den, xs, torch.full((len(x0[0]), 1), s_, device=DEV), a, mem, fx, cache, clamp, sd)
                for part, f in fs.items():
                    out.setdefault(f"field_{part}", []).append(f(D[0][:, 0]))
            for k in [k for k in out if isinstance(out[k], list)]:
                out[k] = torch.stack(out[k], 1)
    return out


@torch.no_grad()
def objective(model, cfg, inp, mask, seed):
    """B4: model.denoise_loss's computation on [8 context frames + target] windows, element-wise at the target. Every
    frame its own physical and glimpse sigma (TR.sample_sigma, independent: factorised P1), fresh noise, the chunked
    training forward, the unclipped raw output against (x0 - c_skip x) / c_out. Per row and draw: the target's two
    sigmas, the raw loss of the consequence cells (mask [B, 81, 1]), periphery, glimpse and lighting, and c_out^2 of
    each part (raw x c_out^2 = the x-space error)."""
    den, sd = model.denoiser, cfg["edm"]["sigma_data"]
    g = torch.Generator(device=DEV).manual_seed(seed)
    x0 = tuple(torch.cat([inp[k], inp[f"x0{k}"][:, None]], 1) for k in ("p", "g", "v"))      # [B, P + 1, ...]
    B, T = x0[0].shape[:2]
    mem = model.belief.memory(inp["bp"])
    ep, valid = torch.zeros(B, T, dtype=torch.long, device=DEV), torch.ones(B, T, dtype=torch.bool, device=DEV)
    out = {k: [] for k in ("sig_p", "sig_g", "cells", "peri", "glim", "light", "co_p", "co_g")}
    with TR.autocast(DEV, cfg):
        for _ in range(K_OBJ):
            sigma = (TR.sample_sigma((B, T), cfg, DEV, g), TR.sample_sigma((B, T), cfg, DEV, g))
            noise = tuple(torch.randn(t.shape, device=DEV, generator=g) for t in x0)
            pre = [M.precondition(s_, sd) for s_ in M._parts(sigma)]
            xs = tuple(t + M._bcast(s_, t) * n for t, n, s_ in zip(x0, noise, M._parts(sigma)))
            Fo = den(tuple(M._bcast(c[2], t) * t for t, c in zip(xs, pre)), (pre[0][3], pre[1][3]), inp["pa"], mem,
                     inp["fix"], ep, valid)
            se = [(f.float() - (t - M._bcast(c[0], t) * s_) / M._bcast(c[1], t))[:, -1].square()
                  for f, t, s_, c in zip(Fo, x0, xs, pre)]
            for k, v in (("sig_p", sigma[0][:, -1]), ("sig_g", sigma[1][:, -1]), ("cells", (se[0] * mask).sum((1, 2))),
                         ("peri", se[0].sum((1, 2))), ("glim", se[1].sum((1, 2))), ("light", se[2].sum(1)),
                         ("co_p", pre[0][1][:, -1].square()), ("co_g", pre[1][1][:, -1].square())):
                out[k].append(v.float().cpu())
    return {k: torch.stack(v, 1) for k, v in out.items()}                 # [B, K_OBJ]


def measure(model, cfg, enc, L, rows, kind, hud, blocks, levels):
    parts = ("hud", "cell") if kind in MINING else ("hud",) if kind != "random" else ()
    cfg5 = copy.deepcopy(cfg)
    cfg5["edm"]["inference_context_sigma"] = CTX_SIGMA
    res = {v: {} for v in VARIANTS + ("obj",)}
    extra = {k: [] for k in ("valid", "sep", "a2", "b2", "row_ctx")}
    n_ctx = 128 // R
    for b, start in enumerate(range(0, len(rows), n_ctx)):
        batch = rows[start:start + n_ctx]
        idx = torch.arange(len(batch)).repeat_interleave(R)
        extra["row_ctx"].append(start + idx)
        fw = {part: functional(kind, part, batch, enc, L, hud, blocks) for part in parts}
        fs, mask = {}, torch.zeros(len(idx), 81, 1, device=DEV)
        for part, (A, W, *_r) in fw.items():
            A_, W_ = A[idx].to(DEV), W[idx].to(DEV)
            fs[part] = lambda y, A_=A_, W_=W_: ((y.float() - A_) * W_).sum((-1, -2)).cpu()
            mask = torch.maximum(mask, (W_.abs().sum(-1, keepdim=True) > 0).float())
        if parts:
            for k, j in (("sep", 2), ("a2", 3), ("b2", 4)):
                extra[k].append(torch.stack([fw[p_][j][idx] for p_ in parts], 1))
            extra["valid"].append(torch.stack([fw[p_][5][idx] for p_ in parts], 1).all(1))
        else:
            extra["valid"].append(torch.ones(len(idx), dtype=torch.bool))
        cached = {"base": inputs(model, cfg, enc, batch, "base")}
        for k, v in objective(model, cfg, cached["base"], mask, SEED + 7919 * b).items():
            res["obj"].setdefault(k, []).append(v)
        if not parts:                                                    # RANDOM: B4 only
            continue
        cached.update({v: inputs(model, cfg, enc, batch, v) for v in ("rerun", "noop")})
        for variant in VARIANTS:                                         # same seed: every variant gets the same draws
            inp = cached[variant.replace("_ctx5", "") if variant != "ctx5" else "base"]
            o = run(model, cfg5 if variant.endswith("ctx5") else cfg, inp, fs, SEED + b, levels if variant == "base" else None,
                    check=b == 0)
            for k, v in o.items():
                res[variant].setdefault(k, []).append(v)
    flat = {f"{v}_{k}": torch.cat(x).numpy() for v, d in res.items() for k, x in d.items()}
    return {**flat, **{k: torch.cat(v).numpy() for k, v in extra.items() if v}}


def exact_pure(p, sep, a2, b2, sigma):
    """The exact two-outcome conditional mean (outcomes a, b with prior p, Gaussian noise) averaged over a pure-noise
    input x = sigma eps, read through f: E_z sigmoid(logit p + |b - a| z / sigma - (|b|^2 - |a|^2) / (2 sigma^2)).
    Codex's REVIEW.md control in the measured geometry (|a| and |b| are the real tokens' norms: pure noise is centred
    on the origin, not on either outcome). For p -> 1 it is 1 whatever the input."""
    z, w = hermgauss(80)
    z, w = z * math.sqrt(2), w / math.sqrt(math.pi)
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    lo = math.log(p / (1 - p)) + sep[None] * z[:, None] / sigma - (b2 - a2)[None] / (2 * sigma ** 2)
    return (w[:, None] * expit(lo)).sum(0)


def objective_summary(o, sig_d2=None):
    """B4 shares from the training-form draws (raw = as trained; x = the same errors weighted as an x-space loss)."""
    q = {k: o[f"obj_{k}"].astype(float) for k in ("sig_p", "sig_g", "cells", "peri", "glim", "light", "co_p", "co_g")}
    raw = {"phys": q["peri"] + q["light"], "glim": q["glim"], "cells": q["cells"]}
    xsp = {"phys": raw["phys"] * q["co_p"], "glim": raw["glim"] * q["co_g"], "cells": raw["cells"] * q["co_p"]}
    out = {}
    for name, L_ in (("as_trained", raw), ("x_space_weighting", xsp)):
        total = L_["phys"].sum() + L_["glim"].sum()
        band = lambda sg, a, b: (sg >= a) & (sg < b)
        r = {"whole_term_by_band": {f"{a}-{b}": float((L_["phys"][band(q["sig_p"], a, b)].sum()
                                                       + L_["glim"][band(q["sig_g"], a, b)].sum()) / total) for a, b in BANDS},
             "glimpse_share": float(L_["glim"].sum() / total)}
        if L_["cells"].sum() > 0:
            r["cells_share_of_term"] = float(L_["cells"].sum() / total)
            r["cells_share_of_physical_by_band"] = {f"{a}-{b}": float(L_["cells"][band(q["sig_p"], a, b)].sum()
                                                                      / max(L_["phys"][band(q["sig_p"], a, b)].sum(), 1e-12)) for a, b in BANDS}
            if sig_d2 is not None:
                r["cells_loss_above_sigma_d2"] = float(L_["cells"][q["sig_p"] >= sig_d2].sum() / L_["cells"].sum())
        out[name] = r
    out["x_space_scale_matching_constant"] = float((raw["phys"].sum() + raw["glim"].sum()) / (xsp["phys"].sum() + xsp["glim"].sum()))
    return out


def summary(o, rows, kind, L, levels, first, s_prof, e_prof):
    ep = np.array([rows[i][0] for i in o["row_ctx"]])
    valid = o["valid"].astype(bool)
    est = lambda x, m=None: G.estimate(np.asarray(x, float), valid if m is None else valid & m, ep)
    if kind == "random":
        return {"contexts": len(rows), "objective": objective_summary(o)}
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    g = np.array([s_off[rows[i][0]] + rows[i][1] - 1 for i in o["row_ctx"]])
    cell = np.array([rows[i][2] for i in o["row_ctx"]]); start = np.array([rows[i][3] for i in o["row_ctx"]])
    end = np.array([rows[i][4] for i in o["row_ctx"]]); ready = np.array([rows[i][5] for i in o["row_ctx"]], bool)
    after = np.array([L[HUD[c][0]][gg + 1, HUD[c][1]] for c, gg in zip(cell, g)]).clip(min=0)
    parts = ("hud", "cell") if kind in MINING else ("hud",)
    p_class = float((after == end)[valid].mean())                       # recorded event rate of the class
    jf = levels.index(first)
    res = {"contexts": len(rows), "rows_valid": int(valid.sum()), "recorded_reached_end": est(after == end),
           "recorded_kept_start": est(after == start), "geometry": {}, "parts": {}}
    for k, part in enumerate(parts):
        sep = o["sep"][:, k]
        d2 = float(np.median(sep[valid])) / 2
        res["geometry"][part] = {"separation_median": 2 * d2, "sigma_d2": d2,
                                 "separation_over_first_level_median": float(np.median(sep[valid]) / first),
                                 "above_sigma_d2_surrogate": NP.above(d2, s_prof, e_prof)}
        f = lambda v, q: o[f"{v}_{q}_{part}"]
        ref = exact_pure(p_class, sep, o["a2"][:, k], o["b2"][:, k], first)
        r = {"f_recorded": est(f("base", "real")), "first_call": est(f("base", "first")),
             "exact_two_outcome_first_call": est(ref), "first_call_minus_exact": est(f("base", "first") - ref),
             "production": est(f("base", "prod")), "sampler": est(f("base", "prod") - f("base", "first")),
             "real_plus_noise_minus_pure_at_first_level": est(f("base", "field")[:, jf] - f("base", "first")),
             "rerun_control": est(f("rerun", "first") - f("base", "first")),
             "noop": {"first_call": est(f("noop", "first")), "production": est(f("noop", "prod")),
                      "action_contrast": est(f("rerun", "first") - f("noop", "first"))},
             "context_sigma_5_stress": {"first_call": est(f("ctx5", "first")), "production": est(f("ctx5", "prod")),
                                        "noop_first_call": est(f("noop_ctx5", "first")), "noop_production": est(f("noop_ctx5", "prod")),
                                        "action_contrast": est(f("ctx5", "first") - f("noop_ctx5", "first")),
                                        "contrast_change": est(f("ctx5", "first") - f("noop_ctx5", "first")
                                                               - f("rerun", "first") + f("noop", "first"))},
             "field": {str(s): {"mean": est(f("base", "field")[:, j]), "bias": est(f("base", "field")[:, j] - f("base", "real"))}
                       for j, s in enumerate(levels)}}
        if ready.any() and (~ready).any():                               # descriptive: the cooldown is hidden
            r["cooldown_descriptive"] = {name: {"rows": int((valid & m).sum()), "f_recorded": est(f("base", "real"), m),
                                                "first_call": est(f("base", "first"), m), "production": est(f("base", "prod"), m),
                                                "noop_first_call": est(f("noop", "first"), m), "noop_production": est(f("noop", "prod"), m)}
                                         for name, m in (("ready", ready), ("not_ready", ~ready))}
        res["parts"][part] = r
    res["p_class"] = p_class
    res["objective"] = objective_summary(o, min(res["geometry"][p_]["sigma_d2"] for p_ in parts))
    return res


def populations(L, enc):
    """B1 rows (e, t, label, rule inputs): IPB's attack and drink rows; COAL / IRON: awake, faced block, count <= 8 (any
    action); label count + 1 at t; rule (game_logic.do_action): DO, tool, no mob on the faced cell."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    out = {"attack": [(r[0], r[1], r[2], r[3]) for r in IPB.population(L, enc)],
           "drink": [(e, t, y, np.array([face[s_off[e] + t - 1] == WATER], np.float32)) for e, t, y in IPB.drink_population(L, enc)]}
    for kind in ("coal", "iron"):
        block, _, cell, tool = MINING[kind]
        col, rows = HUD[cell][1], []
        for e in range(len(enc)):
            for t in range(IPB.HISTORY, len(enc[e]["p"])):
                g = s_off[e] + t - 1
                if L["s_sleep"][g] or face[g] != block or L["s_inv"][g, col] > 8:
                    continue
                a = int(L["t_action"][a_off[e] + t - 1])
                rule = np.array([a == DO, L["s_inv"][g, tool] >= 1, not mob_at(L, g, L["s_pos"][g].astype(int) + d[g])], np.float32)
                rows.append((e, t, bool(L["s_inv"][g + 1, col] == L["s_inv"][g, col] + 1), rule))
        out[kind] = rows
    return out


@torch.no_grad()
def embeddings(model, cfg, enc, rows):
    """Frame t - 1's percept embedding (Belief.percept_embed: the 6,569 -> 256 projection) per row."""
    out = []
    for i in range(0, len(rows), 4096):
        ch = rows[i:i + 4096]
        st = lambda k: torch.stack([enc[e][k][t - 1] for e, t, *_ in ch]).to(DEV)
        with TR.autocast(DEV, cfg):
            out.append(model.belief.percept_embed(M.Percept(st("p").float(), st("g").float(), st("v").float(), st("fix"))).float().cpu())
    return torch.cat(out).numpy()


def reader_inputs(X, emb):
    """The B1 arms, sliced from IPB.features' layout (per history frame t - 8 .. t - 1: periphery, glimpse, fixation,
    lighting, reward, termination [zeroed], previous action, predicted belief, valid; then the current predicted
    belief, a_{t-1}, the target fixation). Raw / embedding / coarse arms carry NO predicted belief (Codex 11-05)."""
    H, Fw = IPB.HISTORY, IPB.FRAME
    n_last = X.shape[1] - (256 + 17 + 49)
    act = X[:, n_last + 256:n_last + 256 + 17]
    percept = FRAME_P + FRAME_G + FRAME_FIX + FRAME_V
    last = (H - 1) * Fw
    coarse = [np.concatenate([X[:, h * Fw:h * Fw + FRAME_P], X[:, h * Fw + percept - FRAME_V:h * Fw + percept]], 1) for h in range(H)]
    return {"a_full_conditioning": X, "b_belief_action_fixation": X[:, n_last:],
            "c_raw_frame_action": np.concatenate([X[:, last:last + percept], act], 1),
            "e_embedding_action": np.concatenate([emb, act], 1),
            "f_coarse_history_action": np.concatenate(coarse + [act], 1)}


def readers(model, cfg, enc, L):
    res = {}
    for name, rows in populations(L, enc).items():
        y, ep = np.array([r[2] for r in rows]), np.array([r[0] for r in rows])
        if len(rows) < MIN_ROWS or y.all() or not y.any():
            res[name] = {"rows": len(rows), "skipped": "too few rows or one class"}
            continue
        X = IPB.features(enc, rows)
        arms = {**reader_inputs(X, embeddings(model, cfg, enc, rows)), "d_rule_inputs": np.stack([r[3] for r in rows])}
        res[name] = {"rows": len(rows), "rate": float(y.mean())}
        for key, Xi in arms.items():
            res[name][key] = IPB.probe(Xi, y, ep, np.random.default_rng(SEED))[0]
            print("B1", name, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name][key].items()}, flush=True)
        del X, arms
    return res


def main():
    out_dir, ckpt = Path(sys.argv[1]), ROOT / sys.argv[2]
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    assert cfg["denoiser"].get("factorised"), "first_call reproduces the factorised cascade only"
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    s_prof, e_prof = NP.profile(ckpt=int(ckpt.stem.split("_")[1]), log=ckpt.parent / "log.jsonl")
    mon = [float(s) for s in s_prof]
    sig, gam = schedule(cfg)
    first = sig[0] * (1 + gam[0])                                        # sigma-hat of the production first evaluation
    levels = sorted(set(FC.SIGMAS) | set(mon) | {first})
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    enc = S.encode(model, L, cfg)
    a_off = np.r_[0, np.cumsum(L["ep_T"])]
    for e in range(0, len(enc), 97):                                    # labels and encoding agree on a_{t-1}
        assert (enc[e]["prev_action"][1:].numpy() == L["t_action"][a_off[e]:a_off[e + 1]]).all()
    hud, blocks = means(L, enc)
    ctx, counts = contexts(L, enc, np.random.default_rng(SEED))
    frames = int(sum(len(e["p"]) - P for e in enc))
    res = {"checkpoint": str(ckpt.relative_to(ROOT)), "checkpoint_sha256": TR.file_sha256(ckpt),
           "script_sha256": TR.file_sha256(Path(__file__)), "episodes": len(enc), "target_frames": frames,
           "levels": levels, "monitor_levels": mon, "first_level": first, "rows_per_context": R, "objective_draws": K_OBJ,
           "context_sigma_stress": CTX_SIGMA, "sampler": cfg["edm"]["sampler"], "opportunities": counts,
           "opportunity_rate": {k: n / frames for k, n in counts.items()}, "classes": {}}
    for kind in KINDS:
        rows = ctx[kind]
        if not rows or (only and kind not in only):
            continue
        o = measure(model, cfg, enc, L, rows, kind, hud, blocks, levels)
        np.savez_compressed(out_dir / f"{kind}.npz", **o, ctx=np.array(rows, dtype=np.int64))
        res["classes"][kind] = summary(o, rows, kind, L, levels, first, s_prof, e_prof)
        c = res["classes"][kind]
        if kind != "random":
            h = c["parts"]["hud"]
            print("B2", kind, len(rows), "rec", round(h["f_recorded"]["est"], 3), "first", round(h["first_call"]["est"], 3),
                  "exact", round(h["exact_two_outcome_first_call"]["est"], 3), "prod", round(h["production"]["est"], 3),
                  "noop", round(h["noop"]["first_call"]["est"], 3), flush=True)
        print("B4", kind, json.dumps(c["objective"]["as_trained"])[:300], flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    if not only:
        res["readers"] = readers(model, cfg, enc, L)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
