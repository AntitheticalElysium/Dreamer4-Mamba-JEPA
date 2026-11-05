"""Stage B (PLAN.md): where each failing consequence stops in a frozen factorised world. One GPU pass per world.

Classes are functions of the state at t - 1 and a_{t-1} (dev labels, game_logic rules; RESEARCH.md section 1), never of
the outcome: COAL / IRON / WOOD (awake, DO, faced block, tool, count <= 8, no mob on the faced cell), DRINK (awake, DO,
faced water, drink <= 8), and field_calibration's RISKY / ADJACENT / SAFE (10-29 A). Functional: catalogue's linear f
(10-29 F), per part: the HUD count cell (start value -> consequence value, mean real tokens) and, for mining, the faced
map cell (its token at t - 1 -> the mean real token of the block left behind at that screen cell).
B0  size of each consequence in token space |b - a|, sigma* = |b - a| / 2 (noise above which the input no longer
    shows it), training draws / objective above sigma* (noise_profile).
B1  readers (information_probe, 10-29 E) on what the generator receives: full conditioning, predicted belief + action
    + fixation, last frame + action, frame t - 1's percept embedding (the 256-D projection) + action, the rule's inputs.
B2  the field across noise levels (real successor + noise; law of total expectation, 10-29 A / F), the field at the
    production decision point (the cascade's first evaluation, from pure noise; asserted bitwise equal to the
    sampler's own first evaluation), and the production samples drawn from that same start.
B3  routes, same draws: beliefs recomputed (control), recomputed with a NOOP at t - 1 in both action channels (the
    in-support counterfactual), and both with the 8 context frames entered at sigma 5 instead of 0.1 (the direct
    route mostly removed, memory intact): the action contrast DO - NOOP with and without the direct context.
B4  the consequence cells' share of the denoising objective on their event frames, at the 8 equal-mass monitor levels
    of p_train (context at 0.1, not the training form's noisy context: an approximation).
Usage (project venv, GPU): python -m artifacts.experiments.20261105_consequence_trace.trace OUT_DIR CKPT [N_EPISODES]
"""
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")                           # no JAX on the GPU, whatever imports craftax
import copy, importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
CAT = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.catalogue")
IPB = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.information_probe")
IP = importlib.import_module("artifacts.experiments.20261030_increment_diagnosis.increment_probe")
NP = importlib.import_module("artifacts.experiments.20261105_consequence_trace.noise_profile")
ROOT = Path(__file__).resolve().parents[3]
DEV, P, R, SEED = FC.DEV, FC.P, 2, 2105
GRASS, WATER, TREE, PATH, COAL, IRON, DO = 2, 3, 5, 7, 8, 9, 5          # constants.BlockType / Action
HUD = {63: ("s_body", 0), 65: ("s_body", 2), 68: ("s_inv", 0), 70: ("s_inv", 2), 71: ("s_inv", 3)}   # renderer slots
MINING = {"coal": (COAL, PATH, 70, 6), "iron": (IRON, PATH, 71, 7), "wood": (TREE, GRASS, 68, None)}  # block, left, cell, tool
KINDS = ("coal", "iron", "wood", "drink", "risky", "adjacent", "safe")
CAP = {"wood": 1500, "drink": 1500, "risky": 2000, "adjacent": 2000, "safe": 500}
VARIANTS = ("base", "rerun", "noop", "ctx5", "noop_ctx5")
CTX_SIGMA, MIN_ROWS = 5.0, 200


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
    for kind in ("risky", "adjacent", "safe"):
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
    """A, W [B, 81, 16] with f(y) = ((y - A) * W).sum() / |W|^2 (catalogue), the separation |W| and validity."""
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
    sep = W.square().sum((-1, -2))
    return A, W / sep.clamp(min=1e-9)[:, None, None], sep.sqrt(), sep >= CAT.MIN_SEPARATION


def inputs(model, cfg, enc, batch, variant):
    """Stacked denoiser inputs per context: 8 context frames, fixations / previous actions / predicted beliefs t - 8 ..
    t, the real frame t. base / ctx5: stored beliefs; rerun / noop: increment_probe.row_inputs (belief recomputed from
    the stored tokens; noop: NOOP at t - 1 in the belief AND in the denoiser's own action input)."""
    rows = []
    for e, t, *_ in batch:
        if variant in ("rerun", "noop", "noop_ctx5"):
            rows.append(IP.row_inputs(model, cfg, enc[e], t, {"rerun": True, "noop": variant.startswith("noop")}, None))
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
def run(model, cfg, inp, fs, seed, levels=None, mask=None, check=False):
    """One variant on one batch: context passes, the first evaluation, the production sample from the same start
    (generator state restored), and, with `levels`, the field on the real successor + noise at every level and the
    weighted squared error of the consequence cells (mask [B, 81, 1]) and of the whole frame. fs: {part: f(y [B, 81, 16])}."""
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
            for j, s_ in enumerate(levels):
                xs = tuple(t[:, None] + s_ * torch.randn(t[:, None].shape, device=DEV, generator=g2) for t in x0)
                D, _ = M.denoised(den, xs, torch.full((len(x0[0]), 1), s_, device=DEV), a, mem, fx, cache, clamp, sd)
                lam = (s_ ** 2 + sd ** 2) / (s_ * sd) ** 2
                for part, f in fs.items():
                    out.setdefault(f"field_{part}", []).append(f(D[0][:, 0]))
                se = [(d[:, 0].float() - t.float()).square() for d, t in zip(D, x0)]
                out.setdefault("loss_frame", []).append(lam * sum(z.flatten(1).sum(1) for z in se).cpu())
                out.setdefault("loss_cells", []).append(lam * (se[0] * mask).flatten(1).sum(1).cpu())
            for k in [k for k in out if isinstance(out[k], list)]:
                out[k] = torch.stack(out[k], 1)
    return out


def measure(model, cfg, enc, L, rows, kind, hud, blocks, levels):
    parts = ("hud", "cell") if kind in MINING else ("hud",)
    cfg5 = copy.deepcopy(cfg)
    cfg5["edm"]["inference_context_sigma"] = CTX_SIGMA
    res = {v: {} for v in VARIANTS}
    extra = {k: [] for k in ("valid", "sep", "row_ctx")}
    n_ctx = 128 // R
    for b, start in enumerate(range(0, len(rows), n_ctx)):
        batch = rows[start:start + n_ctx]
        fw = {part: functional(kind, part, batch, enc, L, hud, blocks) for part in parts}
        idx = torch.arange(len(batch)).repeat_interleave(R)
        fs, mask = {}, torch.zeros(len(idx), 81, 1, device=DEV)
        for part, (A, W, sep, ok) in fw.items():
            A_, W_ = A[idx].to(DEV), W[idx].to(DEV)
            fs[part] = lambda y, A_=A_, W_=W_: ((y.float() - A_) * W_).sum((-1, -2)).cpu()
            mask = torch.maximum(mask, (W_.abs().sum(-1, keepdim=True) > 0).float())
        valid = torch.stack([fw[p_][3][idx] for p_ in parts], 1).all(1)
        extra["valid"].append(valid); extra["row_ctx"].append(start + idx)
        extra["sep"].append(torch.stack([fw[p_][2][idx] for p_ in parts], 1))
        for variant in VARIANTS:                                         # same seed: every variant gets the same draws
            inp = inputs(model, cfg, enc, batch, variant)
            base = variant == "base"
            o = run(model, cfg5 if variant.endswith("ctx5") else cfg, inp, fs, SEED + b, levels if base else None,
                    mask if base else None, check=b == 0)
            for k, v in o.items():
                res[variant].setdefault(k, []).append(v)
    flat = {f"{v}_{k}": torch.cat(x).numpy() for v, d in res.items() for k, x in d.items()}
    return {**flat, **{k: torch.cat(v).numpy() for k, v in extra.items()}}


def summary(o, rows, kind, L, levels, first, mon, s_prof, e_prof):
    ep = np.array([rows[i][0] for i in o["row_ctx"]])
    valid = o["valid"].astype(bool)
    est = lambda x, m=None: G.estimate(np.asarray(x, float), valid if m is None else valid & m, ep)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    g = np.array([s_off[rows[i][0]] + rows[i][1] - 1 for i in o["row_ctx"]])
    cell = np.array([rows[i][2] for i in o["row_ctx"]]); start = np.array([rows[i][3] for i in o["row_ctx"]])
    end = np.array([rows[i][4] for i in o["row_ctx"]]); ready = np.array([rows[i][5] for i in o["row_ctx"]], bool)
    after = np.array([L[HUD[c][0]][gg + 1, HUD[c][1]] for c, gg in zip(cell, g)]).clip(min=0)
    parts = ("hud", "cell") if kind in MINING else ("hud",)
    j6 = levels.index(first)
    res = {"contexts": len(rows), "rows_valid": int(valid.sum()), "recorded_reached_end": est(after == end),
           "recorded_kept_start": est(after == start), "sigma_star": {}, "parts": {}}
    for k, part in enumerate(parts):
        sep = float(np.median(o["sep"][valid, k]))
        res["sigma_star"][part] = {"separation_median": sep, "sigma_star": sep / 2,
                                   "above_sigma_star": NP.above(sep / 2, s_prof, e_prof)}
        f = lambda v, q: o[f"{v}_{q}_{part}"]
        r = {"f_recorded": est(f("base", "real")), "first_call": est(f("base", "first")),
             "production": est(f("base", "prod")),
             "leak_at_first_level": est(f("base", "field")[:, j6] - f("base", "first")),
             "sampler": est(f("base", "prod") - f("base", "first")),
             "rerun_control": est(f("rerun", "first") - f("base", "first")),
             "noop": {"first_call": est(f("noop", "first")), "production": est(f("noop", "prod")),
                      "action_contrast": est(f("rerun", "first") - f("noop", "first"))},
             "context_sigma_5": {"first_call": est(f("ctx5", "first")), "production": est(f("ctx5", "prod")),
                                 "noop_first_call": est(f("noop_ctx5", "first")), "noop_production": est(f("noop_ctx5", "prod")),
                                 "action_contrast": est(f("ctx5", "first") - f("noop_ctx5", "first")),
                                 "contrast_change": est(f("ctx5", "first") - f("noop_ctx5", "first")
                                                        - f("rerun", "first") + f("noop", "first"))},
             "field": {str(s): {"mean": est(f("base", "field")[:, j]), "bias": est(f("base", "field")[:, j] - f("base", "real"))}
                       for j, s in enumerate(levels)}}
        if ready.any() and (~ready).any():
            r["cooldown"] = {name: {"rows": int((valid & m).sum()), "f_recorded": est(f("base", "real"), m),
                                    "first_call": est(f("base", "first"), m), "production": est(f("base", "prod"), m),
                                    "noop_first_call": est(f("noop", "first"), m), "noop_production": est(f("noop", "prod"), m)}
                             for name, m in (("ready", ready), ("not_ready", ~ready))}
        res["parts"][part] = r
    mon = [levels.index(s) for s in mon]
    lc, lf = o["base_loss_cells"][:, mon], o["base_loss_frame"][:, mon]
    star = min(res["sigma_star"][p_]["sigma_star"] for p_ in parts)
    res["objective_share"] = {"event_frames": float(lc[valid].sum() / lf[valid].sum()),
                              "by_level": {str(levels[j]): float(lc[valid, i].sum() / lf[valid, i].sum()) for i, j in enumerate(mon)},
                              "cells_loss_above_sigma_star": float(lc[valid][:, [levels[j] >= star for j in mon]].sum() / lc[valid].sum())}
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


def readers(model, cfg, enc, L):
    res = {}
    for name, rows in populations(L, enc).items():
        y, ep = np.array([r[2] for r in rows]), np.array([r[0] for r in rows])
        if len(rows) < MIN_ROWS or y.all() or not y.any():
            res[name] = {"rows": len(rows), "skipped": "too few rows or one class"}
            continue
        X = IPB.features(enc, rows)
        n_last = X.shape[1] - (256 + 17 + 49)
        act = X[:, n_last + 256:n_last + 256 + 17]
        inputs_ = {"a_full_conditioning": X, "b_belief_action_fixation": X[:, n_last:],
                   "c_last_frame_action": np.concatenate([X[:, (IPB.HISTORY - 1) * IPB.FRAME:IPB.HISTORY * IPB.FRAME], X[:, n_last + 256:]], 1),
                   "e_embedding_action": np.concatenate([embeddings(model, cfg, enc, rows), act], 1),
                   "d_rule_inputs": np.stack([r[3] for r in rows])}
        res[name] = {"rows": len(rows), "rate": float(y.mean())}
        for key, Xi in inputs_.items():
            res[name][key] = IPB.probe(Xi, y, ep, np.random.default_rng(SEED))[0]
            print("B1", name, key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name][key].items()}, flush=True)
        del X, inputs_
    return res


def main():
    out_dir, ckpt = Path(sys.argv[1]), ROOT / sys.argv[2]
    n_ep = int(sys.argv[3]) if len(sys.argv) > 3 else 0
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
           "levels": levels, "monitor_levels": mon, "first_level": first, "rows_per_context": R, "context_sigma_route": CTX_SIGMA,
           "sampler": cfg["edm"]["sampler"], "opportunities": counts,
           "opportunity_rate": {k: n / frames for k, n in counts.items()}, "classes": {}}
    for kind in KINDS:
        rows = ctx[kind]
        if not rows:
            continue
        o = measure(model, cfg, enc, L, rows, kind, hud, blocks, levels)
        np.savez_compressed(out_dir / f"{kind}.npz", **o, ctx=np.array(rows, dtype=np.int64))
        res["classes"][kind] = summary(o, rows, kind, L, levels, first, mon, s_prof, e_prof)
        h = res["classes"][kind]["parts"]["hud"]
        print("B2", kind, len(rows), "rec", round(h["f_recorded"]["est"], 3), "first", round(h["first_call"]["est"], 3),
              "prod", round(h["production"]["est"], 3), "noop", round(h["noop"]["first_call"]["est"], 3),
              "ctx5", round(h["context_sigma_5"]["first_call"]["est"], 3), flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    res["readers"] = readers(model, cfg, enc, L)
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
