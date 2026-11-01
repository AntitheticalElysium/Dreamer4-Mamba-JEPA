"""Evaluation of the fovea fine-tune (PLAN.md, readouts 1-6). One GPU pass over several checkpoints.

Every checkpoint is read on the same dev rows, the same noise draws (one generator per route and class, fixed seed)
and one shared encoding: only the denoiser differs, which is asserted (every non-denoiser tensor equals the base's).
Routes: PERIPHERY = the S1-seeded random fixations of tests A / F / G; GUIDED = the same frames with every attempt
frame's fixation set by the rule (placement_map; all attempt frames, not half), other frames unchanged. Both routes'
field and production projections read PERIPHERY tokens; only the glimpse instruments read the glimpse output.
Rows: the first 1,000 per class of tests F (move, blocked, wood, sapling, table, drink) and A (adjacent, risky, safe,
random), as test G, plus failed-attempt strata (drink / wood / table attempts whose cell did not change).
Instruments, all qualified on REAL dev tokens before any model is read, and enforced row by row (a reading exists only
where its instrument is qualified and the row's values are in its support; excluded counts are reported):
- periphery HUD values (health, drink, sapling, wood): nearest value mean of real train-400 tokens, accuracy >= 0.99;
- faced-cell table: the S1 reader's terrain head, table recall and non-table specificity >= 0.99;
- glimpse HUD values per (cell, window): multinomial logistic reader on real train-400 guided glimpse tokens, accuracy
  >= 0.99; glimpse value projection: a ridge regression of the value on the token, (v_hat - start) / (end - start),
  qualified per class if its mean on the class's real eye-on rows is within 0.05 of the recorded rate;
- no glimpse instrument exists for the table (unmeasured).
Costs: dev G1 loss by accounting class (forward only) and generated entity counts (S1.3 one-step, S1 reader).
Usage (project venv, GPU): python -m artifacts.experiments.20261030_fovea_placement.evaluate OUT_DIR LABEL=CKPT ...
  (LABEL=base for P1's checkpoint 10; fine-tune checkpoints are finetune/ARM/ft_UUUU.pt; labels random_UUUU /
  guided_UUUU are paired at equal UUUU)
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G
from saccade.eval.probe import KEEP

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
CAT = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.catalogue")
LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
PM = importlib.import_module("artifacts.experiments.20261030_fovea_placement.placement_map")
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
READER = ROOT / "artifacts/saccade_m0/p1_world1/s1/results.reader.pt"
DEV, P = FC.DEV, FC.P
N, READ_SIGMAS = 1000, ("0.05", "0.7", "2.0", "6.67")
ACC_MIN, PROJ_TOL = 0.99, 0.05                                           # declared in PLAN.md before any run
HUD_CELLS = {"health": (63, "s_body", 0), "drink": (65, "s_body", 2), "sapling": (67, "s_inv", 5), "wood": (68, "s_inv", 0)}
CAT_KINDS, FC_KINDS = ("move", "blocked", "wood", "sapling", "table", "drink"), ("adjacent", "risky", "safe", "random")
FAIL_KINDS = ("drink_fail", "wood_fail", "table_fail")
SEEDS = {k: 2040 + i for i, k in enumerate(CAT_KINDS + FC_KINDS + FAIL_KINDS)}
READS = {"wood": "wood", "sapling": "sapling", "drink": "drink", "table": "table", "adjacent": "health", "risky": "health",
         "safe": "health", "drink_fail": "drink", "wood_fail": "wood", "table_fail": "table"}


@torch.no_grad()
def encode_with(model, L, cfg, windows=None):
    """S.encode's tokens and beliefs (keys p, g, v, fix, b, bp, action, prev_action, reward, terminated) with the S1
    seeded fixations, except frames t >= 1 whose decided window (windows[state t - 1]) is >= 0."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    out = []
    for e, ep in enumerate(S.episodes(L)):
        rng = np.random.default_rng([cfg["p1"]["monitor"]["fixed_fixation_seed"], ep.shard, ep.slot])
        fx = [tuple(int(x) for x in D.fixation_prior(rng, ())) for _ in range(ep.T + 1)]
        if windows is not None:
            w = windows[s_off[e]:s_off[e + 1] - 1]
            fx = fx[:1] + [tuple(int(x) for x in w[t - 1]) if w[t - 1, 0] >= 0 else fx[t] for t in range(1, ep.T + 1)]
        c = TR.tensors(D.gather([ep], [[(0, 1, t, *fx[t]) for t in range(ep.T + 1)]]), DEV)
        with TR.autocast(DEV, cfg):
            parts = [model.perception(c["frames"][:, a:a + 512], c["fixation"][:, a:a + 512]) for a in range(0, ep.T + 1, 512)]
            Pe = M.Percept(*(torch.cat(z, 1) for z in zip(*[p.parts() for p in parts])), c["fixation"])
            emb = model.belief.percept_embed(Pe)
            y, _ = model.belief(model.belief.tokens(emb, c["reward"], c["terminated"], c["action"]), model.belief.initial(1, DEV))
            b, bp = y[:, 0::2], torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
        out.append({"p": Pe.p[0].half().cpu(), "g": Pe.g[0].half().cpu(), "v": Pe.v[0].half().cpu(), "fix": c["fixation"][0].cpu(),
                    "b": b[0].float().half().cpu(), "bp": bp[0].float().half().cpu(), "action": c["action"][0].cpu(),
                    "prev_action": c["prev_action"][0].cpu(), "reward": c["reward"][0].cpu(), "terminated": c["terminated"][0].cpu()})
    return out


def hud_value(L, name, g):
    cell, arr, col = HUD_CELLS[name]
    return np.clip(L[arr][g, col], 0, None)


def nearest_means(tok, val):
    """Value -> mean token over real rows; the reader assigns a token to the nearest mean."""
    return {int(k): tok[val == k].float().mean(0) for k in np.unique(val)}


def read(means, tok):
    keys = sorted(means)
    mu = torch.stack([means[k] for k in keys])
    return np.array(keys)[torch.cdist(tok.float(), mu).argmin(1).numpy()]


def fit_linear(tok, val):
    """Multinomial logistic reader of a value from a glimpse token (the fovea mixes its crop, so a token also varies
    with the neighbouring counters; a nearest-mean reader was not accurate enough in the smoke test)."""
    classes = np.unique(val)
    X, y = tok.float().to(DEV), torch.as_tensor(np.searchsorted(classes, val), device=DEV)
    W = torch.zeros(X.shape[1], len(classes), device=DEV, requires_grad=True)
    b = torch.zeros(len(classes), device=DEV, requires_grad=True)
    opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(X @ W + b, y) + 1e-4 * W.square().sum()
        loss.backward()
        return loss
    with torch.enable_grad():
        opt.step(closure)
    return {"classes": classes, "W": W.detach(), "b": b.detach()}


def read_linear(rd, tok):
    return rd["classes"][(tok.float().to(DEV) @ rd["W"] + rd["b"]).argmax(1).cpu().numpy()]


def fit_ridge(tok, val, lam=1e-2):
    """Least-squares value readout v_hat = [tok, 1] . w (ridge lam on the token weights), linear in the token."""
    X = torch.cat([tok.float(), torch.ones(len(tok), 1)], 1).double()
    R = lam * torch.eye(X.shape[1], dtype=torch.float64); R[-1, -1] = 0
    return torch.linalg.solve(X.T @ X + R, X.T @ torch.as_tensor(val, dtype=torch.float64)).float()


def ridge_projection(w, tok, start, end):
    v_hat = (torch.cat([tok.float(), torch.ones(len(tok), 1)], 1) @ w).numpy()
    return (v_hat - start) / np.where(end != start, end - start, np.nan)


def readers(enc_tr, L_tr, enc_dv, L_dv, enc_tr_g, enc_dv_g, s1_reader):
    """Fitted on train-400, qualified on dev (real tokens). Periphery HUD readers; the table reader's dev rates; glimpse
    readers and ridge readouts per (cell, window) from frames whose guided glimpse covers the cell at that window."""
    out, qual = {"p": {}, "g": {}}, {"p": {}, "g": {}}
    stack = lambda enc, key: torch.cat([r[key] for r in enc])
    s_tr, s_dv = (np.concatenate([np.arange(len(r["p"])) + o for r, o in zip(enc, np.r_[0, np.cumsum(L["ep_T"] + 1)][:-1])])
                  for enc, L in ((enc_tr, L_tr), (enc_dv, L_dv)))
    p_tr, p_dv = stack(enc_tr, "p"), stack(enc_dv, "p")
    gl = {key: (stack(enc, "fix"), stack(enc, "g")) for enc, key in ((enc_tr_g, "fit"), (enc_dv_g, "qual"))}
    for name, (cell, arr, col) in HUD_CELLS.items():
        means = nearest_means(p_tr[:, cell], hud_value(L_tr, name, s_tr))
        v_dv = hud_value(L_dv, name, s_dv)
        known = np.isin(v_dv, list(means))
        acc = float((read(means, p_dv[:, cell][known]) == v_dv[known]).mean())
        out["p"][name] = {"means": means, "support": set(means), "qualified": acc >= ACC_MIN}
        qual["p"][name] = {"accuracy": acc, "rows": int(known.sum()), "unknown_values": int((~known).sum()),
                           "qualified": acc >= ACC_MIN}
        for (L, s, key) in ((L_tr, s_tr, "fit"), (L_dv, s_dv, "qual")):
            fix, g = gl[key]
            win = M.window(fix)
            slot = (win == cell).float().argmax(1)
            inside = (win == cell).any(1).numpy()
            centre = (fix[:, 0] * 9 + fix[:, 1]).numpy()
            for wc in np.unique(centre[inside]):
                m = inside & (centre == wc)
                tok = g[torch.as_tensor(m)][torch.arange(int(m.sum())), slot[torch.as_tensor(m)]]
                vals = hud_value(L, name, s[m])
                if key == "fit":
                    if m.sum() >= 200:
                        out["g"][(name, int(wc))] = {"linear": fit_linear(tok, vals), "ridge": fit_ridge(tok, vals),
                                                     "support": set(int(v) for v in np.unique(vals)), "qualified": False}
                elif (name, int(wc)) in out["g"]:
                    rd = out["g"][(name, int(wc))]
                    ok = np.isin(vals, list(rd["support"]))
                    acc = float((read_linear(rd["linear"], tok[ok]) == vals[ok]).mean())
                    rd["qualified"] = acc >= ACC_MIN
                    qual["g"][f"{name}@{int(wc)}"] = {"accuracy": acc, "rows": int(ok.sum()), "qualified": rd["qualified"]}
    # table reader on real faced cells
    d = FC.DIRS[L_dv["s_dir"][s_dv]]
    cell = (3 + d[:, 0]) * 9 + 4 + d[:, 1]
    is_table = L_dv["s_view"][s_dv, 3 + d[:, 0], 4 + d[:, 1]] == CAT.TABLE
    v_dv = stack(enc_dv, "v")
    with torch.no_grad():
        pred = np.concatenate([s1_reader(torch.cat([p_dv[i, cell[i]], v_dv[i]], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy()
                               for i in np.array_split(np.arange(len(cell)), max(1, len(cell) // 65536))]) == CAT.TABLE
    rec, spec = float(pred[is_table].mean()), float((~pred[~is_table]).mean())
    out["table"] = {"qualified": rec >= ACC_MIN and spec >= ACC_MIN}
    qual["table"] = {"table_recall": rec, "non_table_specificity": spec, "table_rows": int(is_table.sum()),
                     "qualified": out["table"]["qualified"]}
    return out, qual


def fail_contexts(L, enc, rng, n):
    """Failed attempts, as FC-style rows (e, t, cell, start, start, False): awake, a_{t-1} = DO facing water (drink) or
    a tree (wood) with the counter unchanged at t; PLACE_TABLE with no table on the faced cell at t - 1 or t."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    out = {k: [] for k in FAIL_KINDS}
    for e in range(len(enc)):
        for t in range(P, len(enc[e]["p"])):
            g, a = s_off[e] + t - 1, int(L["t_action"][a_off[e] + t - 1])
            if L["s_sleep"][g]:
                continue
            for kind, name, block in (("drink_fail", "drink", CAT.WATER), ("wood_fail", "wood", CAT.TREE)):
                v0 = int(hud_value(L, name, g))
                if a == CAT.DO and face[g] == block and int(hud_value(L, name, g + 1)) == v0:
                    out[kind].append((e, t, HUD_CELLS[name][0], v0, v0, False))
            if a == CAT.PLACE_TABLE and face[g] != CAT.TABLE and L["s_view"][g + 1, 3 + d[g, 0], 4 + d[g, 1]] != CAT.TABLE:
                out["table_fail"].append((e, t, (3 + d[g, 0]) * 9 + 4 + d[g, 1], 0, 0, False))
    counts = {k: len(v) for k, v in out.items()}
    for k in out:
        if len(out[k]) > n:
            out[k] = [out[k][i] for i in rng.choice(len(out[k]), n, replace=False)]
    return out, counts


def row_info(kind, rows, L):
    """Per context: (consequence cell or -1, start value, end value, recorded end reached). Failed attempts: start =
    end (their recorded outcome is "unchanged")."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    info = []
    for r in rows:
        e, t = r[0], r[1]
        g0, g1 = s_off[e] + t - 1, s_off[e] + t
        if kind in ("wood", "sapling", "drink"):
            a = int(hud_value(L, kind, g0))
            info.append((HUD_CELLS[kind][0], a, a + 1, int(hud_value(L, kind, g1)) == a + 1))
        elif kind == "table":
            dd = FC.DIRS[L["s_dir"][g0]]
            info.append(((3 + dd[0]) * 9 + 4 + dd[1], 0, 1, L["s_view"][g1, 3 + dd[0], 4 + dd[1]] == CAT.TABLE))
        elif kind in ("adjacent", "risky", "safe"):
            info.append((63, r[3], r[4], int(hud_value(L, "health", g1)) == r[4]))
        elif kind in FAIL_KINDS:
            info.append((r[2], r[3], r[4], True))
        else:
            info.append((-1, 0, 0, False))
    return info


def real_glimpse_check(rows, info, enc_g, rdr, kind):
    """Qualification of the glimpse ridge readout for a class: on its real eye-on rows (guided route), the mean readout
    vs the recorded rate (one row per context)."""
    name = READS.get(kind)
    if name not in HUD_CELLS or kind in FAIL_KINDS:
        return None
    cell = HUD_CELLS[name][0]
    f, rec = [], []
    for i, r in enumerate(rows):
        fix = enc_g[r[0]]["fix"][r[1]]
        win = M.window(fix).numpy()
        rd = rdr["g"].get((name, int(fix[0] * 9 + fix[1])))
        if cell not in win or rd is None or not {info[i][1], info[i][2]} <= rd["support"]:
            continue
        tok = enc_g[r[0]]["g"][r[1]][int(np.nonzero(win == cell)[0][0])][None]
        f.append(float(ridge_projection(rd["ridge"], tok, np.array([info[i][1]]), np.array([info[i][2]]))[0]))
        rec.append(float(info[i][3]))
    if not f:
        return {"rows": 0, "qualified": False}
    return {"rows": len(f), "f_real": float(np.mean(f)), "recorded": float(np.mean(rec)),
            "qualified": abs(np.mean(f) - np.mean(rec)) <= PROJ_TOL}


def reader_outcomes(o, rows, info, enc, rdr, gq, s1_reader, kind, route):
    """Per evaluated row (R per context): eye elsewhere / on; generated and real outcomes by qualified readers, with
    row masks read_ok (periphery or table instrument qualified and values supported) and, on the guided route,
    glimpse_read_ok / glimpse_proj_ok."""
    idx = o["row_ctx"]
    n = len(idx)
    cells = np.array([info[i][0] for i in idx])
    fix = torch.stack([enc[rows[i][0]]["fix"][rows[i][1]] for i in idx])
    win = M.window(fix).numpy()
    covered = (win == cells[:, None]).any(1)
    res = {"eye_elsewhere": ~covered, "covered": covered}
    name = READS.get(kind)
    if name is None:
        return res
    gp = torch.as_tensor(o["gen_p"]).float()
    rp = torch.stack([enc[rows[i][0]]["p"][rows[i][1]] for i in idx]).float()
    ar = np.arange(n)
    start, end = np.array([info[i][1] for i in idx]), np.array([info[i][2] for i in idx])
    if name == "table":
        v_gen = torch.as_tensor(o["gen_v"]).float()
        v_real = torch.stack([enc[rows[i][0]]["v"][rows[i][1]] for i in idx]).float()
        with torch.no_grad():
            vg = s1_reader(torch.cat([gp[ar, cells], v_gen], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
            vr = s1_reader(torch.cat([rp[ar, cells], v_real], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
        vg, vr = vg.astype(int), vr.astype(int)
        res["read_ok"] = np.full(n, rdr["table"]["qualified"])
    else:
        rd = rdr["p"][name]
        vg, vr = read(rd["means"], gp[ar, cells]), read(rd["means"], rp[ar, cells])
        supported = np.array([s in rd["support"] and e in rd["support"] for s, e in zip(start, end)])
        res["read_ok"] = supported & rd["qualified"]
    res.update(gen_end=vg == end, gen_start=vg == start, gen_change=vg != start, real_end=vr == end)
    res["glimpse_read_ok"] = np.zeros(n, bool)
    res["glimpse_proj_ok"] = np.zeros(n, bool)
    if route == "guided" and name in HUD_CELLS:
        gg = torch.as_tensor(o["gen_g"]).float()
        slot = (torch.as_tensor(win) == torch.as_tensor(cells)[:, None]).float().argmax(1)
        centre = (fix[:, 0] * 9 + fix[:, 1]).numpy()
        g_end, g_change, f_gen = np.zeros(n, bool), np.zeros(n, bool), np.full(n, np.nan)
        for wc in np.unique(centre[covered]):
            rdg = rdr["g"].get((name, int(wc)))
            if rdg is None:
                continue
            m = covered & (centre == wc)
            mt = torch.as_tensor(m)
            yg = gg[mt][torch.arange(int(m.sum())), slot[mt]]
            sup = np.array([s in rdg["support"] and e in rdg["support"] for s, e in zip(start[m], end[m])])
            vgg = read_linear(rdg["linear"], yg)
            g_end[m], g_change[m] = vgg == end[m], vgg != start[m]
            res["glimpse_read_ok"][m] = sup & rdg["qualified"]
            if gq is not None and gq.get("qualified"):
                f_gen[m] = ridge_projection(rdg["ridge"], yg, start[m], end[m])
                res["glimpse_proj_ok"][m] = sup & np.isfinite(f_gen[m])
        res.update(glimpse_gen_end=g_end, glimpse_gen_change=g_change, glimpse_f_gen=f_gen)
    return res


def hallucination(o, rows, enc, L, rdr, s1_reader):
    """Random rows (a population diagnostic, not the failed-attempt test): per HUD cell and the faced cell, the
    generated change rate where the recorded cell did not change; qualified readers and supported values only."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    idx = o["row_ctx"]
    g0 = np.array([s_off[rows[i][0]] + rows[i][1] - 1 for i in idx]); g1 = g0 + 1
    gp = torch.as_tensor(o["gen_p"]).float()
    ep = np.array([rows[i][0] for i in idx])
    out = {}
    for name, (cell, arr, col) in HUD_CELLS.items():
        rd = rdr["p"][name]
        if not rd["qualified"]:
            out[name] = "unmeasured: reader not qualified"
            continue
        before, after = hud_value(L, name, g0), hud_value(L, name, g1)
        vg = read(rd["means"], gp[:, cell])
        same = (before == after) & np.isin(before, list(rd["support"]))
        out[name] = {"recorded_unchanged_rows": int(same.sum()),
                     "false_change": G.estimate((vg != before).astype(float), same, ep) if same.any() else None}
    if rdr["table"]["qualified"]:
        dd = FC.DIRS[L["s_dir"][g0]]
        fc_ = (3 + dd[:, 0]) * 9 + 4 + dd[:, 1]
        before = L["s_view"][g0, 3 + dd[:, 0], 4 + dd[:, 1]] == CAT.TABLE
        after = L["s_view"][g1, 3 + dd[:, 0], 4 + dd[:, 1]] == CAT.TABLE
        with torch.no_grad():
            tg = s1_reader(torch.cat([gp[np.arange(len(idx)), fc_], torch.as_tensor(o["gen_v"]).float()], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
        same = before == after
        out["table_faced"] = {"recorded_unchanged_rows": int(same.sum()), "false_change": G.estimate((tg != before).astype(float), same, ep)}
    else:
        out["table_faced"] = "unmeasured: reader not qualified"
    return out


@torch.no_grad()
def class_losses(model, enc, L, cfg, n_win=400):
    """Dev G1 loss by accounting class (loss_accounting's forward, no gradients), fixed windows and noise."""
    cls_cell = LA.cell_classes(L)
    wins, s_off = LA.windows(L, enc, n_win, np.random.default_rng(2034))
    gen = torch.Generator(device=DEV).manual_seed(2034)
    den = model.denoiser
    tot, cnt = np.zeros(len(LA.CLASSES)), np.zeros(len(LA.CLASSES))
    for start in range(0, len(wins), LA.B):
        batch = wins[start:start + LA.B]
        take = lambda key: torch.stack([enc[e][key][s:s + LA.P + LA.T] for e, s in batch]).to(DEV)
        x0 = tuple(take(k).float() for k in ("p", "g", "v"))
        first = torch.stack([torch.as_tensor(np.arange(s, s + LA.P + LA.T) == 0) for e, s in batch]).to(DEV)
        pa = torch.where(first, torch.full_like(take("prev_action"), -1), take("prev_action"))
        episode = torch.tensor([[i] * (LA.P + LA.T) for i in range(len(batch))], device=DEV)
        sigma = TR.sample_sigma(episode.shape, cfg, DEV, gen)
        noise = tuple(torch.randn(t.shape, device=DEV, generator=gen) for t in x0)
        with TR.autocast(DEV, cfg):
            c_skip, c_out, c_in, c_noise = M.precondition(sigma, cfg["edm"]["sigma_data"])
            xs = tuple(t + M._bcast(sigma, t) * n for t, n in zip(x0, noise))
            Fo = den(tuple(M._bcast(c_in, t) * t for t in xs), c_noise, pa, model.belief.memory(take("bp").float()),
                     take("fix"), episode, torch.ones_like(episode, dtype=torch.bool))
        se = [((f.float() - (t - M._bcast(c_skip, t) * s) / M._bcast(c_out, t)) ** 2)[:, LA.P:] for f, t, s in zip(Fo, x0, xs)]
        cc = torch.stack([torch.as_tensor(cls_cell[s_off[e] + s + LA.P:s_off[e] + s + LA.P + LA.T].astype(np.int64)) for e, s in batch]).to(DEV)
        keep = ~first[:, LA.P:]
        sp = se[0].sum(-1)
        for k, name in enumerate(LA.CLASSES):
            if name == "glimpse":
                tot[k] += float(se[1].sum((-1, -2))[keep].sum()); cnt[k] += float(keep.sum()) * 9 * 64
            elif name == "lighting":
                tot[k] += float(se[2].sum(-1)[keep].sum()); cnt[k] += float(keep.sum()) * 8
            else:
                m = (cc == k) & keep[..., None]
                tot[k] += float((sp * m).sum()); cnt[k] += float(m.sum()) * 16
    return {name: {"se_per_element": tot[k] / max(cnt[k], 1), "elements": cnt[k]} for k, name in enumerate(LA.CLASSES)}


@torch.no_grad()
def entity_counts(model, enc, cfg, s1_reader, n=5000):
    eps_, ts, gen = S.one_step(model, enc, cfg, n, np.random.default_rng(0))
    real = {k: torch.stack([enc[e][k][t].float() for e, t in zip(eps_, ts)]) for k in ("p", "v")}
    rd = lambda p, v: torch.cat([s1_reader(torch.cat([p[i][:, :63].reshape(-1, p.shape[-1]).half().to(DEV),
                                                      v[i].repeat_interleave(63, 0).half().to(DEV)], 1))["kind"].argmax(1).reshape(len(i), 63).cpu()
                                 for i in torch.arange(len(p)).split(2048)]).numpy()[:, KEEP[:63]]
    kr, kg = rd(real["p"], real["v"]), rd(gen["p"], gen["v"])
    one = np.ones(len(eps_), bool)
    return {kn: {"real": G.estimate((kr == k).sum(1), one, eps_), "generated": G.estimate((kg == k).sum(1), one, eps_)}
            for k, kn in enumerate(G.KINDS, 1)}


SUBSETS = ("all", "eye_elsewhere", "eye_on")


def subset_mask(ro, valid, subset):
    return valid if subset == "all" else valid & (ro["eye_elsewhere"] if subset == "eye_elsewhere" else ro["covered"])


def summarise(o, rows, info, ro, kind):
    ep = np.array([rows[i][0] for i in o["row_ctx"]])
    valid = o["valid"] if "valid" in o else np.ones(len(ep), bool)
    est = lambda v, m: G.estimate(np.where(m, np.asarray(v, float), 0.0), m, ep) if m.any() else None
    res = {}
    for subset in SUBSETS:
        m = subset_mask(ro, valid, subset)
        if not m.any():
            res[subset] = None
            continue
        r = {"rows": int(m.sum())}
        if kind not in FAIL_KINDS:
            r.update(f_recorded=est(o["f_real"], m), f_production=est(o["f_gen"], m),
                     field={s: est(o["f_den"][:, FC.SIGMAS.index(float(s))], m) for s in READ_SIGMAS})
        if "read_ok" in ro:
            mr = m & ro["read_ok"]
            rec = np.array([info[i][3] for i in o["row_ctx"]], float)
            r.update(rows_read=int(mr.sum()), rows_excluded=int((m & ~ro["read_ok"]).sum()),
                     recorded_end=est(rec, mr), reader_real_end=est(ro["real_end"], mr))
            if kind in FAIL_KINDS:
                r["reader_production_false_change"] = est(ro["gen_change"], mr)
            else:
                r.update(reader_production_end=est(ro["gen_end"], mr), reader_production_start=est(ro["gen_start"], mr))
            if "glimpse_gen_end" in ro:
                mg, mp = m & ro["glimpse_read_ok"], m & ro["glimpse_proj_ok"]
                r["glimpse_reader_rows"] = int(mg.sum())
                r["glimpse_reader_production"] = est(ro["glimpse_gen_change" if kind in FAIL_KINDS else "glimpse_gen_end"], mg)
                r["glimpse_projection_rows"] = int(mp.sum())
                r["glimpse_projection_production"] = est(ro["glimpse_f_gen"], mp) if kind not in FAIL_KINDS else None
        res[subset] = r
    res["covered_share"] = float(ro["covered"][valid].mean())
    return res


def paired(a, b, rows, key, mask):
    """Paired difference b - a of a per-row array (same rows and noise), episode-bootstrap SE and a 95% interval;
    rows where either value is undefined leave the mask."""
    ep = np.array([rows[i][0] for i in a["row_ctx"]])
    assert (a["row_ctx"] == b["row_ctx"]).all()
    d = np.asarray(b[key], float) - np.asarray(a[key], float)
    m = mask & np.isfinite(d)
    if not m.any():
        return None
    r = G.estimate(np.where(m, d, 0.0), m, ep)
    r["ci95"] = [r["est"] - 1.96 * r["se"], r["est"] + 1.96 * r["se"]]
    return r


def differences(A, B, rows, kind):
    out = {}
    for subset in SUBSETS:
        m = subset_mask(A, A["valid"], subset)
        if not m.any():
            out[subset] = None
            continue
        r = {}
        if kind not in FAIL_KINDS:
            r["field_6.67"] = paired({"row_ctx": A["row_ctx"], "x": A["f_den"][:, FC.SIGMAS.index(6.67)]},
                                     {"row_ctx": B["row_ctx"], "x": B["f_den"][:, FC.SIGMAS.index(6.67)]}, rows, "x", m)
            r["f_production"] = paired(A, B, rows, "f_gen", m)
        if "read_ok" in A:
            key = "gen_change" if kind in FAIL_KINDS else "gen_end"
            r["reader_production"] = paired(A, B, rows, key, m & A["read_ok"])
            if "glimpse_gen_end" in A:
                gkey = "glimpse_gen_change" if kind in FAIL_KINDS else "glimpse_gen_end"
                r["glimpse_reader_production"] = paired(A, B, rows, gkey, m & A["glimpse_read_ok"])
                if kind not in FAIL_KINDS:
                    r["glimpse_projection_production"] = paired(A, B, rows, "glimpse_f_gen", m & A["glimpse_proj_ok"])
        out[subset] = r
    return out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    args = sys.argv[2:]
    n_ep = int(args.pop(0).split("=")[1]) if args and args[0].startswith("--episodes=") else 0   # smoke tests only
    n_rows = 50 if n_ep else N
    pairs = [a.split("=", 1) for a in args]
    ck = torch.load(BASE, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    frozen = {k: v.clone() for k, v in ck["model"].items() if not k.startswith("denoiser.")}
    del ck
    rule = {int(k): tuple(v) for k, v in json.loads((HERE / "map.json").read_text())["rule"].items()}
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    Ltr = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_train400_labels.npz")), n_ep)
    enc = {"periphery": encode_with(model, L, cfg), "guided": encode_with(model, L, cfg, PM.window_of_state(L, rule))}
    ref = S.encode(model, S.first_episodes(L, 3), cfg)                   # encode_with reproduces S.encode
    for e in range(3):
        for k in ("fix", "prev_action"):
            assert torch.equal(ref[e][k], enc["periphery"][e][k]), k
        for k in ("p", "g", "v", "bp"):
            assert (ref[e][k].float() - enc["periphery"][e][k].float()).abs().max() <= 1e-3, k
    s1_reader = S.load_reader(READER, BASE, DEV)
    enc_tr, enc_tr_g = encode_with(model, Ltr, cfg), encode_with(model, Ltr, cfg, PM.window_of_state(Ltr, rule))
    rdr, qual = readers(enc_tr, Ltr, enc["periphery"], L, enc_tr_g, enc["guided"], s1_reader)
    del enc_tr, enc_tr_g
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    e0 = enc["periphery"]
    hm, tm, means = CAT.hud_means(L, e0), CAT.table_means(L, e0), FC.value_means(L, e0)
    ctx = CAT.contexts(L, e0, np.random.default_rng(2031))[0]
    for kind, (cell, arr, col) in CAT.HUD.items():
        x = L["s_body" if arr == 0 else "s_inv"]
        ctx[kind] = [r for r in ctx[kind] if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= set(hm[kind])]
    fc = FC.contexts(L, e0, np.random.default_rng(2029))                 # its own "drink" class is A's, not F's
    ctx.update({k: fc[k] for k in FC_KINDS})
    fails, fail_counts = fail_contexts(L, e0, np.random.default_rng(2035), n_rows)
    ctx.update(fails)
    # value means for FC.run on the failed attempts (its projection is undefined there: start = end; only the
    # production samples are read)
    means_all = {**means, 68: hm["wood"], **{c: {0: v} for c, v in tm.items()}}
    rows = {k: ctx[k][:n_rows] for k in CAT_KINDS + FC_KINDS + FAIL_KINDS if ctx[k]}
    info = {k: row_info(k, rows[k], L) for k in rows}
    gq = {k: real_glimpse_check(rows[k], info[k], enc["guided"], rdr, k) for k in rows}
    qual["glimpse_projection"] = gq
    res = {"base": str(BASE), "rows_per_class": n_rows, "dev_episodes": int(len(L["ep_T"])),
           "rows": {k: len(v) for k, v in rows.items()}, "failed_attempts_available": fail_counts,
           "instrument_qualification": qual, "unmeasured": {"table_glimpse": "no glimpse instrument for the table"},
           "checkpoints": {}, "differences": {}}
    print("qualification", json.dumps(qual, default=str)[:3000], flush=True)
    per_row = {}
    for label, path in pairs:
        full = (ROOT / path).resolve()
        state = torch.load(full, weights_only=False, map_location="cpu")["model"]
        assert (label == "base") == (full == BASE.resolve()), "label base must name P1's checkpoint 10"
        assert all(torch.equal(state[k], v) for k, v in frozen.items()), "non-denoiser state differs from the base"
        model.denoiser.load_state_dict({k[len("denoiser."):]: v for k, v in state.items() if k.startswith("denoiser.")})
        del state
        row = {"path": path, "sha256": TR.file_sha256(full)}
        for route in ("periphery", "guided"):
            row[route] = {}
            for kind in rows:
                gen = torch.Generator(device=DEV).manual_seed(SEEDS[kind] + (0 if route == "periphery" else 100))
                if kind in CAT_KINDS:
                    o = CAT.run(model, enc[route], L, rows[kind], kind, hm, tm, cfg, gen, heun=False, keep=True)
                else:
                    o = FC.run(model, enc[route], rows[kind], means_all, cfg, gen, keep=True)
                ro = reader_outcomes(o, rows[kind], info[kind], enc[route], rdr, gq.get(kind), s1_reader, kind, route)
                row[route][kind] = summarise(o, rows[kind], info[kind], ro, kind)
                if kind == "random":
                    row[route][kind]["hallucination_population"] = hallucination(o, rows[kind], enc[route], L, rdr, s1_reader)
                keep = {k: o[k] for k in ("row_ctx", "f_real", "f_gen", "f_den") if k in o}
                keep.update({k: v for k, v in ro.items()}, valid=o.get("valid", np.ones(len(o["row_ctx"]), bool)))
                per_row[(label, route, kind)] = keep
                np.savez_compressed(out_dir / f"{label}_{route}_{kind}.npz", **keep)
            sub = "eye_elsewhere" if route == "periphery" else "eye_on"
            print(label, route, sub, {k: (round(v[sub]["field"]["6.67"]["est"], 3), round(v[sub]["f_production"]["est"], 3))
                                      for k, v in row[route].items() if v[sub] and "field" in v[sub]}, flush=True)
        row["costs"] = {"class_losses": class_losses(model, enc["periphery"], L, cfg),
                        "entity_counts": entity_counts(model, enc["periphery"], cfg, s1_reader)}
        res["checkpoints"][label] = row
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    labels = [l for l, _ in pairs]
    combos = [(f"random_{u}", f"guided_{u}") for u in sorted({l.split("_")[1] for l in labels if "_" in l})]
    combos += [("base", l) for l in labels if l != "base"]
    for a, b in combos:
        if a in labels and b in labels:
            res["differences"][f"{b}-{a}"] = {route: {k: differences(per_row[(a, route, k)], per_row[(b, route, k)], rows[k], k)
                                                      for k in rows} for route in ("periphery", "guided")}
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("saved", out_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
