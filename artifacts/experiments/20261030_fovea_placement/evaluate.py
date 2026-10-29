"""Evaluation of the fovea fine-tune (PLAN.md, readouts 1-5). One GPU pass over several checkpoints.

Every checkpoint is read on the same dev rows, the same noise draws (one generator per route and class, fixed seed)
and one shared encoding: only the denoiser differs, which is asserted (every non-denoiser tensor equals the base's).
Routes: PERIPHERY = the S1-seeded random fixations of tests A / F / G; GLIMPSE = the same frames with every attempt
frame's fixation set by the rule (placement_map; all attempt frames, not half), other frames unchanged.
Rows: the first 1,000 per class of tests F (move, blocked, wood, sapling, table, drink) and A (adjacent, risky, safe,
random), as test G. Per row: the field projection at sigma 0.05 / 0.7 / 2 / 6.67 and the production projection
(catalogue / field_calibration functionals), and the production sample read by qualified readers: HUD values by
the nearest value mean of real train-400 periphery tokens (glimpse route: of real glimpse tokens at the same cell and
window), the faced-cell table by the S1 reader's terrain head. "Eye elsewhere": the target frame's glimpse does not
cover the consequence cell. Hallucination: the random rows, a generated change where the recorded cell did not change.
Costs: dev G1 loss by accounting class (forward only) and generated entity counts (S1.3 one-step, S1 reader).
Usage (project venv, GPU): python -m artifacts.experiments.20261030_fovea_placement.evaluate OUT_DIR LABEL=CKPT ...
  (LABEL=base for P1's checkpoint 10; fine-tune checkpoints are finetune/ARM/ft_UUUU.pt)
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
HUD_CELLS = {"health": (63, "s_body", 0), "drink": (65, "s_body", 2), "sapling": (67, "s_inv", 5), "wood": (68, "s_inv", 0)}
CAT_KINDS, FC_KINDS = ("move", "blocked", "wood", "sapling", "table", "drink"), ("adjacent", "risky", "safe", "random")
SEEDS = {k: 2040 + i for i, k in enumerate(CAT_KINDS + FC_KINDS)}


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


def projection(means, tok, start, end):
    """Linear functional per row, as the catalogue's: (y - mu_start) . (mu_end - mu_start) / |mu_end - mu_start|^2;
    NaN where a value has no mean."""
    out = np.full(len(tok), np.nan)
    for i, (s, e) in enumerate(zip(start, end)):
        if s in means and e in means:
            w = means[e] - means[s]
            out[i] = float(((tok[i].float() - means[s]) * w).sum() / w.square().sum().clamp(min=1e-9))
    return out


def readers(enc_tr, L_tr, enc_dv, L_dv, enc_tr_g, enc_dv_g):
    """Periphery HUD readers (train-400 means, dev qualification); glimpse HUD readers per (cell, window) from frames
    whose glimpse covers the cell at that window (train-400 under the rule; dev under the rule for qualification)."""
    out, qual = {"p": {}, "g": {}}, {"p": {}, "g": {}}
    stack = lambda enc, key: torch.cat([r[key] for r in enc])
    s_tr, s_dv = (np.concatenate([np.arange(len(r["p"])) + o for r, o in zip(enc, np.r_[0, np.cumsum(L["ep_T"] + 1)][:-1])])
                  for enc, L in ((enc_tr, L_tr), (enc_dv, L_dv)))
    p_tr, p_dv = stack(enc_tr, "p"), stack(enc_dv, "p")
    gl = {key: (stack(enc, "fix"), stack(enc, "g")) for enc, key in ((enc_tr_g, "fit"), (enc_dv_g, "qual"))}
    for name, (cell, arr, col) in HUD_CELLS.items():
        means = nearest_means(p_tr[:, cell], hud_value(L_tr, name, s_tr))
        out["p"][name] = means
        v_dv = hud_value(L_dv, name, s_dv)
        known = np.isin(v_dv, list(means))
        qual["p"][name] = {"accuracy": float((read(means, p_dv[:, cell][known]) == v_dv[known]).mean()),
                           "rows": int(known.sum()), "unknown_values": int((~known).sum())}
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
                        out["g"][(name, int(wc))] = {"means": nearest_means(tok, vals), "linear": fit_linear(tok, vals)}
                elif (name, int(wc)) in out["g"]:
                    rd = out["g"][(name, int(wc))]
                    ok = np.isin(vals, rd["linear"]["classes"])
                    acc = float((read_linear(rd["linear"], tok[ok]) == vals[ok]).mean())
                    rd["qualified"] = acc >= 0.99
                    qual["g"][f"{name}@{int(wc)}"] = {"accuracy": acc, "rows": int(ok.sum()), "qualified": rd["qualified"]}
    return out, qual


def row_info(kind, rows, L):
    """Per context: (consequence cell or -1, reader name or None, start value, end value, recorded end-reached)."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    info = []
    for r in rows:
        e, t = r[0], r[1]
        g0, g1 = s_off[e] + t - 1, s_off[e] + t
        if kind in ("wood", "sapling", "drink"):
            cell = HUD_CELLS[kind][0]
            a = int(hud_value(L, kind, g0))
            info.append((cell, kind, a, a + 1, int(hud_value(L, kind, g1)) == a + 1))
        elif kind == "table":
            dd = FC.DIRS[L["s_dir"][g0]]
            info.append(((3 + dd[0]) * 9 + 4 + dd[1], "table", 0, 1, L["s_view"][g1, 3 + dd[0], 4 + dd[1]] == CAT.TABLE))
        elif kind in ("adjacent", "risky", "safe"):
            info.append((63, "health", r[3], r[4], int(hud_value(L, "health", g1)) == r[4]))
        else:
            info.append((-1, None, 0, 0, False))
    return info


def reader_outcomes(o, rows, info, enc, rdr, s1_reader, route):
    """Per evaluated row (R per context): generated / real outcome by the readers; eye elsewhere; glimpse readout."""
    idx = o["row_ctx"]
    cells = np.array([info[i][0] for i in idx])
    fix = torch.stack([enc[rows[i][0]]["fix"][rows[i][1]] for i in idx])
    win = M.window(fix).numpy()
    covered = (win == cells[:, None]).any(1)
    res = {"eye_elsewhere": ~covered, "covered": covered}
    name = info[idx[0]][1] if len(idx) else None
    if name is None:
        return res
    gp = torch.as_tensor(o["gen_p"]).float()
    rp = torch.stack([enc[rows[i][0]]["p"][rows[i][1]] for i in idx]).float()
    ar = np.arange(len(idx))
    start, end = np.array([info[i][2] for i in idx]), np.array([info[i][3] for i in idx])
    if name == "table":
        v_gen = torch.as_tensor(o["gen_v"]).float()
        v_real = torch.stack([enc[rows[i][0]]["v"][rows[i][1]] for i in idx]).float()
        with torch.no_grad():
            t_gen = s1_reader(torch.cat([gp[ar, cells], v_gen], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
            t_real = s1_reader(torch.cat([rp[ar, cells], v_real], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
        res.update(gen_end=t_gen, gen_start=~t_gen, real_end=t_real)
        return res
    means = rdr["p"][name]
    vg, vr = read(means, gp[ar, cells]), read(means, rp[ar, cells])
    res.update(gen_end=vg == end, gen_start=vg == start, real_end=vr == end)
    if route == "glimpse":
        gg = torch.as_tensor(o["gen_g"]).float()
        rg = torch.stack([enc[rows[i][0]]["g"][rows[i][1]] for i in idx]).float()
        slot = (torch.as_tensor(win) == torch.as_tensor(cells)[:, None]).float().argmax(1)
        centre = (fix[:, 0] * 9 + fix[:, 1]).numpy()
        g_end = np.zeros(len(idx), bool); g_ok = np.zeros(len(idx), bool)
        fg_gen, fg_real = np.full(len(idx), np.nan), np.full(len(idx), np.nan)
        for wc in np.unique(centre[covered]):
            rd = rdr["g"].get((name, int(wc)))
            if rd is None:
                continue
            m = covered & (centre == wc)
            mt = torch.as_tensor(m)
            yg, yr = gg[mt][torch.arange(int(m.sum())), slot[mt]], rg[mt][torch.arange(int(m.sum())), slot[mt]]
            fg_gen[m], fg_real[m] = projection(rd["means"], yg, start[m], end[m]), projection(rd["means"], yr, start[m], end[m])
            if rd.get("qualified"):
                g_end[m] = read_linear(rd["linear"], yg) == end[m]
                g_ok[m] = True
        res.update(glimpse_gen_end=g_end, glimpse_read=g_ok, glimpse_f_gen=fg_gen, glimpse_f_real=fg_real)
    return res


def hallucination(o, rows, enc, L, rdr, s1_reader):
    """Random rows: per HUD cell and the faced cell, generated change rate where the recorded cell did not change."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    idx = o["row_ctx"]
    g0 = np.array([s_off[rows[i][0]] + rows[i][1] - 1 for i in idx]); g1 = g0 + 1
    gp = torch.as_tensor(o["gen_p"]).float()
    ep = np.array([rows[i][0] for i in idx])
    out = {}
    for name, (cell, arr, col) in HUD_CELLS.items():
        before, after = hud_value(L, name, g0), hud_value(L, name, g1)
        vg = read(rdr["p"][name], gp[:, cell])
        same = before == after
        out[name] = {"recorded_unchanged_rows": int(same.sum()),
                     "false_change": G.estimate((vg != before).astype(float), same, ep),
                     "recorded_changed_rows": int((~same).sum()),
                     "change_generated_when_recorded": G.estimate((vg == after).astype(float), ~same, ep) if (~same).any() else None}
    dd = FC.DIRS[L["s_dir"][g0]]
    fc_ = (3 + dd[:, 0]) * 9 + 4 + dd[:, 1]
    before = L["s_view"][g0, 3 + dd[:, 0], 4 + dd[:, 1]] == CAT.TABLE
    after = L["s_view"][g1, 3 + dd[:, 0], 4 + dd[:, 1]] == CAT.TABLE
    with torch.no_grad():
        tg = s1_reader(torch.cat([gp[np.arange(len(idx)), fc_], torch.as_tensor(o["gen_v"]).float()], 1).half().to(DEV))["terr"].argmax(1).cpu().numpy() == CAT.TABLE
    same = before == after
    out["table_faced"] = {"recorded_unchanged_rows": int(same.sum()), "false_change": G.estimate((tg != before).astype(float), same, ep)}
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


def summarise(o, rows, info, ro):
    ep = np.array([rows[i][0] for i in o["row_ctx"]])
    valid = o["valid"] if "valid" in o else np.ones(len(ep), bool)
    res = {}
    for subset in ("all", "eye_elsewhere", "eye_on"):
        m = valid if subset == "all" else valid & ro["eye_elsewhere"] if subset == "eye_elsewhere" else valid & ro["covered"]
        if not m.any():
            res[subset] = None
            continue
        est = lambda v: G.estimate(np.asarray(v, float), m, ep)
        r = {"rows": int(m.sum()), "f_recorded": est(o["f_real"]), "f_production": est(o["f_gen"]),
             "field": {s: est(o["f_den"][:, FC.SIGMAS.index(float(s))]) for s in READ_SIGMAS}}
        if "gen_end" in ro:
            rec = np.array([info[i][4] for i in o["row_ctx"]])
            r.update(recorded_end=est(rec), reader_real_end=est(ro["real_end"]), reader_production_end=est(ro["gen_end"]),
                     reader_production_start=est(ro["gen_start"]))
        res[subset] = r
    if "glimpse_gen_end" in ro and ro["glimpse_read"].any():
        m = valid & ro["glimpse_read"]
        res["glimpse_reader_production_end"] = G.estimate(ro["glimpse_gen_end"].astype(float), m, ep)
        res["glimpse_rows"] = int(m.sum())
    if "glimpse_f_gen" in ro:
        m = valid & np.isfinite(ro["glimpse_f_gen"])
        if m.any():
            res["glimpse_projection"] = {"rows": int(m.sum()),
                                         "f_real": G.estimate(np.where(m, ro["glimpse_f_real"], 0.0), m, ep),
                                         "f_production": G.estimate(np.where(m, ro["glimpse_f_gen"], 0.0), m, ep)}
    res["covered_share"] = float(ro["covered"][valid].mean())
    return res


def paired(a, b, rows, key, subset_mask):
    """Paired difference b - a of a per-row array (same rows and noise) with an episode-bootstrap SE; rows where
    either value is undefined (NaN) leave the subset."""
    ep = np.array([rows[i][0] for i in a["row_ctx"]])
    assert (a["row_ctx"] == b["row_ctx"]).all()
    d = np.asarray(b[key], float) - np.asarray(a[key], float)
    m = subset_mask & np.isfinite(d)
    return G.estimate(np.where(m, d, 0.0), m, ep) if m.any() else None


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
    enc = {"periphery": encode_with(model, L, cfg), "glimpse": encode_with(model, L, cfg, PM.window_of_state(L, rule))}
    ref = S.encode(model, S.first_episodes(L, 3), cfg)                   # encode_with reproduces S.encode
    for e in range(3):
        for k in ("fix", "prev_action"):
            assert torch.equal(ref[e][k], enc["periphery"][e][k]), k
        for k in ("p", "g", "v", "bp"):
            assert (ref[e][k].float() - enc["periphery"][e][k].float()).abs().max() <= 1e-3, k
    enc_tr, enc_tr_g = encode_with(model, Ltr, cfg), encode_with(model, Ltr, cfg, PM.window_of_state(Ltr, rule))
    rdr, qual = readers(enc_tr, Ltr, enc["periphery"], L, enc_tr_g, enc["glimpse"])
    del enc_tr, enc_tr_g
    s1_reader = S.load_reader(READER, BASE, DEV)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    e0 = enc["periphery"]
    hm, tm, means = CAT.hud_means(L, e0), CAT.table_means(L, e0), FC.value_means(L, e0)
    ctx = CAT.contexts(L, e0, np.random.default_rng(2031))[0]
    for kind, (cell, arr, col) in CAT.HUD.items():
        x = L["s_body" if arr == 0 else "s_inv"]
        ctx[kind] = [r for r in ctx[kind] if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= set(hm[kind])]
    fc = FC.contexts(L, e0, np.random.default_rng(2029))                 # its own "drink" class is A's, not F's
    ctx.update({k: fc[k] for k in FC_KINDS})
    rows = {k: ctx[k][:n_rows] for k in CAT_KINDS + FC_KINDS if ctx[k]}
    info = {k: row_info(k, rows[k], L) for k in rows}
    res = {"base": str(BASE), "rows_per_class": n_rows, "dev_episodes": int(len(L["ep_T"])),
           "rows": {k: len(v) for k, v in rows.items()}, "reader_qualification": qual, "checkpoints": {}, "differences": {}}
    per_row = {}
    for label, path in pairs:
        full = (ROOT / path).resolve()
        state = torch.load(full, weights_only=False, map_location="cpu")["model"]
        assert (label == "base") == (full == BASE.resolve()), "label base must name P1's checkpoint 10"
        assert all(torch.equal(state[k], v) for k, v in frozen.items()), "non-denoiser state differs from the base"
        model.denoiser.load_state_dict({k[len("denoiser."):]: v for k, v in state.items() if k.startswith("denoiser.")})
        del state
        row = {"path": path, "sha256": TR.file_sha256(full)}
        for route in ("periphery", "glimpse"):
            row[route] = {}
            for kind in rows:
                gen = torch.Generator(device=DEV).manual_seed(SEEDS[kind] + (0 if route == "periphery" else 100))
                if kind in CAT_KINDS:
                    o = CAT.run(model, enc[route], L, rows[kind], kind, hm, tm, cfg, gen, heun=False, keep=True)
                else:
                    o = FC.run(model, enc[route], rows[kind], means, cfg, gen, keep=True)
                ro = reader_outcomes(o, rows[kind], info[kind], enc[route], rdr, s1_reader, route)
                row[route][kind] = summarise(o, rows[kind], info[kind], ro)
                if kind == "random":
                    row[route][kind]["hallucination"] = hallucination(o, rows[kind], enc[route], L, rdr, s1_reader)
                keep = {k: o[k] for k in ("row_ctx", "f_real", "f_gen", "f_den") if k in o}
                keep.update({k: v for k, v in ro.items()}, valid=o.get("valid", np.ones(len(o["row_ctx"]), bool)))
                per_row[(label, route, kind)] = keep
                np.savez_compressed(out_dir / f"{label}_{route}_{kind}.npz", **keep)
            sub = "eye_elsewhere" if route == "periphery" else "eye_on"
            print(label, route, sub, {k: (round(v[sub]["field"]["6.67"]["est"], 3), round(v[sub]["f_production"]["est"], 3))
                                      for k, v in row[route].items() if v[sub]}, flush=True)
        row["costs"] = {"class_losses": class_losses(model, enc["periphery"], L, cfg),
                        "entity_counts": entity_counts(model, enc["periphery"], cfg, s1_reader)}
        res["checkpoints"][label] = row
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    # paired differences guided - random at each fine-tune checkpoint, and each fine-tune checkpoint - base
    labels = [l for l, _ in pairs]
    combos = [(f"random_{u}", f"guided_{u}") for u in sorted({l.split("_")[1] for l in labels if "_" in l})]
    combos += [("base", l) for l in labels if l != "base"]
    for a, b in combos:
        if a not in labels or b not in labels:
            continue
        d = res["differences"][f"{b}-{a}"] = {}
        for route in ("periphery", "glimpse"):
            d[route] = {}
            for kind in rows:
                A, B = per_row[(a, route, kind)], per_row[(b, route, kind)]
                m_all = A["valid"]
                d[route][kind] = {}
                for subset, m in (("all", m_all), ("eye_elsewhere", m_all & A["eye_elsewhere"]), ("eye_on", m_all & A["covered"])):
                    if not m.any():
                        d[route][kind][subset] = None
                        continue
                    r = {"field_6.67": paired({"row_ctx": A["row_ctx"], "x": A["f_den"][:, FC.SIGMAS.index(6.67)]},
                                              {"row_ctx": B["row_ctx"], "x": B["f_den"][:, FC.SIGMAS.index(6.67)]}, rows[kind], "x", m),
                         "f_production": paired(A, B, rows[kind], "f_gen", m)}
                    if "gen_end" in A:
                        r["reader_production_end"] = paired(A, B, rows[kind], "gen_end", m)
                    if "glimpse_f_gen" in A:
                        r["glimpse_f_production"] = paired(A, B, rows[kind], "glimpse_f_gen", m)
                        r["glimpse_reader_production_end"] = paired(A, B, rows[kind], "glimpse_gen_end", m & A["glimpse_read"])
                    d[route][kind][subset] = r
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    print("saved", out_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
