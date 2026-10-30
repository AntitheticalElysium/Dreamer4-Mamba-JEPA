"""M0.5 evaluation (PLAN.md). Per checkpoint, with its own encoder (every module trains in M0.5): dev encoded twice
(route `periphery`: S.encode, the S1-seeded fixations; route `guided`: every attempt frame fixated by the rule),
train-400 once (guided) to fit the readers (instruments.fit), qualified on dev real tokens. Rows, seeds and generation
are v2's (label-defined rows, identical for every checkpoint; catalogue.run / field_calibration.run with keep=True; one
seed per route and class, so the same draws for every checkpoint).
Readouts (rows of drink, wood, table, sapling, damage (A's adjacent / risky), failed drink and table attempts):
- PREDICTION, primary: the generated glimpse at the consequence cell (guided route, cell inside the target frame's
  window), read by the checkpoint's glimpse value reader (per window centre) or glimpse table reader; secondary: the
  generated periphery cell (periphery route, cell outside the window), periphery value or table reader. A reading
  exists only where the instrument is qualified and the token lies in its real support (instruments.py).
  success = read == recorded end, copy = read == start; failed attempts: false change = read != start.
- PERCEPTION, co-primary: real dev frames of the same rows, guided route, eye on: pixel NLL (nats per pixel) of the
  consequence cell under the fovea reconstruction (glimpse tokens) and under TSP (predicted belief + periphery), and
  the mean over the window's 9 cells.
- MEMORY RELIANCE, co-primary (drink, wood, table; guided route, eye on): the generated glimpse with the belief
  recomputed from the stored tokens ("rerun") vs recomputed after a NOOP at t - 1 with the denoiser's own action input
  kept ("noop_belief"); the same draws. Reliance = success(rerun) - success(noop_belief).
- REGRESSION PANEL: dev reward NLL, continuation BCE and TSP NLL over all frames (S.encode), movement (MOVE rows'
  shift projection of the sample and of the field at 6.67, periphery route), dev G1 loss by class (forward only, the
  checkpoint's own token space: descriptive), reader qualification and value separation, and at `base` and every
  `*_1500` the S1 entity counts (S1's reader protocol refitted on the checkpoint's own tokens, real vs generated).
Drink and wood are split into appearance (start 0) and increment (start >= 1).
Usage (project venv, GPU): python -m artifacts.experiments.20261030_m05_guided_fovea.evaluate OUT_DIR
  [--episodes=K] LABEL=CKPT ...   (LABEL=base for P1's checkpoint 10; differences: every label minus base, and
  guided_UUUU minus random_UUUU)
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
IN = importlib.import_module("artifacts.experiments.20261030_m05_guided_fovea.instruments")
IP = importlib.import_module("artifacts.experiments.20261030_increment_diagnosis.increment_probe")
CAT, FC, PM = EV.CAT, EV.FC, EV.PM
DEV, N, P = EV.DEV, EV.N, EV.P
CONSEQ = ("drink", "wood", "table", "sapling")
KINDS = CONSEQ + ("move", "adjacent", "risky", "drink_fail", "table_fail")
READS = {"drink": "drink", "wood": "wood", "sapling": "sapling", "table": "table", "adjacent": "health",
         "risky": "health", "drink_fail": "drink", "table_fail": "table"}
MEMORY_KINDS, MEM_SEED, R_MEM, NOOP = ("drink", "wood", "table"), 2060, 2, 0


def load(path):
    full = (EV.ROOT / path).resolve()
    ck = torch.load(full, weights_only=False, map_location="cpu")
    if "cfg" not in ck:
        assert ck["identity"]["base_sha256"] == TR.file_sha256(EV.BASE)
    cfg = ck.get("cfg") or torch.load(EV.BASE, weights_only=False, map_location="cpu")["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    return model, cfg, TR.file_sha256(full)


def rows_of(L, enc, n_rows):
    """v2's rows (labels only; `enc` gives episode lengths)."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    ctx = CAT.contexts(L, enc, np.random.default_rng(2031))[0]
    for kind, (cell, arr, col) in CAT.HUD.items():
        x = L["s_body" if arr == 0 else "s_inv"]
        keys = set(np.unique(x[:, col]).astype(int).tolist())
        ctx[kind] = [r for r in ctx[kind] if {int(x[s_off[r[0]] + r[1] - 1, col]), int(x[s_off[r[0]] + r[1] - 1, col]) + 1} <= keys]
    fc = FC.contexts(L, enc, np.random.default_rng(2029))
    ctx.update({k: fc[k] for k in EV.FC_KINDS})
    fails, counts = EV.fail_contexts(L, enc, np.random.default_rng(2035), n_rows)
    ctx.update(fails)
    return {k: ctx[k][:n_rows] for k in KINDS if ctx.get(k)}, counts


def read_glimpse(rd, name, gen_g, slot, centre, start, end, use):
    """Glimpse readings of the cell's generated token: (read value, read_ok) where `use`."""
    n = len(slot)
    read, ok = np.full(n, -1), np.zeros(n, bool)
    tok = gen_g[np.arange(n), slot]
    if name == "table":
        r = rd["table_g"]
        if r["qualified"] and use.any():
            read[use] = IN.predict(r, IN.feats(tok[use], slot[use]))
            ok[use] = IN.readable(r["support"], tok[use], read[use], slot[use])
        return read, ok
    for c in np.unique(centre[use]):
        r = rd["g"].get((name, int(c)))
        m = use & (centre == c)
        if r is None or not r["qualified"]:
            continue
        read[m] = IN.predict(r, tok[m])
        ok[m] = IN.readable(r["support"], tok[m], read[m]) & np.isin(start[m], r["classes"]) & np.isin(end[m], r["classes"])
    return read, ok


def read_periphery(rd, name, tok, v, start, end, use):
    """Periphery readings of a cell's token (tables: with the lighting latent v): (read value, read_ok) where `use`."""
    n = len(tok)
    read, ok = np.full(n, -1), np.zeros(n, bool)
    if name == "table":
        r = rd["table_p"]
        if r["qualified"] and use.any():
            read[use] = IN.predict(r, torch.cat([tok, v], 1)[use])
            ok[use] = IN.readable(r["support"], tok[use], read[use])
    else:
        r = rd["p"][name]
        if r["qualified"] and use.any():
            read[use] = EV.read(r["means"], tok[use])
            ok[use] = IN.readable(r["support"], tok[use], read[use]) & np.isin(start[use], list(r["means"])) & np.isin(end[use], list(r["means"]))
    return read, ok


def outcomes(o, rows, info, enc, rd, kind, route):
    """Per evaluated row: masks and outcomes of this route's generated output (guided: the glimpse; periphery: the
    periphery cell), the same instrument on the row's REAL token (instrument check: it must reproduce the recorded
    outcome), and on the guided route the generated periphery cell's reading beside the glimpse's (coherence)."""
    idx = o["row_ctx"]
    n = len(idx)
    cell = np.array([info[i][0] for i in idx]); start = np.array([info[i][1] for i in idx]); end = np.array([info[i][2] for i in idx])
    fix = torch.stack([enc[rows[i][0]]["fix"][rows[i][1]] for i in idx])
    win = M.window(fix).numpy()
    inw = (win == cell[:, None]).any(1)
    use = (inw if route == "guided" else ~inw) & o.get("valid", np.ones(n, bool))
    name = READS[kind]
    real = {k: torch.stack([enc[rows[i][0]][k][rows[i][1]] for i in idx]).float() for k in ("p", "g", "v")}
    gp, gv = torch.as_tensor(o["gen_p"]).float()[np.arange(n), cell], torch.as_tensor(o["gen_v"]).float()
    if route == "guided":
        slot, centre = (win == cell[:, None]).argmax(1), (fix[:, 0] * 9 + fix[:, 1]).numpy()
        read, ok = read_glimpse(rd, name, torch.as_tensor(o["gen_g"]).float(), slot, centre, start, end, use)
        rread, rok = read_glimpse(rd, name, real["g"], slot, centre, start, end, use)
        pread, pok = read_periphery(rd, name, gp, gv, start, end, use)
    else:
        read, ok = read_periphery(rd, name, gp, gv, start, end, use)
        rread, rok = read_periphery(rd, name, real["p"][np.arange(n), cell], real["v"], start, end, use)
        pread, pok = np.full(n, -1), np.zeros(n, bool)
    recorded = np.array([bool(info[i][3]) for i in idx])                # the consequence really happened (row_info)
    rec_end = np.where(recorded, end, start)                            # the recorded outcome's value
    return {"use": use, "read_ok": use & ok, "unreadable": use & ~ok, "read": read, "start": start, "end": end,
            "recorded": recorded, "success": read == end, "copy": read == start, "false_change": read != start,
            "real_ok": use & rok, "real_correct": rread == rec_end,
            "coherence_ok": use & ok & pok, "coherent": read == pread}


def groups(kind, start):
    if kind in ("drink", "wood"):
        return {"appearance": start == 0, "increment": start >= 1}
    return {"all": np.ones(len(start), bool)}


def est(x, m, ep):
    return G.estimate(np.where(m, np.asarray(x, float), 0.0), m, ep) if m.any() else None


def pdiff(a, b, m, ep):
    if not m.any():
        return None
    e = G.estimate(np.where(m, np.asarray(b, float) - np.asarray(a, float), 0.0), m, ep)
    return {**e, "ci95": [e["est"] - 1.96 * e["se"], e["est"] + 1.96 * e["se"]], "rows": int(m.sum())}


def keys_of(kind):
    return ("false_change",) if kind.endswith("_fail") else ("success", "copy")


def summarise(oc, ep, kind):
    out = {}
    for g, m in groups(kind, oc["start"]).items():
        mu, mr = oc["use"] & m, oc["read_ok"] & m
        out[g] = {"rows": int(mu.sum()), "read": int(mr.sum()), "unreadable_share": est(oc["unreadable"], mu, ep),
                  "recorded": est(oc["recorded"], mr, ep), **{k: est(oc[k], mr, ep) for k in keys_of(kind)},
                  "real_token_correct": est(oc["real_correct"], oc["real_ok"] & m, ep),
                  "glimpse_periphery_coherent": est(oc["coherent"], oc["coherence_ok"] & m, ep)}
    return out


@torch.no_grad()
def generate(model, cfg, inps):
    """Row inputs (increment_probe.row_inputs) -> the generated frame (p, g, v), one shared generator."""
    den = model.denoiser
    st = lambda k: torch.stack([i[k] for i in inps]).to(DEV)
    fix, pa, bp = st("fix"), st("pa"), st("bp")
    gen = torch.Generator(device=DEV).manual_seed(int(inps[0]["seed"]))
    cache = den.new_cache()
    with TR.autocast(DEV, cfg):
        ctx = {k: st(k) for k in ("p", "g", "v")}
        for k in range(P):
            cache = M.context_pass(den, tuple(ctx[n][:, k][:, None] for n in ("p", "g", "v")), pa[:, k][:, None],
                                   model.belief.memory(bp[:, k])[:, None], fix[:, k][:, None], cache, cfg, gen)
        x0 = (st("x0p"), st("x0g"), st("x0v"))
        xg = M.sample(den, [tuple(t.shape[1:]) for t in x0], pa[:, P][:, None], model.belief.memory(bp[:, P])[:, None],
                      fix[:, P][:, None], cache, cfg, gen)
    return tuple(t[:, 0].float().cpu() for t in xg)


def memory_reliance(model, cfg, rows, info, enc_g, rd, kind):
    """Per variant ("rerun", "noop_belief"): per row (R_MEM draws per eye-on context) read_ok, success, start."""
    ctxs = [(i, r[0], r[1]) for i, r in enumerate(rows) if (M.window(enc_g[r[0]]["fix"][r[1]]) == info[i][0]).any()]
    out = {}
    for variant in ("rerun", "noop_belief"):
        res = {k: [] for k in ("ctx", "read_ok", "success", "copy", "start")}
        for b0 in range(0, len(ctxs), 256 // R_MEM):
            batch = ctxs[b0:b0 + 256 // R_MEM]
            inps = []
            for i, e, t in batch:
                inp = IP.row_inputs(model, cfg, enc_g[e], t, {"rerun": True, "noop": variant == "noop_belief"}, None)
                inp["pa"] = inp["pa"].clone(); inp["pa"][-1] = enc_g[e]["prev_action"][t]   # the denoiser keeps a_{t-1}
                inp["seed"] = MEM_SEED + MEMORY_KINDS.index(kind) * 100_000 + b0
                inps += [inp] * R_MEM
            gp, gg, gv = generate(model, cfg, inps)
            ii = np.repeat([i for i, _, _ in batch], R_MEM)
            cell = np.array([info[i][0] for i in ii]); start = np.array([info[i][1] for i in ii]); end = np.array([info[i][2] for i in ii])
            fix = torch.stack([enc_g[rows[i][0]]["fix"][rows[i][1]] for i in ii])
            win = M.window(fix).numpy()
            read, ok = read_glimpse(rd, READS[kind], gg, (win == cell[:, None]).argmax(1), (fix[:, 0] * 9 + fix[:, 1]).numpy(),
                                    start, end, np.ones(len(ii), bool))
            for k, v in (("ctx", ii), ("read_ok", ok), ("success", read == end), ("copy", read == start), ("start", start)):
                res[k].append(v)
        out[variant] = {k: np.concatenate(v) for k, v in res.items()}
    return out


@torch.no_grad()
def perception(model, cfg, L, rows, info, enc_g):
    """Real frames of the eye-on rows: per row (one per context) the consequence cell's and the window's mean pixel NLL
    under the fovea reconstruction and under TSP."""
    eps = S.episodes(L)
    by_ep = {}
    for i, r in enumerate(rows):
        if (M.window(enc_g[r[0]]["fix"][r[1]]) == info[i][0]).any():
            by_ep.setdefault(r[0], []).append((i, r[1]))
    out = {k: [] for k in ("ctx", "start", "rec_cell", "tsp_cell", "rec_window", "tsp_window")}
    for e, items in by_ep.items():
        ts = [t for _, t in items]
        fx = enc_g[e]["fix"][ts].to(DEV)
        frames = D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"][torch.as_tensor(ts)].to(DEV)  # data.gather's source
        pix = M.cell_pixels(M.crop(frames, fx))                              # [n, 9, 147]
        tk = {k: enc_g[e][k][ts].float().to(DEV) for k in ("p", "g", "v", "bp")}
        cells = M.window(fx)
        with TR.autocast(DEV, cfg):
            rec = M.pixel_nll(*model.pixels(model.tsp.reconstruct(tk["g"], tk["v"], cells), tk["v"], cells), pix).mean(-1)
            tsp = M.pixel_nll(*model.pixels(model.tsp.predict(tk["bp"], tk["p"], tk["v"], cells), tk["v"], cells), pix).mean(-1)
        slot = torch.stack([(cells[j] == info[i][0]).float().argmax() for j, (i, _) in enumerate(items)])
        ar = torch.arange(len(items), device=DEV)
        for k, v in (("ctx", [i for i, _ in items]), ("start", [info[i][1] for i, _ in items]),
                     ("rec_cell", rec[ar, slot].float().cpu()), ("tsp_cell", tsp[ar, slot].float().cpu()),
                     ("rec_window", rec.float().mean(-1).cpu()), ("tsp_window", tsp.float().mean(-1).cpu())):
            out[k].append(np.asarray(v, float) if k not in ("ctx", "start") else np.asarray(v))
    return {k: np.concatenate(v) for k, v in out.items()} if by_ep else None


def regression(enc_p, L):
    """Dev, S1-seeded fixations, frames t >= 1: reward NLL, continuation BCE, TSP NLL (S.encode's per-frame values)."""
    ep = np.concatenate([np.full(len(r["p"]) - 1, e) for e, r in enumerate(enc_p)])
    one = np.ones(len(ep), bool)
    pt = np.concatenate([r["p_true"][1:].numpy() for r in enc_p]); pc = np.concatenate([r["p_cont"][1:].numpy() for r in enc_p])
    term = np.concatenate([r["terminated"][1:].numpy() for r in enc_p]).astype(bool)
    tsp = np.concatenate([r["tsp"][1:].numpy() for r in enc_p])
    eps_ = 1e-6
    return {"reward_nll": G.estimate(-np.log(np.clip(pt, eps_, 1)), one, ep),
            "continuation_bce": G.estimate(-np.where(term, np.log(np.clip(1 - pc, eps_, 1)), np.log(np.clip(pc, eps_, 1))), one, ep),
            "tsp_nll": G.estimate(tsp, one, ep)}


def separation(L, enc):
    """Periphery value separation per transition: |mean(v + 1) - mean(v)| over the RMS within-value spread."""
    st = IP.value_stats(L, enc)
    out = {}
    for name, s in st.items():
        out[name] = {}
        for v in range(9):
            if not (torch.isnan(s["means"][v]).any() or torch.isnan(s["means"][v + 1]).any()):
                step = float((s["means"][v + 1] - s["means"][v]).norm())
                out[name][v] = {"step": step, "step_over_spread": step / float(((s["spread"][v] ** 2 + s["spread"][v + 1] ** 2) / 2) ** 0.5)}
    return out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    args = sys.argv[2:]
    n_ep = int(args.pop(0).split("=")[1]) if args and args[0].startswith("--episodes=") else 0
    n_rows = 50 if n_ep else N
    pairs = [a.split("=", 1) for a in args]
    assert pairs and all(len(p) == 2 for p in pairs), "arguments: LABEL=CKPT"
    rule = {int(k): tuple(v) for k, v in json.loads((PM.HERE / "map.json").read_text())["rule"].items()}
    L = S.first_episodes(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    Ltr = S.first_episodes(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_train400_labels.npz")), n_ep)
    res = {"checkpoints": {}, "differences": {}}
    per, mem, perc = {}, {}, {}
    rows = info = None
    for label, path in pairs:
        model, cfg, sha = load(path)
        enc = {"periphery": S.encode(model, L, cfg), "guided": EV.encode_with(model, L, cfg, PM.window_of_state(L, rule))}
        enc_tr = EV.encode_with(model, Ltr, cfg, PM.window_of_state(Ltr, rule))
        rd, qual = IN.fit(enc_tr, Ltr, enc["guided"], L)
        if rows is None:
            rows, counts = rows_of(L, enc["periphery"], n_rows)
            info = {k: EV.row_info(k, rows[k], L) for k in rows}
            res.update(rows={k: len(v) for k, v in rows.items()}, failed_attempts_available=counts, dev_episodes=int(len(L["ep_T"])))
        e0 = enc["periphery"]
        hm, tm, means = CAT.hud_means(L, e0), CAT.table_means(L, e0), FC.value_means(L, e0)
        means_all = {**means, 68: hm["wood"], **{c: {0: v} for c, v in tm.items()}}
        row = {"path": path, "sha256": sha, "instrument_qualification": qual, "separation": separation(L, e0),
               "regression": regression(e0, L), "prediction": {}, "perception": {}, "memory": {}}
        for route in ("guided", "periphery"):
            row["prediction"][route] = {}
            for kind in rows:
                if route == "guided" and kind == "move":
                    continue
                gen = torch.Generator(device=DEV).manual_seed(EV.SEEDS[kind] + (0 if route == "periphery" else 100))
                if kind in EV.CAT_KINDS:
                    o = CAT.run(model, enc[route], L, rows[kind], kind, hm, tm, cfg, gen, heun=False, keep=True)
                else:
                    o = FC.run(model, enc[route], rows[kind], means_all, cfg, gen, keep=True)
                ep = np.array([rows[kind][i][0] for i in o["row_ctx"]])
                if kind == "move":
                    ok = o["valid"]
                    row["prediction"][route][kind] = {"shift_sample": est(o["f_gen"], ok, ep),
                                                      "shift_field_6.67": est(o["f_den"][:, FC.SIGMAS.index(6.67)], ok, ep)}
                    per[(label, route, kind)] = {"row_ctx": o["row_ctx"], "f_gen": o["f_gen"], "valid": ok}
                    continue
                oc = outcomes(o, rows[kind], info[kind], enc[route], rd, kind, route)
                row["prediction"][route][kind] = summarise(oc, ep, kind)
                per[(label, route, kind)] = {**oc, "row_ctx": o["row_ctx"]}
                np.savez_compressed(out_dir / f"{label}_{route}_{kind}.npz", row_ctx=o["row_ctx"], **oc)
        for kind in CONSEQ:
            pc = perception(model, cfg, L, rows[kind], info[kind], enc["guided"])
            if pc is None:
                continue
            perc[(label, kind)] = pc
            ep = np.array([rows[kind][i][0] for i in pc["ctx"]])
            row["perception"][kind] = {g: {k: est(pc[k], m, ep) for k in ("rec_cell", "tsp_cell", "rec_window", "tsp_window")}
                                       for g, m in groups(kind, pc["start"]).items()}
            np.savez_compressed(out_dir / f"{label}_perception_{kind}.npz", **pc)
        for kind in MEMORY_KINDS:
            mr = memory_reliance(model, cfg, rows[kind], info[kind], enc["guided"], rd, kind)
            mem[(label, kind)] = mr
            a, b = mr["rerun"], mr["noop_belief"]
            assert (a["ctx"] == b["ctx"]).all()
            ep = np.array([rows[kind][i][0] for i in a["ctx"]])
            both = a["read_ok"] & b["read_ok"]
            row["memory"][kind] = {g: {"rerun_success": est(a["success"], a["read_ok"] & m, ep),
                                       "noop_belief_success": est(b["success"], b["read_ok"] & m, ep),
                                       "reliance": pdiff(b["success"], a["success"], both & m, ep)}
                                   for g, m in groups(kind, a["start"]).items()}
            np.savez_compressed(out_dir / f"{label}_memory_{kind}.npz", **{f"{v}_{k}": x for v, d in mr.items() for k, x in d.items()})
        if label == "base" or label.endswith("_1500"):
            tr, dv = S.cell_table(enc_tr, Ltr), S.cell_table(e0, L)
            reader, kp, _ = S.token_probe(tr, dv, 0)
            row["entities"] = {"reader_retention": S.retention(dv, kp), "counts": EV.entity_counts(model, e0, cfg, reader)}
            del tr, dv, reader
        row["class_losses"] = EV.class_losses(model, e0, L, cfg)
        res["checkpoints"][label] = row
        print(label, json.dumps({r: {k: {g: (None if (v.get("success") or v.get("false_change")) is None else
                                             round((v.get("success") or v.get("false_change"))["est"], 3)) for g, v in s.items()}
                                     for k, s in d.items() if k != "move"} for r, d in row["prediction"].items()}), flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
        del model, enc, enc_tr
        torch.cuda.empty_cache()
    labels = [l for l, _ in pairs]
    combos = [("base", l) for l in labels if l != "base"]
    combos += [(f"random_{u}", f"guided_{u}") for u in sorted({l.split("_")[1] for l in labels if "_" in l})]
    for a, b in combos:
        if a not in labels or b not in labels:
            continue
        d = {"prediction": {}, "perception": {}, "memory": {}}
        for route in ("guided", "periphery"):
            d["prediction"][route] = {}
            for kind in rows:
                if (a, route, kind) not in per:
                    continue
                A, B = per[(a, route, kind)], per[(b, route, kind)]
                assert (A["row_ctx"] == B["row_ctx"]).all()
                ep = np.array([rows[kind][i][0] for i in A["row_ctx"]])
                if kind == "move":
                    d["prediction"][route][kind] = {"shift_sample": pdiff(A["f_gen"], B["f_gen"], A["valid"] & B["valid"], ep)}
                    continue
                d["prediction"][route][kind] = {g: {k: pdiff(A[k], B[k], A["read_ok"] & B["read_ok"] & m, ep) for k in keys_of(kind)}
                                                for g, m in groups(kind, A["start"]).items()}
        for kind in CONSEQ:
            if (a, kind) in perc and (b, kind) in perc:
                A, B = perc[(a, kind)], perc[(b, kind)]
                assert (A["ctx"] == B["ctx"]).all()
                ep = np.array([rows[kind][i][0] for i in A["ctx"]])
                d["perception"][kind] = {g: {k: pdiff(A[k], B[k], m, ep) for k in ("rec_cell", "tsp_cell", "rec_window", "tsp_window")}
                                         for g, m in groups(kind, A["start"]).items()}
        for kind in MEMORY_KINDS:
            A, B = mem[(a, kind)], mem[(b, kind)]
            assert (A["rerun"]["ctx"] == B["rerun"]["ctx"]).all()
            ep = np.array([rows[kind][i][0] for i in A["rerun"]["ctx"]])
            ra = A["rerun"]["success"].astype(float) - A["noop_belief"]["success"].astype(float)
            rb = B["rerun"]["success"].astype(float) - B["noop_belief"]["success"].astype(float)
            ok = A["rerun"]["read_ok"] & A["noop_belief"]["read_ok"] & B["rerun"]["read_ok"] & B["noop_belief"]["read_ok"]
            d["memory"][kind] = {g: {"reliance": pdiff(ra, rb, ok & m, ep),
                                     "rerun_success": pdiff(A["rerun"]["success"], B["rerun"]["success"],
                                                            A["rerun"]["read_ok"] & B["rerun"]["read_ok"] & m, ep)}
                                 for g, m in groups(kind, A["rerun"]["start"]).items()}
        res["differences"][f"{b}-{a}"] = d
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print("saved", out_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
