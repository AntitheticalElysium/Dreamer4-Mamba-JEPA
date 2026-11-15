"""F6.1 dissection, checks 13-15 (PLAN.md): why the denoiser does not draw the awake clocks from the per-cell traces.

  separable  check 13 (CPU): check 11's label-free next-change predictor refitted line for line (APs reproduced), scored
             on both branches of every exact-history pair at the due transition; saved for check 15.
  expect     check 14 (GPU): on the same pairs and cells as check 12, the denoiser's prediction at each sampler step
             projected on the due branch's true change (first step: its conditional mean), F6.1 and traces at 2,000.
  catalogue  check 15 (GPU, after 13): the same reading on natural DEV fold-4 frames per single-cell event class.

    GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m artifacts.experiments.20261115_f61_dissection.use_check separable
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.use_check expect
    MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261115_f61_dissection.use_check catalogue
"""
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
G = importlib.import_module("artifacts.experiments.20261115_gaze_learnability.gaze")
TT = importlib.import_module("artifacts.experiments.20261115_f61_dissection.token_trace")
TF = importlib.import_module("artifacts.experiments.20261115_f61_dissection.traces_ft")
M, C, D, TR, DS, TJ = TF.M, TF.C, TF.D, TF.TR, TF.DS, TF.TJ
PREDICTOR = TF.OUT / "next_change.txt"
CLOCK = {"food_down": 64, "drink_down": 65, "energy_down_awake": 66, "energy_up_asleep": 66, "recovery": 63}


def feat(x, tr):
    """token_trace's feature row for every cell: current change, the two traces, their difference, the cell index."""
    return np.c_[x, tr, tr[:, 0] - tr[:, 1], np.arange(81)]


def change(p):
    return np.r_[np.zeros((1, 81), np.float32), np.square(p[1:] - p[:-1]).sum(-1)]   # x[t]: change into frame t


def coarse_np(model, frames):
    fr = torch.as_tensor(frames)
    return torch.cat([C.coarse(model.perception, z)[0] for z in fr.split(512)]).float().numpy()


def ci(d, seed=1111):
    d = np.asarray(d, float)
    ix = np.random.default_rng(seed).integers(len(d), size=(10000, len(d)))
    return [float(d.mean()), *np.quantile(d[ix].mean(1), [.025, .975]).tolist()]


@torch.no_grad()
def separable():
    import lightgbm as lgb
    torch.set_num_threads(6)
    model, cfg = G.CR.NP.IF.load(TF.CKPT.resolve())
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))               # token_trace.main, line for line
    y, ok = G.LB.events(L)
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = G.S.episodes(L)
    train_eps = [e for e in range(len(eps)) if e % 5 != 4][:400]
    test_eps = [e for e in range(len(eps)) if e % 5 == 4]
    rng = np.random.default_rng(0)
    X, Y, rows = [], [], []
    for e in train_eps + test_eps:
        p = coarse_np(model, G.D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy())
        x = change(p)
        tr = TT.traces(x)
        T = len(p)
        if e in train_eps:
            for t in rng.choice(np.arange(1, T - 1), min(T - 2, 120), replace=False):
                X.append(feat(x[t], tr[t]))
                Y.append(x[t + 1] > 1.0)
        else:
            for t in range(9, T):
                g = off[e] + t
                for j, (n, c) in enumerate(TT.CELL.items()):
                    if after[g, j]:
                        rows.append((j, int(y[g, j]), *feat(x[t - 1], tr[t - 1])[c]))
    X, Y = np.concatenate(X), np.concatenate(Y)
    par = dict(objective="binary", learning_rate=.06, num_leaves=31, min_data_in_leaf=50, num_threads=6, verbosity=-1, seed=0)
    m = lgb.train(par, lgb.Dataset(X, Y.astype(float)), num_boost_round=200)
    R = np.array(rows, dtype=np.float64)
    s = m.predict(R[:, 2:])
    ref = json.loads((HERE / "token_trace.json").read_text())
    res = {"reproduced_ap": {}, "train_cells": int(len(Y))}
    for j, n in enumerate(TT.CELL):
        k = R[:, 0] == j
        res["reproduced_ap"][n] = G.AP(R[k, 1].astype(bool), s[k])
        assert abs(res["reproduced_ap"][n] - ref[n]["full"]["ap"]) < 0.002, (n, res["reproduced_ap"][n], ref[n]["full"]["ap"])
    PREDICTOR.parent.mkdir(parents=True, exist_ok=True)
    m.save_model(str(PREDICTOR))
    lt = np.log1p(X[:, 1:3])                                            # check 12's input, log(1 + trace), every cell
    hud = np.isin(X[:, 4], (63, 64, 65, 66))
    res["log_trace_sd"] = {"all_cells": lt.std(0).tolist(), "hud_digit_cells": lt[hud].std(0).tolist()}
    for name, brows, slot, sign in TF.pairs():
        P, Z, S8 = [], [], []
        for prefix, frames, rewards, terms, q, acts, truth in brows:
            sc, lz, s8 = [], [], []
            for b in (0, 1):
                x = change(coarse_np(model, np.concatenate([prefix[0], frames[b, :q + 1]])))
                tr = TT.traces(x)
                sc.append(m.predict(feat(x[-1], tr[-1])[slot:slot + 1])[0])
                lz.append(np.log1p(tr[-1, slot]))
                s8.append(m.predict(feat(x[-1], TT.traces(x[-9:], start=1)[-1])[slot:slot + 1])[0])   # check 11's 8-frame
            P.append(sc), Z.append(lz[0] - lz[1]), S8.append(s8)
        P, Z, S8 = map(np.array, (P, Z, S8))
        res[name] = {"pairs": len(P), "p_due": float(P[:, 0].mean()), "p_delayed": float(P[:, 1].mean()),
                     "due_minus_delayed": ci(P[:, 0] - P[:, 1]),
                     "due_above_delayed": float(np.mean((P[:, 0] > P[:, 1]) + 0.5 * (P[:, 0] == P[:, 1]))),
                     "p_8frame_due_delayed": S8.mean(0).tolist(),
                     "log_trace_diff_in_sd_all": (Z.mean(0) / np.array(res["log_trace_sd"]["all_cells"])).tolist(),
                     "log_trace_diff_in_sd_hud": (Z.mean(0) / np.array(res["log_trace_sd"]["hud_digit_cells"])).tolist(),
                     "per_pair": P.round(5).tolist()}
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "per_pair"}), flush=True)
    (HERE / "separable.json").write_text(json.dumps(res, indent=1) + "\n")


def physical(den, shapes, prev_action, memory, fix, cache, cfg, gen):
    """model.sample's physical stage with its noise (the glimpse stage after it never changes p or v); also returns
    the prediction of p at each of the K evaluations."""
    ladder, e, d = M.shortcut_ladder(cfg)
    k, dev, B = den.k_max, memory.device, memory.shape[0]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    code = lambda s_, e_: torch.full((B, 1), M.shortcut_code(s_, e_, k), device=dev)
    move = lambda z, x, tau: z + (x.float().clamp(-clamp, clamp) - z) / (1 - tau) * d
    p, v = randn((B, 1, *shapes[0])), randn((B, 1, *shapes[2]))
    g0 = torch.zeros((B, 1, *shapes[1]), device=dev)
    xs = []
    for s_, tau in ladder:
        out, _ = den.step((p, g0, v), code(s_, e), prev_action, memory, fix, cache)
        xs.append(out[0].float().clamp(-clamp, clamp))
        p, v = move(p, out[0], tau), move(v, out[2], tau)
    return p, v, xs


def load(trained=True):
    model, cfg = DS.NP.IF.load(TF.CKPT.resolve())
    for p in model.parameters():
        p.requires_grad_(False)
    TF.install(model.denoiser)
    model.eval()
    base = {k: v.clone() for k, v in model.denoiser.state_dict().items()}
    sd = torch.load(TF.OUT / "denoiser_2000.pt", map_location=DS.DEV, weights_only=False)["denoiser"]
    return model, cfg, base, sd


def kappa(z, last, delta):
    """Per stream: (z - last) . delta / |delta|^2 at one cell."""
    return ((z.float() - last) @ delta) / delta.square().sum().clamp_min(1e-6)


@torch.no_grad()
def expect():
    model, cfg, base, trained = load()
    den, dev = model.denoiser, DS.DEV
    readers = TJ.readers(model, cfg, slots=(63, 64, 66))
    banks = TF.pairs()
    ref = json.loads((HERE / "traces_ft.json").read_text())["reads"]
    res = {}
    for arm, sd, u in (("f61", base, "0"), ("traces_2000", trained, "2000")):
        den.load_state_dict(sd)
        res[arm] = {}
        for name, rows, slot, sign in banks:
            cells = []
            for i, (prefix, frames, rewards, terms, q, acts, truth) in enumerate(rows):   # traces_ft.pair_read's setup
                old_f, old_a, old_r, old_d = prefix
                n = len(old_f)
                f = np.concatenate([np.repeat(old_f[None], 2, 0), frames[:, :q + 1]], 1)
                a = np.concatenate([np.repeat(old_a[None], 2, 0), acts], 1)
                rw = np.column_stack([np.repeat(np.r_[0., old_r][None], 2, 0), rewards[:, :q]])
                td = np.column_stack([np.repeat(np.r_[False, old_d][None], 2, 0), terms[:, :q]])
                fix = torch.as_tensor(np.repeat(D.fixation_prior(np.random.default_rng(12000 + i), (len(f[0]),))[None], 2, 0), device=dev)
                query = torch.as_tensor(D.fixation_prior(np.random.default_rng(15000 + i), ()), device=dev)
                act = torch.as_tensor(a, device=dev)
                prev = torch.cat([act.new_full((2, 1), -1), act[:, :-1]], 1)
                ft = torch.as_tensor(f, device=dev)
                pc = torch.stack([C.coarse(model.perception, ft[b])[0] for b in (0, 1)])
                first = torch.zeros(pc.shape[:2], dtype=torch.bool, device=dev); first[:, 0] = True
                tin, tst = TF.traces(pc, first, torch.zeros(2, TF.N, 2, device=dev))
                cell = {}
                with TR.autocast(dev, cfg):
                    p = model.perception(ft, fix)
                    y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(p), torch.as_tensor(rw, device=dev, dtype=torch.float32),
                                                            torch.as_tensor(td, device=dev), act), model.belief.initial(2, dev))
                    bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
                    nxt = model.perception(torch.as_tensor(frames[0:1, q + 1:q + 2], device=dev), query[None, None])
                    last, delta = p.p[0, -1, slot].float(), nxt.p[0, 0, slot].float() - p.p[0, -1, slot].float()
                    for m, c in ((0, 0), (1, 1), (0, 1), (1, 0)):
                        rng = torch.Generator(device=dev).manual_seed(14000 + i)
                        cache = den.new_cache()
                        for t in range(n + q - 7, n + q + 1):
                            den.trace_input = tin[c:c + 1, t:t + 1]
                            cache = M.context_pass(den, tuple(z[0:1, t:t + 1] for z in p.parts()), prev[0:1, t:t + 1],
                                                   model.belief.memory(bp[m:m + 1, t:t + 1]), fix[0:1, t:t + 1], cache, cfg, rng)
                        cache = {"t": cache["t"], "kv": [tuple(z.expand(DS.DRAWS, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}
                        den.trace_input = tst[c:c + 1, -1:].expand(DS.DRAWS, -1, -1, -1)
                        pz, vz, xs = physical(den, [tuple(x.shape[2:]) for x in p.parts()], act.new_zeros(DS.DRAWS, 1),
                                              model.belief.memory(y[m:m + 1, -1:]).expand(DS.DRAWS, -1, -1, -1),
                                              query[None, None].expand(DS.DRAWS, 1, -1), cache, cfg,
                                              torch.Generator(device=dev).manual_seed(16000 + i))
                        digits = readers[slot][0](torch.cat([pz[:, 0, slot], vz[:, 0]], 1).float())["y"].argmax(-1).cpu().numpy()
                        kf = kappa(pz[:, 0, slot], last, delta)
                        cell[f"m{m}c{c}"] = {"rate": float(((digits < truth[0, 0]) if sign < 0 else (digits > truth[0, 0])).mean()),
                                             "kappa": [float(kappa(z[:, 0, slot], last, delta).mean()) for z in xs],
                                             "kappa_final": float(kf.mean()), "drawn_kappa": float((kf > .5).float().mean())}
                cells.append(cell)
            out = {"pairs": len(cells)}
            for key in cells[0]:
                out[key] = {"rate": float(np.mean([c[key]["rate"] for c in cells])),
                            "kappa": np.mean([c[key]["kappa"] for c in cells], 0).tolist(),
                            "kappa_final": float(np.mean([c[key]["kappa_final"] for c in cells])),
                            "drawn_kappa": float(np.mean([c[key]["drawn_kappa"] for c in cells]))}
                dev_rate = abs(out[key]["rate"] - ref[u][name]["rate"][key])
                assert dev_rate <= 2 / (DS.DRAWS * len(cells)), (arm, name, key, out[key]["rate"], ref[u][name]["rate"][key])
                out[key]["rate_vs_check12"] = dev_rate
            k0 = lambda key: np.array([c[key]["kappa"][0] for c in cells])
            out["first_step_due_minus_delayed"] = ci(k0("m0c0") - k0("m1c1"))
            out["first_step_trace_effect"] = ci((k0("m0c0") + k0("m1c0") - k0("m0c1") - k0("m1c1")) / 2)
            out["first_step_memory_effect"] = ci((k0("m0c0") + k0("m0c1") - k0("m1c0") - k0("m1c1")) / 2)
            res[arm][name] = out
            print(arm, name, json.dumps(out), flush=True)
    (HERE / "expect.json").write_text(json.dumps(res, indent=1) + "\n")


def classes(L, after):
    """name -> (event mask, eligible-without-event mask, cell or None (measured), where the cue lies), per state t
    (transition t-1 -> t), t >= 9 in its episode, DEV fold 4."""
    y, ok = G.LB.events(L)
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    n = len(L["s_body"])
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"] + 1)
    base = (np.arange(n) - off[ep] >= 9) & (ep % 5 == 4)
    body, inv = L["s_body"].astype(int), L["s_inv"].astype(int)
    pb, pi = np.roll(body, 1, 0), np.roll(inv, 1, 0)                   # t >= 9: never across an episode start
    out = {}
    for j, name in enumerate(G.LB.EVENTS):
        cue = "older (time)" if j < 4 else "older (memory, rewarded)"
        out[name] = (y[:, j] & base, ok[:, j] & ~y[:, j] & base, CLOCK[name], cue)
        if j < 3:
            out[name + "_after_reset"] = (y[:, j] & after[:, j] & base, ok[:, j] & after[:, j] & ~y[:, j] & base, CLOCK[name], cue)
    dmg = body[:, 0] < pb[:, 0]
    out["damage"] = (dmg & base, ~dmg & base, 63, "mixed")
    for col, name in ((1, "food_up"), (2, "drink_up")):
        ev = body[:, col] > pb[:, col]
        out[name] = (ev & base, ~ev & base, 63 + col, "window (action)")
    for i in range(inv.shape[1]):
        for sgn, s in ((1, "up"), (-1, "down")):
            ev = np.sign(inv[:, i] - pi[:, i]) == sgn
            if (ev & base).sum() >= 30:
                out[f"inv{i}_{s}"] = (ev & base, ~ev & base, None, "window (action)")
    return out


@torch.no_grad()
def catalogue(events=128, quiet=64, batch=8):
    import lightgbm as lgb
    model, cfg, base, trained = load()
    den, dev = model.denoiser, DS.DEV
    booster = lgb.Booster(model_file=str(PREDICTOR))
    L = dict(np.load(G.S.LABELS / "s0_dev_labels.npz"))
    after = G.cohorts(L)["after_reset"]
    off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    eps = G.S.episodes(L)
    ep_of = np.repeat(np.arange(len(eps)), L["ep_T"] + 1)
    cls = classes(L, after)
    # pass 1: token changes on fold 4 (inventory cells measured)
    xs = {}
    for e in range(4, len(eps), 5):
        ft = torch.as_tensor(G.D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"].numpy(), device=dev)
        xs[e] = change(torch.cat([C.coarse(model.perception, z)[0] for z in ft.split(512)]).float().cpu().numpy())
    X = np.concatenate([xs[e] for e in range(4, len(eps), 5)])
    rows_g = np.concatenate([off[e] + np.arange(len(xs[e])) for e in range(4, len(eps), 5)])
    pos = np.full(len(L["s_body"]), -1); pos[rows_g] = np.arange(len(rows_g))
    cellmap = {}
    for name, (ev, no, cell, cue) in list(cls.items()):
        if cell is None:
            mean = X[pos[np.flatnonzero(ev)], 63:].mean(0)
            cell = 63 + int(mean.argmax())
            cellmap[name] = {"cell": cell, "change": float(mean.max()), "next": float(np.sort(mean)[-2])}
            cls[name] = (ev, no, cell, cue)
    # rows
    rows = []                                                           # (state g, class, event?)
    for ci_, (name, (ev, no, cell, cue)) in enumerate(cls.items()):
        r = np.random.default_rng(1515 + ci_)
        for flag, mask, k in ((1, ev, events), (0, no, quiet)):
            idx = np.flatnonzero(mask)
            rows += [(int(g), name, flag) for g in r.choice(idx, min(k, len(idx)), replace=False)]
    arms = (("f61", base, False), ("traces_own", trained, True), ("traces_zero", trained, False))
    rec = {a: [] for a, _, _ in arms}
    acts = np.zeros(81)
    by_ep = {}
    for j, (g, name, flag) in enumerate(rows):
        by_ep.setdefault(ep_of[g], []).append(j)
    shapes = None
    for e, js in sorted(by_ep.items()):
        ef = G.D.RP.episode_fields(eps[e].shard, eps[e].slot)
        f = ef["observations"].numpy()
        n = len(f)
        a = np.r_[ef["actions_taken"].numpy(), 0]
        rw, td = np.r_[0., ef["rewards"].numpy()], np.r_[False, ef["terminated"].numpy()]
        assert len(a) == len(rw) == len(td) == n == L["ep_T"][e] + 1
        fix = torch.as_tensor(D.fixation_prior(np.random.default_rng([15000, int(e)]), (1, n)), device=dev)
        act = torch.as_tensor(a[None], device=dev)
        prev = torch.cat([act.new_full((1, 1), -1), act[:, :-1]], 1)
        ft = torch.as_tensor(f[None], device=dev)
        pc = torch.cat([C.coarse(model.perception, z)[0] for z in ft[0].split(512)])[None]
        first = torch.zeros(1, n, dtype=torch.bool, device=dev); first[0, 0] = True
        tin, _ = TF.traces(pc, first, torch.zeros(1, TF.N, 2, device=dev))
        x = change(pc[0].float().cpu().numpy())
        tr = TT.traces(x)
        with TR.autocast(dev, cfg):
            parts = [model.perception(ft[:, s:s + 512], fix[:, s:s + 512]) for s in range(0, n, 512)]
            parts = tuple(torch.cat(z, 1) for z in zip(*(q.parts() for q in parts)))
            pp = M.Percept(*parts, fix)
            y, _ = model.belief(model.belief.tokens(model.belief.percept_embed(pp), torch.as_tensor(rw[None], device=dev, dtype=torch.float32),
                                                    torch.as_tensor(td[None], device=dev), act), model.belief.initial(1, dev))
            bp = torch.cat([torch.zeros_like(y[:, :1]), y[:, 1::2][:, :-1]], 1)
            shapes = [tuple(z.shape[2:]) for z in parts]
            for b0 in range(0, len(js), batch):
                jb = js[b0:b0 + batch]
                ts = [int(rows[j][0] - off[e]) for j in jb]
                cells = [cls[rows[j][1]][2] for j in jb]
                B = len(ts)
                last = torch.stack([parts[0][0, t - 1, c].float() for t, c in zip(ts, cells)])
                delta = torch.stack([parts[0][0, t, c].float() for t, c in zip(ts, cells)]) - last
                ptrace = booster.predict(np.stack([feat(x[t - 1], tr[t - 1])[c] for t, c in zip(ts, cells)]))
                first_pred = {}
                for arm, sd, own in arms:
                    den.load_state_dict(sd)
                    seed = 1_000_003 * int(e) + b0
                    rng = torch.Generator(device=dev).manual_seed(seed)
                    cache = den.new_cache()
                    for k in range(8):
                        fi = [t - 8 + k for t in ts]
                        den.trace_input = tin[0, fi][:, None] if own else torch.zeros(B, 1, TF.N, 2, device=dev)
                        cache = M.context_pass(den, tuple(z[0, fi][:, None] for z in parts), prev[0, fi][:, None],
                                               model.belief.memory(bp[0, fi][:, None]), fix[0, fi][:, None], cache, cfg, rng)
                    R = DS.DRAWS
                    cache = {"t": cache["t"], "kv": [tuple(z.repeat_interleave(R, 0) for z in kv) for kv in cache["kv"]]}
                    den.trace_input = (tin[0, ts][:, None] if own else torch.zeros(B, 1, TF.N, 2, device=dev)).repeat_interleave(R, 0)
                    pz, vz, xp = physical(den, shapes, prev[0, ts][:, None].repeat_interleave(R, 0),
                                          model.belief.memory(bp[0, ts][:, None]).repeat_interleave(R, 0),
                                          fix[0, ts][:, None].repeat_interleave(R, 0), cache, cfg,
                                          torch.Generator(device=dev).manual_seed(seed + 1))
                    first_pred[arm] = xp[0][:, 0].reshape(B, R, 81, -1)
                    for r_, (j, c) in enumerate(zip(jb, cells)):
                        sl = slice(r_ * R, (r_ + 1) * R)
                        fin = pz[sl, 0, c]
                        rec[arm].append({"row": j, "class": rows[j][1], "event": rows[j][2], "p_trace": float(ptrace[r_]),
                                         "kappa": [float(kappa(z[sl, 0, c], last[r_], delta[r_]).mean()) for z in xp],
                                         "kappa_final": float(kappa(fin, last[r_], delta[r_]).mean()),
                                         "drawn_toward": float((kappa(fin, last[r_], delta[r_]) > .5).float().mean()),
                                         "drawn_change": float(((fin.float() - last[r_]).square().sum(-1) > 1).float().mean()),
                                         "true_change": float(delta[r_].square().sum())})
                acts += (first_pred["traces_own"] - first_pred["traces_zero"]).square().sum(-1).mean(1).sum(0).cpu().numpy()
        print("episode", e, "rows", len(js), flush=True)
    res = {"cells": {k: int(v[2]) for k, v in cls.items()}, "cue": {k: v[3] for k, v in cls.items()}, "inventory_cells": cellmap,
           "trace_acts_per_cell": (acts / len(rows)).tolist()}
    groups = {"map": range(0, 63), "hud_digits": range(63, 67), "hud_inventory": range(67, 81)}
    res["trace_acts_by_group"] = {k: float(np.mean(res["trace_acts_per_cell"][v.start:v.stop])) for k, v in groups.items()}
    for arm in rec:
        res[arm] = {}
        for name in cls:
            ev = [r for r in rec[arm] if r["class"] == name and r["event"]]
            no = [r for r in rec[arm] if r["class"] == name and not r["event"]]
            if not ev:
                continue
            res[arm][name] = {"events": len(ev), "first_step": ci([r["kappa"][0] for r in ev]),
                              "steps": np.mean([r["kappa"] for r in ev], 0).tolist(),
                              "kappa_final": float(np.mean([r["kappa_final"] for r in ev])),
                              "drawn_toward": ci([r["drawn_toward"] for r in ev]),
                              "drawn_change_event": float(np.mean([r["drawn_change"] for r in ev])),
                              "drawn_change_quiet": float(np.mean([r["drawn_change"] for r in no])) if no else None,
                              "true_change": float(np.mean([r["true_change"] for r in ev])),
                              "p_trace_event": float(np.mean([r["p_trace"] for r in ev])),
                              "p_trace_quiet": float(np.mean([r["p_trace"] for r in no])) if no else None}
            print(arm, name, json.dumps(res[arm][name]), flush=True)
    (HERE / "catalogue.json").write_text(json.dumps(res, indent=1) + "\n")
    torch.save(rec, TF.OUT / "catalogue_rows.pt")


if __name__ == "__main__":
    {"separable": separable, "expect": expect, "catalogue": catalogue}[sys.argv[1]]()
