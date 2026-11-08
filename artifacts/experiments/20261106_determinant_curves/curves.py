"""Determinant curves (PLAN.md; user 11-06, "the second experiment"): does P1-F3 condition each failing consequence on
what determines it? One-step futures at 16 sampler steps from real dev histories through the actor interface's
production path, unchanged (interface.futures: 8 real context frames, the fixed random gaze, K = 4), on populations
defined by the situation at t - 1 and the action, stratified by the consequence's determinant (Craftax Classic
game_logic.py). Each future is read by the outcome heads (interface.futures) and by frozen readers of the generated
frame (asleep, HUD values, the faced cell's entity and terrain); the same readers read the recorded successor, the
simulator labels give the truth. The generated percepts are captured from model.sample's return inside futures.
R0: with the capture active, futures reproduces F3's 16-step dev COLLECT_COAL reading exactly.
Usage (project venv, GPU): MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_determinant_curves.curves
  [--smoke | --summarise]   (--summarise: readings/raw.npz -> readings/curves.json, CPU)
"""
import contextlib, copy, importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

IF = importlib.import_module("artifacts.experiments.20261030_actor_interface.interface")
HERE = Path(__file__).resolve().parent
CKPT = "artifacts/experiments/20261105_p1f3/run/ckpt_10.pt"
S1_READER = TR.ROOT / "artifacts/experiments/20261105_p1f3/run/s1/results.reader.pt"
DEV_READING = TR.ROOT / "artifacts/experiments/20261105_p1f3/readings/interface_steps.json"
LABELS = TR.ROOT / "artifacts/eda/s0"
STEPS, K, DEV, P = 16, 4, IF.DEV, IF.P
HUD = {63: ("s_body", 0), 64: ("s_body", 1), 66: ("s_body", 3), 68: ("s_inv", 0), 69: ("s_inv", 1), 70: ("s_inv", 2),
       71: ("s_inv", 3)}                                                 # renderer row 7: health food (drink) energy . wood ..
MINING = {"coal": (8, 7, 70, 6), "iron": (9, 7, 71, 7), "wood": (5, 2, 68, None), "stone": (4, 7, 69, 6)}  # block, left
SWORDS = ((9, 2), (10, 3), (11, 5))                                      # behind, cell, tool; s_inv column, damage (else 1)
DO, PLAYER, WAKE_UP, EAT_COW, DEFEAT_ZOMBIE = 5, 31, 15, 2, 8
MOB = {"zombie": ("zombies", 1), "cow": ("cows", 2)}                    # S1 reader kinds: background, zombie, cow, ...
CAPS = {"e<=5": 500, "e6": 500, "e7_fresh": 300, "e7_stale": 500, "e8_fresh": 800, "e8_stale": 1200, "e9": 1000,
        "wood": 150, "stone": 150, "coal": None, "iron": None, "zombie": 1000, "hit": 800}
POPS = ("asleep", "mining", "zombie", "hit")


@contextlib.contextmanager
def captured():
    """model.sample's generated periphery and lighting, in call order: interface.futures calls it once per batch, on
    the rows of its outputs."""
    got, sample = [], M.sample

    def capture(*a, **k):
        x = sample(*a, **k)
        got.append((x[0][:, 0].float().cpu(), x[2][:, 0].float().cpu()))
        return x
    M.sample = capture
    try:
        yield got
    finally:
        M.sample = sample


def transitions(L):
    """Per transition (the generated frame t = index + 1 within its episode): the situation at state t - 1, the truth
    at t."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    ep = np.repeat(np.arange(len(L["ep_T"])), L["ep_T"])
    t = np.arange(len(ep)) - a_off[ep] + 1
    g = s_off[ep] + t - 1
    d = IF.DIRS[L["s_dir"][g]]
    pos, face = L["s_pos"][g].astype(int), L["s_pos"][g].astype(int) + d
    on = {k: L[f"s_{k}_m"][g] & (L[f"s_{k}_p"][g].astype(int) == face[:, None]).all(-1) for k in G.KINDS}
    after = {k: (L[f"s_{k}_m"][g + 1] & (L[f"s_{k}_p"][g + 1].astype(int) == face[:, None]).all(-1)).any(1) for k in G.KINDS}
    zp, zm, zc = L["s_zombies_p"][g].astype(int), L["s_zombies_m"][g], L["s_zombies_cd"][g]
    adj = zm & (np.abs(zp - pos[:, None]).sum(-1) == 1)
    asleep, run, starts = L["s_sleep"], np.zeros(len(L["s_sleep"]), int), set(s_off.tolist())
    for s in range(len(run)):                                            # consecutive asleep states up to and with s
        run[s] = (run[s - 1] + 1 if s not in starts else 1) if asleep[s] else 0
    inv = L["s_inv"][g]
    return {"ep": ep, "t": t, "g": g, "action": L["t_action"].astype(int), "asleep": asleep[g], "body": L["s_body"][g],
            "inv": inv, "fresh": L["s_body"][g, 3] != L["s_body"][np.maximum(g - 1, 0), 3], "asleep_for": run[g],
            "ach": L["s_ach"][g], "cell": (3 + d[:, 0]) * 9 + 4 + d[:, 1], "view": L["s_view"][g, 3 + d[:, 0], 4 + d[:, 1]],
            "on": on, "mob_on": np.any([v.any(1) for v in on.values()], 0),
            "hp": {k: np.where(on[k], L[f"s_{k}_hp"][g], 99).min(1) for k in G.KINDS},
            "damage": np.max([np.ones(len(g), int)] + [m * (inv[:, c] > 0) for c, m in SWORDS], 0),
            "adjacent": adj.any(1), "ready": (adj & (zc <= 0)).any(1), "min_cd": np.where(adj, zc, 99).min(1),
            "true_awake": ~asleep[g + 1], "true_body": L["s_body"][g + 1], "true_inv": L["s_inv"][g + 1],
            "true_term": L["t_terminated"], "true_view": L["s_view"][g + 1, 3 + d[:, 0], 4 + d[:, 1]], "true_on": after}


def strata(T):
    """{population: per-transition stratum label ('' = not in it)}: functions of the situation only."""
    n, ok = len(T["g"]), T["t"] >= P
    lab = {k: np.full(n, "", object) for k in POPS}
    awake, act, en = ~T["asleep"], T["action"], T["body"][:, 3]
    for i in np.nonzero(ok & T["asleep"])[0]:
        e = en[i]
        lab["asleep"][i] = "e<=5" if e <= 5 else "e6" if e == 6 else "e9" if e >= 9 else f"e{e}_{'fresh' if T['fresh'][i] else 'stale'}"
    for kind, (block, _, cell, tool) in MINING.items():
        c = T["inv"][:, HUD[cell][1]]
        m = ok & awake & (act == DO) & ~T["mob_on"] & (T["view"] == block) & (c <= 8)
        m &= T["inv"][:, tool] >= 1 if tool is not None else True
        lab["mining"][m] = [f"{kind}_{x}" for x in c[m]]
    m = ok & T["adjacent"]
    lethal = T["body"][:, 0] <= np.where(T["asleep"], 7, 2)
    lab["zombie"][m] = [f"{'asleep' if s else 'awake'}_{'ready' if r else 'cooling'}_{'lethal' if l else 'safe'}"
                        for s, r, l in zip(T["asleep"][m], T["ready"][m], lethal[m])]
    for name, (k, _) in MOB.items():
        m = ok & awake & (act == DO) & T["on"][k].any(1)
        lab["hit"][m] = [f"{name}_{'kill' if h <= x else 'hit'}" for h, x in zip(T["hp"][k][m], T["damage"][m])]
    return lab


def select(lab, rng, scale=1.0):
    """Per population: the transitions kept (sorted), every stratum capped by CAPS (its prefix's cap, else the
    population's)."""
    out = {}
    for pop in POPS:
        keep = []
        for s in sorted(set(lab[pop]) - {""}):
            idx = np.nonzero(lab[pop] == s)[0]
            cap = CAPS.get(s, CAPS.get(s.split("_")[0], CAPS.get(pop)))
            cap = None if cap is None else max(1, int(cap * scale))
            keep.append(idx if cap is None or len(idx) <= cap else rng.choice(idx, cap, replace=False))
        out[pop] = np.sort(np.concatenate(keep))
    return out


def fit_readers(model, cfg, Ld, enc_dev, n_train):
    """Asleep and HUD-value readers on F3's tokens of the train400 frames (s1.fit, the S0 probe protocol, seed 0), then
    their accuracy on every dev real frame (PLAN 'reader validity')."""
    Lt = S.first_episodes(dict(np.load(LABELS / "s0_train400_labels.npz")), n_train)
    enc = S.encode(model, Lt, cfg)
    feats = lambda E: (lambda p, v: {"asleep": torch.cat([p[:, PLAYER], v, p[:, :63].mean(1)], 1),
                                     **{c: torch.cat([p[:, c], v], 1) for c in HUD}})(
        torch.cat([e["p"] for e in E]).float(), torch.cat([e["v"] for e in E]).float())
    target = lambda L: {"asleep": torch.as_tensor(L["s_sleep"]).long(),
                        **{c: torch.as_tensor(L[a][:, k].clip(0, 9)).long() for c, (a, k) in HUD.items()}}
    X, Y, Xd, Yd = feats(enc), target(Lt), feats(enc_dev), target(Ld)
    held = torch.as_tensor(np.repeat(np.arange(len(Lt["ep_T"])), Lt["ep_T"] + 1) % 10 == 9)
    R, valid = {}, {}
    for name in X:
        xt, yt, xv, yv = (t.to(DEV) for t in (X[name][~held], Y[name][~held], X[name][held], Y[name][held]))
        R[name], log = S.fit(lambda idx, val: (xv if val else xt)[idx],
                             lambda o, idx, val: F.cross_entropy(o["y"], (yv if val else yt)[idx]),
                             len(yt), len(yv), {"y": 2 if name == "asleep" else 10}, 0)
        with torch.no_grad():
            pred = torch.cat([R[name](x.to(DEV))["y"].argmax(1).cpu() for x in Xd[name].split(65536)])
        ok = pred == Yd[name]
        per = {int(k): [int((Yd[name] == k).sum()), float(ok[Yd[name] == k].float().mean())] for k in Yd[name].unique()}
        valid[str(name)] = {"accuracy": float(ok.float().mean()), "per_value": per, "epochs": len(log),
                            "valid": bool(ok.float().mean() >= 0.99 and all(a >= 0.95 for n, a in per.values() if n >= 100))}
        print("reader", name, json.dumps(valid[str(name)]), flush=True)
    return R, valid


@torch.no_grad()
def read(R, s1, p, v, cell):
    """Frozen readers on percepts p [B, 81, 16], v [B, 8]: P(asleep), per HUD cell the value and its probability, and
    the S1 reader's entity kind and terrain at `cell` [B]."""
    out = {k: [] for k in ("asleep", "kind", "terr", *[f"hud{c}" for c in HUD], *[f"hud{c}_p" for c in HUD])}
    for i in range(0, len(p), 32768):
        pp, vv, cc = p[i:i + 32768].to(DEV), v[i:i + 32768].to(DEV), torch.as_tensor(cell[i:i + 32768], device=DEV)
        out["asleep"].append(R["asleep"](torch.cat([pp[:, PLAYER], vv, pp[:, :63].mean(1)], 1))["y"].softmax(1)[:, 1].cpu())
        for c in HUD:
            q = R[c](torch.cat([pp[:, c], vv], 1))["y"].softmax(1)
            out[f"hud{c}"].append(q.argmax(1).cpu()), out[f"hud{c}_p"].append(q.max(1).values.cpu())
        o = s1(torch.cat([pp[torch.arange(len(pp), device=DEV), cc], vv], 1))
        out["kind"].append(o["kind"].argmax(1).cpu()), out["terr"].append(o["terr"].argmax(1).cpu())
    return {k: torch.cat(x).numpy() for k, x in out.items()}


def generate(model, cfg, enc, L, T, rows, seed, R, s1):
    """interface.futures on the transitions `rows` under the capture; the readers on the generated and the recorded
    successor. Returns per future: the transition, the heads (gen_ / rec_head_), the readers (gen_ / rec_)."""
    tuples = [(int(T["ep"][j]), int(T["t"][j])) for j in rows]
    with captured() as got:
        o = IF.futures(model, enc, L, tuples, K, cfg, torch.Generator(device=DEV).manual_seed(seed))
    p, v = (torch.cat(x) for x in zip(*got))
    assert len(p) == len(o["tuple"]), "capture misaligned with the futures' rows"
    j = rows[o["tuple"]]
    cell = T["cell"][j]
    rp = torch.stack([enc[T["ep"][x]]["p"][T["t"][x]] for x in j]).float()
    rv = torch.stack([enc[T["ep"][x]]["v"][T["t"][x]] for x in j]).float()
    out = {"j": j, **{k: o[k] for k in o if k.startswith(("gen_", "rec_"))}}
    out.update({f"gen_{k}": x for k, x in read(R, s1, p, v, cell).items()})
    out.update({f"rec_{k}": x for k, x in read(R, s1, rp, rv, cell).items()})
    return out


def r0(model, cfg, enc, L):
    """interface.futures under the capture on dev COLLECT_COAL equals F3's stored 16-step reading."""
    ref = json.loads(DEV_READING.read_text())["checkpoints"]["f3_16steps"]
    assert ref["sha256"] == TR.file_sha256(TR.ROOT / CKPT), "dev reading is another checkpoint"
    cls = IF.classes(L, np.random.default_rng(2031))
    name = "ach_COLLECT_COAL"
    with captured() as got:
        o = IF.futures(model, enc, L, cls[name], 4, cfg, torch.Generator(device=DEV).manual_seed(3000 + list(cls).index(name)))
    s = json.loads(json.dumps(IF.summary(o, np.array([cls[name][j][0] for j in o["tuple"]])), default=str))
    assert sum(len(x[0]) for x in got) == len(o["tuple"]) and s == ref["classes"][name], "R0: dev coal does not reproduce"
    return "dev COLLECT_COAL reproduced exactly with the capture active"


def main(smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    model, cfg = IF.load(CKPT)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = STEPS
    L = S.first_episodes(dict(np.load(LABELS / "s0_dev_labels.npz")), 60 if smoke else 0)
    enc = S.encode(model, L, cfg)
    res = {"checkpoint": CKPT, "sha256": TR.file_sha256(TR.ROOT / CKPT), "steps": STEPS, "K": K}
    if not smoke:
        res["r0"] = r0(model, cfg, enc, L)
        print("R0 passed", flush=True)
    R, res["readers"] = fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    s1 = S.load_reader(S1_READER, TR.ROOT / CKPT, DEV)
    T = transitions(L)
    lab = strata(T)
    rows = select(lab, np.random.default_rng(2106), 0.02 if smoke else 1.0)
    res["populations"] = {pop: {s: int((lab[pop][rows[pop]] == s).sum()) for s in sorted(set(lab[pop][rows[pop]]))} for pop in POPS}
    print(json.dumps(res["populations"]), flush=True)
    raw = {}
    for i, pop in enumerate(POPS):
        o = generate(model, cfg, enc, L, T, rows[pop], 4100 + i, R, s1)
        if smoke and i == 0:                                             # sham: the capture changes nothing
            again = IF.futures(model, enc, L, [(int(T["ep"][j]), int(T["t"][j])) for j in rows[pop]], K, cfg,
                               torch.Generator(device=DEV).manual_seed(4100 + i))
            assert all(np.array_equal(again[k], o[k]) for k in again if k.startswith(("gen_", "rec_"))), "capture sham differs"
        o["stratum"] = lab[pop][o["j"]].astype(str)
        raw.update({f"{pop}/{k}": x for k, x in o.items()})
        print(pop, "done", len(o["j"]), "futures", flush=True)
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "run.json").write_text(json.dumps(res, indent=1, default=str))
    summarise(out)


# ---- reading (PLAN 'Declared readings'); CPU, from raw.npz ----

def pop_events(pop, o, T):
    """{event: (generated, recorded through the readers, truth)} per future, and {head event: (gen P, rec-head P,
    truth, rows where the head's event is defined)}."""
    j = o["j"]
    h0, f0 = T["body"][j, 0], T["body"][j, 1]
    ev = {"awake": (o["gen_asleep"] < 0.5, o["rec_asleep"] < 0.5, T["true_awake"][j]),
          "damage": (o["gen_hud63"] < h0, o["rec_hud63"] < h0, T["true_body"][j, 0] < h0),
          "death": (o["gen_hud63"] == 0, o["rec_hud63"] == 0, T["true_term"][j])}
    heads = {"death": (o["gen_term_p"], o["rec_head_term_p"], T["true_term"][j], np.ones(len(j), bool)),
             "damage": (o["gen_neg_p"], o["rec_head_neg_p"], o["rec_reward"] < -0.05, np.ones(len(j), bool))}
    if pop == "asleep":
        ev["energy9"] = (o["gen_hud66"] == 9, o["rec_hud66"] == 9, T["true_body"][j, 3] == 9)
        heads["wake_unlock"] = (o["gen_unlock_p"], o["rec_head_unlock_p"], o["rec_event"], ~T["ach"][j, WAKE_UP])
    if pop == "mining":
        kind = np.array([s.split("_")[0] for s in o["stratum"]])
        n = np.array([int(s.split("_")[1]) for s in o["stratum"]])
        col = {k: c for k, (_, _, c, _) in MINING.items()}
        left = np.array([MINING[k][1] for k in kind])
        g, r = (np.choose(np.searchsorted(list(HUD), [col[k] for k in kind]), [o[f"{w}_hud{c}"] for c in HUD]) for w in ("gen", "rec"))
        tv = np.array([T["true_inv"][x, HUD[col[k]][1]] for x, k in zip(j, kind)])
        ev.update({"count_next": (g == n + 1, r == n + 1, tv == n + 1), "count_copy": (g == n, r == n, tv == n),
                   "cell_cleared": ((o["gen_kind"] == 0) & (o["gen_terr"] == left), (o["rec_kind"] == 0) & (o["rec_terr"] == left),
                                    T["true_view"][j] == left)})
    if pop == "hit":
        kind = np.array([s.split("_")[0] for s in o["stratum"]])
        code = np.array([MOB[k][1] for k in kind])
        gone_true = ~np.array([T["true_on"][MOB[k][0]][x] for x, k in zip(j, kind)])
        ev.update({"mob_gone": (o["gen_kind"] != code, o["rec_kind"] != code, gone_true),
                   "food_up": (o["gen_hud64"] > f0, o["rec_hud64"] > f0, T["true_body"][j, 1] > f0)})
        first = np.where(kind == "cow", ~T["ach"][j, EAT_COW], ~T["ach"][j, DEFEAT_ZOMBIE])
        heads["kill_unlock"] = (o["gen_unlock_p"], o["rec_head_unlock_p"], o["rec_event"], first)
    return ev, heads


def boot_rates(x, ep, masks, n_boot=2000, seed=2107):
    """Point and bootstrap (episode resampling) rates of x within each mask: [len(masks)], [n_boot, len(masks)]."""
    ids, inv = np.unique(ep, return_inverse=True)
    num = np.stack([np.bincount(inv, x * m, len(ids)) for m in masks])
    den = np.stack([np.bincount(inv, m.astype(float), len(ids)) for m in masks])
    w = np.random.default_rng(seed).integers(0, len(ids), (n_boot, len(ids)))
    cnt = np.stack([np.bincount(r, minlength=len(ids)) for r in w]).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return num.sum(1) / den.sum(1), (cnt @ num.T) / (cnt @ den.T)


def contrast(world, true, ep, m1, m0):
    """T = (P_w(d1) - P_w(d0)) / (P_t(d1) - P_t(d0)), Q = P_w(d1) / P_t(d1), false = P_w(d0) - P_t(d0); 95% intervals
    from the same episode resamples (seed 2107), and the declared verdict."""
    pw, bw = boot_rates(world.astype(float), ep, [m1, m0])
    pt, bt = boot_rates(true.astype(float), ep, [m1, m0])
    stat = lambda w, t: np.stack([(w[..., 0] - w[..., 1]) / (t[..., 0] - t[..., 1]), w[..., 0] / t[..., 0], w[..., 1] - t[..., 1]], -1)
    point, b = stat(pw, pt), stat(bw, bt)
    lo, hi = np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)
    verdict = "TRACKS" if lo[0] >= 2 / 3 else "BLIND" if hi[0] < 1 / 3 else "PARTIAL"
    return {"world": pw.tolist(), "true": pt.tolist(), "n_futures": [int(m1.sum()), int(m0.sum())],
            **{k: {"est": float(point[i]), "ci95": [float(lo[i]), float(hi[i])]} for i, k in enumerate(("T", "Q", "false"))},
            "verdict": verdict}


def agreement(frame, head_p, m):
    """P(frame event | head P > 0.5) and P(head P > 0.5 | frame event) on rows m; WORLD-DRAWN / HEAD-READ (PLAN)."""
    hf, ff = (head_p > 0.5) & m, frame & m
    a = float((frame & hf).sum() / max(hf.sum(), 1))
    return {"P_frame_given_head": a, "n_head": int(hf.sum()), "P_head_given_frame": float((hf & ff).sum() / max(ff.sum(), 1)),
            "n_frame": int(ff.sum()), "verdict": "WORLD-DRAWN" if a >= 2 / 3 else "HEAD-READ" if a <= 1 / 3 else "MIXED"}


def summarise(out):
    raw = dict(np.load(out / "raw.npz", allow_pickle=False))
    res = json.loads((out / "run.json").read_text())
    smoke = out.name == "smoke"
    L = S.first_episodes(dict(np.load(LABELS / "s0_dev_labels.npz")), 60 if smoke else 0)
    T = transitions(L)
    bad = {k for k, v in res["readers"].items() if not v["valid"]}
    curves, prim = {}, {}
    for pop in POPS:
        o = {k.split("/", 1)[1]: v for k, v in raw.items() if k.startswith(pop + "/")}
        ep, st = T["ep"][o["j"]], o["stratum"]
        ev, heads = pop_events(pop, o, T)
        curves[pop] = {}
        for s in sorted(set(st)):
            m = st == s
            row = {"futures": int(m.sum()), "histories": int(len(np.unique(o["j"][m])))}
            for k, (g, r, t) in ev.items():
                row[k] = {w: G.estimate(np.asarray(x, float), m, ep)["est"] for w, x in (("gen", g), ("rec_reader", r), ("true", t))}
            for k, (g, r, t, d) in heads.items():
                if (m & d).any():
                    row[f"head_{k}"] = {w: G.estimate(np.asarray(x, float), m & d, ep)["est"] for w, x in (("gen", g), ("rec_head", r), ("true", t))}
            row["readable_gen"] = {c: float((o[f"gen_hud{c}_p"][m] >= 0.9).mean()) for c in HUD}
            curves[pop][s] = row
        o["ev"], o["heads"], o["ep"], o["st"] = ev, heads, ep, st
        prim[pop] = o
    rd = {}
    A, Z, H = prim["asleep"], prim["zombie"], prim["hit"]
    in_ = lambda o, *names: np.isin(o["st"], names)
    rd["wake_value"] = contrast(A["ev"]["awake"][0], A["ev"]["awake"][2], A["ep"], in_(A, "e9"), in_(A, "e8_fresh"))
    pv, bv = boot_rates(A["ev"]["awake"][0].astype(float), A["ep"], [in_(A, "e9"), in_(A, "e8_fresh"), in_(A, "e8_stale")])
    V, C = pv[0] - pv[1], pv[1] - pv[2]
    Vb, Cb = bv[:, 0] - bv[:, 1], bv[:, 1] - bv[:, 2]
    ci = lambda b: [float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))]
    rd["wake_value_vs_change"] = {"V": {"est": float(V), "ci95": ci(Vb)}, "C": {"est": float(C), "ci95": ci(Cb)},
                                  "verdict": "VALUE-READ" if ci(Vb)[0] > 0 and C < V / 3 else
                                  "CHANGE-TRIGGERED" if ci(Cb)[0] > 0 and C >= V else "MIXED"}
    dur = T["asleep_for"][A["j"]]
    m8 = in_(A, "e8_stale")
    rd["wake_time_asleep_e8_stale"] = {f"{a}-{b}": G.estimate(A["ev"]["awake"][0].astype(float), m8 & (dur >= a) & (dur <= b), A["ep"])
                                       for a, b in ((1, 10), (11, 20), (21, 40), (41, 10 ** 6)) if (m8 & (dur >= a) & (dur <= b)).any()}
    rd["cooldown"] = contrast(Z["ev"]["damage"][0], Z["ev"]["damage"][2], Z["ep"], in_(Z, "awake_ready_safe"), in_(Z, "awake_cooling_safe"))
    rd["lethality"] = contrast(Z["ev"]["death"][0], Z["ev"]["death"][2], Z["ep"], in_(Z, "awake_ready_lethal"), in_(Z, "awake_ready_safe"))
    rd["sleep_attack"] = contrast(Z["ev"]["damage"][0], Z["ev"]["damage"][2], Z["ep"], in_(Z, "asleep_ready_lethal", "asleep_ready_safe"),
                                  in_(Z, "asleep_cooling_lethal", "asleep_cooling_safe"))
    for k in MOB:
        rd[f"{k}_kill"] = contrast(H["ev"]["mob_gone"][0], H["ev"]["mob_gone"][2], H["ep"], in_(H, f"{k}_kill"), in_(H, f"{k}_hit"))
    for name, src in (("cooldown", "damage"), ("lethality", "death"), ("sleep_attack", "damage")):  # heads beside
        rd[name]["heads"] = contrast(Z["heads"][src][0], Z["heads"][src][2].astype(float), Z["ep"], *(
            (in_(Z, "awake_ready_safe"), in_(Z, "awake_cooling_safe")) if name == "cooldown" else
            (in_(Z, "awake_ready_lethal"), in_(Z, "awake_ready_safe")) if name == "lethality" else
            (in_(Z, "asleep_ready_lethal", "asleep_ready_safe"), in_(Z, "asleep_cooling_lethal", "asleep_cooling_safe"))))
    wk = A["heads"]["wake_unlock"]
    rd["world_vs_head"] = {"wake_false (asleep, energy <= 8, WAKE_UP not unlocked)": agreement(A["ev"]["awake"][0], wk[0], wk[3] & ~in_(A, "e9")),
                           "wake_true (energy 9, WAKE_UP not unlocked)": agreement(A["ev"]["awake"][0], wk[0], wk[3] & in_(A, "e9")),
                           "death (zombie rows)": agreement(Z["ev"]["death"][0], Z["heads"]["death"][0], np.ones(len(Z["st"]), bool)),
                           "damage (zombie rows)": agreement(Z["ev"]["damage"][0], Z["heads"]["damage"][0], np.ones(len(Z["st"]), bool))}
    Mn = prim["mining"]
    kind = np.array([s.split("_")[0] for s in Mn["st"]]); n = np.array([int(s.split("_")[1]) for s in Mn["st"]])
    rd["mining"] = {k: {"appearance_n0": G.estimate(Mn["ev"]["count_next"][0].astype(float), (kind == k) & (n == 0), Mn["ep"]),
                        "increments_n>=1": G.estimate(Mn["ev"]["count_next"][0].astype(float), (kind == k) & (n >= 1), Mn["ep"]),
                        "n1-3": G.estimate(Mn["ev"]["count_next"][0].astype(float), (kind == k) & (n >= 1) & (n <= 3), Mn["ep"])}
                    for k in MINING}
    for k, r in {"wake_value": "asleep", "cooldown": "63", "lethality": "63", "sleep_attack": "63"}.items():
        if r in bad:                                                     # the kills read F3's own S1 reader
            rd[k]["verdict"] += " (reader-limited)"
    (out / "curves.json").write_text(json.dumps({**res, "readings": rd, "curves": curves}, indent=1, default=str))
    for k, v in rd.items():
        print(k, json.dumps({q: v[q] for q in ("T", "Q", "false", "verdict", "V", "C") if isinstance(v, dict) and q in v}), flush=True)
    print("saved", out / "curves.json", flush=True)


if __name__ == "__main__":
    if "--summarise" in sys.argv:
        summarise(HERE / ("smoke" if "--smoke" in sys.argv else "readings"))
    else:
        main("--smoke" in sys.argv)
