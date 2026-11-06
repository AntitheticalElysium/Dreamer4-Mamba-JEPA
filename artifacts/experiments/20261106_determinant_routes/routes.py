"""Determinant routes (PLAN.md): where each blind determinant stops in frozen P1-F3. Three stages:
  build    (CPU, JAX): simulator-edited pairs on exactly replayed dev backgrounds -> bank.pt (renders of frame t - 1 and
           of both successors from the recorded step key; natural successor asserted equal to the stored frame t).
  run      (GPU): per pair, the belief over the full prefix (batch 2), the cache of frames t - 8 .. t - 1 (shared context
           noise), the 2 x 2 (cache x memory) factorial at 16 steps (K = 8, common random numbers), the first-call
           functional, and the availability features s1-s7; part B's belief features for the curves' natural rows.
           -> readings/run.pt, readings/run.json
  analyse  (CPU): PLAN's declared verdicts -> readings/routes.json
Usage (project venv):
  JAX_PLATFORMS=cpu python -m artifacts.experiments.20261106_determinant_routes.routes build [--smoke]
  MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_determinant_routes.routes run [--smoke]
  python -m artifacts.experiments.20261106_determinant_routes.routes analyse [--smoke]
"""
import copy, importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
TRC = importlib.import_module("artifacts.experiments.20261105_consequence_trace.trace")
HERE = Path(__file__).resolve().parent
KINDS = {"sleep": 240, "energy": 200, "coal": 200, "wood": 200}
SLOT = {"coal": 70, "wood": 68}
NOOP, K, CTX_TRAIN = 0, 8, 0.67
COND = ((0, 0), (1, 0), (0, 1), (1, 1))                                  # (denoiser-side member, memory member); the
                                                                         # denoiser side is the cache t - 1 (sleep,
                                                                         # energy) or the action input (counts)


# ---- build (CPU) ----

def candidates(L):
    """{kind: [transition index j]} in label order: the situation at state t - 1 (curves.transitions' g), PLAN table A."""
    T = CV.transitions(L)
    lab = CV.strata(T)
    g, ok, sl = T["g"], T["t"] >= CV.P, T["asleep"]
    pos = L["s_pos"][g].astype(int)
    near = np.zeros(len(g), bool)
    for k in ("zombies", "skeletons"):
        near |= (L[f"s_{k}_m"][g] & (np.abs(L[f"s_{k}_p"][g].astype(int) - pos[:, None]).sum(-1) <= 3)).any(1)
    near |= L["s_arrows_m"][g].any(1)
    en = T["body"][:, 3]
    quiet = ok & sl & ~near & (en <= 8)
    out = {"sleep": np.nonzero(quiet)[0], "energy": np.nonzero(quiet & (en == 8) & ~T["fresh"])[0],
           **{k: np.nonzero(np.isin(lab["mining"], [f"{k}_{n}" for n in (1, 2, 3)]))[0] for k in SLOT}}
    return T, out


def choose(T, L, idx, n, per_episode, strata=None):
    """First eligible transitions per episode in label order, at most `per_episode` each, until n (per stratum)."""
    keep, seen = [], {}
    groups = {None: idx} if strata is None else {s: idx[strata[idx] == s] for s in np.unique(strata[idx])}
    for s, ix in groups.items():
        got = 0
        for j in ix:
            e = int(T["ep"][j])
            if seen.get((s, e), 0) < per_episode and got < n:
                keep.append(j); seen[(s, e)] = seen.get((s, e), 0) + 1; got += 1
    return np.sort(np.array(keep))


def build(smoke):
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T, cand = candidates(L)
    light = np.where(L["s_light"][T["g"]] >= 0.5, "day", "night")
    _, _, _, step, render = D.RP.env_and_render()
    rows, excluded = [], {}
    for kind, n in KINDS.items():
        n = 3 if smoke else n
        j = choose(T, L, cand[kind], n // 2 if kind == "sleep" else n, 2 if kind in SLOT else 1,
                   light if kind == "sleep" else None)
        for jj in j:
            e, t = int(T["ep"][jj]), int(T["t"][jj])                     # generated frame t; edited state t - 1
            shard, slot = int(L["ep_shard"][e]), int(L["ep_slot"][e])
            st = D.RP.advance_to(shard, slot, t - 1)
            obs = D.RP.episode_fields(shard, slot)["observations"]
            assert np.array_equal(np.asarray(render(st)), obs[t - 1].numpy()), ("replay mismatch", kind, e, t)
            edit = {"sleep": lambda s: s.replace(is_sleeping=jnp.asarray(False)),   # counts: the action is the edit (PLAN)
                    "energy": lambda s: s.replace(player_energy=jnp.asarray(9, s.player_energy.dtype))}.get(kind, lambda s: s)(st)
            actions = (NOOP, NOOP) if kind in ("sleep", "energy") else (int(L["t_action"][jj]), NOOP)
            key = D.RP._slot_keys(shard, slot)[1][t - 1]
            nxt = [step(key, s, jnp.asarray(a, jnp.int32))[1] for s, a in zip((st, edit), actions)]
            succ = [np.asarray(render(s)) for s in nxt]
            assert np.array_equal(succ[0], obs[t].numpy()), ("natural successor differs from the stored frame", kind, e, t)
            truth = {"asleep": [bool(s.is_sleeping) for s in nxt], "energy": [int(s.player_energy) for s in nxt],
                     "count": [int(getattr(s.inventory, kind)) for s in nxt] if kind in SLOT else None}
            ok = {"sleep": truth["asleep"] == [True, False], "energy": truth["asleep"] == [True, False],
                  "coal": truth["count"] and truth["count"][0] == truth["count"][1] + 1,
                  "wood": truth["count"] and truth["count"][0] == truth["count"][1] + 1}[kind]
            if not ok:
                excluded.setdefault(kind, []).append({"e": e, "t": t, "truth": truth})
                continue
            rows.append({"kind": kind, "e": e, "t": t, "j": int(jj), "light": str(light[jj]), "actions": actions,
                         "frame": np.stack([np.asarray(render(st)), np.asarray(render(edit))]), "successor": np.stack(succ),
                         "truth": truth})
            D.RP._shard.cache_clear()
        print(kind, sum(r["kind"] == kind for r in rows), "pairs, excluded", len(excluded.get(kind, [])), flush=True)
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out / "bank.pt")
    (out / "bank.json").write_text(json.dumps({"pairs": {k: sum(r["kind"] == k for r in rows) for k in KINDS},
                                               "excluded": excluded, "bank_sha256": TR.file_sha256(out / "bank.pt")}, indent=1))


# ---- run (GPU) ----

def stack(caches, rep):
    """Per-condition caches (each batch 1) -> one batched cache, each repeated `rep` times (rows condition-major)."""
    return {"t": caches[0]["t"], "kv": [tuple(torch.cat([c["kv"][i][z].expand(rep, *c["kv"][i][z].shape[1:]) for c in caches])
                                              for z in (0, 1)) for i in range(len(caches[0]["kv"]))]}


def last_kv(cache, tokens):
    """Frame t - 1's temporal keys / values at `tokens` (mean over them), per temporal layer -> [B, layers * 2 * W]."""
    return torch.cat([t[:, tokens, :, -1].mean(1).flatten(1) for k, v in cache["kv"] for t in (k, v)], -1)


@torch.no_grad()
def pair_state(model, cfg, enc, r):
    """The pair's two members: frame t - 1 percepts, beliefs (full prefix, batch 2), caches at the inference and the
    training-median context level, consequence / determinant token sets, the successors' tokens."""
    den, bel, e, t = model.denoiser, model.belief, r["e"], r["t"]
    E = enc[e]
    dev = CV.DEV
    fx1, fx = E["fix"][t - 1].to(dev), E["fix"][t].to(dev)
    with TR.autocast(dev, cfg):
        P2 = model.perception(torch.as_tensor(r["frame"], device=dev), fx1.expand(2, -1))
        Ps = model.perception(torch.as_tensor(r["successor"], device=dev), fx.expand(2, -1))
        hist = M.Percept(*(E[k][:t - 1].to(dev).float()[None].expand(2, -1, -1, *([-1] if k != "v" else []))
                           for k in ("p", "g", "v")), E["fix"][:t - 1].to(dev)[None].expand(2, -1, -1))
        emb = torch.cat([bel.percept_embed(hist), bel.percept_embed(P2)[:, None]], 1)
        act = E["action"][:t].to(dev)[None].expand(2, -1).clone()
        act[:, t - 1] = torch.as_tensor(r["actions"], device=dev)
        y, _ = bel(bel.tokens(emb, E["reward"][:t].to(dev)[None].expand(2, -1), E["terminated"][:t].to(dev)[None].expand(2, -1), act),
                   bel.initial(2, dev))
        b, bp = y[:, 2 * (t - 1)].float(), y[:, 2 * (t - 1) + 1].float()
    caches = {}
    for level in (cfg["edm"]["inference_context_sigma"], CTX_TRAIN):
        c2 = copy.deepcopy(cfg); c2["edm"]["inference_context_sigma"] = level
        out = []
        for m in (0, 1):
            gen = torch.Generator(device=dev).manual_seed(r["j"])            # one context draw per pair, both members
            cache = den.new_cache()
            with TR.autocast(dev, cfg):
                for f in range(t - CV.P, t):
                    x = tuple(E[k][f].to(dev).float()[None, None] for k in ("p", "g", "v")) if f < t - 1 else \
                        tuple(z[m:m + 1, None].float() for z in P2.parts())
                    cache = M.context_pass(den, x, E["prev_action"][f].to(dev)[None, None],
                                           bel.memory(E["bp"][f].to(dev).float()[None])[:, None], E["fix"][f].to(dev)[None, None],
                                           cache, c2, gen)
            out.append(cache)
        caches[level] = out
    return P2, Ps, b, bp, caches, fx


def tokens_of(kind, r):
    """(determinant tokens of frame t - 1, consequence tokens of frame t) as periphery indices (counts: the faced
    cell, identical in both members since the action is the edit)."""
    if kind == "sleep":
        return [CV.PLAYER] + list(range(63)), list(range(63))
    if kind == "energy":
        return [66], list(range(63))
    return [r["face_cell"]], [SLOT[kind]]


@torch.no_grad()
def run(smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out = HERE / ("smoke" if smoke else "readings")
    rows = torch.load(out / "bank.pt", weights_only=False)
    model, cfg = CV.IF.load(CV.CKPT)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = CV.STEPS
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    with torch.enable_grad():                                           # the readers are fitted here (s1.fit)
        R, valid = CV.fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    s1 = S.load_reader(CV.S1_READER, TR.ROOT / CV.CKPT, CV.DEV)
    den, ob, dev = model.denoiser, model.outcome, CV.DEV
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    hidden = {}
    hooks = [blk.register_forward_hook(lambda mod, a, o, i=i: hidden.__setitem__(i, o[0])) for i, blk in enumerate(den.blocks)]
    T = CV.transitions(L)
    res = []
    for r in rows:
        g = int(T["g"][r["j"]])
        d = CV.IF.DIRS[L["s_dir"][g]]
        r["face_cell"] = int((3 + d[0]) * 9 + 4 + d[1])
        det, cons = tokens_of(r["kind"], r)
        P2, Ps, b, bp, caches, fx = pair_state(model, cfg, enc, r)
        r1 = {"tokens": float((P2.p[0].float() - enc[r["e"]]["p"][r["t"] - 1].to(dev).float()).abs().max()),
              "belief": float((bp[0] - enc[r["e"]]["bp"][r["t"]].to(dev).float()).norm() / bp[0].norm())}
        assert r1["tokens"] < 2e-2, ("R1: natural member's tokens differ from S1 encode", r1)
        lo = cfg["edm"]["inference_context_sigma"]
        gen_out, first = [], []
        for ck, mm in COND:                                              # the same generator per condition: CRN
            a = torch.full((K, 1), r["actions"][ck], device=dev, dtype=torch.long)   # the denoiser's action input
            mem = model.belief.memory(bp[mm:mm + 1])[:, None]
            with TR.autocast(dev, cfg):
                xs = M.sample(den, shapes, a, mem.expand(K, -1, -1, -1), fx.expand(K, -1)[:, None],
                              stack([caches[lo][ck]], K), cfg, torch.Generator(device=dev).manual_seed(r["j"]))
                hidden.clear()
                x1 = TRC.first_call(den, shapes, a[:1], mem, fx[None, None], caches[lo][ck], cfg,
                                    torch.Generator(device=dev).manual_seed(r["j"] + 1))
            parts = tuple(z[:, 0] for z in xs)
            rd = CV.read(R, s1, parts[0].float().cpu(), parts[2].float().cpu(), np.full(K, cons[0]))
            with TR.autocast(dev, cfg):
                h = ob.features(bp[mm:mm + 1].expand(K, -1), model.belief.percept_embed(M.Percept(*parts, fx.expand(K, -1)))).float()
                unlock = (ob.bin_logp(h).exp() * (ob.bin_value > 0.5)).sum(-1)
            gen_out.append({"asleep": rd["asleep"], "slot": rd[f"hud{cons[0]}"] if r["kind"] in SLOT else None,
                            "energy": rd["hud66"], "unlock": unlock.cpu().numpy()})
            first.append({"x": (x1[0][0, 0].float().cpu(), x1[2][0, 0].float().cpu()),
                          "hidden": torch.stack([hidden[i][0, 0, cons].float().mean(0).cpu() for i in range(len(den.blocks))])})
        A, B = Ps.p[0].float().cpu(), Ps.p[1].float().cpu()
        Av, Bv = Ps.v[0].float().cpu(), Ps.v[1].float().cpu()
        vec = lambda p, v: torch.cat([p[cons].flatten(), v if r["kind"] not in SLOT else v[:0]])
        a_, b_ = vec(A, Av), vec(B, Bv)
        f = [float(((vec(*x["x"]) - a_) @ (b_ - a_)) / (b_ - a_).square().sum()) for x in first]
        rec = CV.read(R, s1, Ps.p.float().cpu(), Ps.v.float().cpu(), np.full(2, cons[0]))
        with TR.autocast(dev, cfg):
            emb = model.belief.percept_embed(P2).float()
        feats = {"s1": torch.cat([P2.p[:, det].float().mean(1) if r["kind"] != "sleep" else
                                  torch.cat([P2.p[:, CV.PLAYER].float(), P2.p[:, :63].float().mean(1)], -1), P2.v.float()], -1).cpu(),
                 "s2": emb.cpu(), "s3": b.cpu(), "s4": bp.cpu(), "s5": model.belief.memory(bp).flatten(1).float().cpu(),
                 **{f"s6_{w}_{lv}": torch.cat([last_kv(caches[lv][m], tk) for m in (0, 1)]).float().cpu()
                    for lv in (lo, CTX_TRAIN) for w, tk in (("det", det), ("cons", cons))},
                 **{f"s7_{i}": torch.stack([first[0]["hidden"][i], first[3]["hidden"][i]]) for i in range(len(den.blocks))}}
        res.append({"kind": r["kind"], "e": r["e"], "t": r["t"], "j": r["j"], "light": r["light"], "truth": r["truth"], "r1": r1,
                    "rec": {"asleep": rec["asleep"], "slot": rec[f"hud{cons[0]}"] if r["kind"] in SLOT else None},
                    "gen": gen_out, "f": f, "feats": feats})
        if len(res) % 50 == 0 or smoke:
            print("pairs", len(res), r["kind"], "f", [round(x, 3) for x in f], flush=True)
    for hk in hooks:
        hk.remove()
    raw = dict(np.load(CV.HERE / "readings/raw.npz"))                   # part B: the curves' natural rows' beliefs
    partb = {}
    for pop in ("zombie", "hit"):
        j = raw[f"{pop}/j"][::CV.K]                                     # one row per history (K futures each)
        partb[pop] = {"j": j, "bp": torch.stack([enc[int(T["ep"][x])]["bp"][int(T["t"][x])] for x in j]).float()}
    torch.save({"pairs": res, "partb": partb}, out / "run.pt")
    (out / "run.json").write_text(json.dumps({"checkpoint": CV.CKPT, "sha256": TR.file_sha256(TR.ROOT / CV.CKPT), "readers": valid,
                                              "pairs": {k: sum(x["kind"] == k for x in res) for k in KINDS}}, indent=1))
    print("saved", out, flush=True)


# ---- analyse (CPU): PLAN's declared readings ----

def logistic(Xtr, ytr, Xte, C=1.0):
    """L2 logistic regression (C x summed log-loss + |w|^2 / 2, sklearn's objective), standardised on the fitting rows."""
    m, sd = Xtr.mean(0), Xtr.std(0).clamp(min=1e-6)
    Xtr, Xte = (Xtr - m) / sd, (Xte - m) / sd
    w = torch.zeros(Xtr.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=500, line_search_fn="strong_wolfe")
    y = torch.as_tensor(ytr, dtype=torch.float64)

    def closure():
        opt.zero_grad()
        lo = C * torch.nn.functional.binary_cross_entropy_with_logits(Xtr @ w + b, y, reduction="sum") + 0.5 * w.square().sum()
        lo.backward()
        return lo
    opt.step(closure)
    return torch.sigmoid(Xte @ w + b).detach().numpy()


def readability(X, ep, seed=1106):
    """X [pairs, 2, D]: held-out accuracy of member identity (5 folds by episode, both members in a fold), the
    random-label control (labels flipped per pair), and the paired ratio |x1 - x0| / background spread."""
    X = X.double()
    folds = np.unique(ep, return_inverse=True)[1] % 5
    flip = np.random.default_rng(seed).integers(0, 2, len(X)).astype(bool)
    acc = {}
    for name, lab in (("semantic", np.zeros(len(X), bool)), ("control", flip)):
        y = np.stack([lab, ~lab], 1).astype(float)                        # member 0 -> label lab, member 1 -> not lab
        hit = []
        for f in range(5):
            tr, te = folds != f, folds == f
            p = logistic(X[tr].flatten(0, 1), y[tr].ravel(), X[te].flatten(0, 1))
            hit.append((p > 0.5) == y[te].ravel().astype(bool))
        acc[name] = float(np.concatenate(hit).mean())
    diff = (X[:, 1] - X[:, 0]).norm(dim=-1).median()
    spread = (X[:, 0] - X[:, 0].mean(0)).norm(dim=-1).median()
    v = acc["semantic"]
    return {"accuracy": v, "control": acc["control"], "paired_ratio": float(diff / spread),
            "class": "READABLE" if v >= 0.9 else "WEAK" if v >= 0.6 else "ABSENT"}


def boot(x, n=2000, seed=1106):
    x = np.asarray(x, float)
    b = x[np.random.default_rng(seed).integers(0, len(x), (n, len(x)))].mean(1)
    return {"est": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def route(E, tr):
    """E [pairs, 4 conditions (COND order)], tr [pairs] true contrast: PLAN's route verdict."""
    sign = np.sign(tr.mean())
    N, Cc, Mm = (sign * (E[:, i] - E[:, 0]) for i in (3, 1, 2))
    Tr = abs(tr.mean())
    n, c, m = boot(N), boot(Cc), boot(Mm)
    if n["ci95"][1] < Tr / 3:
        v = "NONE"
    else:
        N0 = n["est"]
        hi, low = lambda z: z["est"] >= 2 * N0 / 3, lambda z: z["est"] <= N0 / 3
        v = "CACHE" if hi(c) and low(m) else "MEMORY" if hi(m) and low(c) else "EITHER" if hi(c) and hi(m) else \
            "JOINT" if low(c) and low(m) else "MIXED"
    return {"N": n, "C": c, "M": m, "true_contrast": float(Tr), "conditions": [float(x) for x in E.mean(0)], "verdict": v}


def history_ceiling(L, T, j, target):
    """Hand-coded from the visible history (labels): READY = no damage taken while an adjacent zombie was adjacent in
    the previous 5 steps (an attack resets the cooldown to 5); KILL = initial health minus the earlier DO hits on the
    faced mob's slot (since it last appeared) times the sword's damage, <= this hit's damage."""
    out = []
    for x in j:
        g, e = int(T["g"][x]), int(T["ep"][x])
        start = int(np.r_[0, np.cumsum(L["ep_T"] + 1)][e])
        if target == "ready":
            zp, zm = L["s_zombies_p"].astype(int), L["s_zombies_m"]
            pos = L["s_pos"].astype(int)
            adj = lambda s, z: zm[s, z] and np.abs(zp[s, z] - pos[s]).sum() == 1
            cur = [z for z in range(zp.shape[1]) if adj(g, z)]
            hit = lambda z: any(s >= start and adj(s, z) and L["s_body"][s + 1, 0] < L["s_body"][s, 0] for s in range(g - 5, g))
            out.append(float(any(not hit(z) for z in cur)))
        else:
            kind = "cows" if T["on"]["cows"][x].any() else "zombies"
            z = int(T["on"][kind][x].argmax())
            s, dmg = g - 1, 0
            while s >= start and L[f"s_{kind}_m"][s, z]:
                d = CV.IF.DIRS[L["s_dir"][s]]
                faced = (L[f"s_{kind}_p"][s, z].astype(int) == L["s_pos"][s].astype(int) + d).all()
                if faced and int(L["t_action"][x - (g - s)]) == CV.DO and not L["s_sleep"][s]:
                    dmg += int(T["damage"][x - (g - s)])
                s -= 1
            out.append(float({"cows": 3, "zombies": 5}[kind] - dmg <= T["damage"][x]))
    return np.array(out)


def analyse(smoke):
    out = HERE / ("smoke" if smoke else "readings")
    run_ = torch.load(out / "run.pt", weights_only=False)
    res, readings = run_["pairs"], {}
    for kind in KINDS:
        P_ = [r for r in res if r["kind"] == kind]
        if not P_:
            continue
        ep = np.array([r["e"] for r in P_])
        if kind in SLOT:
            n = np.array([r["truth"]["count"][1] for r in P_])           # the edited member's count = n
            ev = lambda r, c, i: (r["gen"][c]["slot"] == n[i] + 1).mean()
            tr = np.full(len(P_), -1.0)                                  # d0 increments, d1 does not
        else:
            ev = lambda r, c, i: (r["gen"][c]["asleep"] < 0.5).mean()
            tr = np.full(len(P_), 1.0)                                   # d0 asleep, d1 awake
        E = np.array([[ev(r, c, i) for c in range(4)] for i, r in enumerate(P_)])
        F = np.array([r["f"] for r in P_])
        rd = {"pairs": len(P_), "route": route(E, tr), "functional": {f"{COND[c]}": boot(F[:, c]) for c in range(4)},
              "r1_belief_rel_diff_max": max(r["r1"]["belief"] for r in P_)}
        if kind == "sleep":
            for lt in ("day", "night"):
                m = np.array([r["light"] == lt for r in P_])
                rd[f"route_{lt}"] = route(E[m], tr[m])
        rd["availability"] = {k: readability(torch.stack([r["feats"][k] for r in P_]), ep) for k in P_[0]["feats"]}
        rd["recorded_reader_check"] = float(np.mean([(r["rec"]["asleep"] < 0.5).tolist() == [False, True] if kind not in SLOT
                                                     else (r["rec"]["slot"] == [n[i] + 1, n[i]]).all() for i, r in enumerate(P_)]))
        av = rd["availability"]
        cls = lambda k: av[k]["class"]
        s6c = [k for k in av if k.startswith("s6_cons")]
        if rd["route"]["verdict"] == "NONE":                             # counts: the action enters at s4 (frames equal)
            inputs = cls("s5") == "READABLE" or (kind not in SLOT and any(cls(k) == "READABLE" for k in s6c if "_0.1" in k))
            if not inputs:
                order = ["s4", "s5"] if kind in SLOT else ["s2", "s3", "s4", "s5"] + s6c
                rd["locus"] = "NOT AT INPUT: " + next((k for k in order if cls(k) != "READABLE"), "?")
            else:
                last = cls(f"s7_{len([k for k in av if k.startswith('s7_')]) - 1}")
                rd["locus"] = "OUTPUT" if last == "READABLE" else "UNUSED"
        readings[kind] = rd
        print(kind, rd["route"]["verdict"], {k: (round(v["accuracy"], 3), round(v["paired_ratio"], 4)) for k, v in av.items()}, rd.get("locus"), flush=True)
    if "coal" in readings and "wood" in readings:
        last = f"s7_{len([k for k in readings['coal']['availability'] if k.startswith('s7_')]) - 1}"
        c, w = readings["coal"], readings["wood"]
        slot_c, slot_w = c["availability"][last]["class"], w["availability"][last]["class"]
        readings["counts"] = {"coal_last_block": slot_c, "wood_last_block": slot_w,
                              "verdict": "OPERATION" if slot_c == "READABLE" and slot_w == "READABLE" and c["route"]["N"]["est"] < w["route"]["N"]["est"] / 3
                              else "BINDING" if slot_c != "READABLE" and slot_w == "READABLE" and c["availability"]["s5"]["class"] == "READABLE"
                              else "NEITHER"}
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    raw = dict(np.load(CV.HERE / "readings/raw.npz"))
    for pop, target in (("zombie", "ready"), ("hit", "kill")):
        B = run_["partb"][pop]
        j = B["j"]
        strata = raw[f"{pop}/stratum"][::CV.K]
        keep = np.array([s.startswith("awake") for s in strata]) if pop == "zombie" else np.ones(len(j), bool)
        j, X = j[keep], B["bp"][keep].double()
        y = T["ready"][j] if target == "ready" else np.array([s.endswith("kill") for s in strata[keep]])
        ep = T["ep"][j]
        folds = np.unique(ep, return_inverse=True)[1] % 5
        score = np.zeros(len(j))
        for f in range(5):
            score[folds == f] = logistic(X[folds != f], y[folds != f], X[folds == f])
        ceil = history_ceiling(L, T, j, target)
        ap_b, ap_c = G.average_precision(y, score), G.average_precision(y, ceil)
        ev = {"zombie": "damage", "hit": "mob_gone"}[pop]
        world = raw[f"{pop}/gen_hud63"].reshape(-1, CV.K) < T["body"][raw[f"{pop}/j"].reshape(-1, CV.K)[:, 0], 0][:, None] if ev == "damage" \
            else raw[f"{pop}/gen_kind"].reshape(-1, CV.K) != np.array([CV.MOB[s.split("_")[0]][1] for s in strata])[:, None]
        world = world.mean(1)[keep]
        med = {f"truth_{int(t_)}_belief_{int(b_)}": float(world[(y == t_) & ((score > 0.5) == b_)].mean()) if ((y == t_) & ((score > 0.5) == b_)).any() else None
               for t_ in (0, 1) for b_ in (0, 1)}
        readings[f"hidden_{target}"] = {"rows": int(len(j)), "prevalence": float(y.mean()), "belief_ap": ap_b, "history_ceiling_ap": ap_c,
                                        "verdict": "BELIEF-LIMITED" if ap_b < ap_c - 0.15 else "AVAILABLE" if ap_b >= ap_c - 0.05 else "BETWEEN",
                                        "world_event_by_truth_x_belief": med}
        print(target, json.dumps(readings[f"hidden_{target}"]), flush=True)
    (out / "routes.json").write_text(json.dumps(readings, indent=1, default=str))
    print("saved", out / "routes.json", flush=True)


if __name__ == "__main__":
    {"build": build, "run": run, "analyse": analyse}[sys.argv[1]]("--smoke" in sys.argv)
