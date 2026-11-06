"""Determinant routes (PLAN.md, amended after Codex's review 11-06): does changing a blind determinant through either
input route change P1-F3's generated consequence, and is the determinant recovered from the actual inputs and
location-resolved states by the declared readers? Three stages:
  build    (CPU, JAX): simulator-edited pairs on exactly replayed dev backgrounds -> bank.pt (renders of frame t - 1 and
           of both successors from the recorded step key; the natural successor asserted equal to the stored frame t).
  run      (GPU): per pair, the belief over the full prefix (batch 3: the two members and a same-input sham), the cache
           of frames t - 8 .. t - 1 (one context draw shared by both members), the 2 x 2 factorial at 16 steps (K = 8,
           common random numbers), the first-call functional, location-resolved state tensors, raw generated percepts;
           part B's belief and current-percept features for the curves' natural rows. -> readings/run.pt, run.json
  analyse  (CPU): PLAN's declared readings -> readings/routes.json
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
KINDS = {"sleep": 240, "energy": 200, "coal": 201, "wood": 201}         # counts: 67 per starting count 1, 2, 3
SLOT = {"coal": 70, "wood": 68}
NOOP, K, CTX_STRESS, LIGHT_TOKEN = 0, 8, 0.67, M.N + M.G              # denoiser token order: periphery, glimpse, lighting
COND = ((0, 0), (1, 0), (0, 1), (1, 1))                                  # (denoiser-side member, memory member); the
                                                                         # denoiser side is the cache t - 1 (sleep,
                                                                         # energy) or the action input (counts)
SOURCES = ("artifacts/experiments/20261106_determinant_routes/routes.py", "artifacts/experiments/20261106_determinant_routes/PLAN.md",
           "artifacts/experiments/20261106_determinant_curves/curves.py", "artifacts/experiments/20261105_consequence_trace/trace.py",
           "saccade/model.py", "saccade/eval/s1.py")


# ---- build (CPU) ----

def candidates(L):
    """{kind: [transition index j]} in label order and their strata: the situation at state t - 1 (curves.transitions'
    g), PLAN table A."""
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
    light = np.where(L["s_light"][g] >= 0.5, "day", "night")
    strata = {"sleep": light, "energy": None, **{k: lab["mining"] for k in SLOT}}
    return T, out, strata


def choose(T, idx, n, per_episode, strata=None):
    """First eligible transitions per episode in label order, at most `per_episode` per (stratum, episode), until n
    per stratum."""
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
    T, cand, strata = candidates(L)
    _, _, _, step, render = D.RP.env_and_render()
    rows, excluded = [], {}
    for kind, n in KINDS.items():
        groups = {"sleep": 2, "energy": 1}.get(kind, 3)
        j = choose(T, cand[kind], 3 if smoke else n // groups, 2 if kind in SLOT else 1, strata[kind])
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
            rows.append({"kind": kind, "e": e, "t": t, "j": int(jj), "stratum": None if strata[kind] is None else str(strata[kind][jj]),
                         "recorded_action": int(L["t_action"][jj]), "actions": actions,
                         "frame": np.stack([np.asarray(render(st)), np.asarray(render(edit))]), "successor": np.stack(succ),
                         "truth": truth})
            D.RP._shard.cache_clear()
        print(kind, sum(r["kind"] == kind for r in rows), "pairs, excluded", len(excluded.get(kind, [])), flush=True)
    out = HERE / ("smoke" if smoke else "readings"); out.mkdir(parents=True, exist_ok=True)
    torch.save(rows, out / "bank.pt")
    (out / "bank.json").write_text(json.dumps({"pairs": {k: sum(r["kind"] == k for r in rows) for k in KINDS},
                                               "episodes": {k: len({r["e"] for r in rows if r["kind"] == k}) for k in KINDS},
                                               "strata": {k: {s: sum(r["kind"] == k and r["stratum"] == s for r in rows)
                                                              for s in {r["stratum"] for r in rows if r["kind"] == k}} for k in KINDS},
                                               "excluded": excluded, "bank_sha256": TR.file_sha256(out / "bank.pt")}, indent=1))


# ---- run (GPU) ----

def stack(cache, rep):
    """A batch-1 cache repeated `rep` times."""
    return {"t": cache["t"], "kv": [tuple(z.expand(rep, *z.shape[1:]) for z in kv) for kv in cache["kv"]]}


def frame_kv(cache, tokens):
    """Frame t - 1's temporal keys / values at `tokens`, location-resolved: [B, tokens, layers x (K, V) x W]."""
    return torch.cat([z[:, tokens, :, -1].flatten(2) for kv in cache["kv"] for z in kv], -1)


def sites(kind, r):
    """(determinant tokens of frame t - 1, consequence tokens of frame t) in the denoiser's token order (periphery
    0-80, glimpse 81-89, lighting 90). Counts: the faced cell is identical in both members (the action is the edit)."""
    if kind == "sleep":
        return [CV.PLAYER] + list(range(63)), list(range(63)) + [LIGHT_TOKEN]
    if kind == "energy":
        return [66], list(range(63)) + [LIGHT_TOKEN]
    return [r["face_cell"]], [SLOT[kind]]


@torch.no_grad()
def pair_state(model, cfg, enc, r):
    """The pair's members: frame t - 1 percepts, beliefs (full prefix, batch 3 = members + the same-input sham: the
    natural frame with the RECORDED action, i.e. S1 encode's inputs), caches at the inference context level and the
    stress level (every context frame at 0.67), a sham cache from S1 encode's stored tokens, the successors' tokens."""
    den, bel, e, t, dev = model.denoiser, model.belief, r["e"], r["t"], CV.DEV
    E = enc[e]
    fx1, fx = E["fix"][t - 1].to(dev), E["fix"][t].to(dev)
    with TR.autocast(dev, cfg):
        P2 = model.perception(torch.as_tensor(r["frame"], device=dev), fx1.expand(2, -1))
        Ps = model.perception(torch.as_tensor(r["successor"], device=dev), fx.expand(2, -1))
        rep = lambda z: z.to(dev)[None].expand(3, *z.shape)
        hist = M.Percept(*(rep(E[k][:t - 1]).float() for k in ("p", "g", "v")), rep(E["fix"][:t - 1]))
        e2 = bel.percept_embed(P2)                                       # members; the sham repeats the natural frame
        emb = torch.cat([bel.percept_embed(hist), torch.cat([e2, e2[:1]])[:, None]], 1)
        act = rep(E["action"][:t]).clone()
        act[:, t - 1] = torch.as_tensor([*r["actions"], r["recorded_action"]], device=dev)
        y, _ = bel(bel.tokens(emb, rep(E["reward"][:t]), rep(E["terminated"][:t]), act), bel.initial(3, dev))
        b, bp = y[:, 2 * (t - 1)].float(), y[:, 2 * (t - 1) + 1].float()
    caches = {}
    for level, x_last in ((cfg["edm"]["inference_context_sigma"], P2), (CTX_STRESS, P2), ("sham", None)):
        c2 = copy.deepcopy(cfg)
        if level == CTX_STRESS:
            c2["edm"]["inference_context_sigma"] = level
        out = []
        for m in ((0,) if level == "sham" else (0, 1)):
            gen = torch.Generator(device=dev).manual_seed(r["j"])            # one context draw per pair, both members
            cache = den.new_cache()
            with TR.autocast(dev, cfg):
                for f in range(t - CV.P, t):
                    x = tuple(E[k][f].to(dev).float()[None, None] for k in ("p", "g", "v")) if f < t - 1 or x_last is None else \
                        tuple(z[m:m + 1, None].float() for z in x_last.parts())
                    cache = M.context_pass(den, x, E["prev_action"][f].to(dev)[None, None],
                                           bel.memory(E["bp"][f].to(dev).float()[None])[:, None], E["fix"][f].to(dev)[None, None],
                                           cache, c2, gen)
            out.append(cache)
        caches[level] = out
    return P2, Ps, b, bp, caches, fx


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
    lo = cfg["edm"]["inference_context_sigma"]
    hidden = {}
    hooks = [blk.register_forward_hook(lambda mod, a, o, i=i: hidden.__setitem__(i, o[0])) for i, blk in enumerate(den.blocks)]
    T = CV.transitions(L)
    res, shams = [], []
    for i, r in enumerate(rows):
        g = int(T["g"][r["j"]])
        d = CV.IF.DIRS[L["s_dir"][g]]
        r["face_cell"] = int((3 + d[0]) * 9 + 4 + d[1])
        det, cons = sites(r["kind"], r)
        P2, Ps, b, bp, caches, fx = pair_state(model, cfg, enc, r)
        sham = {"tokens_max_abs": float((P2.p[0].float() - enc[r["e"]]["p"][r["t"] - 1].to(dev).float()).abs().max()),
                "belief_rel": float((bp[2] - enc[r["e"]]["bp"][r["t"]].to(dev).float()).norm() / bp[2].norm()),
                "cache_rel": max(float((a - c).float().norm() / c.float().norm()) for (a, _), (c, _) in
                                 zip(caches[lo][0]["kv"], caches["sham"][0]["kv"]))}
        assert sham["tokens_max_abs"] < 0.04, ("natural member's tokens differ from S1 encode", sham)
        gen_out, first, raw = [], [], []
        for ck, mm in COND:                                              # the same generator per condition: CRN
            a = torch.full((K, 1), r["actions"][ck], device=dev, dtype=torch.long)   # the denoiser's action input
            mem = model.belief.memory(bp[mm:mm + 1])[:, None]
            with TR.autocast(dev, cfg):
                xs = M.sample(den, shapes, a, mem.expand(K, -1, -1, -1), fx.expand(K, -1)[:, None],
                              stack(caches[lo][ck], K), cfg, torch.Generator(device=dev).manual_seed(r["j"]))
                if (ck, mm) == (0, 0) and (smoke or i % 25 == 0):        # generation sham: same inputs, same draws
                    again = M.sample(den, shapes, a, mem.expand(K, -1, -1, -1), fx.expand(K, -1)[:, None],
                                     stack(caches[lo][ck], K), cfg, torch.Generator(device=dev).manual_seed(r["j"]))
                    assert all(torch.equal(u, v) for u, v in zip(xs, again)), "generation sham differs"
                    sham["generation_repeat"] = "bitwise equal"
                hidden.clear()
                x1 = TRC.first_call(den, shapes, a[:1], mem, fx[None, None], caches[lo][ck], cfg,
                                    torch.Generator(device=dev).manual_seed(r["j"] + 1))
            parts = tuple(z[:, 0] for z in xs)
            rd = CV.read(R, s1, parts[0].float().cpu(), parts[2].float().cpu(), np.full(K, r["face_cell"]))
            with TR.autocast(dev, cfg):
                h = ob.features(bp[mm:mm + 1].expand(K, -1), model.belief.percept_embed(M.Percept(*parts, fx.expand(K, -1)))).float()
                unlock = (ob.bin_logp(h).exp() * (ob.bin_value > 0.5)).sum(-1)
            gen_out.append({"asleep": rd["asleep"], "slot": rd[f"hud{cons[0]}"] if r["kind"] in SLOT else None,
                            "energy": rd["hud66"], "unlock": unlock.cpu().numpy()})
            raw.append((parts[0].half().cpu(), parts[2].half().cpu()))
            first.append({"x": (x1[0][0, 0].float().cpu(), x1[2][0, 0].float().cpu()),
                          "hidden": torch.stack([hidden[k][0, 0, cons].float().cpu() for k in range(len(den.blocks))])})
        A, B = Ps.p[0].float().cpu(), Ps.p[1].float().cpu()
        Av, Bv = Ps.v[0].float().cpu(), Ps.v[1].float().cpu()
        pc = [c for c in cons if c < M.N]
        vec = lambda p, v: torch.cat([p[pc].flatten(), v if LIGHT_TOKEN in cons else v[:0]])
        a_, b_ = vec(A, Av), vec(B, Bv)
        f = [float(((vec(*x["x"]) - a_) @ (b_ - a_)) / (b_ - a_).square().sum()) for x in first]
        rec = CV.read(R, s1, Ps.p.float().cpu(), Ps.v.float().cpu(), np.full(2, r["face_cell"]))
        with TR.autocast(dev, cfg):
            emb = model.belief.percept_embed(M.Percept(*(z for z in P2.parts()), P2.fix)).float()
        pd = [c for c in det if c < M.N]
        feats = {"s1": torch.cat([P2.p[:, pd].float().flatten(1), P2.v.float()], -1).cpu(),
                 "s2": emb.cpu(), "s3": b[:2].cpu(), "s4": bp[:2].cpu(), "s5": model.belief.memory(bp[:2]).flatten(1).float().cpu(),
                 **{f"s6_{w}_{lv}": torch.cat([frame_kv(caches[lv][m], tk) for m in (0, 1)]).half().cpu()
                    for lv in (lo, CTX_STRESS) for w, tk in (("det", det), ("cons", cons))},
                 **{f"s7_{k}": torch.stack([first[0]["hidden"][k], first[3]["hidden"][k]]).half() for k in range(len(den.blocks))}}
        res.append({"kind": r["kind"], "e": r["e"], "t": r["t"], "j": r["j"], "stratum": r["stratum"], "truth": r["truth"], "sham": sham,
                    "rec": {"asleep": rec["asleep"], "slot": rec[f"hud{cons[0]}"] if r["kind"] in SLOT else None},
                    "gen": gen_out, "f": f, "feats": feats, "raw_generated": raw, "seed": r["j"]})
        shams.append(sham)
        if len(res) % 50 == 0 or smoke:
            print("pairs", len(res), r["kind"], "f", [round(x, 3) for x in f], "sham", {k: (round(v, 5) if isinstance(v, float) else v)
                                                                                    for k, v in sham.items()}, flush=True)
    for hk in hooks:
        hk.remove()
    raw = dict(np.load(CV.HERE / "readings/raw.npz"))                   # part B: the curves' natural rows
    partb = {}
    for pop in ("zombie", "hit"):
        j = raw[f"{pop}/j"][::CV.K]                                     # one row per history (K futures each)
        ix = [(int(T["ep"][x]), int(T["t"][x])) for x in j]
        P1 = M.Percept(*(torch.stack([enc[e][k][t - 1] for e, t in ix]).float() for k in ("p", "g", "v")),
                       torch.stack([enc[e]["fix"][t - 1] for e, t in ix]))
        with TR.autocast(dev, cfg):
            cur = torch.cat([model.belief.percept_embed(M.Percept(*(z[a:a + 4096].to(dev) for z in P1.parts()), P1.fix[a:a + 4096].to(dev))).float().cpu()
                             for a in range(0, len(ix), 4096)])
        act = torch.nn.functional.one_hot(torch.as_tensor(L["t_action"][j].astype(np.int64)), 17).float()
        partb[pop] = {"j": j, "bp": torch.stack([enc[e]["bp"][t] for e, t in ix]).float(), "current": torch.cat([cur, act], 1)}
    pins = {p: TR.file_sha256(TR.ROOT / p) for p in SOURCES}
    pins.update({CV.CKPT: TR.file_sha256(TR.ROOT / CV.CKPT), "bank.pt": TR.file_sha256(out / "bank.pt")})
    torch.save({"pairs": res, "partb": partb, "readers": {k: v.state_dict() for k, v in R.items()}}, out / "run.pt")
    (out / "run.json").write_text(json.dumps({"checkpoint": CV.CKPT, "pins": pins, "steps": CV.STEPS, "K": K, "readers": valid,
                                              "context_sigma": {"inference": lo, "stress": CTX_STRESS},
                                              "pairs": {k: sum(x["kind"] == k for x in res) for k in KINDS},
                                              "sham_max": {k: max(s[k] for s in shams) for k in ("tokens_max_abs", "belief_rel", "cache_rel")},
                                              "generation_repeats": sum("generation_repeat" in s for s in shams)}, indent=1))
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


def folds_of(ep, k=5):
    return np.unique(ep, return_inverse=True)[1] % k


def readability(X, ep, seed=1106):
    """X [pairs, 2, ...] (flattened per member): held-out accuracy of member identity from ONE state (5 folds by
    episode, both members in a fold), the random-label control (labels flipped per pair), and the paired ratio
    |x1 - x0| / background spread. RECOVERED / PARTIAL / NOT RECOVERED by this probe; CONTROL FAILED if the control
    leaves 0.4-0.6."""
    X = X.flatten(2).double()
    if (X[:, 1] - X[:, 0]).abs().max() == 0:
        return {"class": "IDENTICAL INPUTS"}
    if len(np.unique(ep)) < 5:
        return {"class": "TOO FEW EPISODES"}                             # smoke only
    folds = folds_of(ep)
    flip = np.random.default_rng(seed).integers(0, 2, len(X)).astype(bool)
    acc = {}
    for name, lab in (("probe", np.zeros(len(X), bool)), ("control", flip)):
        y = np.stack([lab, ~lab], 1).astype(float)                        # member 0 -> lab, member 1 -> not lab
        hit = [((logistic(X[folds != f].flatten(0, 1), y[folds != f].ravel(), X[folds == f].flatten(0, 1)) > 0.5)
                == y[folds == f].ravel().astype(bool)) for f in range(5)]
        acc[name] = float(np.concatenate(hit).mean())
    diff = (X[:, 1] - X[:, 0]).norm(dim=-1).median()
    spread = (X[:, 0] - X[:, 0].mean(0)).norm(dim=-1).median()
    v, c = acc["probe"], acc["control"]
    cls = "CONTROL FAILED" if not 0.4 <= c <= 0.6 else "RECOVERED" if v >= 0.9 else "PARTIAL" if v >= 0.6 else "NOT RECOVERED"
    return {"accuracy": v, "control": c, "paired_ratio": float(diff / spread), "class": cls}


def boot(x, ep, n=2000, seed=1106):
    """Mean of per-pair values with an EPISODE bootstrap (pairs from one episode resampled together)."""
    x = np.asarray(x, float)
    ids, inv = np.unique(ep, return_inverse=True)
    s, c = np.bincount(inv, x, len(ids)), np.bincount(inv, None, len(ids))
    w = np.stack([np.bincount(r, minlength=len(ids)) for r in np.random.default_rng(seed).integers(0, len(ids), (n, len(ids)))])
    b = (w @ s) / (w @ c)
    return {"est": float(x.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def effect_class(z, tr):
    """An oriented effect (truth direction positive) against the true contrast tr: CARRIES (lower >= tr / 3), REVERSED
    (upper < 0), NO EFFECT (upper < tr / 3 and lower > -tr / 3), otherwise UNRESOLVED."""
    lo, hi = z["ci95"]
    return "CARRIES" if lo >= tr / 3 else "REVERSED" if hi < 0 else "NO EFFECT" if hi < tr / 3 and lo > -tr / 3 else "UNRESOLVED"


def factorial(E, tr, ep):
    """E [pairs, 4] (COND order: 00, 10, 01, 11; first index the denoiser-side input), tr [pairs] true contrast. Every
    cell, both simple effects of each input at each level of the other, the interaction, the diagonal; oriented to the
    truth. Sufficiency = putting d1 into the d0 run (10 - 00, 01 - 00); necessity = removing d1 from the d1 run (11 - 01,
    11 - 10)."""
    sign, Tr = np.sign(tr.mean()), abs(tr.mean())
    o = lambda x: sign * x
    eff = {"cells": {f"{c[0]}{c[1]}": boot(E[:, i], ep) for i, c in enumerate(COND)},
           "diagonal": boot(o(E[:, 3] - E[:, 0]), ep),
           "denoiser_side_sufficiency (10-00)": boot(o(E[:, 1] - E[:, 0]), ep),
           "denoiser_side_necessity (11-01)": boot(o(E[:, 3] - E[:, 2]), ep),
           "memory_sufficiency (01-00)": boot(o(E[:, 2] - E[:, 0]), ep),
           "memory_necessity (11-10)": boot(o(E[:, 3] - E[:, 1]), ep),
           "interaction (11-10-01+00)": boot(o(E[:, 3] - E[:, 1] - E[:, 2] + E[:, 0]), ep)}
    cls = {k: effect_class(v, Tr) for k, v in eff.items() if k not in ("cells", "interaction (11-10-01+00)")}
    lo, hi = eff["interaction (11-10-01+00)"]["ci95"]
    cls["interaction"] = "PRESENT" if lo > 0 or hi < 0 else "NOT DETECTED"
    return {"true_contrast": float(Tr), **eff, "classes": cls}


def observable_heuristic(L, T, j, target):
    """Reference heuristics from the VISIBLE history only (labels; not ceilings): READY = no damage taken while an
    adjacent zombie was adjacent in the previous 5 steps (an attack resets the cooldown to 5; other damage sources can
    mislead it); KILL = initial health (cow 3, zombie 5) minus the DO hits on the faced mob's slot over its CONTIGUOUS
    visible span ending now (the scan stops at the first step it is off the 9 x 7 view or absent), times the sword's
    damage, <= this hit's damage."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    pos = L["s_pos"].astype(int)
    out = []
    for x in j:
        g, start = int(T["g"][x]), int(s_off[int(T["ep"][x])])
        if target == "ready":
            zp, zm = L["s_zombies_p"].astype(int), L["s_zombies_m"]
            adj = lambda s, z: zm[s, z] and np.abs(zp[s, z] - pos[s]).sum() == 1
            hit = lambda z: any(s >= start and adj(s, z) and L["s_body"][s + 1, 0] < L["s_body"][s, 0] for s in range(g - 5, g))
            out.append(float(any(not hit(z) for z in range(zp.shape[1]) if adj(g, z))))
        else:
            kind = "cows" if T["on"]["cows"][x].any() else "zombies"
            z, s, dmg = int(T["on"][kind][x].argmax()), g - 1, 0
            mp, mm = L[f"s_{kind}_p"].astype(int), L[f"s_{kind}_m"]
            seen = lambda s: mm[s, z] and abs(mp[s, z, 0] - pos[s, 0]) <= 3 and abs(mp[s, z, 1] - pos[s, 1]) <= 4
            while s >= start and seen(s):
                d = CV.IF.DIRS[L["s_dir"][s]]
                if (mp[s, z] == pos[s] + d).all() and int(L["t_action"][x - (g - s)]) == CV.DO and not L["s_sleep"][s]:
                    dmg += int(T["damage"][x - (g - s)])
                s -= 1
            out.append(float({"cows": 3, "zombies": 5}[kind] - dmg <= T["damage"][x]))
    return np.array(out)


def ap_boot(y, s, ep, n=1000, seed=1106):
    """AP with an episode-bootstrap 95% interval."""
    ids, inv = np.unique(ep, return_inverse=True)
    groups = [np.nonzero(inv == k)[0] for k in range(len(ids))]
    rng, b = np.random.default_rng(seed), []
    for _ in range(n):
        m = np.concatenate([groups[k] for k in rng.integers(0, len(ids), len(ids))])
        if y[m].any() and not y[m].all():
            b.append(G.average_precision(y[m], s[m]))
    return {"ap": G.average_precision(y, s), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]}


def analyse(smoke):
    out = HERE / ("smoke" if smoke else "readings")
    run_ = torch.load(out / "run.pt", weights_only=False)
    res = run_["pairs"]
    mx = {k: max(r["sham"][k] for r in res) for k in ("tokens_max_abs", "belief_rel", "cache_rel")}
    readings = {"shams": {**mx, "valid": mx["tokens_max_abs"] < 0.04 and mx["belief_rel"] < 0.02 and mx["cache_rel"] < 0.011,
                          "generation_repeats_bitwise": sum("generation_repeat" in r["sham"] for r in res)}}
    for kind in KINDS:
        P_ = [r for r in res if r["kind"] == kind]
        if not P_:
            continue
        ep = np.array([r["e"] for r in P_])
        if kind in SLOT:
            n = np.array([r["truth"]["count"][1] for r in P_])           # the NOOP member's count = the start n
            ev = lambda r, c, i: (r["gen"][c]["slot"] == n[i] + 1).mean()
            tr = np.full(len(P_), -1.0)                                  # d0 (DO) increments, d1 (NOOP) does not
        else:
            ev = lambda r, c, i: (r["gen"][c]["asleep"] < 0.5).mean()
            tr = np.full(len(P_), 1.0)                                   # d0 asleep, d1 awake
        E = np.array([[ev(r, c, i) for c in range(4)] for i, r in enumerate(P_)])
        F = np.array([r["f"] for r in P_])
        rd = {"pairs": len(P_), "episodes": int(len(np.unique(ep))), "factorial": factorial(E, tr, ep),
              "functional": {f"{COND[c][0]}{COND[c][1]}": boot(F[:, c], ep) for c in range(4)}}
        strata = sorted({r["stratum"] for r in P_ if r["stratum"] is not None})
        for s in strata:                                                 # sleep: light; counts: the starting count
            m = np.array([r["stratum"] == s for r in P_])
            if len(np.unique(ep[m])) >= 5:
                rd[f"factorial_{s}"] = factorial(E[m], tr[m], ep[m])
        rd["recorded_reader_check"] = float(np.mean([(r["rec"]["asleep"] < 0.5).tolist() == [False, True] if kind not in SLOT
                                                     else (r["rec"]["slot"] == [n[i] + 1, n[i]]).all() for i, r in enumerate(P_)]))
        rd["availability"] = {k: readability(torch.stack([r["feats"][k].float() for r in P_]), ep) for k in P_[0]["feats"]}
        rd["availability_site_mean"] = {k: readability(torch.stack([r["feats"][k].float().mean(1) for r in P_]), ep)
                                        for k in P_[0]["feats"] if k.startswith(("s6", "s7"))}   # descriptive: Codex's
        readings[kind] = rd                                               # pooling caution, both reported
        print(kind, json.dumps(rd["factorial"]["classes"]), {k: (v["class"], round(v.get("accuracy", 0), 3)) for k, v in rd["availability"].items()}, flush=True)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    raw = dict(np.load(CV.HERE / "readings/raw.npz"))
    for pop, target in (("zombie", "ready"), ("hit", "kill")):
        B = run_["partb"][pop]
        strata = raw[f"{pop}/stratum"][::CV.K]
        keep = np.array([s.startswith("awake") for s in strata]) if pop == "zombie" else np.ones(len(B["j"]), bool)
        j = B["j"][keep]
        y = T["ready"][j] if target == "ready" else np.array([s.endswith("kill") for s in strata[keep]])
        ep, folds = T["ep"][j], folds_of(T["ep"][j])
        scores = {}
        for name, X in (("belief", B["bp"][keep].double()), ("current_percept_and_action", B["current"][keep].double())):
            s = np.zeros(len(j))
            for f in range(5):
                s[folds == f] = logistic(X[folds != f], y[folds != f], X[folds == f])
            scores[name] = s
        scores["observable_heuristic"] = observable_heuristic(L, T, j, target)
        sub = {"all": np.ones(len(j), bool)}
        sub.update({k: np.array([x.split("_")[0] == k for x in strata[keep]]) for k in ("cow", "zombie")} if pop == "hit" else
                   {k: np.array([x.endswith(k) for x in strata[keep]]) for k in ("lethal", "safe")})
        world = (raw[f"{pop}/gen_hud63"].reshape(-1, CV.K) < T["body"][raw[f"{pop}/j"].reshape(-1, CV.K)[:, 0], 0][:, None]) if pop == "zombie" \
            else raw[f"{pop}/gen_kind"].reshape(-1, CV.K) != np.array([CV.MOB[s.split("_")[0]][1] for s in strata])[:, None]
        world = world.mean(1)[keep]
        readings[f"hidden_{target}"] = {
            "rows": int(len(j)), "episodes": int(len(np.unique(ep))), "prevalence": float(y.mean()),
            "ap": {k: {sn: ap_boot(y[m], sc[m], ep[m]) for sn, m in sub.items()} for k, sc in scores.items()},
            "world_event_by_truth_x_belief_decision": {f"truth_{int(a)}_belief_{int(b)}": float(world[(y == a) & ((scores['belief'] > 0.5) == b)].mean())
                                                       for a in (0, 1) for b in (0, 1) if ((y == a) & ((scores['belief'] > 0.5) == b)).any()},
            "seen_in_smoke": "yes: the 11-06 smoke computed the earlier version of these values on the same rows (PLAN)"}
        print(target, json.dumps({k: v["all"] for k, v in readings[f"hidden_{target}"]["ap"].items()}), flush=True)
    (out / "routes.json").write_text(json.dumps(readings, indent=1, default=str))
    print("saved", out / "routes.json", flush=True)


if __name__ == "__main__":
    {"build": build, "run": run, "analyse": analyse}[sys.argv[1]]("--smoke" in sys.argv)
