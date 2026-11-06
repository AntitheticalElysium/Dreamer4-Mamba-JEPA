"""Frozen P1-F3 tests of PLAN.md. Stages:
  movement  (CPU, JAX): 64 MOVEMENT pairs on exactly replayed dev backgrounds -> readings/movement_bank.pt
  t1        (GPU): decision levels (first evaluation, pure noise / own target / conflict target, sigma grid) for the
            audit's 208 pairs + the movement pairs; movement generation at the original and the prefix-20 start
            -> readings/t1.pt, t1.json
  t2        (GPU): the determinant curves re-generated with the prefix-20 start (same tuples and seeds) -> readings/t2/
Usage (project venv): JAX_PLATFORMS=cpu python -m artifacts.experiments.20261106_decision_levels.frozen movement
  MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261106_decision_levels.frozen t1|t2 [--smoke]
"""
import copy, importlib, json, math, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

R = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
A_ = importlib.import_module("artifacts.experiments.20261106_conditioning_audit.gpu")
CV = R.CV
HERE = Path(__file__).resolve().parent
OUT = HERE / "readings"
GRID = (0.25, 0.5, 1.0, 2.0, 3.5, 5.0, 8.0, 12.0, 20.0)
K1, MOVES, TOP, N_PREFIX = 8, (1, 2, 3, 4), 20.0, 4


# ---- movement pairs (CPU) ----

def movement():
    import jax, jax.numpy as jnp
    assert jax.default_backend() == "cpu"
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    g, pos = T["g"], L["s_pos"][T["g"]].astype(int)
    near = L["s_arrows_m"][g].any(1)
    for k in ("zombies", "skeletons"):
        near |= (L[f"s_{k}_m"][g] & (np.abs(L[f"s_{k}_p"][g].astype(int) - pos[:, None]).sum(-1) <= 6)).any(1)
    moved = (L["s_pos"][g + 1] != L["s_pos"][g]).any(1)
    idx = np.nonzero((T["t"] >= CV.P) & ~T["asleep"] & np.isin(T["action"], MOVES) & moved & ~near)[0]
    j = R.choose(T, idx, 64, 1)
    _, _, _, step, render = D.RP.env_and_render()
    rows = []
    for jj in j:
        e, t = int(T["ep"][jj]), int(T["t"][jj])
        shard, slot = int(L["ep_shard"][e]), int(L["ep_slot"][e])
        st = D.RP.advance_to(shard, slot, t - 1)
        obs = D.RP.episode_fields(shard, slot)["observations"]
        frame = np.asarray(render(st))
        assert np.array_equal(frame, obs[t - 1].numpy()), ("replay mismatch", e, t)
        actions = (int(L["t_action"][jj]), R.NOOP)
        key = D.RP._slot_keys(shard, slot)[1][t - 1]
        nxt = [step(key, st, jnp.asarray(a, jnp.int32))[1] for a in actions]
        succ = [np.asarray(render(s)) for s in nxt]
        assert np.array_equal(succ[0], obs[t].numpy()), ("natural successor differs", e, t)
        assert (np.asarray(nxt[0].player_position) != np.asarray(st.player_position)).any()
        assert (np.asarray(nxt[1].player_position) == np.asarray(st.player_position)).all()
        rows.append({"kind": "movement", "e": e, "t": t, "j": int(jj), "stratum": None, "recorded_action": actions[0],
                     "actions": actions, "frame": np.stack([frame, frame]), "successor": np.stack(succ), "truth": {}})
        D.RP._shard.cache_clear()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save(rows, OUT / "movement_bank.pt")
    print("movement pairs", len(rows), "episodes", len({r["e"] for r in rows}), flush=True)


# ---- shared GPU setup ----

def setup(model=None):
    m, cfg = CV.IF.load(CV.CKPT)
    if model is not None:
        m.load_state_dict(model)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = CV.STEPS
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    return m, cfg, L


def readers():
    """The determinant-routes run's fitted readers (the audit's loader)."""
    states = torch.load(R.HERE / "readings/run.pt", map_location="cpu", weights_only=False)["readers"]
    out = {}
    for key, state in states.items():
        r = S.MLP(state["body.0.weight"].shape[1], {"y": state["heads.y.weight"].shape[0]}).to(CV.DEV)
        r.load_state_dict(state)
        out[key] = r.eval()
    return out


def pairs(smoke):
    bank = torch.load(R.HERE / "readings/bank.pt", map_location="cpu", weights_only=False)
    rows = A_.select(bank, False) + torch.load(OUT / "movement_bank.pt", weights_only=False)
    if smoke:
        rows = [next(r for r in rows if r["kind"] == k) for k in ("sleep", "energy", "coal", "wood", "movement")]
    return rows


@torch.no_grad()
def first_eval(den, x, sigma, a, mem, fix, cache, cfg):
    with TR.autocast(CV.DEV, cfg):
        D_, _ = M.denoised(den, x, torch.full((x[0].shape[0], 1), sigma, device=CV.DEV), a, mem, fix, cache,
                           cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"])
    return D_


def ideal_reliance(d, sigmas, n=20001):
    """Equal-prior two-point posterior: mean posterior of the true target under its own noise, minus under the other's
    (= |f(conflict) - f(own)| for an ideal denoiser): 1 - 2 E[sigmoid(-d^2 / (2 s^2) + (d / s) z)]."""
    z = np.linspace(-8, 8, n); w = np.exp(-z ** 2 / 2); w /= w.sum()
    return np.array([1 - 2 * (w / (1 + np.exp(np.clip(d ** 2 / (2 * s ** 2) - (d / s) * z, -60, 60)))).sum() for s in sigmas])


def crossing(r, sigmas):
    """Where a decreasing reliance curve falls through 0.5 (log-sigma interpolation); None if it never does."""
    for i in range(len(r) - 1):
        if r[i] >= 0.5 > r[i + 1]:
            a, b = math.log(sigmas[i]), math.log(sigmas[i + 1])
            return float(math.exp(a + (r[i] - 0.5) / (r[i] - r[i + 1]) * (b - a)))
    return None if r[-1] >= 0.5 else float(sigmas[0])


@torch.no_grad()
def t1(smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg, L = setup()
    enc = S.encode(model, L, cfg)
    rd = readers()
    den, shapes = model.denoiser, [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    lo = cfg["edm"]["inference_context_sigma"]
    res = []
    for i, r in enumerate(pairs(smoke)):
        T = CV.transitions(L) if i == 0 else T
        if r["kind"] in R.SLOT:
            d = CV.IF.DIRS[L["s_dir"][int(T["g"][r["j"]])]]
            r["face_cell"] = int((3 + d[0]) * 9 + 4 + d[1])
        P2, Ps, b, bp, caches, fx = R.pair_state(model, cfg, enc, r)
        kind = "sleep" if r["kind"] == "movement" else r["kind"]            # map + lighting vector for movement
        A, B = (A_.vector(Ps.p[m].float(), Ps.v[m].float(), kind) for m in (0, 1))
        proj = lambda D_: ((A_.vector(D_[0][:, 0].float(), D_[2][:, 0].float(), kind) - A) * (B - A)).sum(-1) / (B - A).square().sum()
        gen = torch.Generator(device=CV.DEV).manual_seed(r["j"])
        eps = (torch.randn((K1, 1, *shapes[0]), device=CV.DEV, generator=gen), torch.randn((K1, 1, *shapes[2]), device=CV.DEV, generator=gen))
        g0 = torch.zeros((K1, 1, *shapes[1]), device=CV.DEV)
        tgt = [(Ps.p[m:m + 1, None].float(), Ps.v[m:m + 1, None].float()) for m in (0, 1)]
        rec = {"kind": r["kind"], "e": r["e"], "j": r["j"], "stratum": r["stratum"], "d": float((B - A).norm()), "members": []}
        for m in (0, 1):
            a = torch.full((K1, 1), r["actions"][m], device=CV.DEV, dtype=torch.long)
            mem = model.belief.memory(bp[m:m + 1])[:, None].expand(K1, -1, -1, -1)
            fix, cache = fx.expand(K1, -1)[:, None], R.stack(caches[lo][m], K1)
            f = np.zeros((3, len(GRID), K1))
            for si, s in enumerate(GRID):
                for w, base in enumerate((None, tgt[m], tgt[1 - m])):
                    x = tuple((0 if base is None else base[q]) + s * eps[q] for q in (0, 1))
                    f[w, si] = proj(first_eval(den, (x[0], g0, x[1]), s, a, mem, fix, cache, cfg)).cpu().numpy()
            gen_out = {}
            if r["kind"] == "movement":                                  # generation at the two starts (K = 16)
                for recipe in ("base5", "prefix20"):
                    a16 = torch.full((16, 1), r["actions"][m], device=CV.DEV, dtype=torch.long)
                    mem16 = model.belief.memory(bp[m:m + 1])[:, None].expand(16, -1, -1, -1)
                    p, v, _ = A_.physical(den, shapes, a16, mem16, fx.expand(16, -1)[:, None], R.stack(caches[lo][m], 16), cfg,
                                         r["j"], kind, rd["asleep"], A, B, recipe, record=False)
                    fin = ((A_.vector(p[:, 0].float(), v[:, 0].float(), kind) - A) * (B - A)).sum(-1) / (B - A).square().sum()
                    gen_out[recipe] = fin.cpu().numpy()
            rec["members"].append({"f": f, "generated_f": gen_out})
        res.append(rec)
        if (i + 1) % 25 == 0 or smoke:
            print("t1", i + 1, r["kind"], "d", round(rec["d"], 2), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"rows": res, "grid": GRID}, OUT / ("t1_smoke.pt" if smoke else "t1.pt"))
    summarise_t1(smoke)


def summarise_t1(smoke=False):
    from scipy.stats import spearmanr
    res = torch.load(OUT / ("t1_smoke.pt" if smoke else "t1.pt"), weights_only=False)["rows"]
    out, per = {}, []
    for r in res:
        f = np.stack([m["f"] for m in r["members"]])                   # [member, (pure, own, conflict), sigma, K]
        rel = np.abs(f[:, 2] - f[:, 1]).mean((0, 2))                     # reliance per sigma
        ideal = ideal_reliance(r["d"], GRID)
        per.append({"kind": r["kind"], "e": r["e"], "d": r["d"], "reliance": rel, "net": crossing(rel, GRID),
                    "ideal": crossing(ideal, GRID), "ideal_curve": ideal,
                    "pure20": [float(f[m, 0, -1].mean()) for m in (0, 1)],
                    "gen": {k: [float((r["members"][m]["generated_f"][k] < 0.5).mean()) for m in (0, 1)] for k in r["members"][0]["generated_f"]}})
    for kind in ("sleep", "energy", "coal", "wood", "movement"):
        P_ = [p for p in per if p["kind"] == kind]
        if not P_:
            continue
        nets = [p["net"] for p in P_ if p["net"] is not None]
        out[kind] = {"pairs": len(P_), "d_median": float(np.median([p["d"] for p in P_])),
                     "reliance_by_sigma": dict(zip(map(str, GRID), np.mean([p["reliance"] for p in P_], 0).round(3).tolist())),
                     "ideal_by_sigma": dict(zip(map(str, GRID), np.mean([p["ideal_curve"] for p in P_], 0).round(3).tolist())),
                     "sigma_star_net_median": float(np.median(nets)) if nets else None, "never_below_half": len(P_) - len(nets),
                     "sigma_star_ideal_median": float(np.median([p["ideal"] for p in P_ if p["ideal"] is not None])),
                     "pure_noise_f_at_20 (member 0, member 1)": np.mean([p["pure20"] for p in P_], 0).round(3).tolist()}
        if kind == "movement":
            out[kind]["generated_moved (member 0 = move, member 1 = NOOP)"] = {k: np.mean([p["gen"][k] for p in P_], 0).round(3).tolist()
                                                                               for k in P_[0]["gen"]}
    ok = [p for p in per if p["net"] is not None]
    out["spearman_sigma_star_vs_d"] = float(spearmanr([p["d"] for p in ok], [p["net"] for p in ok])[0]) if len(ok) > 2 else None
    (OUT / ("t1_smoke.json" if smoke else "t1.json")).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1), flush=True)


# ---- T2: the determinant curves with the prefix-20 start ----

def prefix_sample(den, shapes, prev_action, memory, fix, cache, cfg, gen=None, choose=None, n_prefix=N_PREFIX, top=TOP):
    """model.sample's factorised cascade with n_prefix deterministic evaluations from `top` down to sigma_max before the
    original ladder (churn only on the ladder, so the random draws match the original); the glimpse stage unchanged.
    n_prefix = 0 is model.sample (sham)."""
    e, s = cfg["edm"], cfg["edm"]["sampler"]
    n, dev, sd = s["steps"], memory.device, e["sigma_data"]
    i = torch.arange(n, device=dev, dtype=torch.float64)
    sig = (s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
    sig = torch.cat([sig, sig.new_zeros(1)]).tolist()
    pre = ((top ** (1 / 7) + np.arange(n_prefix + 1) / max(n_prefix, 1) * (s["sigma_max"] ** (1 / 7) - top ** (1 / 7))) ** 7)[:-1].tolist() if n_prefix else []
    B = memory.shape[0]
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    full = lambda a: torch.full((B, 1), a, device=dev)
    gammas = [min(s["s_churn"] / n, math.sqrt(2) - 1) if s["s_tmin"] <= sig[k] <= s["s_tmax"] else 0.0 for k in range(n)]
    start = pre[0] if pre else sig[0]
    p, v = start * randn((B, 1, *shapes[0])), start * randn((B, 1, *shapes[2]))
    g0 = torch.zeros((B, 1, *shapes[1]), device=dev)
    for k, hat in enumerate(pre):                                       # deterministic prefix down to sigma_max
        nxt = pre[k + 1] if k + 1 < len(pre) else sig[0]
        D_, _ = M.denoised(den, (p, g0, v), full(hat), prev_action, memory, fix, cache, clamp, sd)
        p, v = p + (nxt - hat) * (p - D_[0]) / hat, v + (nxt - hat) * (v - D_[2]) / hat
    for k in range(n):                                                  # model._cascade's physical stage
        hat = sig[k] * (1 + gammas[k])
        if gammas[k] > 0:
            p, v = (t + math.sqrt(hat ** 2 - sig[k] ** 2) * s["s_noise"] * randn(t.shape) for t in (p, v))
        D_, _ = M.denoised(den, (p, g0, v), full(hat), prev_action, memory, fix, cache, clamp, sd)
        p, v = p + (sig[k + 1] - hat) * (p - D_[0]) / hat, v + (sig[k + 1] - hat) * (v - D_[2]) / hat
    chosen = choose(p, v) if choose is not None else None
    fix = fix if chosen is None else chosen
    sa = s["glimpse_conditioning_sigma"]
    pc, vc = (t + sa * randn(t.shape) for t in (p, v))
    g, kept = sig[0] * randn((B, 1, *shapes[1])), []
    for k in range(n):                                                  # model._cascade's glimpse stage, unchanged
        hat = sig[k] * (1 + gammas[k])
        if gammas[k] > 0:
            g = g + math.sqrt(hat ** 2 - sig[k] ** 2) * s["s_noise"] * randn(g.shape)
        if k == 0:
            Dg = M.denoised(den, (pc, g, vc), (full(sa), full(hat)), prev_action, memory, fix, cache, clamp, sd, kept)[0][1]
        else:
            c_skip, c_out, c_in, c_noise = M.precondition(full(hat), sd)
            Fg = den.glimpse_step(M._bcast(c_in, g) * g, c_noise, prev_action, fix, cache, kept)
            Dg = M.clean_estimate((g,), (Fg,), c_skip, c_out, clamp)[0]
        g = g + (sig[k + 1] - hat) * (g - Dg) / hat
    return (p, g, v) if chosen is None else ((p, g, v), fix)


def t2(smoke):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    out = OUT / ("t2_smoke" if smoke else "t2"); out.mkdir(parents=True, exist_ok=True)
    model, cfg, L = setup()
    L = S.first_episodes(L, 60 if smoke else 0)
    enc = S.encode(model, L, cfg)
    orig = M.sample
    T = CV.transitions(L)
    lab = CV.strata(T)
    rows = CV.select(lab, np.random.default_rng(2106), 0.02 if smoke else 1.0)
    tuples = [(int(T["ep"][j]), int(T["t"][j])) for j in rows["asleep"][:16]]   # sham on the first asleep batch
    with torch.no_grad():
        a = CV.IF.futures(model, enc, L, tuples, CV.K, cfg, torch.Generator(device=CV.DEV).manual_seed(4100))
        M.sample = lambda *x, **k: prefix_sample(*x, **k, n_prefix=0)
        b = CV.IF.futures(model, enc, L, tuples, CV.K, cfg, torch.Generator(device=CV.DEV).manual_seed(4100))
    M.sample = orig
    assert all(np.array_equal(a[k], b[k]) for k in a), "prefix sampler with no prefix differs from model.sample"
    with torch.enable_grad():
        R_, valid = CV.fit_readers(model, cfg, L, enc, 40 if smoke else 0)
    s1 = S.load_reader(CV.S1_READER, TR.ROOT / CV.CKPT, CV.DEV)
    res = {"recipe": "prefix-20 (4 deterministic evaluations 20 -> 5, then the original ladder)", "sham": "bitwise equal",
           "readers": valid, "populations": {pop: {s: int((lab[pop][rows[pop]] == s).sum()) for s in sorted(set(lab[pop][rows[pop]]))} for pop in CV.POPS}}
    raw = {}
    M.sample = prefix_sample
    try:
        for i, pop in enumerate(CV.POPS):
            o = CV.generate(model, cfg, enc, L, T, rows[pop], 4100 + i, R_, s1)
            o["stratum"] = lab[pop][o["j"]].astype(str)
            raw.update({f"{pop}/{k}": x for k, x in o.items()})
            print(pop, "done", len(o["j"]), flush=True)
    finally:
        M.sample = orig
    np.savez_compressed(out / "raw.npz", **raw)
    (out / "run.json").write_text(json.dumps(res, indent=1, default=str))
    CV.summarise(out)


if __name__ == "__main__":
    {"movement": lambda s: movement(), "t1": t1, "t2": t2, "summarise_t1": summarise_t1}[sys.argv[1]]("--smoke" in sys.argv)
