"""Denoiser teaching (PLAN.md 13): what the denoiser's loss teaches Mamba. Frozen F4 on P1's training chunks (Streams,
64 steps, one bootstrap row in four past bootstrap_start; bootstrap STUDENT rows keep their memory gradient, model.py
576 detaches only the teacher's), each loss term backpropagated alone at its harmoniser weight. M1 magnitude of
dL/d bp_t per term and event class (flow / bootstrap rows apart); M2 consistency |sum g| / sum |g| of the per-frame bp
gradients over a class's events; M3 reach through time: one event per stream row, that frame's G1 alone (restricted to
its token), |dL_t / d emb_{t-k}| on the belief's input embeddings for k = 1..40, and at the counter's previous reset;
M4 (dev) is the reset visible in the belief's input: digit changed from emb_t - emb_{t-1}, [emb_t, emb_{t-1}] vs the
HUD token difference, and digit == 0 from emb_t, b_t vs the HUD token.
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.denoiser_teaching CKPT OUT N_MICRO"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
HERE = Path(__file__).resolve().parent
TOK = {"food": 64, "health": 63, "energy": 66}
CLASSES = ("food_down", "recovery", "hit", "wake", "ordinary")
STREAMS, STEPS, LAGS = 4, 64, 40


def last_reset(counter, s_off, normal):
    """Per state: the index of the last state reached by a reset of this counter (a change other than a normal step),
    the episode's first state if none."""
    reset = np.zeros(len(counter), bool)
    reset[1:] = ~np.isin(np.round(np.diff(counter) * 2) / 2, normal)
    reset[s_off[:-1]] = True                                            # the episode start counts as a (visible) reset
    idx = np.where(reset, np.arange(len(counter)), 0)
    return np.maximum.accumulate(idx)


def main(ckpt, out, n_micro):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = CV.IF.load(ckpt)
    model.train()
    model.denoiser.checkpoint = cfg["p1"]["activation_checkpointing"]
    dev, P = CV.DEV, cfg["data"]["context_prefix"]
    eps = D.load_split(cfg)["train"]
    targets = np.load(TR.ROOT / cfg["p1"]["out_dir"] / "targets/change_targets.npy", mmap_mode="r")
    streams = D.Streams(eps, STREAMS, P, seed=2119, targets_offsets=D.frame_offsets(eps))
    L, C = np.load(CV.LABELS / "s0_train_labels.npz"), np.load(HERE / "readings/counters_train.npz")
    lab = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(L["ep_shard"], L["ep_slot"]))}
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    body, sleep, inv, zc, zm = L["s_body"], L["s_sleep"], L["s_inv"], L["s_zombies_cd"], L["s_zombies_m"]
    lr_hunger = last_reset(C["hunger"], s_off, (1.0, 0.5))
    lr_recover = last_reset(C["recover"], s_off, (1.0, 2.0, -1.0, -0.5))
    w = (1 / model.harmoniser.log_sigma.detach().exp()).tolist()
    carry = model.belief.initial(STREAMS, dev)
    cache = torch.zeros(STREAMS, P + 1, cfg["belief"]["width"], device=dev)
    st = {"mode": None, "y": None, "emb": None, "g1f": None, "recs": None}
    rng = np.random.default_rng(2119)

    gather, embed, sc_loss, fwd = D.gather, model.belief.percept_embed, M.shortcut_loss, model.denoiser.forward

    def gather_(episodes, recs, offsets=None):
        st["recs"] = recs
        return gather(episodes, recs, offsets)

    def embed_(Pc):
        e = embed(Pc)
        e.retain_grad()
        st["emb"] = e
        return e

    def sc_loss_(*a, **k):
        o = sc_loss(*a, **k)
        st["g1f"] = o
        return o

    def mask(i):
        def f(g):
            m = st["mode"]
            if m is None or m == "total":
                return g
            if i != 0:
                return torch.zeros_like(g)
            keep = torch.zeros(g.shape[-2], device=g.device, dtype=g.dtype)
            keep[TOK[m]] = 1
            return g * keep[:, None]
        return f

    def fwd_(*a, **k):
        o = fwd(*a, **k)
        if torch.is_grad_enabled():
            for i, t in enumerate(o):
                if t.requires_grad:
                    t.register_hook(mask(i))
        return o

    def keep_y(mod, args, output):
        output[0].retain_grad()
        st["y"] = output[0]

    def classify(recs):
        f = {c: np.zeros((STREAMS, STEPS), bool) for c in CLASSES}
        lag = {c: np.full((STREAMS, STEPS), -1) for c in ("food_down", "recovery")}
        for i in range(STREAMS):
            for k in range(1, STEPS):
                r = recs[i][P + k]
                if r is None or r[2] == 0:
                    continue
                ep = eps[r[0]]
                s = s_off[lab[(ep.shard, ep.slot)]] + r[2]
                f["food_down"][i, k] = body[s, 1] - body[s - 1, 1] == -1
                f["recovery"][i, k] = body[s, 0] - body[s - 1, 0] == 1 and lr_recover[s] == s
                f["hit"][i, k] = bool((zm[s - 1] & (zc[s - 1] <= 0) & (zc[s] == 5)).any())
                f["wake"][i, k] = sleep[s - 1] and not sleep[s]
                f["ordinary"][i, k] = (body[s] == body[s - 1]).all() and (inv[s] == inv[s - 1]).all() and sleep[s] == sleep[s - 1]
                lag["food_down"][i, k] = s - lr_hunger[s - 1]                  # steps back to the reset before this one
                lag["recovery"][i, k] = s - lr_recover[s - 1]
        return f, lag

    passes = [("g1", "g1", 0, "total")] + [(f"g1/{t}", "g1", 0, t) for t in TOK] + [("reward", "reward", 2, None), ("change", "change", 1, None)]
    m1 = {p[0]: {c: {"flow": [0.0, 0], "boot": [0.0, 0]} for c in CLASSES} for p in passes}
    m2_pairs = [("g1/food", "food_down"), ("g1/health", "recovery"), ("g1/health", "hit"), ("reward", "hit"), ("change", "wake"), ("g1", "ordinary"), ("g1", "wake")]
    m2 = {f"{p} @ {c}": [torch.zeros(cfg["belief"]["width"]), 0.0, 0] for p, c in m2_pairs}
    m3_types = {"food_down": "food", "recovery": "health", "hit": "health", "wake": "total", "ordinary": "total"}
    m3 = {c: {"profile": np.zeros(LAGS + 1), "n": 0, "at_reset": [], "near_reset": []} for c in m3_types}

    D.gather, model.belief.percept_embed, M.shortcut_loss, model.denoiser.forward = gather_, embed_, sc_loss_, fwd_
    hb = model.belief.register_forward_hook(keep_y)
    try:
        for n in range(n_micro):
            chunk = streams.next(STEPS)
            boot = np.zeros(STREAMS, bool)
            boot[n % STREAMS] = True
            terms, c_new, cache_new = TR.p1_losses(model, chunk, carry, cache, targets, cfg, boot, 10 ** 6)
            y, flags, lags = st["y"], *classify(st["recs"])
            rowtype = np.where(boot, "boot", "flow")
            for name, term, gi, mode in passes:                          # M1, M2
                st["mode"] = mode
                model.zero_grad(set_to_none=True)
                y.grad, st["emb"].grad = None, None
                (terms[term] * w[gi]).backward(retain_graph=True)
                g = torch.zeros(STREAMS, STEPS, y.shape[-1])
                if y.grad is not None:
                    g[:, 1:] = y.grad[:, 1::2][:, :-1].float().cpu()      # bp of new frame k = nxt[k - 1]
                gn = g.norm(dim=-1)
                for c in CLASSES:
                    for i in range(STREAMS):
                        m = torch.as_tensor(flags[c][i])
                        m1[name][c][rowtype[i]][0] += float(gn[i][m].sum())
                        m1[name][c][rowtype[i]][1] += int(m.sum())
                    key = f"{name} @ {c}"
                    if key in m2:
                        m = torch.as_tensor(flags[c])
                        m2[key][0] += g[m].sum(0)
                        m2[key][1] += float(gn[m].sum())
                        m2[key][2] += int(m.sum())
            for c, mode in m3_types.items():                              # M3: one event per row, its frame's loss alone
                pick = {}
                for i in range(STREAMS):
                    ks = np.nonzero(flags[c][i][LAGS + 1:])[0] + LAGS + 1
                    if len(ks):
                        pick[i] = int(rng.choice(ks))
                if not pick:
                    continue
                st["mode"] = mode
                model.zero_grad(set_to_none=True)
                y.grad, st["emb"].grad = None, None
                loss = sum(st["g1f"][i, P + k] for i, k in pick.items()) * w[0]
                loss.backward(retain_graph=True)
                ge = st["emb"].grad.float().norm(dim=-1).cpu()          # [B, STEPS]
                for i, k in pick.items():
                    prof = np.array([float(ge[i, k - j]) if j > 0 else 0.0 for j in range(LAGS + 1)])
                    m3[c]["profile"] += prof
                    m3[c]["n"] += 1
                    if c in lags and 2 < lags[c][i, k] <= LAGS - 2:
                        r = int(lags[c][i, k])
                        m3[c]["at_reset"].append(prof[r])
                        m3[c]["near_reset"].append(float(np.mean([prof[r - 2], prof[r - 1], prof[r + 1], prof[r + 2]])))
            M.carry_put(carry, list(range(STREAMS)), c_new)
            cache = cache_new
            del terms, c_new, y
            st["y"], st["emb"], st["g1f"] = None, None, None
            if n % 25 == 0:
                print(f"micro {n}/{n_micro}", {c: m3[c]["n"] for c in m3}, flush=True)
    finally:
        D.gather, model.belief.percept_embed, M.shortcut_loss, model.denoiser.forward = gather, embed, sc_loss, fwd
        hb.remove()

    res = {"checkpoint": ckpt, "microbatches": n_micro, "harmoniser_weights": w,
           "M1 mean |dL/d bp| x1e6 (pass / class / flow, boot)": {f"{p} / {c}": {rt: 1e6 * v[0] / max(1, v[1]) for rt, v in m1[p][c].items()}
                                                                    | {"frames": {rt: v[1] for rt, v in m1[p][c].items()}}
                                                                    for p in m1 for c in CLASSES},
           "M2 consistency |sum g| / sum |g| (and 1/sqrt(n))": {k: {"consistency": float(v[0].norm() / max(v[1], 1e-30)), "n": v[2],
                                                                    "random reference": float(1 / np.sqrt(max(v[2], 1)))} for k, v in m2.items()},
           "M3 reach": {}}
    for c, v in m3.items():
        prof = v["profile"] / max(1, v["n"])
        res["M3 reach"][c] = {"events": v["n"], "|dL/d emb_{t-k}| x1e9 for k = 1..40": [1e9 * float(x) for x in prof[1:]],
                              "relative to k = 1 at k = 5, 10, 20, 30, 40": [float(prof[k] / prof[1]) if prof[1] > 0 else None for k in (5, 10, 20, 30, 40)]}
        if v["at_reset"]:
            a, b = np.array(v["at_reset"]), np.array(v["near_reset"])
            res["M3 reach"][c]["at the previous reset vs its 4 neighbours (mean x1e9, ratio, n)"] = [1e9 * float(a.mean()), 1e9 * float(b.mean()), float(a.mean() / b.mean()) if b.mean() > 0 else None, int(len(a))]
    print(json.dumps({k: res[k] for k in ("M2 consistency |sum g| / sum |g| (and 1/sqrt(n))", "M3 reach")}, indent=1), flush=True)

    # M4: is the reset visible in the belief's input? (dev)
    Ld = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    model.eval()
    enc = S.encode(model, Ld, cfg)
    T = CV.transitions(Ld)
    j = np.sort(np.random.default_rng(2120).choice(np.nonzero(T["t"] >= CV.P)[0], 40000, replace=False))
    at = lambda key, dt: torch.stack([enc[int(T["ep"][x])][key][int(T["t"][x]) + dt] for x in j])

    @torch.no_grad()
    def embed_at(dt):
        Pq = M.Percept(*(at(k, dt).float().to(dev) for k in ("p", "g", "v")), at("fix", dt).to(dev))
        return torch.cat([model.belief.percept_embed(M.Percept(*(z[a:a + 4096] for z in Pq.parts()), Pq.fix[a:a + 4096])).float().cpu()
                          for a in range(0, len(j), 4096)]).double()
    e0, e1 = embed_at(-1), embed_at(0)                                   # frames t - 1 and t
    p0, p1, b1 = at("p", -1).double(), at("p", 0).double(), at("b", 0).double()
    ep = T["ep"][j]
    folds = RT.folds_of(ep)

    def probe(X, y):
        s = np.zeros(len(y))
        for f in range(5):
            s[folds == f] = RT.logistic(X[folds != f], y[folds != f], X[folds == f])
        return RT.ap_boot(y, s, ep)
    res["M4 visibility (AP, prevalence)"] = {}
    for name, col, cell in (("health", 0, 63), ("food", 1, 64), ("drink", 2, 65), ("energy", 3, 66)):
        y = (T["true_body"][j, col] != T["body"][j, col]).astype(float)
        res["M4 visibility (AP, prevalence)"][f"{name} changed"] = {"prevalence": float(y.mean()),
            "HUD token difference": probe(p1[:, cell] - p0[:, cell], y)["ap"],
            "embedding difference": probe(e1 - e0, y)["ap"], "both embeddings": probe(torch.cat([e1, e0], 1), y)["ap"]}
        if name != "health":
            z = (T["true_body"][j, col] == 0).astype(float)
            res["M4 visibility (AP, prevalence)"][f"{name} == 0"] = {"prevalence": float(z.mean()), "HUD token": probe(p1[:, cell], z)["ap"],
                                                                    "embedding": probe(e1, z)["ap"], "corrected belief b_t": probe(b1, z)["ap"]}
        print(name, res["M4 visibility (AP, prevalence)"][f"{name} changed"], flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
