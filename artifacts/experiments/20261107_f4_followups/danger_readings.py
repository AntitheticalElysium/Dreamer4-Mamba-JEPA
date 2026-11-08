"""Danger readings R1-R3 (PLAN.md 8). Either code tree, as danger_checks.py (F4 canonical, F3 under fa6edd1a's core).
  phase CKPT STEPS OUT     (GPU) R1: timer events by steps since the HUD value last changed the same way, same rows
  attn CKPT STEPS OUT      (GPU) R2: attention mass of the health and lighting queries at the decision level
  reliance CKPT STEPS OUT  (GPU) R3: the output's health / wake projections with the hit (own) or no-hit (conflict)
                                    successor in its input, per signal level
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.danger_readings STAGE CKPT STEPS OUT"""
import importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
import torch.nn.functional as F

from saccade import model as M, train as TR
from saccade.eval import s1 as S

DC = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_checks")
CV, RT, DM, OLD = DC.CV, DC.RT, DC.DM, DC.OLD
R1 = {"food_down_awake": (64, 1, -1, False), "energy_up_asleep": (66, 3, 1, True), "recovery_awake": (63, 0, 1, False),
      "energy_down_awake": (66, 3, -1, False)}
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 25), (26, 30), (31, 10 ** 6)]
PER_BIN = 250
PLAYER, BESIDE = 31, (22, 30, 32, 40)                                  # the player's cell and its 4 neighbours
n_tok = M.N + M.G + 1 + 4


def since_last(L, T, col, d):
    """Per transition: steps from state t - 1 back to the last state whose value changed by sign d (same episode),
    -1 if none yet."""
    b, s_off = L["s_body"][:, col].astype(int), np.r_[0, np.cumsum(L["ep_T"] + 1)]
    last = np.full(len(b), -1)
    for e in range(len(L["ep_T"])):
        lo, hi, cur = s_off[e], s_off[e + 1], -1
        for s in range(lo, hi):
            if s > lo and np.sign(b[s] - b[s - 1]) == d:
                cur = s
            last[s] = cur
    g = T["g"]
    return np.where(last[g] >= 0, g - last[g], -1)


def phase(ckpt, steps, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = DC.load(ckpt, steps)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    with torch.enable_grad():
        R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(ckpt).parent / "s1/results.reader.pt", TR.ROOT / ckpt, CV.DEV)
    q, body, rng = DC.quiet(L, T), T["body"], np.random.default_rng(2113)
    met = (body[:, 1] > 0) & (body[:, 2] > 0) & (body[:, 3] > 0)
    cond = {"food_down_awake": ~T["asleep"] & (body[:, 1] > 0), "energy_up_asleep": T["asleep"] & (body[:, 3] < 9),
            "recovery_awake": ~T["asleep"] & met & (body[:, 0] < 9), "energy_down_awake": ~T["asleep"] & (body[:, 3] > 0)}
    res = {"checkpoint": ckpt, "steps": steps, "tree": "fa6edd1a" if OLD else "canonical", "strata": {}}
    for i, (name, (cell, col, d, _)) in enumerate(R1.items()):
        since = since_last(L, T, col, d)
        lab = np.full(len(since), "", object)
        lab[since == -1] = "none"
        for lo_, hi_ in BINS:
            lab[(since >= lo_) & (since <= hi_)] = f"{lo_}-{hi_}" if hi_ < 10 ** 6 else f"{lo_}+"
        rows = []
        for bname in [f"{a}-{b}" if b < 10 ** 6 else f"{a}+" for a, b in BINS] + ["none"]:
            idx = np.nonzero(q & cond[name] & (lab == bname))[0]
            rows.append(np.sort(rng.choice(idx, PER_BIN, replace=False)) if len(idx) > PER_BIN else idx)
        rows = np.sort(np.concatenate(rows))
        o = CV.generate(model, cfg, enc, L, T, rows, 4500 + i, R, s1)
        j = o["j"]; v0, v1, g = body[j, col], T["true_body"][j, col], o[f"gen_hud{cell}"]; ep = T["ep"][j]
        drawn, true = (g == v0 + d).astype(float), (v1 == v0 + d).astype(float)
        res["strata"][name] = {b: {"rows": int((lab[rows] == b).sum()), "drawn": RT.boot(drawn[lab[j] == b], ep[lab[j] == b]),
                                   "true": RT.boot(true[lab[j] == b], ep[lab[j] == b]),
                                   "wrong_way": float((((g - v0) * d) < 0)[lab[j] == b].mean())}
                               for b in sorted(set(lab[rows]), key=lambda s: (s == "none", 0 if s == "none" else int(s.split("-")[0].rstrip("+"))))}
        print(name, {b: (round(v["drawn"]["est"], 3), round(v["true"]["est"], 3)) for b, v in res["strata"][name].items()}, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


def decision_inputs(model, cfg, enc, r, K):
    """M2 pair -> (both members' (memory, cache), the successors, actions, fixation) at natural inputs per member."""
    P2, Ps, b, bp, caches, fx = RT.pair_state(model, cfg, enc, r)
    lo = cfg["edm"]["inference_context_sigma"]
    dev = CV.DEV
    a = torch.full((K, 1), r["actions"][0], device=dev, dtype=torch.long)
    members = [(model.belief.memory(bp[m:m + 1])[:, None].expand(K, -1, -1, -1), RT.stack(caches[lo][m], K)) for m in (0, 1)]
    return members, Ps, a, fx.expand(K, -1)[:, None]


def first_eval(model, cfg, x, a, mem, fix, cache, level):
    """The decision-level evaluation: F4 the sampler's code at signal `level` (an index), F3 the denoised estimate at
    sigma `level`. x = (p, g, v) already at that level. Returns the clean estimate (p, g, v)."""
    den, dev, K = model.denoiser, CV.DEV, x[0].shape[0]
    with TR.autocast(dev, cfg):
        if OLD:
            return M.denoised(den, x, torch.full((K, 1), float(level), device=dev), a, mem, fix, cache,
                              cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"])[0]
        _, e, _ = M.shortcut_ladder(cfg)
        return den.step(x, torch.full((K, 1), M.shortcut_code(int(level), e, den.k_max), device=dev), a, mem, fix, cache)[0]


@torch.no_grad()
def attn(ckpt, steps, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = DC.load(ckpt, steps)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    den, dev, K = model.denoiser, CV.DEV, 4
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    phys = torch.cat([torch.arange(M.N), torch.arange(M.N + M.G, n_tok)])
    pos = {int(t): i for i, t in enumerate(phys.tolist())}                 # token -> physical query / key index
    Q = {"health": 63, "lighting": M.N + M.G}
    state = {"block": None, "rows": []}
    hooks = [blk.register_forward_pre_hook(lambda mod, args, i=i: state.__setitem__("block", i)) for i, blk in enumerate(den.blocks)]
    sdpa = F.scaled_dot_product_attention

    def groups_of(token_of_key, slot_of_key, now, qtok):
        g = {}
        for name, sel in (("beside_now", lambda t, s: (s == now) & np.isin(t, BESIDE)), ("beside_past", lambda t, s: (s < now) & np.isin(t, BESIDE)),
                          ("self_now", lambda t, s: (s == now) & (t == qtok)), ("self_past", lambda t, s: (s < now) & (t == qtok)),
                          ("memory", lambda t, s: t >= M.N + M.G + 1), ("hud_other", lambda t, s: (t >= 63) & (t < M.N) & (t != qtok))):
            g[name] = sel(token_of_key, slot_of_key)
        g["rest"] = ~np.any(list(g.values()), 0)
        return g

    def capture(q, k, v, attn_mask=None, **kw):
        blk = state["block"]; nq, nk = q.shape[-2], k.shape[-2]
        w = (q.float() @ k.float().transpose(-1, -2)) / q.shape[-1] ** 0.5
        if attn_mask is not None:
            w = w.masked_fill(~attn_mask, float("-inf"))
        w = w.softmax(-1)
        if nq == len(phys) and nk == len(phys):                          # spatial, physical queries and keys
            tok, slot = phys.numpy(), np.zeros(nk, int)
            for name, t in Q.items():
                gr = groups_of(tok, slot, 0, t)
                state["rows"].append((blk, "spatial", name, {k_: float(w[..., pos[t], :][..., torch.as_tensor(m, device=w.device)].sum(-1).mean()) for k_, m in gr.items()}))
        elif nq == len(phys) and nk % n_tok == 0 and nk > len(phys):     # F4 dense, physical queries (keys: frames x tokens)
            tok, slot = np.arange(nk) % n_tok, np.arange(nk) // n_tok
            for name, t in Q.items():
                gr = groups_of(tok, slot, nk // n_tok - 1, t)
                state["rows"].append((blk, "dense", name, {k_: float(w[..., pos[t], :][..., torch.as_tensor(m, device=w.device)].sum(-1).mean()) for k_, m in gr.items()}))
        elif nq == 1 and OLD:                                           # F3 per-token time attention: [B n, H, 1, F + 1]
            wt = w.reshape(-1, n_tok, *w.shape[1:])
            for name, t in Q.items():
                state["rows"].append((blk, "time", name, {"self_past": float(wt[:, t, ..., :-1].sum(-1).mean()), "self_now": float(wt[:, t, ..., -1].mean())}))
        return sdpa(q, k, v, attn_mask=attn_mask, **kw)

    rows = torch.load(DM.OUT / "m2_bank.pt", weights_only=False)
    acc = {}
    try:
        for r in rows:
            members, Ps, a, fix = decision_inputs(model, cfg, enc, r, K)
            gen = torch.Generator(device=dev).manual_seed(r["j"])
            eps = (torch.randn((K, 1, *shapes[0]), device=dev, generator=gen), torch.randn((K, 1, *shapes[2]), device=dev, generator=gen))
            g0 = torch.zeros((K, 1, *shapes[1]), device=dev)
            for m, (mem, cache) in enumerate(members):
                state["rows"] = []
                F.scaled_dot_product_attention = capture
                lv, s = (5.0, 5.0) if OLD else (0, 1.0)
                first_eval(model, cfg, (s * eps[0], g0, s * eps[1]), a, mem, fix, cache, lv)
                F.scaled_dot_product_attention = sdpa
                for blk, kind, name, gm in state["rows"]:
                    for k_, x in gm.items():
                        acc.setdefault((["zombie", "removed"][m], blk, kind, name, k_), []).append(x)
    finally:
        F.scaled_dot_product_attention = sdpa
        for h in hooks:
            h.remove()
    res = {"checkpoint": ckpt, "tree": "fa6edd1a" if OLD else "canonical", "pairs": len(rows),
           "mass (member / block / kind / query / key group)": {" / ".join(map(str, k)): float(np.mean(v)) for k, v in sorted(acc.items())}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v, 3) for k, v in res["mass (member / block / kind / query / key group)"].items() if "beside" in k and "zombie /" in k}), flush=True)


@torch.no_grad()
def reliance(ckpt, steps, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = DC.load(ckpt, steps)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    dev, K = CV.DEV, RT.K
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    levels = (5.0, 2.0, 1.0, 0.5, 0.2) if OLD else (0, 2, 4, 6)
    k_max = None if OLD else model.denoiser.k_max
    rows = torch.load(DM.OUT / "m2_bank.pt", weights_only=False)
    acc = {}
    for r in rows:
        members, Ps, a, fix = decision_inputs(model, cfg, enc, r, K)
        mem, cache = members[0]                                          # natural inputs: the hit member's
        A, B = Ps.p[0].float(), Ps.p[1].float()
        Wa, Wb = (torch.cat([p[:63].flatten(), v]) for p, v in ((A, Ps.v[0].float()), (B, Ps.v[1].float())))
        f_h = lambda p: ((p[:, 0, 63].float() - B[63]) * (A[63] - B[63])).sum(-1) / (A[63] - B[63]).square().sum()
        f_w = lambda p, v: ((torch.cat([p[:, 0, :63].float().flatten(1), v[:, 0].float()], 1) - Wb) * (Wa - Wb)).sum(-1) / (Wa - Wb).square().sum()
        gen = torch.Generator(device=dev).manual_seed(r["j"])
        eps = (torch.randn((K, 1, *shapes[0]), device=dev, generator=gen), torch.randn((K, 1, *shapes[2]), device=dev, generator=gen))
        g0 = torch.zeros((K, 1, *shapes[1]), device=dev)
        for lv in levels:
            for tname, (tp, tv) in (("own", (Ps.p[0], Ps.v[0])), ("conflict", (Ps.p[1], Ps.v[1]))):
                tp, tv = tp.float()[None, None], tv.float()[None, None]
                if OLD:
                    x = (tp + lv * eps[0], g0, tv + lv * eps[1])
                else:
                    tau = lv / k_max
                    x = ((1 - tau) * eps[0] + tau * tp, g0, (1 - tau) * eps[1] + tau * tv)
                est = first_eval(model, cfg, x, a, mem, fix, cache, lv)
                acc.setdefault((lv, tname, "health"), []).append(float(f_h(est[0]).mean()))
                acc.setdefault((lv, tname, "wake"), []).append(float(f_w(est[0], est[2]).mean()))
    ep = np.array([r["e"] for r in rows])
    res = {"checkpoint": ckpt, "tree": "fa6edd1a" if OLD else "canonical", "pairs": len(rows),
           "levels": "sigma" if OLD else "signal index of k_max (tau = index / k_max)",
           "projection (0 = no-hit successor, 1 = hit successor)": {f"{lv} / {t} / {w}": RT.boot(np.array(v), ep) for (lv, t, w), v in acc.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: round(v["est"], 3) for k, v in res["projection (0 = no-hit successor, 1 = hit successor)"].items()}), flush=True)


if __name__ == "__main__":
    {"phase": phase, "attn": attn, "reliance": reliance}[sys.argv[1]](sys.argv[2], int(sys.argv[3]), sys.argv[4])
