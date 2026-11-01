"""Part 1 of PLAN.md: substitution tests A-C on the frozen P1 world 1 checkpoint 10. One GPU pass.
Rows: the fovea evaluation's test-F rows (drink, wood; MOVE as B's positive control), eye elsewhere at the target
frame, R draws per context; every variant uses the same draws (one generator per batch, seeded by the batch).
Per row and variant: the generated counter token's read value (nearest dev value mean of real tokens) and its distance
over that value's within-value RMS spread; the field at sigma 20 projected on the variant's own axis (shown start u ->
u + 1). MOVE: catalogue.functional's shift projection of the sample and of the field.
Variants: base (stored beliefs, S.encode) and base_rerun (the belief recomputed from the stored tokens: the reference
of every variant that recomputes it); B1 NOOP at the denoiser's a_{t-1} only; B2 NOOP also in the belief; C counter
substitution (periphery token, and glimpse token where a window covers the cell, set to a value mean): _den in the 8
context frames, beliefs stored; _full in every frame up to t - 1, belief recomputed. u: identity (each frame its own
value), zero (increment rows, u = 0), two (appearance rows, u = 2), shift (increment rows, u = v +- 3, so that u,
u + 1, v, v + 1 are distinct).
Also from the train labels: what changes each counter, by start value.
Usage (project venv, GPU): python -m artifacts.experiments.20261030_increment_diagnosis.increment_probe OUT_DIR
  [N_EPISODES: smoke tests only]
"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

EV = importlib.import_module("artifacts.experiments.20261030_fovea_placement.evaluate")
DS = importlib.import_module("artifacts.experiments.20261030_fovea_placement.dissect")
CAT, FC = EV.CAT, EV.FC
DEV, P, R = FC.DEV, FC.P, CAT.R
NOOP, SIG_FIELD, SEED, OFF = 0, 20.0, 2040, 3.0
KINDS = ("drink", "wood")


def value_stats(L, enc):
    """Per counter: periphery value means and within-value RMS spread; glimpse value means [value, centre] (fallback:
    the value's mean over centres)."""
    out = {}
    fix = torch.cat([e["fix"] for e in enc]); g = torch.cat([e["g"].float() for e in enc])
    win = M.window(fix)
    for name in KINDS:
        cell, arr, col = CAT.HUD[name]
        x = torch.as_tensor(L["s_body" if arr == 0 else "s_inv"][:, col].astype(np.int64))
        p = torch.cat([e["p"][:, cell].float() for e in enc])
        vals = sorted(int(k) for k in x.unique() if (x == k).sum() >= 2)
        means = torch.full((10, p.shape[-1]), float("nan"))
        spread = torch.full((10,), float("nan"))
        for k in vals:
            means[k] = p[x == k].mean(0)
            spread[k] = (p[x == k] - means[k]).square().sum(-1).mean().sqrt()
        r_, slot = (win == cell).nonzero(as_tuple=True)
        centre = fix[r_, 0] * 9 + fix[r_, 1]
        gt = torch.zeros(10, 81, g.shape[-1])
        for k in vals:
            mk = x[r_] == k
            any_ = g[r_[mk], slot[mk]].mean(0) if mk.any() else torch.zeros(g.shape[-1])
            gt[k] = any_
            for c in centre.unique().tolist():
                m = mk & (centre == c)
                if m.sum() >= 5:
                    gt[k, c] = g[r_[m], slot[m]].mean(0)
        out[name] = {"cell": cell, "vals": vals, "means": means, "spread": spread, "gtab": gt}
    return out


def row_inputs(model, cfg, ep, t, sub, st):
    """Context frames t - P .. t - 1, fixations / previous actions / predicted beliefs t - P .. t and the real target
    frame t under a variant sub = {noop, counter: (per-frame values [t], level), rerun}."""
    p, g, v = (ep[k][:t + 1].float().clone() for k in ("p", "g", "v"))
    fix, act, pa = ep["fix"][:t + 1], ep["action"][:t + 1].clone(), ep["prev_action"][:t + 1].clone()
    if sub.get("noop"):
        act[t - 1] = NOOP; pa[t] = NOOP
    if sub.get("counter") is not None:
        vals, level = sub["counter"]
        fr = torch.arange(t - P if level == "den" else 0, t)
        u = torch.as_tensor(vals)[fr]
        p[fr, st["cell"]] = st["means"][u]
        hit, slot = (M.window(fix[fr]) == st["cell"]).nonzero(as_tuple=True)
        g[fr[hit], slot] = st["gtab"][u[hit], fix[fr[hit], 0] * 9 + fix[fr[hit], 1]]
    if sub.get("rerun"):
        with torch.no_grad(), TR.autocast(DEV, cfg):
            Pe = M.Percept(p[:t][None].to(DEV), g[:t][None].to(DEV), v[:t][None].to(DEV), fix[:t][None].to(DEV))
            tok = model.belief.tokens(model.belief.percept_embed(Pe), ep["reward"][:t][None].to(DEV),
                                      ep["terminated"][:t][None].to(DEV), act[:t][None].to(DEV))
            y, _ = model.belief(tok, model.belief.initial(1, DEV))
        ya = y[0, 1::2].float().cpu()                                    # after action s: the predicted belief of s + 1
        bp = torch.stack([ya[s - 1] if s >= 1 else torch.zeros_like(ya[0]) for s in range(t - P, t + 1)])
    else:
        bp = ep["bp"][t - P:t + 1].float()
    return {"p": p[t - P:t], "g": g[t - P:t], "v": v[t - P:t], "fix": fix[t - P:t + 1], "pa": pa[t - P:t + 1], "bp": bp,
            "x0p": p[t], "x0g": g[t], "x0v": v[t]}


@torch.no_grad()
def generate(model, cfg, inps, seed):
    """Row inputs -> (field periphery at sigma 20, generated periphery), each [B, 81, 16]; same seed, same draws."""
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    gen = torch.Generator(device=DEV).manual_seed(seed)
    st = lambda k: torch.stack([i[k] for i in inps]).to(DEV)
    fix, pa, bp = st("fix"), st("pa"), st("bp")
    cache = den.new_cache()
    with TR.autocast(DEV, cfg):
        ctx = {k: st(k) for k in ("p", "g", "v")}
        for k in range(P):
            x = tuple(ctx[n][:, k][:, None] for n in ("p", "g", "v"))
            cache = M.context_pass(den, x, pa[:, k][:, None], model.belief.memory(bp[:, k])[:, None], fix[:, k][:, None],
                                   cache, cfg, gen)
        a, mem, fx = pa[:, P][:, None], model.belief.memory(bp[:, P])[:, None], fix[:, P][:, None]
        x0 = (st("x0p"), st("x0g"), st("x0v"))
        xs = tuple(t[:, None] + SIG_FIELD * torch.randn(t[:, None].shape, device=DEV, generator=gen) for t in x0)
        Dd, _ = M.denoised(den, xs, torch.full((len(inps), 1), SIG_FIELD, device=DEV), a, mem, fx, cache, clamp, sd)
        xg = M.sample(den, [tuple(t.shape[1:]) for t in x0], a, mem, fx, cache, cfg, gen)
    return Dd[0][:, 0].float().cpu(), xg[0][:, 0].float().cpu()


def read(st, tok):
    d = (tok[:, None] - st["means"][None]).norm(dim=-1)                 # [B, 10], nan for absent values
    d = torch.where(torch.isnan(d), torch.full_like(d, float("inf")), d)
    k = d.argmin(1)
    return k.numpy(), (d.min(1).values / st["spread"][k]).numpy()


def composition(Ltr):
    """Train split: each counter's changes by start value (+1, -1, other)."""
    s_off = np.r_[0, np.cumsum(Ltr["ep_T"] + 1)]
    first = np.zeros(len(Ltr["s_body"]), bool); first[s_off[:-1]] = True
    prev = np.maximum(np.arange(len(first)) - 1, 0)
    out = {}
    for name in KINDS:
        cell, arr, col = CAT.HUD[name]
        x = Ltr["s_body" if arr == 0 else "s_inv"][:, col].astype(int)
        d, sv = (x - x[prev])[~first], x[prev][~first]
        out[name] = {int(v): {"frames": int((sv == v).sum()), "up": int(((sv == v) & (d == 1)).sum()),
                              "down": int(((sv == v) & (d == -1)).sum()), "other": int(((sv == v) & (d != 0) & (abs(d) != 1)).sum())}
                     for v in range(10) if (sv == v).any()}
    return out


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(EV.BASE, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    del ck
    L = S.first_episodes(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), int(sys.argv[2]) if len(sys.argv) > 2 else 0)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    enc = S.encode(model, L, cfg)
    stats = value_stats(L, enc)
    rows = DS.rows_from_labels(L)
    res = {"checkpoint": str(EV.BASE), "sigma_field": SIG_FIELD, "seed": SEED, "draws_per_context": R,
           "composition_train": composition(dict(np.load(EV.ROOT / "artifacts/eda/s0/s0_train_labels.npz"))),
           "value_stats": {n: {"vals": s["vals"], "spread": s["spread"].tolist()} for n, s in stats.items()}, "kinds": {}}
    for kind in KINDS + ("move",):
        st = stats.get(kind)
        cell = st["cell"] if st else None
        x = L["s_body" if CAT.HUD.get(kind, (0, 0, 0))[1] == 0 else "s_inv"][:, CAT.HUD[kind][2]] if st else None
        ctxs = []
        for (e, t, a) in rows[kind]:
            if st and (M.window(enc[e]["fix"][t]) == cell).any():
                continue                                                 # eye elsewhere at the target frame only
            v = int(x[s_off[e] + t - 1]) if st else None
            ctxs.append((e, t, a, v))
        variants = {"base": lambda c: {}, "base_rerun": lambda c: {"rerun": True},
                    "B1_noop_den": lambda c: {"noop": True}, "B2_noop_full": lambda c: {"noop": True, "rerun": True}}
        if st:
            true = lambda c: x[s_off[c[0]]:s_off[c[0]] + c[1]].astype(np.int64)
            const = lambda c, u: np.full(c[1], u, np.int64)
            shift = lambda v: v + 3 if v + 4 <= max(st["vals"]) else v - 3
            subst = {"identity": lambda c: true(c), "zero": lambda c: const(c, 0) if c[3] >= 1 else None,
                     "two": lambda c: const(c, 2) if c[3] == 0 else None,
                     "shift": lambda c: const(c, shift(c[3])) if c[3] >= 1 else None}
            for nm, fn in subst.items():
                for level in ("den", "full"):
                    variants[f"C_{nm}_{level}"] = (lambda fn, level: lambda c: (
                        None if fn(c) is None else {"counter": (fn(c), level), "rerun": level == "full"}))(fn, level)
        K = {"contexts": len(ctxs), "episode": [c[0] for c in ctxs], "t": [c[1] for c in ctxs],
             "start_value": [c[3] for c in ctxs], "variants": {}}
        # rerun fidelity: the recomputed belief against S.encode's, on the first contexts
        d_bp = [float((row_inputs(model, cfg, enc[c[0]], c[1], {"rerun": True}, st)["bp"] - enc[c[0]]["bp"][c[1] - P:c[1] + 1].float()).abs().max())
                for c in ctxs[:20]]
        K["rerun_bp_max_abs_diff"] = max(d_bp)
        K["bp_abs_scale"] = float(torch.stack([enc[c[0]]["bp"][c[1]].float().abs().mean() for c in ctxs[:20]]).mean())
        n_ctx = 256 // R
        for vname, vf in variants.items():
            outs = {k: [] for k in ("ctx", "read", "off", "field", "shown")}
            for b0 in range(0, len(ctxs), n_ctx):
                batch = list(enumerate(ctxs[b0:b0 + n_ctx], b0))
                subs = [vf(c) for _, c in batch]
                if all(s is None for s in subs):
                    continue
                inps = []                                                # rows a variant does not apply to run as base
                for (i, c), sub in zip(batch, subs):                     # (dropped below) so every row keeps its draws
                    inps += [row_inputs(model, cfg, enc[c[0]], c[1], sub if sub is not None else {}, st)] * R
                fd, xg = generate(model, cfg, inps, SEED + b0)
                keep = np.repeat([s is not None for s in subs], R)
                outs["ctx"].append(np.repeat([i for i, _ in batch], R)[keep])
                if st:
                    shown = np.repeat([int(s["counter"][0][c[1] - 1]) if s is not None and "counter" in s else c[3]
                                       for (_, c), s in zip(batch, subs)], R)
                    k, off = read(st, xg[:, cell])
                    a_ = st["means"][shown]; w_ = st["means"][np.minimum(shown + 1, 9)] - a_
                    den_ = w_.square().sum(-1)
                    f = ((fd[:, cell] - a_) * w_).sum(-1) / den_
                    f = torch.where(den_ >= CAT.MIN_SEPARATION, f, torch.full_like(f, float("nan")))
                    outs["read"].append(k[keep]); outs["off"].append(off[keep]); outs["field"].append(f.numpy()[keep])
                    outs["shown"].append(shown[keep])
                else:
                    A, W, valid = CAT.functional("move", [c[:3] for _, c in batch], enc, L, s_off, None, None)
                    rep = np.repeat(np.arange(len(batch)), R)
                    A, W, ok = A[rep], W[rep], valid.numpy()[rep]
                    fg = ((xg - A) * W).sum((-1, -2)).numpy(); ff = ((fd - A) * W).sum((-1, -2)).numpy()
                    outs["read"].append(np.where(ok, fg, np.nan)[keep]); outs["field"].append(np.where(ok, ff, np.nan)[keep])
            o = {k: np.concatenate(v) for k, v in outs.items() if v}
            np.savez_compressed(out_dir / f"{kind}_{vname}.npz", **o)
            K["variants"][vname] = int(len(o.get("ctx", [])))
            print(kind, vname, "rows", K["variants"][vname], flush=True)
        res["kinds"][kind] = K
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    print("saved", out_dir / "results.json", flush=True)


if __name__ == "__main__":
    main()
