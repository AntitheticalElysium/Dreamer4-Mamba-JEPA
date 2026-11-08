"""Danger checks (PLAN.md 7). Runs under either code tree: F4 under the canonical core, F3 under its own (commit
fa6edd1a's saccade/, run with that tree first on sys.path); the sampler and the first evaluation follow the tree.
  hud CKPT STEPS OUT [--health]  (GPU) C1: HUD timer strata (food, drink, energy); --health adds M1's health strata
                                 and the R0 sham on exp 2's zombie population (C2 on F3)
  probe CKPT STEPS OUT           (GPU) C3: health / wake projections at the decision level and in the final sample
  data OUT                       (CPU) D1 / D2
Usage: MAMBA_DETERMINISTIC=1 python -m artifacts.experiments.20261107_f4_followups.danger_checks STAGE ..."""
import copy, importlib, json, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S

CV = importlib.import_module("artifacts.experiments.20261106_determinant_curves.curves")
RT = importlib.import_module("artifacts.experiments.20261106_determinant_routes.routes")
DM = importlib.import_module("artifacts.experiments.20261107_f4_followups.danger_m")
HERE = Path(__file__).resolve().parent
OLD = hasattr(M, "precondition")                                        # commit fa6edd1a's EDM core
CV.HUD = {**CV.HUD, 65: ("s_body", 2)}                                  # drink's HUD cell gets its reader (PLAN 7)
TIMERS = {"food_down_awake": (64, 1, -1, False), "food_down_asleep": (64, 1, -1, True), "drink_down_awake": (65, 2, -1, False),
          "drink_down_asleep": (65, 2, -1, True), "energy_down_awake": (66, 3, -1, False), "energy_up_asleep": (66, 3, 1, True)}


def load(ckpt, steps):
    model, cfg = CV.IF.load(ckpt)
    cfg = copy.deepcopy(cfg); cfg["edm"]["sampler"]["steps"] = steps
    return model, cfg


def quiet(L, T):
    g, pos = T["g"], L["s_pos"][T["g"]].astype(int)
    dist = lambda k: np.where(L[f"s_{k}_m"][g], np.abs(L[f"s_{k}_p"][g].astype(int) - pos[:, None]).sum(-1), 99).min(1)
    return (dist("zombies") > 3) & (dist("skeletons") > 3) & ~L["s_arrows_m"][g].any(1) & (T["t"] >= CV.P) & (T["action"] != CV.DO)


def hud(ckpt, steps, out, health):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = load(ckpt, steps)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    T = CV.transitions(L)
    with torch.enable_grad():
        R, valid = CV.fit_readers(model, cfg, L, enc, 0)
    s1 = S.load_reader(TR.ROOT / Path(ckpt).parent / "s1/results.reader.pt", TR.ROOT / ckpt, CV.DEV)
    res = {"checkpoint": ckpt, "steps": steps, "tree": "fa6edd1a" if OLD else "canonical", "readers": valid, "strata": {}}
    if health:                                                           # R0: exp 2's zombie population, its seed
        rows = CV.select(CV.strata(T), np.random.default_rng(2106), 1.0)["zombie"]
        o = CV.generate(model, cfg, enc, L, T, rows, 4100 + CV.POPS.index("zombie"), R, s1)
        raw = np.load(CV.HERE / "readings/raw.npz", allow_pickle=True)
        res["sham_r0"] = {k: bool(np.array_equal(o[k], raw[f"zombie/{k}"])) for k in ("j", "gen_hud63", "gen_asleep")}
        print("R0", res["sham_r0"], flush=True)
    q, rng = quiet(L, T), np.random.default_rng(2112)
    strata = {n: (q & (T["asleep"] == sl) & ((T["body"][:, col] > 0) if d < 0 else (T["body"][:, col] < 9)), cell, col, d)
              for n, (cell, col, d, sl) in TIMERS.items()}
    if health:
        lab = DM.m1_strata(L, T)
        strata.update({n: (lab == n, 63, 0, None) for n in sorted(set(lab) - {""})})
    for i, (name, (m, cell, col, d)) in enumerate(strata.items()):
        idx = np.nonzero(m)[0]
        idx = np.sort(rng.choice(idx, DM.CAP, replace=False)) if len(idx) > DM.CAP else idx
        o = CV.generate(model, cfg, enc, L, T, idx, 4400 + i, R, s1)
        j = o["j"]; v0, v1, g = T["body"][j, col], T["true_body"][j, col], o[f"gen_hud{cell}"]; ep = T["ep"][j]
        if d:                                                            # a timer: the declared direction, and the wrong one
            ev = {"change": ((g == v0 + d).astype(float), (v1 == v0 + d).astype(float)),
                  "wrong_way": (((g - v0) * d < 0).astype(float), ((v1 - v0) * d < 0).astype(float))}
        else:                                                            # M1's health strata: drop and rise
            ev = {"drop": ((g < v0).astype(float), (v1 < v0).astype(float)), "rise": ((g > v0).astype(float), (v1 > v0).astype(float))}
        res["strata"][name] = {"rows": int(len(idx)), "episodes": int(len(np.unique(ep))), "cell": cell,
                               **{k: {"gen": RT.boot(a, ep), "true": RT.boot(b, ep), "gen_over_true": DM.ratio(a, b, ep)} for k, (a, b) in ev.items()}}
        print(name, {k: (round(v["gen"]["est"], 4), round(v["true"]["est"], 4)) for k, v in res["strata"][name].items() if isinstance(v, dict)}, flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))


@torch.no_grad()
def probe(ckpt, steps, out):
    assert os.environ.get("MAMBA_DETERMINISTIC") == "1"
    model, cfg = load(ckpt, steps)
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    den, dev, K = model.denoiser, CV.DEV, RT.K
    shapes, lo = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")], cfg["edm"]["inference_context_sigma"]
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    rows = torch.load(DM.OUT / "m2_bank.pt", weights_only=False)
    levels = ("sigma_5", "sigma_20") if OLD else ("tau_0",)
    proj = {k: [] for k in (*levels, "sample")}
    for r in rows:
        P2, Ps, b, bp, caches, fx = RT.pair_state(model, cfg, enc, r)
        A, B = Ps.p[0].float(), Ps.p[1].float()
        Wa, Wb = (torch.cat([p[:63].flatten(), v]) for p, v in ((A, Ps.v[0].float()), (B, Ps.v[1].float())))
        f_h = lambda p: ((p[:, 63].float() - B[63]) * (A[63] - B[63])).sum(-1) / (A[63] - B[63]).square().sum()
        f_w = lambda p, v: ((torch.cat([p[:, :63].float().flatten(1), v.float()], 1) - Wb) * (Wa - Wb)).sum(-1) / (Wa - Wb).square().sum()
        a = torch.full((K, 1), r["actions"][0], device=dev, dtype=torch.long)
        mem, fix, cache = model.belief.memory(bp[:1])[:, None].expand(K, -1, -1, -1), fx.expand(K, -1)[:, None], RT.stack(caches[lo][0], K)
        gen = torch.Generator(device=dev).manual_seed(r["j"])
        eps = (torch.randn((K, 1, *shapes[0]), device=dev, generator=gen), torch.randn((K, 1, *shapes[2]), device=dev, generator=gen))
        g0 = torch.zeros((K, 1, *shapes[1]), device=dev)
        with TR.autocast(dev, cfg):
            for lv in levels:                                            # the decision level: the estimate from pure noise
                if OLD:
                    s = float(lv.split("_")[1])
                    x1 = M.denoised(den, (s * eps[0], g0, s * eps[1]), torch.full((K, 1), s, device=dev), a, mem, fix, cache, clamp,
                                    cfg["edm"]["sigma_data"])[0]
                else:
                    _, e, _ = M.shortcut_ladder(cfg)
                    x1 = den.step((eps[0], g0, eps[1]), torch.full((K, 1), M.shortcut_code(0, e, den.k_max), device=dev), a, mem, fix, cache)[0]
                proj[lv].append((f_h(x1[0][:, 0]).cpu().numpy(), f_w(x1[0][:, 0], x1[2][:, 0]).cpu().numpy()))
            xs = M.sample(den, shapes, a, mem, fix, cache, cfg, torch.Generator(device=dev).manual_seed(r["j"] + 7))
        proj["sample"].append((f_h(xs[0][:, 0]).cpu().numpy(), f_w(xs[0][:, 0], xs[2][:, 0]).cpu().numpy()))
    ep = np.array([r["e"] for r in rows])
    res = {"checkpoint": ckpt, "steps": steps, "tree": "fa6edd1a" if OLD else "canonical", "pairs": len(rows),
           "projection (0 = no-hit successor, 1 = hit successor)": {
               k: {"health_mean": RT.boot(np.array([h.mean() for h, _ in v]), ep), "health_above_half": RT.boot(np.array([(h > .5).mean() for h, _ in v]), ep),
                   "wake_mean": RT.boot(np.array([w.mean() for _, w in v]), ep), "wake_above_half": RT.boot(np.array([(w > .5).mean() for _, w in v]), ep)}
               for k, v in proj.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: {q: round(x["est"], 3) for q, x in v.items()} for k, v in res["projection (0 = no-hit successor, 1 = hit successor)"].items()}), flush=True)


def data(out):
    """D1: zombie hits per (state, starting health, drop) in the train labels vs exp 2's drawn success per starting
    health (F4 K = 4 and F3 16 steps); D2: tools made while already held, per tool and count, vs first tools."""
    from scipy.stats import spearmanr
    Ltr = dict(np.load(CV.LABELS / "s0_train_labels.npz"))
    Ttr = CV.transitions(Ltr)
    h0, h1 = Ttr["body"][:, 0], Ttr["true_body"][:, 0]
    hits = Ttr["ready"] & (h1 < h0)
    exposure = {f"{'asleep' if s else 'awake'}_h{h}": int((hits & (Ttr["asleep"] == s) & (h0 == h)).sum()) for s in (False, True) for h in range(1, 10)}
    L = dict(np.load(CV.LABELS / "s0_dev_labels.npz"))
    T = CV.transitions(L)
    res = {"exposure_per_pass (train labels: ready zombie hits by state and starting health)": exposure, "success": {}}
    for name, p in (("f4_k4", HERE.parent / "20261106_p1f4/readings/consequences/curves/raw.npz"), ("f3_16", CV.HERE / "readings/raw.npz")):
        r = np.load(p, allow_pickle=True); j = r["zombie/j"]
        g, a0, a1 = r["zombie/gen_hud63"], T["body"][j, 0], T["true_body"][j, 0]
        hit = T["ready"][j] & (a1 < a0)
        cells, xs, ys = {}, [], []
        for s in (False, True):
            for h in range(1, 10):
                m = hit & (T["asleep"][j] == s) & (a0 == h)
                if m.sum() >= 30:
                    k = f"{'asleep' if s else 'awake'}_h{h}"
                    cells[k] = {"futures": int(m.sum()), "drawn_drop": float((g[m] < a0[m]).mean())}
                    xs.append(exposure[k]); ys.append(cells[k]["drawn_drop"])
        res["success"][name] = {"cells": cells, "spearman_exposure_vs_success": float(spearmanr(xs, ys)[0]) if len(xs) > 2 else None}
    CR = importlib.import_module("artifacts.experiments.20261106_decision_levels.crafting")
    s_off, a_off = np.r_[0, np.cumsum(Ltr["ep_T"] + 1)], np.r_[0, np.cumsum(Ltr["ep_T"])]
    ep = np.repeat(np.arange(len(Ltr["ep_T"])), Ltr["ep_T"]); gg = s_off[ep] + np.arange(len(ep)) - a_off[ep]
    tools = {}
    for act, (col, cell, _) in CR.TOOLS.items():
        n0, n1 = Ltr["s_inv"][gg, col], Ltr["s_inv"][gg + 1, col]
        made = (n1 == n0 + 1) & (Ltr["t_action"] == act)
        tools[str(act)] = {"first (0 -> 1)": int((made & (n0 == 0)).sum()), **{f"{n} -> {n + 1}": int((made & (n0 == n)).sum()) for n in range(1, 4)}}
    res["tools_made_per_pass (train labels)"] = tools
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res)[:1500], flush=True)


if __name__ == "__main__":
    st = sys.argv[1]
    if st == "hud":
        hud(sys.argv[2], int(sys.argv[3]), sys.argv[4], "--health" in sys.argv)
    elif st == "probe":
        probe(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        data(sys.argv[2])
