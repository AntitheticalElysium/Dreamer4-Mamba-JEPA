"""Learned-fixation design reading F (PLAN.md F): can the world point its own eye at where the action will act?
Label-free signal from the frozen world, pre-outcome inputs only: the action contingency of its predicted next
periphery, per cell ||D(a_taken) - D(NOOP)||^2, D the denoiser's clean estimate at production's first evaluation from
pure noise, the belief recomputed for each action. Competing: predicted change, spread of production samples;
post-outcome reference: the realized change. Window score = sum over the window's 9 cells; chooser = arg-max window.
Added 11-02 (decision 2, TEACHING 15; re-read on P1-F1.5): surprise, ||p_t - D(a_taken)||^2 per cell, the observed
periphery of frame t against the same expectation: pre-crop under the same-frame timeline (TEACHING 14), not
pre-outcome; older outputs lack it and are summarised without it. --ckpt=NAME=PATH reads that checkpoint instead.
Labels select contexts and score the choice only.
Usage (project venv, GPU): python -m artifacts.experiments.20261031_learned_fixation_sweep.locus OUT_DIR [--smoke] [--ckpt=NAME=PATH]
                           python -m artifacts.experiments.20261031_learned_fixation_sweep.locus --summary OUT_DIR
"""
import importlib, json, math, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")
LA = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.loss_accounting")
DC = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.decompose")
DEV = FC.DEV
W1 = TR.ROOT / "artifacts/saccade_m0/p1_world1"
CKPTS = {"w1_02": W1 / "ckpt_02.pt", "w1_05": W1 / "ckpt_05.pt", "w1_10": W1 / "ckpt_10.pt", "p1f_10": DC.P1F}
ATTEMPTS = {"drink": (FC.WATER, 65, None), **DC.COLLECT}                 # faced block, counter cell, tool (s_inv index)
MINED = ("stone", "coal", "iron")                                        # the faced block becomes path
NOOP, N_EP, CAP, N_RANDOM, P, K, DRAWS, NB = 0, 400, 300, 600, 8, 4, 2, 8
SIGNALS = ("contingency", "contingency_marginal", "predicted_change", "spread", "realized_change", "surprise")
CENTRES = torch.stack(torch.meshgrid(torch.arange(1, 8), torch.arange(1, 8), indexing="ij"), -1).reshape(-1, 2)
WIN = M.window(CENTRES)                                                  # [49 windows, 9 cells]


def contexts(L, rng):
    """(episode, t) per class, from state t - 1 and a_{t-1}: DO attempts as decompose, stone_no_tool, random frames."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    d = FC.DIRS[L["s_dir"]]
    face = L["s_view"][np.arange(len(d)), 3 + d[:, 0], 4 + d[:, 1]]
    e = np.concatenate([np.full(T + 1 - P, i) for i, T in enumerate(L["ep_T"])])
    t = np.concatenate([np.arange(P, T + 1) for T in L["ep_T"]])
    g = s_off[e] + t - 1
    awake_do = (L["t_action"][a_off[e] + t - 1] == FC.DO) & ~L["s_sleep"][g].astype(bool)
    masks = {}
    for c, (block, cell, tool) in ATTEMPTS.items():
        k, col = LA.HUD_SRC[cell]
        masks[c] = awake_do & (face[g] == block) & (L[k][g, col] < 9) & (True if tool is None else L["s_inv"][g, tool] > 0)
    masks["stone_no_tool"] = awake_do & (face[g] == ATTEMPTS["stone"][0]) & (L["s_inv"][g, ATTEMPTS["stone"][2]] == 0)
    masks["random"] = np.ones(len(e), bool)
    out = {}
    for c, m in masks.items():
        idx, n = np.flatnonzero(m), (N_RANDOM if c == "random" else CAP)
        idx = np.sort(rng.choice(idx, n, replace=False)) if len(idx) > n else idx
        out[c] = [(int(e[i]), int(t[i])) for i in idx]
    return out, face, d


@torch.no_grad()
def beliefs(model, cfg, enc, rows):
    """bp(t) for every action a_{t-1}: one step of the action token from the carry after frame t - 1's percept token.
    Returns {(e, t): [actions, W]} and the worst relative L2 of the recorded action against S.encode's bp."""
    need, out, worst = {}, {}, 0.0
    for e, t in rows:
        need.setdefault(e, set()).add(t)
    acts = torch.arange(cfg["data"]["actions"], device=DEV)
    for e, ts in need.items():
        ep = enc[e]
        Pe = M.Percept(*(ep[k].float()[None].to(DEV) for k in ("p", "g", "v")), ep["fix"][None].to(DEV))
        with TR.autocast(DEV, cfg):
            tok = model.belief.tokens(model.belief.percept_embed(Pe), ep["reward"][None].to(DEV),
                                      ep["terminated"][None].to(DEV), ep["action"][None].to(DEV))
            a_tok = model.belief.action(acts).to(tok.dtype)[:, None]
            carry, at = model.belief.initial(1, DEV), 0
            for t in sorted(ts):
                _, carry = model.belief(tok[:, at:2 * t - 1], carry); at = 2 * t - 1
                rep = tuple(M.MambaCarry(c.conv.expand(len(acts), *c.conv.shape[1:]).contiguous(),
                                         c.ssm.expand(len(acts), *c.ssm.shape[1:]).contiguous()) for c in carry)
                y, _ = model.belief(a_tok, rep, None, "reference")
                out[(e, t)] = y[:, 0].float().cpu()
                ref = ep["bp"][t].float()
                worst = max(worst, float((out[(e, t)][int(ep["prev_action"][t])] - ref).norm() / ref.norm().clamp(min=1e-8)))
    return out, worst


def context_cache(model, cfg, take, rep, gen):
    """M.context_pass over the P frames before t, each context repeated `rep` times with the SAME context noise."""
    den, e = model.denoiser, cfg["edm"]
    sc, cache = e["inference_context_sigma"], den.new_cache()
    for dt in range(-P, 0):
        xs = tuple((x + sc * torch.randn(x.shape, device=DEV, generator=gen)).repeat_interleave(rep, 0)
                   for x in (take(k, dt).float()[:, None] for k in ("p", "g", "v")))
        _, _, c_in, c_noise = M.precondition(torch.full((xs[0].shape[0], 1), sc, device=DEV), e["sigma_data"])
        r = lambda x: x.repeat_interleave(rep, 0)
        _, new = den.step(tuple(M._bcast(c_in, x) * x for x in xs), c_noise, r(take("prev_action", dt).long()[:, None]),
                          r(model.belief.memory(take("bp", dt).float())[:, None]), r(take("fix", dt).long()[:, None]), cache)
        cache = den.commit(cache, new)
    return cache


@torch.no_grad()
def run(model, cfg, enc, rows, bps, gen):
    """Per row: window scores [49] of each signal, and the contingency per cell [81]."""
    den, clamp, sd, s = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"], cfg["edm"]["sampler"]
    hat = s["sigma_max"] * (1 + min(s["s_churn"] / s["steps"], math.sqrt(2) - 1))    # production's first evaluation
    A, win = cfg["data"]["actions"], WIN.to(DEV)
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    out = {k: [] for k in (*SIGNALS, "cell_contingency")}
    for b0 in range(0, len(rows), NB):
        br = rows[b0:b0 + NB]
        n, ar = len(br), torch.arange(len(br), device=DEV)
        take = lambda k, dt: torch.stack([enc[e][k][t + dt] for e, t in br]).to(DEV)
        taken = take("prev_action", 0).long()
        bp = torch.stack([bps[r] for r in br]).to(DEV)                    # [n, A, W]
        with TR.autocast(DEV, cfg):
            cache = context_cache(model, cfg, take, A, gen)
            pa, fx = torch.arange(A, device=DEV).repeat(n)[:, None], take("fix", 0).long().repeat_interleave(A, 0)[:, None]
            mem = model.belief.memory(bp.flatten(0, 1))[:, None]
            Ds = []
            for _ in range(DRAWS):
                x = tuple(hat * torch.randn((n, 1, *shp), device=DEV, generator=gen).repeat_interleave(A, 0) for shp in shapes)
                Dd, _ = M.denoised(den, x, torch.full((n * A, 1), hat, device=DEV), pa, mem, fx, cache, clamp, sd)
                Ds.append(Dd[0][:, 0].float().view(n, A, *shapes[0]))
            cache_k = context_cache(model, cfg, take, K, gen)
            smp = M.sample(den, shapes, taken.repeat_interleave(K)[:, None],
                           model.belief.memory(bp[ar, taken]).repeat_interleave(K, 0)[:, None],
                           take("fix", 0).long().repeat_interleave(K, 0)[:, None], cache_k, cfg, gen)[0][:, 0].float()
        prev, real = take("p", -1).float(), take("p", 0).float()
        cells = {"contingency": sum(((d[ar, taken] - d[:, NOOP]) ** 2).sum(-1) for d in Ds) / DRAWS,
                 "contingency_marginal": sum(((d[ar, taken] - d.mean(1)) ** 2).sum(-1) for d in Ds) / DRAWS,
                 "predicted_change": sum(((d[ar, taken] - prev) ** 2).sum(-1) for d in Ds) / DRAWS,
                 "spread": smp.view(n, K, *shapes[0]).var(1).sum(-1),
                 "realized_change": ((real - prev) ** 2).sum(-1),
                 "surprise": sum(((real - d[ar, taken]) ** 2).sum(-1) for d in Ds) / DRAWS}
        for k, v in cells.items():
            out[k].append(v[:, win].sum(-1).cpu())
        out["cell_contingency"].append(cells["contingency"].cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summary(out_dir):
    """PLAN F's readings from the saved arrays (CPU)."""
    win, res = WIN.numpy(), {}
    hud = np.array([c[0] == 7 for c in CENTRES.tolist()])
    for f in sorted(out_dir.glob("*.npz")):
        z, r = np.load(f), {}
        for c in np.unique(z["cls"]):
            m = z["cls"] == c
            ep, cons = z["ep"][m], [z["counter"][m]] + ([z["faced"][m]] if c in MINED else [])
            covered = lambda ch: np.stack([(win[ch] == x[:, None]).any(1) for x in cons]).any(0)
            chance = np.stack([(win[None] == x[:, None, None]).any(-1) for x in cons]).any(0).mean(1)
            est = lambda v: G.estimate(np.asarray(v, float), np.ones(len(ep), bool), ep)
            r[c] = {"rows": int(m.sum())}
            for s in (s for s in SIGNALS if s in z):
                ch = z[s][m].argmax(1)
                e_ = est(covered(ch)) if c != "random" else None
                r[c][s] = {"coverage": e_, "counter": est((win[ch] == z["counter"][m][:, None]).any(1)) if c != "random" else None,
                           "faced": est((win[ch] == z["faced"][m][:, None]).any(1)),
                           "max_score": est(z[s][m].max(1)), "hud_share": est(hud[ch])}
                if e_ is not None:
                    r[c][s]["locates"] = bool(e_["est"] >= 0.5 and e_["est"] - 3 * e_["se"] > 3 * chance.mean())
                if c == "random":
                    for name, lm in (("bright", z["light"][m] >= 0.7), ("dark", z["light"][m] <= 0.3)):
                        if lm.any():
                            r[c][s][name] = {"rows": int(lm.sum()), "hud_share": float(hud[ch[lm]].mean()),
                                         "faced_share": float((win[ch[lm]] == z["faced"][m][lm][:, None]).any(1).mean()),
                                         "max_score": float(z[s][m][lm].max(1).mean())}
            r[c]["chance"] = float(chance.mean())
            if c != "random":
                top = z["cell_contingency"][m].argmax(1)
                r[c]["top_cell_is_consequence"] = est(np.stack([top == x for x in cons]).any(0))
        res[f.stem] = r
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1))
    for w, r in res.items():
        print("==", w)
        for c, x in r.items():
            print(f"  {c:14s} n{x['rows']:4d} chance {x['chance']:.2f}", " | ".join(
                f"{s[:12]} {x[s]['coverage']['est']:.2f}({x[s]['coverage']['se']:.2f}){'*' if x[s].get('locates') else ''}"
                if x[s]["coverage"] else f"{s[:12]} hud {x[s]['hud_share']['est']:.2f}" for s in SIGNALS if s in x),
                  f"| top cell {x['top_cell_is_consequence']['est']:.2f}" if "top_cell_is_consequence" in x else "")
    return res


def main():
    if sys.argv[1] == "--summary":
        summary(Path(sys.argv[2]))
        return
    out_dir, smoke = Path(sys.argv[1]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 12 if smoke else N_EP)
    ctx, face, d = contexts(L, np.random.default_rng(2031))
    if smoke:
        ctx = {c: rows[:6] for c, rows in ctx.items() if rows}
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    meta = {"episodes": int(len(L["ep_T"])), "contexts": {c: len(v) for c, v in ctx.items()}, "checkpoints": {}}
    ckpts = dict(a.split("=", 2)[1:] for a in sys.argv if a.startswith("--ckpt=")) or CKPTS
    for name, path in (list(ckpts.items())[2:3] if smoke else ckpts.items()):
        ck = torch.load(path, weights_only=False, map_location="cpu")
        cfg = ck["cfg"]
        model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
        del ck
        enc = S.encode(model, L, cfg)
        rows = [r for c in ctx for r in ctx[c]]
        bps, worst = beliefs(model, cfg, enc, rows)
        meta["checkpoints"][name] = {"path": str(path), "sha256": TR.file_sha256(path), "belief_replay_worst_rel_l2": worst}
        print(name, "belief replay worst relative L2", round(worst, 5), flush=True)
        if worst > 0.02:
            raise RuntimeError(f"belief replay differs: {worst}")
        gen = torch.Generator(device=DEV).manual_seed(2031)
        o = run(model, cfg, enc, rows, bps, gen)
        g = np.array([s_off[e] + t - 1 for e, t in rows])
        cls = np.array([c for c in ctx for _ in ctx[c]])
        counter = np.array([ATTEMPTS[c][1] if c in ATTEMPTS else ATTEMPTS["stone"][1] if c == "stone_no_tool" else -1 for c in cls])
        np.savez_compressed(out_dir / f"{name}.npz", **o, cls=cls, ep=np.array([e for e, _ in rows]),
                            t=np.array([t for _, t in rows]), counter=counter, faced=(3 + d[g, 0]) * 9 + 4 + d[g, 1],
                            light=L["s_light"][g], action=L["t_action"][np.r_[0, np.cumsum(L["ep_T"])][[e for e, _ in rows]] + np.array([t for _, t in rows]) - 1])
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        del model, enc
        torch.cuda.empty_cache()
    summary(out_dir)


if __name__ == "__main__":
    main()
