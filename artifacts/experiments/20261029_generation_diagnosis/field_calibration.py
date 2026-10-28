"""PLAN v2 test A (field calibration on context classes) and test C's native-head part, one GPU pass.

CLAIM (A): the trained denoiser's conditional mean underweights the damage consequence in low-health / adjacent-zombie
contexts at the noise levels where the context decides it.
COMPETING: the conditional mean carries it at the recorded rate (then the deployed deficit enters during sampling).
CONTROL: the law of total expectation. For the minimum-MSE denoiser D*(x_sigma, c) = E[x0 | x_sigma, c], and for a class
C that is a function of the conditioning c, E[f(D*(x0 + sigma eps, c)) | C] = E[f(x0) | C] at every sigma, for any
fixed linear functional f. Classes are defined from the state at t-1 (rendered in context frame t-1) and a_{t-1}, never
from the outcome. Controls: DRINK (deterministic +1, game_logic.do_action) and SAFE (health stays 9) must show
|bias| < 0.05 at every sigma <= 6.67 (declared in PLAN v2 before running).
OUTCOMES: controls pass and RISKY / ADJACENT bias < -0.05 beyond 3 SE at sigma in [2, 6.67] -> the field underweights
damage there; |bias| < 0.05 at all sigma while production samples stay far below -> the deficit enters in sampling;
controls fail -> the readout is unusable. Not established by either: joint correctness, per-history sharpness, event
odds.
C (native heads): death / negative-reward probabilities of the native heads on denoised-real successors (sigma 0.05,
0.2, 0.377) vs the same heads on the recorded successors: robustness near real tokens only.
Frozen P1 world seed 1, checkpoint 10; dev split; simulator labels select contexts and define the functional's values.
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.field_calibration OUT_DIR [N_EPISODES]
"""
import json, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
DEV = torch.device("cuda")
P, R = 8, 4                                                              # context frames; rows (noise draws) per context
SIGMAS = (0.05, 0.2, 0.377, 0.7, 1.2, 2.0, 3.5, 5.0, 6.67, 10.0, 20.0)  # 0.377 / 6.67: production's evaluations
HEAD_SIGMAS = 3                                                          # heads on the first three levels
HEALTH, DRINK = 63, 65                                                   # HUD cells (renderer: health (0, 0), drink (2, 0))
WATER, LAVA, DO = 3, 14, 5
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])              # constants.DIRECTIONS[:5]
CLASSES = ("risky", "adjacent", "drink", "safe", "random")
N_MAX = {"risky": None, "adjacent": 3000, "drink": 3000, "safe": 3000, "random": 3000}


def contexts(L, enc, rng):
    """(episode, t, class, cell, start value, end value, cooldown-ready) for targets t >= P; state g = t - 1."""
    s_off, a_off = np.r_[0, np.cumsum(L["ep_T"] + 1)], np.r_[0, np.cumsum(L["ep_T"])]
    pos, body, sleep, view = L["s_pos"].astype(int), L["s_body"], L["s_sleep"], L["s_view"]
    dist = {k: np.abs(L[f"s_{k}_p"].astype(int) - pos[:, None]).sum(-1) for k in ("zombies", "skeletons", "arrows")}
    rel = {k: L[f"s_{k}_p"].astype(int) - pos[:, None] for k in ("skeletons", "arrows")}
    in_view = {k: (L[f"s_{k}_m"] & (np.abs(rel[k][..., 0]) <= 3) & (np.abs(rel[k][..., 1]) <= 4)).any(1) for k in rel}
    zm = L["s_zombies_m"]
    adj, ready = (zm & (dist["zombies"] == 1)).any(1), (zm & (dist["zombies"] == 1) & (L["s_zombies_cd"] <= 0)).any(1)
    near3 = (zm & (dist["zombies"] <= 3)).any(1)
    d = DIRS[L["s_dir"]]
    face = view[np.arange(len(view)), 3 + d[:, 0], 4 + d[:, 1]]
    lava = np.stack([view[:, 3 + dr, 4 + dc] for dr, dc in DIRS[1:]], 1) == LAVA
    out = {c: [] for c in CLASSES}
    for e in range(len(enc)):
        for t in range(P, len(enc[e]["p"])):
            g, a = s_off[e] + t - 1, int(L["t_action"][a_off[e] + t - 1])
            h, dr = int(body[g, 0]), int(body[g, 2])
            hit = (HEALTH, h, max(h - 2, 0), bool(ready[g]))             # value 0 = dead (health <= 0 pooled)
            if h <= 2 and adj[g]:
                out["risky"].append((e, t, *hit))
            if h >= 3 and adj[g] and not sleep[g]:
                out["adjacent"].append((e, t, *hit))
            if not sleep[g] and face[g] == WATER and a == DO and dr <= 8:
                out["drink"].append((e, t, DRINK, dr, dr + 1, False))
            if (h == 9 and not sleep[g] and not near3[g] and not in_view["skeletons"][g] and not in_view["arrows"][g]
                    and not lava[g].any() and (body[g, 1:] >= 1).all()):
                out["safe"].append((e, t, HEALTH, 9, 7, False))
            out["random"].append((e, t, *hit))
    for c, n in N_MAX.items():
        if n is not None and len(out[c]) > n:
            out[c] = [out[c][i] for i in rng.choice(len(out[c]), n, replace=False)]
    return out


def value_means(L, enc):
    """Mean real token of the health cell per health value (<= 0 pooled as 0) and of the drink cell per drink value."""
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    p = torch.cat([enc[e]["p"][:, [HEALTH, DRINK]].float() for e in range(len(enc))])
    v = np.concatenate([L["s_body"][s_off[e]:s_off[e + 1]][:, [0, 2]] for e in range(len(enc))]).clip(min=0)
    return {cell: {int(k): p[v[:, j] == k, j].mean(0) for k in np.unique(v[:, j])} for j, cell in enumerate((HEALTH, DRINK))}


def geometry(p):
    """Median over periphery tokens of the per-token channel variance after inverting the 3 tanh(. / 3) clamp."""
    z = 3 * torch.atanh((p.float() / 3).clamp(-1 + 1e-6, 1 - 1e-6))
    return z.var(-1, unbiased=False).median(-1).values


@torch.no_grad()
def run(model, enc, ctx, means, cfg, gen):
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    neg_bins = model.outcome.bin_value < -0.05
    keys = ("f_real", "geo_real", "death_rec", "death_real", "neg_real", "f_gen", "geo_gen", "death_gen", "neg_gen",
            "f_den", "f_den_raw", "geo_den", "death_den", "neg_den", "row_ctx")
    out = {k: [] for k in keys}
    n_ctx = 256 // R
    for start in range(0, len(ctx), n_ctx):
        batch = ctx[start:start + n_ctx]
        rows = [i for i in range(len(batch)) for _ in range(R)]
        B = len(rows)
        take = lambda key, dt: torch.stack([enc[batch[i][0]][key][batch[i][1] + dt] for i in rows]).to(DEV)
        cell = torch.tensor([batch[i][2] for i in rows], device=DEV)
        a = torch.stack([means[batch[i][2]][batch[i][3]] for i in rows]).to(DEV)
        b = torch.stack([means[batch[i][2]][batch[i][4]] for i in rows]).to(DEV)
        u = (b - a) / (b - a).square().sum(-1, keepdim=True)
        f = lambda pp: ((pp.float()[torch.arange(B), cell] - a) * u).sum(-1).cpu()

        def heads(parts, r=None):
            Pc = M.Percept(*parts, take("fix", 0))
            h = model.outcome.features(bp, model.belief.percept_embed(Pc)).float()
            r = model.outcome.sample_reward(h, gen) if r is None else r
            return (torch.sigmoid(-model.outcome.cont_logit(h, r).float()).cpu(),
                    (model.outcome.bin_logp(h).exp() * neg_bins).sum(-1).cpu())

        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-P, 0):
                x = tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v"))
                cache = M.context_pass(den, x, take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None],
                                       cache, cfg, gen)
            bp = take("bp", 0).float()
            pa, mem, fx = take("prev_action", 0)[:, None], model.belief.memory(bp)[:, None], take("fix", 0)[:, None]
            x0 = tuple(take(k, 0).float() for k in ("p", "g", "v"))
            out["f_real"].append(f(x0[0])); out["geo_real"].append(geometry(x0[0]).cpu())
            out["death_rec"].append(heads(x0, take("reward", 0).float())[0])
            dr, nr = heads(x0); out["death_real"].append(dr); out["neg_real"].append(nr)
            fd, fdr, gd, dd, nd = [], [], [], [], []
            for k, s in enumerate(SIGMAS):
                sig = torch.full((B, 1), s, device=DEV)
                xs = tuple(t[:, None] + s * torch.randn(t[:, None].shape, device=DEV, generator=gen) for t in x0)
                c_skip, c_out, c_in, c_noise = M.precondition(sig, sd)
                Fo, _ = den.step(tuple(M._bcast(c_in, t) * t for t in xs), c_noise, pa, mem, fx, cache)
                raw = tuple(M._bcast(c_skip, t) * t + M._bcast(c_out, t) * o.float() for t, o in zip(xs, Fo))
                D = tuple(t.clamp(-clamp, clamp)[:, 0] for t in raw)         # = model.denoised (clean_estimate)
                fd.append(f(D[0])); fdr.append(f(raw[0][:, 0])); gd.append(geometry(D[0]).cpu())
                if k < HEAD_SIGMAS:
                    h_ = heads(D); dd.append(h_[0]); nd.append(h_[1])
            for key, v in (("f_den", fd), ("f_den_raw", fdr), ("geo_den", gd), ("death_den", dd), ("neg_den", nd)):
                out[key].append(torch.stack(v, 1))
            shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
            xg = tuple(t[:, 0] for t in M.sample(den, shapes, pa, mem, fx, cache, cfg, gen))
            out["f_gen"].append(f(xg[0])); out["geo_gen"].append(geometry(xg[0]).cpu())
            dg, ng = heads(xg); out["death_gen"].append(dg); out["neg_gen"].append(ng)
        out["row_ctx"].append(torch.tensor([start + i for i in rows]))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def summary(o, ctx, L, enc):
    ep = np.array([ctx[i][0] for i in o["row_ctx"]])
    one = np.ones(len(ep), bool)
    est = lambda v, m=one: G.estimate(np.asarray(v, float), m, ep)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    g = np.array([s_off[ctx[i][0]] + ctx[i][1] for i in o["row_ctx"]])           # state of the target frame
    cell = np.array([ctx[i][2] for i in o["row_ctx"]])
    after = np.where(cell == HEALTH, L["s_body"][g, 0], L["s_body"][g, 2])
    start = np.array([ctx[i][3] for i in o["row_ctx"]]); end = np.array([ctx[i][4] for i in o["row_ctx"]])
    res = {"rows": int(len(ep)), "contexts": int(len(ctx)),
           "recorded_reached_end_value": est(np.clip(after, 0, None) == end),
           "recorded_kept_start_value": est(after == start),
           "f_recorded": est(o["f_real"]), "f_production": est(o["f_gen"]),
           "bias_production": est(o["f_gen"] - o["f_real"]),
           "geometry_recorded": float(np.median(o["geo_real"])), "geometry_production": float(np.median(o["geo_gen"])),
           "death_recorded": est(L["t_terminated"][_trans(L, ctx, o)]),
           "death_head_recorded_reward": est(o["death_rec"]), "death_head_real": est(o["death_real"]),
           "death_head_production": est(o["death_gen"]), "neg_head_real": est(o["neg_real"]),
           "neg_head_production": est(o["neg_gen"]), "by_sigma": {}}
    for k, s in enumerate(SIGMAS):
        r = {"bias": est(o["f_den"][:, k] - o["f_real"]), "bias_unclamped": est(o["f_den_raw"][:, k] - o["f_real"]),
             "geometry": float(np.median(o["geo_den"][:, k]))}
        if k < HEAD_SIGMAS:
            r["death_head"], r["neg_head"] = est(o["death_den"][:, k]), est(o["neg_den"][:, k])
        res["by_sigma"][str(s)] = r
    rdy = np.array([ctx[i][5] for i in o["row_ctx"]])
    if rdy.any() and (~rdy).any():
        res["cooldown_split"] = {name: {"rows": int(m.sum()), "f_recorded": est(o["f_real"], m),
                                        "f_production": est(o["f_gen"], m),
                                        "bias_by_sigma": {str(s): est(o["f_den"][:, k] - o["f_real"], m)
                                                          for k, s in enumerate(SIGMAS)}}
                                 for name, m in (("ready", rdy), ("not_ready", ~rdy))}
    return res


def _trans(L, ctx, o):
    """Global transition index of a_{t-1} -> frame t for each row (t_terminated marks the transition into frame t)."""
    a_off = np.r_[0, np.cumsum(L["ep_T"])]
    return np.array([a_off[ctx[i][0]] + ctx[i][1] - 1 for i in o["row_ctx"]])


def main():
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True)
    n_ep = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_ep)
    enc = S.encode(model, L, cfg)
    a_off = np.r_[0, np.cumsum(L["ep_T"])]
    for e in range(0, len(enc), 97):                                    # labels and encoding agree on a_{t-1}
        assert (enc[e]["prev_action"][1:].numpy() == L["t_action"][a_off[e]:a_off[e + 1]]).all()
    means = value_means(L, enc)
    ctx = contexts(L, enc, np.random.default_rng(2029))
    gen = torch.Generator(device=DEV).manual_seed(2029)
    res = {"checkpoint_sha256": TR.file_sha256(CKPT), "sigmas": SIGMAS, "rows_per_context": R, "context_frames": P,
           "sampler": cfg["edm"]["sampler"], "episodes": int(len(enc)),
           "value_counts": {str(c): sorted(int(k) for k in m) for c, m in means.items()}}
    for c in CLASSES:
        o = run(model, enc, ctx[c], means, cfg, gen)
        np.savez_compressed(out_dir / f"{c}.npz", **o, ctx=np.array(ctx[c], dtype=np.int64))
        res[c] = summary(o, ctx[c], L, enc)
        r = res[c]
        print(c, r["contexts"], "f rec", round(r["f_recorded"]["est"], 4), "f prod", round(r["f_production"]["est"], 4),
              "bias", {s: round(v["bias"]["est"], 4) for s, v in r["by_sigma"].items()}, flush=True)
        (out_dir / "results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
