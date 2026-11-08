"""P1-F1.5 readings 3 (per-query consistency) and 4 (real- vs generated-conditioning gap), PLAN.md; GPU.
Contexts: the first 150 dev episodes (S1's encoding, its seeded fixations), rows with a full 8-frame context, 128 drawn
uniformly and 128 whose previous action is DO (attempt consequences), rng 610400. Per batch of contexts: the real
context committed by context_pass, then M.sample with a choose hook that returns the real fixation and keeps the
stage-1 periphery / lighting and the generator state, so the glimpse stage below (a copy of M._cascade's stage 2) is
checked against M.sample's own glimpse, bitwise, on every batch.
3. Per legal fixation (49), K = 8 glimpses on that one sampled periphery (independent draws): the reward head's
   expectation and the continuation probability at a reward drawn from the head; per context the spread across
   fixations (std of the 49 means over K) against the RMS standard error of a mean. Coherence at every fixation: pixel
   MSE between the TSP render (predicted belief, periphery, lighting) and the reconstruction render (glimpse,
   lighting) of the crop, on the generated percept and on the real one (frame t's periphery and its real glimpse at
   that fixation, encoded from the frame), by window type (HUD / map).
4. At the real fixation, K glimpses given the real periphery and lighting and K given the sampled ones (each noised
   to the conditioning level by the stage itself): token MSE to the real glimpse; reconstruction render's pixel MSE
   to the real crop.
Usage (project venv, GPU): python -m artifacts.experiments.20261101_p1f15.consistency CKPT OUT_DIR [--smoke]
"""
import importlib, json, math, os, sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import torch

from saccade import data as D, model as M, train as TR
from saccade.eval import s1 as S, gates as G

F = importlib.import_module("artifacts.experiments.20261031_learned_fixation_sweep.locus")
DEV = torch.device("cuda")
K, BATCH, N_EPISODES, N_EACH = 8, 8, 150, 128
CEN, HUD = F.CENTRES, (F.WIN >= 63).any(-1).numpy()                  # [49, 2]; windows touching the HUD rows


def glimpse_stage(den, p, v, prev_action, memory, fix, cache, cfg, gen):
    """M._cascade's stage 2 (the stage-2 cache included): its schedule, arithmetic and draw order, given (p, v)."""
    s, sd = cfg["edm"]["sampler"], cfg["edm"]["sigma_data"]
    n, dev, B = s["steps"], memory.device, memory.shape[0]
    i = torch.arange(n, device=dev, dtype=torch.float64)
    sig = (s["sigma_max"] ** (1 / s["rho"]) + i / (n - 1) * (s["sigma_min"] ** (1 / s["rho"]) - s["sigma_max"] ** (1 / s["rho"]))) ** s["rho"]
    sig = torch.cat([sig, sig.new_zeros(1)]).tolist()
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    clamp, full = cfg["perception"]["periphery"]["tanh_clamp"], lambda a: torch.full((B, 1), a, device=dev)
    gammas = [min(s["s_churn"] / n, math.sqrt(2) - 1) if s["s_tmin"] <= sig[k] <= s["s_tmax"] else 0.0 for k in range(n)]
    sa = s["glimpse_conditioning_sigma"]
    pc, vc = (t + sa * randn(t.shape) for t in (p, v))
    g, kept = sig[0] * randn((B, 1, M.G, cfg["perception"]["fovea"]["token_dim"])), []
    for k in range(n):
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
    return g


def rows_of(enc, rng, n_each):
    full = [(e, t) for e in range(len(enc)) for t in range(8, len(enc[e]["p"]))]
    do = [(e, t) for e, t in full if int(enc[e]["prev_action"][t]) == F.FC.DO]
    pick = lambda r: [r[i] for i in sorted(rng.choice(len(r), min(n_each, len(r)), replace=False))]
    return [(e, t, "random") for e, t in pick(full)] + [(e, t, "do") for e, t in pick(do)]


def heads(model, bp, Pc, gen):
    """Reward expectation and continuation probability (at a reward drawn from the head)."""
    h = model.outcome.features(bp, model.belief.percept_embed(Pc)).float()
    er = (model.outcome.bin_logp(h).exp() * model.outcome.bin_value).sum(-1)
    return er, torch.sigmoid(model.outcome.cont_logit(h, model.outcome.sample_reward(h, gen)).float())


def render_gap(model, bp, Pc, cells):
    """Pixel MSE between the TSP and the reconstruction renders of the crop, per row."""
    tsp = model.pixels(model.tsp.predict(bp, Pc.p, Pc.v, cells), Pc.v, cells)[0].float().clamp(-1, 1)
    rec = model.pixels(model.tsp.reconstruct(Pc.g, Pc.v, cells), Pc.v, cells)[0].float().clamp(-1, 1)
    return (tsp - rec).square().flatten(1).mean(1)


@torch.no_grad()
def main():
    ckpt, out_dir, smoke = Path(sys.argv[1]), Path(sys.argv[2]), "--smoke" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ckpt, weights_only=False, map_location="cpu"); cfg = ck["cfg"]
    assert cfg["denoiser"].get("factorised")
    model = M.M0(cfg).to(DEV).eval(); model.load_state_dict(ck["model"]); del ck
    den, sd, cen = model.denoiser, cfg["edm"]["sigma_data"], CEN.to(DEV)
    L = S.first_episodes(dict(np.load(S.LABELS / "s0_dev_labels.npz")), 6 if smoke else N_EPISODES)
    enc, eps = S.encode(model, L, cfg), S.episodes(L)
    rows = rows_of(enc, np.random.default_rng(610400), 4 if smoke else N_EACH)
    frames = {e: D.RP.episode_fields(eps[e].shard, eps[e].slot)["observations"] for e in sorted({e for e, _, _ in rows})}
    shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
    rec = {k: [] for k in ("er", "er_se", "pc", "pc_se", "coh_gen", "coh_real", "tok_real_cond", "tok_gen_cond",
                           "pix_real_cond", "pix_gen_cond")}
    checks = []
    for b0 in range(0, len(rows), BATCH):
        batch = rows[b0:b0 + BATCH]; B = len(batch)
        take = lambda key, dt: torch.stack([enc[e][key][t + dt] for e, t, _ in batch]).to(DEV)
        cache, gen = den.new_cache(), torch.Generator(device=DEV).manual_seed(610401 + b0)
        with TR.autocast(DEV, cfg):
            for dt in range(-8, 0):
                cache = M.context_pass(den, tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v")), take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None], cache, cfg, gen)
            pa, bp, fix = take("prev_action", 0)[:, None], take("bp", 0).float(), take("fix", 0)[:, None]
            mem = model.belief.memory(bp)[:, None]
            kept = {}
            def hook(p, v):
                kept.update(p=p, v=v, state=gen.get_state())
                return fix
            (frame, _) = M.sample(den, shapes, pa, mem, fix, cache, cfg, gen, choose=hook)
            g_copy = glimpse_stage(den, kept["p"], kept["v"], pa, mem, fix, cache, cfg, torch.Generator(device=DEV).set_state(kept["state"]))
            checks.append(bool(torch.equal(g_copy, frame[1])))
            assert checks[-1], "the glimpse stage copy differs from M.sample's"
            p, v = kept["p"], kept["v"]                                  # one sampled periphery / lighting per context
            rep = lambda t: t.repeat_interleave(K, 0)
            cache_k = {"t": cache["t"], "kv": [(rep(a), rep(c)) for a, c in cache["kv"]]}
            pk, vk, pak, memk, bpk = rep(p), rep(v), rep(pa), rep(mem), rep(bp)
            g2 = torch.Generator(device=DEV).manual_seed(610402 + b0)
            fr = torch.stack([torch.as_tensor(frames[e][t]) for e, t, _ in batch]).to(DEV)
            preal, vreal = take("p", 0).float(), take("v", 0).float()
            er_m, er_s, pc_m, pc_s, coh_g, coh_r = (np.zeros((B, 49)) for _ in range(6))
            for w in range(49):
                fw = cen[w].expand(B * K, 1, 2)
                gw = glimpse_stage(den, pk, vk, pak, memk, fw, cache_k, cfg, g2)
                Pw = M.Percept(pk[:, 0], gw[:, 0].float(), vk[:, 0], fw[:, 0])
                e_, c_ = (x.view(B, K).cpu().numpy() for x in heads(model, bpk, Pw, g2))
                er_m[:, w], er_s[:, w], pc_m[:, w], pc_s[:, w] = e_.mean(1), e_.std(1, ddof=1) / math.sqrt(K), c_.mean(1), c_.std(1, ddof=1) / math.sqrt(K)
                cells = M.window(fw[:, 0])
                coh_g[:, w] = render_gap(model, bpk, Pw, cells).view(B, K).mean(1).cpu().numpy()
                g_real = model.perception(fr, cen[w].expand(B, 2)).g
                coh_r[:, w] = render_gap(model, bp, M.Percept(preal, g_real.float(), vreal, cen[w].expand(B, 2)), M.window(cen[w].expand(B, 2))).cpu().numpy()
            greal, crop = take("g", 0).float(), M.cell_pixels(M.crop(fr, fix[:, 0])).float() / 127.5 - 1
            cells_r, fixk = M.window(rep(fix)[:, 0]), rep(fix)
            for name, (pp, vv) in (("real_cond", (rep(preal[:, None]), rep(vreal[:, None]))), ("gen_cond", (pk, vk))):
                gk = glimpse_stage(den, pp, vv, pak, memk, fixk, cache_k, cfg, g2)[:, 0].float()
                rec[f"tok_{name}"].append((gk - rep(greal)).square().flatten(1).mean(1).view(B, K).mean(1).cpu().numpy())
                rmean = model.pixels(model.tsp.reconstruct(gk, rep(vreal), cells_r), rep(vreal), cells_r)[0].float().clamp(-1, 1)
                rec[f"pix_{name}"].append((rmean - rep(crop)).square().flatten(1).mean(1).view(B, K).mean(1).cpu().numpy())
        for k, x in (("er", er_m), ("er_se", er_s), ("pc", pc_m), ("pc_se", pc_s), ("coh_gen", coh_g), ("coh_real", coh_r)):
            rec[k].append(x)
        print("contexts", b0 + B, "/", len(rows), flush=True)
    arr = {k: np.concatenate(v) for k, v in rec.items()}
    ep = np.array([e for e, _, _ in rows]); kind = np.array([k for _, _, k in rows])
    np.savez_compressed(out_dir / "consistency.npz", ep=ep, t=np.array([t for _, t, _ in rows]), kind=kind, **arr)
    res = {"checkpoint": str(ckpt), "sha256": TR.file_sha256(ckpt), "script_sha256": TR.file_sha256(Path(__file__)),
           "contexts": {k: int((kind == k).sum()) for k in ("random", "do")}, "K": K, "glimpse_stage_equal_to_M_sample": all(checks),
           "batches_checked": len(checks)}
    for k in ("random", "do"):
        m = kind == k
        r = res[k] = {}
        for h in ("er", "pc"):
            spread, se = arr[h][m].std(1, ddof=1), np.sqrt((arr[f"{h}_se"][m] ** 2).mean(1))
            r[h] = {"spread_over_se_median": float(np.median(spread / np.maximum(se, 1e-12))),
                    "spread_median": float(np.median(spread)), "se_rms_median": float(np.median(se)),
                    "mean_over_fixations": float(arr[h][m].mean())}
        gap = arr["coh_gen"][m] - arr["coh_real"][m]
        r["coherence"] = {wt: {"gen": G.estimate(arr["coh_gen"][m][:, sel].mean(1), np.ones(m.sum(), bool), ep[m]),
                               "real": G.estimate(arr["coh_real"][m][:, sel].mean(1), np.ones(m.sum(), bool), ep[m]),
                               "gen_minus_real": G.estimate(gap[:, sel].mean(1), np.ones(m.sum(), bool), ep[m])}
                          for wt, sel in (("hud", HUD), ("map", ~HUD))}
        r["conditioning_gap"] = {q: {c: G.estimate(arr[f"{q}_{c}"][m], np.ones(m.sum(), bool), ep[m]) for c in ("real_cond", "gen_cond")}
                                 | {"gen_minus_real": G.estimate(arr[f"{q}_gen_cond"][m] - arr[f"{q}_real_cond"][m], np.ones(m.sum(), bool), ep[m])}
                                 for q in ("tok", "pix")}
    (out_dir / "consistency.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res, default=float)[:3000], flush=True)


if __name__ == "__main__":
    main()
