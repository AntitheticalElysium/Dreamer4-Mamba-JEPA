"""Glimpse weighting diagnostic (11-05; user: "is there a way to check the best weighting for the glimpse?"). No frozen
measurement selects a weighting (it acts through training); this asks whether the glimpse has the deficit a weighting
toward high noise would address: does its field under-use its conditioning when its own input is hidden (the counts'
failure, RESULTS.md 3)?
Field: on the trace's RANDOM dev contexts (1,000 x 2 rows), the glimpse output of model.denoised with the REAL periphery
and lighting noised to edm.sampler.glimpse_conditioning_sigma (the cascade's conditioning), the real glimpse + noise at
sigma_g, the real context passes, memory, fixation and action (trace.run's context path): per-element MSE vs the real
glimpse at each sigma_g. Control: an MLP from STRICTLY LESS than the generator sees (the fixated window's 9 coarse
tokens, lighting, fixation one-hot) to the 576 glimpse values, fit 2-fold by episode on dev frames; its out-of-fold MSE
on the same rows bounds the window-determined irreducible error from above. Reference: the mean glimpse.
Declared reading: field MSE at sigma_g 20 > probe MSE beyond 3 SE -> the glimpse leaves window-determined content
unused at high noise (an N1-type deficit: a weighting toward high noise is indicated for the glimpse); otherwise no
such deficit is shown (EDM's low-noise emphasis costs no measured conditional content). The curve is descriptive.

Usage: python -m artifacts.experiments.20261105_consequence_trace.glimpse_check OUT_JSON CKPT"""
import importlib, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

T = importlib.import_module("artifacts.experiments.20261105_consequence_trace.trace")
M, DEV, P = T.M, T.DEV, T.P
LEVELS = (0.05, 0.2, 0.5, 1.0, 2.0, 3.5, 6.67, 20.0)


def window_inputs(ep, ts):
    fix = ep["fix"][ts].long()
    cells = M.window(fix)                                                # [n, 9]
    win = ep["p"][ts][torch.arange(len(ts))[:, None], cells].float().flatten(1)
    oh = F.one_hot((fix[:, 0] - 1) * 7 + fix[:, 1] - 1, 49).float()
    return torch.cat([win, ep["v"][ts].float(), oh], 1)


def probe(enc, rows, seed=2105):
    """2-fold by episode: fit on one half's frames, predict the other half's rows."""
    eps = np.unique([e for e, *_ in rows]); rng = np.random.default_rng(seed); rng.shuffle(eps)
    fold = {int(e): i % 2 for i, e in enumerate(eps)}
    all_eps = np.arange(len(enc)); rng.shuffle(all_eps)
    ep_fold = {int(e): fold.get(int(e), i % 2) for i, e in enumerate(all_eps)}
    out = {}
    for k in (0, 1):
        tr = [e for e in range(len(enc)) if ep_fold[e] != k]
        X = torch.cat([window_inputs(enc[e], torch.arange(P, len(enc[e]["p"]))) for e in tr]).to(DEV)
        Y = torch.cat([enc[e]["g"][P:].float().flatten(1) for e in tr]).to(DEV)
        torch.manual_seed(seed + k)
        net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 512), torch.nn.GELU(), torch.nn.Linear(512, 512),
                                  torch.nn.GELU(), torch.nn.Linear(512, Y.shape[1])).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator(device=DEV).manual_seed(seed + k)
        for epoch in range(8):
            perm = torch.randperm(len(X), device=DEV, generator=g)
            for i in range(0, len(X), 2048):
                b = perm[i:i + 2048]
                loss = F.mse_loss(net(X[b]), Y[b])
                opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for j, (e, t, *_r) in enumerate(rows):
                if ep_fold[e] == k:
                    x = window_inputs(enc[e], torch.tensor([t])).to(DEV)
                    out[j] = float(F.mse_loss(net(x), enc[e]["g"][t].float().flatten()[None].to(DEV)))
        del X, Y
    return np.array([out[j] for j in range(len(rows))])


@torch.no_grad()
def field(model, cfg, enc, rows, seed=2105):
    den, clamp, sd = model.denoiser, cfg["perception"]["periphery"]["tanh_clamp"], cfg["edm"]["sigma_data"]
    sa = cfg["edm"]["sampler"]["glimpse_conditioning_sigma"]
    res = []
    for b, start in enumerate(range(0, len(rows), 64)):
        batch = rows[start:start + 64]
        inp = T.inputs(model, cfg, enc, batch, "base")                   # R = 2 rows per context
        gen = torch.Generator(device=DEV).manual_seed(seed + b)
        cache = den.new_cache()
        with T.TR.autocast(DEV, cfg):
            for k in range(P):
                x = tuple(inp[n][:, k][:, None] for n in ("p", "g", "v"))
                cache = M.context_pass(den, x, inp["pa"][:, k][:, None], model.belief.memory(inp["bp"][:, k])[:, None],
                                       inp["fix"][:, k][:, None], cache, cfg, gen)
            a, mem, fx = inp["pa"][:, P][:, None], model.belief.memory(inp["bp"][:, P])[:, None], inp["fix"][:, P][:, None]
            x0 = (inp["x0p"][:, None], inp["x0g"][:, None], inp["x0v"][:, None])
            B = len(x0[0])
            full = lambda s: torch.full((B, 1), s, device=DEV)
            pc, vc = (t + sa * torch.randn(t.shape, device=DEV, generator=gen) for t in (x0[0], x0[2]))
            per = []
            for s in LEVELS:
                g = x0[1] + s * torch.randn(x0[1].shape, device=DEV, generator=gen)
                D, _ = M.denoised(den, (pc, g, vc), (full(sa), full(s)), a, mem, fx, cache, clamp, sd)
                per.append((D[1].float() - x0[1].float()).square().flatten(1).mean(1).cpu())
        res.append(torch.stack(per, 1))
    return torch.cat(res).numpy()                                        # [rows x R, levels]


def main():
    out, ckpt = Path(sys.argv[1]), T.ROOT / sys.argv[2]
    ck = torch.load(ckpt, weights_only=False, map_location="cpu"); cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(T.ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    enc = T.S.encode(model, L, cfg)
    ctx, _ = T.contexts(L, enc, np.random.default_rng(T.SEED))
    rows = ctx["random"]
    fm = field(model, cfg, enc, rows)                                    # [2 x rows, levels]
    pm = np.repeat(probe(enc, rows), T.R)
    gbar = torch.cat([e["g"][P:].float().flatten(1) for e in enc]).mean(0)
    mm = np.repeat(np.array([float((enc[e]["g"][t].float().flatten() - gbar).square().mean()) for e, t, *_ in rows]), T.R)
    ep = np.repeat(np.array([e for e, *_ in rows]), T.R)
    m = np.ones(len(ep), bool)
    est = lambda x: T.G.estimate(np.asarray(x, float), m, ep)
    res = {"checkpoint": str(ckpt.relative_to(T.ROOT)), "checkpoint_sha256": T.TR.file_sha256(ckpt),
           "script_sha256": T.TR.file_sha256(Path(__file__)), "rows": int(len(ep)), "levels": LEVELS,
           "field_mse": {s: est(fm[:, i]) for i, s in enumerate(LEVELS)}, "probe_mse": est(pm), "mean_predictor_mse": est(mm),
           "field20_minus_probe": est(fm[:, -1] - pm), "field_first_call_minus_probe": est(fm[:, LEVELS.index(6.67)] - pm)}
    d = res["field20_minus_probe"]
    res["reading"] = ("deficit: field at sigma_g 20 worse than the window probe" if d["est"] - 3 * d["se"] > 0
                      else "no deficit shown")
    out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or "est" in v else {kk: round(vv["est"], 4) for kk, vv in v.items()})
                      for k, v in res.items() if k not in ("levels",)}, indent=1, default=str))


if __name__ == "__main__":
    main()
