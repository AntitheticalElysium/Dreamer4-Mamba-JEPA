"""PLAN v2 test D: do the frozen checkpoint's consequence rates depend on the sampling algorithm? One GPU pass.

CLAIM: the deployed consequence rates of checkpoint 10 depend on the sampling algorithm.
COMPETING: they do not (the algorithm is not the limit within the tested range).
CONTROL: one common panel (test A's contexts, first rows of each class), the same native heads, the same HUD-cell
functional as A and the same geometry statistic, under fully specified algorithms; the production sampler
(model.sample, immutable) is the baseline row. SAFE gives the false-positive rate.
Predictions written before the run (from A's field means and B's exact-field results): production reproduces A's
production rows; deterministic Heun from the production start (sigma_max * eps at sigma_max 5) may OVERSHOOT the
field's own conditional mean (B: an exact field gave 0.15 -> 0.24 for the health carrier); the sigma_max 10 arm
separates that start effect from recovery. A rate rising toward the recorded with resolution says inference settings
matter for this checkpoint; it does not say the learned distribution is right (REVIEW sec. 3), and no additive split is
read. A's field mean already bounds what any sampler of this field can return in the mean at high sigma.
Algorithms (EDM Algorithm 2 as in model.sample; Heun = EDM Algorithm 1's second-order correction; network evaluations
per frame in brackets): production Euler N 3 churn 1 [3]; Heun N 8 / 16 / 32, no churn [15 / 31 / 63]; Heun N 32 with
EDM's ImageNet churn (S_churn 40, S_tmin 0.05, S_tmax 14 so that sigma_hat <= 20, S_noise 1.003) [63]; Heun N 32 at
sigma_max 10 [63].
Usage (project venv, GPU): python -m artifacts.experiments.20261029_generation_diagnosis.solver_sensitivity A_DIR OUT_JSON
"""
import importlib, json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import model as M, train as TR
from saccade.eval import s1 as S, gates as G
FC = importlib.import_module("artifacts.experiments.20261029_generation_diagnosis.field_calibration")

ROOT = Path(__file__).resolve().parents[3]
DEV = FC.DEV
R = 2                                                                    # futures per context per algorithm
PANEL = {"risky": 1500, "adjacent": 1000, "drink": 1000, "safe": 1000, "random": 1500}
ALGORITHMS = {                                    # name: (steps, heun, s_churn, s_tmin, s_tmax, s_noise, sigma_max)
    "heun_8": (8, True, 0.0, 0.0, math.inf, 1.0, 5.0),
    "heun_16": (16, True, 0.0, 0.0, math.inf, 1.0, 5.0),
    "heun_32": (32, True, 0.0, 0.0, math.inf, 1.0, 5.0),
    "heun_32_edm_churn": (32, True, 40.0, 0.05, 14.0, 1.003, 5.0),
    "heun_32_sigma_max_10": (32, True, 0.0, 0.0, math.inf, 1.0, 10.0),
}


def sample(den, shapes, prev_action, memory, fix, cache, alg, cfg, gen):
    """EDM Algorithm 2 with the Heun correction (NVIDIA generate.py edm_sampler), sigma_max-scaled start, the network
    evaluated through model.denoised (range clamp), never committing to the cache."""
    n, heun, churn, tmin, tmax, s_noise, smax = alg
    e = cfg["edm"]; smin, rho = e["sampler"]["sigma_min"], e["sampler"]["rho"]
    i = torch.arange(n, dtype=torch.float64)
    sig = ((smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho).tolist() + [0.0]
    B, dev = memory.shape[0], memory.device
    randn = lambda shp: torch.randn(shp, device=dev, generator=gen)
    clamp = cfg["perception"]["periphery"]["tanh_clamp"]
    D = lambda x, s: M.denoised(den, x, torch.full((B, 1), s, device=dev), prev_action, memory, fix, cache, clamp,
                                e["sigma_data"])[0]
    x = tuple(sig[0] * randn((B, 1, *shp)) for shp in shapes)
    top = 0.0
    for k in range(n):
        gamma = min(churn / n, math.sqrt(2) - 1) if tmin <= sig[k] <= tmax else 0.0
        hat = sig[k] * (1 + gamma); top = max(top, hat)
        if gamma > 0:
            x = tuple(t + math.sqrt(hat ** 2 - sig[k] ** 2) * s_noise * randn(t.shape) for t in x)
        d1 = tuple((t - d) / hat for t, d in zip(x, D(x, hat)))
        xn = tuple(t + (sig[k + 1] - hat) * d for t, d in zip(x, d1))
        if heun and sig[k + 1] > 0:
            d2 = tuple((t - d) / sig[k + 1] for t, d in zip(xn, D(xn, sig[k + 1])))
            xn = tuple(t + (sig[k + 1] - hat) * (a + b) / 2 for t, a, b in zip(x, d1, d2))
        x = xn
    assert top <= 20.0 + 1e-9, top
    return tuple(t[:, 0] for t in x)


@torch.no_grad()
def run(model, enc, ctx, means, cfg, gen):
    den, neg_bins = model.denoiser, model.outcome.bin_value < -0.05
    names = ["production", *ALGORITHMS]
    out = {f"{k}_{a}": [] for a in names for k in ("f", "geo", "death", "neg")}
    out["row_ctx"] = []
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
        cache = den.new_cache()
        with TR.autocast(DEV, cfg):
            for dt in range(-FC.P, 0):
                x = tuple(take(k, dt).float()[:, None] for k in ("p", "g", "v"))
                cache = M.context_pass(den, x, take("prev_action", dt)[:, None],
                                       model.belief.memory(take("bp", dt).float())[:, None], take("fix", dt)[:, None],
                                       cache, cfg, gen)
            bp = take("bp", 0).float()
            pa, mem, fx = take("prev_action", 0)[:, None], model.belief.memory(bp)[:, None], take("fix", 0)[:, None]
            shapes = [tuple(enc[0][k].shape[1:]) for k in ("p", "g", "v")]
            for name in names:
                xg = (tuple(t[:, 0] for t in M.sample(den, shapes, pa, mem, fx, cache, cfg, gen)) if name == "production"
                      else sample(den, shapes, pa, mem, fx, cache, ALGORITHMS[name], cfg, gen))
                h = model.outcome.features(bp, model.belief.percept_embed(M.Percept(*xg, take("fix", 0)))).float()
                r = model.outcome.sample_reward(h, gen)
                out[f"f_{name}"].append(((xg[0].float()[torch.arange(B), cell] - a) * u).sum(-1).cpu())
                out[f"geo_{name}"].append(FC.geometry(xg[0]).cpu())
                out[f"death_{name}"].append(torch.sigmoid(-model.outcome.cont_logit(h, r).float()).cpu())
                out[f"neg_{name}"].append((model.outcome.bin_logp(h).exp() * neg_bins).sum(-1).cpu())
        out["row_ctx"].append(torch.tensor([start + i for i in rows]))
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main():
    a_dir, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    ck = torch.load(FC.CKPT, weights_only=False, map_location="cpu")
    cfg = ck["cfg"]
    model = M.M0(cfg).to(DEV); model.load_state_dict(ck["model"]); model.eval()
    L = dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz"))
    enc = S.encode(model, L, cfg)
    means = FC.value_means(L, enc)
    gen = torch.Generator(device=DEV).manual_seed(2030)
    a_res = json.loads((a_dir / "results.json").read_text())
    res = {"checkpoint_sha256": TR.file_sha256(FC.CKPT), "rows_per_context": R, "panel": PANEL,
           "algorithms": {k: list(v) for k, v in ALGORITHMS.items()}}
    for c, n in PANEL.items():
        ctx = [tuple(int(v) if j < 5 else bool(v) for j, v in enumerate(row)) for row in np.load(a_dir / f"{c}.npz")["ctx"][:n]]
        o = run(model, enc, ctx, means, cfg, gen)
        np.savez_compressed(out_path.parent / f"solver_{c}.npz", **o)
        ep = np.array([ctx[i][0] for i in o["row_ctx"]])
        est = lambda v: G.estimate(np.asarray(v, float), np.ones(len(ep), bool), ep)
        res[c] = {"contexts": len(ctx), "A_f_recorded": a_res[c]["f_recorded"],
                  "A_field_mean_sigma_6.67": a_res[c]["by_sigma"]["6.67"]["bias"]["est"] + a_res[c]["f_recorded"]["est"],
                  "by_algorithm": {name: {"f": est(o[f"f_{name}"]), "death_head": est(o[f"death_{name}"]),
                                          "neg_head": est(o[f"neg_{name}"]), "geometry": float(np.median(o[f"geo_{name}"]))}
                                   for name in ["production", *ALGORITHMS]}}
        print(c, {k: round(v["f"]["est"], 4) for k, v in res[c]["by_algorithm"].items()}, flush=True)
        out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
