"""PLAN v2 test B: the sampling algorithms on an EXACT field (EDM sec. 4's analytical-denoiser method), CPU only.

CLAIM: the production algorithm (EDM Algorithm 2, Euler, N = 3, rho = 7, sigma_max 5, S_churn 1, evaluated at sigma_hat,
start sigma_max * eps), given the exact posterior of a two-outcome mixture, returns the rare outcome far less often
than its probability.
COMPETING: it returns it at its probability; then the algorithm cannot by itself explain the deployed deficit.
CONTROL: the same exact field under many-step algorithms started from the true noised mixture, which must return the
mixture weight (a check of this code), and the law of total expectation evaluated on the exact field (zero bias at
every sigma: PLAN v2 A's amendment, check iv).
Geometry: the two outcomes are the mean real tokens of a HUD cell at two values (frozen P1 world seed 1, checkpoint 10,
dev episodes), repeated over k cells for a consequence with a larger footprint; all other percept coordinates are equal
across the outcomes and do not enter the exact posterior. Says nothing about the trained field.
Usage (project venv, CPU): python -m artifacts.experiments.20261029_generation_diagnosis.exact_field OUT_JSON
"""
import json, math, sys
from pathlib import Path

import numpy as np
import torch

from saccade import data as D, model as M
from saccade.eval import s1 as S

ROOT = Path(__file__).resolve().parents[3]
CKPT = ROOT / "artifacts/saccade_m0/p1_world1/ckpt_10.pt"
HEALTH, DRINK = 63, 65
PROBS = (0.01, 0.03, 0.1, 0.15, 0.3, 0.5)
FOOTPRINT = (1, 2, 4, 9)
RUNS = 20000
ALGORITHMS = {                                    # name: (steps, heun, s_churn, sigma_max, start)
    "production": (3, False, 1.0, 5.0, "sigma_max_eps"),
    "production_no_churn": (3, False, 0.0, 5.0, "sigma_max_eps"),
    "euler_3_true_start": (3, False, 1.0, 5.0, "true"),
    "heun_8": (8, True, 0.0, 5.0, "sigma_max_eps"),
    "heun_32": (32, True, 0.0, 5.0, "sigma_max_eps"),
    "heun_32_true_start": (32, True, 0.0, 5.0, "true"),
    "heun_64_sigma_max_19_true_start": (64, True, 0.0, 19.0, "true"),
    "churn_32_true_start": (32, True, 1.0, 5.0, "true"),
}


def token_means(n_episodes):
    """Mean real periphery token of the health and drink cells per value (health <= 0 pooled as 0), CPU encoder."""
    ck = torch.load(CKPT, weights_only=False, map_location="cpu")
    model = M.M0(ck["cfg"]); model.load_state_dict(ck["model"]); model.eval()
    L = S.first_episodes(dict(np.load(ROOT / "artifacts/eda/s0/s0_dev_labels.npz")), n_episodes)
    s_off = np.r_[0, np.cumsum(L["ep_T"] + 1)]
    toks, vals = [], []
    with torch.no_grad():
        for e, ep in enumerate(S.episodes(L)):
            fr = torch.as_tensor(D.RP.episode_fields(ep.shard, ep.slot)["observations"].numpy())
            p = model.perception(fr, torch.ones(len(fr), 2, dtype=torch.long)).p
            toks.append(p[:, [HEALTH, DRINK]].double()); vals.append(L["s_body"][s_off[e]:s_off[e + 1]][:, [0, 2]])
    t, v = torch.cat(toks), np.concatenate(vals).clip(min=0)
    return {c: {int(k): t[v[:, j] == k, j].mean(0) for k in np.unique(v[:, j])} for j, c in enumerate((HEALTH, DRINK))}


def schedule(n, smax, smin=0.002, rho=7):
    i = np.arange(n)
    s = (smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho
    return np.r_[s, 0.0]


def exact_denoiser(xa, xb, p):
    def D(x, s):
        logit = math.log(p / (1 - p)) + ((x - xa).square().sum(-1) - (x - xb).square().sum(-1)) / (2 * s * s)
        return xa + torch.sigmoid(logit)[:, None] * (xb - xa)
    return D


def sample(D, xa, xb, p, alg, gen):
    """EDM Algorithm 2 (Algorithm 1 when S_churn = 0) as in model.sample, with an optional Heun correction."""
    n, heun, churn, smax, start = alg
    sig = schedule(n, smax)
    eps = lambda: torch.randn(RUNS, len(xa), dtype=torch.float64, generator=gen)
    if start == "true":
        x = torch.where((torch.rand(RUNS, 1, dtype=torch.float64, generator=gen) < p), xb, xa) + sig[0] * eps()
    else:
        x = sig[0] * eps()
    for k in range(n):
        gamma = min(churn / n, math.sqrt(2) - 1)
        hat = sig[k] * (1 + gamma)
        if gamma > 0:
            x = x + math.sqrt(hat ** 2 - sig[k] ** 2) * eps()
        d1 = (x - D(x, hat)) / hat
        xn = x + (sig[k + 1] - hat) * d1
        if heun and sig[k + 1] > 0:
            xn = x + (sig[k + 1] - hat) * (d1 + (xn - D(xn, sig[k + 1])) / sig[k + 1]) / 2
        x = xn
    f = ((x - xa) * (xb - xa)).sum(-1) / (xb - xa).square().sum()
    return {"mean_f": float(f.mean()), "share_nearer_rare": float((f > 0.5).double().mean()),
            "max_sigma_hat": float(sig[0] * (1 + min(churn / n, math.sqrt(2) - 1)))}


def tower_check(D, xa, xb, p, gen):
    """Exact field: mean f(D(x0 + sigma eps)) over x0 ~ mixture must equal p at every sigma."""
    out = {}
    for s in (0.05, 0.377, 2.0, 6.67, 20.0):
        x0 = torch.where((torch.rand(RUNS, 1, dtype=torch.float64, generator=gen) < p), xb, xa)
        d = D(x0 + s * torch.randn(x0.shape, dtype=torch.float64, generator=gen), s)
        out[str(s)] = float((((d - xa) * (xb - xa)).sum(-1) / (xb - xa).square().sum()).mean() - p)
    return out


def main():
    means = token_means(150)
    carriers = {"health_2_to_dead": (means[HEALTH][2], means[HEALTH][0]),
                "health_5_to_3": (means[HEALTH][5], means[HEALTH][3]),
                "drink_4_to_5": (means[DRINK][4], means[DRINK][5])}
    gen = torch.Generator().manual_seed(2029)
    res = {"runs": RUNS, "algorithms": {k: list(v) for k, v in ALGORITHMS.items()}, "carriers": {}}
    for name, (a, b) in carriers.items():
        res["carriers"][name] = {"separation_norm": float((b - a).norm()), "start_norm": float(a.norm()), "by_k": {}}
        for k in FOOTPRINT:
            xa, xb = a.repeat(k), b.repeat(k)
            row = {}
            for p in PROBS:
                Dx = exact_denoiser(xa, xb, p)
                row[str(p)] = {alg: sample(Dx, xa, xb, p, ALGORITHMS[alg], gen) for alg in ALGORITHMS}
                row[str(p)]["tower_bias"] = tower_check(Dx, xa, xb, p, gen)
            res["carriers"][name]["by_k"][str(k)] = row
            print(name, k, {p: {a_: round(v[a_]["mean_f"], 4) for a_ in ("production", "heun_32", "heun_32_true_start")}
                            for p, v in row.items()}, flush=True)
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
