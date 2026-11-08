"""Where each recipe spends its denoising objective across noise levels (RESEARCH.md, "Noise"; CPU, no model).

Reads F2's saved P1 monitor (G1 at 8 fixed noise levels, `artifacts/experiments/20261103_p1f2/run/log.jsonl`) and
turns it into F2's x-space error profile e(sigma) = G1(sigma) / lambda(sigma), lambda = 1 + 1 / sigma^2 (EDM / DIAMOND
raw-output loss, sigma_data = 1). That profile is then weighted by each recipe's training distribution and loss weight:
- ours (m0.yaml): ln sigma ~ N(-0.4, 1.2), clipped to [0.002, 20]; weight lambda.
- DIAMOND as published (config/trainer.yaml + agent/default.yaml): the same lognormal in ABSOLUTE units with
  sigma_data = 0.5, i.e. sigma / sigma_data = 2 sigma; weight lambda (its per-channel offset noise 0.3 is ignored).
- Dreamer 4 (arXiv 2509.24527 eq. 4, 6, 8; nicklashansen/dreamer4 train_dynamics.py): signal level tau uniform on
  [0, 1), x~ = (1 - tau) z0 + tau z1, so sigma / scale = (1 - tau) / tau (latent scale assumed 1, see RESEARCH.md);
  weight 0.9 tau + 0.1 on the x-space error. Also the reimplementations' default grid k_max = 8 (tau in {0, 1/8, ...,
  7/8}: pure noise is 1/8 of the draws).
It is F2's error profile throughout: the shares say what OUR model's objective would be made of under each recipe, not
what Dreamer 4's model optimises. Monitor context frames sit at the inference level 0.1 (training contexts are noisier).
Usage: python artifacts/experiments/20261105_consequence_trace/noise_profile.py   (writes noise_profile.json beside it)
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

HERE = Path(__file__).resolve().parent
LOG = HERE.parents[1] / "experiments/20261103_p1f2/run/log.jsonl"
BANDS = [(0, 0.3), (0.3, 1), (1, 3), (3, 7), (7, np.inf)]
FIRST_CALL = {"ours": 5 * (1 + 1 / 3), "diamond": 5 / 0.5, "d4": np.inf}     # sigma / sigma_data of the first evaluation


def profile(ckpt=10, log=LOG):
    rows = [json.loads(l) for l in Path(log).read_text().splitlines()]
    m = next(r["monitor"] for r in rows if r.get("checkpoint") == ckpt and "monitor" in r)
    s, g = np.array(m["levels"]), np.array(m["g1_by_level"])
    return s, g / (1 + 1 / s ** 2)


def error(sig, s, e, high):
    """Log-log interpolation; below the lowest level the first segment's slope; above the highest either flat
    ('flat') or the last segment's slope up to sigma 20, flat beyond ('rising'). Bracketing, not a fit."""
    ls, le = np.log(s), np.log(e)
    x = np.log(np.minimum(sig, 1e12))
    out = np.interp(x, ls, le)
    lo = (le[1] - le[0]) / (ls[1] - ls[0])
    out = np.where(x < ls[0], le[0] + lo * (x - ls[0]), out)
    if high == "rising":
        hi = (le[-1] - le[-2]) / (ls[-1] - ls[-2])
        out = np.where(x > ls[-1], le[-1] + hi * (np.minimum(x, np.log(20.0)) - ls[-1]), out)
    else:
        out = np.where(x > ls[-1], le[-1], out)
    return np.exp(out)


def lognormal_draws(loc, scale, lo, hi, unit, n=200_000):
    """Quantile grid of the clipped lognormal (absolute sigma), returned in sigma / sigma_data units, equal mass."""
    return np.clip(np.exp(loc + scale * ndtri((np.arange(n) + 0.5) / n)), lo, hi) / unit


def recipes(n=200_000):
    tau = (np.arange(n) + 0.5) / n
    k8 = np.repeat(np.arange(8) / 8, n // 8)
    d4 = lambda t: (np.where(t > 0, (1 - t) / np.maximum(t, 1e-300), np.inf), 0.9 * t + 0.1)
    ours = lognormal_draws(-0.4, 1.2, 0.002, 20.0, 1.0, n)
    dia = lognormal_draws(-0.4, 1.2, 0.002, 20.0, 0.5, n)
    lam = lambda s: 1 + 1 / s ** 2
    return {"ours": (ours, lam(ours)), "diamond": (dia, lam(dia)), "d4": d4(tau), "d4_kmax8": d4(k8)}


def above(threshold, s, e):
    """Per recipe: the share of training draws, and of the objective (both extrapolations), with sigma >= threshold."""
    out = {}
    for name, (sig, w) in recipes().items():
        m = sig >= threshold
        out[name] = {"draws": float(m.mean()), **{f"objective_{h}": float((c := w * error(sig, s, e, h))[m].sum() / c.sum())
                                                  for h in ("flat", "rising")}}
    return out


def main():
    s, e = profile()
    res = {"f2_profile": {"sigma": s.tolist(), "x_space_error": e.tolist()}, "recipes": {}}
    for name, (sig, w) in recipes().items():
        r = {"draws_above_first_call": float((sig >= FIRST_CALL.get(name.split("_")[0], np.inf)).mean()),
             "draws_sigma_above_5": float((sig > 5).mean()), "draws_sigma_below_0.2": float((sig < 0.2).mean()),
             "draws_by_band": {f"{a}-{b}": float(((sig >= a) & (sig < b)).mean()) for a, b in BANDS}}
        for high in ("flat", "rising"):
            c = w * error(sig, s, e, high)
            r[f"objective_by_band_{high}"] = {f"{a}-{b}": float(c[(sig >= a) & (sig < b)].sum() / c.sum()) for a, b in BANDS}
        res["recipes"][name] = r
    (HERE / "noise_profile.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
