"""POST-HOC (11-05): is the 3-step sampler's loss of minority outcomes a property of the sampler itself? An EXACT
two-outcome denoiser (clean in {0, s}, P(s) = p; D = s * posterior) run through saccade.model.sample's schedule
(EDM Euler, Karras rho 7, sigma 5 -> 0.002, churn gamma = min(S_churn / N, sqrt 2 - 1) every step, evaluated at
sigma_hat, last step to 0), started from the CORRECT marginal at sigma_max (isolates the step count from the start
distribution). Output: P(final > s / 2) per p and step count; with many steps it must return p.

Usage: python sampler_rounding.py  (CPU, seconds)"""
import json, math
from pathlib import Path
import numpy as np
from scipy.special import expit

RNG = np.random.default_rng(0)


def run(p, s, n, churn=1.0, smax=5.0, smin=0.002, rho=7, N=400000):
    i = np.arange(n)
    sig = list((smax ** (1 / rho) + i / (n - 1) * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho) + [0.0]
    x = s * (RNG.random(N) < p) + sig[0] * RNG.standard_normal(N)
    for k in range(n):
        hat = sig[k] * (1 + min(churn / n, math.sqrt(2) - 1))
        x = x + math.sqrt(hat ** 2 - sig[k] ** 2) * RNG.standard_normal(N)
        d = s * expit(math.log(p / (1 - p)) + (2 * x * s - s * s) / (2 * hat * hat))
        x = x + (sig[k + 1] - hat) * (x - d) / hat
    return float(np.mean(x > s / 2))


def main():
    out = {}
    for s in (3.15, 4.96):                                               # drink / coal HUD separations (F2)
        out[s] = {p: {n: run(p, s, n) for n in (3, 8, 32, 128)} | {"3_no_churn": run(p, s, 3, churn=0)}
                  for p in (0.05, 0.15, 0.3, 0.45, 0.7, 0.9)}
        for p, r in out[s].items():
            print(f"sep {s} p {p:.2f}: " + " | ".join(f"{k}: {v:.3f}" for k, v in r.items()))
    Path(__file__).with_name("readings").joinpath("sampler_rounding.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
