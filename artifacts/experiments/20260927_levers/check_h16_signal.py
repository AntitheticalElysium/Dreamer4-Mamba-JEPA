"""Panel check (2026-10-03): how much of the H16 panel's across-action variation is a real first-action effect, and how much is
32-key sampling noise? (deepeval, sealed: oracle 0.711 at k = 16, prior 0.586, real16 (one draw, hindsight) 0.652, gen16 of every
world 0.593-0.602; under NO action effect the minimum of 17 Binomial(32, p) / 32 estimates already looks like a ~0.7 oracle.)
deeppanel FIT + DEV rows (replayed seeds; the sealed judge block is not read), recorded continuation, kept as deepeval keeps them
(P varies at some judged depth). Per root and depth k in (1, 4, 16), over the 17 first actions:
  P        the 32-key P(dead by k) (key sequences shared across the 17 actions)
  d0       key sequence 0's realized dead-by-k (one of the 32)
  P_rest   (32 P - d0) / 31: the other 31 keys
Given the true p_a, d0 and P_rest are independent draws, so the within-root covariance of d0 and P_rest across actions estimates
the variance of the TRUE p_a across actions ("signal"); the within-root variance of P is signal + noise.
  signal_share     sum over roots of the within-root covariance(d0, P_rest) / sum of the within-root variance of P
  noise_oracle     expected safe of argmin over 17 simulated Binomial(32, mean_a P) / 32 draws per root (no action effect),
                   200 simulations, vs the observed oracle 1 - mean min_a P
Readings, declared before running: h16_noise_dominated = signal_share < 0.5 at k = 16; oracle_is_noise = noise_oracle within
0.02 of the observed oracle at k = 16.
Usage: check_h16_signal.py   (CPU)
Result (2026-10-03; 9,489 FIT + DEV roots): the across-action variation is a real first-action effect at every depth.
  k = 1 / 4 / 16: opportunity roots 1,900 / 4,032 / 7,920; signal_share 0.999 / 0.991 / 0.985 (binomial noise 0.0003 / 0.026 /
  0.061); observed oracle 1.000 / 0.775 / 0.745 vs noise-only oracle 0.571 / 0.664 / 0.655; uniform 0.480 / 0.598 / 0.573.
  h16_noise_dominated FALSE, oracle_is_noise FALSE: the H16 panel measures a real decision; my noise conjecture is refuted.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
STORES = [ROOT / "artifacts/eda/deeppanel_fit_v1", ROOT / "artifacts/eda/deeppanel_dev_v1"]
DEPTHS = (1, 4, 16)


def load(stores=STORES):
    """Kept roots of the stores: P [R,17,3] (32-key P(dead by k)), D [R,17,3] (key 0's dead-by-k), split [R] (store index)."""
    P, D, S = [], [], []
    for i, store in enumerate(stores):
        for f in sorted(store.glob("seed-*.pt")):
            for r in torch.load(f, weights_only=False):
                p = r["p_dead_by"]["recorded"][:, [k - 1 for k in DEPTHS]].float()                  # [17, 3]
                if bool((p.amax(0) > p.amin(0)).any()):
                    P.append(p); D.append(r["depth_dead"][:, [k - 1 for k in DEPTHS]].float()); S.append(i)
    return torch.stack(P), torch.stack(D), torch.tensor(S)


def main():
    P, D, _ = load()
    rest = (32 * P - D) / 31
    out = {"roots": len(P)}
    g = torch.Generator().manual_seed(20261003)
    for j, k in enumerate(DEPTHS):
        p, d, q = P[:, :, j], D[:, :, j], rest[:, :, j]
        opp = p.amax(1) > p.amin(1)
        p, d, q = p[opp], d[opp], q[opp]
        cov = ((d - d.mean(1, keepdim=True)) * (q - q.mean(1, keepdim=True))).sum(1) / 16
        var = ((p - p.mean(1, keepdim=True)) ** 2).sum(1) / 16
        m = p.mean(1, keepdim=True).expand_as(p)
        sims = torch.stack([torch.binomial(torch.full_like(m, 32.0), m, generator=g).amin(1) / 32 for _ in range(200)])
        prior = None
        out[f"k{k}"] = {"opportunity_roots": int(opp.sum()), "signal_share": float(cov.sum() / var.sum()),
                        "noise_share_binomial": float((p * (1 - p) / 31).mean(1).sum() / var.sum()),
                        "observed_oracle": float(1 - p.amin(1).mean()), "noise_oracle": float(1 - sims.mean()),
                        "uniform": float(1 - p.mean())}
    out["readings"] = {"h16_noise_dominated": out["k16"]["signal_share"] < 0.5,
                       "oracle_is_noise": abs(out["k16"]["noise_oracle"] - out["k16"]["observed_oracle"]) <= 0.02}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
