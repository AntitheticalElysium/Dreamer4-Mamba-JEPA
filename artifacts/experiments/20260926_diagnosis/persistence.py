"""D19. Is the TRUE latent process a near-integrator (poles ~1), the regime where Lambert et al. 2022 find
one-step errors compound? And does each world propagate inherited error per direction with coefficient ~1?

Per world, in the eigenbasis of its true states (futures sample 0, all depths):
  true_rho_i   one-step autocorrelation of direction i along the factual trajectories: regression slope of
               s_{t+1,i} - mu_i on s_{t,i} - mu_i over consecutive steps (root -> k1 -> ... -> k16)
  error_rho_i  slope of (g_k - t_k)_i on (g_{k-1} - s_{k-1})_i, k = 2..16: how much of an inherited error in
               direction i the world passes on (1 = kept intact, < 1 = corrected)
Reported: variance-weighted shares with true_rho >= 0.95 / 0.9 / 0.8, median, and the error-variance-weighted
mean of error_rho; also error_rho in the top-42 vs the rest (tail) directions.
"""
import json
from pathlib import Path

import torch

HERE = Path(__file__).parent
ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
DATA = ROOT / "artifacts/eda/diagnosis_rollouts_v1"
WORLDS = ("H2", "Z", "sZ", "U", "W", "T")


def main():
    meta = torch.load(DATA / "meta.pt")
    alive = ~meta["future_dead"][:, 0].cumsum(1).bool()
    result = {}
    for w in WORLDS:
        d = torch.load(DATA / f"{w}.pt")
        s = torch.cat([d["root"][:, None], d["true"][:, 0]], 1).double()       # [R,17,D]
        g, t = d["gen"].double(), d["tf"].double()
        mu = s[:, 1:].flatten(0, 1).mean(0)
        ev, V = torch.linalg.eigh(torch.cov((s[:, 1:].flatten(0, 1) - mu).T))
        ev, V = ev.flip(0), V.flip(1)
        ok = torch.cat([torch.ones(len(s), 1, dtype=torch.bool), alive], 1)
        pairs = ok[:, :-1] & ok[:, 1:]
        a = ((s[:, :-1] - mu) @ V)[pairs]
        b = ((s[:, 1:] - mu) @ V)[pairs]
        rho = (a * b).sum(0) / (a * a).sum(0)
        wv = ev / ev.sum()
        # model error propagation
        inherited = ((g[:, :-1] - s[:, 1:-1]) @ V)[alive[:, 1:]]
        passed = ((g[:, 1:] - t[:, 1:]) @ V)[alive[:, 1:]]
        erho = (inherited * passed).sum(0) / (inherited * inherited).sum(0)
        ew = (inherited ** 2).sum(0) / (inherited ** 2).sum()
        top = torch.arange(len(ev)) < 42
        result[w] = {"true_rho_share_ge_0.95": float(wv[rho >= 0.95].sum()), "true_rho_share_ge_0.9": float(wv[rho >= 0.9].sum()),
                     "true_rho_share_ge_0.8": float(wv[rho >= 0.8].sum()),
                     "true_rho_variance_weighted_mean": float((wv * rho).sum()),
                     "error_rho_errorweighted_mean": float((ew * erho).sum()),
                     "error_rho_top42": float((ew[top] * erho[top]).sum() / ew[top].sum()),
                     "error_rho_tail": float((ew[~top] * erho[~top]).sum() / ew[~top].sum()) if (~top).any() else float("nan"),
                     "true_rho_top42": float((wv[top] * rho[top]).sum() / wv[top].sum()),
                     "true_rho_tail": float((wv[~top] * rho[~top]).sum() / wv[~top].sum()) if (~top).any() else float("nan")}
        print(w, json.dumps({k: round(v, 3) for k, v in result[w].items()}), flush=True)
    (HERE / "persistence.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
