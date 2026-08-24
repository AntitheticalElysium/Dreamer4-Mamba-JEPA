"""Did coverage, at fixed data volume, reduce the error?

Both arms trained the promoted mixer Direct on 1,825 fit roots with the same
initialisation, updates, lambda, streams and evaluator. They differ only in which roots:
uniform against k-center spread in pooled space, selected from fit roots only.

The primary reading is action-effect MSE, and specifically its value on the held-out
roots that were originally farthest from fit coverage -- the stratum the coverage
hypothesis actually predicts. That stratification is fixed once, from distance to the
FULL 3,651-root fit set under the abm0 checkpoint, so both arms are scored on the same
partition rather than on their own subsets.

  B wins on the far quartile  coverage is an actionable lever, and expanding or
                              balancing paired data has justification
  B does not win              the adjusted distance association was describing hard
                              states rather than identifying a repair, and collecting
                              more data is not warranted on this evidence

Uncertainty is a root-clustered bootstrap on the paired difference, since the arms are
scored on identical roots.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_phase1b_fork import predict_branches
from legacy import open_checkpoint
from probe_action_matching import classes_of
from probe_decoder_ceiling import cache as cache_backbone
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import fork_actions, load_forkset, seed_split

from d4mj.config import Config

DEVICE = "cuda"
ARMS = {"random": "cvr0", "spread": "cvs0"}


def measures(pred, truth):
    hat = pred - pred.mean(1, keepdim=True)
    eff = truth - truth.mean(1, keepdim=True)
    per_root_action = (hat - eff).pow(2).mean(-1)                     # action-effect MSE
    nse = ((hat - eff).pow(2).sum(-1) / eff.pow(2).sum(-1).clamp(min=1e-12))
    cos = torch.nn.functional.cosine_similarity(
        hat.reshape(-1, hat.shape[-1]), eff.reshape(-1, eff.shape[-1]), dim=1).view(-1, 17)
    gram = torch.cdist(eff, eff).pow(2).numpy()
    pgram = torch.cdist(hat, hat).pow(2).numpy()
    cross = torch.cdist(hat, eff).pow(2).numpy()
    hits, geo = [], []
    for i in range(len(cross)):
        label = classes_of(gram[i])
        if label.max() < 1:
            continue
        best = cross[i].argmin(1)
        hits.append(np.mean([label[best[a]] == label[a] for a in range(17)]))
        u = np.triu_indices(17, 1)
        if gram[i][u].std() > 0 and pgram[i][u].std() > 0:
            geo.append(np.corrcoef(gram[i][u], pgram[i][u])[0, 1])
    return {"action_mse": per_root_action, "nse": nse, "cosine": cos,
            "retrieval": float(np.mean(hits)), "geometry": float(np.mean(geo))}


def main() -> None:
    rows = load_forkset(HERE / "forkset_abm0_n64")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows]).float()
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1].float()
    masks = tuple(torch.from_numpy(splits == s) for s in ("fit", "tune", "test"))
    test = masks[2]
    led = fork_actions(rows)

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=64, d_bottleneck=16, seed=Config().seed)

    # fixed stratification: distance to the FULL fit set, under the reference checkpoint
    reference = open_checkpoint(HERE / "phase1b_abm0_n64" / "world_020000.pt",
                                config, "promoted")
    pooled = cache_backbone(reference, config, history, led, "pooled").flatten(1).float()
    fit_idx, test_idx = np.where(splits == "fit")[0], np.where(splits == "test")[0]
    f = torch.nn.functional.normalize(pooled[fit_idx], dim=1)
    t = torch.nn.functional.normalize(pooled[test_idx], dim=1)
    near = torch.cat([(1 - t[lo:lo+256] @ f.T).min(1).values for lo in range(0, len(t), 256)])
    order = near.argsort()
    quarter = len(order) // 4
    quartile = torch.empty(len(order), dtype=torch.long)
    for q in range(4):
        quartile[order[q * quarter:(q + 1) * quarter if q < 3 else len(order)]] = q
    far = quartile == 3
    del reference

    truth = branch[test]
    out = {"roots_test": int(test.sum()), "far_quartile_roots": int(far.sum()), "arms": {}}
    stats = {}
    for name, suffix in ARMS.items():
        world = open_checkpoint(HERE / f"phase1b_{suffix}_n64" / "world_020000.pt",
                                config, "promoted")
        pred = predict_branches(world, config, history[test], led[test]).float()
        stats[name] = measures(pred, truth)
        stats[name]["pred"] = pred
        m = stats[name]
        out["arms"][name] = {
            "action_mse": float(m["action_mse"].mean()),
            "action_mse_far": float(m["action_mse"][far].mean()),
            "action_mse_near": float(m["action_mse"][quartile == 0].mean()),
            "nse": float(m["nse"].mean()), "cosine": float(m["cosine"].mean()),
            "retrieval": m["retrieval"], "geometry": m["geometry"],
        }
        a = out["arms"][name]
        print(f"{name:<8} action MSE {a['action_mse']:.5f}   far quartile "
              f"{a['action_mse_far']:.5f}   near {a['action_mse_near']:.5f}   "
              f"NSE {a['nse']:.4f}  cosine {a['cosine']:.4f}  geometry {a['geometry']:.4f}",
              flush=True)
        del world

    # frozen damage probe, for R_delta as a secondary reading
    width = branch.shape[-1]
    delta_true = branch - root[:, None]
    probe, _ = fit_probe(delta_true[masks[0]].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "fit"].reshape(-1)).float().to(DEVICE),
                         delta_true[masks[1]].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(labels[splits == "tune"].reshape(-1)).float().to(DEVICE),
                         seed=11)

    def read(mix):
        with torch.no_grad():
            s = torch.cat([probe(mix[lo:lo+128].reshape(-1, width).to(DEVICE)).cpu()
                           for lo in range(0, len(mix), 128)]).numpy().reshape(-1, 17)
        return float(np.mean(within_state(s, labels[test.numpy()])))

    auc_true = read(delta_true[test])
    for name in ARMS:
        auc = read(stats[name]["pred"] - root[test][:, None])
        out["arms"][name]["R_delta"] = (auc - 0.5) / (auc_true - 0.5)
        print(f"{name:<8} R_delta {out['arms'][name]['R_delta']:.3f}", flush=True)

    # paired, root-clustered
    generator = np.random.default_rng(20260825)
    n = int(test.sum())
    d_all = (stats["spread"]["action_mse"] - stats["random"]["action_mse"]).numpy()
    d_far = d_all[far.numpy()]
    d_nse = (stats["spread"]["nse"] - stats["random"]["nse"]).numpy()
    draws = np.array([[d_all[i].mean(), d_far[j].mean(), d_nse[i].mean()]
                      for i, j in ((generator.integers(0, n, n),
                                    generator.integers(0, len(d_far), len(d_far)))
                                   for _ in range(10000))])
    band = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    out["paired_spread_minus_random"] = {
        "action_mse": float(d_all.mean()), "action_mse_ci": band(draws[:, 0]),
        "action_mse_far": float(d_far.mean()), "action_mse_far_ci": band(draws[:, 1]),
        "nse": float(d_nse.mean()), "nse_ci": band(draws[:, 2]),
    }
    p = out["paired_spread_minus_random"]
    print(f"\nspread minus random, paired over {n} roots (negative favours spread)")
    print(f"  action MSE, all        {p['action_mse']:+.6f} "
          f"[{p['action_mse_ci'][0]:+.6f}, {p['action_mse_ci'][1]:+.6f}]")
    print(f"  action MSE, far quart  {p['action_mse_far']:+.6f} "
          f"[{p['action_mse_far_ci'][0]:+.6f}, {p['action_mse_far_ci'][1]:+.6f}]")
    print(f"  NSE                    {p['nse']:+.6f} "
          f"[{p['nse_ci'][0]:+.6f}, {p['nse_ci'][1]:+.6f}]")

    (HERE / "coverage_ab_result.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
