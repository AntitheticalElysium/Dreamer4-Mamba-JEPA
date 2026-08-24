"""Does Direct associate the right successor with the right action?

Read-only. No training, no new data: the held-out 17-way fork archive and an existing
checkpoint contain everything.

Averaged metrics cannot answer this. Cosine and NSE average over the 17 candidates, so
a model that predicts the right *set* of successors while attaching them to the wrong
actions scores well; R_delta measures transfer of a damage readout, not action
identity. The per-root matching matrix separates those cases:

    D[a, b] = || predicted centred effect for requested action a
               - true centred effect produced by action b ||^2

Scored against true equivalence classes, not against 17 distinct labels. 45.5% of
action pairs at a root produce bitwise-identical successors -- invalid or irrelevant
Craftax actions genuinely lead to the same state -- and the distribution of true
pairwise distances is cleanly bimodal, with nothing between 1e-1 and the median 1.28,
so TAU sits in an empty gap rather than on a judgement call. Demanding 17 distinct
predictions would manufacture a failure; the model should reproduce the true
equivalence classes and invent no others.

Reported:

  retrieval      is argmin_b D[a, b] inside class(a), against the chance rate implied
                 by that root's own class sizes
  rank, margin   where the correct class ranks, and how much closer it is
  hungarian      optimal one-to-one assignment -- structure recovered even if action
                 identities are permuted
  geometry       correlation between the predicted and true 17x17 distance matrices
  collapse       predicted separation on pairs the simulator makes identical, against
                 pairs it makes distinct
  per action     which actions are retrieved and which collapse or get confused
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import interval
from evaluate_phase1b_fork import predict_branches
from reevaluate_phase1b_delta import fit_probe, within_state
from train_phase1b_fork import MixerWorld, NoTanhWorld, fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World

DEVICE = "cuda"
TAU = 1e-6          # true effects are bitwise equal or far apart; nothing lands between


def classes_of(distance: np.ndarray) -> np.ndarray:
    """Label the 17 actions by which true successor they produce."""
    label = -np.ones(17, dtype=int)
    nxt = 0
    for a in range(17):
        if label[a] >= 0:
            continue
        label[(distance[a] <= TAU) & (label < 0)] = nxt
        nxt += 1
    return label


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int, default=64)
    parser.add_argument("--suffix", type=str, default="abt0")
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--no-tanh", action="store_true")
    parser.add_argument("--mixer", action="store_true")
    args = parser.parse_args()

    rows = load_forkset(HERE / f"forkset_{args.suffix}_n{args.n_latents}")
    splits = np.array([seed_split(r["seed"]) for r in rows])
    test = torch.from_numpy(splits == "test")
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=args.n_latents, d_bottleneck=16, seed=Config().seed)
    world = (MixerWorld if args.mixer else NoTanhWorld if args.no_tanh else World)(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n{args.n_latents}" /
         f"world_{args.milestone:06d}.pt", config, part0=world)
    world.eval()
    predicted = predict_branches(world, config, history[test], fork_actions(rows)[test]).float()
    truth = branch[test].float()

    hat = predicted - predicted.mean(1, keepdim=True)
    eff = truth - truth.mean(1, keepdim=True)
    cross = torch.cdist(hat, eff).pow(2).numpy()          # D[a, b]
    true_gram = torch.cdist(eff, eff).pow(2).numpy()
    pred_gram = torch.cdist(hat, hat).pow(2).numpy()

    hits, chance, ranks, margins, hungarian, geometry = [], [], [], [], [], []
    same_pred, diff_pred, classes = [], [], []
    by_action = {a: [] for a in range(17)}
    for i in range(len(cross)):
        label = classes_of(true_gram[i])
        classes.append(label.max() + 1)
        sizes = np.bincount(label)
        if len(sizes) < 2:                                # nothing to distinguish here
            continue
        order = np.argsort(cross[i], axis=1)
        for a in range(17):
            correct = label == label[a]
            hit = bool(correct[order[a, 0]])
            hits.append(hit)
            by_action[a].append(hit)
            chance.append(sizes[label[a]] / 17.0)
            ranked = np.array([label[b] == label[a] for b in order[a]])
            ranks.append(int(np.argmax(ranked)) + 1)
            best_right = cross[i, a, correct].min()
            best_wrong = cross[i, a, ~correct].min()
            margins.append(float(best_wrong - best_right))
        rowi, coli = linear_sum_assignment(cross[i])
        hungarian.append(float(np.mean(label[coli] == label[rowi])))
        upper = np.triu_indices(17, 1)
        t, p = true_gram[i][upper], pred_gram[i][upper]
        if t.std() > 0 and p.std() > 0:
            geometry.append(float(np.corrcoef(t, p)[0, 1]))
        identical = (true_gram[i] <= TAU) & ~np.eye(17, dtype=bool)
        distinct = true_gram[i] > TAU
        if identical.any():
            same_pred.append(float(pred_gram[i][identical].mean()))
        if distinct.any():
            diff_pred.append(float(pred_gram[i][distinct].mean()))

    per_root = []
    for i in range(len(cross)):
        label = classes_of(true_gram[i])
        best = cross[i].argmin(1)
        per_root.append(float(np.mean([label[best[a]] == label[a] for a in range(17)])))
    per_root = np.array(per_root)

    hits = np.array(hits, dtype=float)
    retrieval, (lo, hi) = interval(hits, 11)
    result = {
        "arm": args.suffix, "milestone": args.milestone,
        "roots_scored": len(hungarian), "roots_total": int(test.sum()),
        "mean_true_classes": float(np.mean(classes)),
        "retrieval": retrieval, "retrieval_ci": [lo, hi], "chance": float(np.mean(chance)),
        "mean_rank": float(np.mean(ranks)), "median_rank": float(np.median(ranks)),
        "mean_margin": float(np.mean(margins)),
        "hungarian": float(np.mean(hungarian)),
        "geometry_correlation": float(np.mean(geometry)),
        "pred_separation_on_identical": float(np.mean(same_pred)),
        "pred_separation_on_distinct": float(np.mean(diff_pred)),
        "per_action": {a: float(np.mean(v)) for a, v in by_action.items() if v},
    }
    print(f"\narm {args.suffix} @ {args.milestone}: {result['roots_scored']} of "
          f"{result['roots_total']} test roots have >1 distinct successor")
    print(f"  true distinct successors per root  {result['mean_true_classes']:.2f} of 17")
    print(f"  correct-action retrieval           {retrieval:.4f} [{lo:.4f}, {hi:.4f}]"
          f"   chance {result['chance']:.4f}")
    print(f"  rank of correct class              mean {result['mean_rank']:.2f}  "
          f"median {result['median_rank']:.0f}")
    print(f"  hungarian (identity-free)          {result['hungarian']:.4f}")
    print(f"  17x17 distance-matrix correlation  {result['geometry_correlation']:.4f}")
    print(f"  predicted separation: simulator-identical pairs "
          f"{result['pred_separation_on_identical']:.4f}  vs distinct pairs "
          f"{result['pred_separation_on_distinct']:.4f}")
    worst = sorted(result["per_action"].items(), key=lambda kv: kv[1])[:5]
    best = sorted(result["per_action"].items(), key=lambda kv: -kv[1])[:5]
    print("  weakest actions " + ", ".join(f"{a}:{v:.2f}" for a, v in worst))
    print("  strongest actions " + ", ".join(f"{a}:{v:.2f}" for a, v in best))

    result["per_root_retrieval"] = per_root.tolist()

    # ---- where the residual sits, relative to the damage-discriminative direction
    fit = torch.from_numpy(splits == "fit")
    train_eff = branch[fit].float() - branch[fit].float().mean(1, keepdim=True)
    train_lab = torch.stack([r["label"] for r in rows]).numpy()[fit.numpy()].reshape(-1)
    flat = train_eff.reshape(-1, train_eff.shape[-1])
    axis = flat[train_lab > 0].mean(0) - flat[train_lab <= 0].mean(0)
    axis = axis / axis.norm()
    at, ap = eff @ axis, hat @ axis
    perp_t, perp_p = eff - at[..., None] * axis, hat - ap[..., None] * axis
    labels_test = torch.stack([r["label"] for r in rows]).numpy()[test.numpy()]
    from reevaluate_phase1b_delta import within_state as _within
    result["subspace"] = {
        "damage_axis_energy_share": float((at ** 2).sum() / (eff ** 2).sum()),
        "r2_along_damage_axis": 1 - float(((ap - at) ** 2).sum() / (at ** 2).sum()),
        "r2_orthogonal": 1 - float(((perp_p - perp_t) ** 2).sum() / (perp_t ** 2).sum()),
        "linear_axis_auc_true": float(np.mean(_within(at.numpy(), labels_test))),
        "linear_axis_auc_pred": float(np.mean(_within(ap.numpy(), labels_test))),
    }
    b = result["subspace"]
    print(f"  damage axis holds {b['damage_axis_energy_share']:.2%} of effect energy; "
          f"reproduced R2 {b['r2_along_damage_axis']:.4f} along it, "
          f"{b['r2_orthogonal']:.4f} elsewhere")
    print(f"  that linear axis alone scores {b['linear_axis_auc_true']:.4f} on true effects "
          f"and {b['linear_axis_auc_pred']:.4f} on predicted -- the damage signal the "
          f"nonlinear probe uses is not this axis")

    # ---- does restoring the missing detail restore the mechanic?
    #
    # The subspace split above shows only that the linear damaging-minus-safe axis is
    # cheap to reproduce and worth little. It does NOT localise the nonlinear mechanic
    # and does not show the residual is what costs us R_delta. This does: add known
    # fractions of the true residual back and watch the frozen probe respond.
    root = history[:, -1]
    delta_true = branch.float() - root.float()[:, None]
    y_all = torch.stack([r["label"] for r in rows]).numpy()
    tune = torch.from_numpy(splits == "tune")
    width = delta_true.shape[-1]
    probe, _ = fit_probe(delta_true[fit].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(y_all[fit.numpy()].reshape(-1)).float().to(DEVICE),
                         delta_true[tune].reshape(-1, width).to(DEVICE),
                         torch.from_numpy(y_all[tune.numpy()].reshape(-1)).float().to(DEVICE),
                         seed=11)
    dt = delta_true[test]
    dp = predicted - root.float()[test][:, None]

    def read(mix):
        with torch.no_grad():
            score = torch.cat([probe(mix[lo : lo + 128].reshape(-1, width).to(DEVICE)).cpu()
                               for lo in range(0, len(mix), 128)]).numpy().reshape(-1, 17)
        return float(np.mean(within_state(score, labels_test)))

    auc_true = read(dt)
    curve = [{"alpha": a, "auc": (v := read(dp + a * (dt - dp))),
              "R_delta": (v - 0.5) / (auc_true - 0.5)}
             for a in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0)]
    result["restoration"] = {
        "auc_true": auc_true, "curve": curve,
        "true_axis_pred_complement": read(dp + ((dt @ axis) - (dp @ axis))[..., None] * axis),
        "pred_axis_true_complement": read(dt + ((dp @ axis) - (dt @ axis))[..., None] * axis),
    }
    print(f"\n  residual restoration, frozen probe on true dz (true AUC {auc_true:.4f}):")
    for row in curve:
        print(f"    alpha {row['alpha']:.2f}  AUC {row['auc']:.4f}  R_delta {row['R_delta']:.3f}")
    print(f"    true axis + predicted complement  AUC "
          f"{result['restoration']['true_axis_pred_complement']:.4f}")
    print(f"    predicted axis + true complement  AUC "
          f"{result['restoration']['pred_axis_true_complement']:.4f}")

    (HERE / f"action_matching_{args.suffix}_{args.milestone:06d}.json").write_text(
        json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
