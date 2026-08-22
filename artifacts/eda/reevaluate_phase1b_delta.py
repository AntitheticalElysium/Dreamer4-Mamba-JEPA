"""Task-matched re-evaluation of the existing Phase-1B checkpoints. No training.

The first evaluator read damage off z_{t+1} with one global logit. That is not the
object our own probe work showed consequence lives in -- it lives in the action
induced change dz = z_{t+1} - z_t, and nonlinearly. It also compared two arms whose
true-successor reference AUCs differ, so the raw cross-arm difference was not a clean
translation measure.

Statistics fixed before any result was inspected:

  primary    R_delta = (AUC_pred - 0.5) / (AUC_true - 0.5), per arm
  secondary  damaging-minus-safe probe-logit contrast on true and predicted dz,
             and its recovery ratio
  label-free centred all-17 action-effect fidelity: e_a = z_{t+1,a} - mean_b z_{t+1,b}
             against its predicted counterpart, by cosine and normalised squared error
  sensitivity  64 -> 512 fixed random projections

The probe is fitted on true dz from fit roots, selected on tune, frozen, then read on
both true and predicted dz over held-out test roots.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path("/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from evaluate_damage_classifier import auc, interval
from evaluate_phase1b_fork import Readout, predict_branches
from train_phase1b_fork import fork_actions, load_forkset, seed_split

from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.transition import World

DEVICE = "cuda"


def fit_probe(x_fit, y_fit, x_tune, y_tune, seed, epochs=80):
    """Nonlinear probe on true dz; epoch chosen on tune, never on test."""
    torch.manual_seed(seed)
    model = Readout(x_fit.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    weight = torch.tensor(float((y_fit <= 0).sum() / y_fit.sum().clamp(min=1)), device=DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    index = np.arange(len(x_fit))
    best = {"auc": -1.0, "state": None}
    for epoch in range(epochs):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(index)
        for lo in range(0, len(order), 512):
            batch = torch.from_numpy(order[lo : lo + 512]).to(DEVICE)
            opt.zero_grad()
            criterion(model(x_fit[batch]), y_fit[batch]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            value = auc(model(x_tune).cpu().numpy(), y_tune.cpu().numpy())
        if value > best["auc"]:
            best = {"auc": value, "state": {k: v.clone() for k, v in model.state_dict().items()}}
    model.load_state_dict(best["state"])
    return model.eval(), best["auc"]


def within_state(scores, labels):
    values = np.array([auc(scores[i], labels[i]) for i in range(len(labels))])
    return values[~np.isnan(values)]


def contrast(scores, labels):
    out = []
    for i in range(len(labels)):
        y, s = labels[i], scores[i]
        if not (y > 0).any() or not (y <= 0).any():
            continue
        scale = s.std() if s.std() > 0 else 1.0
        out.append((s[y > 0].mean() - s[y <= 0].mean()) / scale)
    return np.array(out)


BOOT = 10_000


def combine(args) -> None:
    """Primary endpoint: R_delta(64) - R_delta(32), bootstrapped over shared roots.

    R_delta is a ratio of means, so it is resampled rather than averaged: each draw
    takes the same roots in both arms and recomputes both ratios on them.
    """
    arms = {}
    for n in (32, 64):
        arms[n] = json.loads(
            (HERE / f"phase1b_delta_{args.suffix}_n{n}_{args.milestone:06d}.json").read_text())
    assert arms[32]["root_keys"] == arms[64]["root_keys"], "arms scored different roots"
    n_roots = len(arms[32]["root_keys"])

    pred = {n: np.array(arms[n]["none"]["pred_values"]) for n in (32, 64)}
    true = {n: np.array(arms[n]["none"]["true_values"]) for n in (32, 64)}
    nse = {n: np.array(arms[n]["action_effect"]["per_root_nse"]) for n in (32, 64)}
    cos = {n: np.array(arms[n]["action_effect"]["per_root_cosine"]) for n in (32, 64)}
    for n in (32, 64):
        assert len(pred[n]) == n_roots and len(nse[n]) == n_roots

    ratio = lambda p, t: (p.mean() - 0.5) / (t.mean() - 0.5)
    point = {n: ratio(pred[n], true[n]) for n in (32, 64)}
    generator = np.random.default_rng(20260821)
    draws = np.array([
        [ratio(pred[n][idx], true[n][idx]) for n in (32, 64)]
        + [nse[64][idx].mean() - nse[32][idx].mean(),
           cos[64][idx].mean() - cos[32][idx].mean()]
        for idx in (generator.integers(0, n_roots, n_roots) for _ in range(BOOT))])
    delta = draws[:, 1] - draws[:, 0]

    def band(values):
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    out = {
        "suffix": args.suffix, "milestone": args.milestone, "roots": n_roots,
        "R_delta_32": point[32], "R_delta_64": point[64],
        "R_delta_difference": point[64] - point[32],
        "R_delta_difference_ci": band(delta),
        "R_delta_difference_p_le_0": float((delta <= 0).mean()),
        "nse_32": float(nse[32].mean()), "nse_64": float(nse[64].mean()),
        "nse_difference": float(nse[64].mean() - nse[32].mean()),
        "nse_difference_ci": band(draws[:, 2]),
        "cosine_difference": float(cos[64].mean() - cos[32].mean()),
        "cosine_difference_ci": band(draws[:, 3]),
    }
    print(f"\n{args.suffix} @ {args.milestone}, {n_roots} shared test roots")
    print(f"  R_delta   32x16 {out['R_delta_32']:.3f}   64x16 {out['R_delta_64']:.3f}   "
          f"64-32 {out['R_delta_difference']:+.3f} "
          f"[{out['R_delta_difference_ci'][0]:+.3f}, {out['R_delta_difference_ci'][1]:+.3f}]"
          f"  P(<=0) {out['R_delta_difference_p_le_0']:.4f}")
    print(f"  NSE       32x16 {out['nse_32']:.4f}  64x16 {out['nse_64']:.4f}  "
          f"64-32 {out['nse_difference']:+.4f} "
          f"[{out['nse_difference_ci'][0]:+.4f}, {out['nse_difference_ci'][1]:+.4f}]"
          f"   (lower is better)")
    print(f"  cosine    64-32 {out['cosine_difference']:+.4f} "
          f"[{out['cosine_difference_ci'][0]:+.4f}, {out['cosine_difference_ci'][1]:+.4f}]")
    (HERE / f"phase1b_paired_{args.suffix}_{args.milestone:06d}.json").write_text(
        json.dumps(out, indent=2))


def trajectory(args) -> None:
    """Is R_delta still climbing, or has Direct plateaued?

    Registered before any milestone was inspected: an arm is *still climbing* iff the
    paired 10k->20k increment in R_delta has a bootstrap CI excluding zero, and
    *plateaued* iff that CI includes zero. Direct64 is extended only if both tokenizer
    seeds are still climbing.
    """
    miles = tuple(args.milestones)
    arms = {}
    for m in miles:
        arms[m] = json.loads(
            (HERE / f"phase1b_delta_{args.suffix}_n{args.n_latents}_{m:06d}.json").read_text())
    keys = arms[miles[0]]["root_keys"]
    for m in miles[1:]:
        assert arms[m]["root_keys"] == keys, f"milestone {m} scored different roots"
    n_roots = len(keys)

    pred = {m: np.array(arms[m]["none"]["pred_values"]) for m in miles}
    true = {m: np.array(arms[m]["none"]["true_values"]) for m in miles}
    ratio = lambda p, t: (p.mean() - 0.5) / (t.mean() - 0.5)
    point = {m: ratio(pred[m], true[m]) for m in miles}

    generator = np.random.default_rng(20260822)
    draws = np.array([[ratio(pred[m][idx], true[m][idx]) for m in miles]
                      for idx in (generator.integers(0, n_roots, n_roots) for _ in range(BOOT))])
    band = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
    steps = {}
    for i in range(1, len(miles)):
        delta = draws[:, i] - draws[:, i - 1]
        steps[f"{miles[i-1]//1000}k_{miles[i]//1000}k"] = {
            "increment": point[miles[i]] - point[miles[i - 1]], "ci": band(delta)}
    final = list(steps)[-1]
    climbing = steps[final]["ci"][0] > 0

    out = {"suffix": args.suffix, "n_latents": args.n_latents, "roots": n_roots,
           "R_delta": {str(m): point[m] for m in miles}, "increments": steps,
           "final_increment": final, "still_climbing": bool(climbing)}
    print(f"\n{args.suffix} {args.n_latents}x16 trajectory, {n_roots} roots")
    print("  R_delta  " + "  ".join(f"{m//1000}k {point[m]:.3f}" for m in miles))
    for name, row in steps.items():
        mark = ""
        if name == final:
            mark = f"   -> {'STILL CLIMBING' if climbing else 'PLATEAUED'}"
        print(f"  {name.replace('_', '->'):<10} {row['increment']:+.3f} "
              f"[{row['ci'][0]:+.3f}, {row['ci'][1]:+.3f}]{mark}")
    (HERE / f"phase1b_trajectory_{args.suffix}_n{args.n_latents}.json").write_text(
        json.dumps(out, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-latents", type=int)
    parser.add_argument("--suffix", type=str, default="s1")
    parser.add_argument("--milestone", type=int, default=20000)
    parser.add_argument("--combine", action="store_true",
                        help="no model: read both arms' results and bootstrap 64 minus 32")
    parser.add_argument("--milestones", type=int, nargs="+",
                        default=(5_000, 10_000, 20_000),
                        help="trajectory only: the checkpoints to compare, in order")
    parser.add_argument("--trajectory", action="store_true",
                        help="no model: bootstrap one arm's R_delta increments across milestones")
    args = parser.parse_args()

    if args.combine:
        combine(args)
        return
    if args.trajectory:
        trajectory(args)
        return

    folder = HERE / f"forkset_{args.suffix}_n{args.n_latents}"
    rows = load_forkset(folder)
    splits = np.array([seed_split(r["seed"]) for r in rows])
    history = torch.stack([r["z_history"] for r in rows])
    branch = torch.stack([r["z_branch"] for r in rows])
    labels = torch.stack([r["label"] for r in rows]).numpy()
    root = history[:, -1]                                   # z_t

    config = replace(Config(transition="direct", time_mixer="attention"),
                     n_latents=args.n_latents, d_bottleneck=16, seed=Config().seed)
    world = World(config).to(DEVICE)
    load(HERE / f"phase1b_{args.suffix}_n{args.n_latents}" /
         f"world_{args.milestone:06d}.pt", config, part0=world)
    world.eval()

    fit, tune, test = (splits == s for s in ("fit", "tune", "test"))
    predicted_test = predict_branches(world, config, history[test],
                                      fork_actions(rows)[torch.from_numpy(test)])

    true_delta = branch - root[:, None]
    pred_delta_test = predicted_test - root[test][:, None]

    y_all = labels[test]
    kept = [i for i in range(len(y_all))
            if (y_all[i] > 0).any() and (y_all[i] <= 0).any()]
    keys = [[int(rows[i]["seed"]), int(rows[i]["step"])] for i in np.where(test)[0]]
    result = {"arm": f"{args.n_latents}x16", "roots_test": int(test.sum()),
              "root_keys": [keys[i] for i in kept]}
    projections = {"none": None}
    if args.n_latents == 64:
        for seed in (20260820, 7, 101):
            generator = torch.Generator().manual_seed(seed)
            projections[f"rp512-s{seed}"] = (
                torch.randn(true_delta.shape[2], 512, generator=generator)
                / true_delta.shape[2] ** 0.5)

    for name, projection in projections.items():
        td = true_delta if projection is None else true_delta @ projection
        pd = pred_delta_test if projection is None else pred_delta_test @ projection
        x_fit = td[fit].reshape(-1, td.shape[2]).to(DEVICE)
        y_fit = torch.from_numpy(labels[fit].reshape(-1)).float().to(DEVICE)
        x_tune = td[tune].reshape(-1, td.shape[2]).to(DEVICE)
        y_tune = torch.from_numpy(labels[tune].reshape(-1)).float().to(DEVICE)
        probe, tune_auc = fit_probe(x_fit, y_fit, x_tune, y_tune, seed=11)
        with torch.no_grad():
            s_true = probe(td[test].reshape(-1, td.shape[2]).to(DEVICE)).cpu().numpy()
            s_pred = probe(pd.reshape(-1, pd.shape[2]).to(DEVICE)).cpu().numpy()
        s_true = s_true.reshape(-1, 17)
        s_pred = s_pred.reshape(-1, 17)
        y_test = labels[test]
        v_true, v_pred = within_state(s_true, y_test), within_state(s_pred, y_test)
        a_true, _ = interval(v_true, 17)
        a_pred, (lo, hi) = interval(v_pred, 17)
        c_true, c_pred = contrast(s_true, y_test), contrast(s_pred, y_test)
        m_true, m_pred = float(c_true.mean()), float(c_pred.mean())
        result[name] = {
            "auc_true_delta": a_true, "auc_pred_delta": a_pred, "auc_pred_ci": [lo, hi],
            "R_delta": (a_pred - 0.5) / (a_true - 0.5) if a_true > 0.5 else float("nan"),
            "contrast_true": m_true, "contrast_pred": m_pred,
            "contrast_recovery": m_pred / m_true if abs(m_true) > 1e-9 else float("nan"),
            "tune_auc": tune_auc,
            "pred_values": v_pred.tolist(), "true_values": v_true.tolist(),
        }
        print(f"  {args.n_latents}x16 {name:<14} true dz {a_true:.4f}  pred dz {a_pred:.4f} "
              f"[{lo:.4f}, {hi:.4f}]  R_delta {result[name]['R_delta']:.3f}  "
              f"contrast {m_pred:+.3f}/{m_true:+.3f} = {result[name]['contrast_recovery']:.3f}",
              flush=True)

    # label-free: centred action effect, no damage classifier involved
    true_centred = branch[test] - branch[test].mean(1, keepdim=True)
    pred_centred = predicted_test - predicted_test.mean(1, keepdim=True)
    cosine = torch.nn.functional.cosine_similarity(
        pred_centred.reshape(-1, pred_centred.shape[2]),
        true_centred.reshape(-1, true_centred.shape[2]), dim=1)
    nse = ((pred_centred - true_centred).pow(2).sum(-1)
           / true_centred.pow(2).sum(-1).clamp(min=1e-12))
    per_root_nse = nse.mean(1)
    per_root_cos = cosine.view(-1, 17).mean(1)
    result["action_effect"] = {
        "cosine_mean": float(cosine.mean()), "cosine_median": float(cosine.median()),
        "normalised_squared_error": float(nse.mean()),
        "true_effect_norm": float(true_centred.pow(2).sum(-1).sqrt().mean()),
        "pred_effect_norm": float(pred_centred.pow(2).sum(-1).sqrt().mean()),
        "per_root_nse": per_root_nse.tolist(),
        "per_root_cosine": per_root_cos.tolist(),
    }
    e = result["action_effect"]
    print(f"  {args.n_latents}x16 action-effect  cosine {e['cosine_mean']:.4f} "
          f"(median {e['cosine_median']:.4f})  NSE {e['normalised_squared_error']:.4f}  "
          f"|e| true {e['true_effect_norm']:.4f} pred {e['pred_effect_norm']:.4f}", flush=True)

    (HERE / f"phase1b_delta_{args.suffix}_n{args.n_latents}_{args.milestone:06d}.json"
     ).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
