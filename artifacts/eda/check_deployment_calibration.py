"""Deployment calibration: the RAW head on unfiltered BC-policy states.

The old terminal gate scored BCE only where death varies across actions, which excludes
every all-safe state and lifts prevalence from ~0.12 to 0.64-0.92, and it fitted its action
marginal on the test labels themselves. A model calibrated on the states a policy visits
must lose that. This asks the deployment question instead: does the raw head beat a frozen
action marginal, fitted on separate roots, over ALL sampled states?

The affine calibrator is reported beside it as a recorded negative -- it improved aggregate
BCE only by wrecking the safe regimes, and is not in the production path.

Primary is per-stratum, aggregate co-reported: this project has already been misled once by
an aggregate that was the mean of two opposite-signed strata (af62689).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from d4mj.agent import Heads
from d4mj.checkpoint import load
from d4mj.config import Config
from d4mj.counterfactual import collect_outcome_forks
from d4mj.representation import Decoder, Encoder
from d4mj.transition import World

DEVICE = "cuda"
ENCODER = HERE / "capacity6k" / "n64d16_s1" / "encoder_006000.pt"
REPORT = HERE / "capacity6k" / "n64d16_s1" / "training_report.json"
EPS = 1e-6
BLOCKS = {"calibrate": range(12_200, 12_232), "tune": range(12_300, 12_316),
          "test": range(12_400, 12_416)}


def bce_rows(p, y):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean(1)


def within_auc(score, truth):
    out = []
    for s, y in zip(score, truth):
        pos, neg = s[y > 0], s[y == 0]
        if len(pos) and len(neg):
            out.append(float((pos[:, None] > neg[None]).mean()
                             + 0.5 * (pos[:, None] == neg[None]).mean()))
    return float(np.mean(out)) if out else float("nan")


def clustered(diff, seeds, draws=4000, seed=20260826):
    """Paired interval over the test seeds, not the states."""
    g = np.random.default_rng(seed)
    uniq = np.unique(seeds)
    index = {s: np.where(seeds == s)[0] for s in uniq}
    boot = [diff[np.concatenate([index[s] for s in g.choice(uniq, len(uniq))])].mean()
            for _ in range(draws)]
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("attention", "mamba"))
    args = parser.parse_args()
    folder = HERE / f"v2_phase2_{args.arm}"

    base = replace(Config(), n_latents=64, d_bottleneck=16)
    saved = replace(base, transition="direct", time_mixer=args.arm)
    world, heads = World(saved).to(DEVICE), Heads(saved).to(DEVICE)
    load(folder / "phase2_final.pt", saved, part0=world, part1=heads)
    stored = json.loads(REPORT.read_text())
    encoder = Encoder(base).to(DEVICE)
    load(ENCODER, replace(base, batch=stored["batch"], seed=stored["seed"]),
         part0=encoder, part1=Decoder(base))
    world, heads, encoder = world.eval(), heads.eval(), encoder.eval()
    gate_config = replace(saved, horizon=saved.direct_rollout)

    data = {}
    for name, seeds in BLOCKS.items():
        seeds = tuple(seeds)
        assert not set(seeds) & set(gate_config.outcome_gate_seeds), "overlaps the live gate"
        f = collect_outcome_forks(world, encoder, heads,
                                  replace(gate_config, outcome_gate_seeds=seeds))
        data[name] = {"p": f.model_death.numpy(), "y": f.true_death.float().numpy(),
                      "seed": f.seed.numpy(), "step": f.step.numpy()}
        d = data[name]
        print(f"{args.arm} {name}: {len(d['y'])} states, {d['y'].mean():.3f} lethal, "
              f"{(d['y'].sum(1) == 0).sum()} all-safe", flush=True)

    # the baseline is frozen on the calibration roots, never on the labels it is scored against
    marginal = data["calibrate"]["y"].mean(0)
    t = data["test"]
    lethal = t["y"].sum(1)
    floor = np.tile(marginal, (len(t["y"]), 1))
    raw_rows, floor_rows = bce_rows(t["p"], t["y"]), bce_rows(floor, t["y"])

    print(f"\n{args.arm}: raw head against a frozen action marginal, "
          f"clustered over {len(np.unique(t['seed']))} test seeds")
    print(f"{'':<18}{'n':>5}{'BCE raw':>9}{'BCE floor':>11}"
          f"{'raw - floor (95%)':>28}{'pred':>7}{'true':>7}{'AUC':>7}")
    report = {"arm": args.arm, "strata": {}}
    for name, mask in (("all test", np.ones(len(lethal), bool)),
                       ("  all-safe", lethal == 0),
                       ("  escape-rich", (lethal >= 1) & (lethal <= 2)),
                       ("  trap-heavy", lethal >= 14)):
        if mask.sum() == 0:
            print(f"  {name:<16}{0:>5}   none"); continue
        mean, lo, hi = clustered(raw_rows[mask] - floor_rows[mask], t["seed"][mask])
        star = "*" if hi < 0 else (" " if lo <= 0 else "!")
        print(f"  {name:<16}{int(mask.sum()):>5}{raw_rows[mask].mean():>9.3f}"
              f"{floor_rows[mask].mean():>11.3f}"
              f"{f'{mean:+.3f} [{lo:+.3f},{hi:+.3f}]{star}':>28}"
              f"{t['p'][mask].mean():>7.3f}{t['y'][mask].mean():>7.3f}"
              f"{within_auc(t['p'][mask], t['y'][mask]):>7.3f}")
        report["strata"][name.strip()] = {
            "n": int(mask.sum()), "bce_raw": float(raw_rows[mask].mean()),
            "bce_floor": float(floor_rows[mask].mean()),
            "difference": mean, "lower": lo, "upper": hi,
            "predicted": float(t["p"][mask].mean()), "true": float(t["y"][mask].mean()),
            "within_auc": within_auc(t["p"][mask], t["y"][mask])}

    beats = report["strata"]["all test"]["upper"] < 0
    print(f"\n  raw beats the frozen marginal on all states: {beats}"
          f"   (* interval below zero, ! above)")
    report["raw_beats_frozen_marginal"] = bool(beats)
    np.savez(folder / "deployment_calibration.npz", **{f"test_{k}": v for k, v in t.items()},
             marginal=marginal)
    (folder / "deployment_calibration.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
