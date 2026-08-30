"""Paired comparisons the marginal intervals could not settle.

Each arm was reported against the action-only floor, and each interval excluded zero
while the production control's did not. That does not establish that the arms beat the
control: a significant arm-versus-floor result beside an insignificant control-versus-
floor result is not a significant arm-versus-control difference. Nor were the arms ever
compared to each other, and their marginal intervals overlap heavily.

Every reading here is paired over the same 197 test roots, scored by the same frozen
probe, so the differences can be bootstrapped directly:

  1  factual minus production at 20k
  2  counterfactual minus production at 20k
  3  counterfactual minus factual at every milestone
  4  within-arm change across 5k, 10k, 13,592 and 20k
  5  the reversal interaction, (cf - f at 20k) - (cf - f at 13,592), which asks whether
     the ordering swap between the first complete pass and the end is real or is two
     noisy snapshots

Read-only arithmetic over the saved per-root readings.

The seed-2 matrix adds two things. `--arms` names which readings to compare, so the five
new arms can be read the same way as the seed-0 pair. And every comparison is repeated
inside prespecified strata, because the roots are bimodal on how many of their seventeen
successors are fatal: at the training roots 87.8% carry thirteen or more and 10.6% carry
two or fewer. A gain concentrated in one population would otherwise be diluted into an
aggregate null, or an aggregate win read as general when it is not. The strata are fixed
here before any seed-2 arm has finished, and are cut on the evaluation roots' own lethal
counts, which `evaluate_death_transfer` now persists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOOT = 10000
MILESTONES = (5000, 10000, 13592, 0)
NAMES = {5000: "5k", 10000: "10k", 13592: "13,592", 0: "20k"}

# prespecified, on the evaluation roots' own count of fatal successors out of seventeen
STRATA = (("escape-rich", 0, 2), ("middle", 3, 13), ("trap-heavy", 14, 17))


def load(arm: str, milestone: int = 0):
    tag = f"_{milestone:06d}" if milestone else ""
    path = HERE / f"death_transfer_{arm}{tag}.json"
    return json.loads(path.read_text())


def name_of(milestone: int) -> str:
    return NAMES.get(milestone, f"{milestone:,}")


def band(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", default="production")
    parser.add_argument("--arms", nargs="+", default=("factual", "counterfactual"))
    parser.add_argument("--milestones", type=int, nargs="+", default=MILESTONES)
    parser.add_argument("--out", default="death_paired_analysis.json")
    args = parser.parse_args()
    milestones = tuple(args.milestones)

    generator = np.random.default_rng(20260826)
    control_reading = load(args.control)
    floor = np.array(control_reading["per_root_floor"])
    n = len(floor)
    print(f"paired over {n} test roots, {BOOT:,} root-clustered draws\n")

    arms = {arm: {m: np.array(load(arm, m)["per_root_pred"]) for m in milestones}
            for arm in args.arms}
    control = np.array(control_reading["per_root_pred"])
    for series in list(arms.values()) + [{0: control}]:
        for value in series.values():
            assert len(value) == n, "arms scored different roots"

    # the strata come from the readings themselves, so they cannot drift between arms
    lethal = control_reading.get("per_root_lethal")
    masks = {"all": np.ones(n, bool)}
    if lethal is None:
        print("  (no per-root lethal counts stored; aggregate only)\n")
    else:
        lethal = np.array(lethal)
        assert len(lethal) == n
        for name, lo, hi in STRATA:
            masks[name] = (lethal >= lo) & (lethal <= hi)

    # one resample per stratum, drawn before any comparison, so every reading inside a
    # stratum is paired on the same draws. The aggregate is drawn first and over all n,
    # which is what the seed-0 analysis did, so that reading reproduces exactly.
    resamples = {}
    for name, mask in masks.items():
        kept = np.where(mask)[0]
        resamples[name] = kept[generator.integers(0, len(kept), (BOOT, len(kept)))]

    def report(label, delta, mask, stratum):
        if mask.sum() < 20:
            print(f"  {label:<48}n={int(mask.sum())}, too few roots to read")
            return
        lo, hi = band(delta[resamples[stratum]].mean(1))
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"  {label:<48}{delta[mask].mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")

    summary = {}
    for stratum, mask in masks.items():
        print(f"=== {stratum} ({int(mask.sum())} roots) ===\n")

        print(f"against the {args.control} control, at 20k")
        for arm in args.arms:
            report(f"{arm} - {args.control}", arms[arm][0] - control, mask, stratum)
        print()

        print("against the action-only floor, at 20k")
        report(f"{args.control} - floor", control - floor, mask, stratum)
        for arm in args.arms:
            report(f"{arm} - floor", arms[arm][0] - floor, mask, stratum)
        print()

        print("between arms, at 20k")
        for i, first in enumerate(args.arms):
            for second in args.arms[i + 1:]:
                report(f"{second} - {first}", arms[second][0] - arms[first][0], mask, stratum)
        print()

        print("within-arm change from the previous milestone")
        for arm in args.arms:
            for a, b in zip(milestones, milestones[1:]):
                report(f"{arm} {name_of(a)} -> {name_of(b)}", arms[arm][b] - arms[arm][a], mask, stratum)
        print()

        if len(args.arms) == 2 and 13592 in milestones:
            first, second = args.arms
            print("reversal interaction")
            report(f"({second} - {first} at 20k) - (at 13,592)",
                   (arms[second][0] - arms[first][0])
                   - (arms[second][13592] - arms[first][13592]), mask, stratum)
            print()

        summary[stratum] = {"roots": int(mask.sum()),
                            args.control: float(control[mask].mean()),
                            "floor": float(floor[mask].mean()),
                            **{f"{arm} {name_of(m)}": float(arms[arm][m][mask].mean())
                               for arm in args.arms for m in milestones}}

    print("  * marks an interval excluding zero")
    (HERE / args.out).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
