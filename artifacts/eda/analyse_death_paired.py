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
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BOOT = 10000
MILESTONES = (5000, 10000, 13592, 0)
NAMES = {5000: "5k", 10000: "10k", 13592: "13,592", 0: "20k"}


def load(arm: str, milestone: int = 0):
    tag = f"_{milestone:06d}" if milestone else ""
    path = HERE / f"death_transfer_{arm}{tag}.json"
    return json.loads(path.read_text())


def band(values):
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def main() -> None:
    generator = np.random.default_rng(20260826)
    production = load("production")
    floor = np.array(production["per_root_floor"])
    n = len(floor)
    draws = generator.integers(0, n, (BOOT, n))
    print(f"paired over {n} test roots, {BOOT:,} root-clustered draws\n")

    arms = {arm: {m: np.array(load(arm, m)["per_root_pred"]) for m in MILESTONES}
            for arm in ("factual", "counterfactual")}
    control = np.array(production["per_root_pred"])
    for series in list(arms.values()) + [{0: control}]:
        for value in series.values():
            assert len(value) == n, "arms scored different roots"

    def report(label, delta):
        lo, hi = band(delta[draws].mean(1))
        star = "  *" if lo > 0 or hi < 0 else ""
        print(f"  {label:<44}{delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]{star}")

    print("against the production control, at 20k")
    for arm in ("factual", "counterfactual"):
        report(f"{arm} - production", arms[arm][0] - control)
    print()

    print("against the action-only floor, at 20k")
    report("production - floor", control - floor)
    for arm in ("factual", "counterfactual"):
        report(f"{arm} - floor", arms[arm][0] - floor)
    print()

    print("counterfactual minus factual, by milestone")
    for m in MILESTONES:
        report(NAMES[m], arms["counterfactual"][m] - arms["factual"][m])
    print()

    print("within-arm change from the previous milestone")
    for arm in ("factual", "counterfactual"):
        for a, b in zip(MILESTONES, MILESTONES[1:]):
            report(f"{arm} {NAMES[a]} -> {NAMES[b]}", arms[arm][b] - arms[arm][a])
    print()

    print("reversal interaction")
    interaction = ((arms["counterfactual"][0] - arms["factual"][0])
                   - (arms["counterfactual"][13592] - arms["factual"][13592]))
    report("(cf - f at 20k) - (cf - f at 13,592)", interaction)
    print("\n  * marks an interval excluding zero")

    out = {"roots": n, "milestones": {NAMES[m]: {
        "factual": float(arms["factual"][m].mean()),
        "counterfactual": float(arms["counterfactual"][m].mean())} for m in MILESTONES},
        "production": float(control.mean()), "floor": float(floor.mean())}
    (HERE / "death_paired_analysis.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
