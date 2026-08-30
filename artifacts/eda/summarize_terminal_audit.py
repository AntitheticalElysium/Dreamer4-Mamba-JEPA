"""Turn the terminal census into the two strata and a matched factual index.

Reads whatever shards `audit_terminal_tails` has finished, so it can be run partway
through for a running estimate and again at the end for the census.

Reports the actionable rate overall and by collection epsilon, the distribution of safe
alternatives, which actions are lethal and which the policy took, and how actionability
relates to the covariates that would otherwise confound a tail-versus-all-17 matchup --
health, light, timestep and mob counts.

Then writes `terminal_strata.json`:

  unavoidable   every action kills. Teaches terminal appearance and continuation, and
                cannot teach an action-conditioned mechanic.
  actionable    the taken action killed and at least one other survives. The only tails
                carrying the mechanic production fails to learn.

For the factual arm it also emits, per actionable root, the lethal action actually taken
and the safe alternatives available at that same state, so a factual-versus-
counterfactual comparison can be matched on root rather than on volume.
"""

from __future__ import annotations

import glob
import json
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def wilson(k, n, z=1.96):
    if not n:
        return (float("nan"), float("nan"))
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return (centre - half, centre + half)


def main() -> None:
    rows = []
    for path in sorted(glob.glob(str(HERE / "terminal_audit" / "shard-*.json"))):
        rows += json.loads(Path(path).read_text())
    if not rows:
        raise SystemExit("no audited shards yet")
    shards = len({r["shard"] for r in rows})
    actionable = [r for r in rows if r["actionable"]]
    unavoidable = [r for r in rows if r["all_lethal"]]
    survived = [r for r in rows if not r["taken_lethal"]]

    lo, hi = wilson(len(actionable), len(rows))
    print(f"{len(rows):,} TRAIN terminal tails from {shards} shards")
    print(f"  actionable    {len(actionable):,} ({len(actionable)/len(rows):.1%}) "
          f"[{lo:.1%}, {hi:.1%}]")
    print(f"  all lethal    {len(unavoidable):,} ({len(unavoidable)/len(rows):.1%})")
    print(f"  taken action survived  {len(survived):,} "
          f"({len(survived)/len(rows):.1%})  -- terminal by other means")

    print("\n  by collection epsilon")
    for eps in sorted({r["epsilon"] for r in rows}, key=lambda x: (x is None, x)):
        sub = [r for r in rows if r["epsilon"] == eps]
        act = sum(r["actionable"] for r in sub)
        l, h = wilson(act, len(sub))
        print(f"    {str(eps):<6} {len(sub):>6,} tails   actionable {act/len(sub):>6.1%} "
              f"[{l:.1%}, {h:.1%}]")

    safe = np.array([17 - r["n_lethal"] for r in rows])
    print(f"\n  safe alternatives per tail: mean {safe.mean():.2f}, median "
          f"{np.median(safe):.0f}, zero for {(safe == 0).mean():.1%}")
    among = np.array([17 - r["n_lethal"] for r in actionable])
    if len(among):
        print(f"  among actionable tails:     mean {among.mean():.2f}, median "
              f"{np.median(among):.0f}")

    taken = Counter(r["taken_action"] for r in actionable)
    print("  most common lethal action taken at actionable roots: " +
          ", ".join(f"{a}:{c}" for a, c in taken.most_common(5)))

    print("\n  covariates, actionable against unavoidable")
    for key in ("health", "light", "steps", "n_zombies", "n_skeletons", "food", "energy"):
        a = np.array([r[key] for r in actionable], dtype=float)
        u = np.array([r[key] for r in unavoidable], dtype=float)
        if len(a) and len(u):
            print(f"    {key:<12} actionable {a.mean():8.3f}   unavoidable {u.mean():8.3f}")

    strata = {
        "counts": {"total": len(rows), "actionable": len(actionable),
                   "unavoidable": len(unavoidable), "shards": shards},
        "unavoidable": [{"shard": r["shard"], "slot": r["slot"], "steps": r["steps"]}
                        for r in unavoidable],
        "actionable": [{"shard": r["shard"], "slot": r["slot"], "steps": r["steps"],
                        "lethal_action": r["taken_action"], "safe_actions": r["safe_actions"],
                        "n_lethal": r["n_lethal"], "epsilon": r["epsilon"],
                        "health": r["health"], "light": r["light"]}
                       for r in actionable],
    }
    (HERE / "terminal_strata.json").write_text(json.dumps(strata, indent=1))
    print(f"\nwrote terminal_strata.json: {len(actionable):,} actionable roots with their "
          f"lethal action and safe alternatives, {len(unavoidable):,} unavoidable")


if __name__ == "__main__":
    main()
