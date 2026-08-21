"""TABLE 4 and 5 -- action-dependent non-death consequences, and where they live."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
records = torch.load(HERE / "nonterminal_forks.pt", weights_only=False)
BANDS = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 22)]
EPS = [0.1, 0.25, 0.5, 1.0]
NAMES = ["NOOP", "LEFT", "RIGHT", "UP", "DOWN", "DO", "SLEEP", "PLACE_STONE",
         "PLACE_TABLE", "PLACE_FURNACE", "PLACE_PLANT", "MAKE_WOOD_PICK",
         "MAKE_STONE_PICK", "MAKE_IRON_PICK", "MAKE_WOOD_SWORD",
         "MAKE_STONE_SWORD", "MAKE_IRON_SWORD"]

rows = []
for r in records:
    health = np.array(r["health"], float)
    reward = np.array(r["reward"], float)
    ach = np.array(r["achievements"], float)
    inventory = np.array(r["inventory"], float)
    y = np.array(r["y"], float)
    rows.append(dict(
        band=tuple(r["band"]), eps=r["epsilon"], health=health, y=y,
        health_spread=float(health.max() - health.min()),
        health_distinct=len(set(health.tolist())),
        any_damage=bool((health <= -1).any()),
        damage_and_safe=bool((health <= -1).any() and (health >= 0).any()),
        reward_spread=float(reward.max() - reward.min()),
        reward_distinct=len(set(reward.tolist())),
        ach_any=bool((ach > 0).any()), inv_any=bool((inventory > 0).any()),
        inv_distinct=len(set(inventory.tolist())),
        dead_any=bool(any(r["dead"])), n_dead=int(sum(r["dead"])),
        y_spread=float(y.max() - y.min()),
        zombies=r["zombies"], skeletons=r["skeletons"],
        base_health=r["base_health"], sleeping=r["sleeping"],
    ))
n = len(rows)
print("=" * 132)
print(f"TABLE 4 -- DOES THE ACTION CHANGE A NON-DEATH OUTCOME?  ({n} ordinary TRAIN "
      f"states, all 17 actions each)")
print("=" * 132)
for label, test in [
    ("action changes health at all", lambda r: r["health_spread"] > 0),
    ("  offers both a damaging and a non-damaging action", lambda r: r["damage_and_safe"]),
    ("  health spread >= 2", lambda r: r["health_spread"] >= 2),
    ("  health spread >= 4", lambda r: r["health_spread"] >= 4),
    ("action changes reward", lambda r: r["reward_spread"] > 0),
    ("action changes achievements", lambda r: r["ach_any"]),
    ("action changes inventory", lambda r: r["inv_any"]),
    ("action changes terminal status", lambda r: r["dead_any"] and r["n_dead"] < 17),
]:
    hits = sum(test(r) for r in rows)
    print(f"  {label:<52}{hits:>7} / {n}{hits/n:>10.1%}")

print()
print("  distinct values across the 17 actions (mean)")
for label, key in (("health delta", "health_distinct"), ("reward", "reward_distinct"),
                   ("inventory change", "inv_distinct")):
    print(f"    {label:<24}{np.mean([r[key] for r in rows]):>6.2f} / 17")

print()
print("  fatality-direction delta y across the 17 actions from the same state")
ys = np.array([r["y_spread"] for r in rows])
print(f"    within-state spread of y: median {np.median(ys):.4f}  mean {ys.mean():.4f}  "
      f"p90 {np.quantile(ys, 0.9):.4f}")
print("    for reference: Experiment 1's whole-corpus target std is 0.1386, and the")
print("    fork gate's true fatal-minus-safe contrast is about 0.33")
both = [r for r in rows if r["damage_and_safe"]]
contrasts = []
for r in both:
    hurt, fine = r["y"][r["health"] <= -1], r["y"][r["health"] >= 0]
    if len(hurt) and len(fine):
        contrasts.append(hurt.mean() - fine.mean())
contrasts = np.array(contrasts)
if len(contrasts):
    print(f"    within-state damage-minus-safe y contrast on the {len(contrasts)} states")
    print(f"      offering both: mean {contrasts.mean():+.4f}  median "
          f"{np.median(contrasts):+.4f}  share positive {(contrasts>0).mean():.1%}")
    boot = np.array([contrasts[np.random.default_rng(s).integers(0, len(contrasts),
                     len(contrasts))].mean() for s in range(2000)])
    print(f"      95% interval [{np.quantile(boot,0.025):+.4f}, {np.quantile(boot,0.975):+.4f}]")

print()
print("=" * 132)
print("TABLE 5 -- WHERE THE INFORMATIVE EXAMPLES LIVE")
print("=" * 132)
tests = [("any damage", lambda r: r["any_damage"]),
         ("damage+safe", lambda r: r["damage_and_safe"]),
         ("health spread>=2", lambda r: r["health_spread"] >= 2),
         ("reward differs", lambda r: r["reward_spread"] > 0),
         ("any death", lambda r: r["dead_any"])]
print(f"{'':<18}" + "".join(f"{f'{lo}-{hi} ach':>14}" for lo, hi in BANDS) + f"{'all':>12}")
for label, test in tests:
    line, hits, tot = f"{label:<18}", 0, 0
    for band in BANDS:
        group = [r for r in rows if r["band"] == band]
        h = sum(test(r) for r in group)
        hits, tot = hits + h, tot + len(group)
        line += f"{f'{h}/{len(group)}':>14}"
    print(line + f"{f'{hits}/{tot}':>12}")
print()
print(f"{'':<18}" + "".join(f"{f'eps {e}':>14}" for e in EPS) + f"{'all':>12}")
for label, test in tests:
    line, hits, tot = f"{label:<18}", 0, 0
    for e in EPS:
        group = [r for r in rows if r["eps"] == e]
        h = sum(test(r) for r in group)
        hits, tot = hits + h, tot + len(group)
        line += f"{f'{h}/{len(group)}':>14}"
    print(line + f"{f'{hits}/{tot}':>12}")

print()
print("  hazard context of the states offering a damage / no-damage choice")
if both:
    print(f"    n = {len(both)}; zombies present {np.mean([r['zombies']>0 for r in both]):.0%}, "
          f"skeletons present {np.mean([r['skeletons']>0 for r in both]):.0%}, "
          f"asleep {np.mean([r['sleeping'] for r in both]):.0%}, "
          f"mean health {np.mean([r['base_health'] for r in both]):.2f}")
    counter = Counter()
    for r in both:
        for a in np.where(r["health"] <= -1)[0]:
            counter[int(a)] += 1
    print("    actions that cause the damage: " + ", ".join(
        f"{NAMES[a]} {c}" for a, c in counter.most_common(6)))

share = float(np.mean([r["damage_and_safe"] for r in rows]))
print()
print(f"  Extrapolated to 3,111,438 TRAIN transitions, a damage / no-damage choice is")
print(f"  available at about {share:.1%} of states -- roughly {share*3_111_438:,.0f} states,")
print(f"  against ~374 action-decidable death roots in the whole corpus.")
