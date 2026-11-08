# Additional danger coverage — 2026-11-04

Explicitly authorized by the user's follow-up: collect enough risk/death coverage,
not just rare achievements. The first collection completed3,072 episodes; actual
TRAIN additions were2,311 deaths,12,194 risk transitions and759 deaths within
the risk class, across3.711M transitions. This is less danger-dense than the old
corpus, so adding it indiscriminately would dilute danger supervision.

## New bounded acquisition component

New independent reset keys, seed202611500, checked against support-v2 AND the
completed coverage-v1 collection. Same environment and frozen expert. All
episodes retained through first termination or declared cap; no filtering on
success, damage, death or achieved return. Old dev/final sources stay untouched.

Budget:512 batches x24 environments =12,288 fresh episodes maximum, two-hour
wall cap. Interleave eight reset-exploration batches with one late-failure batch:

- **Reset exploration:** epsilon cycle .5,1,.5,1,.5,1,.25,1, limit1,024 actions.
  Fresh natural episodes cover low-health encounters, starvation/thirst, sleep,
  projectiles and unsafe/unsuccessful actions. Surviving cap-limited episodes
  remain in the store with truncation, never relabeled as death.
- **Later-stage failures:** epsilon .02 for a fixed512 or1,024 actions, then
  completely uniform actions until termination or2,500 actions. Prefix alternates
  across these batches. The switch is clock-based, not selected from an eventual
  outcome. This keeps complete real histories into failure from more competent
  inventory/world states rather than only random early-game deaths.

Keep separate store `artifacts/craftax_risk_v1/`, with the same per-transition
epsilon, actual random-branch flags, action probabilities and state sidecars.
Retain frozen collector mechanics; first full reset-exploration and first full
late-failure batch get independent exact replay checks. Same CPU contract tests,
source pins, atomic commits, monitor assignment and all22-skill accounting.

## Required reading before tonight's model run

Report unique source episodes, damage, death, low-health-adjacent-zombie risk,
risk deaths, day/night, death health/lava causes, and early/late components.
Inspect conditional health, sleep, cooldown, projectile and starvation support
from the saved sidecars if those cells remain thin. Do not count duplicated
prefixes as new worlds: this collector has fresh roots and no copied prefixes.

Counts AND per-frame rates must be shown for each component and the proposed
training mixture. More independent events do not automatically give more
optimization pressure at a fixed25.6M-frame budget. Keep the data components
separate so a long competent-play addition cannot silently swamp danger.
Collection does not choose the final training mixture, change acuity, or launch
the overnight model. The data-only comparison should preserve F2's architecture,
3x3 input acuity, learned chooser, objectives and evaluation conditions.
