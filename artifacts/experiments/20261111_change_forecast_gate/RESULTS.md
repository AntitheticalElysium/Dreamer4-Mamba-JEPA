# Change-map forecast mode: objective gate results

2026-11-11, CPU only, no training. Protocol and decision rules: `PLAN.md` (committed before running, a426e8f0).
Raw outputs: `STAGE0.json`, `CHECKS.json`; decision arithmetic: `summarize.py` -> `SUMMARY.json`.

**Verdict under the preregistered rules: fail, narrowly. No training plan follows.** The forecast target prices older
evidence about 2.7-2.8x better than G1 does at pure noise, relative to each objective's natural noise floor. That is below
the 3x bar, the advantage is concentrated in dark scenes, and a +1/255 brightness step out-prices the consequence in 2 of
32 zombie scenes.

## Controls

- The local copy of the target function reproduces `D.change_probabilities` exactly on a full DEV episode (182
  transitions, identical values and validity).
- The 24-pair hunger bank was rebuilt bitwise; 23 qualify (episode 84 terminates, as before). The 11-11 gate's
  one-step hunger cost on scene 9 reproduces exactly (.0044957).
- All 32 zombie pairs came from distinct DEV episodes (37 states scanned): frames 3-6 byte-identical, frames 1-2
  different, the hit at transition 6 only in member A. Branch frame 0 equals the recorded frame in every pair.
- Control transitions (before the informative one) cost exactly 0 in every C1 and C2 scene.

## Stage 0: the 11-11 foil failure was a support-window artifact

| Scene 9, one step at the informative transition | Hunger | +1 | +3 | +6 |
|---|---:|---:|---:|---:|
| Support over a 17-frame branch window (as the 11-11 gate) | .0045 | | .0555 | |
| Support over the episode (as the canonical store) | .0084 | .0022 | .0004 | .0003 |

With the short window, the support is concentrated on the few cells that vary in 17 frames, and the plain member (not
the foil) picks a wrong camera shift at 3 of 16 transitions although the player never moves. The foil makes every cell
vary, which restores the correct shift in the foil member only, so the two members differ in their translation planes.
With episode-level support every shift is (0, 0) and the foil is cheap. The pre-declared photometric candidate changed
nothing and was not needed; the motion check was therefore not run.

## Stage 1

| | C1 hunger (23 pairs) | C2 zombie readiness (32 pairs) |
|---|---:|---:|
| Consequence cost, change target (median, range) | .0084 (.0084-.0084) | .0083 (.0021-.0255) |
| Consequence cost, G1 at pure noise (median) | .00223 | .00218 |
| Event cost / natural floor, change target | .63 | .62 |
| Event cost / natural floor, G1 | .23 | .22 |
| **Pricing ratio R, median [95% CI]** | **2.74 [2.40, 3.18]** | **2.83 [1.46, 4.78]** |
| R against the day floor / the dark floor | 1.92 / 5.83 | 1.98 / 6.00 |
| Worst foil / consequence, change target (+1, +3, +6) | .29, .06, .07 | 1.39, .55, .71 |
| Worst foil / consequence, G1 (+1, +3, +6) | .02, .16, .64 | .06, .48, 1.86 |

Natural floors (240 DEV states x 8 keys): change target .0134 (day .0137, dark .0123); G1 .0097 (day .0070, dark .0190).
C1's consequence is one cell (HUD row 7, col 1) in both comparisons. In 25 of 32 zombie scenes it is only the health
cell; in the other 7 it also includes cells around the player, because the non-attacking zombie moves while the
attacking one stays.

| Rule | C1 | C2 |
|---|---|---|
| Consequence priced in every pair | pass | pass |
| Controls cost zero | pass | pass |
| Median R >= 3 | fail (2.74) | fail (2.83) |
| Every foil cheaper than the consequence (C3) | pass | fail: +1 in 2/32 scenes (e 40, 988); +3, +6 pass |

## What this shows

1. **Changing the currency buys roughly 2-3x, not an order of magnitude.** Both objectives are per-cell averages, and the
   consequence is one cell competing with the few cells that are genuinely random each step (mob moves). That spatial
   ratio survives the switch from squared error to BCE. Against the day floor the gain is about 2x; most of the
   advantage comes from dark scenes, where key-drawn night static nearly triples G1's floor and leaves the change
   target's unchanged.
2. **The change target has two quality problems of its own.** Soft detector outputs under-price real changes: the
   same kind of event, a 2-point health drop, costs .0021, .0045 or .0083 depending on the digits. And any non-exact
   difference switches off the exact-zero rule on every cell, so a +1/255 step is the target's worst nuisance
   (G1 is robust there, but G1 pays for large brightness steps).
3. **Not measured here:** the gradient route. The forecast head would read the belief directly, whereas G1 reaches it
   through the denoiser, alongside the cache, the noisy target and the ramp weighting (pure-noise targets get a tenth
   of the weight of near-clean ones). This gate prices targets only; it compares against G1 at pure noise, unweighted,
   which is G1's most favourable case.

One training seed of the F4 encoder for G1's targets; constructed pairs, not prevalence estimates; oracle incentives,
not proof that SGD would learn or use the distinction.
