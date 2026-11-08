# P1-F2.1 vs P1-F2: stage-by-stage reading (11-05 evening; held-out own-gaze stages pending)

Source: `../20261104_p1f21/readings/` (Codex's queue, F2.1 ckpt_10 vs F2 ckpt_10, same instruments, one seed each).
"Beyond within-run noise" = |difference| > 3 SE (the instruments' episode-bootstrap SEs, or non-overlapping 95% CIs
where only CIs are given). Seed-level spread is not measured: F2 vs F1.6 moved classes 2-10 pp both ways (a recipe
change too), so differences under ~10 pp are read as patterns only (F2.1 PLAN). Declared expectations (F2.1 PLAN): if
exposure governs, up for plant 1.60x, diamond 1.45x, drink 1.19x, iron increments 1.08x; slightly down for most first
unlocks (0.71-0.83x) and danger (0.77-0.86x); if distinct instances govern, gains also where exposure fell.

| stage | beyond within-run noise (F2.1 - F2) | reading |
|---|---|---|
| S1.1 perception probes | none | unchanged |
| S1.2 belief | diamond unlocked AP 0.41 / 0.52 -> 0.64 / 0.65 (both probe seeds); health exact +0.010 (z 4) | diamond tracked better (union: ~5x diamond episodes) |
| S1.2 percept controls | HUD body values exact from the percept: food -9.5 / -5.6 pp (z -15 / -10), drink -4.3 / -3.0, energy -4.0 / -4.4, health -2.6 | the encoder's tokens carry body stats less exactly; the belief compensates |
| S1.3 generation | generated cows 0.237 -> 0.278 (z ~5.5; real 0.47); arrows 0.103 -> 0.082 (real 0.091); C2ST generated vs real 0.968 -> 0.976; reward NLL 0.0094 -> 0.0081 | small; samples slightly MORE distinguishable from real |
| interface (random gaze; guided agrees) | up: wake +0.120 (z 8.6), eat cow +0.099 (7.6), zombie +0.070 (6.4), stone +0.045 (3.5), stone pickaxe +0.044 (3.4); down: wood -0.060 (-4.3), skeleton -0.061 (-4.4), iron sword -0.051 (-4.3), iron -0.038 (-3.5); within 3 SE: furnace +0.031, stone sword +0.037, drink +0.027 (z 2.5-2.8); coal, sapling, diamond, damage, death: no change | the four primaries unchanged (as the train-vs-dev prior predicted); exposure ratios predict few of the moves (iron down at 1.08x; first unlocks up where exposure fell) |
| M0.5 (samples' readers) | table placement +0.23 / +0.26 (z 16-18) with false tables +0.033 (z 6-7); drink APPEARANCE -0.23 / -0.31 (z -6 / -7.5) with false drinks -0.066 (z -3.9); wood increment -0.089 (z -8.7); risky +0.013 (z 3) | the largest effects of the run: tables much better (exposure fell: context diversity, not exposure), drink appearance much worse though drink exposure rose 1.19x |
| token geometry | health token step / radius 3.10 -> 2.74 (no SE) | consistent with the percept's weaker body-stat decoding |
| render (decoder, no SE) | rare blocks worse (coal +0.32, iron +0.35, diamond +1.46 MAE), water better (-0.22) | descriptive |
| Codex panel / counters / NOOP counters / movement | 0 of 56 metrics with non-overlapping CIs; paired dark-scene differences <= 0.006 | unchanged |
| Codex sampler factorial | (both worlds) entity counts in generated frames rise with steps: F2 cows 0.25 (3) -> 0.45 (8) -> 0.50 (32), real 0.49 | independent support for the sampler's majority rounding (RESULTS.md 5, 7) |
| consistency | reward-head view-dependence: mean over fixations 0.008 -> 0.031, spread / SE 21.6 -> 26.2 | small, up |
| locus (world's action contingency) | more on the counter: iron 0.23 -> 0.33, stone 0.37 -> 0.50 (z ~4), coal 0.25 -> 0.31 | modest shift; no class changes classification |
| head / fovea / gaze / chooser trace | changes <= 0.05 within their intervals; coal / iron stay surprise-led | unchanged |
| own-gaze dev (each world's closed-loop gaze) | wake +0.120 (z 8.6), cow +0.104 (8.7), zombie +0.064 (5.9), stone pickaxe / sword +0.049 / +0.041; wood -0.059, skeleton -0.060, iron sword -0.051, iron -0.034, place plant -0.034 (all z 3.2-4.5); damage, coal, sapling unchanged | reproduces the random-gaze interface: F2.1 is gaze-insensitive like F2 |
| own-gaze held-out COVERAGE (new states; F2 never trained on them) | cow +0.101 (z 4.8), stone pickaxe / sword +0.052 / +0.033, wake +0.061 (3.1); skeleton -0.093 (-4.6), coal -0.081 (-3.1), iron sword -0.075 (-3.7); risky damage 0.006 / 0.011 and death 0.001 / 0.002 vs recorded 0.069 / 0.064 (both worlds ~0) | the dev pattern repeats on new states |
| own-gaze held-out RISK (danger-rich; F2.1 trained on its source) | cow +0.140 (z 11.6), wake +0.131 (9.4), zombie +0.068 (4.4), stone +0.051 (3.7); iron -0.048 (-3.0); risky damage 0.066 / 0.066 vs 0.169, DEATH 0.022 / 0.028 vs 0.167 (no change) | danger is not improved even on the danger-rich distribution F2.1 trained on |

Overall (all stages read except the deferred CPU uptake): the union corpus at F2's compute reshuffles classes rather than repairing the
primaries: large gains (tables, wake, cow, zombie) and large losses (drink appearance, wood) beside unchanged coal /
iron / damage / death. The exposure-ratio pattern declared in the PLAN does not organise the changes; the table gain
where exposure fell favours context diversity for that class. Some swings may be the 3-step sampler's threshold
amplifying small field changes (hypothesis; the equal-budget 16-step interface on both worlds tests it).
The held-out readings settle the F2.1 question: diversity at F2's compute does not improve danger or death, even on
the danger-rich source it trained on (death 2.2% -> 2.8% vs 16.7% recorded); consistent with the 3-step sampler erasing
per-row minority outcomes regardless of data (RESULTS.md 5, 7), and with the adjacent-damage field deficit.
