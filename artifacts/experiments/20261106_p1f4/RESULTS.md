# P1-F4 results (11-07): shortcut forcing + dense long-context attention, against the declared readings

Run: 25.6M frames, world seed 1, training 00:39-15:02 (14.4 h, as estimated), monitor falling at every checkpoint (no
three-increase trigger) -> checkpoint 10 (sha 1b70edcc...). Readout complete (S1, consequences, interface, recursion),
all at the sampler's K = 4 from pure noise, no start recipe. F3's references as declared: curves exp 2 (16 EDM steps)
and decision levels T2 (prefix-20); crafting (prefix-20); F3's readout (interface and recursion at 3 and 16 steps).
Tables not printed by the instruments themselves: `compare.py` -> `readings/comparison.json` (saved outputs only).
This tests the combined recipe: no reading is attributed to shortcut forcing or to dense attention alone.

## Declared readings

| declared reading | F3 (exp 2 / prefix-20) | F4 | verdict |
|---|---|---|---|
| woken from asleep at energy <= 8 (truth 0.01-0.03) | 0.42-0.52 / 0.06-0.11 | 0.006-0.023 per stratum | sleep HELD (<= 0.15) |
| ... with correct waking: woken at energy 9 (truth 1.00) | 0.54 / 0.15 | 1.00 | holds (>= 0.15) |
| ... woken by a ready zombie's hit while asleep (truth 1.00) | 0.65 / 0.22 | 0.56 | holds (>= 0.23) |
| energy 9 -> wake, contrast T (learned >= 0.33) | 0.02 / 0.04 | 0.99 (1.00 vs 0.006) | LEARNED |
| coal 0 -> 1 (support >= 0.50) | 0.28 / 0.25 | 0.99 | supported |
| iron 0 -> 1 (support >= 0.35) | 0.18 / 0.13 | 0.92 | supported |
| increments keep following per-transition exposure | n = 1-4: 0.04-0.86 | n = 1-4: 0.79-1.00 (iron 4: 0.56, 36 futures) | REFUTED (my prediction) |
| control: cleared cell (coal / iron / wood / stone) | 0.94-0.95 / 0.75-0.76 / 0.97 / 0.94 | 0.96 / 0.69 / 0.99 / 0.95 | holds (iron -6 pp) |
| control: crafted tool appears when feasible (6 tools) | 0.79-0.90 | 0.81-0.96 | holds (better) |
| hidden counters: no change predicted (cooldown / lethality T) | 0.23 / 0.27 (T2 0.24 / 0.22) | 0.16 / 0.12 | lethality CHANGED (worse) |
| movement | - | - | UNMEASURED (needs F4-fitted readers) |

## What changed beyond the thresholds

- **Counts.** Every (item, n) transition up to n = 4 is drawn 0.79-1.00 (F3 0.04-0.89) except iron 4 -> 5 (0.56, 36
  futures), the rare items included (coal 4 -> 5 at 200 futures: 0.79 vs 0.04). The one transition that fails in every item is 5 -> 6 (coal 0.02, iron 0.00,
  wood 0.00, stone 0.33 at 600 futures); high wood counts (6-8) stay low. Exp 2 already saw 5 -> 6 as the smallest
  token step; it is not exposure-governed (stone's exposure is the highest). A held tool's count + 1 is still copied
  (0.00-0.06; F3 0.01-0.19). Crafting coherence (materials spent and tool drawn) rose: e.g. wood pickaxe 0.80 -> 0.92,
  stone pickaxe 0.50 -> 0.70; the iron tools stay low (coherent 0.24 / 0.21 -> 0.30 / 0.32: their materials are
  rarely all spent).
- **Actor interface (random policy).** Unlocks drawn on recorded unlock events: coal 0.30 / 0.44 (F3 3 / 16 steps) ->
  0.98, iron 0.11 / 0.24 -> 0.90, wake 0.35 / 0.44 -> 0.88, stone tools 0.81-0.86 -> 0.95-0.96. Unchanged: eat cow
  0.24, sapling 0.06, defeat zombie 0.29, diamond 0.05.
- **Danger regressed.** Risky class (recorded damage 0.127, death 0.128): F4 0.042 / 0.036 vs F3 16 steps 0.110 / 0.088
  (3 steps 0.057 / 0.036). Beside a zombie (recorded damage 0.159): 0.054 vs 0.097. Curves: a ready zombie damages 0.08
  (truth 0.32; F3 0.14), lethal hits 0.04 (truth 0.34; F3 0.09), a sleeping agent hit 0.10 (truth 1.00; F3 0.38 / 0.15).
  The heads read the frame (world-drawn, all four agreements). F3 drew more danger with more sampler steps; F4 was
  read only at K = 4.
- **Waking beside a zombie is partly decoupled from the hit:** asleep with a ready zombie, woken 0.56 but damaged 0.10;
  with a cooling zombie (no hit possible) woken 0.18 (truth 0.007; F3 T2 0.19).
- **Own-output recursion (16 steps, 128 windows x 4):** free-running visible entities 1.02 / 0.97 / 0.89 / 0.86 at
  steps 1 / 4 / 8 / 16 (truth 1.11 / 1.22 / 1.33 / 1.27): a slow decline, against F3's collapse at 3 steps (0.13) and
  overshoot at 16 (1.63). Positive reward per step on real history 0.015 vs truth 0.014 (F3 0.039 / 0.050: the 3.4x
  over-prediction is gone); free running 0.007. Free-running terminations by step 16: 1.6%.
- **S1.** Belief probes (S1.2) within 0.01 of F3's or better on every target but one (departed cows' side 0.44 ->
  0.43); e.g. HUD exact 0.41 -> 0.45 on the hardest slot, achievement 19 AP 0.43 -> 0.50: no sign in these probes that
  the dense path drained the belief. Perception (S1.1) and
  TSP (S1.4) unchanged. Real-vs-generated C2ST 0.978 -> 0.999 (power references 0.98 / 0.95 / 0.90): generated next
  frames are more separable from real ones at the token level than F3's; which features separate them is unmeasured.

## Open, not measured tonight

Movement (F4-fitted readers); whether the danger regression is the K = 4 sampler (the shortcut model also samples at
K = 8) or the trained world; what the C2ST separates; memory versus cache routes on F4 (exp 3's swap).
