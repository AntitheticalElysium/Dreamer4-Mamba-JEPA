# Decision levels: results

PLAN.md (929bb069) and code (8dda93b8) were committed before any measurement. Frozen P1-F3 checkpoint 10 for T1 / T2;
T3 is in section 3.

## 1. T1: where the world stops following the target, and what it predicts from context alone

272 pairs (the audit's 208 + 64 MOVEMENT pairs: all 64 replayed bitwise, natural successors equal to the stored
frames), K = 8 shared draws, first evaluation only. Reliance = |f(conflicting target) - f(own target)| (1 = follows
the noisy target, 0 = follows its context); sigma* = where it falls through 0.5; ideal = an equal-prior two-point
reader at the pair's own separation d.

| family | d (median) | reliance at sigma = 1 / 2 / 5 / 12 / 20 | sigma* network | sigma* ideal | pure-noise first evaluation at 20 (member 0 / 1) |
|---|---|---|---|---|---|
| sleep (asleep / woken) | 28.9 | 0.80 / 0.58 / 0.43 / 0.12 / 0.04 | 3.3 | 15.9 | 0.06 / 0.98 (0 = asleep) |
| energy (8 / 9, asleep) | 29.4 | 0.99 / 0.99 / 0.90 / 0.26 / 0.09 | 8.7 | 16.1 | 0.06 / 0.06 (both asleep) |
| coal (DO / NOOP) | 3.7 | 0.48 / 0.18 / 0.03 / 0.00 / 0.00 | 0.9 | 2.0 | 0.93 / 1.01 (1 = copy) |
| wood (DO / NOOP) | 4.6 | 0.60 / 0.18 / 0.03 / 0.01 / 0.00 | 1.2 | 2.5 | 0.12 / 1.00 |
| movement (move / NOOP) | 18.0 | 0.36 / 0.07 / 0.01 / 0.00 / 0.00 | 0.85 | 9.9 | 0.10 / 0.98 (0 = moved) |

Spearman(sigma*, d) over pairs 0.78.

- **H-dec (the level is set by the consequence's size): REFUTED by the declared rule.** Movement sits 11.6x below
  its ideal level and out of order (largest-but-two d, lowest sigma*); sleep 4.8x. The level tracks d only where the
  context says nothing about the outcome (energy: 1.9x); wherever the world has a confident context prediction
  (movement, persistence; counts at small d) it stops consulting the target far earlier. The governing variable is
  the confidence of the learned context prediction together with d, which is what a posterior-mean denoiser
  combining a learned conditional prior with the noisy target does (EDM: D = E[x | x_sigma, c]).
- **H-ctx (above that level the output is the context-only prediction): SUPPORTED.** The pure-noise first evaluation
  at sigma 20 matches the generated rates of the prefix-20 recipe (audit) in every family: sleep 0.06 vs 0.08 woken;
  energy 0.06 / 0.06 vs 0.06 / 0.07 woken; coal increment 0.07 vs 0.09; wood 0.88 vs 0.89; movement 0.90 moved. The
  energy-9 context-only prediction is the marginal (asleep), not the rule.
- **Movement, the case that could break H-start: not broken, and it explains the start effect.** Generated movement
  is 100% correct at both starts (move member moved 1.00, NOOP member 0.00, K = 16). At the original start the
  world's reliance on the target is 0.01 for movement, 0.03 for coal / wood, but 0.43 for sleep and 0.90 for energy:
  pure noise at sigma 5 replaces a target only where the world still consults one, and those are exactly the families
  the higher start changes (sleep -37 pp, energy -28 pp; coal / wood / movement within 4 pp).

## 2. T2: the determinant curves with the prefix-20 start (same tuples, seeds; sham bitwise)

| declared reading | original start (exp 2) | prefix-20 | prediction |
|---|---|---|---|
| woken from asleep, energy <= 8 (truth 0.01-0.03) | 0.42-0.52 | 0.06-0.11 | <= 0.15: holds |
| energy value contrast T | 0.02 | 0.04 (woken at 9: 0.54 -> 0.15; truth 1.00) | < 0.10: holds |
| cooldown T | 0.23 | 0.24 | change < 0.10: holds |
| lethality T | 0.27 | 0.22 | holds |
| sleep-attack T | 0.08 | 0.04 | holds |
| cow / zombie kill T | 0.16 / 0.17 | 0.20 / 0.20 | holds |
| coal / wood increments (n >= 1) | 0.105 / 0.587 | 0.093 / 0.607 | holds |

The start lever is confined to the sleep / wake state and what rides on it: a ready zombie hitting a SLEEPING agent
(truth: damage 1.00, wakes 1.00) is drawn with damage 0.43 -> 0.18 and woken 0.63 -> 0.23. All six BLIND verdicts of
the determinant curves survive the corrected start. The heads still read what the frame draws (wake unlocks 321 / 321
and 284 / 284 on awake frames, death 1,003 / 1,003). The false-wake reward exploit shrinks (head unlock per asleep step
0.15-0.36 -> 0.03-0.10), but true waking at energy 9 falls to 0.15: with the start fixed, imagined sleep becomes a
state the agent rarely leaves.
