# Stage B results (F2; F2.1 for agreement below when it lands)

Run 11-05 14:13-14:35 (unit trace-f2, MAMBA_DETERMINISTIC=1; 21 min 26 s, host peak 9.8 GB, GPU 3.6 GB), F2 ckpt_10
(sha256 e0f6a24e...), trace.py sha256 63b8b1df..., all 1,009 dev episodes. Declared readings applied by `readings.py`
(committed 5d41d156 before any F2 number); `readings/f2/readings.json`. Section 3 is POST-HOC (`posthoc.py`).

## 1. Declared readings

R0 passes everywhere: SAFE max |f| 0.007 (<= first level, first call, production), bias at sigma 0.05 and rerun
control < 0.01 on every class and part, recorded outcome reached in >= 99% of mining / drink rows. Measurability gate
(reader (b) or (c) AP >= 0.95): coal passes (b 0.99999); iron fails (b 0.891, c 0.708) and drink fails (b 0.944,
c 0.879), so their R2 / R5 are descriptive; wood has no B1 population; the attack population is not qualified (rule
inputs AP 0.939 < 0.95).

| class / part | rec f | first call | rho (vs rec) | R2 | lowest under-weighted sigma / sigma_d2 | R4 sampler | production | declared interpretation |
|---|---|---|---|---|---|---|---|---|
| coal HUD | 1.000 | 0.159 | 0.16 | lacks | 0.81 / 2.48 | -0.045 (first estimate's) | 0.114 | not localised |
| coal map cell | 1.018 | 0.952 | 0.95 (0.94) | exact-like | 1.94 / 1.68 | +0.023 | 0.975 | carried to the samples |
| iron HUD (descr.) | 0.999 | 0.051 | 0.05 | lacks | 1.2 / 2.98 | -0.028 | 0.022 | not localised |
| iron map cell (descr.) | 0.761 | 0.652 | 0.65 (0.86) | between | 0.7 / 1.09 | +0.014 | 0.666 | not localised |
| wood HUD | 1.002 | 0.685 | 0.69 | between | 1.2 / 2.29 | +0.012 | 0.697 | not localised |
| wood map cell | 1.041 | 0.980 | 0.98 (0.94) | exact-like | 20 / 1.93 | +0.022 | 1.002 | carried to the samples |
| drink HUD (descr.) | 1.000 | 0.482 | 0.48 | lacks | 0.56 / 1.58 | -0.042 | 0.440 | not localised |
| risky damage | 0.125 | 0.105 | 0.73 | between | none / 2.96 | -0.075 (between) | 0.029 | not localised |
| adjacent damage | 0.180 | 0.063 | 0.34 | lacks | 1.94 / 2.03 | -0.045 | 0.018 | not localised |

What the declared readings rule out, per class:
- Sampler (N2): mining and drink R4 within +-0.05: the deficit is the first estimate's. Risky damage is "between" by
  the declared absolute threshold; the sampler removes 72% of its first estimate (0.105 -> 0.029), but a class with a
  0.13 rate cannot reach the -0.1 threshold (a plan defect for p < 1 classes, mine; descriptive).
- Start distribution: R3 ~ 0 everywhere (real successor + noise at the first level gives what the pure-noise first call
  gives, |diff| <= 0.06); no class is calibrated at the first level with R2 <= 0.5.
- N1 as declared: NOT shown on any class. Its signature (calibrated below sigma_d2, under-weighted above) fails: the
  HUD field departs from calibration at sigma 0.56-1.2, a third to a half of sigma_d2.
- H1 / H2 as declared: no class. Descriptive: coal is in the predicted belief (b 0.99999) beyond what the single-frame
  readers reach (c 0.76, e 0.77); iron's projection output carries it (e 0.98) more than the belief (b 0.89); attack is
  not accessible to any arm (best b 0.62) nor fully to its rule inputs (0.94: the hidden cooldown).

R6 (descriptive): NOOP removes the mined cell (coal contrast 0.83, wood 0.97) and the wood / drink counts (0.69 / 0.44);
coal / iron HUD contrast 0.16 / 0.04. R7 (descriptive stress): context at sigma 5 collapses the cell's action contrast
(coal -0.79, wood -0.83) and wood / drink HUD's (-0.46 / -0.36). R8 (the measured training term, random frames):
by band 51.5 / 25.1 / 15.4 / 6.0 / 2.0% (sigma < 0.3 / 0.3-1 / 1-3 / 3-7 / > 7), glimpse 23%; with the same errors
weighted as an x-space loss 4.3 / 22.9 / 43.2 / 21.7 / 7.8%, constant 3.86; RESEARCH.md's surrogate (56 / 24 / 13 / 5 /
2-3) was close. The consequence cells carry 51-82% (mining, drink; damage 31-48%) of their own event-frame loss
above sigma_d2 as trained.

## 2. Consequence for the overnight proposal

The N1 training test was proposed only if B showed the N1 signature on F2. It does not, and the map cell of the same
events is drawn from the conditioning under the same noise draws and weights (section 3): an allocation account alone
predicts both parts starved. The N1 overnight run is withdrawn.

Correction (11-05, after F2.1, mine): the declared signature's threshold sigma_d2 = separation / 2 assumes even odds.
A field that ignores the conditioning and holds the event's marginal prior p0 departs at sigma ~ separation /
sqrt(2 |logit p0|) (coal 0.4%: 1.49, below sigma_d2 2.48), which is where the coal / iron HUD departs. So "signature
absent" does not refute N1's mechanism (input read, conditioning unused); only the cell / count split argues against
allocation as the sole cause. Allocation as a multiplier on per-slot exposure is not separated by this run.

Training exposure at F2's budget (`../20261104_dataset_eda/RESULTS.md`, events per 25.6M frames) beside the count's
use of the conditioning (field at sigma 20, F2 / F2.1): wood 682k -> 0.72 / 0.64; drink 414k -> 0.44 / 0.44
(increments); coal 102k -> 0.10 / 0.11; iron 63k -> 0.02 / 0.10; the cell rule, shared by all mined materials (stone
929k + wood + coal + iron + ...), 0.94 / 0.84. Monotone in exposure on four slots; stone (the most exposed count, the
same HUD row as coal) was not measured.

## 3. POST-HOC (after the declared readings; descriptive, `posthoc.py`, `readings/f2/posthoc.json`)

Same rows, first estimate (sample in brackets), threshold 0.5 on each part's f:
- coal: block removed but count unchanged in 86% (84%); both in 11.5% (11%). iron: 68% (65%) vs 1.4%. wood: 21% (26%)
  vs 75% (69%).
- HUD by start count, first call: coal appearance 0 -> 1 0.21 vs increment 0.12; iron 0.07 vs 0.03; drink 0.78 vs
  0.44; wood 0.62 vs 0.71.
- HUD field on real successor + noise vs a conditioning-blind exact two-outcome denoiser (prior = the event's rate over
  all target frames, 0.2-2.7%), sigma 0.7 / 1.2 / 2 / 3.5 / 20:
  iron increments 0.92 / 0.65 / 0.27 / 0.07 / 0.02 vs blind 0.95 / 0.65 / 0.20 / 0.02 / 0.00;
  coal increments 0.92 / 0.65 / 0.37 / 0.20 / 0.10 vs 0.93 / 0.61 / 0.22 / 0.03 / 0.00;
  wood 0.98 / 0.92 / 0.84 / 0.79 / 0.72 vs 0.96 / 0.75 / 0.37 / 0.11 / 0.03;
  map cell of the same coal rows 1.01 / 0.98 / 0.95 / 0.95 / 0.94.

Reading (descriptive): the generator receives the event and uses it for the faced cell; for the coal and iron counts it
behaves almost as if it never saw the event (within ~0.07 of the blind denoiser up to sigma 2 for iron increments);
wood's count uses the conditioning. Open, not tested here: why the count token ignores what the cell token uses
(candidates: per-slot exposure, wood being ~7-12x more frequent than coal / iron, while the cell rule is shared across all
mined materials; the count needs the event from elsewhere on the screen; the digit increment's composition).

## 4. F2.1 (second world, read for agreement)

Run 14:35-14:56 (unit trace-f21; 21 min 43 s, host peak 11.8 GB: within 0.2 GB of the 12G cap, tighter than estimated),
F2.1 ckpt_10 (sha256 de007bf1...), same script, dev rows and seeds. R0 passes (SAFE 0.007); gate: coal passes
(b 0.9998), iron (b 0.889, c 0.758) and drink (b 0.948, c 0.876) fail again; attack not qualified (rule 0.939).

Agreement with F2 on every declared reading's direction: HUD first calls lack the consequence (coal 0.152, iron 0.095,
drink 0.483, wood 0.617), cells carried (coal 0.886, wood 0.984), N1 signature absent (HUD departs at sigma 0.56-1.2),
R3 ~ 0 (<= 0.05), no H1 / H2 pattern (coal b 0.9998 vs c 0.76 / e 0.78; iron e 0.991 vs b 0.889). Difference: the HUD
R4 is "between" for coal (-0.078 +- 0.003), iron (-0.075) and drink (-0.057) where F2 was within 0.05: F2.1's sampler
loses more of an already small first estimate (coal 0.152 -> 0.075, iron 0.095 -> 0.020; F2 0.159 -> 0.114, 0.051 ->
0.022). Damage as in F2 (risky 0.111 -> 0.028). B4 allocation identical to F2 within 0.002 per band (c 3.88).

POST-HOC (`readings/f21/posthoc.json`): coal block removed / count unchanged 85% (F2 86%); iron 65% (68%); wood 26%
(21%). Increments vs the blind denoiser at sigma 0.7 / 1.2 / 2 / 3.5 / 20: iron 0.91 / 0.64 / 0.32 / 0.14 / 0.10 vs
0.94 / 0.65 / 0.22 / 0.02 / 0.00; coal 0.88 / 0.63 / 0.37 / 0.20 / 0.11 vs 0.91 / 0.60 / 0.23 / 0.03 / 0.00; wood count
0.64 at sigma 20; coal cell 0.84 at sigma 20 (F2 0.94). The union corpus did not change the pattern (train-vs-dev
prior: coal / iron not expected to improve).

Descriptive, both worlds: where the first estimate is fractional, the 3-step sampler resolves most rows to "no
consequence" (relative loss coal 28 / 51%, iron 57 / 79%, risky damage 72 / 75%, F2 / F2.1). The declared absolute
R4 thresholds cannot register this for small first estimates (plan defect, mine). It matters after the field's own
deficit, not before it.

## 5. The sampler rounds (POST-HOC analysis + simulation, 11-05; `sampler_rounding.py`)

Per row, both worlds: the production sample's rate equals the fraction of rows whose first estimate exceeds 0.5
(coal HUD F2 P(first > 0.5) 0.115, production 0.114; drink 0.457 / 0.440; risky 0.023 / 0.029; F2.1 alike). Rows with
a first estimate in (0.1, 0.5) produce the consequence 0.6-11% of the time (mean first 0.18-0.35); risky damage: 28-39%
of rows are there (mean first 0.19-0.21) and 0.6-0.9% of them draw damage.
An EXACT two-outcome denoiser under the same schedule, started from the correct marginal (only the step count
varies): p 0.05 / 0.15 / 0.30 / 0.70 / 0.90 -> 3 steps 0.000-0.001 / 0.02-0.03 / 0.15-0.17 / 0.83-0.85 / 0.99;
8 steps 0.03 / 0.12 / 0.28 / 0.72 / 0.93; 32 steps 0.045 / 0.145 / 0.295 / 0.705 / 0.905; 128 steps = p. The first
Euler step (6.67 -> 0.283) lands on the posterior MEAN, a blend; the next call reads the blend as a noisy copy of the
nearer outcome. So with 3 steps the world is a majority-vote decoder of each row's probability: an outcome with
per-row probability < 0.5 (damage under a hidden cooldown: p ~ 0.11, predicted 0.02-0.03, measured 0.028-0.029) is
almost never drawn even by a perfect field, and one > 0.5 is drawn almost always. Churn does not help at 3 steps
(p 0.15: 0.019-0.027 with churn, 0.028-0.050 without). Cross-instrument check: the actor interface's generated first
unlocks agree with the trace's appearance rates (drink 0.86-0.89 vs 0.85; coal 0.24 vs 0.11-0.18; iron 0.05-0.09 vs
0.02-0.04): no readout mismatch on the HUD appearances.

## 6. Stage B2: stone (declared reading, PLAN "Stage B2"; run 15:14-15:22, trace.py sha256 ed314162...)

R0 passes both worlds (recorded count + 1 and PATH in 100%; bias at sigma 0.05 and rerun < 0.01). Primary: the stone
count's field at sigma 20 = 0.632 +- 0.007 (F2), 0.633 +- 0.007 (F2.1): >= 0.6 in both -> **exposure-consistent**
(margin ~4.6 SE). Stone shares coal's HUD row, faced-cell geometry, tool and rule and has increments at every digit,
so neither the route / locality (b) nor the digit composition (c) blocks a count at this exposure; per-slot exposure is
the leading account (not proven: stone's slot also changes by placing / crafting; its look differs).

Descriptive: first call 0.627 / 0.610, production 0.620 / 0.593 (sampler within 0.02: most rows exceed 0.5); start 0
0.78 / 0.78 vs increments 0.59 / 0.57 at the first call (the appearance advantage again: a secondary cost, not a block);
increments at sigma 20 0.58 / 0.59 vs the blind reference 0.04 (coal 0.10 / 0.11); NOOP contrast 0.59; R7 -0.41 / -0.45
(the count's action dependence also needs clean context). Cell 0.74 / 0.70 at sigma 20 (recorded 0.85 / 0.80).
Exposure (events per 25.6M frames) vs the count's conditioning use at sigma 20 (F2 / F2.1): iron 63k 0.02-0.10, coal
102k 0.10-0.11, drink 414k 0.44 (increments), wood 682k 0.70 / 0.62, stone 929k 0.63 / 0.63; the shared cell rule
~1.8M 0.84-0.94. Rising, then flat near 0.6-0.7 for counts: even the most exposed count stays well short of the exact
1.0, so a second limit caps counts (not identified here).

## 7. Stage B3: the sampler's step count (declared reading, PLAN "Stage B3"; 15:51-16:31, both worlds)

R0 passes: 3 steps reproduce the trace's production exactly (max |diff| 0) on every class and part, both worlds; SAFE
production <= 0.016 at every N. Primary, recovery r(32) = production(32 steps) / first call(3 steps), F2 / F2.1:
risky damage 1.07 / 1.09, adjacent damage 1.01 / 0.98, coal HUD 1.16 / 1.10, iron HUD 2.05 / 1.21: all >= 0.75 ->
**the 3-step rounding is the samples' main loss of what the field holds; a sampling fix recovers it.**

| class (F2) | recorded | 3 steps | 8 | 16 | 32 | first call |
|---|---|---|---|---|---|---|
| risky damage | 0.125 | 0.029 | 0.088 | 0.105 | 0.112 | 0.105 |
| adjacent damage | 0.180 | 0.018 | 0.049 | 0.060 | 0.064 | 0.063 |
| coal HUD | 1.0 | 0.114 | 0.166 | 0.180 | 0.185 | 0.159 |
| iron HUD | 1.0 | 0.022 | 0.080 | 0.097 | 0.104 | 0.051 |
| wood / stone HUD | 1.0 | 0.70 / 0.62 | 0.63 / 0.56 | 0.63 / 0.54 | 0.63 / 0.54 | 0.69 / 0.63 |
| coal / iron cell | 1.02 / 0.76 | 0.98 / 0.67 | 0.89 / 0.58 | 0.87 / 0.55 | 0.86 / 0.55 | 0.95 / 0.65 |

Descriptive: with more steps risky damage in samples goes from 23% to 90% of the recorded rate (F2.1 0.028 -> 0.120);
adjacent damage only to its field's value (36%: field-limited; its readiness is not readable from the conditioning, B1
attack best arm 0.58-0.62 vs rule 0.94). Majority outcomes fall below their first call by 0.05-0.1 (the 3-step vote
was inflating them; the field's later calls are not fully self-consistent). 8 steps recovers most (risky 84% of 32
steps' value); cost per generated frame ~N.

## 8. Evening readings (11-05, 17:28-18:33): actor interface at 3 vs 16 steps, swing test, own-history pilot

All instrument checks pass: each 3-step pass reproduces its earlier reading exactly (interface: max |diff| 0 over 104
class x quantity pairs, both worlds); own-history pilot: first step identical across modes, real-replay belief RMS
<= 0.011 (Codex's asserts). One world seed each; per-class seed spread unmeasured (2-10 pp between recipes).

**Actor interface (native reward / termination heads on generated futures; F2, F2.1 agrees):** 3 -> 16 steps, recorded:
- danger: risky damage 0.042 -> 0.115 (rec 0.127; F2.1 0.038 -> 0.124), risky DEATH 0.013 -> 0.062 (rec 0.128: half
  recovered; F2.1 0.063), adjacent damage 0.012 -> 0.056 (rec 0.159: field-limited, F2.1 0.067); all z 19-27.
- stochastic: sapling context reward 0.056 -> 0.142 (rec 0.135: calibrated); random-history damage 0.004 -> 0.027
  (rec 0.020: now 35% OVER), death 0.000 -> 0.003 (rec 0.004); wake 0.40 -> 0.51 (F2.1 0.52 -> 0.60).
- false positives rise with steps: random-history unlocks 0.040 -> 0.048 (rec 0.034; 3 steps already +18%), adjacent
  death 0 -> 0.003 (rec 0). The vote hid false outcomes as well as true minorities.
- deterministic, under-learned: coal 0.24 -> 0.36, iron 0.09 -> 0.19 (head on the real successor 0.99): field-limited.
- majority outcomes fall to the field's own level: skeleton defeat 0.57 -> 0.42 (both worlds), wood -0.05 / -0.06,
  stone -0.06 / -0.12, iron sword -0.06 / -0.11, cow -0.02 / -0.08.

**Swing test (F2.1 - F2 at equal budgets; unpaired SE, conservative):** vanish at 16 steps: zombie defeat +0.070 ->
+0.002, stone +0.045 -> -0.016, iron -0.038 -> +0.001; shrink: cow +0.099 -> +0.040; persist: wake +0.120 -> +0.085
(z 4.2), wood -0.06 -> -0.07, skeleton -0.06 -> -0.06; grows: iron sword -0.05 -> -0.105. Several F2.1 "gains" were the
3-step vote amplifying small field differences; worlds must be compared at >= 16 steps (and on fields).

**Own-history pilot (Codex's recursion.py, F2, 128 exogenous windows x 4 futures, same seeds):** mean visible
entities, free running, steps 1 / 4 / 8 / 16: 3 steps 0.72 / 0.37 / 0.17 / 0.10 (collapse, as on F1.5); 16 steps
1.13 / 1.20 / 1.31 / 1.49 (reader on the real percepts 1.08 / 1.21 / 1.28 / 1.23). The collapse is almost entirely
the 3-step vote compounding through the history. S2.1 count gate (mean over 16 steps, > 30% and > 3 SE): 3 steps 7 of
8 kind x mode checks fire (free: zombies 0.27x, cows 0.17x, skeletons 0.07x; real history: cows 0.59x, skeletons 0.34x,
arrows 3.06x); 16 steps 2 of 8 (free zombies 1.34x, rising 0.18 -> 0.47 vs real 0.15 -> 0.31; real-history arrows
1.88x). Not measured here (user: necessary, not sufficient): persistence, identity, positions; deaths are not
calibratable on these survival-selected windows (free running samples 2.7% deaths by step 16, recorded 0%).
