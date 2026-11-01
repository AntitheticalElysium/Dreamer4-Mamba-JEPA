# Scale-weighted TSP from initialization against P1-F: results (11-01)

Run: unit saccade-beta-tsp, commit a21ebce9, 00:08-09:32, 25.6 M frames, clean exit; checkpoint 10 by P1-F's rule
(monitor fell at every checkpoint). Readings as declared in PLAN.md, run by `queue.sh` (eab9a890) with P1-F re-run
beside this run on the same rows and draws (`beta` / `beta10` / `beta_1500` = this run; `p1f` / `p1f10` / `base` =
P1-F checkpoint 10). Summaries: `readings/codex_summary.json` (`readings_summary.py`, Codex's own estimators),
`readings/eval_m05/readings_beta_1500-base.{json,txt}`, `readings/interface_{random,guided}.json`, `run/s1/results.json`,
`readings/render_beta.json`. One seed; descriptive, no gate.

## Instrument checks
- Interface (random gaze), P1-F's re-run against its 10-31 output, 4,552 numbers: recorded quantities identical; the
  model's readings within 0.0012 (head on real) and 0.0023 (generated) absolute: GPU evaluation is not bitwise
  deterministic, below the SEs.
- Codex's noise panel on P1-F checkpoint 10 equals Codex's earlier run on all 32 scenes (sensor variance, all groups).
- Codex's sampler factorial on P1-F equals Codex's `sampler_followup/results.json` (Euler-3 / churn 1 and Euler-8 /
  churn 0 counts identical to 4 digits).

## 1. Manipulation: renderer static in the periphery tokens (Codex's fixed-world re-render panel)
| | P1-F | beta | paired difference [95% CI] |
|---|---|---|---|
| night periphery token variance under renderer noise | 0.0482 | 0.0402 | -0.0080 [-0.0093, -0.0065] (ratio 0.83) |
| day | 0 | 0 | (the renderer adds no static by day) |
| night glimpse token variance | 0.0011 | 0.0012 | +0.0001 [0.0001, 0.0001] |
Noise fell by 17% at night (Codex's 256-update fork: 13.8%). The manipulation worked, partially.

## 2. Evidence retained (grounding)
- HUD: every digit and icon rendered on both paths (1.000, as P1-F); real crops decoded correctly at the rule's crop
  and the other covering crop (1.0 / 1.0). TSP's HUD error halved (MAE drink 1.60 -> 0.47, health 1.98 -> 0.86).
- Common map content on TSP's path sharper (grass 1.10 -> 0.80, water 2.00 -> 1.06, stone 1.77 -> 1.08, tree 2.31 -> 1.46).
- **Rare content worse on both paths**: TSP coal 3.7 -> 13.8, iron 7.8 -> 22.8, diamond 27 -> 37, skeletons 13.1 -> 27.0,
  arrows 17.6 -> 19.1; reconstruction (still faithful) coal 2.6 -> 5.3, iron 4.1 -> 12.6, skeletons 13.8 -> 21.3, cows
  4.4 -> 6.4, zombies 5.4 -> 6.9 (they share the pixel decoder and the grounding weight). This is the documented risk of
  uncertainty-weighted losses (Seitzer et al. 2022: hard, rare examples are down-weighted with the noise).
- S1.1 perception unchanged (day recall 1.0 both; night presence recall 0.967 / 0.966, arrows 0.854 / 0.849).

## 3. Physical generation
| reading (beta vs P1-F) | P1-F | beta | |
|---|---|---|---|
| S1.3 C2ST (0.5 = indistinguishable) | 0.960 | 0.973 | worse; power controls equal (shuffled 0.975 / 0.978, frozen time 0.914 / 0.914, mismatched 0.941 / 0.931) |
| generated zombies / cows / skeletons / arrows (S1.3; real 0.42 / 0.47 / 0.16 / 0.09) | 0.224 / 0.281 / 0.052 / 0.132 | 0.306 / 0.197 / 0.049 / 0.159 | zombies better, cows and arrows worse |
| sampler factorial, production Euler-3 / churn 1 (real 0.50 / 0.48-0.50 / 0.13-0.14 / 0.074) | 0.344 / 0.287 / 0.051 / 0.111 | 0.430 / 0.201 / 0.051 / 0.143 | same pattern |
| Euler-8 / churn 0 (secondary) | 0.480 / 0.447 / 0.162 / 0.141 | 0.555 / 0.432 / 0.137 / 0.123 | |
| Euler-32 / churn 1 (secondary) | 0.482 / 0.490 / 0.182 / 0.098 | 0.572 / 0.434 / 0.176 / 0.064 | |
| movement shift, generated (real-token reference) | 0.865 (0.862) | 0.874 (0.874) | both at their own reference; M0.5 panel paired +0.0075 [0.005, 0.010]; Codex panel residual +0.004 [-0.0004, 0.0086] |
| night panel generation, mean mismatch (own token space) | 0.061 | 0.050 | -0.011 [-0.028, -0.001]; different token spaces, not a physical claim |
| M0.5 entity gap, generated minus real (zombies / cows / arrows) | -0.19 / -0.19 / +0.04 | -0.11 / -0.27 / +0.06 | |

## 4. Consequences and the actor interface
M0.5's evaluator (beta minus P1-F, same rows): under the rule's gaze the generated glimpse draws drink increments
+0.10 [0.08, 0.13] and tables +0.11 [0.08, 0.13] more often (wood appearance -0.02, increment +0.02); the periphery
draws placed tables +0.28 [0.25, 0.31] more often under random gaze; HUD consequences in the periphery under random
gaze stay at 0 in both (the gate). False changes on failed attempts rise (table +0.019 guided, +0.030 periphery; drink
+0.023 periphery). Codex's crop test (64 contexts, 16 episodes; wide CIs): rule crop wood 0.98 / 0.97, drink 0.84 / 0.94;
the other covering crop wood 0.14 / 0.14, drink 0.14 / 0.08; NOOP false increases 0.016 -> 0. The gate persists, as
the declared expectation said (the rule is kept).

Actor interface, generated rates, beta minus P1-F (paired, same draws; * = CI excludes 0; per-achievement rows are
outcome-selected recall diagnostics, as declared):
| | random gaze | guided gaze |
|---|---|---|
| random histories, P(unlock) (recorded 0.034; P1-F 0.044) | -0.005* (0.039) | -0.004* |
| death, risky (recorded 12.8%; P1-F 1.2% / 7.9%) | +0.002 | -0.029* (5.0%) |
| damage, risky (12.7%; P1-F 2.6% / 11.9%) | -0.006* | -0.040* |
| damage, adjacent (15.9%) | +0.000 | -0.011* |
| sapling context (13.5%) | -0.007 | -0.034* |
| place table | **-0.468*** (head on the real successor 0.79 -> 0.33) | +0.015* |
| wake up | **-0.334*** | **-0.312*** |
| defeat zombie / eat cow | -0.010 / -0.022 | -0.159* / -0.139* |
| defeat skeleton | -0.058* | -0.071* |
| collect stone / coal / iron | +0.036* / -0.042* / -0.014* | -0.026 / -0.056* / -0.016* |
| iron pickaxe / place plant | -0.094* / -0.072* | -0.125* / -0.050* |
| first wood / first drink | -0.007 / -0.000 | +0.004 / +0.003 |

## Declared reading (OVERNIGHT_PROPOSAL.md)
Noise down (night static -17%), the HUD evidence kept, but physical prediction not improved and at the actor
interface mostly worse: the declared "noise decreases but generation does not improve: reject noise filtering as a
sufficient fix at this setting". Two further costs the proposal named as risks: rare blocks and mobs lose grounding
on both pixel paths (the scale weighting down-weights hard pixels, Seitzer et al.), and the table reward readout
under random gaze falls 0.79 -> 0.33 on REAL successors (a stronger view dependence for one rule class). Gains: drink
increments and tables under the rule's gaze, zombie counts, movement at its reference, random-history unlock
calibration nearer the record. P1-F remains the better recipe of the two for P1; scale-weighted TSP is not adopted.
What this does not establish: that faithful TSP caused P1-F's regressions against world 1 (no faithful-vs-NLL run
from scratch); anything across world seeds; anything about a learned chooser.
