# P1-F3 results (11-06; F2 + physical G1 as x-space loss x 3.425, glimpse EDM)

Run: launched 11-05 22:41 AEDT (saccade-p1f3), training complete 11-06 10:13, automatic readout complete 11:49 (13 h 08
wall, host peak 21.4 GB under the 24 GB cap). Checkpoint 10 selected (monitor decreasing at every checkpoint; sha256
9695c6d6...). Every number below is from `readings/` (instruments of `../20261105_consequence_trace/`) against F2's
pinned references; verdicts are `compare.py`'s (`readings/comparison.json`). Single world seed. SEs are unpaired episode
bootstraps (conservative) unless marked paired.

## 1. Declared verdicts (PLAN.md)

| reading | F2 | F3 | F3 - F2 | declared rule | verdict |
|---|---|---|---|---|---|
| coal HUD field, sigma 20 | 0.124 | 0.155 | +0.030 +- 0.007 | >= 0.25 and beyond 3 SE | lever **partial** |
| iron HUD field, sigma 20 | 0.046 | 0.089 | +0.043 +- 0.005 | >= 0.15 and beyond 3 SE | (both beyond 3 SE, both under threshold) |
| wood / stone HUD field, sigma 20 | 0.704 / 0.632 | 0.842 / 0.807 | +0.138 / +0.174 | both >= 0.80 | plateau **signal-limited** |
| drink HUD field, sigma 20 (descr.) | 0.446 | 0.643 | +0.197 +- 0.012 | | |
| first coal, 16 steps | 0.364 | 0.444 | +0.080 +- 0.025 | +0.05 beyond 3 SE | not improved (z 3.1) |
| first iron, 16 steps | 0.193 | 0.240 | +0.047 +- 0.019 | " | not improved (z 2.5) |
| risky death, 16 steps (rec 0.128) | 0.062 | 0.088 | +0.026 +- 0.005 | " | not improved (z 5.4) |
| adjacent damage, 16 steps (rec 0.159) | 0.056 | 0.097 | +0.040 +- 0.005 | " | not improved (z 8.2) |
| risky damage, 16 steps (rec 0.127) | 0.115 | 0.110 | -0.005 +- 0.006 | " | unchanged |
| sapling context vs recorded 0.135 | +0.007 | +0.033 | | within 0.02 | **BREACH** (0.168 +- 0.005 vs 0.135 +- 0.007) |
| low-noise bias, max over class/part | 0.0018 | 0.0085 | | < 0.01 | pass (4.7x F2: drink HUD -0.0085, stone cell -0.0079) |
| S1.1 presence, worst F3 - F2 | | -0.005 | | >= -0.02 | pass |
| S1.4 TSP nats/pixel | 2.209 | 2.218 | +0.4% | <= +5% | pass |
| common classes, 16 steps | | | none down; wood +0.109, stone +0.126 | not down > 0.05 beyond 3 SE | pass |
| false rewards (random histories), 16 steps (rec 0.034) | 0.048 | 0.045 | -0.003 +- 0.004 | not up > 0.01 | pass |
| own-history count gate, 16 steps | 2 / 8 | 0 / 8 | | <= 2 | pass, but a threshold effect (section 4) |

## 2. Fields (consequence trace, first call / field across sigma)

- The weighting raised the field's use of the conditioning on every consequence the trace measures, in the direction
  N1 predicts. HUD field at sigma 20: coal 0.124 -> 0.155, iron 0.046 -> 0.089, wood 0.70 -> 0.84, stone 0.63 -> 0.81,
  drink 0.45 -> 0.64; damage: adjacent 0.050 -> 0.128 (recorded 0.180), risky 0.064 -> 0.091 (0.125). Adjacent damage's
  field is now nearly flat across sigma (0.18 at 0.37 -> 0.12-0.13 above 3.5; F2 0.18 -> 0.05).
- Cells are now conditioning-driven at every sigma: coal cell 1.02 flat from 0.37 to 20 (F2 0.94-0.97), wood 1.05,
  stone 0.84-0.86 (F2 0.74-0.85). Iron cell unchanged at ~0.60 in all three worlds (its reader on recorded frames reads
  0.76: partly reader-limited).
- Appearance vs increment (posthoc, HUD field at sigma 20, F2 -> F3): coal appearance (0 -> 1) 0.163 -> 0.229, coal
  increments (> 0 -> +1) 0.096 -> **0.100 (unchanged)**; iron 0.065 -> 0.096 / 0.022 -> 0.080 (F2.1, by data alone:
  0.090 / 0.096); drink 0.73 -> 0.73 / 0.41 -> 0.63; stone 0.83 -> 0.89 / 0.58 -> 0.79; wood 0.65 -> 0.78 / 0.72 -> 0.87.
  First estimates: coal "block gone, count unchanged" 0.86 -> 0.90 (the cell/count split of RESULTS.md 3 persists).
- Lift is roughly multiplicative across slots (F3 / F2 at sigma 20: wood 1.20, coal 1.25, stone 1.28, drink 1.44, iron
  1.92); the exposure ordering is preserved. The rare counts stay near the floor because a factor on a near-zero use
  is still near zero: the weighting is not an exposure substitute.
- Action routes: the coal cell's DO-NOOP contrast 0.83 -> 0.995, wood HUD 0.69 -> 0.83. Damage now depends on the action
  much more strongly: NOOP at t - 1 raises adjacent damage by 0.216 over the recorded action (F2 0.034), risky by 0.118
  (F2 0.072); context noised to sigma 5 removes most of it (+0.164 / +0.102). Direction plausible (standing next to a
  zombie gets you hit); its size is not verified against the simulator.
- Sampler steps (production, 3 / 32 steps): risky damage 0.048 / 0.125 (recorded 0.125; F2 0.029 / 0.112), adjacent
  0.073 / 0.106 (0.180; F2 0.018 / 0.064), coal HUD 0.102 / 0.202, iron 0.039 / 0.150. F2's coal cell degraded with
  steps (0.975 -> 0.858); F3's holds (1.018 -> 0.995). Common counts fall with steps in both worlds (stone 0.82 -> 0.73):
  more steps sample the field's own uncertainty instead of the 3-step majority vote.
- F3's actual allocation on its own errors (`f3_allocation.json`): glimpse 22.66%, sigma < 0.3 18.99% (F2-calibrated
  prediction 18.83%), teaching share x3.3-3.6 over all-EDM on the same errors (x2.8-3.1 on F2's errors).
- Monitor (EDM units, unchanged): final 0.03055 vs F2 0.03022 (+1.1%). By level: sigma 0.106 +13%, 0.231 +3%, 0.373 0%,
  0.555 -3%, 0.81 -6%, 1.2 -8%, 1.94 -9%, 4.22 -8%: the low-noise / high-noise trade the weighting buys.

## 3. Actor interface at 16 steps (native heads on generated futures)

Broad lift: 20 of 22 achievement classes up, 9 readings beyond 3 SE (wood +0.109, stone +0.126, table +0.036, zombie
defeat +0.070, stone pickaxe +0.076, furnace +0.071, coal +0.080, risky death +0.026, adjacent damage +0.040). Head
ceilings (heads on recorded futures) unchanged within 0.03. Share of the ceiling reached (gen / rec_head), F2 -> F3:
adjacent damage 0.35 -> 0.61, risky death 0.48 -> 0.69, coal 0.37 -> 0.45, iron 0.20 -> 0.24, zombie 0.35 -> 0.44, cow
0.23 -> 0.29, skeleton 0.65 -> 0.72, eat plant 0.50 -> 0.48.

Regressions: **WAKE_UP 0.510 -> 0.439 (-0.071 +- 0.020, z -3.5)**, not a declared guard (F2.1 had +0.085 at 16 steps);
**sapling context over-produced** (0.168 vs 0.135; F2 0.142). Both are hidden-state / stochastic mechanics (wake via the
hidden fatigue timer; sapling 10% per DO on grass); neither is covered by the trace.

## 4. Own-history (16 steps, 128 exogenous windows x 4, paired windows)

No change beyond noise. The count gate's 2 -> 0 is a threshold effect: free zombies 1.339x -> 1.355x (F2 z 3.2, F3 z
2.8); real-history arrows 1.88x -> 1.50x (paired excess -0.014 +- 0.005: the one real gain); free cows drift up (0.945 vs
0.797 real at step 16); free total excess at step 16 0.26 -> 0.39 entities (paired +0.135 +- 0.073, z 1.8). Positive
reward per step on real history 0.065 -> 0.057 vs truth 0.017 (3.4x; free running 0.022 both): pre-existing, unchanged.

## 5. Other instruments

- Glimpse check: glimpse field MSE lower at every sigma (0.05: 4.7e-4 vs 5.4e-4; 1.0: 3.0e-3 vs 3.4e-3; 20: 5.7e-3 vs
  6.4e-3) although its recipe is unchanged; still beats the window probe (no deficit).
- S1: perception unchanged; at 3 steps generated reader counts up slightly (cows 0.237 -> 0.261, zombies 0.286 -> 0.317;
  real 0.47 / 0.42); C2ST on generated frames 0.968 -> 0.978 (no SE; not gaze-invariant).
- Training: tsp, rec, reward, cont within ~1% of F2 throughout; speed identical; G1's gradient into the encoder ~3-5x F2's
  (median of 10 snapshots 0.16 vs 0.03; noisy).

## 6. Reading

1. N1 is a real lever and broader than its target: every conditional consequence the trace measures moved the predicted
   way; the wood / stone plateau was a teaching-signal cap. Damage gained most at the actor interface.
2. It does not fix the rare counts. Coal increments did not move; iron's gain matches what F2.1's data change gave. Two
   explanations remain for coal: (a) examples: the corpus lacks per-transition exposure; (b) still a learning-signal or
   route deficit (F2 did not fit first coal even on training histories seen 10.5 times, 11-04). Separating control: the
   11-04 train-vs-dev instrument on F3: fits training coal but not dev -> (a); fails on training too -> (b).
3. Costs: low-noise fidelity (monitor +13% at 0.106; bias 4.7x, under the guard) and two actor regressions (wake,
   sapling) on hidden / stochastic mechanics. Single seed: field effects (z 4-19, one direction, mechanism-consistent)
   are safe; actor differences of 0.03-0.07 are on the scale of F2 vs F2.1 swings and need a second seed before they
   are read as properties of the weighting.
4. Own-history drift is unchanged (P2's job).

## 7. Post-hoc 11-06 (existing outputs only): false rewards are premature wake-ups; damage ignores the cooldown

- Own-history real-history ("teacher") false rewards (`posthoc_wake.py`, `readings/posthoc_wake.json`): 87% (F2 90%) of
  the false positive-reward mass falls on steps where the agent is asleep and stays asleep. There P(reward > 0.5) is
  0.42 (F2 0.50) while WAKE_UP is not yet unlocked and 0.00001 once it is: the head's first-time bookkeeping is right;
  the false reward is a WAKE_UP. Its dependence on the visible energy digit is graded, the rule is a step (Craftax
  Classic: asleep, actions become NOOP; wake at energy >= 9, ~100% in the labels, or when hit, 1-2%): world P per
  step at energy 6 / 7 / 8 = 0.00 / 0.16 / 0.20 (F2 0.00 / 0.19 / 0.24); on true wakes 0.26 (F2 0.31) vs 0.52 rewarded.
  Not action-driven (every ignored action 0.08-0.28). Free running shows fewer (0.017) because a falsely woken
  agent stays awake; the real history keeps re-offering the sleeping frame. Training data is not short: ~20k wakes
  at energy 9 and ~197k asleep-at-energy-8 transitions in the train split (x ~10 epochs). The WAKE_UP regression of
  section 3 is the same mechanism re-calibrated (fewer wakes, true and false), not a separate failure. Open: whether
  the generated frame shows the agent awake (world) or the head reads a wake from an asleep frame (head).
  CORRECTION 11-06 (`../20261106_determinant_curves/RESULTS.md` 2): "world P per step" above is the HEAD's P(reward >
  0.5) on generated futures, not the world's wake rate. Read by a frame reader, the world wakes the agent in ~50% of
  futures at every energy (flat; 10x the training rate); the energy grading is the head's. The open question is
  answered: WORLD-DRAWN (every head-reported wake is an awake frame).
- Damage by the hidden zombie cooldown (trace's cooldown strata, adjacent damage, first call): recorded ready 0.314 /
  not ready 0.018 (18x); F2 0.049 / 0.080 (blind, inverted); F3 0.145 / 0.089 (1.6x). The trace's attack readers:
  belief AP 0.64 vs rule inputs 0.94: the belief carries the cooldown only partly (history-inferable: <= 5 steps).
