# M0.5 arm G: results (10-30)

Arm G: P1's joint update from checkpoint 10, Q=1 guided placement, 1,500 updates (log 646af765, code eb77f348).
Evaluation: PLAN.md's declared readouts on base, guided_0300, guided_0750, guided_1500 (`eval/results.json`, 48 min);
readings by `readings.py` (`eval/readings_*-base.json`). Every comparison here is guided minus base: continued joint
training at the re-warmed learning rate and the placement are NOT separated (arm R does that; not run, decided by
hand after this dissection). The intervals cover evaluation rows only, not training noise.

## Training

Clean run: guided share 0.13, cumulative coverage 0.952, denoiser update norm 0.007-0.044 per update (0.042
predicted at 5.8e-5), peak 3.8 GiB, 5.2 s/update. Gradient norm after warm-up 19-129, median 70 (P1's last 625
updates, learning rate 1.0-1.2e-5: 8-44, median 17); clip factor 0.008-0.052. P1's monitor G1 (held-out, its own placements) rose 0.01908 -> 0.01937 / 0.01945 /
0.01977 at 300 / 750 / 1,500 (+3.6%, about P1's checkpoint 8 level), in every noise level.

## Primary: the generated glimpse at the consequence (guided route)

Verified success over the input-defined population (unknown share in brackets); reading = guided_1500 - base.

| class | base | 300 | 750 | 1500 | 1500 - base [95%] | reading |
|---|---|---|---|---|---|---|
| drink appearance (0->1) | 0.064 (0.11) | 0.172 (0.10) | **0.422** (0.09) | 0.191 (0.14) | +0.127 [+0.085, +0.170] | DETECTED, unknown-heavy |
| wood appearance (0->1) | 0.123 (0.19) | 0.235 (0.23) | **0.430** (0.21) | 0.373 (0.27) | +0.250 [+0.207, +0.293] | POSITIVE, unknown-heavy |
| drink increment | 0.038 | 0.028 | 0.033 | 0.044 | +0.006 [-0.003, +0.014] | BOUNDED NULL |
| wood increment | 0.083 | 0.087 | 0.104 | 0.101 | +0.018 [+0.004, +0.033] | DETECTED |
| table placed | 0.001 | 0.002 | 0.002 | 0.009 | +0.008 [+0.004, +0.011] | DETECTED |

MEI (0.25 x the base gap to the recorded rate) is 0.22-0.25 for every class. At 750 both appearances read POSITIVE
(drink +0.358, wood +0.307). Bounds at 1500: drink [0.19, 0.33], wood [0.37, 0.65] (base [0.06, 0.17], [0.12, 0.31]).
The instrument holds on the same rows' real tokens (0.99-1.00 correct against the observed value) and every reader
stays qualified at every checkpoint.

**Where the copy mass went.** Appearance rows left copy (drink 0.57 -> 0.10 at 750 -> 0.26 at 1500; wood 0.63 -> 0.21
-> 0.24) for the right value AND for other values: drink "other" rose 0.26 -> 0.30 -> 0.38 -> 0.42 (wood 0.06 -> 0.12
-> 0.15 -> 0.11). Post-hoc description of the saved per-row readings (`eval/*_guided_*.npz`, counts of read - start over eligible
readable samples): these other values are
far jumps, not off-by-one. Drink 0->1 is read as 4, 5 or 8 (base 43 of 182 readable samples, 1500 60 of 176); on
failed drink attempts (start 9 in all 564 samples) the glimpse is read as 2 or 3 in 278 of 558 at base and 228 at
1500; on risky damage rows (health 1-2) it is read as 8-9 in 1,652 of 2,208 at base and 377 of 2,205 at 1500. The
periphery route on the same classes has little of this (drink appearance other 0.03-0.12, failed drink false change
0.09-0.12). Two readings fit and the evaluation cannot separate them: the glimpse generates the wrong digit, or a
degraded generated token falls inside a wrong value's support and the reader misreads it.

**Not changed by placement.** Increments stay at copy (drink 0.64-0.66, wood 0.74-0.76). The glimpse almost never
generates a placed table (copy 0.99), although the periphery route's generated cell (faced cell outside the window)
shows it on 0.22-0.24 of its rows (0.12 at 750). Failed attempts:
guided false change NO CHANGE (drink 0.66 -> 0.67, table ~0). Stochastic classes (calibration): sapling 0 generated
against 0.092 recorded at every checkpoint; adjacent damage 0.072 -> 0.035 against 0.22 recorded; risky ~0 against
0.21, its "other" falling 0.90 -> 0.50 toward copy.

## Secondary: the generated periphery cell (periphery route)

drink appearance DETECTED +0.049 [+0.009, +0.090] (0.132 -> 0.181; 0.291 at 750); wood appearance DETECTED +0.077
[+0.050, +0.105] (0.164 -> 0.242; 0.348 at 750); both unknown-heavy (0.23-0.40). Increments BOUNDED NULL. Table
NEGATIVE -0.018 [-0.028, -0.007] (0.242 -> 0.224), and -0.123 at 750 (0.119). Failed attempts LESS false change.

## Monitored

- Perception, real frames at the consequence cell: BETTER at 1500 for every class on the cell's reconstruction and TSP
  NLL; largest on tables (rec 7.10 -> 4.24 nats/pixel, TSP 8.19 -> 4.77, monotone over checkpoints) and wood
  increments (TSP 1.92 -> 1.68). Window means mostly BETTER; sapling TSP window WORSE (0.650 -> 0.663).
- Action routing (diagnostic, not memory): MORE for drink / wood appearance, wood increment and table; with the NOOP
  belief appearance success is 0.000-0.002 at every checkpoint, so every appearance gain goes through the latest
  action.
- Belief content (S1.2 probe, seed 0, base vs 1500): departed-entity AP within 0.005 in every kind and stratum, HUD
  exact mean 0.767 -> 0.768; achievement AP within 0.004 except MAKE_IRON_PICKAXE 0.952 -> 0.962 and COLLECT_DIAMOND
  0.47 -> 0.59 (rare labels, one probe seed).
- Regression panel (1500 vs base): dev TSP NLL 2.177 -> 2.171 (750: 2.199), reward NLL 0.025 -> 0.024, continuation
  0; movement shift 0.910 -> 0.915; entity counts (generated vs real) zombies 0.23 -> 0.27 (0.43), cows 0.25 -> 0.27
  (0.47), skeletons 0.14 -> 0.12 (0.16), arrows 0.13 -> 0.17 (0.09, over-generation grew); no reader lost
  qualification.

## Limits of these numbers

- The trajectory is not monotone. 750 is the best checkpoint for drink and wood appearance in both routes and the
  worst for the periphery's tables and dev TSP; drink appearance falls 0.42 -> 0.19 from 750 to 1500, more than the
  width of the 1500 - base interval. At a constant learning rate the checkpoints differ by training noise that no
  interval here covers, so a single checkpoint's reading is fragile.
- Placement and continued training are confounded (arm R). So is the G1 rise: the learning-rate re-warm against the
  shifted fixation distribution.
- The glimpse "other" readings are post-hoc descriptions of saved arrays, not a declared readout; their nature
  (generation vs instrument) is open.
- One world seed, one base checkpoint, labels as the placement rule.
