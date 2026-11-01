# Generation diagnosis: results (2026-10-29)

Frozen P1 world seed 1, checkpoint 10 (sha256 `74eccc43...`), dev split. Plan: PLAN.md v2 with its amendment to test A
(made after a 60-episode pilot, recorded before the full run). Episode-bootstrap SE in parentheses.

## A. The field's conditional mean on context classes (`field_calibration.py`, `field_calibration/results.json`)
Unit: the HUD cell token projected so that the mean real token of the starting value is 0 and that of the
consequence value is 1 (health h -> h - 2, or dead; drink d -> d + 1). bias(sigma) = mean f(D(x0 + sigma eps)) - mean
f(x0); zero at every sigma for the minimum-MSE denoiser on classes defined by the conditioning.

Instrument checks (model-independent, amendment): |bias| at sigma 0.05 <= 0.006 in every class; DRINK reaches d + 1
in 100.0% of rows and SAFE keeps 9 in 99.97%; SAFE bias <= 0.022 at every sigma <= 6.67 (0.031 at 10-20); B's exact
field gives |bias| <= 0.01. All pass.

| Class (contexts) | Recorded f | Field mean at sigma 0.7 / 2 / 6.67 / 20 | Production samples f |
|---|---|---|---|
| RISKY: health <= 2, zombie at L1 1 (3,805) | 0.127 (0.004) | 0.125 / 0.067 / 0.025 / 0.026 | 0.0018 (0.0003) |
| ADJACENT: health >= 3, zombie at L1 1, awake (3,000) | 0.189 (0.007) | 0.097 / 0.082 / 0.086 / 0.100 | 0.059 (0.002) |
| DRINK: facing water, DO, awake (3,000) | 1.002 (0.000) | 0.596 / 0.292 / 0.178 / 0.150 | 0.131 (0.004) |
| SAFE: health 9, no threat (3,000) | 0.004 | 0.016 / 0.021 / 0.025 / 0.035 | 0.011 |
| RANDOM (3,000) | 0.022 (0.003) | 0.026 / 0.032 / 0.033 / 0.045 | 0.018 (0.001) |

Bias SEs at sigma >= 2: 0.003-0.007. By the declared outcome rule (controls pass; RISKY and ADJACENT bias below -0.05
beyond 3 SE at sigma 2-6.67): the trained field's conditional mean underweights damage: about 5x in RISKY (0.025 vs
0.127), 2.2x in ADJACENT. It also misses a DETERMINISTIC visible consequence: drinking (+1 always, game_logic
do_action) is carried at 0.18 at sigma 6.67 and is already pulled toward "no change" at sigma 0.2 (bias -0.075) and
0.7 (-0.41), levels well inside the training distribution. The DRINK class is a function of the conditioning: the
player's facing direction is read exactly from the 3 x 3-pooled player cell (1-NN accuracy 1.000 day and night,
chance 0.29) and from its learned periphery token (1.000), the faced water is visible and a_{t-1} is conditioned on.

Secondary split by the simulator's cooldown readiness (not a function of the conditioning, so descriptive): the field
is insensitive to it, the wrong way round in ADJACENT. RISKY ready: recorded 0.389, field 0.027; not ready: 0.004,
field 0.024. ADJACENT ready: recorded 0.326, field 0.039; not ready (a zombie that just attacked): 0.034, field 0.138.
The hidden cooldown is largely inferable from the context (a zombie that attacked within the last 5 steps left a
visible HUD drop inside the 8 context frames); the field instead predicts more damage right after damage.

C (native heads on denoised-real successors, sigma 0.05 / 0.2 / 0.377): RISKY death probability 0.1277 / 0.1277 /
0.1276 vs 0.1277 recorded; the heads read model-output geometry correctly where the content is preserved. Production
samples: death 0.0001 (RISKY).

Geometry (median per-token inverse-tanh variance): recorded 0.982 (RISKY); denoised-real falls with sigma (0.981 at
0.05, 0.970 at 5, 0.967 at 6.67, 0.956 at 20); production samples 0.970. I first read the match with the clean
estimate at sigma 5-6.67 as "the sample is essentially that estimate"; D does not support it (more steps move the
statistic further from the real value, see D).

Not established: which property of training produces the underweighting (capacity, optimisation length, noise-level
coverage, loss weighting or the consequence's share of the per-element loss: the drink cell is 16 of 1,880 scalars
in ~1% of frames); joint correctness; per-history calibration.

## B. The sampling algorithms on an exact field (`exact_field.py`, `exact_field.json`)
Exact posterior of a two-outcome mixture of real token means (k cells), 20,000 runs. Checks: every many-step
algorithm started from the TRUE noised mixture returns the weight (e.g. health 2 -> dead, p 0.15: Heun 32 0.152,
churn 32 0.152, Heun 64 at sigma_max 19 0.151); the exact field's tower bias <= 0.01 at every sigma.

| Carrier (separation), k = 1 | p = 0.03 | 0.10 | 0.15 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| health 2 -> dead (5.66): production | 0.0007 | 0.019 | 0.050 | 0.215 | 0.502 |
| health 5 -> 3 (1.91): production | 0.0027 | 0.018 | 0.036 | 0.170 | 0.485 |
| drink 4 -> 5 (2.21): production | 0.0013 | 0.013 | 0.034 | 0.177 | 0.505 |
| health 2 -> dead: 3 Euler steps from the true start | 0.0003 | 0.010 | 0.034 | 0.185 | 0.500 |
| health 2 -> dead: Heun 32 from the production start | 0.083 | 0.185 | 0.239 | 0.361 | 0.503 |

- The production algorithm suppresses minority outcomes and inflates majorities even with a perfect field: 3x at
  p 0.15, 10-40x at p 0.03. The 3-step discretisation causes it (the true start does not help); a larger footprint
  helps only for well-separated carriers (health 2 -> dead, k = 9: 0.03 -> 0.094; health 5 -> 3, k = 9: 0.03 -> 0.0002).
- The production start (sigma_max * eps at sigma_max 5) is itself biased for these geometries: with many steps it
  OVER-produces the rare outcome (0.15 -> 0.24). sigma_max 5 is not >> the token scale (EDM sec. 2's condition).
- Consistency with A: the field carries RISKY damage at ~0.025; B's production algorithm turns p 0.03 into
  0.0007-0.005 (k = 1-2); the production samples show 0.0018. Descriptive agreement, not an additive split.

## D. Inference settings on the trained field (`solver_sensitivity.py`, `solver_sensitivity/results.json`)
A's contexts (first rows per class), 2 futures per context per algorithm, native heads, A's unit and geometry.

| Class (contexts) | Recorded f / field mean (A) | Production | Heun 8 / 16 / 32 | Heun 32 + EDM churn | Heun 32, sigma_max 10 |
|---|---|---|---|---|---|
| RISKY (1,500) | 0.127 / 0.025 | 0.002 (0.001) | 0.024 / 0.019 / 0.022 (0.003) | 0.008 | 0.015 |
| ADJACENT (1,000) | 0.189 / 0.086 | 0.066 (0.005) | 0.102 / 0.102 / 0.104 (0.008) | 0.077 | 0.091 |
| DRINK (1,000) | 1.002 / 0.178 | 0.143 (0.007) | 0.226 / 0.227 / 0.234 (0.009) | 0.202 | 0.209 |
| SAFE (1,000) | 0.004 / 0.025 | 0.011 | 0.015 / 0.013 / 0.019 | 0.003 | 0.011 |
| RANDOM (1,500) | 0.022 / 0.033 | 0.019 | 0.036 / 0.035 / 0.029 | 0.018 | 0.026 |

Native death head (recorded RISKY 12.77%, SAFE 0%, RANDOM 0.20%): production 0.000% in every class; Heun 32: RISKY
1.57%, SAFE 0.37%, RANDOM 0.41%; churn: 0.56% / 0% / 0.07%. Geometry statistic (real 0.982-0.991): production
0.973-0.980; Heun 0.958-0.969; churn 0.981-0.985; sigma_max 10 0.972-0.977.

- Inference settings matter for this checkpoint: resolved deterministic sampling raises RISKY damage ~10x over
  production, to about the field's own conditional mean (0.022 vs 0.025), and no setting exceeds it. The field bounds
  every tested sampler: RISKY stays 6x below the recorded and DRINK 4x below with the best one.
- Resolution also creates deaths where none occur (SAFE 0.37% under Heun 32) and separates RISKY from SAFE only
  ~4x (recorded: 12.8% vs 0): what the extra samples carry is not the rule.
- The token-geometry mismatch is NOT a by-product of unresolved sampling: more steps move it further from the real
  value; only stochastic churn (EDM's Langevin correction) restores it, and churn lowers the consequences again. The
  geometry is a property of how this field is integrated, not a symptom of the missing rare events.
- B's warning about the production start (overshoot from sigma_max * eps) is not visible here beyond small excesses
  over the field mean (ADJACENT 0.104 vs 0.086, DRINK 0.23 vs 0.18); sigma_max 10 sits slightly lower.

## E. Information in the conditioning (`information_probe.py`, `information_probe/results.json`)
Event: a zombie attack during t-1 -> t (cooldown reset to 5). Population: a zombie within L1 2 at t-1, 40,056 rows,
attack rate 11.3% (test 11.0%; the rule gives 89.0% when a ready zombie is adjacent after the move, 0.0% otherwise);
one third of physical episodes held out (13,685 test rows). S1 probe protocol (MLP 256-256).

| Probe input | AP | Log loss (base 0.346) | ECE | Mean P: ready + adjacent after the move / other |
|---|---|---|---|---|
| (d) rule's causal inputs (positive control) | 0.933 | 0.046 | 0.005 | 0.886 / 0.0004 |
| (b) predicted belief + a_{t-1} + fixation | 0.592 | 0.226 | 0.023 | 0.448 / 0.071 |
| (a) full conditioning (8 context frames + their beliefs + the above) | 0.448 | 0.269 | 0.022 | 0.337 / 0.086 |
| (c) last context frame + a_{t-1} | 0.410 | 0.297 | 0.028 | 0.282 / 0.080 |

- One-sided reading: the denoiser's conditioning carries at least the belief probe's information: it separates
  ready from non-ready adjacent zombies (0.45 vs 0.07), i.e. part of the hidden cooldown. The field does not use it
  (A: RISKY ready 0.027 vs not ready 0.024; ADJACENT reversed). So the field's underweighting is not only missing
  information; it ignores information present in its own conditioning.
- (a) below (b) is a probe limitation (18k inputs, 24k training rows), not less information: (a) contains (b). The
  belief stays below the rule (0.59 vs 0.93): whether the rest is absent or unextracted is not established.

## F. Consequence catalogue (`catalogue.py`, `catalogue/results.json`; descriptive)
A's instrument on six classes (rules in the script), 2,000 contexts each, 2 rows per context; opportunity rate = share
of the 297,835 dev frames with t >= 8. Instrument check: |bias| at sigma 0.05 <= 0.002 in every class; recorded
outcomes as defined in 100% (MOVE, BLOCKED, WOOD, DRINK), 96.9% (TABLE), 9.75% (SAPLING, rule p = 0.10).

| Class | Opportunity | Consequence | Recorded f | Field mean at sigma 0.7 / 2 / 6.67 / 20 | Production | Heun 32 |
|---|---|---|---|---|---|---|
| MOVE | 35.3% | view shift, ~60 cells, deterministic | 0.930 | 0.921 / 0.918 / 0.910 / 0.896 | 0.909 | 0.872 |
| BLOCKED | 3.6% | no shift, deterministic | 0.038 | 0.052 / 0.065 / 0.099 / 0.136 | 0.082 | 0.143 |
| WOOD | 2.7% | HUD wood + 1, deterministic | 0.994 | 0.720 / 0.385 / 0.238 / 0.128 | 0.148 | 0.204 |
| SAPLING | 1.6% | HUD sapling + 1 with p 0.10 | 0.072 | 0.059 / -0.004 / -0.018 / -0.024 | -0.025 | 0.018 |
| DRINK | 1.6% | HUD drink + 1, deterministic | 1.001 | 0.634 / 0.317 / 0.182 / 0.146 | 0.140 | 0.227 |
| TABLE | 0.76% | faced map cell -> table, deterministic | 0.918 | 0.818 / 0.621 / 0.404 / 0.151 | 0.310 | 0.377 |

SE 0.001-0.008. Reading (descriptive; frequency, footprint, location and separation vary together):
- The common, large-footprint consequence is carried (MOVE within 0.02 at every sigma <= 6.67; BLOCKED drifts
  toward a shift that does not happen, 0.10 at 6.67). Single-cell consequences are not, deterministic or not: HUD
  increments are carried at 0.18-0.24 of their size at sigma 6.67, the 10% sapling at ~0.
- Frequency alone does not order them: TABLE (0.76%) is carried better (0.40) than WOOD (2.7%, 0.24) and DRINK (1.6%,
  0.18). A per-element loss's share of an event is its frequency times its token change (larger for a new table than
  for a changed digit); this is a candidate, not a measured cause.
- DRINK replicates A on fresh rows (field 0.182 vs A's 0.178 at sigma 6.67).

## Reading so far
The deployed one-step deficit has two measured components: (1) the trained field's conditional mean underweights
consequences, including a deterministic visible one (drink), and ignores the hidden cooldown; this bounds every tested
sampler; (2) the 3-step production algorithm suppresses most of what the field still carries for a rare branch
(RISKY 0.025 -> 0.002, as B predicts for an exact field). Of the two, (1) is binding: no inference setting recovers
the recorded rates. Neither is "only a rare-event symptom" of the other, and the geometry mismatch is a separate,
integration-dependent property. E: the conditioning carries part of the hidden cooldown that the field ignores.
F: the field carries the common large-footprint consequence (movement) and misses single-cell consequences,
deterministic or stochastic, HUD most; frequency alone does not order them. Bulk / tail summaries (rest of F) not run.

## Corrections after Codex's review (10-29, `../20261029_completed_diagnosis_review/REVIEW.md`; verified, conceded)
1. The "field mean" columns are projections of the conditional mean at one noise level, not branch probabilities
   (they equal the event rate on recorded data because the recorded tokens sit at the two reference values). Read
   "no TESTED sampler exceeded the field's high-sigma projection", not "the field bounds every sampler". A and B
   cannot be chained into an independent tenfold sampler factor: B shows the production algorithm distorts exact
   mixture weights; in RISKY the production samples sit ~10x below A's high-sigma projection; that is all.
2. E's 0.45 vs 0.07 compared ready-AND-adjacent rows with everything else, so it mixed adjacency with readiness; the
   isolated readout (adjacent after the move: ready vs not ready) is in the revised E below.
3. DRINK measurability: the facing direction is decodable from the player-cell token, which supports one
   prerequisite; the full information pathway through the denoiser's noised conditioning is not certified by it. The
   revised E adds a direct drink probe.
4. The sigma 0.05 check involves the network (its output is scaled by c_out = 0.05): nearly, not fully,
   model-independent.
5. TABLE reached its outcome in 96.9% of rows, below the >= 99% declared in catalogue.py: in 70 failed placements a
   zombie or cow stood on the faced cell in almost every case (my class did not exclude mobs). TABLE is not
   deterministic as defined; the WOOD / DRINK readings do not depend on it.
6. Geometry: that it behaves differently across samplers shows it is not explained as a by-product of the 3-step
   blend; it does not show that geometry and consequence errors have independent causes. "Separate" is withdrawn.
7. Scope: F's bulk / tail summaries were not run, and C's external HUD-reader qualification was not run either.

## E revised (10-29, after the review; `information_probe/results.json` overwritten, first version in git history)
Same probes, seeds and splits (attack AP unchanged: 0.448 / 0.592 / 0.410 / 0.933).
- Cooldown isolated, rows with a zombie adjacent after the player's move (4,454 test rows; rule: attack 88.6% if
  ready, 0.15% if not): mean P(attack) ready vs not ready: belief + action 0.448 vs 0.167 (AUC 0.77); full
  conditioning 0.345 vs 0.153 (AUC 0.72); last frame 0.281 vs 0.157 (AUC 0.64); rule 0.886 vs 0.002 (AUC 1.00). The
  conditioning carries PART of the cooldown (AUC 0.77, well short of the rule); the field's flat readiness split (A)
  uses less than that.
- Drink accessible (DO actions, awake, drink <= 8, any facing; drink + 1 in 12.1%; 13,835 test rows): AP 0.860 full
  conditioning, 0.901 belief + action + fixation, 0.904 last frame + action; log loss 0.11-0.13 vs base 0.37. A reader
  recovers most of the drink consequence from the conditioning (not exactly: the rule is deterministic); the field
  carries 0.18 of it in the DRINK class. One-sided, as before.

## G. Checkpoint trajectory, P1 world seed 1 (`checkpoint_trajectory.py`, `checkpoint_trajectory/results.json`)
Matched rows (first 1,000 per class of A / F), matched noise, each checkpoint's own references. Declared checks per
checkpoint: checkpoints 1-4 FAIL (|bias| at sigma 0.05 up to 0.028 on HUD classes: early denoisers do not yet
reproduce a HUD cell even at sigma 0.05; SAFE up to 0.072 at checkpoint 1) and are not read; 5-10 pass (|bias| at
0.05 <= 0.0044, SAFE <= 0.036). Checkpoint 10 reproduces A / F (wood 0.242 vs 0.238, drink 0.190 vs 0.182, table
0.406 vs 0.404, risky 0.026 vs 0.025 at sigma 6.67).

Field projection at sigma 6.67 (sigma 2 in brackets), checkpoints 5 -> 10 (12.8 M -> 25.6 M frames); SE <= 0.01:

| Class | Recorded | ckpt 5 | 6 | 7 | 8 | 9 | 10 | Production 5 -> 10 |
|---|---|---|---|---|---|---|---|---|
| MOVE | 0.932 | 0.906 (0.913) | 0.893 | 0.917 | 0.911 | 0.913 | 0.911 (0.919) | 0.904 -> 0.908 |
| BLOCKED | 0.039 | 0.176 (0.096) | 0.123 | 0.147 | 0.115 | 0.111 | 0.100 (0.064) | 0.167 -> 0.081 |
| WOOD | 0.994 | 0.096 (0.306) | 0.198 | 0.225 | 0.241 | 0.275 | 0.242 (0.383) | 0.043 -> 0.140 |
| DRINK | 1.002 | 0.142 (0.268) | 0.171 | 0.172 | 0.187 | 0.179 | 0.190 (0.313) | 0.095 -> 0.152 |
| TABLE | 0.920 | 0.184 (0.558) | 0.239 | 0.272 | 0.275 | 0.322 | 0.406 (0.623) | 0.070 -> 0.318 |
| SAPLING (p 0.1) | 0.070 | -0.025 (-0.024) | -0.019 | -0.016 | -0.017 | -0.018 | -0.019 (-0.004) | -0.036 -> -0.025 |
| RISKY damage | 0.140 | 0.027 (0.058) | 0.027 | 0.030 | 0.025 | 0.029 | 0.026 (0.075) | 0.008 -> 0.003 |
| ADJACENT damage | 0.166 | 0.109 (0.096) | 0.108 | 0.099 | 0.099 | 0.097 | 0.092 (0.083) | 0.078 -> 0.061 |

Unread checkpoints 1-4 (descriptive only): MOVE rises 0.26 -> 0.86 and BLOCKED is first predicted to shift like a
move (0.72-0.81 at checkpoints 2-4) before the distinction appears at checkpoint 5.
Reading (this run only; the learning rate decays by cosine over the same frames, so plateaus cannot be attributed):
- Over the second half, movement is flat near the recorded value and the blocked distinction keeps sharpening.
- The single-cell deterministic consequences improve slowly and unevenly: TABLE still rises at checkpoint 10 (0.32
  -> 0.41), WOOD rises to checkpoint 9 and stops near 0.24-0.28, DRINK is nearly flat (0.14 -> 0.19); all remain far
  below the recorded values.
- The stochastic branch (SAPLING) and the hidden-state damage (RISKY, ADJACENT) show no approach in the second half.
- Production samples follow the field's trend at lower values (WOOD 0.04 -> 0.14, TABLE 0.07 -> 0.32).
Not established: what longer training, a different schedule or more capacity would do (Codex review), or why WOOD
and TABLE move while DRINK hardly does.

## H. Loss and gradient accounting of the G1 objective (`loss_accounting.py`, `loss_accounting/`)
P1's exact G1 objective on train-400 (1,760 windows of 8 + 64 frames, 112,640 target frames, diffusion forcing with
sigma ~ p_train), every element in one class; gradients w.r.t. the denoiser (the only parameters G1 alone trains);
Adam preconditioning from each checkpoint's own optimizer state; split halves (A / B, alternating windows) for
noise-free norms, cosines and drives (`halves_ckpt_05/10.json`; checkpoint 2 single-sample, `ckpt_02.json`). Every step
in this run was clipped (mean clip factor 0.012 / 0.017 / 0.059 at checkpoints 2 / 5 / 10); learning rate 9.4e-5 /
5.8e-5 / 1.0e-5; harmoniser sigma_G1 0.413 at checkpoint 10 (G1 weight 2.42).

"Drive" = first-order change of the class's loss under the expected Adam update (E[g] / (sqrt(v_hat) + eps)),
relative to its loss, per 1,000 steps (linearised: compare classes, not absolute values); "own" = the same if the step
followed only that class's gradient. Checkpoint 10 for the first five columns:

| Class | Frames with it | Elements | G1 loss share | Error per element | Gradient-norm share (A x B) | Cos with total | Drive ckpt 2 / 5 / 10 | Own 2 / 5 / 10 |
|---|---|---|---|---|---|---|---|---|
| drink + 1 | 1.75% | 0.015% | 0.092% | 0.156 | 0.87% | 0.034 | -77 / -162 / -7.5 | -2.5 / -11 / -8.5 |
| wood + 1 | 2.73% | 0.023% | 0.121% | 0.132 | 1.86% | 0.016 | -148 / -221 / -9.3 | -7.6 / -31 / -31 |
| sapling + 1 | 0.13% | 0.001% | 0.016% | 0.365 | 0.30% | -0.000 | -23 / -144 / **+16.5** | -1.7 / -2.3 / -5.4 |
| table placed | 0.74% | 0.006% | 0.034% | 0.137 | 1.01% | -0.042 | -292 / -55 / **+128** | -3.8 / -26 / -32 |
| zombie hit | 0.82% | 0.007% | 0.049% | 0.178 | 0.52% | 0.020 | -325 / -56 / 0.2 | -3.1 / -6.4 / -5.2 |
| map, moved frames | | 19.4% | 24.4% | 0.032 | 31.6% | 0.604 | -1169 / -257 / -27 | -494 / -111 / -20 |
| map, still frames | | 34.2% | 28.6% | 0.021 | 48.6% | 0.678 | -1525 / -438 / -38 | -988 / -296 / -32 |
| HUD, unchanged | | 15.1% | 4.8% | 0.008 | 42.4% | 0.419 | -1520 / -1510 / -164 | -1515 / -1320 / -173 |
| glimpse | | 30.6% | 40.5% | 0.034 | 53.2% | 0.560 | -724 / -1407 / -39 | -715 / -1387 / -37 |

Split-half reliability of every class gradient: cos(G_A, G_B) 0.92-0.99 at both checkpoints; the two halves give the
same sign of drive for every consequence at checkpoints 5 and 10 (table +139 / +116, sapling +8 / +25 at 10).
- Allocation: the five single-cell consequences hold ~0.31% of the G1 loss on 0.05% of its elements; their error per
  element is 4-20x the bulk's, so their gradient-norm shares (0.3-1.9% each) exceed their loss shares (Codex: a small
  share of dimensions is not a small gradient).
- Direction: their gradients are nearly orthogonal to the total (cos -0.04 to 0.03; bulk 0.42-0.68).
- Over training: at checkpoints 2 and 5 their losses fell mostly by transfer from the bulk's update (drive 2-100x
  their own); by checkpoint 10 (bulk converged, learning rate at its floor) the drive is ~0 (zombie), 3-20x below the
  bulk's (drink, wood) or REVERSED (table, sapling): the bulk-dominated update now raises their loss although their own
  gradient would lower it.
- Noise levels: the drink error per element rises from 0.04 (sigma < 0.17) to 0.27 (sigma > 2.67); its loss mass sits
  above sigma ~0.5, where the context must decide.
Limits: first-order at each checkpoint, noise-induced (second-order) effects ignored; the realised TABLE projection
still rose from checkpoint 9 to 10 while the endpoint drift is positive (different quantities: outcome-defined cells
over p_train vs the context class at sigma 6.67). The accounting makes loss allocation and interference plausible
mechanisms for the slow, uneven learning; it does not establish them as the cause, nor what another recipe would do.
