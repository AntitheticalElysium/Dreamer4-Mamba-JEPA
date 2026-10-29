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

## Reading so far
The deployed one-step deficit has two measured components: (1) the trained field's conditional mean underweights
consequences, including a deterministic visible one (drink), and ignores the hidden cooldown; this bounds every tested
sampler; (2) the 3-step production algorithm suppresses most of what the field still carries for a rare branch
(RISKY 0.025 -> 0.002, as B predicts for an exact field). Of the two, (1) is binding: no inference setting recovers
the recorded rates. Neither is "only a rare-event symptom" of the other, and the geometry mismatch is a separate,
integration-dependent property. E (information probes) is running; F follows PLAN v2.
