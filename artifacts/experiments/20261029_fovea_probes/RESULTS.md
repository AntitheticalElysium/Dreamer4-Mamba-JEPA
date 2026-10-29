# Fovea probes: results (2026-10-29)

Frozen P1 world seed 1, checkpoint 10; first 250 dev episodes (CPU). Plan and addendum: PLAN.md. Episode-bootstrap SE
in parentheses. No fixation behaviour exists to observe (random prior in P1-P3); these read the world's substrate.

## F1. Retention through the predicted belief (TSP head; `run/results.json` F1)
705,042 glimpse cells. Retention gain = TSP NLL (nats per pixel) on cells NOT fixated in the last 8 frames minus on
cells fixated k frames earlier, within screen cell x day / night strata (positive = the belief kept something).

| Stratum | Cells seen at lag 1 | Gain lag 1 | lag 2 / 4 / 8 | Input ceiling (TSP - reconstruction) | TSP NLL |
|---|---|---|---|---|---|
| all, day | 64,141 | -0.071 (0.019) | -0.108 / -0.109 / -0.136 | 0.078 | 1.91 |
| all, night | 30,707 | -0.021 (0.011) | -0.018 / -0.029 / -0.008 | 0.028 | 2.75 |
| terrain, day | 57,072 | -0.095 (0.016) | -0.116 / -0.133 / -0.134 | 0.061 | 1.87 |
| HUD, day / night | 6,106 / 2,880 | 0.002 (0.010) / -0.007 (0.010) | ~0 | 0.053 / 0.056 | 0.68 / 0.82 |
| addendum: static camera, unchanged cell, day | 42,149 | -0.064 (0.016) | -0.069 / -0.087 / -0.092 | 0.072 | 1.85 |
| addendum: camera moved, day / night | | -0.071 (0.034) / -0.035 (0.018) | | | |

Input ceiling at night by content: arrows 0.82, cows 0.42, skeletons 0.05, zombies 0.04 nats per pixel; entity
retention gains have SE 0.3-2.7 (few cells) and are not read.
- No positive retention through the belief output anywhere measurable; by day a small, randomised (so causal) COST
  of 0.06-0.14 nats per pixel (3-7% of TSP's NLL) on cells the fovea saw, present on static unchanged cells, so not
  stale memory of changed content. Mechanism not established.
- Limit: TSP reads only the 256-dimensional predicted-belief output, which cannot hold pixel detail of the cells; the
  denoiser's second route (8 context frames holding the raw glimpse tokens) is measured by F2, not here.
- What the glimpse adds over the 3 x 3 periphery at all is small (0.03-0.08 nats per pixel) except for night arrows
  and cows, as the acuity EDA predicted (PERCEPTION, acuity table).

## F2. Value of looking for the world's one-step prediction (`run/results.json` F2)
1,339 frames (782 random day, 218 random night, 339 night with a mob in view, reported separately). Value(L) = mean
over the 49 fixations of the next frame's periphery + lighting G1 loss minus that loss under fixation L.

| Set | Frames | Noise-set agreement | Total loss | Mean spread over L | R2: per-frame best minus best fixed (cross-validated) |
|---|---|---|---|---|---|
| random day | 782 | 0.58 | 9.2 | 0.041 | 0.056 (0.005) |
| random night | 218 | 0.75 | 43.3 | 0.085 | 0.112 (0.016) |
| night with a mob | 339 | 0.74 | 48.5 | 0.084 | 0.123 (0.012) |

- (R2) State dependence is real (7-11 SE) but tiny: choosing the best fixation per frame improves the next-frame loss
  by 0.6% by day and 0.25% at night relative to the best fixed position.
- (R1) No strong positional structure by day (+-0.007); at night, centres on the bottom row (window over the HUD rows)
  are worse than average (-0.02 to -0.04): looking at the HUD at night costs map information the HUD does not repay.
- (R3 / R4) Excess value by window content is small and mostly within 3 SE: cows +0.011 (0.003) by day; zombies
  -0.003 (0.004) day, +0.008 (0.005) night with mob; the zombie that attacks at t + 1: +0.050 (0.031) (153 windows),
  the one that just attacked +0.030 (0.019); the faced cell ~0 or slightly negative; HUD ~0.
- (R5) Value does not follow the glimpse's pixel information (rank correlation 0.02 in entity-free frames, -0.02 with
  entities): no noisy-TV signature, and no information-seeking signature either.

## Reading against PLAN outcomes
The world offers a statistically real but very small state-dependent value of looking (outcome b-c boundary: R2 > 0,
but not concentrated on consequential content); it does not chase night noise; the belief output keeps nothing of
what the fovea saw and shows a small unexplained cost. Two measured reasons the substrate is weak: at 3 x 3 acuity
the periphery already carries almost everything the fovea sees (input ceiling), and the field does not model the
single-cell consequences a decision-relevant fovea would look for (generation diagnosis A / F: drink, wood, damage
carried at 0.18-0.24 of their size). Not established: value for decisions over longer horizons (one-step periphery
loss only), or what a learned controller would do.

## Corrections after Codex's review (10-29, `../20261029_completed_diagnosis_review/`; verified, conceded)
1. F1's "cost of looking" is WITHDRAWN. My comparator (glimpsed at lag k vs not glimpsed in any of 8 frames)
   conditions on the camera's path, which is not randomised. The one-step contrast (glimpsed vs not glimpsed at t - 1,
   inverse-probability weighted by the known coverage of the 49 crops; Codex, checks.json): day +0.009 (0.018), night
   -0.007 (0.016), static camera / same terrain code / player excluded, day -0.012 (0.018). No detectable effect
   either way. The addendum used the same comparator, and its "unchanged" meant no entity, not equal terrain.
2. "The belief keeps nothing" overstates F1: the TSP head shows no benefit from previous glimpses; information could
   remain in the belief unused by that head, and the denoiser's 8-frame context route is not tested by F1.
3. The TSP-minus-reconstruction gap is not a ceiling on the glimpse's information (both pixel heads have a
   content-independent scale). What 3 x 3 allows is better measured by the acuity EDA's supervised probes.
4. F2's per-frame best fixation is chosen with the realised next frame: a hindsight upper bound for the one-step
   periphery loss, not a deployable policy, and not decision value.
5. R3's HUD excess is zero by construction (position-centred, and HUD coverage is fixed by position); withdrawn.
   R1's uncentred map (HUD windows below average at night) is the interpretable HUD result.
6. R5: no positive correlation in this proxy; it does not certify the absence of noisy-TV behaviour (no learned
   fixation exists).
