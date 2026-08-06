# D4-lite reboot: executed-control preflight protocol

Date registered: 2026-07-20, before execution.

## Question

Does the first source-pinned `T-BASE` checkpoint produce any usable executed
Crafter behavior, and do its known offline weaknesses visibly induce planner
exploitation?

This is an instrumentation check, not an official Crafter evaluation and not a
model-selection tier.

## Frozen checkpoint

- Path: `outputs/d4_mamba_jepa/preflight_t_base_5k/world_t_base.pt`
- SHA-256:
  `6d4a2a18ed968ab29b0ef32d02f656284647b50714b25d54abfd90884ed079e4`
- Arm: `T-BASE`
- World updates: 5,000
- Training seed: 20260720

## Fixed execution contract

- Environment seeds: 2000, 2001, 2002. These are outside all repository seed
  tiers found during the preflight audit.
- Policies: uniform random and evaluation-only categorical random shooting.
- Episode cap: 200 environment steps.
- Receding horizon: 8; execute only the first selected action.
- Context: the latest 8 observations and their led-to action tokens.
- Candidates: 34, with the first action stratified so every one of Crafter's 17
  actions appears twice.
- Denoising: the source-aligned finest schedule, K=4.
- Score:
  `sum_k 0.99^k * product_{j<k}(continuation_j) * reward_k`.
- Common deterministic policy-RNG derivation is fixed in the runner and must be
  retained for future checkpoint comparisons.
- No learning, checkpoint selection, reward calibration, action filtering, or
  replanning-budget tuning is permitted from these episodes.

## Required outputs

- Per seed: return, achievements, length, action histogram, termination,
  planner predicted-return spread, and throughput.
- Paired planner-minus-random return and achievement differences.
- Crafter success rates and geometric-mean score are computed with the official
  formula but labelled non-publishable because three 200-step episodes are not
  the benchmark's one-million-step evaluation.

## Interpretation

A planner loss does not reject the architecture: this checkpoint is a
single-seed, 5k-update preflight with known reward-magnitude and terminal
calibration defects. A win does not validate the architecture either. The run
only decides whether executed behavior is wired correctly and whether further
baseline training has an observable target.
