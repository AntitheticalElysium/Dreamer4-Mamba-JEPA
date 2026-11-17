# Change-map forecast mode: objective gate (CPU only, no training)

2026-11-11, approved by the user ("let's run it, proceed as you discussed"). Background and reasoning: `DISCUSSION.md`.

## Question

The candidate is a second mode for the existing change-map head: on some samples the next frame is hidden, and the
head forecasts the canonical G2 change target for transition t -> t+1 from what is available before frame t+1 (the
current frame and the post-action belief). Before any implementation or training, does that target

1. charge a forecaster materially for forgetting older evidence, after its real averaging over all cells, and more
   so than G1 does at its most belief-dependent noise level;
2. do the same for a consequence with a different mechanism from the periodic hunger counter;
3. charge less for a predictable appearance nuisance than for those consequences?

These are oracle incentive checks on the target, as in `../20261111_observable_rollout_gate/` and
`../20261111_phase_erasure_gate/`: the best forecast that knows the history is compared with the best forecast when two
histories are tied. No model is trained or run except the frozen F4 encoder, which gives G1's targets. Passing is a
necessary incentive, not proof that SGD will learn to use it.

## Fixed definitions

- **Target:** the canonical `full_frame_a9b_v2` change target (`saccade/data.py`: global shift by support-weighted
  alignment, exact-match patches -> 0, A9b LightGBM teacher `artifacts/saccade_full_frame_change/targets/change_teacher.txt`
  elsewhere), 81 cells x 2 comparisons, BCE.
- **Support as in the canonical store.** The store computes the temporal-variability support over the whole episode.
  The 11-11 gate computed it over a 17-frame branch window. Here an "episode" for a constructed branch is the recorded
  prefix up to the branch start plus the branch frames. A local copy of `change_correspondence` takes the support as an
  argument; it must reproduce `D.change_probabilities` exactly on a full recorded DEV episode (reproduction control).
- **Cost of forgetting (both objectives):** Jensen gap between the two members, i.e. the excess loss of the best tied
  forecast. Change target: mean over valid cell/comparison entries pooled across members (the G2 reduction, `scores` in
  the 11-11 gate). G1: ||y_A - y_B||^2 / (4 n) over the n = 1,880 percept numbers of frame t+1 (F4 encoder, CPU fp32),
  averaged over the 49 fixation centres of the training prior. For G1 this is its pure-noise (tau = 0) cost; at every
  other noise level the noisy target reveals part of the answer, so this is G1's most favourable case.
- **Noise floor (both objectives):** the same Jensen gap / variance, but across 8 fresh environment keys from one real
  state, averaged over 240 natural DEV states (120 seeded episodes x 2 seeded times, recorded action, one fixation per
  state drawn from the prior and held across its keys; change support from the recorded episode). This is the
  irreducible per-step loss any forecaster pays, a lower bound since the state is more informative than history.
- **Pricing ratio:** R = (S_F / N_F) / (S_G1 / N_G1) per scene, S = cost of forgetting at the informative transition,
  N = natural noise floor. Median over scenes with a joint bootstrap over scenes and natural states. Day and night
  floors are reported separately; the decision uses the natural mix.

## Stage 0: the brightness foil and the camera shift

0a. Reproduce the 11-11 gate's +3 foil and hunger costs for episode 9 with its 17-frame support (control).
0b. Recompute with episode-level support.
0c. For each transition, report the chosen global shift in the plain and foil members and which entries change.

If check 3 passes with episode-level support on every scene, no target change is needed. If it fails, test exactly one
pre-declared, generic candidate: photometric offset compensation in the alignment cost (for each candidate shift,
subtract the median per-channel difference between the current frame and the shifted previous frame over supported
pixels before taking absolute differences). The candidate must also match the simulator's camera displacement (player
position change) on 2,000 seeded natural DEV transitions at least as well as the canonical estimator, day and night
separately; the simulator is used only to verify, not to build the target. No other candidate is tried in this gate.

## Stage 1: the three checks, one step ahead

- **C1, hunger (evidence 17-25 steps old).** The 24 constructed pairs of `../20261108_full_frame_change/phase_bank.pt`,
  rebuilt bitwise (hunger 20 vs 12 at t0, same keys, NOOP). Query transition 31 -> 32: the early member's food drops,
  the other's does not; the last 8 frames are identical. Control transitions 24 -> 31 must cost zero.
- **C2, zombie readiness (an entity interaction, evidence 4-6 steps old).** Constructed the same way from real DEV
  states: awake, health >= 6, exactly one zombie at Manhattan distance 1, no health drop in the 6 recorded frames before
  t0. Member A sets that zombie's hidden cooldown to 0 and member B to 2 (the zombie attacks at transitions 0 and 6 in A,
  2 and 8 in B). Same keys, NOOP; keys are searched (up to 64 per state) for branches where frames 3-6 are byte-identical,
  frames 1-2 differ, A loses health at transition 6 and B does not. Query transition 6 -> 7. Up to 32 pairs from
  distinct episodes. Readiness at frame 6 is fixed by the time since the last visible hit, whatever happened before t0.
  This is inside the denoiser's 8-frame cache but outside what the forecast mode sees except through the belief.
- **C3, nuisance foil.** On every C1 and C2 scene, copy the non-event member and brighten frames from the query's
  successor on by +1, +3 or +6 out of 255 (support recomputed with the foil); cost at the query transition.

## Decision rules (fixed before running)

- C3 passes only if the foil costs less than that scene's consequence for every C1 and C2 scene and every amplitude.
- C1 and C2 pass only if the consequence costs more than zero on every qualified pair, the control transitions cost
  zero, and the median R >= 3 in each.
- R between 1 and 3: the forecast prices old evidence no better than a weighting change inside G1 could plausibly
  provide, so this target alone does not justify a head; R < 1: it prices it worse than G1. In either case we report and
  revisit the target before any training.
- Only if C1, C2 and C3 all pass does a training plan follow, with native generation under the full cache as its
  acceptance test (more true drops when due, no rise when not due), reviewed before launch.

## Out of scope

Training, gradient paths, the harmoniser's group scaling, the belief-input question for the grounding mode, canonical
edits, and the GPU. Jobs run on CPU under `systemd-run --user -p MemoryMax=`, with `JAX_PLATFORMS=cpu` and
`MAMBA_DETERMINISTIC=1`.
