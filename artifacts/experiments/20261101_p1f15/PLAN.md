# P1-F1.5: the architectural baseline, everything but the learned fixation (11-01 night; for review before launch)

**Why (user 11-01; Codex supports).** The learned chooser (decision 2) is not settled (H3: the endogenous floor as
built fails; the design needs discussion, not another ablation). Decision 1 is built and checked
(`../20261101_factorised_denoiser/PLAN.md`, results 1-7). P1-F1.5 trains the system with every agreed piece except the
learned chooser, so the chooser's later run has its control: P1-F2 = P1-F1.5 + the chooser, nothing else changed.

**Recipe** (`train_p1f15.py`, saccade.train.p1 itself): P1 world 1's config, data streams and world seed 1, with three
declared changes written into the checkpointed config: (1) P1-F's faithful pixel decoder (mean by squared error, scale
by NLL on the detached mean); (2) decision 1's factorised denoiser (physical tokens never attend to glimpse tokens in
spatial layers; independent physical / glimpse noise levels and adaLN, UniDiffuser); (3) the cascade's glimpse
conditioning level 0.1 (used only by the sampler, i.e. by readings). Fixations: P1 world 1's, uniform over the 49
windows (data.fixation_prior, seeded per stream); no hand placement, no learned chooser, no reference model, no floor.
Budget, schedule (25.6 M frames, 64- then 256-step chunks), optimizer, harmoniser, checkpoints and monitor unchanged.
MAMBA_DETERMINISTIC=1 (required by the launcher; mamba_ssm otherwise autotunes per process, so a resume would not
continue the same computation; deterministic mode costs no time here: 1.32 s / update joint, as P1-F's 1.36).
Expected ~11 h (P1-F 9.5 h x 1.18-1.21, decision 1's measured training cost).

**What it can and cannot be compared with.** Against P1-F it changes the fixations (rule-placed on 13% of frames ->
uniform) AND the generator (joint -> factorised): neither change is isolated (Codex). Against P1 world 1 it changes the
decoder and the generator. Its role is the control for the learned chooser, and the first trained factorised world.

**Pre-launch checks** (`checks.py` -> `checks.json`): the real launcher for 25 updates (every logged term finite; time,
peak memory); resume with TR.p1 at a shrunken budget (4 straight vs 2 + resume + 2 against a second straight run,
bitwise); identity (validate_resume refuses P1-F's checkpoint under this config and this run's checkpoint without the
factorised key; the launcher refuses a directory with another identity).

**Not addressed tonight, before any adaptive chooser or reading:**
- Readings must use the cascade: saccade/eval/s1.one_step and Codex's sampler factorial sample the periphery and the
  glimpse jointly (Codex exercised the S1 path: zero cascade calls). Fixed and checked before the readings run.
- The chooser hook's inputs (Codex): `choose(p, v)` reads the clean generated periphery, the glimpse stage its
  conditioning-augmented version; with an adaptive chooser the window can carry information the conditional lacks.
  Inert under uniform fixations; to be reconciled with the real step's inputs before decision 2 is built.
**Readings: required, fixed before launch (11-01, Codex's pre-launch review section 4).** Checkpoint by P1-F's rule
(10 unless the monitor rose at three successive checkpoints). Every generation reading goes through the cascade with
the stage-2 cache (the same optimized sampler in every later arm); the evaluators are adapted and checked during
training (CPU; the GPU is the run's) and on the GPU after it, before any result is produced. Descriptive, no pass rule:
1. Canonical S1 (TEST_SPEC S1.1-S1.4), S1.3's one-step generation through the cascade.
2. P1-F's battery as applied to beta-TSP (its PLAN, Readings): the actor interface (actor-facing consequences) under
   random gaze, and guided gaze for comparability with P1-F; M0.5's evaluator; the decoder render; Codex's re-render
   noise panel by light; HUD counter consequences at the rule's crop and at another crop containing the counter, DO
   and NOOP (the real-crop evidence check); entity counts; the sampler factorial (Euler-3 / churn 1 primary, Euler-8 /
   churn 0 and Euler-32 / churn 1 secondary, never a best-of), each sampler in its cascade form.
3. Per-query consistency (TEACHING 10a): per legal fixation (all 49), glimpses drawn through stage 2 on one sampled
   periphery per context; the reward / continuation expectation per fixation and its spread across fixations against
   sampling error; glimpse-periphery coherence at every fixation (the label-leak check).
4. Real- versus generated-conditioning gap (TEACHING 10a): the glimpse stage given the real periphery (noised at the
   conditioning level) versus given the cascade's own generated periphery.
5. Useful past glimpse -> belief -> later physical prediction (decision 1's standing reading, TEACHING 10a): next-frame
   periphery / lighting prediction with the real past glimpses versus the same history with them replaced by
   glimpses at another window, the belief recomputed (base: `20261029_fovea_probes/measure.py` F2), by light and by
   whether the past glimpse held an entity the periphery missed.
Comparisons with P1-F and P1 world 1 are descriptive (two changes each); this run is the control for P1-F2.

## Pre-launch checks: results (11-01 night; `checks.json`; all pass)
- First attempt: the real launcher ran out of memory after update 1 (default CUDA allocator: 4.0 GiB allocated +
  0.76 GiB reserved unused, the factorised path's temporaries fragmenting the 6 GB card). Fixed by expandable
  segments, set by the launcher and recorded in the identity (allocation only; no arithmetic changes).
- Launcher, 25 updates: every logged term finite. Resume: 4 straight = 2 + resume + 2 = a second straight run,
  bitwise (max abs 0, state equal). Identity: P1-F's checkpoint refused under this config, this run's refused
  without the factorised key, the launcher refuses a directory whose identity differs.
- Memory measured on fresh processes (nvidia-smi, every second): 64-step phase (60 updates) process max 4,624 MiB,
  card max 5,245 MiB of ~5,850 usable (desktop ~620 MiB): ~0.6 GiB headroom; 256-step phase with 3 checkpoints
  (encoder diagnostics) process max 4,492 MiB. Time: 1.66 s / update over the first 60 at 64 steps, 6.6 s at 256
  (the GPU's thermal state moves absolute times; decision 1's measured cost is +18-21%): ~11 h.
- Codex's pre-launch review (`../20261101_p1f15_review/REVIEW.md`): supports launch. Qualification kept: beyond the
  model, 4 optimizer-moment tensors (pixels.cell / tsp.cell) differ at <= 2.3e-13 (first moment) in both the resumed
  and the uninterrupted repeat comparisons; weights, carry, cache, RNG, streams and counters equal. Not a
  resume-specific defect; the optimizer state is not bitwise across runs. The bf16 cascade glimpse is not bitwise
  between cached and uncached computation (rounding, 7e-7 in float32).
