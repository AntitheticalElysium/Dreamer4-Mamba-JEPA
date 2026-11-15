# P1-F6.1 dissection (2026-11-15)

User, 11-15: "Once it completes, let's dissect the results. Run whatever additional checks you need, be thorough ...
it will be essential for the next overnight run." The fixed readings are F6.1's own (`../20261114_p1f61/PLAN.md`);
these are the checks added for the dissection, each committed with its code before it is read.

## Check 1: the sleep phase where no loss asks for it (`sleep_phase.py`, CPU)

F6.1's forecast heads use the asleep energy-rise phase only before WAKE_UP has paid (`readings/ahead_10.json`:
phase effect 0.66 nats at distance 2 not yet paid, -0.008 already paid), while the belief probe's asleep-rise AP rose
from F6's 0.180 to 0.471 over all asleep frames (`trajectory/belief_probe_10.json`).

- **Claim (Codex's competing explanation (b) in F6.1's plan):** the belief carries the phase only on sleeps where the
  reward forecast prices it, before WAKE_UP has paid.
- **Competing explanations:** (i) the phase is one belief feature that serves both, so repeat sleeps carry it too and
  only the heads ignore it there; (ii) part of the gain is elapsed time or lighting (sleep falls at night).
- **Separating reading:** F6.1 against F6 at checkpoint 10 under the same instrument as the gaze test (learnable.probe
  on all DEV, folds 0-3 fit, fold 4 read): AP, and held-out log-loss gain over Codex's time-and-light reader, on asleep
  frames before and after WAKE_UP has paid, and on the paid cohort with 8 constant asleep frames (the gaze test's
  `after_reset` asleep cohort); the five events on the natural, steady8 and after_reset rows beside them. F6.1 minus F6
  per-row loss with episode-bootstrap intervals. No bar: a dissection reading. CPU encodings (reference scan) differ
  from the GPU readings in the last digits; F6 and F6.1 are encoded the same way.

## Check 2: waking on time on repeat sleeps (`sleep_paid.py`; simulator build on CPU, generation on GPU)

F6.1's primary sleep bank has 32 scenes, of which only 7 had WAKE_UP already earned, so its plan reads that cohort as
descriptive. In DEV, 1,978 of 2,995 wakes happen after WAKE_UP has paid, where the reward forecast has nothing to
price.

- **Claim (F6.1's competing explanation (b)):** repeat sleeps stay untimed in generation.
- **Competing explanation:** the phase the belief learned on first sleeps is general, and the denoiser draws repeat
  wakes on time too.
- **Separating reading:** the quiet bank's own construction and generation, restricted to scenes with WAKE_UP already
  earned, 48 scenes (one per episode), for F6.1 and F6 at checkpoint 10; F6.1's sleep_split readings (phase test
  against the cohort shuffle, awake by step 8-48 against the simulator, early and late wakes, wait ratio). No bar: the
  phase advantage and its p beside F6's on the same scenes.

## Check 3: where the late waking comes from (`hazard.py`, CPU, saved rollouts)

F6.1 draws the phase on first sleeps (phase advantage 1.31 steps against F6's 0.46) and wakes at once after drawing
energy 9 (224 / 224 transitions, F6 0.50), yet late or missing wakes rose (0.91 against F6's 0.775).

- **Claim:** the remaining lateness is the asleep energy rise drawn too rarely (an under-drawn hazard, as in F4-F6).
- **Competing explanation:** the rise is drawn at a consistent but shifted step (a timed clock with an offset).
- **Separating reading:** per scene the simulator's rise step (deterministic in the fatigue phase), per draw the first
  generated energy 9; the generated rise hazard at energy 8 before, at, 1-3 and 4+ steps after the due step; the rank
  correlation of the scenes' generated and true rise steps; F6.1, F6 and F5 on the same 32 quiet-bank scenes, and F6.1
  and F6 on the repeat-sleep bank of check 2. A timed world rises near the due step; an under-drawn one spreads its
  rises after it.

## Check 4: does the belief expect the rise the denoiser draws late? (`rollout_belief.py`; generation on GPU)

- **Claim:** along the world's own rollouts the predicted belief still expects the rise at the due step, and the
  denoiser does not draw it (a use failure, like F4's zombie hits: the belief knew, generation did not follow).
- **Competing explanation:** the belief loses the phase along its own rollouts (the generated frames do not carry what
  the recorded ones did), so the denoiser draws what the belief expects.
- **Separating reading:** the banks' own generation, unchanged, with each per-step belief output recorded (bitwise
  equal to the queue's run on the quiet bank, the known answer); the world's own asleep-rise probe (learnable.probe on
  its DEV beliefs from check 1) read on those beliefs, while the draw is still asleep at energy 8, before, at, 1-3 and
  4+ steps after the due step, beside the drawn rise rate at the same steps; F6.1 and F6, both banks. The repeat-sleep
  bank's runs are check 2's (same generation, the recording does not change it).

## Check 5: why the mining stage stopped (`mining_check.py`, GPU)

The queue's `mining` stage (the coal and wood DO / NOOP pairs, F6's coal and iron guard) failed its own pre-check: a
pair's frame-(t-1) periphery tokens recomputed from the bank's stored frame must match S1's encoding of the recorded
episode to within 0.04. It passed for F5 and F6 with the same bank and code.

- **Claim:** a numerical tolerance exceeded by F6.1's tokens (batch-2 against batch-512 bf16 arithmetic), not a
  mismatch of inputs.
- **Competing explanation:** a real input mismatch (the stored frame or fixation differs from the recorded one).
- **Separating reading:** per pair, the stored frame against the recorded frame (exact), and the largest token
  difference with its distribution and the tokens' scale, for F6.1 and F6. If it is numerical, the stage is rerun with
  the pre-check reported rather than asserted, as a dissection reading, never by editing the pinned queue.

## Check 6: sampler or learned conditional? (`sampler_steps.py`, GPU)

Check 4 shows a use failure: along F6.1's rollouts the belief expects the rise at the due step (0.61) while the
denoiser draws it 0.05 of the time.

- **Claim:** the learned conditional under-draws the change (the denoiser was never taught to turn this belief state
  into the digit change; the clock is about 1% of G1's price).
- **Competing explanation:** the few-step sampler (K = 4) under-draws a low-probability, single-cell flip that more
  evaluations would draw.
- **Separating reading:** the same quiet-bank generation for F6.1 with K = 8 (the only allowed value above 4,
  k_max = 8), same scenes and seeds; check 3's hazard at, before and after the due step and the rank correlation,
  beside the K = 4 queue run. F4's readings were step-invariant at 8 for other events (`../20261107_f4_followups`).
