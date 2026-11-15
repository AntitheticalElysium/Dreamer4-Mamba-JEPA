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
