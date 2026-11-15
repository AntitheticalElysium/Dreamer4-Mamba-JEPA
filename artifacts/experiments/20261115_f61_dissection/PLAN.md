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

Result (`mining_check.json`): the stored frames equal the recorded ones on all 402 pairs; the largest token difference
is 0.0408 for F6.1 (one pair over 0.04; median 0.012, 99th percentile 0.032) and 0.0398 for F6 (none over), with
tokens up to 2.4 in magnitude, where bf16's spacing is about 0.016. Numerical, as claimed. **Check 5b**
(`mining_rerun.py`): the stage's own code with the pre-check recorded, F6.1, beside F6's queue result.

## Check 7: does the denoiser's use lag the belief? (check 3's reader on the checkpoint-3 rollouts, CPU)

The early readings saved checkpoint 3's quiet-bank rollouts (`../20261114_p1f61/readings/early_03/`, and F6's in
`reference_f6/ckpt_03/`). If drawn timing improves from checkpoint 3 to 10 while the belief's phase grows little, the
denoiser's use lags the belief (as recovery's did in F5 and F6.1) rather than being capped.

## Check 8: each clock event's size in the denoiser's currency (`token_share.py`, CPU)

The user asked whether the use failure is the loss share. Within G1, an event is priced by its token change. F6.1's
periphery tokens on held-out DEV: the squared change at the digit's cell, over the whole frame, and the cell's share
of the frame's change, at events and at eligible frames without one, for the four pixel clocks and recovery. (Sleep
dims the map, not the HUD: Craftax renders the inventory after the sleep tint.)

## Check 9: still learning at checkpoint 10? (`trend.py`, GPU) and check 10: memory or cache? (`swap.py`)

User 11-15: "run them directly". Check 9: the quiet bank at F6.1 checkpoints 5 and 8 beside 3 and 10 (check 3's
reader). Check 10 (Codex's swap, with a cache arm): asleep exact-history pairs (F5's builder, asleep start, fatigue -5 /
-3, the previous rise 2 steps apart beyond the 8-frame window); on frozen F6.1 one generated frame per cell, common
random numbers: cached memory from branch c, current memory from branch m, cache context signal trained (7) or
stressed (4). Readings: native due minus delayed; the current- and cached-memory effects; the stress effect on the due
cell. Rise follows memory on clean prefixes -> the free-running delay is recursion; only under stress -> the cache
holds it back; neither -> a weak learned conditional.

## Check 10c result and the fine-tune (`swax_ft.py`; user 11-15: run it, no control)

10c (natural onsets 2-3 steps apart, `swap_onset.json`, `swap_belief_onset.json`): the belief separates the branches
(logit +1.30 [0.47, 2.25], 25 / 32 pairs); the drawn rise follows memory weakly (due 0.034 against delayed 0.017;
current-memory effect +0.012 [0.006, 0.019], cached +0.006 [0.001, 0.012]); a stressed cache raises it 2.6x and doubles
the memory effect (due 0.089, delayed 0.043). A soft cache shortcut: the clean local context outweighs memory.

Fine-tune: SWAX's stochastic short window (Cabannes et al. 2025, section 4.4: the short window per batch with p = 0.5,
none in the last 10% of training, evaluated at the long window; long windows leave the recurrent memory
under-trained). Ours: from F6.1 checkpoint 10, the denoiser alone on G1 alone, 2,000 updates (F6.1's streams, carry,
cache; AdamW 1e-4, 100 warm-up, clip 1; as learnable.g1_arm with the belief frozen), temporal window 0 with p = 0.5
before update 1,800, else 4. Fixed before results: memory use is improved if, at 2,000 updates, the swap's native due
rise (trained cache) is at least 3x F6.1's (>= 0.10) with a current-memory effect above F6.1's interval (> 0.019), and
the quiet bank's due-step release rises without the before-due rate exceeding 0.02; the monitor's G1 within 10% of
F6.1's 0.00383 (guard).

## Check 11: can F6.1's own tokens feed a generic slow trace? (`token_trace.py`, CPU)

Codex's analytic traces time the awake clocks from coarse pixels (post-sleep AP 0.42 / 0.39 / 0.44) but weakly from
F6's tokens through a fitted scalar (0.11 / 0.21 / 0.06). Here the per-cell scalar is the cell's own squared token
change (F6.1), two time constants (8, 32), a shared label-free next-change predictor; AP at the after_reset clock
events of DEV fold 4, full-prefix traces against traces started 8 frames back. High full and low short: our tokens
carry what a slow trace needs.

## Check 12: access or incentive? (`traces_ft.py`; user 11-15: run it, no control)

A substitution test, not a design: perfect, cheap per-cell recency (check 11's traces of each cell's own token change,
time constants 8 and 32, through the previous frame) as a zero-initialised additive input to that periphery cell's
denoiser token; the denoiser alone fine-tuned on G1 alone from F6.1 checkpoint 10, 2,000 updates (as the SWAX
fine-tune, full window). Read with one generated frame on the exact-history pairs (awake energy 32, food 24, recovery
32, asleep onset 32), each branch's own memory and traces, and traces exchanged; update 0 (inert input) reproduces F6.1.
Fixed before results: **access** if at 2,000 updates the drawn due change on the awake energy and food pairs is at
least 0.10 with due minus delayed above zero (95% lower bound) and a positive trace effect; **incentive** if the due
change stays below 0.05 with no separation. Anything else is reported as is.

## Checks 13-15: why the denoiser does not draw the awake clocks from the traces (`use_check.py`; user 11-15: "run that check ... why doesn't the denoiser use it? Are there other examples of small pixel changes that are learnt by G1?")

Check 12 gave the denoiser per-cell traces and the drawn due change did not move. Three links separate the causes:

```text
traces on these pairs carry the timing     check 13 (CPU)
denoiser's conditional uses them           check 14: its expected change before sampling
the sampler expresses the conditional      check 14: expected change against drawn rate
```

- **Check 13 (separable).** Check 11's label-free next-change predictor refitted line for line (its three full-trace
  APs reproduced to 0.002 before use), scored on both branches of every exact-history pair at the due transition
  (features through the last given frame, the digit's cell). Also the input's own scale: the due-minus-delayed
  difference of log(1 + trace) at the cell in units of that feature's spread over training cells. Reading:
  **separable** when the predictor's due-minus-delayed probability has a pair-bootstrap 95% lower bound above zero
  (awake energy, food); its due probability is what a denoiser using the traces as well as the predictor could draw.
- **Check 14 (expect).** The same pairs and four cells as check 12, the physical stage of `model.sample` with its noise
  (the drawn frames reproduce check 12's reader rates exactly at update 0 and 2,000, asserted), recording the
  denoiser's prediction at each of the 4 sampler steps. The expected change at the digit's cell is the projection of
  (prediction - last frame) on the due branch's true change; at the first step (pure noise) it is the denoiser's
  conditional mean, i.e. its own probability of the change. F6.1 and the trace fine-tune at 2,000. Reading: **learned
  use** when the fine-tune's first-step due-minus-delayed (own traces) has its 95% lower bound above zero on awake energy
  or food; then **sampler** if the drawn rate stays below half the first-step expectation; **not learned** otherwise.
- **Check 15 (catalogue, no bars).** On natural DEV fold-4 frames (128 events per class, 9 frames of history or more),
  F6.1's first-step expectation and drawn rate at the event's cell, and the drawn change rate (squared token change
  above 1, check 11's threshold) on 128 eligible frames without the event. Classes: the clock drops (all and
  after_reset), the asleep rise, recovery, damage, food and drink intake, and every inventory count change with at
  least 30 events (its cell measured as the HUD cell with the largest mean token change at those events). Each class is
  tagged by where its cue lies (in the last 8 frames and the action, or older). The trace fine-tune on the same rows
  with its own and with zeroed traces, and where the traces act (squared change of the first-step prediction per cell
  group, own minus zeroed). Check 13's predictor scores each clock row, so the denoiser's expectation sits beside the
  probability the traces allow.

Bounds: one seed, one fine-tune length (2,000 updates; the map's norm was still growing linearly at the end), linear
projection probes of a token change, simulator labels for selection only.

## Check 16: input form or incentive? (`recency_ft.py`; after check 14: the traces separate the pairs, the denoiser's conditional ignores them)

Check 13 put the traces' due-minus-delayed difference at the digit's cell at 2-3% of log(1 + trace)'s spread over
cells: present, but nearly invisible to a linear map that every cell shares. Check 12 rerun with that one change: each
periphery cell's exact recency (frames since its own token last changed by more than 1, capped at 63; unknown before
any change) through a zero-initialised embedding, a lookup the denoiser can read at any value. Same fine-tune (G1 only,
denoiser only, 2,000 updates from F6.1), same pair reads and seeds, then check 14's expectation read. The recency at
the due frame is recorded per pair (it must differ between the branches for the test to bear).

Fixed before results: **form** if at 2,000 updates the drawn due change on the awake energy and food pairs is at least
0.10 with due minus delayed above zero (95% lower bound) and a positive recency effect (check 12's trace-effect
reading); **incentive** if the due change stays below 0.05 with no separation and the first-step expectation shows no
separation either. Anything else is reported as is. A learned use at the first step that the drawn rate does not show
is the sampler, reported beside it.

**Check 16b** (`recency_catalogue.py`, added after check 16's 1,000-update read: recency used for food, not yet for
awake energy, on the stand-still pairs): check 15's natural-frame catalogue on the recency fine-tune at 2,000, own
against zeroed recency, same rows and seeds; no bars. Also from the labels (DEV, awake): the probability that the digit
drops at the next frame given frames since its last change peaks at 0.96 for awake energy (30 frames, 97% of drops),
0.83 for drink (20) and 0.73 for food (25), so the readable cue is sharpest for the clock not yet used.
