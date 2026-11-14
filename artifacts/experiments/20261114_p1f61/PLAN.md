# P1-F6.1: no visual forecast, Dreamer 4's multi-distance reward forecast (2026-11-14 night)

Ordered by the user on 11-14 night as tonight's run: "remove the token forecaster and add the reward forecast ...
bundle these changes together: I don't think we'll solve the non-consequence timer issue but it might make progress."
Codex reviews this plan and the code before launch. Evidence behind it: F6's dissection (`../20261113_p1f6/RESULTS.md`),
the currency EDA (`../20261114_currency_eda/RESULTS.md`), Codex's independent sleep reanalysis (`../NOTEBOOK.md`,
"Independent check of the longer reward forecast") and the learnability result (`../20261114_learnability/PLAN.md`).

## Question

Does pricing the reward 2-8 steps ahead teach Mamba the sleep clock (the energy rise while asleep, which decides when
the player wakes), and does the world then draw waking on time, including on sleeps where WAKE_UP has already paid?

- **Claim.** With reward forecasts at distances 1-8 from the predicted belief, the asleep energy-rise phase is taken up
  by the belief and used by the denoiser: waking in native generation moves toward the simulator's timing on the quiet
  sleep bank. Basis: knowing the fatigue phase is worth 0.036-0.047 nats per asleep frame at 2-8 steps ahead, about
  half of what the recovery counter was worth at one step (0.068) when the one-step forecast taught it.
- **Competing explanations.** (a) The forecast becomes a better reward predictor that the denoiser does not use (belief
  probe up, generation unchanged). (b) The price is a first-sleep price: 85-87% of the gain comes from asleep frames
  before WAKE_UP has paid (Codex: 0.150 vs 0.010 nats at four steps), while 1,978 of 2,995 DEV wakes happen after it
  has paid and the clock governs them as strongly (221 / 230 at the due phase vs 48 / 2,527 at other phases). Then
  repeat sleeps stay untimed. (c) The pixel-only clocks with no reward consequence within 32 steps (food, drink, awake
  energy) stay at their base rates. This is predicted, not tested.
- **Separating control.** F6 at the same checkpoints: same seed, same initial weights for every shared parameter, same
  data order and schedule. F5 beside it. On the sleep bank every scene starts from a recorded asleep frame with its
  visual history; the split by WAKE_UP status separates (b) from the claim.
- **Bundled (user's choice).** The arm differs from F6 by two things: the token forecaster removed and distances 2-8
  added. F6 matched F5 on every clock while its token head supplied 1-2% of the belief's gradient, so the removal is
  not expected to move a clock, but a regression it causes (for example on what the token head did for the belief)
  cannot be separated from the new heads.

## The arm (`train.py`; F5's recipe plus `heads.forecast` = `{tsp: false, reward_distances: 8, reward_init_seed: 615003, reward_ahead_reduction: mean}`)

- **No visual forecast at all.** F6's token forecaster is removed and TSP's forecast mode is not restored
  (`tsp: false`): F6's glimpse prediction recovered when that mode was removed (2.28 against F5's 2.57 nats per pixel).
- **Reward forecast, distances 1-8.** Frame t's reward is predicted from the belief predicted before frame t - d + 1,
  with frame t's percept replaced by the forecast mask, for d = 1..8. Distance 1 is F5's and F6's forecast mode,
  unchanged (it shares the reactive head's output layer). Distances 2-8 each have their own output layer (7 x 65,792
  parameters, 460,544 in all) on the shared outcome trunk, built under their own seed so every other parameter and the
  training RNG keep F5's draws. A pair counts only when no episode starts in [t - d + 1, t]. Beliefs older than the
  chunk come from the detached cache (gradient to the head only), as for distance 1 at a chunk's first frame.
- **Weighting (Codex's review, adopted): L1 + mean(L2..L8)** in G4 with the one-step continuation forecast (unchanged).
  Distance 1 keeps its full weight (the objective that taught recovery); distances 2-8 together count as one more
  term, each averaged over its elements as LOSSES principle 1 requires. At initialisation: 11.15 nats against 5.61 for
  F5's reward forecast (the raw sum would be 44.42); first-update gradient norm 13.3 against F5's 9.3 (sum 45.3).
- Unchanged: G1 (shortcut forcing), reactive TSP, fovea reconstruction, the reactive reward and continuation heads,
  the learned gaze, the harmoniser, the corpus, the optimiser, the curriculum, the schedule (25.6M frames).

### Against Dreamer 4 (section 3.3, verified 11-14)

Dreamer 4 trains its policy and reward heads with multi-token prediction of length L = 8: distances n = 0..8, an
unweighted sum, small MLPs with one output layer per distance, a symexp two-hot reward, conditioned only on the task
embedding at t. It does this in phase 2 (agent finetuning) with the video loss still applied, it normalises every loss
term by a running RMS, and its dynamics model runs on a frozen tokenizer. Two consequences for us: the scale of its
sum is neutralised by the RMS normalisation, so averaging is what its sum amounts to; and its reward heads never shape
the tokens, whereas ours reach the encoder through Mamba's inputs (agreed: "Encoder trained jointly with the world").
Here: n = 0 is our reactive head, n = 1 F5's forecast mode, n = 2..8 new; our reward stays a categorical over symlog
bins; there are no policy heads (P1 has no actor); the heads train from the first update, as F5's forecast modes do.

### What the weighting does to the loss share (measured on F6, for review)

- **Adam rescales each parameter's gradient by its own running size**, so a loss weight changes only the share of
  modules that several losses train (LOSSES principle 3). Parameters one loss trains alone (denoiser, decoders, head
  layers) learn at their own pace; their only cost is the clip: every one of F6's 626 logged updates was clipped at
  1.0 (median norm 2-5), and the step-to-step variation of the clip factor costs them 4-10% of their effective step
  (E[c] / sqrt(E[c^2]) = 0.90-0.96 over F6's log).
- **Mamba is already the reward forecast's.** On F6's checkpoint-10 training chunk (`../20261114_f6_dissection`,
  `batch_10.json`): the reward forecast's weighted gradient into the belief is 4.54 against G1's 0.031 (about 150x), its
  cosine to the total 0.997, and G1's 0.02 (orthogonal: no conflict in direction, only in scale).
- **The encoder is where a larger reward share could cost the rest.** F6's logged per-term gradients into the encoder
  (raw, unweighted; group weights are within 2.0-2.7 of each other): the reward forecast's is 1.4-62x the reactive
  TSP's, the periphery tokens' designated owner (LOSSES 7b). Adding the reward forecast from F4 to F6 cost little there
  (TSP 2.22 against 2.28 nats per pixel), so a step of up to 2x is within what we have seen; the raw sum's 2.8-8x is not.
- **The harmoniser does not absorb this.** Its divisors fell from 1.0 to 0.38-0.51 in F6 (weights rose to 1.96-2.68,
  Codex's correction of my earlier reading). Its stationary point is sigma = (L + sqrt(L^2 + 4L)) / 2 (HarmonyDream App.
  B), about sqrt(L) for small losses: G4 sat near it (0.51), G1 far above it (0.38 against about 0.07).
- Expected reward-forecast pull on Mamba and the encoder against F6's (arithmetic, between uncorrelated and aligned
  distances, G4's divisor at its stationary point): L1 + mean about 1.1-2.2x; the raw sum about 1.4-3.9x late in
  training and up to 8x early, while the divisor is still near 1. The sleep price for the run is not additive across
  distances (Codex): the 2-8 step gains are largely the same phase information, so the earlier "0.29 nats, four times
  recovery's" was withdrawn.

## Preflight (`checks.py`, before launch; results in `checks.json`; the raw-sum variant's in `checks_sum_variant.json`)

- Same seed: every F5 parameter bitwise equal (TSP's forecast mask dropped), the only new parameters are the seven
  output layers, global RNG after construction equal to F5's.
- The F5 recipe and the F6 recipe under the changed code each reproduce their logged first update exactly.
- The arm's first update: every term F5 also has equals F5's logged value; distance 1 equals F5's reward forecast; no
  visual forecast term.
- Known answer: for each distance, the rows the head reads are exactly the predicted beliefs of frame t - d + 1 and the
  rewards of frame t on the pairs found independently from episode ids; the grouped term is distance 1 plus the mean of
  the others in the trainer's order; each distance's term reaches Mamba. (Codex separately checked the episode-boundary
  mask against an independent reference on 1.79M synthetic decisions, both chunk lengths.)
- Exact resume (2 + 2 = 4 updates), finite 12-update smoke, peak memory (2.54 GiB, 1.9 s per update).
- `eval_checks.json`: the evaluation queue's smoke on a fixture checkpoint from the preflight's short run. The stages
  the smoke does not cover were run on F6's own checkpoints (`reference.py`, below), which also gives the review its
  F6 numbers.

## Readings (fixed before any result)

**Primary: waking on the quiet sleep bank** (`stages.py sleep` + `sleep_split`; 32 recorded asleep scenes at energy 8,
no mob within 6 cells, NOOP, 8 draws x 48 frames), split by whether WAKE_UP had already paid at the start (25 not yet,
7 already). Two readings per cohort:

1. **Phase (Codex's test).** Per scene, the 1-Wasserstein distance between generated and simulator wake times (no wake
   = 49); the real pairing against generated samples shuffled across scenes of the same cohort (10,000 shuffles). The
   phase advantage is the shuffled mean minus the real distance. Known answers on F6's archived draws, not-yet-paid
   cohort: a clockless predictor (one draw at step 2, seven at step 12, every scene) has advantage 0 (p = 1.0); copying
   the simulator has 4.09 steps (p = 0.0001). **F6 at checkpoint 10: advantage 0.46 steps (real 9.31 against 9.77,
   p = 0.006), so F6 already beats the shuffle slightly; the bar is F6's advantage, not the shuffle.**
2. **Rollout quality.** Awake and alive by steps 8 / 12 / 16 / 32 / 48 against the simulator, the restricted wait ratio
   (bootstrap over scenes), early wakes (before the scene's earliest simulator wake), late or missing wakes. A clockless
   predictor can improve these (Codex: wait ratio 2.25 -> 1.60) while knowing nothing about which scene is due, so they
   are read only beside the phase reading.

F6, not-yet-paid cohort (25 scenes), checkpoint 10: awake by step 12 0.46 (simulator 1.00), wait ratio 2.25, early 0.155,
late or none 0.775, phase advantage 0.46 steps. Already paid (7 scenes): 0.286, 2.97, 0.214, 0.768, no advantage (p 0.95).
F6 at checkpoint 3 (`readings/reference_f6/ckpt_03/followups/sleep_split.json`), not-yet-paid cohort: awake by step
12 0.08, wait ratio 5.49, early 0.03, late or none 0.955, phase advantage 0.13 steps (p 0.08): no phase yet.

- Supports the claim: on the not-yet-paid cohort, a phase advantage clearly above F6's at the same checkpoint with a
  lower real distance, and late wakes falling without early wakes rising.
- Supports (b): the not-yet-paid cohort improves and the already-paid cohort does not (descriptive: 7 scenes).
- Supports (a): the readout below shows the heads using the sleep phase, with the bank unchanged.

**The new heads (`readout.py ahead`, Codex).** On held-out DEV asleep origins, each distance's reward log-loss with two
donors' predicted beliefs in place of the origin's own: asleep origins of other episodes with the same energy and health
digits and WAKE_UP status, one at the same simulator fatigue phase (the control: everything else about a foreign
belief) and one at a different phase (labels choose donors only). The phase effect is the different-phase donor's loss
minus the same-phase donor's, split by WAKE_UP status, with episode-bootstrap intervals; each distance's own-belief
log-loss on every frame beside it. A phase effect above F6's distance-1 effect at the same checkpoint
(`reference_f6/ahead_0k.json`) means the forecast uses the phase; a better every-frame log-loss without it means the
heads mostly learned to predict zero reward. F6, distance 1, checkpoints 1 / 2 / 3: phase effect 0.0001 / 0.0035 /
0.0066 nats (intervals include 0; not-yet-paid 0.0093 [-0.011, 0.028] at 3) against a foreign-belief effect of 0.13 /
0.31 / 0.39: without the same-phase control almost all of a donor gap is not phase. Every-frame log-loss 0.177 /
0.139 / 0.115.

**The share, measured (`readout.py budget`).** Each term's weighted gradient into Mamba's module and into the encoder,
with cosines to the total, on the checkpoint's own next training chunk (8 streams x 64 frames), at checkpoints 1-3,
beside F6's at 1-3 with the same instrument (`reference_f6/budget_0k.json`); the per-term encoder norms logged by the
trainer at each checkpoint, beside F6's log. F6, checkpoints 1 / 2 / 3, weighted norms into Mamba's module: reward
forecast 1.63 / 3.91 / 3.05, reactive reward 1.41 / 2.42 / 0.98, G1 0.007 / 0.012 / 0.012, token forecast 0.05 / 0.06 /
0.04; into the encoder: reward forecast 3.0 / 7.8 / 6.1, reactive reward 5.9 / 5.6 / 2.8, TSP 0.69 / 2.04 / 0.92,
reconstruction 0.11 / 0.33 / 0.14. Read at the review: the reward heads' share of the encoder against TSP and
reconstruction (F6: about 6-10x), G1's share of Mamba's.

**Secondary.** The asleep energy rise read from the predicted belief, by checkpoint (F6 0.218 / 0.214 / 0.214 at
checkpoints 1-3, 0.180 at 10; base rate 0.086), and the generated rise hazard per asleep step (F6 0.069, flat over 48
steps). **Predicted negative:** food, drink and awake-energy drops from the belief at their base rates (F6 0.038 / 0.046
/ 0.034); hunger and energy exact-history pairs flat.

**Guards, beside F6 at the same update or checkpoint.** The logged G1, TSP and reconstruction terms and the monitor's
G1; recovery and cooldown (belief AP, F6 0.05 / 0.24 / 0.34 at checkpoints 1-3); zombie readiness. At the full run's
end: danger, C2ST, four-step recursion and entities, coal and iron. A regression here beside a larger reward share on
the encoder points to the share (or to the removed token head; the bundle cannot tell them apart).

## Schedule and launch

Segment 1 trains to checkpoint 3 (7,500 updates, about 4-5 h) and then reads it: `readout.py` (trajectory, registration,
trace, budget, ahead) at checkpoints 1-3, and the sleep stage with its split at checkpoint 3 into `readings/early_03/`.
**Then it stops for review (Codex): the user decides whether to continue.** In F6 neither recovery nor cooldown was drawn
timed by checkpoint 3 (pairs 0.005 / 0.000 and 0.067 / 0.037), so a null in generation there is not a failure; the
heads' readout, the belief probe and the guards are the early reads. Continuing is `run.py` (F5's and F6's one-shot
worker): it resumes exactly to the full budget, applies the fixed checkpoint rule and runs `evaluation.py`. The saccade
sources must not change between segments (resume checks their hashes; `run.py` checks `SOURCE_PINS.json`).

    systemd-run --user --unit=saccade-p1f61 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" bash -c 'set -e; X=artifacts.experiments.20261114_p1f61; D=artifacts/experiments/20261114_p1f61
        .venv/bin/python -u -m $X.train $D/run --updates 7500 >> $D/training.log 2>&1
        for k in 1 2 3; do .venv/bin/python -u -m $X.readout $k >> $D/early.log 2>&1; done
        for s in sleep sleep_split; do .venv/bin/python -u -m $X.stages $s $D/run/ckpt_03.pt $D/readings/early_03 >> $D/early.log 2>&1; done'

Continuation, after review only:

    systemd-run --user --unit=saccade-p1f61-full -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261114_p1f61.run
