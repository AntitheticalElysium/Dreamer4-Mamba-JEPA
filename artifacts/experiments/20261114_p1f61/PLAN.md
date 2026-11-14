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

## The arm (`train.py`; F5's recipe plus `heads.forecast` = `{tsp: false, reward_distances: 8, reward_init_seed: 615003}`)

- **No visual forecast at all.** F6's token forecaster is removed and TSP's forecast mode is not restored
  (`tsp: false`): F6's glimpse prediction recovered when that mode was removed (2.28 against F5's 2.57 nats per pixel).
- **Reward forecast, distances 1-8.** Frame t's reward is predicted from the belief predicted before frame t - d + 1,
  with frame t's percept replaced by the forecast mask, for d = 1..8. Distance 1 is F5's and F6's forecast mode,
  unchanged (it shares the reactive head's output layer). Distances 2-8 each have their own output layer (7 x 65,792
  parameters, 460,544 in all) on the shared outcome trunk, built under their own seed so every other parameter and the
  training RNG keep F5's draws. A pair counts only when no episode starts in [t - d + 1, t]. Beliefs older than the
  chunk come from the detached cache (gradient to the head only), as for distance 1 at a chunk's first frame.
- **The sum goes into G4** with the one-step continuation forecast (unchanged): Dreamer 4's Eq. 9 sums the distances
  without weights.
- Unchanged: G1 (shortcut forcing), reactive TSP, fovea reconstruction, the reactive reward and continuation heads,
  the learned gaze, the harmoniser, the corpus, the optimiser, the curriculum, the schedule (25.6M frames).

### Against Dreamer 4 (section 3.3, verified 11-14)

Dreamer 4 trains its policy and reward heads with multi-token prediction of length L = 8: distances n = 0..8, an
unweighted sum, small MLPs with one output layer per distance, a symexp two-hot reward, conditioned only on the task
embedding at t. It does this in phase 2 (agent finetuning) with the video loss still applied, and it normalises every
loss term by a running RMS. Here: n = 0 is our reactive head, n = 1 F5's forecast mode, n = 2..8 new; our reward stays a
categorical over symlog bins; there are no policy heads (P1 has no actor); the heads train from the first update, as
F5's forecast modes do.

### What the sum does to the loss share (for review)

Our stand-in for Dreamer 4's RMS normalisation is the harmoniser, and it barely moves: in F6 every group's weight
drifted from 1.0 to 0.38-0.51 over the whole run. So the raw sizes decide the shares. The sum makes the reward forecast
about 8 times F6's (update 1: 44.5 nats against 5.6), and the first-update gradient norm is 45.3 against F5's 9.3
(every update in F6 was clipped at 1.0; median norm 2-5). The reward heads already supplied about 98% of the belief's
gradient in F6; this raises their share further. The alternative is the mean over distances, which keeps F6's total
reward-forecast weight but gives sleep's phase about 0.035 nats per asleep frame, half of what recovery had. The sum is
implemented (Dreamer 4's equation); the mean is a one-line change and a rerun of the preflight.

## Preflight (`checks.py`, before launch; results in `checks.json`)

- Same seed: every F5 parameter bitwise equal (TSP's forecast mask dropped), the only new parameters are the seven
  output layers, global RNG after construction equal to F5's.
- The F5 recipe and the F6 recipe under the changed code each reproduce their logged first update exactly.
- The arm's first update: every term F5 also has equals F5's logged value; distance 1 equals F5's reward forecast; no
  visual forecast term.
- Known answer: for each distance, the rows the head reads are exactly the predicted beliefs of frame t - d + 1 and the
  rewards of frame t on the pairs found independently from episode ids; the grouped term is the unweighted sum in the
  trainer's order; each distance's term reaches Mamba.
- Exact resume (2 + 2 = 4 updates), finite 12-update smoke, peak memory.
- `eval_checks.json`: the evaluation queue's smoke on a fixture checkpoint from the preflight's short run. The new
  sleep split was checked on F6's archived sleep draws: it reproduces F6's sleep stage (awake by step 12 0.422, wait
  ratio 2.44) and Codex's split (0.46 on the 25 starts where WAKE_UP had not paid).

## Readings (fixed before any result)

**Primary: waking on the quiet sleep bank** (`stages.py sleep` + `sleep_split`; 32 recorded asleep scenes at energy 8,
no mob within 6 cells, NOOP, 8 draws x 48 frames), split by whether WAKE_UP had already paid at the start (25 not yet,
7 already). Per cohort: awake and alive by steps 8 / 12 / 16 / 32 / 48 against the simulator's draws, the restricted
wait ratio (bootstrap over scenes), early wakes (before the scene's earliest simulator wake) and late or missing wakes
(after its latest). F6 at checkpoint 10:

| Cohort | Awake by step 12 (simulator 1.00) | Wait ratio | Early | Late or none |
| :--- | ---: | ---: | ---: | ---: |
| WAKE_UP not yet paid (25) | 0.46 | 2.25 | 0.155 | 0.775 |
| Already paid (7) | 0.286 | 2.97 | 0.214 | 0.768 |
| All (32) | 0.422 | 2.44 | 0.168 | 0.773 |

- Supports the claim: the not-yet-paid cohort's wait ratio falls below F6's 2.25 with its interval excluding it, with
  late wakes falling and early wakes not rising.
- Supports (b): the not-yet-paid cohort improves and the already-paid cohort does not. With 7 scenes that cohort is
  descriptive only.
- Supports (a): the belief probe rises (below) with the bank unchanged.

**Secondary.** The asleep energy rise read from the predicted belief, by checkpoint (F6 0.218 / 0.214 / 0.214 at
checkpoints 1-3, 0.180 at 10; base rate 0.086), and the generated rise hazard per asleep step (F6 0.069, flat over 48
steps) from the sleep stage. **Predicted negative:** food, drink and awake-energy drops from the belief at their base
rates (F6 0.038 / 0.046 / 0.034); hunger and energy exact-history pairs flat.

**Guards, beside F6 at the same checkpoint.** Recovery and cooldown (belief AP, F6 0.05 / 0.24 / 0.34 at checkpoints
1-3; pairs drawn at 10, F6 0.466 / 0.000 and 0.912 / 0.048), danger, the monitor's G1 loss against F6 at the same
update, reactive TSP, C2ST, four-step recursion and entities, coal and iron. **This is also the first reading of what a
larger reward share costs the world:** a regression in G1, TSP, recursion or entities beyond F6's points to the share
(or to the removed token head; the bundle cannot tell them apart).

## Schedule and launch

Segment 1 trains to checkpoint 3 (7,500 updates; F6 took 1.4-1.8 h per checkpoint, about 5 h) and reads it: the clock
readout (`readout.py`: trajectory, registration, trace) at checkpoints 1-3 and the sleep stage with its split at
checkpoint 3 into `readings/early_03/`. In F6 neither recovery nor cooldown was drawn timed by checkpoint 3 (pairs 0.005
/ 0.000 and 0.067 / 0.037), so a null in generation at checkpoint 3 is not read as a failure; the belief probe and the
reward-forecast separation are the early reads. Segment 2 (`run.py`, F5's and F6's one-shot worker) resumes exactly to
the full budget, applies the fixed checkpoint rule and runs `evaluation.py` (F5's queue without the forecast scores and
control pairs, then `stages.py`). The two segments run back to back in one service; the user can stop it after
reading checkpoint 3. The saccade sources must not change between segments (resume checks their hashes; `run.py` checks
`SOURCE_PINS.json`).

    systemd-run --user --unit=saccade-p1f61 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" bash -c 'set -e; X=artifacts.experiments.20261114_p1f61; D=artifacts/experiments/20261114_p1f61
        .venv/bin/python -u -m $X.train $D/run --updates 7500 >> $D/training.log 2>&1
        for k in 1 2 3; do .venv/bin/python -u -m $X.readout $k >> $D/early.log 2>&1; done
        for s in sleep sleep_split; do .venv/bin/python -u -m $X.stages $s $D/run/ckpt_03.pt $D/readings/early_03 >> $D/early.log 2>&1; done
        .venv/bin/python -u -m $X.run'
