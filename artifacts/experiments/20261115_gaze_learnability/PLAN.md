# Can the gaze forecast teach Mamba the pixel clocks? (plan, 2026-11-15)

User, 11-15: "Write it, I'll have Codex review it." After the gaze-forecast EDA (`../20261115_gaze_forecast_eda/`): with
F6's frozen chooser, remembering a clock's phase from before the last 8 frames makes the chooser's gaze distribution
easier to forecast (Codex's strict cohorts: repeat sleep 0.036 nats per row, drink 0.008, food 0.005, awake energy
0.004, the last only before the first sleep). That is the first of three links:

```text
older history -> predicts where the eye will look      supported (CPU, held-out episodes)
gaze loss     -> makes Mamba carry the clock phase      THIS TEST
Mamba phase   -> makes the world draw changes on time   untested for this loss (a joint run only)
```

The earlier label-free targets failed at the second link (the token forecast's push into Mamba was 1e-5 to 1e-4 of
the reward forecast's; the change-form forecast asked for nothing the belief did not already carry). This test puts
the gaze forecast at its extreme, Mamba with the whole optimiser to itself, in the instrument that already has a
known answer on both sides.

- **Claim.** The gaze forecast, as the only loss, makes F6's Mamba carry the pixel clocks' phase.
- **Competing explanations.** (a) It reinforces only the trace the belief already holds (the asleep rise: F6's AP 0.17)
  and cannot seed the absent awake clocks, as the change-form forecast's request could not (11-13); (b) it teaches a
  correlate, not the phase: steps since the episode began (awake energy before the first sleep), "look at the digit
  more often", or the reward-earned state; (c) any training on these windows moves the probe (the instrument's drift).
- **Separating controls.** (c): A5 (the denoiser alone, every update, 2,000 updates: no pixel clock moved more than
  0.006 AP) and A0 (F6's belief unchanged); the positive reference A1 (the direct event request, simulator labels,
  same instrument and draws: food 0.208, drink 0.200, awake energy 0.650, asleep rise 0.961 at 1,500 updates). (b):
  cohorts where the shortcuts fail and F6's exact-history banks (identical last 8 frames, one branch due). (a): read
  per clock.

## Instrument (`gaze.py`)

`../20261114_learnability/learnable.py` unchanged in everything but the loss: F6 checkpoint 10, perception frozen,
windows of 128 consecutive steps from random training episodes, the belief run from its zero state, uniform prior
fixations, 32 windows per update, 3,000 updates, Adam (head 1e-3, belief 1e-4), 100 warm-up updates, gradient clip 1,
`default_rng(7)` draws in learnable.main's order (so A1's windows and fixations exactly), head initialised after
`torch.manual_seed(2)`.

- **Target:** at every window frame t = 1..127, F6's frozen final chooser's 49-window distribution
  (`../20261113_p1f6/run/chooser_last.pt`, `chooser.policy` of its contingency and surprise, fp32, from the 8 frames
  before t in the episode, including frames before the window, the action into t and frame t), as in the gaze EDA and
  Codex's CPU fits. No gradient reaches the chooser.
- **Head:** fair.py's forecaster (the chooser head's trunk on the chooser's own 8-frame inputs, a score per cell, a
  window's logit the sum of its nine cells' plus a bias) with one more input added to every cell: a linear map of the
  predicted belief before frame t (oracle.py's structure, the belief in place of the exact counters). The explicit
  8-frame baseline means the belief is asked only for what the window lacks (11-13: a head denied the present spends
  its error on the present). Loss: soft cross-entropy to the chooser's distribution.
- **Arms**, same draws, schedule and head initialisation: **none** (belief input held at zero: the 8-frame baseline),
  **frozen** (F6's belief, unchanged: what it already gives the forecast), **trained** (F6's belief trained by this loss
  alone; the test). Only *trained* changes Mamba.
- **References**, read by the same probes on all DEV: A0 and A5 (`../20261114_learnability/g1_run/run/ckpt_02.pt`).
  Known answers, checked before anything else: their natural APs must reproduce `reference.json` and `g1.json` to
  0.002 (the run stops otherwise).

## Readings (fixed before results)

Probes are learnable.probe's (one linear logistic probe per event on the predicted belief, fit on DEV folds 0-3, AP on
fold 4), on the natural eligible frames and on two cohorts per event (`cohorts`, labels only):

- *strict*: eligible and the last 8 frames in the eligible sleep state with the event's digit unchanged;
- *shortcut-free*: strict and, for food / drink, an intake rise of that digit or a sleep earlier in the episode; for
  the awake energy drop, a sleep earlier in the episode; for the asleep rise, WAKE_UP already earned (Codex's
  repeat-sleep cohort); recovery: strict.

**Primary, per pixel clock, trained arm at 1,500 updates (A1's reading point):**

1. Natural AP at least a quarter of the way from A0 to A1: food >= 0.080, drink >= 0.085, awake energy >= 0.189,
   asleep rise >= 0.367 (A0 0.037 / 0.047 / 0.035 / 0.170).
2. Shortcut-free cohort AP above both A0's and A5's: episode-bootstrap (1,000 draws, held-out episodes) 95% lower
   bounds of both differences above zero.
3. Awake energy and food only (banks exist): F6's exact-history banks (32 awake energy drops, 24 food drops), the
   forecast probability that the chosen crop covers the digit, due minus delayed, pair-bootstrap (10,000 draws) 95%
   lower bound above zero.

A clock is **taught** when 1-3 hold (1-2 for drink and the asleep rise). **Drop the target** when every pixel clock's
natural AP is below A0 + 0.02 (A5's drift was at most 0.006): given Mamba to itself, the gaze request builds no
clock where the direct request built them. Anything else is reported for the user.

What each outcome rules out: *drop* rules out this target for a joint run (bounded by fine-tuning from F6 for
12.3M frames). *Taught* on some clocks rules out explanations (a)-(c) for those clocks and makes the next questions the
dose inside the joint recipe and the world drawing the clock (a joint run, planned and reviewed separately). Only the
asleep rise taught would support (a) for the awake clocks.

**Secondary (no bars):**

- AP at 750 and 3,000 updates; strict-cohort APs; recovery and its decay without any reward request (A5 lost 0.26).
- What the belief adds to the gaze forecast on held-out DEV (fold 4): KL(none) - KL(frozen) and KL(none) - KL(trained)
  over all frames, repeat sleep, food / drink strict, awake energy strict before and after the first sleep, with
  episode-bootstrap intervals. Codex's DEV-fitted oracle values are context only (other fit data and budget).
- Banks: the chooser's own coverage, due / delayed, must reproduce Codex's read of the same banks (energy 0.503 /
  0.133, hunger 0.553 / 0.101, recovery 0.381 / 0.075) to about 0.005: the bank reader's known answer.
- Dose context, for any later joint arm (the 11-14 rule): on F6's own belief with the frozen arm's fitted head, the
  gradients into the belief's parameters from the gaze forecast, F6's reward forecast (distance 1) and F6's token
  forecast, on the same 16 fresh windows (`default_rng(8)`): median norms, ratios, cosines. The token forecast is the
  known failure; its norm ratio to the reward forecast here is read against the F6 dissection's in-training ratio
  (`../20261114_f6_dissection/readings/batch_10.json`: 0.032 / 2.32, about 0.014), and the reward forecast's loss here
  against F6's training log (about 0.06), as checks that these windows stand for training.

## Bounds

A fine-tune from F6's checkpoint with the whole budget, not a joint or from-scratch result; perception frozen; prior
fixations; a fixed teacher (F6's final chooser), where a from-scratch chooser would co-evolve; one training seed per
arm; linear probes. The cohorts and banks use simulator labels as evaluation devices only; nothing in the loss names
the HUD.

## Run

One GPU job, after F6.1's evaluation queue finishes; never alongside another GPU job.

```sh
systemd-run --user --unit gaze-learnability -p MemoryMax=16G --working-directory=$PWD \
  --setenv=MAMBA_DETERMINISTIC=1 --setenv=PYTHONPATH=$PWD \
  /bin/sh -c ".venv/bin/python -u -m artifacts.experiments.20261115_gaze_learnability.gaze all \
  > artifacts/experiments/20261115_gaze_learnability/gaze.log 2>&1"
```

Steps in order (each skipped if its output exists): references, none, frozen, trained, summary (`result.json`).
Time is not yet measured: A1 took 0.3 s per update; this adds the chooser and the trunk on 4,064 frames per update. The
log records seconds per update and peak memory after the first update. Saved: the trained belief and head at each
reading (`run/`, not in git).

**Smoke (CPU, before review):** `GAZE_SMOKE=1 GAZE_OUT=<scratch>` runs every step on 10 DEV episodes, windows of 16 x 2,
3 updates, Mamba's reference scan, two pairs per bank; a code-path check only (no reading, no known-answer assert).
Run 11-15 with F6.1's evaluation on the GPU: all five steps complete (it caught the chooser being run on an
episode's first frame, which has no previous action; fixed by scoring frames 1.. only, as fair.py). Bank reader's known
answer checked on CPU on every pair: the chooser's coverage due / delayed is energy 0.5026 / 0.1333 (32 pairs),
hunger 0.5526 / 0.1014 (24), recovery 0.3808 / 0.0745 (32), rising in every pair, exactly Codex's read.
