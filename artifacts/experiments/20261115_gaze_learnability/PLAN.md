# Can the gaze forecast teach Mamba the pixel clocks? (plan, 2026-11-15; revised after Codex's review)

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
the gaze forecast at its extreme, Mamba with the whole optimiser to itself, in the instrument where a direct request
(A1, simulator labels) built the energy clocks.

- **Claim.** The gaze forecast, as the only loss, makes F6's Mamba carry the pixel clocks' phase.
- **Competing explanations.** (a) It reinforces only the trace the belief already holds (the asleep rise: F6's AP 0.17)
  and cannot seed the absent awake clocks, as the change-form forecast's request could not (11-13); (b) it teaches a
  correlate of elapsed time, not the phase: Codex's reader of the episode step and the previous frame's lighting alone
  reaches natural AP 0.33 food, 0.36 drink, 0.24 awake energy (`time_light`, reproduced exactly below), far above F6's
  belief (0.04); (c) the probe or head already had the phase in F6's belief.
- **Separating readings.** (c): everything is read against A0, F6's belief unchanged, by the same probes on the same
  rows and pairs. (b): the belief's held-out gain over the time-and-light reader, and F6's exact-history banks, whose
  two branches share the episode step, the last 8 frames and their lighting. (a): read per clock.

## Instrument (`gaze.py`)

`../20261114_learnability/learnable.py` unchanged in everything but the loss: F6 checkpoint 10, perception frozen,
windows of 128 consecutive steps from random training episodes, the belief run from its zero state, uniform prior
fixations, 32 windows per update, 3,000 updates, Adam (head 1e-3, belief 1e-4), 100 warm-up updates, gradient clip 1,
`default_rng(7)` draws in learnable.main's order (A1's windows and fixations exactly), head initialised after
`torch.manual_seed(2)`.

- **Target:** at every window frame t = 1..127, F6's frozen final chooser's 49-window distribution
  (`../20261113_p1f6/run/chooser_last.pt`, `chooser.policy` of its contingency and surprise, fp32, from the 8 frames
  before t in the episode, including frames before the window, the action into t and frame t), as in the gaze EDA and
  Codex's CPU fits. No gradient reaches the chooser. Codex checked the alignment on CPU (NOTEBOOK 11-15): editing only
  the target frame, its arrival reward and outgoing action changes the teacher's distribution and leaves the student's
  8-frame inputs and predicted belief bitwise identical.
- **Head:** fair.py's forecaster (the chooser head's trunk on the chooser's own 8-frame inputs, a score per cell, a
  window's logit the sum of its nine cells' plus a bias) with one more input added to every cell: a linear map of the
  predicted belief before frame t (oracle.py's structure, the belief in place of the exact counters). The explicit
  8-frame baseline means the belief is asked only for what the window lacks (11-13: a head denied the present spends
  its error on the present). Loss: soft cross-entropy to the chooser's distribution.
- **One trained arm** (user, 11-15: no control arms; every pressure that does not ask has come back without clocks,
  A5 the latest). Dropped from the first version with it: the belief-zeroed and frozen-belief head arms (they could not
  change Mamba; they served only head-level readings), the A5 re-read (a different window, carry and optimiser
  contract, Codex), and the gradient-norm dose (Codex: a shared-batch norm does not give the direction of change on
  matched phase; a joint run will need its own dose at its own starting point).
- **A0 reading** (F6's belief unchanged, no training): predicted beliefs on all DEV and on the banks, saved for the
  summary. Known answer, asserted before training: its natural probe APs reproduce `reference.json` to 0.002.

## Readings (fixed before results)

Probes are learnable.probe's (one linear logistic probe per event on the standardised predicted belief, fit on DEV
folds 0-3, read on fold 4), refitted on CPU from the saved beliefs for A0 and the trained belief at 750, 1,500 and
3,000 updates. The time-and-light reader is Codex's (LightGBM on the episode step and the previous frame's lighting,
31 leaves, learning rate 0.06, min leaf 50, 200 rounds, seed 0), fit per event on the same rows; for fold 4 from folds
0-3, for folds 0-3 cross-fitted by fold, so a probe fitted on top never sees a reader trained on its own rows.

**Per pixel clock, at 1,500 and at 3,000 updates:**

1. **Screen** (a threshold, not a finding; A1 has another target and head): natural AP at least a quarter of the way
   from A0 to A1 at 1,500 updates: food >= 0.080, drink >= 0.085, awake energy >= 0.189, asleep rise >= 0.367.
2. **Beyond time and light:** a probe with the reader's logit as a fixed offset; held-out log-loss per eligible row,
   trained belief against A0's on the natural rows; episode-bootstrap (10,000 draws, seed 215) 95% lower bound of
   A0's loss minus the trained belief's above zero.
3. **Exact pairs** (awake energy and food: the banks of 32 awake energy drops and 24 food drops): the natural probe's
   logit, due branch minus delayed, trained minus A0 on the same pairs; pair-bootstrap 95% lower bound above zero.

Readings: **phase taught** when 1-3 hold (awake energy, food); **event readability beyond time and light** when 1-2
hold (the most drink and the asleep rise can show here, which have no exact-pair bank; or awake energy and food
without 3). **Drop the target** when at 3,000 updates no pixel clock passes 2 (the whole budget). Anything else is
for the user. A null bounds this gaze-only continuation from F6 at this budget, not training from scratch or a joint
recipe. One seed suffices for a screen (Codex); a result near a bar gets a second seed before any decision.

**Reported, no bars:** reading 2 on the two cohorts, named by what they select (`steady8`: the last 8 frames in one
sleep state with the digit unchanged; `after_reset`: steady8 plus an earlier intake or sleep for food / drink, an
earlier sleep for awake energy, WAKE_UP already earned for the asleep rise) and each cohort's time-and-light AP;
recovery (no reward request here; A5 lost 0.26 of its AP); the trained head on the banks, forecast and chooser
coverage of the digit, due and delayed, and the forecast's coverage KL to the chooser's.

## Bounds

A fine-tune from F6's checkpoint with the whole budget, not a joint or from-scratch result; perception frozen; prior
fixations; a fixed teacher (F6's final chooser), where a from-scratch chooser would co-evolve; one training seed; linear
probes. The cohorts, banks and time-and-light reader use simulator labels as evaluation devices only; nothing in the
loss names the HUD.

## Run (not queued: launched by hand after F6.1's evaluation, with the user's go)

```sh
systemd-run --user --unit gaze-learnability -p MemoryMax=16G --working-directory=$PWD \
  --setenv=MAMBA_DETERMINISTIC=1 --setenv=PYTHONPATH=$PWD \
  /bin/sh -c ".venv/bin/python -u -m artifacts.experiments.20261115_gaze_learnability.gaze gpu \
  > artifacts/experiments/20261115_gaze_learnability/gpu.log 2>&1"
# then, on CPU (GPU free):
MAMBA_DETERMINISTIC=1 GAZE_CPU=1 JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= \
  .venv/bin/python -m artifacts.experiments.20261115_gaze_learnability.gaze summary   # result.json
```

The first update is the GPU shape check at the real batch (4,064 scored frames with retained gradients): it logs peak
memory, and the run is stopped by hand if that exceeds 5 GiB or the logged seconds per update imply more than a few
hours. Time is not yet measured (A1: 0.3 s per update; this adds the chooser and the trunk on 4,064 frames). Saved in
`run/` (not in git): the trained belief and head, the predicted beliefs on DEV and on the banks at each reading.

**Checks before review (CPU).** Smoke (`GAZE_SMOKE=1 GAZE_OUT=<scratch> ... gaze all`: 10 DEV episodes, windows 16 x 2,
3 updates, Mamba's reference scan, two pairs per bank) runs every step. It caught the chooser being run on an
episode's first frame, which has no previous action (fixed by scoring frames 1.. only, as fair.py). The bank reader
reproduces Codex's chooser read on every pair (coverage due / delayed: energy 0.5026 / 0.1333, 32 pairs; hunger
0.5526 / 0.1014, 24; recovery 0.3808 / 0.0745, 32; rising in every pair). `time_light` reproduces Codex's held-out APs
exactly in all 15 cells (natural food 0.3289, drink 0.3585, awake energy 0.2426, asleep rise 0.0970; after_reset
0.2132, 0.2413, 0.0432, 0.2543).
