# Separating local prediction, feedback drift, history extrapolation and decoder mismatch

Status: **complete**, 2026-09-17. Diagnostic over frozen checkpoints and the worlds trained
by [`20260917_state_transition`](../20260917_state_transition/). Trains nothing, changes no
objective. Runner: [`drift.py`](drift.py), evidence in [`evidence/drift.json`](evidence/drift.json).

**It overturns the headline conclusion of [`20260917_closed_loop`](../20260917_closed_loop/).**

## Two defects in that experiment

**The memory lengths never matched.** The worlds were trained on windows of **three**
action-pairs with memory reset every batch. The closed-loop evaluation prefilled three pairs
and then advanced eight more, so *every* scored prediction — the first included — sat beyond
the trained history length. Feedback drift and history extrapolation were measured together
and reported as if they were one thing.

**Termination was never tested.** That sampler requested one frame and one action it never
used, which excluded every episode-ending transition, so `termination` had no positives and
was silently dropped from the label set. Sampling is fixed here — episode-ending transitions
are now drawn — but **termination is still not tested**: 7 DEV positives is below the probe's
coverage floor, so it reports `insufficient_coverage` at every depth and contributes to no
reported average. The fix makes the gap visible; it does not close it.

## The factorial

Normalized MSE against the truly-encoded latent, divided by that latent's own variance.
Trained history is 3 pairs.

| cell | d1 | d2 | d4 | d6 | d8 |
|---|---|---|---|---|---|
| `z` persistence | 0.872 | 1.131 | 1.516 | 1.689 | 1.768 |
| `z` teacher, trained-length | 0.074 | 0.075 | 0.068 | 0.077 | **0.091** |
| `z` teacher, persistent | 0.086 | 0.089 | 0.083 | 0.092 | 0.105 |
| `z` generated, trained-length | 0.074 | 0.092 | 0.120 | 0.175 | **0.239** |
| `z` generated, persistent | 0.086 | 0.107 | 0.163 | 0.269 | 0.322 |
| `u` persistence | 0.099 | 0.201 | 0.325 | 0.498 | 0.637 |
| `u` teacher, trained-length | 0.036 | 0.036 | 0.030 | 0.046 | **0.072** |
| `u` teacher, persistent | 0.054 | 0.056 | 0.057 | 0.072 | 0.095 |
| `u` generated, trained-length | 0.036 | 0.075 | 0.152 | 0.265 | **0.373** |
| `u` generated, persistent | 0.054 | 0.128 | 0.331 | 0.588 | 0.863 |

### `u` does beat persistence — when evaluated as it was trained

Skill at depth 8 (model ÷ persistence):

| arm | trained-length memory | persistent memory |
|---|---|---|
| `z` | 0.135 | 0.182 |
| `u` | **0.586 — beats holding still** | **1.356 — loses to it** |

The previous report's "`u→u` loses to holding still from depth 3" was measured **only** in
the persistent-memory column. Matched to its training, `u` beats persistence comfortably.
That conclusion is withdrawn.

### Local prediction is excellent for both, and *better* for `u`

Teacher-fed at trained length, depth 8: `u` 0.072 against `z` 0.091. `u` is not harder to
predict — it is easier, and the world learned it. So "the one-step advantage was an easier
prediction task" is also withdrawn: easiness shows up in the *persistence baseline*, not in
the model's own accuracy, and conflating those was the error.

### What actually breaks: 182 abandoned coordinates

Error split by the variance partition that raw-coordinate MSE optimizes. The ten
highest-variance coordinates hold **93.1%** of `u`'s variance and **12.9%** of `z`'s:

| cell (depth 8) | all | top-10 | remaining 182 |
|---|---|---|---|
| `z` generated, trained-length | 0.239 | 0.196 | 0.246 |
| `z` teacher, trained-length | 0.091 | 0.055 | 0.096 |
| `u` generated, trained-length | 0.373 | 0.327 | **0.949** |
| `u` teacher, trained-length | 0.072 | 0.060 | **0.220** |

For `z` the two partitions track each other. For `u` the remaining 182 coordinates reach
**0.949** — barely better than predicting their mean — while the top ten stay at 0.327. The
world preserved the ten numbers its loss was almost entirely about and abandoned the rest.

This is a **measurement mismatch, not a representation verdict**: the world trains on raw
MSE, where `u`'s minor coordinates are worth 6.9% of the objective, while the probe
standardizes every coordinate and weights all 192 equally. The world did what it was asked.

## Decoding is mixed, and the instrument is weak

Native readouts, probe fitted on each condition's own latents. Each cell is a mean over the
labels **measured at that depth**, which is 2 or 3 of the 4 — `termination` never qualifies,
and `negative_reward` qualifies only at some depths. **The label set therefore changes between
columns, so these numbers are not comparable across depth.** They are only comparable within a
column, between a world and its own persistence baseline.

| cell | d1 (n=2) | d2 (n=3) | d4 (n=2) | d6 (n=2) | d8 (n=3) |
|---|---|---|---|---|---|
| `z` persistence | 0.7919 | 0.6706 | 0.7129 | 0.8499 | 0.7123 |
| `z` generated, trained-length | **0.8105** | 0.6518 | **0.7486** | 0.8381 | 0.6864 |
| `u` persistence | 0.7371 | 0.6783 | 0.6841 | 0.7657 | 0.6843 |
| `u` generated, trained-length | **0.7910** | **0.7056** | 0.6728 | **0.7795** | 0.6828 |

Counting all eight depths, the generated latent decodes better than persistence at **3 of 8**
for `z` and **4 of 8** for `u`. So the result is **mixed**, and no paired test was run, so no
consistent advantage is established in either direction.

An earlier revision of this file claimed neither world beats persistence "at any depth". That
is numerically wrong — `u` reaches 0.7910 against 0.7371 at depth 1 — and is withdrawn.

Reward and achievement are strongly action-driven, and two of four labels are unusable, so
this label family is a weak instrument either way. The richer M03 successor-state panel is
the one that would settle it, and it was not used here.

## What this establishes, and what it does not

Established: the closed-loop inversion was an evaluation artifact; `u`'s local dynamics are
learned well; and `u`'s degradation under feedback is concentrated in exactly the coordinates
its training objective de-weights, rising from 0.220 teacher-fed to 0.949 generated while the
top ten go 0.060 to 0.327.

Not established: that standardizing the world's loss would fix it — that is an untested
intervention, not a conclusion. Nor that all 182 coordinates were "abandoned": 0.949 is a
mean over them, and the split reported here is two buckets, not a per-coordinate profile.
Nor anything about semantics — the decoding above is mixed and its instrument is weak. Nor does this isolate CLS, pooling, projection, predictor capacity
or action conditioning; there is still no `CLS→CLS` world here, and the arms differ in both
representation and trained world. One seed, 4,000-step worlds, frozen encoder, 512 DEV
windows, and `termination` is below the coverage floor throughout, so the label set is
effectively three at best.
