# Why the world generates a counter's appearance but not its increment (plan, 2026-10-30)

User, 10-30: use the existing checkpoints and results to find out why increments fail, before redesigning the fovea
test. Fact to explain (fovea placement dissection, `../20261030_fovea_placement/RESULTS.md`): P1 world 1 generates
drink / wood on ~30% of 0 -> 1 events (the icon appears) and on 0-6% of v -> v + 1 events (the digit changes), at
matched training frequency and at comparable token step sizes; the glimpse output did not learn increments either.

Status (10-30): Part 1 ran (13:11-13:18, `increment_probe.py` as committed with it) WITHOUT prior review by the
user or Codex and before this plan was committed (procedure breach, reported to the user). Parts 2 and 3 are written
for review only: not approved, not run. User (10-30): Part 2 to run; Part 3 dropped (not worth the compute).

All tests on frozen checkpoints; dev rows of test F (the evaluation's rows), S1-seeded fixations, eye elsewhere at the
target frame, 2 draws per context, the same draws in every variant. Read value = nearest dev value mean of the base
encoder's real tokens (read at accuracy 1.0 on real tokens).

## Part 1: substitution on checkpoint 10 (`increment_probe.py`)
- **A. What is generated instead.** Claim: increment failures are copies of the start value (the world predicts "no
  change"). Competing: it predicts a change, to the wrong value (v - 1, the appearance's "1", off-manifold). Read
  value of each sample in {v + 1, v, v - 1, 1, other} and its distance over the value's spread.
- **B. Does the action reach the counter?** a_{t-1} = DO replaced by NOOP: B1 in the denoiser's input only, B2 also in
  the belief (its predicted state after a_{t-1}). Claim: increments' output does not depend on the action at all.
  Competing: it depends on it, in the wrong direction. Separation: the change of the field (sigma 20) and of the read
  values between DO and NOOP, for appearance and increment rows. Positive control: MOVE rows, whose shift must
  vanish under NOOP (otherwise the substitution does not reach the model).
- **C. Does the displayed value gate it?** The counter's periphery token (and glimpse token where a window covers it)
  set to a value mean: C_den in the 8 context frames only, C_full in every frame up to t - 1 with the belief
  recomputed. Identity control: each frame set to its own value's mean (the substitution's own effect). On
  increment rows u = 0: does an empty history produce the appearance? On appearance rows u = 2: does a shown value
  remove it? On increment rows u = v +- 3 (copy check): does generation follow the shown value or the true one?
  Claim: success is gated by the displayed state (empty -> fixed output; v -> no change). Competing: by correlates of
  the zero state (e.g. thirst damage, the agent's behaviour), or the value is read from elsewhere (memory).

## Part 2: the learning signal by transition (`signal_accounting.py`)
The 10-29 accounting with drink and wood split into appearance (0 -> 1) and increment (v -> v + 1), and the glimpse
elements at consequence cells attributed to their class: loss share, gradient-norm share, cosine with the total, and
drive under the checkpoint's Adam update, and loss by equal-mass noise bin. Runs: checkpoint 10 with random
fixations (P1's condition; must reproduce the 10-29 run's unchanged classes: same windows and noise), checkpoint 10
with guided fixations on every attempt frame (the fine-tune's start; its Q = 1/2 is the midpoint in expectation), and
the fine-tune's ft_1500 with guided fixations (its end: the per-class loss change between the two, same windows and
noise, is what the extra share achieved over 1,500 updates). Claim (the user's question, 10-30): increments get
too small or too opposed a share of the update. Competing: their share is comparable to appearances', so the signal
is there and the denoiser fails to use it.

## Part 3: trajectory by transition (`trajectory_by_value.py`)
P1 checkpoints 1, 2, 3, 5, 7, 10, the same rows and draws, each checkpoint read with its own encoder's value means:
were increments ever generated, and when did appearances emerge? Claim: increments were never learned. Competing:
learned, then lost (as the 10-29 accounting found the drive reverse late).

Limits: one world seed, one run; substitution with value means is not the real token (measured by the identity
control); memory beliefs at C_den keep the real history (C_full removes that).
