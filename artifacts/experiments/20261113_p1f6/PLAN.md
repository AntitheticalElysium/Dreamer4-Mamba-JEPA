# P1-F6: F5 with a change-form token forecast (2026-11-13; first an early acquisition test to checkpoint 2)

Ordered by the user on 11-13 ("proceed with running this to checkpoint 1; and then extend to checkpoint 2 if
necessary"), after the frozen gate `../20261113_token_forecast_gate/` (round 3). Design: Codex's review of the same day,
adopted except where the user's order differs (the extension rule below).

## Question

The frozen gate showed what a change-form token forecast can extract from F5's existing belief: a request as aligned as
the reward forecast's for the clocks F5's belief carries (recovery, cooldown), nothing for energy and hunger. It could
not show whether this objective, present from the first update with Mamba trainable, builds the timing information
that F5's belief never held. That is the question here.

- **Claim.** Trained from scratch with the change-form token forecast, the belief takes up the energy reset and the
  forecast times the energy drop (due against not due) on the exact-history pairs. Energy is the primary read, hunger
  secondary.
- **Competing explanation.** No one-step, plain-MSE forecast of a single cell's change creates the route: the timed
  drop is 16 numbers of 1,296 once per ~25-30 steps, and the trace of the reset has to be carried by Mamba before any
  gradient can use it. Then energy and hunger stay at zero as in F5.
- **Separating control.** F5's own archived checkpoints 1 and 2 (`../20261111_p1f5/run/`), read with the same
  instruments: same seed, same initial weights for every shared parameter, same data order, same schedule. On the
  exact-history pairs both members share the current frame and the whole visual cache, so a correctly directed
  difference between them can only come from Mamba's memory.
- **What each outcome rules out.** A timed energy change rules out "the pixel clocks need a priced (reward-like)
  request". A null at checkpoint 1 rules out nothing: F5's reward forecast, the only loss that ever taught a clock, had
  taught nothing by checkpoint 1 (recovery separation -0.04, belief AP 0.05) and only began by checkpoint 2 (2.26,
  0.20). A null at checkpoint 2 is evidence against early acquisition by this one-step form, not a veto on the idea
  (Codex; F5's native recovery gain came two checkpoints after the belief's).

## The arm

F5's recipe (`../20261111_p1f5/train_p1f5.py`), one change: **TSP's masked forecast mode is replaced by the change-form
token forecast.**

    forecast of frame t's periphery tokens = frame t-1's observed tokens + head(belief predicted before frame t)

- Plain MSE over all 81 x 16 numbers, mean over valid non-first frames. Both token sets detached: the term reaches the
  head and the belief (Mamba, its percept projection, and through past percepts the encoder at 0.3x), never the frame
  it forecasts or the tokens it starts from.
- The head is the gate's `TokenHead` (cell embedding 32, belief projection 128, two hidden layers of 256; 147k
  parameters), fp32, its own initialisation seed, built last, so every other parameter and the training RNG keep F5's
  draws.
- It takes the TSP forecast's place in G4 (with the reward and continuation forecasts): one harmonised group, as in F5.
- Kept unchanged: reactive TSP, fovea reconstruction, learned gaze, reward / continuation forecasts, G1, the sampler,
  the corpus, the optimiser, the curriculum and the learning-rate schedule of the full 25.6M-frame run (not compressed).
  The run is stopped by update count: 2,500 updates = checkpoint 1 (2.56M frames), 5,000 = checkpoint 2.

Accepted confound: the arm differs from F5 by two things, the token forecast added and the TSP forecast removed.
Replacing was chosen because it is the candidate design (one forecast of the scene, fewer parts) and because the TSP
forecast asked nothing for any clock in F5 (RESULTS 11.3). A regression on what the TSP forecast did help with (drawing
the belief-held health digit, sharing test) would not be separable from the token forecast's own effect.

## Preflight (`checks.py`, before launch)

- Same seed: every F5 parameter bitwise equal (TSP's unused mask aside), only the token head new, global RNG equal.
- The F5 recipe under the changed code reproduces F5's logged first update exactly.
- The arm's first update: every term F5 also has equals F5's logged value.
- Known answer: with the head's output layer zeroed, the term equals copying the previous frame's tokens, computed
  independently from the chunk (alignment and masking).
- Exact resume (2 + 2 = 4 updates), finite 12-update smoke, peak memory.

## Readings (`readout.py`; fixed before any result)

F5's instruments unchanged, pointed at this run; each number is read beside F5's at the same checkpoint.

1. **Does Mamba take up the reset and carry it?** (a) Registration of the energy / food / drink drops at Mamba's input
   and state (`d1.py`; F5 ckpt 1: energy 0.106 / 0.031, food 0.034 / 0.032, drink 0.083 / 0.075; ckpt 2: energy
   0.274 / 0.116). (b) Event AP from the predicted belief on quiet awake frames (`belief_probe`; F5 ckpt 1: energy
   0.035, food 0.039, drink 0.047; ckpt 2: 0.037, 0.040, 0.046). (c) The earlier reset's trace at the due step
   (`trace.py`; F5 ckpt 1: energy 0.0073, hunger 0.0065).
2. **Does the forecast separate due from not due?** The arm's own trained head on the pairs (`gate3.py`, arm
   `native`): fraction of the true digit-cell change predicted when due and when not due, their difference (kappa),
   bootstrap intervals over pairs. **Primary: energy kappa, interval excluding 0, with the not-due change staying near
   copying.** Beside it: the round-3 fresh skip head on the frozen belief (F5: kappa 0.000 at checkpoints 1 and 2), the
   request cosines, the +1 / +3 / +6 brightness foils, held-out error against copying.
3. **Does the native generator use it?** Generated due / not-due rates on the pairs (`trajectory.py`). Reported; early
   uptake is not expected (F5's came two checkpoints after the belief's).

Guards, same instruments: recovery and cooldown (reward-forecast separation, belief AP, zombie readiness AP), the
monitor's G1 loss and the logged reactive terms against F5's at the same update. Not read at these checkpoints: S1
entity retention and the actor interface (generation at checkpoints 1-2 is not yet usable in F5 either).

The energy pairs all fork at episode step 14 with the drop 8 -> 7 (F5 RESULTS 11.2): a narrow sample. The belief probe
(1b) covers energy drops in general and is read with it.

## Extension rule (user)

Checkpoint 1 is read first. The run is extended to checkpoint 2 (resume, same schedule) unless checkpoint 1 already
shows the primary reading (energy kappa under the arm's own head, interval excluding 0). A null at checkpoint 1 cannot
be read (see above), so it triggers the extension. The run is not extended, and is reported as broken, if a guard shows
a damaged run: non-finite terms, or G1 / reactive terms clearly off F5's at the same update. Nothing runs past
checkpoint 2 without the user.

## Launch

    systemd-run --user --unit=p1f5t -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261113_p1f6.train \
      artifacts/experiments/20261113_p1f6/run --updates 2500

About 1.35 h per checkpoint (F5's log), then `readout CHECKPOINT`. The saccade sources must not change between the
two segments (resume checks their hashes).

## P1-F6: the full run (user, 11-13 night)

The user named this arm P1-F6 and ordered its extension to the usual ten checkpoints, resumed from checkpoint 2, with
the readout queue, without waiting for a review ("F6 shouldn't wait for Codex"). The folder is renamed
`20261113_p1f6`. Nothing in the recipe, the schedule or the code changes: `run.py` (the one-shot worker of
`../20261104_p1f21/run.py`, as F5) resumes `run/latest.pt` (exact resume, preflight), completes the 25.6M frames,
applies the fixed checkpoint rule and runs `evaluation.py` on the selected checkpoint.

Why the full run and not a stop at checkpoint 2: F5's own clocks formed late (recovery from the belief 0.05 / 0.20 /
0.28 / 0.43 / 0.57 / 0.59 at checkpoints 1 / 2 / 3 / 5 / 8 / 10; drawn in generation from checkpoint 5), the arm
showed no energy request yet at checkpoint 1, and chunks grow from 64 to 256 steps at checkpoint 5 (gradients stop at
chunk boundaries). The run also reads what replacing the TSP forecast does to F5's bundle costs (reactive TSP 0.015
against 0.021 at checkpoint 1).

Queue (`evaluation.py`): F5's nine stages with the same tools and order (S1, native pairs, forecast scores read from
the token forecast, consequence curves and crafting, actor interface, four-step recursion, natural thirst, belief
probe, belief-path probe; the F4 / HUD-Change control pairs are F5's), then `stages.py`: the clock readings above at
checkpoints 2 and the selected one, F5's trajectory table at all ten checkpoints, movement, mining, sleep, the entity
breakdown, the comparison table with an F6 column.

Read first, fixed here: (1) the primary reading above at the selected checkpoint, energy then hunger; (2) the asleep
energy rise, the shortest pixel-only clock (11 asleep steps; F5: belief AP 0.18-0.22 at every checkpoint against a
0.086 base rate, generated rises untimed at 0.048 per asleep step): belief AP by checkpoint and the sleep rollout's
wait ratio (F5 3.24 / 3.06); (3) guards: recovery and cooldown (F5 0.508 / 0.000 and 0.862 / 0.025 drawn due / not
due), danger, the standard table against F5. Coal and iron are expected to stay collapsed (cause open; not this arm's
target).

Launch:

    systemd-run --user --unit=saccade-p1f6 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E JAX_PLATFORMS=cpu \
      -E OMP_NUM_THREADS=2 -E MKL_NUM_THREADS=2 -E PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --working-directory="$PWD" "$PWD/.venv/bin/python" -u -m artifacts.experiments.20261113_p1f6.run
