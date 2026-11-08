# P1-F4: Dreamer 4's denoiser training and dense long-context attention (plan before any run)

User 11-06 night: adopt Dreamer 4-style denoising and the dense temporal attention together in tonight's run ("less clean
scientifically, but we're on a deadline"; "don't implement some weird ad hoc hybrid": both as Dreamer 4 specifies them,
from its paper and the public code). F3's world otherwise unchanged: F2's recipe, trainer hooks, chooser, corpus, world
seed 1, 25.6M-frame budget. From scratch. If adopted, the code becomes canonical and the old branches (EDM, per-token
temporal attention) are removed (user 11-07), checkpoints before that then load from the commit before the cleanup.

## What changes, exactly (sources)

Paper: Dreamer 4, arXiv 2509.24527 (`third_party/papers/2509.24527v1.pdf`), eq. 6-8 and sections 3.2 / 3.4, Table 2.
Code: both public reimplementations, identical in training: nicklashansen/dreamer4 b8abafbf (`train_dynamics.py`:
`dynamics_pretrain_loss`, `sample_one_timestep_packed`; `model.py`: `Dynamics`) and edwhu/dreamer4-jax 8144b940
(`scripts/train_dynamics.py`, `dreamer/sampler.py`). Implementation: `saccade/model.py` (`shortcut_*`, `Block.dense_time`),
`saccade/train.py` (`p1_losses`, `bootstrap_rows`, the monitor), behind config keys; absent keys = the previous code
path, which reproduces F3's stored dev reading bitwise.

1. **Shortcut forcing replaces EDM** (and F3's x-space weighting: the flow term is an x-space loss).
   - Interpolation z_tau = (1 - tau) z0 + tau x1, z0 ~ N(0, I), tau = 0 pure noise; the network predicts x1 (x-prediction,
     raw output, zero-initialised heads as theirs); no EDM preconditioning.
   - Per frame (diffusion forcing) and per group (decision 1: physical p, v / glimpse independent): a step index and a
     signal level on that step's grid. Flow rows: finest step d = 1 / k_max, tau = j / k_max; bootstrap rows: d uniform
     over the coarser powers of two, tau = j d.
   - Flow term ||x1_hat - x1||^2; bootstrap term (1 - tau)^2 ||(x1_hat - z) / (1 - tau) - sg(b' + b'') / 2||^2 with the
     two half steps at d / 2 (tau, then tau + d / 2); ramp weight w(tau) = 0.9 tau + 0.1 on both; per frame the mean over
     its percept scalars (each element weighted at its group's level); before update 5,000 the bootstrap rows contribute
     0 (both reimplementations). In the half steps every frame of a row advances its own half step (theirs).
   - k_max 8 (theirs; the paper leaves it open; K = 4 needs >= 4); bootstrap rows 25% of each update's streams.
     Theirs take the batch's last rows of random sequences; our streams persist, so a fresh random subset per update
     (`bootstrap_rows`: its own seed, resumable, the training RNG untouched).
   - Conditioning: discrete embeddings of the signal index (k_max + 1) and step index (log2 k_max + 1), as theirs, summed
     with the action embedding into the existing adaLN projection (sum = concatenation followed by its linear layer).
     The two tables draw from their own seed, so every other parameter starts exactly as F3's.
   - Sampler: K = 4 evaluations from pure noise, z <- z + (x1_hat - z) / (1 - tau) d (theirs; the last lands on x1_hat);
     x1_hat clipped to the token range as every inference path here already does. Factorised cascade kept: periphery
     and lighting first, then the glimpse given them, slightly corrupted to the context level.
   - Past inputs: "slightly corrupt the past inputs ... to signal level tau_ctx" (paper). Both reimplementations label
     the context k_max - 1; nicklashansen passes it clean, edwhu optionally corrupts it. We corrupt it to that level
     (tau = 7 / 8) and label it so: exactly the training distribution of that label (STEP_SPEC 7's one fresh draw).
   - Monitor (checkpoint selection, the rule unchanged): the ramp-weighted flow term at all 8 finest-grid levels on the
     fixed held-out windows, context at its level, fixed draws.
2. **Dense temporal attention** in the long-context layers (blocks 4 and 8): every token attends every token of its own
   frame and of the 4 previous frames of its episode (Dreamer 4 Table 2's dense layers, before its "time factorized
   long context" speed step: FVD 71 -> 91), 1D RoPE over frame positions; physical queries read physical keys only
   (decision 1). Same parameters as the per-token layers.
3. **Memory, not the recipe:** dense attention reads each frame's keys / values once per window slot (+0.5 GiB); each
   update is split into 2 microbatches at 64 steps and 8 at 256 (F3: 1 and 4); gradient accumulation, the same update up
   to per-microbatch averaging of the count-normalised terms (as F3 already had at 256 steps).

## Verified before the preflight

- `saccade/tests`: 76 pass (63 existing unchanged + 13 new): dense streaming = chunked (factorised and not, across an
  episode reset and beyond the window); physical outputs independent of glimpse inputs, glimpse code and fixation; the
  glimpse-only pass = the full step; the whole cached shortcut cascade (with context passes) = a full-history chunked
  recomputation (window overflow, reset, branch duplication, cache round trip); the loss = nicklashansen's
  `dynamics_pretrain_loss` transcribed line by line (bootstrap on and off); the factorised loss = an explicit per-part
  computation; levels on the reimplementations' grid; the sampler = their recursion; bootstrap rows deterministic.
- F3 unchanged: exp 2's R0 (dev COLLECT_COAL through the interface) reproduces F3's stored reading exactly.
- Recipe = F3's checkpoint configuration minus its G1 weighting plus exactly these keys; init identical to F3's except
  the two embedding tables (3,328 parameters).
- Determinism: within a process bitwise. Across processes F3's own configuration already differs (optimizer state 2.3e-13
  after 4 updates, model bitwise); with dense layers it reaches their near-zero adaLN parameters (5e-18). The resume
  check holds model and optimizer to F2.1's 1e-11 and everything else (streams, carry, cache, state, RNG, chooser, gaze,
  every logged term) bitwise. A first version gathered window keys by index, whose backward accumulates with atomics;
  replaced by shifted slices.

## Preflight (`checks.py` -> `checks.json`)

Exact resume across the bootstrap start; update 1 of F3's configuration beside F4's (every logged term except g1
identical); the real launcher; 200 x 64-step updates (bootstrap off, the recipe until 5,000), 100 x 64 and 3 x 256 with it
on, 3 x 256 off; peak memory; the training-time estimate; the monitor. Probe before it: 64-step 2.30 s / update with the
bootstrap (F3 1.68), 256-step 8.1 s (6.9 without; F3 6.14), peak 2.5 GiB: ~15 h of training vs F3's 11.5 (+30%). The
bootstrap's share matches shortcut models' own ~16% (Frans et al. 2410.12557); the rest is dense attention and the
smaller microbatches (FlexAttention's block-sparse windows could remove the key duplication: tomorrow's optimisation).

## Preflight result (11-07 00:15-00:28, `checks.json`, 795 s, peak 2.54 GiB)

All passed. Resume across the bootstrap start: streams, carry, cache, state, RNG, chooser, gaze bytes and every logged
term bitwise; model within 4.7e-18 (225 / 228 tensors bitwise), optimizer 1.1e-19. Update 1 beside F3's configuration:
every logged term identical except g1 (1.121 -> 0.328; grad norm 6.73 -> 6.45). Real launcher in a fresh process.
Speed: 64-step 2.03 s / update (bootstrap off) and 2.19 (on), 256-step 7.30 / 8.06 (F3 1.68 / 6.14): training ~14.4 h.
Readout smoke (every instrument end to end on the preflight's 4-update checkpoint: S1 and its reader, the consequence
curves and crafting at smoke size, the interface on 40 episodes, recursion on 4 windows): all completed. Codex's review
(`../20261107_p1f4_review/REVIEW.md`): supports the run; its points are taken in the declared readings (combined recipe;
correct waking must survive; movement unmeasured; glimpse covered only by S1.3); the full seal was waived by the user.

## Run

`run.py` = F2.1's one-shot worker unchanged (training, F2's checkpoint rule on the monitor, gates `checks.json`, then
`readout.py`); `pin_sources.py` seals the sources after the preflight. Launch (after review):
`systemd-run --user --unit=saccade-p1f4 -p MemoryMax=24G -E MAMBA_DETERMINISTIC=1 -E
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --working-directory=REPO .venv/bin/python -u -m
artifacts.experiments.20261106_p1f4.run`. One GPU job at a time.

## Readout (`readout.py`, ~1 h)

S1 (and its S1 reader); `consequences.py`: exp 2's determinant curves and crafting at the frame level (same
populations, seeds, readers' protocol; F4's own sampler with no start recipe); the actor interface; own-output recursion;
all at K = 4. Dropped from F3's chain: the consequence trace, sampler_steps and glimpse_check (EDM internals), F3's F2
comparisons. F3's references: curves exp 2 (16 EDM steps) and T2 (prefix-20), crafting (prefix-20), F3's readout.

## Declared readings (before any result; amended 11-07 with Codex's review, `../20261107_p1f4_review/REVIEW.md`)

This tests the combined recipe: no reading is attributed to shortcut forcing or to dense attention alone (Codex).
- **Sleep held without any start recipe**: false wakes from asleep at energy <= 8 <= 0.15 (F3: 0.42-0.52 at its start,
  0.06-0.11 with prefix-20); above 0.30 it is not fixed. Only together with correct waking (Codex): woken at energy 9
  and by a ready zombie's hit on a sleeping agent must not fall below F3's prefix-20 levels (0.15 / 0.23); fewer false
  wakes bought by never waking is a failure, not a fix.
- **Energy 9 -> wake** (25% of flow draws now at sigma_eq >= 7 vs ~2%; still a cross-token, smallest-cue, global
  consequence): contrast wake(9) - wake(<= 8) >= 33 pp learned, < 10 pp not (F3: 0.04 with prefix-20).
- **Appearance of a count when its cause is another token's past** (one attention step now): coal 0 -> 1 >= 0.50 and
  iron 0 -> 1 >= 0.35 at unchanged exposure support the route reading (F3 T2: 0.25 / 0.13); both within 10 pp of F3:
  it is not supported. Increments n -> n + 1 are predicted to keep following per-transition exposure (no pooling
  mechanism was added).
- **Controls**: cleared cells and crafted-tool appearance stay at F3's level (0.81-0.98 / 0.79-0.91); a drop of > 10 pp
  is a regression to explain before anything else. Movement is UNMEASURED tonight (T1's movement bank reads moves with
  F3-token readers); to be read from the checkpoint with F4-fitted readers. The generated glimpse is covered only
  inside S1.3's real-vs-generated test (periphery, glimpse, lighting together); no dedicated glimpse check tonight.
- **Hidden counters** (cooldown, kills): no route was added; no change predicted.
- **Few-step generation** at K = 4: interface and recursion against F3's 3- and 16-step readings, descriptive.
