# Curriculum v1 (2026-10-27): context length, self-feeding, rollouts (contracts per Codex)

Three different clocks: (1) how far back the belief must remember (memory), (2) how robust the world is to reading its
own generated percepts (self-feeding), (3) how far imagination runs for the actor (horizon). Working numbers.

## Facts
- Episode lengths (train): eps 0.1 median 478 (p90 1,530, max 2,500); eps 0.25 median 240; eps 0.5 median 184;
  eps 1.0 median 135 (p90 235, max 444). Long histories come from expert-heavy episodes; fully random episodes
  rarely exceed ~235 steps (p90), at most 444.
- Memory needs (M2, of ALL entering-cell events): never seen 41%, last seen <= 16 steps before 27%, 16-64 14%, 64-256
  12%, > 256 5%; i.e. ~17-18% of all entering events, ~30% of previously seen ones, were last seen more than 64 steps
  earlier. Entities reappearing were seen within 64 steps in nearly all cases (> 64: 1-2%).
- Expert decisions barely need long memory (M1), but revisit consistency in imagination does.

## 1. Context length (the Mamba state)
- Recurrent models fail beyond their training length when training never visits the states long sequences reach
  ("unexplored states"); starting training sequences from attainable states fixes it: STATE PASSING (the final state
  of another sequence, 10% zero) or TRUNCATED BPTT (chunks of one sequence, state carried), both equally good, ~0.1% of
  training as post-training (Buitrago & Gu, ICML 2025; Mamba-1 / 2, GLA, RWKV). Longer training contexts help; long
  warm-ups and long training on short contexts hurt.
- Plan (contract): 16 parallel episode STREAMS per optimizer update throughout P1 (E1 / E1d: at equal frames per
  update, more distinct episodes per update won; DRAMA uses 16 x 128), each played in order in consecutive chunks:
  64 steps for the first half of P1's frames, 256 steps for the second half. At 256 only ~4 streams fit in 6 GB, so
  a 256-step update ACCUMULATES four microbatches of four DISTINCT streams (each with its full 256-step gradient path)
  before one optimizer step; loss normalised by valid (non-padding) frames, gradients clipped after accumulation;
  token normalisation (BatchNorm, no affine) uses each microbatch's statistics, 83k tokens in both halves (16 x 64 x
  81 and 4 x 256 x 81), so the estimator is the same before and after the switch. The Mamba state is carried across chunks of the SAME episode and reset at true episode boundaries; one
  optimizer update per chunk, the carried state detached: its last chunk was computed under the previous update's
  parameters and its earlier history under older ones (TBPTT never refreshes the history; a normal approximation, to be
  measured, not a reason for a state-rebuild subsystem); no out-of-order chunks, so no burn-in is needed (a short zero-initialised burn-in
  could not rebuild facts from an omitted long history anyway). Foreign-state or mid-episode zero starts (Buitrago &
  Gu) are a separately declared robustness intervention, OFF by default: an episodic belief carries achievements,
  health and map facts that another episode's state would contradict. Carrying state extends the history represented
  in training, not gradient credit beyond the chunk; long-gap memory is something the probes must show. Streams keep
  their identity, reset masks, padding masks and the denoiser's context aligned; denoiser evaluations within one
  environment step never commit Mamba updates. Denoiser context (Codex, 10-27): temporal attention sees the past 4
  frames at each temporal layer and composes across its 2 temporal layers, so its effective reach is 8 frames; each
  training chunk is preceded by the previous 8 frames of its stream as context-only frames (no loss), episode
  boundaries mask attention, and streaming inference with per-layer caches of the last 4 frames must reproduce the
  chunked computation exactly (an implementation test).

## 2. Self-feeding (robustness to the world's own outputs)
Compounding comes from the belief and denoiser reading their own imperfect outputs (exposure bias). Under stochastic
outcomes a generated history can be correct and still differ from the recorded one, so a pointwise loss against the
recorded future is only valid one step after a real history (DaD, Venkatraman et al. 2015, assumes it otherwise).
- From the start, one robustness procedure, not a solution to stochastic self-feeding:
  (a) context corruption, realised by the denoiser's DIFFUSION-FORCING training form (Dreamer 4; agreed 10-27 with the
      transformer backbone): every frame of a chunk carries its own sampled noise level, given to the denoiser, so the
      past frames it conditions on are noised copies of the recording; at inference past frames are re-entered at a
      small context noise level (m0.yaml edm.inference_context_sigma). Scope (Codex): Mamba reads the clean recorded
      percepts in P1, so the belief update never practises consuming GENERATED percepts there; that is P2's job
      (rollouts read generated percepts through the same update, gradients through the last 2 steps).
  (b) DROPPED for M0 (user, 10-27; a defensible simplification, not an equivalent of the recipe: JEDI's ablation finds
      switching helpful, not critical, and it does not fit a one-pass sequence form). KEPT ON THE DRAWING BOARD as the
      FIRST ablation if the curriculum proves to need more self-feeding. As specified before: JEDI-style switching,
      exactly: with probability 0.5 a recorded percept in the conditioning window is replaced by
      the denoiser's RECONSTRUCTION of a noised copy of that same recorded percept (noise level sampled; stop-gradient
      on the substitute; the real reward and termination accompany it). JEDI's Algorithm 1 is the precedent (random
      switching to its own denoised version of the same recorded latent). A robustness heuristic, tested as such: a
      lossy reconstruction need not preserve every future conditional, so the recorded continuation is a sound target
      only to the extent the reconstruction keeps what that continuation depends on (review, 10-27). An INDEPENDENTLY
      sampled future is never inserted and then supervised against the recording (scheduled sampling's
      inconsistency, Huszar 2015).
- After one-step training (fixed update count, not on a result): multi-step post-training on generated histories with
  a JOINT-TRAJECTORY distributional loss: for one real prefix and the recorded exogenous actions, m >= 2 independent
  free-running trajectories X_i and the recorded Y are each flattened to one vector (per-step percept summaries, reward,
  termination; each component scaled by its data standard deviation, the scaling declared) and scored by the energy
  score (beta = 1); framewise and patched scores are supplementary only (a sum of per-step scores can match every time
  marginal while breaking persistence). Gradient boundary (ours, declared): backpropagate through the sampler of the last
  2 world steps only, earlier steps detached. Self Forcing's DMD / SiD / adversarial objectives are the precedent; the
  energy-score objective is our adaptation, not a validated copy. Horizon schedule 2 -> 4 -> 8 -> 16 by update count.
  The training summary, its scaling and weights are LOSSES G4 (patched space-time + full-trajectory global score,
  m = 2 independent rollouts). Gradient contract (agreed 10-27): a truncated PATHWISE gradient through the continuous
  sampling and the belief update of the last 2 generated steps; reward and termination draws are DETACHED (their heads
  train only by the one-step likelihood losses, G3); fixations come from the random prior, independent of the
  recorded future, and are shared by the samples and the recording. The implementation check confirms nonzero
  gradients into the denoiser and the update, none into the encoder or the outcome heads.
  Termination in these samples is SAMPLED with an explicit terminal convention (after termination: absorbing state,
  percepts masked, reward 0, termination = 1), the same convention applied to recorded trajectories; survival-weighted
  branches are never scored by this ordinary energy score.
  Windows: exogenous only (A8): 16-step windows almost all from random play; 4- and 8-step windows also inside
  expert-mixed episodes. A coverage limit (Codex): random-play windows cannot alone establish behaviour in the states a
  competent actor visits.
- The encoder is frozen during multi-step post-training (stable targets; the S1 reader stays calibrated).
- History guidance (Song et al., ICML 2025: diffusion forcing transformer with flexible history) is the reference if
  long free-running rollouts drift despite the above.

## 3. Imagination (actor training; world frozen)
- Start states: real beliefs after real prefixes of variable length (from the TBTT stream), so imagination starts
  from shallow and deep beliefs alike.
- Returns: survival-weighted continuing branches (STEP_SPEC); the immediate reward of a step, including a terminal
  penalty, is counted before that step's continuation weight multiplies later returns.
- Horizon H = 16 (Dreamer 15-16, JEDI 15), equal to the end of the world's rollout schedule; longer horizons only after
  S2 shows coherence at 16.
- Fixation: a broad random prior during world training; part of the actor's action during imagination.

## Phases
P1 one-step world training (TBTT chunks, state passing, context corruption, self-conditioning) -> S1;
P2 multi-step post-training (horizon schedule, energy score, exogenous windows) -> S2;
P3 actor in imagination (H = 16, variable-depth starts) -> S3.

## Budgets (agreed 10-27; caps fixed before training, never extended after a disappointing reading)
Basis: RTX 3060 Laptop (6 GB), one GPU job at a time; train split 8,069 episodes / 2.56 M transitions; LeWM Raw
(6.2 M encoder + 2.5 M world, 512 frames per update) ran 0.56 s per update and its loss was still falling at 2 epochs;
a stand-in of M0's shapes measured ~1 ms per frame with activation checkpointing (16 x 64 frames: 0.97 s, 3.5 GiB).
- Seeds: 2 world seeds (P1 + P2), each read separately; P3: 3 actor seeds and 3 BC seeds per world seed (nested:
  two independently learned worlds, reported per world).
- P1: 25.6 M valid frames per world seed (~10 epochs; slightly more of the reduced split): 12,500 optimizer steps of
  16 x 64, then 3,125 steps of 4 x 4 x 256 (accumulated). Learning-rate warmup and cosine decay follow FRAMES seen,
  not optimizer steps (80% of the steps fall in the first half). A fixed 5% of train episodes is held out for all
  arms as the monitor; checkpoints every 10% of the frames; monitor metric fixed now: the one-step denoising loss on
  2,000 fixed held-out windows, fixed noise draws at 8 fixed noise levels, fixed random fixations. The S1 reading uses
  the final checkpoint, unless the monitor rose at 3 consecutive checkpoints: then the lowest one. ~9 h per seed.
- P2: 6,000 updates per world seed (the project's continuation dose), 1,500 at each horizon 2, 4, 8, 16; per update
  16 exogenous windows x m = 2 rollouts, plus the one-step G1 loss on 8 real streams x 64; ~2 h per seed.
- P3: 2,500 actor updates (the 09-03 count, the only measured success), batch 64 imagined starts, H = 16; BC on the
  same belief features; evaluation on the 512 paired DEV environment seeds per agent; ~8 h in all.
- S1 / S2 readings ~1 h each. Total ~32 GPU-hours: an ESTIMATE until the full-model throughput check (every loss,
  optimizer state, reset path and real gradient routes, before P1) converts frames into wall time; the frame and
  update budgets stay fixed if it is slower.
Open: context-corruption range.
