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
- Plan (contract): B parallel episode STREAMS, each played in order in consecutive chunks (64 steps, growing to
  256); the Mamba state is carried across chunks of the SAME episode and reset at true episode boundaries; one
  optimizer update per chunk, the carried state detached: its last chunk was computed under the previous update's
  parameters and its earlier history under older ones (TBPTT never refreshes the history; a normal approximation, to be
  measured, not a reason for a state-rebuild subsystem); no out-of-order chunks, so no burn-in is needed (a short zero-initialised burn-in
  could not rebuild facts from an omitted long history anyway). Foreign-state or mid-episode zero starts (Buitrago &
  Gu) are a separately declared robustness intervention, OFF by default: an episodic belief carries achievements,
  health and map facts that another episode's state would contradict. Carrying state extends the history represented
  in training, not gradient credit beyond the chunk; long-gap memory is something the probes must show. Streams keep
  their identity, reset masks, padding masks and the 4-percept window aligned; denoiser evaluations within one
  environment step never commit Mamba updates.

## 2. Self-feeding (robustness to the world's own outputs)
Compounding comes from the belief and denoiser reading their own imperfect outputs (exposure bias). Under stochastic
outcomes a generated history can be correct and still differ from the recorded one, so a pointwise loss against the
recorded future is only valid one step after a real history (DaD, Venkatraman et al. 2015, assumes it otherwise).
- From the start, two robustness procedures, neither a solution to stochastic self-feeding:
  (a) context corruption: Gaussian noise on the conditioning percepts at a level sampled per sequence and given to the
      denoiser as input (GameNGen; Dreamer 4 uses tau_ctx = 0.1);
  (b) JEDI-style switching, exactly: with probability 0.5 a recorded percept in the conditioning window is replaced by
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
Open: chunk growth schedule, corruption range, self-conditioning probability, update counts per phase.
