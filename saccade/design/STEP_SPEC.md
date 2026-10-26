# Step specification v1.3 (2026-10-27) — one real step, one imagined step

Supersedes v0 (world sampling + chance codes) and LEARNING_SKETCH.md. Consolidates PERCEPTION v4, RECORDS v3, GENERATOR
v3, LOSSES v1. Working numbers are placeholders. Nothing here is implemented.

## 1. Persistent state: the belief B_t (the only state; the actor reads it)
- self-location: position + heading in the world frame, with uncertainty;
- store S_t: location-keyed traces (background content, time stamp, reliability); per-branch overlay in imagination;
- records R_t (K = 8): presence probability, appearance descriptor, dynamic state, position hypotheses (H each),
  hidden state = the record's slot of the working state;
- working state W_t (Mamba): one global slot (self-state, time of day, anything not tied to a record or a place) + one
  slot per record;
- pending split associations (rare), kept for N steps.
There is no separate hidden world: simulation runs on the belief (belief-MDP; GENERATOR v3).

## 2. Real step (world training on the corpus)
1. Predict: B-_{t+1} = U_pred(B_t, a_t), deterministic: self-location moved by the action; each record's hypothesis
   distribution propagated; Mamba step on the action; store unchanged (edits arrive as observations).
2. Perceive o_{t+1}: lighting latent v, patch-local periphery tokens p, fixation f_t chosen at t; TSP predicts the
   glimpse pixels at f_t from (B-_{t+1}, p, v) before reading it; glimpse tokens g, always stored together with their
   fixation coordinates f_t (in the recent window and in the Mamba input).
3. Generator loss: the denoiser D, conditioned on B-_{t+1}, a_t and f_t, denoises a noised copy of the percept
   P_{t+1} = (v, p, g) back to its stop-gradient value (EDM noise levels, x-prediction).
4. Correct: B_{t+1} = U_corr(B-_{t+1}, P_{t+1}): self-location correction (alignment with the store's expectation);
   association (hard; split posteriors deferred); owned records corrected; non-detection reweighting; births; learned
   inactivation; store write where the background owns the location; Mamba step on the percept.
5. Outcome signals are OBSERVATIONS too (Codex): U_corr reads the real reward and termination along with the percept.
   In Craftax this is needed: the HUD shows health / food / drink / energy / inventory but NOT which achievements are
   unlocked, so reward is evidence of an unlock (combined with the health change), and later rewards depend on it.
   Outcome models, chained for consistency: reward ~ p(r | B-_{t+1}, P_{t+1}) as a discretised mixture of logistics
   in symlog space (a proper, samplable density that can be sharp for deterministic rewards; a two-hot regression
   head is NOT a sampler: a deterministic reward of 1 gets mass on bins ~0.88 and ~1.20, Codex), with a categorical
   over the empirical reward values as the alternative for discrete-reward environments; continuation ~ p(c |
   B-_{t+1}, P_{t+1}, r). The critic keeps two-hot. Change map rendered from (B-_{t+1}, B_{t+1}).
Losses and weights: LOSSES.md v1. Gradient routes into the encoders: the belief path, TSP, the fovea's own
reconstruction, the change map; every denoiser input and target is stop-gradient (LOSSES principle 7; SIGReg dropped
10-27). No gradient from the fixation head, actor or critic into the world.

## 3. Imagined step (actor training; world frozen)
Start from a real history: B_0 from the update. Repeat:
1. Actor reads B_t (stop-gradient) -> a_t and fixation f_t.
2. B-_{t+1} = U_pred(B_t, a_t).
3. Sample the next percept P^_{t+1} ~ D(. | B-_{t+1}, a_t, f_t) with ~3 Euler steps (periphery, lighting and glimpse
   jointly).
4. Sample r^ from the reward model given (B-_{t+1}, P^_{t+1}); use the SAME r^ for the belief update and the return.
   Termination: survival-conditioned branches — U_corr reads "not terminated" and the return weights later steps by
   the predicted continuation probability (Dreamer), which is the exact expectation when the belief is conditioned on
   survival (Codex: either this or sampling termination, but declared). B_{t+1} = U_corr(B-_{t+1}, P^_{t+1}, r^,
   not terminated). First-seen content is written to the branch store with the reliability the update assigns.
   Conservation and revisit consistency are LEARNED properties to demonstrate.
Cost per imagined step: ~3 denoiser passes + one predict + one correct. The denoiser's passes never commit Mamba
updates; exactly one predict and one correct per environment step. Distributional scoring (P2, S2) uses SAMPLED
termination with the terminal convention of CURRICULUM.md, not these survival-weighted branches.

## 4. Curriculum
Context length, self-feeding and rollouts: CURRICULUM.md v0 (TBTT / state passing for long beliefs; context corruption
and one-step self-conditioning from the start; multi-step distributional post-training on exogenous windows with a
horizon schedule; imagination H = 16 from variable-depth real beliefs).

## 5. Actor and critic inputs
Belief only: self-location and the global slot (self-state, time), an attention summary over records (presence,
appearance, dynamic state, hypothesis spread), a store read around the self, v. Fixation is part of the action.
Critic inputs: belief only by default (there is no hidden world any more).

## 6. Denoiser practicalities (RAE, Zheng et al. ICLR 2026)
Starting choices, checked by smoke tests (Codex: guidance, not a law): denoiser width at least the token dimension;
noise-schedule shift set by the EFFECTIVE dimension (tokens x channels, not channels alone); RAE studied frozen
pretrained representations with velocity prediction, so neither result transfers as a theorem to our jointly learned
tokens with EDM x-prediction. JEDI adds tanh clamping, a slower encoder and own-output conditioning.

## 7. Minimal first build M0 (user, 10-27: minimal first, defined components added on measured need)
- Belief = the Mamba state + the denoiser's recent frames. Mamba reads an INTERLEAVED sequence: the percept of t with
  its reward and termination (flattened and linearly projected, as in DRAMA: position-sensitive and lossy; probes show
  what survives), then the action of t. Its output after the action is the PREDICTED belief, held fixed through the
  denoising steps; the next percept is the correction. No records, no store, no self-location.
- Denoiser (agreed 10-27, user; replaces the grid-conv proposal): a Dreamer-4-style space-time transformer. Tokens per
  frame: 81 periphery tokens, 9 glimpse tokens at their own fixation's cells, 1 lighting token, 4 memory tokens (the
  predicted belief projected); spatial attention every layer, temporal attention every 4th layer over the past 4
  frames (effective reach 8 frames through composition); 2D RoPE plus absolute 2D positions (Heo et al.: absolute
  embeddings helped at a 7 x 7 grid), type embeddings, adaLN-Zero on noise level and action (DiT). Each component keeps
  its own shape and noise draw (Codex). Trained in the diffusion-forcing form (each frame its own noise level) over
  chunks preceded by 8 context-only frames; streaming inference with per-layer caches of the last 4 frames must equal
  the chunked computation (test). In P1 Mamba reads clean recorded percepts; practice on generated percepts is P2's.
  The memory map of the grid-conv proposal is dropped: memory tokens serve global and position-specific reads alike,
  and store reads (I1) and records (I2) will enter as tokens. All numbers: saccade/m0.yaml.
- Periphery acuity (agreed 10-27): each cell's periphery input is downsampled to 3 x 3 pixels (8-bit); the fovea stays
  full resolution (PERCEPTION, acuity section).
- Unchanged from the full design: perception (patch-local tokens, lighting latent, fovea + TSP), denoiser and its loss,
  reward / termination as observations, change map, fovea reconstruction, actor and critic reading the belief, the belief-MDP
  imagination loop. Fixation: a broad random prior (HUD included) while the world model is trained; learned fixation
  once the world model exists (its offline target needs one).
- Increments, each entering through the SAME two interfaces (the belief's contents and the denoiser's conditioning
  set): I1 store + self-location (world-keyed memory; the denoiser also attends to store reads mapped to screen
  positions); I2 records (entity slots of the working state, hypotheses, association, births; the denoiser attends to
  them). The loop does not change.
- Known weak spots of M0 by design: hidden per-entity state (cooldown, health) carried only implicitly; revisits
  beyond Mamba's effective memory; identity through long absences.

## 8. Build order (bounded learned-loop test)
M0 one-step -> M0 rollouts -> M0 actor; then I1 / I2 only on a measured trigger (TEST_SPEC.md). Disabling a structure
in a trained model measures RELIANCE on it; claiming its BENEFIT needs a matched training comparison (Codex).

## Open
Resolved 10-27: budgets and seeds (CURRICULUM, Budgets), the S3 blockers (TEST_SPEC S3.2), the P2 training summary
and gradient contract (LOSSES G4, CURRICULUM §2). Reward bins: LOSSES G3 (256 symlog bins over [-0.9, 1.1]).
N, K, H, denoiser steps, noise weighting, schedule lengths; how U_pred propagates hypothesis distributions (a learned
deterministic map of the set); probes and predeclared readings per stage.
