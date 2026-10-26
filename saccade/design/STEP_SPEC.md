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
Losses and weights: LOSSES.md v1. Gradient routes into the encoders: the belief path, TSP, SIGReg, the change map; the
denoiser's targets are stop-gradient. No gradient from the fixation head, actor or critic into the world.

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
- Belief = a short window of recent percepts (k = 4, working; JEDI uses 4) + the Mamba state (a global working state;
  input per step = an attention-pooled summary of the percept, the action, reward and termination). No records, no
  store, no self-location.
- Predict = Mamba step on the action; correct = append the percept and step Mamba on it. The denoiser is conditioned on
  (Mamba output, recent percepts, action, fixation) and generates the next percept; scrolling is learned from the
  recent percepts, as in JEDI / Dreamer 4.
- Unchanged from the full design: perception (patch-local tokens, lighting latent, fovea + TSP), denoiser and its loss,
  reward / termination as observations, change map, SIGReg, actor and critic reading the belief, the belief-MDP
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
