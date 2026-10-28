# Where does the generated distribution go wrong? Diagnosis plan v2 (2026-10-29, before any run)

v1 (commit 558ba9ba) was reviewed (REVIEW.md, 67d13ca0). v2 keeps the questions, removes the tests whose outcomes
could not identify what v1 claimed, adds two tests with an exact null or an exact reference, and narrows every
outcome-to-cause arrow. Point-by-point responses are at the end.

## What is established (narrowed)
- In the DEPLOYED pipeline (production sampler, native heads), one-step generated successors from real histories
  carry almost no deaths and about a third of the negative rewards, while the native heads on recorded successors match
  the recorded death rate (Codex 10-28; reproduced 10-29 with different code and draws: 0.367% recorded vs 0 of 24,000
  sampled futures on random contexts; 12.77% vs 8 of 60,880 in low-health / adjacent-zombie contexts). Native heads
  are part of the model: this is a predictive failure of the model as deployed, whatever its location inside it.
- Matching the mean death rate on recorded successors is a strong recognition control, not complete conditional
  calibration of the heads.
- Withdrawn from v1: "negative reward without the rule-implied death" as a partial consequence. A generated negative
  reward can be an intrinsic health decrement, and the belief may be uncertain about starting health; it is an
  association to diagnose. The reproduction's predicted-death column used 1 - sigmoid(logit), whose float32
  resolution distorts values near 1e-6 (the sampled counts are unaffected); new scripts use sigmoid(-logit).
- Generated tokens violate the encoder's per-token statistics (Codex's summary-statistics classifier 99.5%); entity
  counts on generated tokens depend on the step count and the readers' support there is poor (Codex). Neither the
  geometry nor the counts are interpreted causally yet.

## Two candidate links and why both are live (sources read 10-29)
(a) THE SAMPLING ALGORITHM. Production: EDM Algorithm 2, Euler, N = 3, rho = 7, sigma_max 5, S_churn 1 (sigma
grid 5, 0.283, 0.002, 0; churn gamma = 1/3). The network is evaluated at sigma_hat 6.67 and 0.377 (the third
evaluation, at 0.0027, approximately returns its input). Each Euler step to the next grid level lands at the clean estimate plus
the remaining noise, so a sample is "the clean estimate at 6.67, re-noised to 0.377, re-estimated": the network is not
evaluated anywhere between 6.67 and 0.377. DIAMOND (2405.12399, sec. 5.2, Fig. 4): "the optimal single-step
prediction is ... the expectation over possible reconstructions"; "an iterative solver is necessary to drive the
sampling procedure towards a particular mode"; DIAMOND chose n = 3 on Atari (crispness was the criterion, not branch
frequencies). Dreamer 4 (2509.24527, Table 2, Fig. 8): plain diffusion forcing degrades from FVD 306 at 64 steps to 875
at 4 steps; shortcut forcing exists to make 4 steps work. M0 is plain EDM diffusion forcing at 3 steps.
(b) THE LEARNED FIELD. Tweedie denoisers favour majority features (Um et al., ICLR 2024, sec. 3.1); conditional
diffusion on long-tailed data loses tail classes (Qin et al., CVPR 2023); EDM's p_train deliberately puts little mass at
high sigma (sec. 5: "at high levels the training targets are always dissimilar from the correct answer"): our
lognormal(-0.4, 1.2) puts 4.70% of draws at sigma >= 5, 1.22% at >= 10, 0.23% clamped at 20; the noise schedule can
cause "mode amplification" (Dennehy et al., 2607.15485, abstract).
Approximate fields couple (a) and (b) (REVIEW): no test below yields an additive split, and no plateau is "the
model's distribution".

## Tests (frozen world; each with claim / competing / control / what each outcome rules out)

A. Field calibration on context classes (replaces v1 T2; sampler-free, no head, no counterfactual frame).
- Identity: for the minimum-MSE denoiser, D*(x_sigma, c) = E[x0 | x_sigma, c], so for any class C that is a
  function of the conditioning c and any fixed linear functional f, E[f(D*(x0 + sigma eps, c)) | C] = E[f(x0) | C] at
  EVERY sigma (law of total expectation; the loss weighting does not move the minimiser). Outcome-stratified errors
  have no such null (an exact posterior mean errs more on rare outcomes by definition), which is why classes are
  defined by the context only.
- Classes, from the state at t-1 (rendered in the context frame t-1) and the action a_{t-1}, both in c:
  RISKY (health <= 2, a zombie at L1 1), ADJACENT (health >= 3, a zombie at L1 1, awake), DRINK positive control
  (awake, facing water, a_{t-1} = DO, drink <= 8: drink + 1 is deterministic, game_logic do_action), SAFE negative
  control (health 9, awake, no zombie within L1 3, no skeleton or arrow in view, no lava adjacent, food / drink /
  energy >= 1), RANDOM.
- Functional: the HUD cell token (health 63, drink 65) projected on the line from the mean real token of the starting
  value to that of the consequence value (health h -> h - 2, or <= 0 for RISKY; drink d -> d + 1), scaled so that
  the two means are 0 and 1. Fixed per context before any denoiser output is seen.
- Readout per class: bias(sigma) = mean f(D(x0 + sigma eps)) - mean f(x0) over sigma in {0.05, 0.2, 0.377, 0.7, 1.2,
  2, 3.5, 5, 6.67, 10, 20}, 4 noise draws per context, episode-bootstrap SE; beside it, mean f of the PRODUCTION samples
  (4 per context) in the same unit. Secondary (not conditioning-measurable, descriptive): RISKY split by the
  simulator's cooldown readiness.
- Controls pass if |bias| < 0.05 at every sigma <= 6.67 in SAFE and DRINK (declared now).
- Outcomes: controls pass and RISKY / ADJACENT bias < -0.05 beyond 3 SE at sigma in [2, 6.67] -> the trained field's
  conditional mean underweights damage there (a property of the network, independent of any sampler). Bias within
  0.05 at all sigma while the production samples' mean f stays far below the recorded -> the field's conditional mean
  carries the consequence and the deficit enters during sampling. Controls fail -> the readout is not usable.
- Does NOT establish: joint correctness of the frame, per-history sharpness, calibrated event odds, or which change
  would fix anything.

B. The production algorithm on an exact field (new; CPU; EDM's own analytical-denoiser method, EDM sec. 4).
- Claim: the production algorithm, given the EXACT posterior of a two-outcome mixture, returns the rare outcome far
  less often than its probability. Competing: it returns it at its probability (then the algorithm cannot by itself
  explain the deficit). Control: the same computation for the many-step stochastic and deterministic algorithms of D,
  which must return the mixture weight (a check of the code).
- Construction: the exact posterior mean of a mixture of two real token configurations that differ in k cells (k = 1:
  the HUD cell only; larger k: a consequence with a larger footprint), real per-cell separations from the same
  token means as A; probabilities 0.01-0.5; 20,000 runs per setting.
- Establishes the algorithm's retention of a rare branch as a function of its footprint (sufficiency or not); says
  nothing about the trained field.

C. Native-head robustness and external-reader qualification (v1 T0 split as the review asks; no universal gate).
- Native heads on denoised-real successors (A's outputs at sigma 0.05-0.377, near real tokens): death and negative-
  reward probabilities vs the same heads on the recorded successors. Robustness near real tokens only; it does not
  certify fully generated percepts.
- External HUD reader (nearest real token per value, support radius from held-out real tokens): qualified on held-out
  real data; on generated tokens it reports abstentions and support coverage, never a forced label.
- The v1 projection map is dropped (not the identity on real tokens, REVIEW sec. 2; normalisation constraints do not
  put a frame on the joint observation support).

D. Inference-settings sensitivity (v1 T1, reworded; runs after A).
- Claim: the frozen checkpoint's consequence rates depend on the sampling algorithm. Competing: they do not.
- One common panel (A's classes) and the same native heads, functional and geometry statistic (median per-token
  inverse-tanh variance; real 0.982). Each algorithm specified in full: Euler N = 3 (production, immutable); Heun
  (EDM Alg. 1) N = 8 / 16 / 32 (15 / 31 / 63 network evaluations); EDM Alg. 2 with churn at N = 32 with S_tmax set so
  that sigma_hat <= 20 (the review's 20.625 at sigma_max 20 is otherwise above the training clamp); sigma_max 5 and 10
  as separate axes. Throughput measured on a pilot before the panel size is fixed.
- Reading: rates rising toward the recorded with resolution -> inference settings matter for this checkpoint (with
  A: whether the field's conditional mean already carried them); no change across settings -> the algorithm is not the
  limit within this range. The geometry statistic per setting answers whether the token-geometry mismatch is a
  by-product of unresolved sampling (the user's question: a symptom, not a separate defect?). No additive split; a
  matched marginal rate does not establish conditional or joint correctness. SAFE gives the false-positive rate; the
  cooldown split shows conditional sharpness.

E. Information in the conditioning (v1 T3; one-sided).
- Probes for P(damage at t) on the denoiser's full conditioning (current predicted belief, the memory tokens of the
  cached context frames, the 8 context percepts with fixations, actions, validity: s1.conditioning_features audited
  and reused), capacity reported; physical-episode splits; log loss and calibration beside AP. Simulator reference
  with the exact rule: adjacency after the player's own action and movement (craftax_step: do_action, move_player, then
  update_mobs), cooldown <= 0, sleep (action -> NOOP, damage 7), no realised random draw.
- A strong probe shows the information is accessible; a failed probe shows nothing (one-sided). Run regardless of A.

F. Consequence catalogue (v1 T4) and bulk / tail summaries (v1 T5): descriptive only, after D. Event definitions with
opportunities, causal timing, positive controls and sample sizes; real-vs-real references; discrete PIT with random
tie-breaking for rank histograms (K = 16 resolves 1/16 per history; aggregate rates only).

Seed 2 (P1 world seed 2) is replication of training, outside this frozen-world diagnosis; it runs in the declared
budget when the GPU is free.

## Order
A (+ C's native-head part, same GPU job) and B (CPU) first; D after A; E; F. The deployed-pipeline numbers stay the
primary record throughout. No sampler, loss or architecture change follows from any of these without a separate
decision.

## Responses to the review (REVIEW.md)
1. Scope. Agreed that native heads are part of the model and that the deployed score stays the primary record.
   Disagree that inference-time substitution on a frozen checkpoint is an intervention outside the brief: diagnosis by
   substitution is the project's standing procedure, and the user approved the solver ladder (10-29). No training,
   loss or architecture change is made.
2. T0 map: accepted (verified: LayerNorm epsilon; median inverse-tanh variance 0.982 on real periphery tokens); the
   map is dropped. Denoised-real is robustness near real tokens only (C).
3. T1: accepted: algorithm-specific limits, no unique model distribution, no additive split, sigma_hat above 20 with
   churn (now capped), NFE stated. The SNR sentence in v1 was wrong: per-coordinate SNR at sigma 5 is 0.034; the data
   mean's norm (~30) is six noise SDs along its direction, and the first clean estimate already recovers it (29.82 vs
   30.01), so v1's start-offset argument is withdrawn.
4. T2: accepted as a probability test; removed. The proposed replacement (outcome-stratified denoising errors) has no
   null for an exact posterior mean, so A uses context classes, where the null is exact.
5. T3: accepted (full conditioning, post-move rule, one-sided reading).
6. T4 / T5: accepted as descriptive.
7. Seed 2: agreed it is replication, not diagnosis.
