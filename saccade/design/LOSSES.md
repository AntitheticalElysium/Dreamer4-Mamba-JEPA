# World-model losses: units and weights v1.1 (2026-10-27; generator v3, outcome models and joint score synced)

Terms from STEP_SPEC.md v0 §3. Starting values are working numbers. The actor's and fixation head's objectives are
separate (no gradient into the world).

## Principles (literature + our history)
1. Likelihood-like units where they exist (DreamerV3: every term a likelihood, fixed weights across domains), and
   each term AVERAGED per element (token, pixel, cell, step), so a term's weight is not set by how many elements it has.
   Sums over elements stay only inside proper scores where the joint matters (the energy score of a patch or trajectory vector). The critic's value head keeps symlog two-hot (it is a regression,
   never sampled).
2. Natural frequencies; no event rebalancing (08-29 terminal tails and E14 positives-only installed shortcuts).
3. Ratios that ARE design choices stay fixed: the encoder's learning-rate scale and SIGReg's lambda (m0.yaml).
   Weak coupling, reconciled 10-27 (Codex: a slower encoder is not the same as weakening only the dynamics route):
   under Adam, rescaling an encoder route's gradient changes only its SHARE of the encoder's update, while the
   encoder's learning-rate scale sets its SPEED relative to the denoiser and heads. M0 uses JEDI's mechanism, the
   speed: encoder lr x 0.3 with stop-gradient targets (Tang et al. 2023: the predictor must learn faster than the
   representation; JEDI's ablation: removing the denoising gradient into the encoder hurts). It slows every route into
   the encoder alike (grounding and SIGReg included); the mix between routes is set by the group weights. DreamerV3's
   per-route 0.1 does not transfer one-to-one: here the denoising gradient reaches the encoder through conditioning
   paths (past percepts, the belief) that also carry the grounding gradients. A per-route scale stays a documented
   alternative.
7. Collapse prevention (decided 10-27 after a dedicated investigation; SIGReg dropped from M0, see READING 10-27):
   collapse is prevented by the ROUTING and by GROUNDING, not by a distribution regulariser. (a) Every denoiser input
   and target is stop-gradient (noisy frames, context frames, targets): REPA-E measured that a diffusion loss reaching
   the encoder through the latent collapses it (spatial variance 17.06 -> 0.02; FID 444 without the stop-gradient);
   JEDI stops its target and its own noisy input. The denoising loss reaches the encoder only through the belief
   (past percepts -> Mamba -> memory tokens), never through the frame being predicted, and the encoder learns 0.3x
   slower (Tang et al.: predictor faster than representation). (b) Each token type has a fixed-target owner: periphery
   tokens by TSP and the change map, glimpse tokens by the fovea's own reconstruction (G2), the lighting latent through
   both pixel likelihoods, all of them through the outcome heads. (c) Token scale for the diffusion (sigma_data = 1)
   is kept by JEDI's tanh clamp followed by REPA-E's normalisation (BatchNorm, no affine), not by shaping the encoder.
   Diagnostics, never gates: token std before normalisation, effective rank, nuisance drift ratio (R56 definition).
4. Terms in different units are equalised by a learned harmoniser per GROUP: L_g / sigma_g + log(1 + sigma_g)
   (HarmonyDream, ICML 2024). CORRECTION (Codex): with mu = E[L_g] the optimum satisfies sigma^2 = mu (1 + sigma), not
   sigma = mu (that is the unrectified log sigma form); the normalised contribution is sigma / (1 + sigma) < 1, and as mu
   -> 0 the weight still grows (~ 1 / sqrt(mu)), only more slowly than with log sigma. Balancing GROUP losses does not
   balance their gradients, nor protect rare entities inside a group (a well-scaled next-percept loss can still be
   mostly terrain); that concern stays open and is a probe target. Kurin et al. 2022: plain weighted sums match fancy
   multi-task optimisers, and HarmonyDream is a learned weighted sum, so no gradient-surgery optimiser.
5. Schedules instead of weights: rollout horizon grows gradually (NeuralGCM: "critical"); context corruption level
   (GameNGen; Dreamer 4 tau_ctx = 0.1); prior-sampled scoring starts after one-step training is stable.
6. The encoder's gradient budget is monitored per route (belief path, denoiser target path is stop-gradient, TSP,
   SIGReg). Patch-local encoding removes the screen-position routes behind the D4MJ transport conflict; it does not
   make the objectives agree (equivariance still depends on fixed viewing conditions; regulariser, grounding and
   prediction can still compete).

## Terms
| group | term | form | unit, normalisation | weight |
|---|---|---|---|---|
| G1 next step | next percept (generator v3) | denoising loss on the next percept (lighting latent, periphery, glimpse jointly), EDM noise levels, x-prediction, conditioned on the predicted belief | squared distance; mean per token | harmonised |
| G2 grounding | change map | BCE against the soft label-free target (A9b) | nats per cell; mean over cells | harmonised |
| G2 grounding | TSP | discretised logistic NLL; mean from (belief, periphery, v), scale from (v, position) | nats per pixel; mean over the glimpse | harmonised |
| G3 consequences | reward (observation model) | negative log-likelihood of a discretised mixture of logistics over symlog(r) (support: the symlog range of the train split, [-0.9, 1.1], 256 uniform bins; sample, then symexp; the same bins score the S1 calibration references, saccade/eval/calib.py) | nats; mean over steps | harmonised |
| G3 consequences | continuation | BCE, conditioned on (predicted belief, percept, reward) | nats; mean over steps | harmonised |
| G2 grounding | fovea reconstruction (10-27) | discretised logistic NLL of the glimpse crop's pixels from its own glimpse tokens and v (TSP's pixel decoder and noise model) | nats per pixel; mean over the crop | harmonised |
| G1 (shared) | own-sample unrolls | the G1 term at each unrolled step while the update reads the generator's own percepts (context corruption / random switching), truncated gradients | mean over steps | G1's sigma |
| G4 sampled rollouts | free-running sampled unrolls on exogenous windows | PATCHED + GLOBAL joint-trajectory energy score (agreed 10-27; beta = 1, fair pairwise term, m = 2 independent rollouts with independent noise and termination draws; Y = the recorded window through the frozen encoder). Patched: one zero-padded 3x3 window of periphery tokens CENTRED on each of the 81 tokens (each token covered 9 times, HUD included), stacked over the h steps. Global: the full flattened trajectory (all periphery tokens, lighting latent, glimpse tokens, symlog reward, termination); a pooled global would miss distant dependencies (Codex's corner counterexample). Every component scaled once by its std over 10,000 recorded windows at the start of P2; absorbing terminal convention; weights: the patch sum 1, the global set once at the start of P2 to equal mean magnitude (Pacchiardi et al., E.4.2; their patches are spatial at one lead time, stacking over time is ours). Gradient: CURRICULUM §2 contract. Distinct from the S0 evaluation summary (simulator labels via the reader) | per trajectory | harmonised |

## Facts used
Change-map positives 1.7% (day) / 2.9% (night) of cells (A9b); reward non-zero on 6.7% of steps, termination 0.3%
(corpus); exogenous windows from A8. (Free bits left with generator v2; they were never an entropy budget, and the
1.37-bit figure is movement-only and conditioned on full simulator state, Codex.)

## Open
Sigma initialisation; noise-level weighting for the denoising
loss; schedule lengths; per-route gradient-norm alarms; how to protect rare entities inside the next-percept group.
