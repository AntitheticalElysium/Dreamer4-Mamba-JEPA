# World-model losses: units and weights v1.1 (2026-10-27; generator v3, outcome models and joint score synced)

Terms from STEP_SPEC.md v0 §3. Starting values are working numbers. The actor's and fixation head's objectives are
separate (no gradient into the world).

## Principles (literature + our history)
1. Likelihood-like units where they exist (DreamerV3: every term a likelihood, fixed weights across domains), and
   each term AVERAGED per element (token, pixel, cell, step), so a term's weight is not set by how many elements it has.
   Sums over elements stay only inside proper scores where the joint matters (the energy score of a patch or trajectory vector). The critic's value head keeps symlog two-hot (it is a regression,
   never sampled).
2. Natural frequencies; no event rebalancing (08-29 terminal tails and E14 positives-only installed shortcuts).
3. Ratios that ARE design choices stay fixed: the encoder's learning-rate scale (m0.yaml).
   Weak coupling, reconciled 10-27 (Codex: a slower encoder is not the same as weakening only the dynamics route):
   under Adam, rescaling an encoder route's gradient changes only its SHARE of the encoder's update, while the
   encoder's learning-rate scale sets its SPEED relative to the denoiser and heads. M0 uses JEDI's mechanism, the
   speed: encoder lr x 0.3 with stop-gradient targets (Tang et al. 2023: the predictor must learn faster than the
   representation; JEDI's ablation: removing the denoising gradient into the encoder hurts). It slows every route into
   the encoder alike (grounding included); the mix between routes is set by the group weights. DreamerV3's per-route
   0.1 does not transfer one-to-one: here the denoising gradient reaches the encoder only through the belief (past
   percepts -> Mamba -> memory tokens), which also carries the outcome heads' gradients. A per-route scale stays a
   documented alternative.
7. Collapse prevention (decided 10-27; decision record: COLLAPSE.md): the chosen mechanism is ROUTING plus
   GROUNDING, not a distribution regulariser; entity retention is still measured in S1 (Codex). (a) Every FRAME
   input of the denoiser and its targets are stop-gradient (noisy frames, context-prefix frames, targets; the memory
   tokens stay differentiable, they are the route below): REPA-E measured that a diffusion loss reaching
   the encoder through the latent collapses it (spatial variance 17.06 -> 0.02; FID 444 without the stop-gradient);
   JEDI stops its target and its own noisy input. The denoising loss reaches the encoder only through the belief
   (past percepts -> Mamba -> memory tokens), never through the frame being predicted, and the encoder learns 0.3x
   slower (Tang et al.: predictor faster than representation). (b) Each token type has a fixed-target owner: periphery
   tokens by TSP and the change map, glimpse tokens by the fovea's own reconstruction (G2), the lighting latent through
   both pixel likelihoods, all of them through the outcome heads. (c) Token scale for the diffusion (sigma_data = 1):
   each token is layer-normalised over its own channels without affine, THEN soft-clamped by JEDI's C(z) = 3 tanh(z/3).
   Precedent: I-JEPA normalises its target tokens this way, and V-JEPA 2-AC, an action-conditioned JEPA world model,
   normalises every latent state this way, encoder outputs and the predictor's outputs before they are fed back alike
   (vendored code). Per token, it is stateless and causal and identical in training, imagination and P2 (Codex:
   BatchNorm over a chunk made a past percept depend on future frames and other episodes, and BatchNorm after tanh
   broke the bound). The second moment of every token is 1 before the clamp (sigma_data = 1) and the clamp bounds each
   element to +-3. The denoiser's denoised estimate passes through the same normalisation and clamp at every sampler
   evaluation (DIAMOND clamps each denoised estimate to its data range; JEDI clamps denoiser outputs with C; V-JEPA
   2-AC re-normalises predicted states); the training loss does not include it (DIAMOND). Diagnostics, never gates:
   token norm before normalisation, effective rank, nuisance drift ratio (R56 definition).
4. Terms in different units are equalised by a learned harmoniser per GROUP: L_g / sigma_g + log(1 + sigma_g)
   (HarmonyDream, ICML 2024). CORRECTION (Codex): with mu = E[L_g] the optimum satisfies sigma^2 = mu (1 + sigma), not
   sigma = mu (that is the unrectified log sigma form); the normalised contribution is sigma / (1 + sigma) < 1, and as mu
   -> 0 the weight still grows (~ 1 / sqrt(mu)), only more slowly than with log sigma. Balancing GROUP losses does not
   balance their gradients, nor protect rare entities inside a group (a well-scaled next-percept loss can still be
   mostly terrain); that concern stays open and is a probe target. Kurin et al. 2022: plain weighted sums match fancy
   multi-task optimisers, and HarmonyDream is a learned weighted sum, so no gradient-surgery optimiser.
5. Schedules instead of weights: rollout horizon grows gradually (NeuralGCM: "critical"); context corruption level
   (GameNGen; Dreamer 4 tau_ctx = 0.1); prior-sampled scoring starts after one-step training is stable.
6. The encoder's gradient budget is monitored per route (belief path, TSP, change map, fovea reconstruction,
   outcome heads; the denoiser's frame path is stop-gradient). Patch-local encoding removes the screen-position routes behind the D4MJ transport conflict; it does not
   make the objectives agree (equivariance still depends on fixed viewing conditions; regulariser, grounding and
   prediction can still compete).

## Terms
| group | term | form | unit, normalisation | weight |
|---|---|---|---|---|
| G1 next step | next percept (generator v3) | denoising loss on the next percept (lighting latent, periphery, glimpse jointly), EDM noise levels, x-prediction, conditioned on the predicted belief. Weighting (pinned 10-27, Codex): EDM's lambda(sigma) = 1 / c_out^2, i.e. MSE on the raw network output against (x0 - c_skip x_sigma) / c_out (DIAMOND's code; JEDI's Algorithm 1 writes the clean-space form without a weight), equivalent to clean-space MSE weighted by 1 + 1/sigma^2 at sigma_data = 1 (Codex: equal to 4.6e-13); no noise offset (DIAMOND's 0.3 offsets image brightness; our lighting has its own token); diffusion forcing gives each frame its own sigma | squared distance; mean over ALL elements of a frame's percept (81 x 16 periphery + 9 x 64 glimpse + 8 lighting = 1,880; each scalar equal), then over frames | harmonised |
| G2 grounding | change map | BCE against the soft label-free target (A9b) | nats per cell; mean over cells | harmonised |
| G2 grounding | TSP | discretised logistic NLL; mean from (belief, periphery, v), scale from (v, position) | nats per pixel; mean over the glimpse | harmonised |
| G3 consequences | reward (observation model) | negative log-likelihood of a discretised mixture of logistics over symlog(r) (support: the train split's raw reward range [-0.9, 1.1], cut into 256 bins uniform in symlog space; the edge bins absorb the tails (PixelCNN++); the same bins score the S1 calibration references, saccade/eval/calib.py). Sampled VALUE (pinned 10-27, Codex: the zero-reward bin's centre decodes to -0.0013): a sampled bin decodes to the mean of the train-split rewards falling in it (the MSE-optimal representative of a quantiser cell, Lloyd 1982; the symexp of the bin centre if none), so deterministic rewards are reproduced exactly (0 -> 0, 1 -> 1) | nats; mean over steps | harmonised |
| G3 consequences | continuation | BCE, conditioned on (predicted belief, percept, reward) | nats; mean over steps | harmonised |
| G2 grounding | fovea reconstruction (10-27) | discretised logistic NLL of the glimpse crop's pixels from its own glimpse tokens and v (TSP's pixel decoder and noise model) | nats per pixel; mean over the crop | harmonised |
| G1 (shared) | own-sample unrolls | the G1 term at each unrolled step while the update reads the generator's own percepts (context corruption via diffusion forcing; switching dropped for M0), truncated gradients | mean over steps | G1's sigma |
| G4 sampled rollouts | free-running sampled unrolls on exogenous windows | PATCHED + GLOBAL joint-trajectory energy score (agreed 10-27; beta = 1, fair pairwise term, m = 2 independent rollouts with independent noise and termination draws; Y = the recorded window through the frozen encoder). Patched: zero-padded 3x3 windows of periphery tokens centred on every position of the 11 x 11 grid that extends the 9 x 9 screen by one cell (121 windows), so every token, HUD and view edges included, lies in exactly 9 windows (Codex: windows centred only on the 81 tokens covered corners 4 times and edges 6); padding is identical in samples and recording, so it adds no distance. Stacked over the h steps. Global: the full flattened trajectory (all periphery tokens, lighting latent, glimpse tokens, symlog reward, termination); a pooled global would miss distant dependencies (Codex's corner counterexample). Every component scaled once by its std over 10,000 recorded windows at the start of P2; absorbing terminal convention; weights: the patch sum 1, the global set once at the start of P2 to equal mean magnitude (Pacchiardi et al., E.4.2; their patches are spatial at one lead time, stacking over time is ours). Gradient: CURRICULUM §2 contract. Distinct from the S0 evaluation summary (simulator labels via the reader) | per trajectory | harmonised |

## Facts used
Change-map positives 1.7% (day) / 2.9% (night) of cells (A9b); reward non-zero on 6.7% of steps, termination 0.3%
(corpus); exogenous windows from A8. (Free bits left with generator v2; they were never an entropy budget, and the
1.37-bit figure is movement-only and conditioned on full simulator state, Codex.)

## Open
Sigma initialisation; schedule lengths; per-route gradient-norm alarms; how to protect rare entities inside the next-percept group.
