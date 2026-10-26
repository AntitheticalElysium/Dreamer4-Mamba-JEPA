# World-model losses: units and weights v1.1 (2026-10-27; generator v3, outcome models and joint score synced)

Terms from STEP_SPEC.md v0 §3. Starting values are working numbers. The actor's and fixation head's objectives are
separate (no gradient into the world).

## Principles (literature + our history)
1. Likelihood-like units where they exist (DreamerV3: every term a likelihood, fixed weights across domains), and
   each term AVERAGED per element (token, pixel, cell, step), so a term's weight is not set by how many elements it has.
   Sums over elements stay only inside proper scores where the joint matters (the energy score of a patch or trajectory vector). The critic's value head keeps symlog two-hot (it is a regression,
   never sampled).
2. Natural frequencies; no event rebalancing (08-29 terminal tails and E14 positives-only installed shortcuts).
3. Ratios that ARE design choices stay fixed: the gradient scale from the next-percept loss into the encoder (the weak
   coupling) and SIGReg's lambda. LeWM's 0.1 belongs to its own two-term objective; with a denoising loss and
   harmonised groups it is a starting value, not an established ratio (Codex).
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
| (fixed) | SIGReg | Epps-Pulley statistic over random projections (LeJEPA) on periphery and glimpse tokens | statistic | lambda = 0.1 relative to G1 |
| G1 (shared) | own-sample unrolls | the G1 term at each unrolled step while the update reads the generator's own percepts (context corruption / random switching), truncated gradients | mean over steps | G1's sigma |
| G4 sampled rollouts | free-running sampled unrolls on exogenous windows | JOINT-TRAJECTORY energy score (beta = 1, m >= 2 independent trajectories with independent termination draws, each trajectory one vector of per-step percept summaries + reward + termination, components scaled by data std; terminal convention per CURRICULUM); framewise / patched scores supplementary. The label-free, differentiable PERCEPT SUMMARY used for training is still to be declared before P2; it is not the S0 evaluation summary (simulator labels via the reader), whose count + centroid scope is documented in saccade/eval/gates.py | per trajectory | harmonised |

## Facts used
Change-map positives 1.7% (day) / 2.9% (night) of cells (A9b); reward non-zero on 6.7% of steps, termination 0.3%
(corpus); exogenous windows from A8. (Free bits left with generator v2; they were never an entropy budget, and the
1.37-bit figure is movement-only and conditioned on full simulator state, Codex.)

## Open
Window size and sample count for the energy scores; sigma initialisation; noise-level weighting for the denoising
loss; schedule lengths; per-route gradient-norm alarms; how to protect rare entities inside the next-percept group.
The P2 training summary for G4 and its gradient boundary (declared before P2).
