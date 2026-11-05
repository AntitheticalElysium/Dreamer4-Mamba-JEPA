# N1 research: which loss weighting, if any, for tonight (11-05, for discussion with the user; nothing launched)

## 1. Where we stand (RESULTS.md 2-6)

- Data: a count uses the conditioning in proportion to its slot's training exposure (iron 63k events -> 0.02-0.10,
  coal 102k -> 0.10, drink 414k -> 0.44, wood 682k -> 0.62-0.72, stone 929k -> 0.63; the shared cell rule ~1.8M ->
  0.84-0.94). Stone (coal's route and composition) settles that route and digit composition do not block. Counts
  plateau near 0.6-0.7 (unexplained).
- Sampling: 3 Euler steps decode each row's probability by majority vote (production = P(first > 0.5); exact
  denoiser p 0.15 -> 0.02-0.03; 32 steps -> 0.145). Kills stochastic minority outcomes (damage) and hides any partial
  field progress. Stage B3 (frozen worlds, 8-32 steps) measures the real network's recovery.
- Noise: the signal that teaches "take this consequence from the conditioning" exists only at noise levels where the
  input hides it (sigma* = separation / sqrt(2 |logit p0|): 1.1-2.0 here). How much of the objective sits there is set
  by the loss weighting: N1 is a multiplier on every slot's exposure, not the cell / count split.

## 2. Verified literature (primary sources read 11-05)

- Kingma & Gao 2023 (NeurIPS, 2303.00848): "The only real difference between diffusion objectives is their difference
  in weighting w(lambda)"; L_w = 1/2 E[w(lambda)/p(lambda) ||eps_hat - eps||^2]; x-loss: w = e^-lambda p(lambda)
  (||x - x_hat||^2 = e^-lambda ||eps - eps_hat||^2); EDM: N(lambda; 2.4, 2.4^2)(e^-lambda + sigma_data^2). So ours
  (sigma_data 1) = x-loss + eps-loss: switching to x-loss divides each level's weight by 1 + 1/sigma^2 (101x at 0.1,
  2x at 1, ~1 above 3). Every positive weighting has the same minimiser (the posterior mean at every level): it changes
  priorities under finite capacity / steps, not the target. Best image FIDs there: EDM / EDM-monotonic, sigmoid(-lambda+2).
- Dreamer 4 (2509.24527): flow x_tau = (1 - tau) x0 + tau x1, tau = 1 clean; tau on a grid {0, 1/d, ...} with d from a
  power-of-two grid (pure noise included); x-prediction because "v-prediction trains the network to produce
  high-frequency outputs" that accumulate over long rollouts; loss in data space; ramp w(tau) = 0.9 tau + 0.1 "to focus
  the model capacity on signal levels with the most learning signal" (at low tau the flow "degenerates to predicting the
  dataset mean"); history lightly noised at inference (tau_ctx 0.1); K = 4 shortcut steps. Table 2 (cascade, FVD):
  diffusion forcing 306; 4 steps 875; + shortcut 329; x-prediction 326; x-loss 151; ramp 102. Caution: for a
  CONDITIONAL world model the "dataset mean" at pure noise is the history-conditioned consequence, exactly what our
  counts lack; the ramp's rationale is image-quality-driven and partly opposes our need.
- Hang et al. 2023, Min-SNR (ICCV, 2303.09556): x0 weight min(SNR, gamma), gamma 5; motivation "conflicting optimization
  directions" across timesteps, "to avoid the model focusing too much on small noise levels"; 3.4x faster to FID 10.
  It also decays as 1/sigma^2 above sigma 0.45.
- Choi et al. 2022, P2 (CVPR, 2204.00227): prioritise the "content" stage (SNR 1e-2-1, here sigma 1-10) over "clean-up"
  (SNR > 1); weight 1 / (k + SNR)^gamma on the eps-loss.
- Karras et al. 2024, EDM2 (2312.02696, eq. 21): per-level learned log-variance u(sigma) (Kendall uncertainty
  weighting) because a fixed lambda(sigma) cannot track per-level losses that change unevenly; drives every level
  toward an equal share (our HarmonyDream harmoniser is the same family and was measured rate-limited, 10-31).
- Kynkaanniemi et al. 2024 (2404.07724): guidance (the conditioning's push) is "harmful" at high noise, "largely
  unnecessary" at low noise, "only beneficial in the middle": where conditional content is decided is the middle.
- Frans et al. 2024, shortcut models (2410.12557): naive few-step predictions "point towards the average of plausible
  data points" and "cannot generate multi-modal data in a single step"; conditioning on the step size with the
  self-consistency s(x, t, 2d) = s(x, t, d)/2 + s(x', t+d, d)/2 fixes it. = our N2 rounding; Dreamer 4's sampling answer.
- Um, Lee & Ye 2024, minority guidance (ICLR, 2301.12334): diffusion samplers favour majority features.
- Qin et al. 2023, class-balancing diffusion (CVPR, 2305.00562): on class-imbalanced data tail classes lose fidelity and
  mode-collapse; fix by a distribution-adjustment regulariser (data-side).
- Kauvar et al. 2023, Curious Replay (2306.15934): curiosity-prioritised replay for model-based agents; DreamerV3 on
  Crafter 14.5 -> 19.4. A data lever with precedent in our domain; its priority is outcome-dependent, so it changes the
  replayed distribution (our E14 history: generic doses failed by prior shift / tail impurity).
- Song et al. 2025, history guidance (2502.06764): CFG over history frames improves consistency in diffusion-forcing
  video models: a sampling-time lever on conditioning use (not tonight).

## 3. Measured on F2's own training term (`weighting_allocation.py`; first-order proxy on F2's current errors)

| weighting (x-space w(sigma)) | < 0.3 | 0.3-1 | 1-3 | 3-7 | > 7 | teaching share vs ours |
|---|---|---|---|---|---|---|
| ours: EDM sigma_data 1, 1 + 1/sigma^2 | 51.5% | 25.1% | 15.4% | 6.0% | 2.0% | 1.0 |
| DIAMOND's EDM sigma_data 0.5 | 30.9% | 24.1% | 27.6% | 12.8% | 4.6% | 2.0-2.1 |
| Dreamer 4 x-loss, 1 | 4.3% | 22.9% | 43.2% | 21.7% | 7.8% | 3.2-3.5 |
| Dreamer 4 x-loss + ramp | 8.3% | 33.7% | 41.4% | 13.4% | 3.1% | 2.3-2.5 |
| EDM2-style equalised | 28.2% | 37.8% | 24.6% | 7.3% | 2.2% | 1.3-1.4 |
| P2 (k 1, gamma 1) | 12.8% | 49.4% | 33.8% | 3.8% | 0.2% | 0.9-1.5 |
| Min-SNR-5 | 21.0% | 61.8% | 15.8% | 1.3% | 0.1% | 0.3-0.7 |
| sigmoid(-lambda + 2) | 27.8% | 53.3% | 17.4% | 1.5% | 0.1% | 0.4-0.75 |

Teaching share = event rate x the consequence cells' weighted x-error on event frames at sigma >= sigma*, over the
whole weighted term (ours: coal 1.7e-4, iron 1.3e-4, wood 3.5e-4, drink 3.0e-4, risky 1.2e-4 of the objective).
F2.1 within 0.1. The image-domain favourites (Min-SNR, sigmoid) would make our failure worse: they move weight to
sigma 0.3-1, just below sigma*.

## 4. Options for tonight (one change per run; F2's exact recipe otherwise)

- A (recommended). x-space loss (Dreamer 4's largest single gain), constant 3.86 matching G1's magnitude (B4),
  training term only (the checkpoint monitor stays in F2's units). Teaching share x3.5, no change to the data or the
  draws (so no prior shift, context noise / N3 unchanged), the optimum unchanged. Risk: sigma < 0.3 share 51.5 -> 4.3%
  (fine detail, the last sampler calls), glimpse share 23 -> 14%.
- B. Dreamer 4's x-loss + ramp: x2.4, sigma < 0.3 8.3%; two changes, and the ramp's rationale opposes our need.
- C. sigma_data 0.5 (DIAMOND): x2.1, but also changes c_skip / c_out / c_in (not a pure weighting).
- D. Data (prioritised replay): precedent (Curious Replay) but an outcome-dependent priority shifts p(consequence |
  context) unless importance-corrected; our own history of prior shift. Not tonight.
- E. Sampling (shortcut forcing): the principled N2 fix and Dreamer 4's; a larger change; Stage B3 says first whether
  plain extra steps recover the frozen worlds (then shortcut forcing is an efficiency question).

Predictions for A, to be declared before launch: on the exposure curve the counts move left (coal, iron, drink up
by a declared margin at sigma 20); the cell stays; if the 0.6-0.7 plateau is signal-limited, wood / stone rise above
it; fidelity: low-sigma bias, regression panel. A null on all counts with the teaching share x3.5 would say the
allocation does not substitute for exposure (the data side then carries the explanation).

## 5. Correction (Codex review 11-05, verified): published weightings need 1 / p(lambda)

Kingma & Gao's weightings are TOTAL weightings over lambda (eq. 6: per-draw weight w / p). Section 3's Min-SNR, P2 and
sigmoid rows transplanted each paper's per-sample weight onto our draws; they are not the published objectives.
Recomputed (`weighting_allocation.py`, "published:" rows; per-draw x-weight = w*(lambda) / p_ours(lambda) x e^lambda,
lambda ~ N(0.8, 2.4^2) on our draws; the machinery reproduces our own row exactly). Teaching share vs ours, bootstrap
95% over rows (coal / drink / risky): published sigmoid(-lambda + 2) 0.62-0.65 / 0.77-0.81 / 0.57-0.59 (below ours; 43%
of the term below sigma 0.3 vs ours 51.5%); Min-SNR-5 0.57-0.62 / 0.89-0.97 / 0.45-0.50; P2 1.35-1.48 / 1.84-1.99 /
1.11-1.24; v-prediction / flow matching 1.4-1.6; Dreamer 4 x-loss + ramp AS PUBLISHED (uniform tau) 3.28-3.59 /
2.96-3.18 / 3.37-3.69 (= our x-loss arm, 3.31-3.59 / 3.05-3.27 / 3.41-3.67); Dreamer 4 x-loss with uniform tau, no ramp,
3.7-5.0. The earlier "Dreamer 4 full" row was our draws + ramp, mislabelled. Conclusions narrowed: the published
sigmoid and Min-SNR put LESS of the objective above sigma* than ours (modestly, not 0.3x); our x-loss arm equals Dreamer
4's published recipe on this measure. Also conceded (Codex): the glimpse is generated from noise through its own
schedule (0.1 is its coarse conditioning's level), so "a low-noise task" was not established; the glimpse's 23.4% share
holds on the calibration measurement, not throughout training.
