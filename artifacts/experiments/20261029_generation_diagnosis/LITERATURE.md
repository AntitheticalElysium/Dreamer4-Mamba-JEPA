# Literature on the diagnosed problem (2026-10-29)

Problem as measured (RESULTS A, E, F, G): the trained denoiser's conditional mean carries common, large-footprint
consequences (movement) and misses single-cell ones (drink, wood, table, damage, sapling), although a probe reads
drink from the denoiser's own conditioning (AP 0.90); over training, table and wood rise slowly, drink barely, damage
and sapling not at all. Question for the literature: why would available conditioning fail to produce small
consequences, and should late learning be expected? Sources read for the passages quoted; "our inference" marks what
the papers do not say themselves. Codex's earlier source audit: `../20261029_generation_literature/READING.md`.

## The papers M0 is built from
- EDM (Karras et al., NeurIPS 2022, sec. 5): lambda(sigma) = 1 / c_out^2 "equalizes the initial training loss over
  the entire sigma range"; p_train is log-normal because "a significant reduction is possible only at intermediate
  noise levels ... whereas at high levels the training targets are always dissimilar from the correct answer that
  approaches dataset average". M0 keeps lambda and DIAMOND's p_train (-0.4, 1.2).
- EDM2 (Karras et al., CVPR 2024, App. B, config B, eq. 21): lambda(sigma) "aims to ensure that the gradients
  originating from each noise level are roughly of the same magnitude", but the per-sigma loss "behaves somewhat
  unevenly over the course of training", so "no static choice of lambda(sigma) is sufficient"; noise levels are treated
  as a continuum of tasks with a learned uncertainty u(sigma) (Kendall et al.). For VAE latents "the in-between region
  has shifted considerably toward higher noise levels", hence P_mean -0.4, P_std 1.0. Our inference: M0 balances noise
  levels statically and balances nothing WITHIN a noise level (all 1,880 scalars share one weight).
- Dreamer 4 (Hafner et al., 2509.24527, sec. 3.2, Table 2): a ramp weight w(tau) = 0.9 tau + 0.1 "to focus the model
  capacity on signal levels with the most learning signal" because at low signal levels the flow term "degenerates to
  predicting the dataset mean"; FVD 151 -> 102 in the cascade. Actions enter as tokens of the interleaved sequence
  ("We encode each action component separately into S_a tokens"). Sec. 4.2: "inventory items are sometimes unclear or
  change over time"; Lucid-v1 generations show "object interactions being ignored"; Oasis "autocompletion". Our
  inference: even the strongest source model reports the inventory (HUD-like) weakness we measure.
- DIAMOND (Alonso et al., 2405.12399, sec. 5.2, App. L): one denoising step returns the conditional mean; three steps
  chosen on Atari; actions and noise level enter by adaptive group norm (our adaLN). M0 conditions the denoiser on the
  action by adaLN-Zero (DiT, global per frame) plus the predicted belief as 4 memory tokens.
- DiT (Peebles & Xie, ICCV 2023): adaLN-Zero beat cross-attention and in-context tokens for CLASS conditioning (a
  global label). Our inference: no source tests adaLN for a consequence that lands in one cell and depends on the
  conjunction action x faced cell; Dreamer 4 chose tokens for actions.
- JEDI (2605.13013, limitations / related work): no rare-event analysis; cites event supervision (EA-SIMULUS) as out
  of scope.
- GameNGen (Valevski et al., 2408.14837, sec. 3.2.2): HUD artefacts came from the pretrained autoencoder and were
  fixed by fine-tuning its decoder: a tokenizer issue, unlike ours (our tokens read the HUD exactly; the field misses
  its changes).
- Delta-IRIS (Micheli et al., ICML 2024): the autoencoder is conditioned on past frames and actions so that its tokens
  "only [have] to capture what has changed and that cannot be inferred from actions"; state of the art on Crafter.
  Our inference: a representation in which the bulk is already explained moves the objective's mass onto changes.

## Small or action-driven changes under uniform objectives
- OC-STORM (Zhang et al., 2501.16443): reconstruction losses "often fail to capture small, task-critical objects";
  an object-centric representation from a pretrained segmenter "can direct model capacity toward semantically
  meaningful entities".
- Segmentation Dreamer (Kim et al., 2410.09972): reconstruct only task-relevant components (segmentation mask), with
  the loss applied selectively.
- DWS (He et al., 2502.07825, sec. 3, eq. 2-3): "standard losses typically consider each pixel equally"; a
  motion-reinforced weight omega = c^softmax(|x_{i+1} - x_i|) "to improve the action-following ability of video
  models"; its ablation is combined with the action module, not isolated.
- EAWM (Peng et al., ICLR 2026, 2601.19336): events derived from raw observations without labels, a segmentor of event
  boundaries, event prediction shaping the representation; +10-45% over strong MBRL baselines incl. Craftax 1M.
- Video prediction copies the last frame when it pays: PredNet "resolves to mere last-frame copying, as it is
  statistically beneficial to do just that" (Rane et al., 1906.11902); separate motion and content pathways (Villegas
  et al., ICLR 2017, 1706.08033) as one remedy.
- Our own history: E14 (10-02) copying = rarity under uniform L1 + diluted precision; an attempt-mask x304 with a skip
  head worked head-only; generic doses failed (tail impurity, prior shift). E14f (10-03): a deterministic world bought
  consequence modes one by one with budget, sometimes in jumps (table 0.11 -> 0.86 between 50k and 54k updates, same
  order DO, stone, furnace, table at both seeds); health never at any budget to 100k. E20 (10-08): "too short /
  grokking late" refuted for the cooldown phase (flat 6k -> 100k); extra exposure to damage moved it.

## Late learning: what the theory says
- Saxe, McClelland & Ganguli (ICLR 2014): each input-output mode is learned on a timescale t ~ (tau / s) ln(s / eps),
  "inversely proportional to the correlation strength" s; trajectories are sigmoidal and "can exhibit sharp transitions
  from a state of no learning to full learning". Weak modes are learned late, and abruptly.
- Grokking: generalisation long after fitting (Power et al. 2022); tied to weight norm and weight decay ("LU
  mechanism", Liu et al., Omnigrok, ICLR 2023); or to a transition from lazy to rich feature learning, also without
  weight decay (Kumar et al., ICLR 2024).
- Noise levels as competing tasks: conflicting gradients between timesteps slow convergence (Min-SNR, Hang et al.,
  ICCV 2023); conditioning is relied on at high noise and "almost entirely ignored" at low noise (eDiff-I, Balaji et
  al. 2022).
- Our inference for the grokking question: the theory predicts that consequences with weak effective signal are
  learned late and possibly in jumps, as E14f showed for deterministic modes; whether that happens here depends on the
  effective strength of each consequence's signal under the actual optimiser (the accounting) and on the learning
  rate, which decays ~6x over P1's second half. E14f / E20 also say budget alone did not buy the hidden-state damage;
  exposure did.
