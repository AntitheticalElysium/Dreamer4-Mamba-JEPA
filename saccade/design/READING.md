# System design discussion — reading notes and measurements (2026-10-23)

Context: design discussion with the user (offline imagination training; reconstruction-free except a fixed-budget,
consequence-driven fovea; jointly trained encoder; grounded targets = reward, termination, fovea pixels). Three open
questions: (1) state structure, (2) keeping imagined states readable, (3) "the model doesn't know" and dynamic horizons.
Nothing here is a decision; decisions are recorded only when the user agrees.
Scripts named below (eda_*.py, ana_*.py, t1_/t2_*) are in artifacts/experiments/20261023_system_design/; dated folders
(2026MMDD_*) are under artifacts/experiments/.

## Our measurements (evaluation-only simulator state; TRAIN split; nothing trained)

| script | data | result |
|---|---|---|
| `consequence_locality.py` | 400 episodes, 125,015 transitions | 28% of steps have a one-step consequence (median 1 cell). 99.8% of consequence cells are the player's 4 neighbours + HUD; 0.16% elsewhere. 2 glimpses cover all in 92-96% of event steps. Zombies at distance 2-3: 72% outside the player's 3x3. Night static at night map events: mean 0.09, p90 0.17. |
| `action_coverage.py` | 1,200 episodes, 375,927 transitions | 94% of steps are in coarse contexts where all 17 actions were observed (89% with >= 5 each). Can sleep + zombie near/adjacent: 719 SLEEPs, death within 16 = 0.70 vs moves 0.15-0.18. Ready adjacent zombie: 166 SLEEPs, 0.74 vs 0.13-0.17. Coverage comes from the epsilon-noise episodes. |
| `entity_stats.py` | 600 episodes, 198,926 frames | Visible mobs per frame: 0: 35%, 1: 33%, 2: 21%, 3: 8%, 4: 2.5%, >= 5: 0.7% (max 8). Re-entries of an already-seen entity outnumber first sightings (zombies 3,316 vs 2,809; cows 5,253 vs 3,227; skeletons 2,848 vs 1,558); median gap 2 / 5 / 3 steps, p90 17 / 29 / 25. |
| corpus rewards | 2,312 episodes, 780,914 transitions | reward != 0 on 6.7% of steps; achievement unlock 3.2% (10.8 per episode); termination 0.3% of steps. |
| source facts | `craftax_classic` | Night static: per-pixel blend with uniform(32,127), strength 2(0.5-light) x (1 - exp(-2 r^2)): 0.02 at the player, up to 0.96 at the edges, none on the HUD. Zombie spawn chance 0.02 + 0.1 (1 - light)^2 (6x at full darkness). Reward = new achievements + 0.1 x health change. |
| prior (WHITEBOARD §2, 10-12 feasibility) | | chance ~1.37 bits per visible mob, near-independent (total correlation 3.8%); hunger / thirst / fatigue need ~64 frames, recover the whole episode. |
| prior (July, experiment-log) | Pong / Maze / Memory Maze | single-model sample disagreement u_s predicts latent error AUC ~0.74 at short horizon, falls to 0.47 (JEPA latent) / 0.64 (generative) at steps 12-16; naive shared-trunk heads give u_s ~ 0; randomized-prior heads fix it. |

## Q1 — state structure

- Grid tokens are the strongest on Craftax-Classic: Dedieu et al. 2025 (67.4% return, 1M online steps); ITC (Kim et al.,
  arXiv 2605.16457) 72.5%: copy-or-generate assignment with a displacement cap, much of the gain from 3D RoPE; per-step
  accuracy with creatures 38.4% vs 65.5% without (baseline 33.8 / 61.0); solver only in the central region.
- Object slots: OC-STORM (arXiv 2501.16443, ICLR 2026) gains where decision-relevant information fits objects, weak on
  geometry; DLPWM (arXiv 2511.06136) object latents drift near interactions and policies underperform DreamerV3 (no
  mitigation tested); slot identity through occlusion / out-of-frame is unsolved (iSEE 2610.01201; Structured World
  Belief 2107.08577 uses SMC + object permanence); slot quality helps planning up to saturation (2608.12078).
- Slots without pixels: DINOSAUR / VideoSAUR reconstruct frozen self-supervised features (pixel-reconstruction slot
  attention fails with ViT encoders); C-JEPA (2602.11389) object-level masked JEPA, control comparable with ~1% of the
  latent features; C-SWM (1911.12247) contrastive, simple environments.

## Q2 — readable imagined states

- TD-MPC2 (2310.16828): loss = sum over an unrolled horizon of lambda^t (||z'_t - sg(h(s'_t))||^2 + CE reward + CE Q),
  lambda 0.5; heads trained on unrolled latents; decoder-free; SimNorm for stability.
- EfficientZero (2111.00210): MuZero + self-supervised consistency between unrolled and encoded latents; "critical" with
  high-dimensional observations and limited data.
- MuZero / Unplugged (2104.06294): heads on unrolled latents only; offline Atari median 265% vs BC ~53%.
- DreamerV3: heads on posterior latents, KL keeps prior ~ posterior. Dreamer-CDP (reconstruction-free): 16.2 full,
  6.3 without the KL alignment (WHITEBOARD §7).
- Stochastic MuZero: chance code from an encoder of the real outcome during the unroll, prior trained to match it.
- Ni et al. 2024 (2401.08898): stop-gradient l2 self-prediction has stationary points satisfying EZP (expected next
  latent); biased in stochastic tasks; stop-gradient / EMA targets avoid rank collapse (also Tang et al. 2023).
- Koopman Dreamer (2607.19719, preprint): multi-step teacher / prior rollout objectives to close the posterior-prior gap.

## Q3 — the model doesn't know

- MOReL (2005.05951): unknown regions -> low-reward absorbing HALT. LOMPO (2012.11547): latent ensemble disagreement
  penalty, offline from images. RWM-U (2504.16680): per-member variance = aleatoric, disagreement = epistemic, offline robot
  world model. STEVE (1807.01675): inverse-variance blending of k-step targets across horizons (dynamic horizon).
- Lu et al. 2022 (2110.04135): ensemble variance tracks true dynamics error best among common penalties; horizon,
  ensemble size and penalty interact strongly.
- Biased Dreams (2604.25416, RLC 2026): latent dynamics have an attractor effect; ensemble disagreement captures local
  uncertainty but not global compounding error; no remedy proposed.
- Hansen & Wang 2026 (2606.27326): hallucination concentrates in low-coverage regions; mitigation by data (coverage-aware
  sampling, curiosity collection, finetuning).
- Ensemble diversity comes mostly from training randomness (2510.22063); shared trunks lose it; randomized priors /
  epinets restore it cheaply.
- Timor et al. (2605.06732): the dynamics term of the return error blows up as gamma L_f (1 + L_pi) -> 1; low-Lipschitz
  dynamics bound compounding error.

## Second reading round (after the user's 2026-10-23 challenge)

**Memory and structure without hand-coded coordinates**
- Po et al. 2025 (2505.20171, ICCV): Memory Maze retrieval PSNR: full-context causal 32.6; plain Mamba2 20.4; Mamba2 +
  frame-local attention 19.3; block-wise (spatial-block) Mamba scan + local attention 30.8. Memory beyond the training
  context length is not handled.
- WorldMem (2504.12369), VMem (2506.18903), Wu et al. 2025 (2506.05284): revisit consistency via memories indexed by pose /
  geometry. EgoMap (2002.02286): egocentric map updated by ego-motion measurements (differentiable affine transform).
- TEM (Whittington et al. 2020, Cell): a learned location code (path integration from actions) bound to sensory content in
  memory; TEM-t (2112.04035, ICLR 2022): a transformer with recurrent (action-updated) position encodings reproduces it.
  So "anchoring" can be learned from one's own actions; it is not inherently a hand-coded coordinate.
- R2I (2403.04253, ICLR 2024): S4-variant SSM in DreamerV3, state of the art on Memory Maze, BSuite, POPGym.
- SlotSSMs (2406.12272, NeurIPS 2024): per-slot SSM states with sparse cross-slot attention; strong on long-context
  multi-object prediction and reasoning.
- Jelassi et al. 2024 (2402.01032): fixed-state SSMs are provably weaker than transformers at copying / retrieval;
  Zoology (2312.04927): 82% of the attention vs gated-convolution LM gap is in-context recall.

**Anti-collapse and unroll length in working systems (no consensus)**
- Dreamer-CDP (2603.07083): stop-gradient cosine to the deterministic embedding (weight 500), KL balancing with a floor of
  1, no EMA; encoder learning rate 6e-6 vs RSSM 4e-4 (~67x slower: the Tang et al. 2023 "faster predictor" condition).
  Crafter 16.2 full; 12.7 without reward gradient; 6.3 without KL; 3.2 without next-embedding prediction. States that
  decoder-free methods have been worse than reconstruction on Crafter.
- NE-Dreamer (2603.02765, ICLR 2026): causal transformer forecasting the next encoder embedding, Barlow Twins loss.
- LeWM (2603.19312): end-to-end, prediction + SIGReg, claims stability without EMA. PLDM (2502.14819): VICReg-style +
  inverse dynamics. AC-MTM "No Gaussian Required" (2608.17542): contrastive inverse dynamics matches SIGReg on 4 tasks and
  beats it on a multi-object scene task (80.0 vs 58.0%), attributing SIGReg's deficit to its environment-independent
  geometry.
- "What drives success in JEPA-WMs" (2512.24497): with frozen encoders, a 2-step rollout loss is best in simulation, more
  steps hurt; 6 steps best on DROID; training context must cover the planning context.
- Tang et al. 2023 (2212.03319): self-predictive learning avoids collapse with semi-gradient (stop-gradient) targets and a
  faster-paced predictor. Ni et al. 2024: stop-gradient l2 stationary points satisfy EZP.

## Memory deep dive (2026-10-23, user-requested; brain + ML)

**Brain (verified sources)**
- CLS (Kumaran, Hassabis & McClelland 2016): slow structured neocortical learning + fast hippocampal storage of specifics;
  replay weights experience by goals; reorganisation over time (not an inert cache).
- Indexing theory (Teyler & DiScenna 1986; Teyler & Rudy 2007): the hippocampal index is distinct from the content;
  reactivating the index reinstates the content. Pattern separation (DG/CA3 sparse codes) at encoding, pattern completion
  from partial cues at retrieval (Rolls). Chickadee "barcodes" (Chettih et al. 2024, Cell): sparse, unique per cache,
  uncorrelated even for nearby sites with similar place codes, reactivated at retrieval.
- Cognitive maps learned from sequences: CSCG (George et al. 2021) and "Space is a latent sequence" (Raju et al.
  2022/2024): allocentric representations from higher-order sequence learning on egocentric inputs, no coordinates.
  TEM (Whittington et al. 2020): trained to predict the next observation; location code g by learned path integration
  from actions; Hebbian memory binds g and content; retrieval content->location (corrects drift) and location->content
  (predicts the view); structure transfers across environments (weights shared, memory reset). Limits: noiseless inputs,
  static content, simple graphs. TEM-t (2112.04035): a transformer with recurrent (action-updated) position encodings.
  Successor representation (Stachenfeld et al. 2017): place codes are predictive (expected future occupancy).
  Caution (Schaeffer, Khona & Fiete 2022): "emergent" grid codes in deep nets depend on post-hoc implementation choices.
- Imagination is reconstruction from memory: hippocampal amnesics' imagined scenes have details but no spatial coherence
  (Hassabis et al. 2007, PNAS); Bicanski & Burgess 2018 (eLife): imagery = allocentric memory transformed to an
  egocentric view (retrosplenial gain fields, head direction).
- Event segmentation: hippocampal encoding peaks at event boundaries and predicts later reinstatement (Baldassano 2017).
- Source monitoring (Johnson): memories carry no explicit source tag; imagined vs perceived is inferred; confabulation.

**ML mechanisms**
- MERLIN (1803.10760): the predictive model writes compressed predictive states into memory; predictor and policy read
  it (policy read-only, separate parameters); "memory is not enough ... the right information in the right format".
- Modern Hopfield = attention (2008.02217): exponential capacity, one-step retrieval, metastable averaging of similar
  patterns. Scene Memory Transformer (1903.03878): stores all observation embeddings, attention, beats RNN policies.
- Titans (2501.00663): short-term attention + surprise-gated neural long-term memory updated at test time + persistent
  weights. EM-LLM (2407.09450, ICLR 2025): event segmentation by Bayesian surprise, retrieval by similarity + temporal
  contiguity. Pose-attention navigation transformer (2609.21212): observations indexed by pose, RoPE on relative pose.
- No latent Dreamer-style world model enforcing imagination consistency with retrieved memory was found (LLM-agent
  retrieval world models: RAWM, R-WoM, AriGraph).

## Memory EDA (2026-10-23)
- M2 `eda_window_memory.py` (301 episodes, 1.03M entering-cell events in 16-step windows): terrain entering view last
  seen before the window: never 41%, <=16 steps 27%, 16-64 14%, 64-256 12%, >256 5%; re-entering terrain within
  Manhattan 7 (p50) / 11 (p90) / 23 (max) of the window start. Entities appearing: never 60-68%, <=16 24-28%,
  16-64 7-11%, >64 1-2%.
- M1 `eda_expert_amnesia.py` (the data-generating PPO-RNN expert; ~1000 envs at t0 = 200 / 500 / 1000; identical states
  and random keys; branches: intact, memory rebuilt from the last 64 / 16 observations, none): first-step action
  agreement with intact 0.68-0.76 (64), 0.61-0.69 (16), 0.31-0.34 (none); 300-step survival and achievements not
  reduced by any branch; water-seeking (thirsty, water out of view) not slowed by M64 or M16 (faster in several cells);
  full amnesia slower only for water last seen 64-256 steps before at t0 = 1000 (+20.8 steps [3.1, 40.3]).

## Catch-up reading (2026-10-24)
- SLIM (2610.03137, full text): LeWM drops bodies that change ~0.37% of the frame; SIGReg "prevents global collapse
  without choosing which variance survives". Fix = inverse dynamics h(z_t, z_t+1) -> a_t with gradient into the encoder
  (encoder term alone gives ~85% of the gain; shuffled-action null 0.002). A patch decoder alone kept plain-level
  probes: a pixel target protects content only in proportion to its share of the target. Frozen DINOv2 + IDM 0.163,
  below half of jointly trained LeWM + IDM.
- TEM-t (2112.04035): position e_t+1 = sigma(e_t W_a), learned per action; keys/queries from positions only, values from
  stimulus only; a memory is added only if no existing memory matches that (position, stimulus) pair; softmax scaled by
  log(n_memories). 4-connected graphs, one-hot random stimuli, no blocked moves, no moving objects, no noise.
- MERLIN (1803.10760): writes the posterior z_t every step (plus a retroactive filtered sum of later z); losses =
  image, reward, return, previous action, velocity; policy reads the same memory read-only (stop-gradient "not necessary").
- Dreamer 4 (2509.24527): offline imagination training. Tokenizer = masked autoencoding (MSE + 0.2 LPIPS), frozen for
  dynamics. BC policy and reward heads on agent tokens with 8-step multi-token prediction; agent tokens attend to all,
  nothing attends back ("crucial for avoiding causal confusion"); transformer frozen during RL; frozen BC copy as prior.
- DINO-WM (2411.04983): frozen DINOv2 patch targets, teacher forcing, no rollout loss; patch >> CLS (PushT 0.90 vs 0.44).
- Reconstruction-free, jointly trained world models that train an actor in imagination are all ONLINE (Dreamer-CDP,
  JEDI, NE-Dreamer, MuDreamer, Contrastive WM 2609.22175). Offline + joint + reconstruction-free EXISTS for planning:
  LeWM, PLDM (corrected 10-24 after Codex; my first version said no offline precedent). Offline imagination-trained
  actors use a reconstruction tokenizer (Dreamer 4). Missing precedent = the full combination (offline, joint,
  reconstruction-free, imagination-trained actor, stochastic sparse-reward game). JEDI frames the JEPA loss as an
  information bottleneck that penalises I(X; Z).
- MuDreamer (action + value prediction, no reconstruction) 7.3 on Crafter vs Dreamer 14.5; Dreamer-CDP blames "the
  comparatively weak action signal in Crafter".
- Object discovery without labels: VideoSAUR / DINOSAUR reconstruct frozen pretrained features; OC-STORM uses a
  pretrained segmenter + annotated frames. No label-free, fixed-target-free method found.
- Behaviour as a representation signal: BPR (2211.00863, BC features for offline RL); ACRO (2211.00164, multi-step
  inverse models drop exogenous noise when the data policy ignores it); copycat problem (Wen 2020, 2010.14876): with
  observation histories, BC predicts the previous action instead of the next; adversarial fix weak on images.
- AD-WM (2609.30264): predictor-level action-recovery regularisation; factual error does not track closed-loop success.

## Decision A: loss currency and ownership reading (2026-10-24)
- Currency. HarmonyDream (2310.00344, ICML 2024): observation loss dominates reward loss in world models; learned
  per-loss scale L_i/sigma_i + log(1 + sigma_i) equalises them (E[L/sigma*] = 1); big gains on Atari100k / Meta-World.
  Kurin et al. (2201.04122, NeurIPS 2022): no multi-task optimiser (PCGrad, MGDA, ...) consistently beats a weighted
  sum with standard regularisation. VaGraM (2204.01464): value-gradient-weighted model loss; couples model and value
  (dependency broken early in training). DreamerV3: every term a likelihood + fixed weights across domains.
- What self-prediction keeps. Klindt, LeCun, Balestriero (2605.26379): LeJEPA's optimum = slowest features; with
  fewer dims than latents the kept subspace is undetermined; policy data breaks identifiability. Sobal et al.
  (2211.10831, offline): JEPA (VICReg / SimCLR) is satisfied by FIXED background noise or a stationary dot and ignores
  the moving dot. 3-dot RMSE (action / random / stationary): VICReg .277/.273/.066, IDM .035/.298/.272, recon
  .021/.182/.026, supervised .010/.180/.005. Proposed remedy: hierarchy or a constraint against temporally constant
  representations (PLDM's time-variance term). Feature suppression (2011.02803): the dominant object suppresses
  smaller ones; Robinson 2106.11230: InfoNCE alone permits suppression shortcuts.
- Contrastive, local, temporal. ST-DIM (1906.08226, offline random-agent Atari): small-object F1 VAE .18, pixel
  prediction .31, CPC .42, ST-DIM .51, supervised .66; patch-level InfoNCE also resists easy features saturating the
  objective (Boxing clock). TWISTER (2503.04416): action-conditioned CPC over K = 10 future steps, negatives = the
  other B x T - 1 batch samples (other times of the same sequence included); removing action conditioning: median
  77 -> 42; keeps reconstruction too. TCN (1704.06888): repelling temporal neighbours makes embeddings encode what
  changes over time. Principle: a contrastive objective keeps what distinguishes positives from its negatives.
- Sparse consequence signals. Lyle et al. (2102.13089): sparse reward lets the predictor sit at ~0 with ~0 weights,
  starving the features (collapse risk); random cumulants help. UNREAL: rewarding / non-rewarding histories shown in
  equal proportion. Ours: event-based rebalancing installs shortcuts (08-29 terminal tails -> action prior; E14
  positives-only -> 24-30% hallucination); a faced-tile mask on EVERY attempt (matching negatives) caught 0.54-0.65.
- Hard attention. Saccader (1908.07644): hard attention is hard to train from sparse signals; pretrained location
  proposals fix it.
- Ownership summary (reasoning on the above; owner = an incentive that survives shortcuts): persisting terrain ->
  self-prediction (over-owns: it can satisfy the objective alone); player-centric dynamics -> inverse dynamics (on
  per-frame encodings, or action history gives a copycat shortcut); faced-cell consequences -> fovea on attempts +
  reward / inventory; adjacent entities -> damage / termination heads (context-rebalanced, action-conditioned);
  entities 2+ cells -> weak: longer-horizon consequences, fovea (from-state target + exploration floor), local
  contrastive temporal term; cows weakest; food / drink / energy -> HUD pixels via the fovea; night static -> nobody.

## Self-prediction objective: families and interactions (2026-10-24)
- Regression to own targets. Tang et al. (2212.03319): stop-gradient self-prediction's critical points span
  eigenvectors of P^pi and drift to the dominant (slowest-mixing) ones; needs a faster predictor + semi-gradient.
  SPR / BYOL / EfficientZero / TD-MPC2 / MLR (2201.12096, masked space-time cubes, momentum targets) / SGF (2506.02612,
  VICReg + temporal consistency + augmentation, online Atari). Klindt / Sobal / Littwin: favours slow, predictable
  content; can be satisfied by fixed background.
- Contrastive (InfoNCE / hinge). CPC, ST-DIM, TWISTER, C-SWM (1911.12247: TransE-style hinge with random buffer
  negatives; "limited to fully deterministic worlds", Markov). InfoNCE estimates a density ratio, so stochastic
  positives are not averaged. What is kept = what distinguishes positives from negatives (Codex: within-sequence
  negatives can still be told apart by camera, player position, lighting, HUD; feature suppression persists).
- Correspondence targets. VideoSAUR (2306.04829): predict the softmax-normalised affinity "where did each patch go
  k steps later" (frozen DINO features); beats GT optical flow as a target, robust to camera motion and static
  objects. Contrastive random walk (Jabri et al. 2020) learns tracking from cycle consistency.
- Distributional latent prediction. Dreamer KL with categorical latents; Stochastic MuZero chance codes; VJEPA
  (2601.14354, theory + toy); MoP-JEPA mixtures; JEDI diffusion.
- Structural separation. Iso-Dream (2205.13817): controllable (action-conditioned + inverse dynamics), uncontrollable
  (action-free) and static branches; reconstruction-grounded. Denoised MDPs (2206.15477): keep controllable AND
  reward-relevant only (too narrow for a task-agnostic world). PLDM time-variance term against temporally constant
  features (no ablation reported). Sobal: hierarchy as a remedy.
- Long-horizon / geometry. Contrastive RL (2206.07568, offline, pixels), FB / successor measures, CTWM (2610.01373:
  isotropic regularisers such as SIGReg distort commute-time geometry; log-det regulariser + displacement predictor).
  ACPC (2608.12939): bisimulation-based diagnostic for JEPA world models (invariance radius / separation rate).
- Interactions. Bisimulation objectives show embedding pathologies under sparse reward (2110.14096). Voelcker et al.
  (2406.17718), Kurin, Yang & Nachum: rankings of auxiliary losses change across benchmarks and distractor types.
- Key structural insight: working systems split roles. A PERCEPTION objective decides what the state holds
  (DreamerV3 reconstruction; TWISTER reconstruction + AC-CPC; Dreamer-CDP CDP + reward). A DYNAMICS objective predicts
  the next perception state distributionally (KL / chance) with LIMITED gradient back into perception: DreamerV3
  loss_scales dyn 1.0 vs rep 0.1, free_nats 1.0 (verified in third_party/sources/danijar__dreamerv3); Dreamer-CDP
  encoder lr 67x below the RSSM; Tang's faster predictor. Our D / IC / PI / LeWM worlds used one deterministic
  regression for both roles.
- Correspondence / tracks as targets: ATM (2401.00025, RSS 2024): forecasting future point tracks from video guides
  policies (130+ tasks; argues tracks carry object-permanence bias). Contrastive random walk (Jabri et al. 2020,
  2006.14613): correspondence from cycle consistency (self-referential, not pixel-grounded).
- Locality in temporal contrast: ATC (2009.08319): global temporal contrast alone matched end-to-end RL in 5/8 Atari
  games but suffered in Breakout and Space Invaders (small moving objects); cf. ST-DIM's local advantage.
- Foveation: Williams (2505.01249, Neural Computation 2025): a fixation = linear downsampling of ONE high-resolution
  latent scene (full resolution at the fovea, lower in the periphery); fixations fused by Bayesian inference; where to
  look = expected information gain. Two-stream foveation (2403.15977): low-res dorsal stream picks fixations, high-res
  ventral stream classifies. No foveated world model for imagination found (niche still open).
- Correspondence beyond Craftax: self-supervised driving splits motion into rigid flow (ego-motion + depth) plus
  residual flow for independently moving objects (GeoNet 1803.02276; Competitive Collaboration, Ranjan et al.;
  point-cloud version 2009.10467). Flow / tracks as world-model currency in manipulation: FLIP (2412.08261, ICLR
  2025, flow from off-the-shelf trackers), 3DFlowAction (2506.06199), FlowWAM (2607.13017), ATM (CoTracker labels).
- Peripheral vision: Rosenholtz's texture tiling model: the periphery = rich local summary statistics over pooling
  regions growing with eccentricity (features kept, arrangement jumbled); crowding, not acuity, is the main
  difference. Herwig & Schneider 2014 (JEP:General 143:1903): the visual system learns transsaccadic associations
  and predicts foveal appearance from peripheral input; altered foveal input shifts peripheral recognition.
- Our disappearance history, by level: perception (learned representations): LeWM CLS z erased the zombie
  (0.947 -> 0.528, 09-26; the edited cell's patch token kept it), R2-Dreamer RSSM posterior mob recall 0 (R44); under
  joint training the PREDICTED next states lost relative linear mob decodability in 1k-4k updates (R63, qualified by R64:
  neither survival nor destruction of information is shown), while the same predictor on frozen Raw did not. Imagination: median erases / mean superimposes (10-10),
  soft-blend fading (R21), D's zombies vanish (adjacency 0.02 at d16, R53), E22M fades x1/3, sleep-onset erasure
  (R46), PI draws no arrivals (R24). Correspondence-as-transport (IC / PI) fixed coherence (valid 94 vs 63%, census x7)
  but not decisions (R50).

## Dynamics research pass (2026-10-24)
- Global stochastic latent per step: DreamerV2 (2010.02193): categorical latents beat Gaussian on 42/55 Atari games
  ("a mixture of categoricals is again a categorical"); KL balancing (prior 0.8 / posterior 0.2) "to avoid
  regularizing the representations toward a poorly trained prior". Delta-IRIS (2406.19320, ICML 2024): encoder and
  decoder conditioned on past frames + actions, so a few discrete tokens per frame carry only "the stochastic
  delta"; Crafter return 16.1 @10M vs DreamerV3 XL 15.1, 10x faster than IRIS; same token count every step
  (stated limitation); reconstruction-based. Genie: global VQ latent action from consecutive frames.
- Per-token: Dedieu (per-patch categorical, independent; wrong counts); ITC (2605.16457): decoding-time copy-or-
  generate assignment with a displacement cap + wildcards; Craftax return 68.6 -> 71.9 from 3D RoPE, +0.6 from ITC;
  transitions with a random creature fully correct 38% (65% without). EMERALD (2507.04075): spatial latent state +
  MaskGIT refinement; reports beating human experts on Crafter at 10M; online, reconstruction.
- Per-entity: LPWM (2603.04553, ICLR 2026): per-particle continuous latent actions (posterior from consecutive
  frames, learned prior, per-particle KL); argues a global latent action "cannot naturally disentangle" independent
  local dynamics; reconstruction (DLP). Structured World Belief (2107.08577): SMC particles, each a set of object
  files; presence in the belief decoupled from visibility, so unseen objects are not deleted (prior models deleted
  them because invisible objects add no reconstruction error). SlotSSMs (2406.12272): per-slot SSM transitions +
  sparse cross-slot self-attention.
- Afterstate + chance: Stochastic MuZero: fixed one-hot codebook (VQ variant, no decoder), chance encoder on the
  real outcome, prior sigma trained toward it, commitment term (via 2212.11087 / LightZero; paper PDF not reachable).

## Dynamics pass 2 (2026-10-25): ground truth, brain, tracking theory, history
- Craftax source (game_logic.py): zombies: within Manhattan < 10, w.p. 0.75 step toward the player on the larger axis
  (random on ties), else a uniform random step; attack iff adjacent and hidden cooldown <= 0 (reset 5; damage 7 asleep /
  2 awake, wakes the player); cows: 8-way choice with 50% stay; skeletons: approach if >= 10, flee if <= 3, else random,
  w.p. 0.85 override by random, shoot at 4-5 or when blocked while fleeing; despawn DETERMINISTIC at distance >= 14;
  mobs updated sequentially, each blocked by cells already occupied (the only entity-entity coupling). Spawns: cow
  w.p. 0.1 on grass at distance 4-13 (so INSIDE the view possible); zombie w.p. 0.02 + 0.1 (1 - light)^2 on grass /
  path at distance 10-13 (always outside the view: zombies only walk in). Caps: 3 zombies, 3 cows, 2 skeletons, 3 arrows.
- Brain / cognition: Battaglia, Hamrick & Tenenbaum (PNAS 2013): an intuitive physics engine of approximate
  probabilistic simulation over objects; noise on perceived positions fit humans (and the world less); heuristics worse.
  Ullman et al. (TICS 2017): mental game engines, approximate and fast. Kay et al. (Cell 2020): the hippocampus
  alternates between possible futures, one coherent scenario per ~125 ms theta cycle. PLATO (Piloto et al., Nat Hum
  Behav 2022): learning continuity / solidity / permanence depended on object-level representations; flat models at
  least as large failed some tests (mask provenance unchecked). Loci-Looped (2310.10372, ICML 2024): per-object outer
  loop (observe) + inner loop (imagine hidden objects), learned fusion; learns permanence, inertia, solidity from video.
  GateL0RD (2110.15949, NeurIPS 2021): LSTM / GRU gates leak and wash out information; a gate + L0 penalty on latent
  CHANGES gives sparsely changing states, better 1-step -> N-step generalisation; tied to event segmentation theory.
- Tracking theory: FastSLAM / Rao-Blackwellised particle filters: given the agent's path, landmark estimates decouple
  into independent filters; particles over path (+ data association) carry the correlations; better than the EKF under
  ambiguous association. Labelled multi-Bernoulli (Reuter, Vo, Vo, Dietmayer 2014): each track = label + existence
  probability + spatial density; births / deaths built in. Propagation + discovery: SQAIR, SCALOR (1910.02384:
  parallel proposal-rejection, background module), STOVE (1910.02425); TrackFormer / MOTR: track queries propagate
  identities, detect queries add new objects.
- Prior usability: PlaNet (1811.04551): latent overshooting (multi-step priors -> posteriors); the RSSM agent did not need
  it; deterministic + stochastic state crucial. Koopman Dreamer (2607.19719): EMA teacher targets, multi-step teacher and
  prior rollouts, spectrally bounded deterministic core (proprioceptive / LiDAR tasks).
- Imperfect information: determinization / PIMC suffers strategy fusion (cannot value information gathering) and
  non-locality; mitigations reason over information sets (ISMCTS; 2408.02380, 2006.08740).
- Our dynamics history (limits: per-token worlds on frozen Raw tokens, screen-anchored, mostly single seed, proxies not
  actor outcomes): stochastic channels go where the loss mass is (E22FJ 92% to entering terrain; E22Z: loss share is
  the cause); a warm decoder ignores a new noise channel (Park & Lee 2108.02446); a global mean-pool posterior fails
  (S4); free per-tile creation makes content appear / vanish from own noise (S5 / S6); sampled zombies multiply x10 by
  d16 (generated, not copied); true posterior codes catch 49% of damage, the learned prior 17%, and the prior overfits
  with training (0.470 -> 0.405); one faithful future = 57% of H16 decision value, 16 samples 90%.

## Discovery, belief interface, prior usability (2026-10-25)
- Object discovery: Bao et al. (CVPR 2022, Discovering Objects that Can Move): a weak motion-segmentation signal on a
  subset of slots generalises to "both moving and static instances of dynamic objects" (KITTI). SAVi (ICLR 2022): slots
  trained to PREDICT OPTICAL FLOW track objects; first-frame hints (object centres) needed for strong segmentation.
  MOREL (NeurIPS 2018): SfM-Net motion segmentation of moving objects improved Atari RL sample efficiency. AdaSlot (CVPR
  2024): slot count chosen per input (discrete sampling + masked decoder). Codex: PLATO used ground-truth segmentation;
  Loci-Looped was given the scene background (neither solves label-free separation).
- Code leakage: Nikulin et al. (2502.00379, ICML 2025): latent actions inferred from consecutive frames absorb
  distractor changes; LAOM improves probes 8x but not to the distractor-free level; 2.5% action labels gave 4.2x.
- World vs belief: POMCP (Silver & Veness 2010) samples hidden states but acts through action-observation histories
  (Codex). Baisero & Amato (2105.11674, AAMAS 2022): state-only critics are biased under partial observability; a
  history-state critic V(h, s) is unbiased with a history-based policy. Church (Goodman et al., UAI 2008): memoised
  random functions have random-world semantics (a value drawn on first use, then looked up); DPmem stochastically
  reuses a cached draw or makes a new one.
- Corrections accepted (Codex, 10-25): lazy revelation is an implementation choice that must preserve the joint
  distribution (sample on first use by ANY process, e.g. an unseen zombie testing an unseen cell), not the necessary
  fix; the necessary condition is that the actor uses only history-available evidence. FastSLAM's independence is
  model-specific: in Craftax, primitive random draws are independent but occupancy, shared uncertain terrain and
  exclusive identity couple the beliefs. A1's tracked-offset R2 is dominated by position persistence and is not
  comparable with binary cell occupancy; withdrawn as evidence. Plain fact kept: visible tracked entities move in 54%
  (zombies) / 38% (cows) of steps.
- Object-centric world models: G-SWM (2010.02054, ICML 2020): propagation + discovery (one discovery latent per grid
  cell, top-K presence kept from the union), local "attention on environment" for situated dynamics, depth for occlusion,
  hierarchical per-object latent for multimodal futures; the posterior of the attributes is REPLACED by the prior's
  generator given the behaviour latent, because otherwise "xt provides all necessary information" and the model ignores
  the behaviour latent (a bypass); discovery uses a fixed prior and runs "only during inference" (imagined scenes get no
  new objects; "temporal discovery" is future work). SOLD (2410.08822, ICML 2025): SAVi slots pretrained by
  reconstruction on ~1M random frames, then fine-tuned jointly; deterministic transformer dynamics (stated weakness);
  fixed slot count per environment; beats DreamerV3 / TD-MPC2 on relational manipulation; entering / leaving objects
  not discussed. VQ planning (Ozair et al. 2106.04615): code k(t+1) = enc(s(1:t+1), a(1:t)) = "the additional
  information" given the past; prior trained by cross-entropy on the encoder's codes; no explicit mechanism keeps
  deterministic information out of the code.
- Brain: object files / FINSTs (Pylyshyn; Kahneman, Treisman & Gibbs 1992): content-free indexes that open a file;
  infants individuate by spatiotemporal continuity first, by kind later (Xu & Carey 1996; Carey & Xu 2001: two systems).
  Scholl & Pylyshyn (Cogn Psych 1999): tracking survives full occlusion when accretion / deletion at fixed contours
  implies an occluder; items that vanish without occlusion cues break tracking. Franconeri, Pylyshyn & Scholl (2012):
  pre- and post-occlusion views linked by spatial proximity.
- Tracking theory: PMBM (Garcia-Fernandez, Williams, Granstrom & Svensson, IEEE TAES 2018, 1703.04264): never-detected
  targets = a Poisson point process intensity (PHD-style prediction, lowered where the sensor looked and saw nothing);
  detected targets = a multi-Bernoulli mixture; delta-GLMB is a special case.
- Craftax-Classic source facts: per-mob hidden state = health and attack_cooldown (Mobs dataclass; only the player's
  health is rendered); skeleton cooldown resets to 4, zombie to 5. Growing plants: up to 10, ripen deterministically at
  age 600, age reset when eaten. Night: fresh per-pixel uniform static each frame (state_rng), intensity
  2 (0.5 - light) times a noise-intensity texture, only when light < 0.5.

## How real observations enter the state (2026-10-25, code + papers)
- Bottleneck rationale: PlaNet (1811.04551): "all information about the observations must pass through the sampling step
  of the encoder to avoid a deterministic shortcut from inputs to reconstructions"; purely deterministic transitions
  make "it easy for the planner to exploit inaccuracies"; without the stochastic part "the agent does not learn".
  World Models (1803.10122): the controller exploited M ("we are also giving the controller access to all of the hidden
  states of M"); MDN sampling with temperature trades realism against exploitability.
- History-conditioned posterior q(z | h, o): DreamerV3 code (rssm.py _observe: obs layers read concat(deter, tokens);
  flag `absolute` drops deter, default False); NE-Dreamer and R2-Dreamer keep it. Per-frame posterior q(z | o): STORM /
  DRAMA (Mamba; code: post_head(encoder(obs)), decoder reconstructs from z alone, sequence model reads z, actor reads
  concat(z, h)), TWM, TransDreamer, R2I. R2I App. M, inside DreamerV3: q(z | o) "retains or elevates performance" on
  Memory Maze, "zero influence" on Atari 26 / DMC-proprio; "an inductive bias that favors memory"; needed for parallel
  scan. TransDreamer: Dreamer with q(z | x) performs similarly.
- Policy input: TWM reads z only ([z, h] lowered scores; h's space shifts during training); R2I: pi(z, h) broke on memory
  tasks, pi(z, x) worked (non-stationarity, online training). IRIS: actor reads reconstructed frames, and "during
  experience collection in the real environment, frames still go through the autoencoder to keep the input distribution
  of the policy unchanged". Delta-IRIS: encoder conditioned on past frames + actions, about 4 tokens per Crafter frame
  carry "the stochastic delta"; random tokens keep layout / movement / crafting right but cows "appear and disappear
  incoherently"; policy reads reconstructed frames. Dreamer 4: tokenizer "causal in time", frozen for dynamics; context
  latents corrupted to tau_ctx = 0.1 "to make the model robust to small imperfections in its generations"; policy /
  reward heads on agent-token outputs (agent tokens attend to the world, not back).
- No stochastic bottleneck: TD-MPC2 (code): pi trained on latents rolled out by the dynamics from encode(obs[0]) along
  real actions (horizon 3) with a consistency loss to encode(obs[t]); MuZero re-encodes observations at each real step.
  Stochastic MuZero (LightZero code): chance encoder reads concat(obs_t, obs_t+1); used in training unrolls and search.
- Objects: SAVi predictor -> corrector (slot attention on the frame, initialised by the predictor). G-SWM separate
  posterior object-state RNN. OP3 (1910.12827): posterior by iterative refinement of the dynamics' prediction using
  reconstruction gradients; "a single forward pass ... is insufficient to break the symmetry". VRNN: h_t = f(phi(x_t),
  phi(z_t), h_t-1), so generation feeds generated x back. KVAE: per-frame recognition a_t from x_t + Kalman over a.
- Brain: imagination relies on mechanisms similar to perception, which makes reality monitoring necessary (Dijkstra, Kok
  & Fleming, Neurosci Biobehav Rev 2022); predictive coding (Rao & Ballard 1999): feedforward signals carry prediction
  errors, feedback carries predictions.
- A2 `eda_association.py` (A1 rollouts, 1.8M transitions; true positions, so perception errors excluded): matching each
  visible entity to the nearest same-kind entity of the previous frame: identity unique by gating in 86% (zombies) /
  94% (cows) / 95% (skeletons) of continuing visible transitions; min-distance assignment resolves most of the rest;
  unresolved (wrong or tied) 0.36 / 0.24 / 0.35% in world coordinates vs 2.0 / 1.2 / 1.2% in screen coordinates;
  entries landing where another entity left 0.22-0.24%. Zombie attacks (10,841): one adjacent zombie 95.1%, two 4.8%,
  three 0.06%; attacker visible for a median 13 steps before attacking (p10 5; 1.9% within 3 steps); 7.3% while asleep.
- Craftax source: in-view disappearances are kills (mask = health > 0) or arrows stopping (wall, mob, player);
  despawn only at distance >= 14 (mob_despawn_distance), always outside the 9x7 view; no occlusion in the top-down view.

## Route 1 trainability, out-of-view beliefs, imagined fixations, contextual memory (2026-10-25)
- Compounding error. GameNGen (2408.14837): teacher forcing then autoregressive sampling "leads to error accumulation";
  corrupting encoded context frames with Gaussian noise of sampled level (level given as input) "allows the network to
  correct information sampled in previous frames"; without it quality "degrades quickly after 10-20 frames". Self
  Forcing (2506.08009): exposure bias; noisy context at inference "does not fundamentally resolve" it; unroll the
  model on its own outputs in training with holistic distribution-matching losses (DMD / SiD / GAN), a post-training
  stage with gradient truncation. SlotFormer (2210.05861): trained on its own predicted slots "simulating the
  autoregressive generation process at test time"; consistent object attributes over long rollouts; clips with newly
  entering objects filtered out of the rollout evaluation; vanilla SAVi misses entering objects (empty-slot symmetry).
  MuZero / Stochastic MuZero / EfficientZero / TD-MPC2: unroll the latent along real actions (and posterior chance
  codes) with losses to real targets / encodings. PlaNet latent overshooting; Gregor et al. (1906.09237): overshooting
  (4-32) plus an expressive generative model "critical for stable" belief states. MBPO (1906.08253): short rollouts
  branched from real states. Offline: MOReL (2005.05951) HALT state for unknown state-actions (ensemble disagreement);
  MOPO (2005.13239) reward penalised by model uncertainty. Asadi et al. (1804.07193): multi-step Wasserstein error of
  Lipschitz models; Lambert et al. (2012.09156): trajectory-based models predict long horizons more stably.
- Memory in world models: Long-Context SSM video world models (2505.20171, Mamba block-wise scan + local attention;
  Memory Maze, Minecraft; random clean-prefix training to force long dependencies). WorldMem (2504.12369): memory bank
  of previously GENERATED latent tokens with states (pose, timestamp), state-aware attention; consistency on revisits.
- Out-of-view beliefs. Brain: social place cells (Omer et al. Science 2018, bats; Danjo et al. 2018, rats) code another
  animal's position allocentrically in CA1; object-vector cells (Hoydal et al. Nature 2019) code distance / direction to
  objects; object-trace cells (Tsao, Moser & Moser 2013) fire where objects used to be (LEC also codes elapsed time);
  posterior parietal neurons are more active when a vanished stimulus is inferred to be moving (Assad & Maunsell, Nature
  1995); human multiple-object tracking matches a Rao-Blackwellised particle filter ideal observer (Vul et al. NIPS
  2009); uncertainty codes debated: probabilistic population codes vs sampling vs distributed distributional codes
  (2409.02709; a sample code yields a PPC in the many-sample limit). Adjacent fields: Deep Tracking (1602.00991): a
  spatial recurrent belief trained to predict FUTURE observations with inputs dropped out (otherwise it "would only
  learn to copy"); PF-RNN (1905.12885): latent particle sets with soft resampling; ideal searcher (Najemnik & Geisler,
  Nature 2005): posterior map over locations updated with visibility-weighted evidence, including target-absent looks;
  PMBM undetected intensity; DDC belief states (Vertes & Sahani, NeurIPS 2019).
- Imagined fixations: no imagination-trained actor with learned foveation found. Principle from active inference /
  Bayesian experimental design (Friston et al. 2012; Daucé 2018; Parr et al. 2021) and POMCP: candidate observations
  are generated from SAMPLED hidden causes, so a look can resolve uncertainty; video world models with camera control
  generate unseen views once and retrieve them on revisits (WorldMem).
- Contextual memory: hippocampal indexing theory (Teyler & DiScenna 1986; Teyler & Rudy 2007): the trace indexes
  co-occurring neocortical activity; constructive episodic simulation (Schacter & Addis 2007): retrieval recombines
  details for imagined events; transformation hypothesis (Winocur & Moscovitch 2011): context-bound episodic traces
  coexist with gist / schema versions, used by retrieval demands.
- A3 `eda_outofview_belief.py` (200 replayed TRAIN episodes, replay exact; perfect-observer belief by 2,048-particle
  simulation of the game's own mob rules from the last sighting along the real player path, conditioned on not being
  seen; zombies 9,403 out-of-view steps, cows 25,071; the true cell fell outside the simulated support in 0.2%):
  effective cells (median) zombies 1.6 at 1 step -> 6.1 at 9-16 steps, cows 2.0 -> 22; separated supports zombies 0% ->
  7% (3.4% overall), cows <= 0.9% (0.5%); share of the predicted mass removed by not seeing it: 45% / 52% at the first
  step, 2-12% per step after. Predictive cost vs the exact belief (nats per step, true position): one moment-matched
  Gaussian 0.52 (zombies) / 0.23 (cows); Gaussian truncated by the view 0.46 / 0.17; 16 samples with a one-cell kernel
  0.57 / 0.58; 4 samples 1.18 / 1.99. Re-entries: zombies median 3 steps (p90 10), exact belief gives the true re-entry
  cell log-lik -1.13 and 60% of its mass is in view at the re-entry step; cows median 4 (p90 12), -2.95, 28%. Reading:
  multiple hypotheses are rarely needed for topology; truncation by the view dominates the first steps; the chase rule
  gives zombie beliefs non-elliptical shapes a Gaussian smooths away; diffuse cow beliefs suit a Gaussian.

## Fovea roles, rollout objectives, learning signals (2026-10-25, later)
- Gaze: Sprague & Ballard (NIPS 2003): eye movements reduce uncertainty about task-relevant variables; each candidate
  scored by "the expected cost of the uncertainty that will result if the movement is not made" (beats round robin);
  Hayhoe & Ballard (TICS 2005): gaze directed by task goals rather than image salience. Noisy TV: raw prediction-error
  curiosity gets stuck on unpredictable stimuli (Burda et al. ICLR 2019). Measured earlier (consequence_locality.py):
  99.8% of immediate consequence cells are the player's 4 neighbours + HUD; 2 glimpses cover all in 92-96% of event steps.
- Proper scoring of sampled rollouts: NeuralGCM (2311.07222): stochastic model trained with CRPS on rollouts from
  6 h to 5 days (gradual increase "critical"), two members per forecast give an unbiased CRPS. AIFS-CRPS (2412.15832):
  almost-fair CRPS, noise-injected members. Energy / kernel scores are the multivariate strictly proper versions
  (Gneiting & Raftery JASA 2007); distributional diffusion with scoring rules (2502.02483).
- Actions as evidence: Ortega et al. (2110.10819): conditioning on actions instead of treating them as interventions
  produces "self-delusion". Our corpus: the expert acts on Craftax-Classic-Symbolic observations (collect_support.py),
  not on our pixels; epsilon = 1.0 episodes (uniform random actions, exogenous) = 2,003 train episodes, 285,705
  transitions (eps 0.1 / 0.25 / 0.5: 1.36M / 550k / 363k); the expert network is available, so per-step noise
  propensities are computable for eps < 1.
- Generative discovery: SQAIR (1806.01794): a LEARNED discovery prior "samples latent variables for new objects that
  enter the frame" during generation; SCALOR sets its discovery prior to zero for conditional generation.
- Learning signals: TEM (Whittington et al. Cell 2020): trained by "predicting the next sensory observation"; path
  integration from the previous location code given the action, refined by memory retrieval via a Bayesian update.
  JEPA implicit bias toward high-influence (predictive) features over noisy ones (Littwin et al. NeurIPS 2024, deep
  linear models).
- A4 `eda_correspondence.py` (120 replayed TRAIN episodes, stride 2; plain block matching of stored frames, simulator
  state for evaluation only): global camera shift recovered 99.3% (still) / 99.7% (moved) by day, 98.0% / 94.4% at night
  (13% ambiguous when moving at night). Entity screen displacement by patch matching in a +-2 window: day 89% (still) /
  86% (moved), night 76% / 72%; arrows 78% / 66%; 15-22% of entity matches ambiguous. Terrain cells: the true shift is
  the selected best match in 17% (day; 93% of cells have a near-tie) / 27% (night) [correction, Codex: argmin correct,
  not "uniquely correct"]. Reading: self-motion is cleanly recoverable from pixels globally; per-cell terrain
  correspondence is ill-posed (aperture) but follows from the shared shift; entity correspondence from raw matching is
  a noisy label (1 in 7 wrong by day, 1 in 4 at night).

## Flow parsing, births, confounding, uncertainty (2026-10-25, evening)
- Flow parsing (Rushton & Warren 2005; Warren & Rushton 2009): the visual system estimates the self-motion component of
  retinal flow, globally discounts it, and attributes the remainder to object movement (described as qualitative).
  Robotics: Competitive Collaboration (Ranjan et al. CVPR 2019) jointly learns depth, camera motion, flow and motion
  segmentation without labels; a moderator assigns pixels to static vs independently moving. Infants perceive object
  unity from common motion, not from static Gestalt cues (Kellman & Spelke 1983).
- Reconstruction-free object world models: C-JEPA (2602.11389) masks object slots from a FROZEN encoder (VideoSAUR on
  frozen DINOv2); reports SlotFormer's CLEVRER accuracy falls 79.4% -> 44.9% without reconstruction; joint encoder
  refinement "without representational collapse" left to future work.
- Uncertainty: PETS (Chua et al. 2018): each probabilistic network = aleatoric, disagreement across the ensemble =
  epistemic. Depeweg et al. (ICML 2018) decompose predictive uncertainty into the two. Yu & Dayan (Neuron 2005):
  acetylcholine signals expected uncertainty (known unreliability within a context), norepinephrine unexpected
  uncertainty (unsignalled context switches).
- A5 `eda_change_residual.py` (150 replayed TRAIN episodes, stride 3): pixel residual after aligning consecutive frames
  by the label-free estimated shift, per cell, vs simulator labels (entity arrived / left / appeared, block edited):
  day AUC 0.978, recall 99.6% at 5% false positives; night AUC 0.87, recall 45-55% at 5% FP (pooling barely helps);
  estimated vs true shift identical. Clean grounded change target by day, noisy at night.
- A6 `eda_births.py` (A1 rollouts): first sightings: zombies 24,480 at the view edge, 1,574 just inside it, 1,337
  appearing in view; cows 27,790 edge, 11,373 in view (28%); skeletons 14,758 edge, 421 in view; arrows 94% appear in
  view (shot by skeletons). Source + data: despawned mobs keep a stale position that update_mobs keeps moving (zombies
  keep chasing), and do_action applies mask = health > 0 to every slot whenever the action is DO, so any DO revives all
  despawned-but-healthy mobs where their ghost now is. Births after DO with continuous position: zombies 23,796
  revivals (1,337 in view) vs 59,080 spawns (all out of view); cows 21,427 revivals vs 88,379 spawns (10,186 in view);
  skeletons 24,573 revivals vs 43,098 spawns. Every in-view zombie / skeleton appearance in the data is a revival.
  Correction to the 10-25 source note: zombies never SPAWN in view, but they can REAPPEAR in view after a DO.
- Corrections (Codex): SlotFormer 79.4 -> 44.9% is CLEVRER question-answering accuracy with pretrained SAVi slots, not
  evidence of encoder erasure; presence AP does not measure how much privileged information changes the expert's
  choices. A5 marks appearance change (entity occupancy changes or terrain edits), not boundaries, identity or
  ownership; terrain edits belong to the store route.
- A8 `eda_exogenous_steps.py`: the collector's random-branch flags regenerated from the seed tree reproduce all 740,693
  stored random actions. Random steps: eps 0.1 135,514; 0.25 138,081; 0.5 181,393; 1.0 285,705 (all). Start positions
  with k consecutive random actions, k = 4 / 8 / 16: eps 0.1 158 / 0 / 0; 0.25 2,155 / 16 / 0; 0.5 22,202 / 1,242 / 7;
  1.0 279,696 / 271,684 / 255,660. Multi-step exogenous windows exist almost only in random-play episodes; one-step
  exogenous steps inside expert-mixed episodes: 455k.
- A9 `eda_change_night.py` (120 episodes; held-out episodes for evaluation; simulator labels for evaluation only):
  night independent-change detection, AUC / recall at 5% false positives: raw residual 0.89 / 51%; detector trained on
  clean day pairs with label-free (Otsu) day targets 0.91 / 58%; detector trained on GENERICALLY corrupted day pairs
  (darken, desaturate, uniform grey noise, independent per frame) with the same label-free targets 0.94 / 77%; ceiling
  (trained on real night pairs with simulator labels) 0.99 / 94%. Day: all 0.97-0.99 / 97-100%. Night evidence exists
  in the pixels; a label-free recipe recovers a large part of it.
- "Doesn't know" reading: Lu et al. (ICLR 2022, 2110.04135): LOO-KL and log-likelihood-variance penalties correlate
  very weakly with true model error; ensemble std is better; "uncertainty is more correlated with dynamics error than
  distribution shift ... can in fact assign high certainty to data far away from the offline data"; "further work is
  needed" to detect the discrepancies that arise in offline MBRL. LOMPO (2012.11547): offline visual MBRL with a
  latent-dynamics ensemble, penalty = variance of ensemble log-likelihoods. Nalisnick et al. (ICLR 2019): deep
  generative models can give out-of-distribution data HIGHER likelihood. Randomized prior functions (Osband et al.
  2018) and epinets (2107.08924, NeurIPS 2023): fixed prior networks are "crucial" for ensemble diversity; joint
  predictions. Gershman, Blei & Niv (Psych Rev 2010): inferring a NEW latent cause explains context effects; hippocampal
  lesions restrict the capacity to infer new causes.

## Sketch v2 inputs (2026-10-25, night)
- Craftax-Classic map writes (game_logic.py): faced-cell edits by the agent (mine, eat plant, place), plants ripening
  at age 600 (update_plants), arrows turning a crafting table / furnace into path (update_mobs). Nothing else.
- A9b `eda_change_night_v2.py` (observation-derived example selection; held-out episodes): average precision / precision
  and recall at a LABEL-FREE operating point / calibration error. Day: raw residual AP 0.24, Otsu threshold flags
  4.2% of cells at precision 0.40 (recall 0.98); corrupted-day detector AP 0.94, p > 0.5 precision 0.85 recall 0.92,
  ECE 0.005. Night (base rate 2.9%): raw residual AP 0.18, Otsu flags 41% at precision 0.07; clean-day detector AP 0.18,
  ECE 0.34; corrupted-day detector AP 0.61, precision 0.63 recall 0.51, ECE 0.016; simulator-label reference AP 0.93.
  Reading: thresholded residuals are poor targets even by day; the corrupted-day detector's soft probability is a
  calibrated target; night supervision stays partial.
- Recent: LeJEPA (2511.08544): SIGReg (random 1-D projections, characteristic-function matching to an isotropic
  Gaussian) replaces stop-gradients / teachers. LeWorldModel (2603.19312): end-to-end JEPA world model from pixels with
  a next-embedding loss + a Gaussianity regulariser. DLPWM (2511.06136): unsupervised object-centric world model,
  strong prediction, policies below DreamerV3; latent drift during multi-object interactions destabilises policy
  learning. Better Slots, Better Worlds (2608.12078, RLC 2026 workshop): planning success rises with slot quality
  (FG-ARI, mBO) and saturates; well-bound slots more robust than LeWM; DINO-WM competitive. Most 2025 object-centric
  RL world models use reconstruction or frozen / pretrained encoders (SOLD, FIOC-WM, ObjectZero). EchoJEPA
  (2602.02603): same-architecture latent vs pixel targets, latent better for representation (ultrasound video with
  stochastic speckle, so relevant to nuisance suppression).

## Perception pass (2026-10-26)
- Dedieu et al. (2502.01591, ICML 2025; Craftax-Classic): 63x63 frame -> 81 patches of 7x7, each tokenised
  independently of its neighbours by a nearest-neighbour tokenizer (code added when no code within tau = 0.75; codes
  never updated; K = 4096) -> stationary targets; VQ-VAE -> NNT 58.9% -> 65.0% reward; 5x5 / 9x9 patches lower but
  competitive; "when the 7x7 patches are not quantized ... MBRL performance collapses" (21.2%), "consistent with ...
  DreamerV2"; tokens sampled independently per patch given history (block teacher forcing). Limitations: patch-size
  sensitivity, large codebooks under appearance variation (lighting / texture randomness "not very severe" in Craftax).
- LeWorldModel (2603.19312): ViT-Tiny, one global CLS vector (192-d), MSE next-embedding prediction + SIGReg
  (M = 1024 projections, lambda = 0.1), deterministic predictor, CEM planning over 5 steps; no policy learning,
  stochasticity not discussed, fine details (end-effector angle) not fully captured; SIGReg struggles in
  low-intrinsic-dimension scenes.
- R2-Dreamer (2603.18202, ICLR 2026; code vendored): decoder-free RSSM with categorical latents; Barlow-Twins-style
  redundancy reduction between projected latent features and the DETACHED encoder embedding (dreamer.py: "this
  detach is important"); no augmentation; competitive with DreamerV3 / TD-MPC2, largest gains on DMC-Subtle (tiny
  task-relevant objects).
- FSQ (Mentzer et al. ICLR 2024): project to < 10 dims, round each to a few levels; implicit codebook; no codebook
  collapse machinery needed. Normalisation as a canonical neural computation: responses divided by pooled activity
  (Carandini & Heeger, Nat Rev Neurosci 2012).
- A10 `eda_cell_evidence.py` (160 replayed TRAIN episodes; one full-resolution 7x7 cell; simulator labels for
  evaluation; regularised multiclass fit after an unstable first fit): terrain type accuracy and entity presence AP by
  Chebyshev ring 1-4: day and dusk ~100% / 1.00 at every ring; night (light < 0.25) terrain 100 / 99.6 / 96.3 / 93.1%,
  entity AP 0.998 / 0.992 / 0.930 / 0.836. [Correction 10-26, user + Codex: this is what a classifier reads from raw
  pixels (binary presence; terrain only where no entity), an upper bound for a high-capacity fovea, NOT what
  bounded-capacity periphery tokens will hold and not a perceptual ceiling; it does not support a cell-local
  encoder or a small fovea evidence role.]

## Lighting, context, observation stage (2026-10-26)
- Lighting in the brain: divisive normalisation / gain control (Carandini & Heeger 2012); lightness constancy discounts
  the illuminant to estimate reflectance, imperfectly (Adelson 2000, "Lightness perception and lightness illusions");
  anchoring theory (Gilchrist et al. 1999); discounting gives relative reflectance, an absolute scale needs anchoring
  (Hong & Grossberg aFILM). Night: rods only, the fovea is rod-free (~1 deg scotopic scotoma), scotopic acuity ~10x
  worse; the brain fills the foveal gap from the surround and observers trust the inferred content more than real
  peripheral input (Gloriani & Schutz 2019). Noisy measurements -> the prior dominates (Weiss, Simoncelli & Adelson,
  Nat Neurosci 2002, slow-speed prior). Attention = inferred precision of prediction errors (Feldman & Friston 2010).
- Periphery in the brain: nonlinear V1 responses pooled over regions that grow linearly with eccentricity (scaling
  ~0.46; Freeman & Simoncelli, Nat Neurosci 2011): the periphery is a pooled, contextual summary, not cell-local.
- Dedieu et al. tokenizer ladder (2502.01591): IRIS-style VQ-VAE on an 8x8 CNN feature map 43.36% -> factorising the
  VQ-VAE over the 81 tile-aligned 7x7 patches 58.92% -> NNT 64.96%; the factorisation "leverages a property of
  Craftax-classic" (patches aligned with the tiles); confounds: 64 vs 81 tokens, reconstruction-based.
- Our history: D4MJ's Raw encoder = ViT-12, width 192, patch 7 (one patch = one tile), global attention (R2); its
  entity content was linearly local at every layer and its persistent-content drift was lighting, not context.

## Encoder context pass (2026-10-26)
- Brain: the feedforward sweep groups hard-wired feature constellations fast; horizontal connections and feedback
  modulate tuning later (~80 ms+; figure-ground), needed for attentive grouping (Lamme & Roelfsema, TINS 2000).
  Surround modulation in V1 = flexible divisive normalisation gated by the statistical dependence of centre and
  surround (Coen-Cagli, Dayan & Schwartz, PLoS CB 2012). Crowding: clutter limits recognition across most of the
  visual field, mostly in the periphery (Whitney & Levi, TICS 2011); holistic information partly survives (Fischer &
  Whitney 2011). Peripheral pooling grows with eccentricity (Freeman & Simoncelli 2011). Reading: the periphery is
  pooled and loses individual identity in clutter; foveation resolves it.
- ML: ViTs repurpose low-information background patches as high-norm tokens holding global information; dedicated
  register tokens remove the artefacts and help dense prediction / object discovery (Darcet et al., ICLR 2024).
  Zero padding lets CNNs encode absolute position; no padding precludes it (Kayhan & van Gemert CVPR 2020; Islam et
  al. 2020/2021). ConViT (d'Ascoli et al., ICML 2021): attention heads initialised local with a learned gate to escape
  locality; more sample-efficient than DeiT. Dreamer 4's tokenizer reads separate learned latent tokens (not tied to
  patches; latents attend to all modalities; tanh bottleneck; MSE + 0.2 LPIPS; masked patches; causal in time).
- Our history: D4MJ backbone review (20261017_backbone_mechanism_review/RESULTS.md): in the jointly trained ViT
  (absolute position embeddings), patch SIGReg pointed toward WORSE camera-aligned token transport at all 8 paired
  checkpoints (25/32 cohorts); a local optimisation conflict, causal link to R68 not identified. P1 (p1_results.json):
  adding the 8 neighbours to a pooled cell never improves entity AP materially at r = 1, 2, 3 (often lower), day or
  night: in Craftax context adds no detection evidence even for degraded cells.
- A11 `eda_entity_adjacency.py` (A1 rollouts): share of visible entity-steps with another visible entity within
  Chebyshev 1 / 2: zombies 25 / 42% (day), 26 / 46% (night); cows 14 / 30%, 19 / 39%; skeletons 28 / 47%, 31 / 50%;
  arrows 34 / 56%, 36 / 60%. Adjacency is common: any receptive field beyond one tile often mixes two entities'
  evidence (crowding).

## Observation stage specifics (2026-10-26)
- Energy score ES_beta(F, y) = E||X - y||^beta - 0.5 E||X - X'||^beta, beta in (0, 2) strictly proper (beta = 2 only
  scores the mean); unbiased from m >= 2 samples (Gneiting & Raftery 2007). It is more sensitive to mean errors than
  to variance / dependence errors and loses discrimination in high dimension (Pinson & Tastu 2013; Scheuerer & Hamill
  2015; Alexander et al. 2101.12693); the variogram score targets dependence but is proper, not strictly.
- Engression (Shen & Meinshausen, JRSSB 2025, 2307.00835): Y = f(X, eps), eps from a fixed reference (pre-additive
  noise into the network), trained by the energy score; generative, extrapolates better than regression.
- Pacchiardi et al. (JMLR 2024, 2112.08217): generative networks trained by (prequential) scoring rules, consistent
  with dependent data; on WeatherBench the PATCHED energy score (energy score on local spatial patches, accumulated over
  the map; proper, not strictly) did best; adding a strictly proper score keeps the sum strictly proper (Lemma 3.4).
- Classifier two-sample tests (Lopez-Paz & Oquab, ICLR 2017): held-out classifier accuracy between real and generated
  samples, near chance iff the distributions match; its errors show where they differ.
- Tracking measurement models (PMBM, 1703.04264): detection probability p_D(conditions) plus clutter independent over
  time; persistence separates targets from clutter.

## Records pass (2026-10-26)
- Tracking: ByteTrack (2110.06864, ECCV 2022): associate low-score detections with existing tracklets instead of
  discarding them; recovers occluded objects; +1-10 IDF1 when added to 9 trackers. DeepSORT (1703.07402): appearance
  metric cuts identity switches 45%. Track-before-detect: unthresholded measurements integrated over time, existence as
  a two-state Markov chain (birth / death), best at low SNR (2312.15823 and refs). PermaTrack (2103.14258, ICCV 2021):
  spatio-temporal recurrent memory tracks fully occluded objects, supervised by synthetic labels for invisible objects.
- Object-centric: SlotSSMs (2406.12272, NeurIPS 2024): per-slot SSM transitions (block-diagonal), slots interact only
  through self-attention; gains on object-centric video, 3D reasoning, long-context video.
- Cognition: Alvarez & Franconeri (JoV 2007): tracking capacity is not fixed (up to 8 slow objects, 1 fast); a
  flexibly allocated resource (FLEX) vs fixed FINST indexes. Visual working memory: 3-4 slots (Luck & Vogel 1997) vs a
  shared resource with precision falling with set size (Wilken & Ma 2004; Bays & Husain 2008); debate unresolved.
  Entorhinal streams (Keene et al., J Neurosci 2016): LEC / perirhinal represent objects and, within them, their
  locations; MEC represents locations and, within them, the objects; object-location associations need both.
- A12 `eda_record_needs.py` (A1 rollouts, perfect observer): load (visible + alive out of view seen within 16 / 64
  steps) median 2, p90 4 / 5, p99 6 / 7, max 11, > 8 in 0.05 / 0.13% of steps. First-sighting urgency: arrows 5%
  adjacent at first sight, 14% within 1 step, 28% within 2; zombies 1.3 / 3.3 / 7.4%, 19% within 4, attack within 16
  steps 17% (re-entries: 0 / 0.9 / 3.0%, 8%, 6%). Re-entry identity: another reachable same-kind out-of-view entity in
  21% (zombies, skeletons) / 36% (cows); nearest last-seen correct 96.7 / 94.5 / 95.3% overall, 92 / 86 / 88% after
  gaps > 32. (Revivals count as first sightings here.)

## Records rethink (2026-10-26, evening)
- Structured World Belief (Singh, Peri, Kim, Kim & Ahn, ICML 2021, 2107.08577): belief = K SMC particles, each a set of
  N object files (ID, visibility, state, RNN history); "decomposes the presence of an object in the belief and its
  visibility in the observation"; learned file-slot matching (GNN interaction summary, softmax over M slots + none,
  sampled per particle); weight = previous weight x p(x | z) x prior / proposal ratios; AESMC ELBO; invisible files
  updated by the prior ("imagination phase"); no explicit deletion (option: delete the longest-invisible file when
  near capacity); soft resampling; more particles (K = 10 -> 30) removed over-counting; pixel reconstruction
  likelihood. Results on video modelling, RL, planning and reasoning.
- AESMC (Le et al. ICLR 2018, 1705.10306) and FIVO (Maddison et al. 2017, 1705.09279): model + proposal learned by an
  SMC lower bound on log marginal likelihood. Differentiable particle filters (Jonschkowski, Rastogi & Brock, RSS 2018):
  learned motion and measurement models trained end to end. JPDA merges neighbouring tracks (track coalescence;
  Blom & Bloem, IEEE TAC 2000). Layers: Wang & Adelson (IEEE TIP 1994): intensity + alpha maps, depth-ordered,
  occluding. Sampling-based neural representation of uncertainty (Fiser, Berkes, Orban & Lengyel, TICS 2010).
- Corrections (Codex, 10-26): Keene et al. show both entorhinal streams encode objects and places with different
  priorities, not a one-to-one map onto records / store; SlotSSMs' object learning used reconstruction (the
  architecture transfers, not the learning result); ByteTrack still thresholds and initialises tracks from high-score
  detections only. A12: load counts simulator-active entities only (no false tracks or alternatives); "another
  candidate" uses a maximum-speed envelope (no terrain); segments break at inactivity, so revival identity is not
  covered.
- A13 `eda_joint_uncertainty.py` (A1 rollouts): entities out of view and tracked at once (seen within 8 / 16 / 32
  steps): none 63 / 49 / 39%, one 27 / 32 / 34%, two 8 / 13 / 18%, three 2 / 4 / 7%, p99 3 / 4 / 4. At ambiguous
  re-entries (another reachable same-kind candidate), cooldown readiness differs between the true entity and a
  candidate in 45% (zombies, 7,446) / 39% (skeletons, 5,555): identity swaps would often change what happens next.

## Particles vs one hypothesis; generator simplification (2026-10-26, night)
- Identity in human tracking: positions tracked far better than identities; swaps when targets come close (Pylyshyn,
  Visual Cognition 2004); identity tracking draws on the location-tracking resource (Cohen, Pinto, Howe & Horowitz,
  APP 2011). Multistability as sampling: one interpretation at a time, stochastic switching (Gershman, Vul & Tenenbaum,
  Neural Computation 2012). RBMCDA (Sarkka, Vehtari & Lampinen, Information Fusion 2007): particles over data
  associations only, per-target Kalman filters, unknown target number with birth / death.
- OC-STORM (2501.16443, ICLR 2026): spatial-temporal transformer over K object tokens + 1 visual token; spatial
  attention across tokens, temporal attention per token type; next-step categorical latent per token (objects 16x16,
  visual 32x32) from an MLP; straight-through sampling; object features from a pretrained segmenter (Cutie / SAM2).
- A14 `eda_identity_posterior.py` (500 replayed episodes, 161,724 steps; perfect observer; candidate identities
  propagated with the game's mob rules, conditioned on non-sighting): zombies 2,555 re-entries (gap <= 32), 458 with a
  reachable alternative; P(true) < 0.9 in 61 (2.4% of re-entries; 0.38 / 1,000 steps); MAP wrong 29 (1.1%); alternatives
  with posterior >= 0.1 differing in cooldown readiness: 0. Cows 4,126 re-entries, 1,186 with an alternative; P(true) <
  0.9 in 138 (3.3%; 0.85 / 1,000 steps); MAP wrong 46 (1.1%). Births excluded; health not recorded.
- Corrections (Codex): "with M = 1 it is the agreed ELBO" was wrong (our recipe uses an energy score and asymmetric
  stop-gradient KLs); AESMC reports that tighter bounds / more particles can worsen proposal learning.

## Generator joint sampling; A14 correction (2026-10-26)
- A14 v1 RETRACTED (Codex): the true returner stayed on its real path while alternatives were tested (could block
  them) and zombies' attack stop was missing. A14b (`eda_identity_posterior_v2.py`, same 500 episodes / 161,724
  steps; identities in question removed from blockers; attack stop with per-particle cooldown; health recorded):
  zombies P(true) < 0.9 in 7.2% of 2,555 re-entries (1.1 / 1,000 steps), MAP wrong 2.7% (gap 1-2: 8, 3-8: 27,
  9-32: 35 errors), alternatives >= 0.1 differing in health 10 (0.06 / 1,000 steps), in cooldown readiness 0; cows
  7.2% of 4,126 (1.8 / 1,000), MAP wrong 2.4%, health 37 (0.23 / 1,000). Births excluded (lower bound).
- EMERALD (2507.04075): spatial categorical latent with a MaskGIT prior trained by KL(sg q(z | o) || p(z | h,
  z_masked)); Crafter score 51.6% (no MaskGIT) / 53.8% (1 decoding step) / 58.1% (3) / 55.1% (8); one-step parallel
  decoding produced incoherent tokens ("hallucinations"); first to exceed human experts on Crafter within 10M steps.
- Genie (2402.15391): dynamics model is a decoder-only MaskGIT transformer (mask rate 0.5-1 in training) over
  spatiotemporal tokens. MaskGIT (Chang et al. CVPR 2022): bidirectional transformer trained to recover random masks;
  parallel decoding refined over a few iterations; up to 48x faster than autoregressive decoding.

## Chance representation fork (2026-10-27)
- JEDI (2605.13013): online end-to-end latent diffusion world model inside a JEPA framework, no decoder; latent 16 x 8 x 8
  (tanh-clamped); denoiser predicts the clean next latent from the past three latents and actions; anti-collapse by
  stop-gradient target, encoder LR 0.3x, tanh clamp, random switching between denoiser and encoder outputs as
  conditioning; reward / termination gradients reach the encoder; decoder gradients slightly hurt. Atari100k mean HNS
  1.450 (DIAMOND 1.459), Har-HNS 0.377 vs 0.319; Craftium 4 tasks; stochastic Atari (random frameskip) beats DreamerV3;
  43% less VRAM, 3x faster sampling, 2.5x faster training than DIAMOND; 38 A100-h, 13.5M params. Limits: Atari100k and
  Craftium only, deterministic Atari, results sensitive to clamping / switching / batch size.
  CORRECTION to 10-26: a reconstruction-free spatial-latent world model with RL results exists (JEDI).
- Shortcut models (Frans et al., ICLR 2025, 2410.12557): one network conditioned on noise level and step size; one to
  few sampling steps; better than consistency models and reflow across step budgets.
- MDLM (Sahoo et al., NeurIPS 2024, 2406.07524): masked (absorbing-state) discrete diffusion; Rao-Blackwellised
  continuous-time ELBO = weighted average of masked cross-entropy losses; approaches autoregressive perplexity.

## Generator v3 sources (2026-10-27)
- Belief MDP (Kaelbling, Littman & Cassandra, Artificial Intelligence 1998; Astrom 1965): a POMDP is an MDP over beliefs;
  its transition samples the next observation from the belief's predictive distribution and applies the belief update.
- JEDI details: EDM log-normal noise levels, Euler sampler with 3 steps, context of 4 observations and actions,
  imagination horizon 15, actor / critic read the clean latent through an LSTM; online loop; per-game better of two
  denoiser batch sizes in the Atari table (Codex).
- Long-Context SSM video world models (Po et al. 2505.20171): block-wise Mamba scan + local attention, trained with
  diffusion forcing (independent per-frame noise levels) plus a random clean prefix to force long dependencies.
- Codex corrections (10-27): categorical vs continuous does not decide whether chance is explicit (continuous
  innovations can keep afterstate + chance); shortcut step-size disagreement measures sampling-resolution consistency,
  not calibrated epistemic uncertainty; HarmonyDream's rectified optimum is sigma^2 = mu (1 + sigma).

## Minimal build and test plan (2026-10-27)
- Craftax-Classic renderer: the HUD shows health, food, drink, energy and the inventory counts; achievements are not
  rendered, so reward is the only evidence of an unlock (reward and termination must be observations for the belief).
- RAE (Zheng, Ma, Tong & Xie, ICLR 2026, 2510.11690): diffusion on frozen SSL latents (DINOv2 etc.) works only with a
  denoiser wider than the token dimension and a noise-schedule shift scaled with token dimensionality; DiT-S trained
  directly on RAE latents failed (gFID 215.8 vs 51.7 on SD-VAE); noise-augmented decoder training.

## Curriculum sources (2026-10-27)
- Buitrago & Gu (ICML 2025, 2507.02782): recurrent models (Mamba-1 / 2, GLA, RWKV) fail beyond training length when
  trained on a subset of attainable states ("unexplored states hypothesis"); state passing (final state of another
  sequence, 10% zero) and truncated BPTT work equally well; ~100-1,000 post-training steps (0.02-0.1%) extend 2k -> 128k;
  longer training contexts help; long warm-ups and over-training on short contexts hurt.
- R2D2 (Kapturowski et al., ICLR 2019): stored recurrent states go stale as the network changes; a burn-in prefix
  without gradient rebuilds them.
- DaD (Venkatraman, Hebert & Bagnell, AAAI 2015): multi-step prediction as imitation learning, training data as the
  demonstrator correcting the model's own rollouts (valid where the recorded future is the target).
- History-Guided Video Diffusion (Song et al., ICML 2025): diffusion forcing transformer with flexible history; history
  guidance enables stable very long rollouts.
- Data: train episode lengths eps 0.1 median 478 (p90 1,530, max 2,500), eps 0.25 240, eps 0.5 184, eps 1.0 135
  (p90 235, max 444).
- Codex (10-27): DreamerV3's two-hot bins put a deterministic reward of 1 on ~0.877 and ~1.198 (checked: symlog bins
  spacing 40/254); a two-hot head is not a sampler.

## S0 (2026-10-27)
- Exact dev replay (1,009 episodes, 304,898 transitions; rewards reproduced to 3.6e-8). Observable gates: real half 0 / 43
  rejections, corruptions 2-38; one-step classifier two-sample test blind to delayed consequences (0.495) and nearly
  to duplication (0.543). Observable cooldown signature: P(health drop | zombie adjacent) 0.02-0.04 within 5 steps of
  the last drop, 0.35 at >= 6. Details: 20261027_s0_harness/RESULTS.md.
- S0 completed (same day). Statistical sources used, checked: energy score ES(F, y) = E||X - y|| - 0.5 E||X - X'||,
  strictly proper (Gneiting & Raftery 2007, JASA 102); its pairwise term over i != j with m(m - 1) is the "fair"
  ensemble form, whose expectation is optimised by members drawn from the true distribution (Ferro 2014, QJRMS 140; the
  form is written out for CRPS in secondary sources, its ES analogue is ours); the energy score's weak sensitivity to
  dependence structure (Pinson & Tastu 2013) is why persistence and counts are the pass / fail gates. Randomised PIT
  for discrete outcomes, uniform under calibration (Czado, Gneiting & Held 2009, Biometrics 65, whose own proposal is
  a non-randomised version). Equal-mass ("adaptive") calibration bins (Nixon et al. 2019, CVPRW). Classifier two-sample
  tests (Lopez-Paz & Oquab, ICLR 2017). Craftax-Classic renderer: night = per-frame random static (state_rng) with an
  intensity texture, desaturation and a night texture (craftax_classic/renderer.py), hence night readability by ring.
- S0 review (Codex, 10-27; verified): the scene stores evidence of past unlocks (tables, furnaces), so same-frame-pair
  counts beat the memory-free reward reference R2 by 13% without any history; Craftax map changes are the player's
  edits, plant ripening after 600 steps (game_logic.py) and arrows destroying tables / furnaces; Craftax entities move
  cardinally, so a one-cell diagonal step is illegal and now has its own gate.

## Denoiser backbone and periphery capacity (2026-10-27)
- Cost at M0 shapes (stand-ins, fwd + bwd, bf16; the full-model check stays the authority): grid conv, 4 residual
  blocks, 0.22 ms per target frame; transformer, 6 layers width 256, Dreamer-4 sequence form (every frame of a 64-step
  chunk a target, temporal attention to the past 4 frames every 2nd layer) 0.52 ms, 8 layers 0.69 ms; transformer with
  a 4-frame context window per target 2.4 ms.
- Positions: Dreamer 4's paper uses RoPE; both public reimplementations (nicklashansen, edwhu) add absolute sinusoids.
  Heo et al. (ECCV 2024): on ViT, absolute embeddings added to 2D RoPE helped at the smallest grid tested (112 px,
  7 x 7 tokens: +2.2 vs +1.1), not at large ones. DiT: adaLN-Zero beat in-context tokens and cross-attention for global
  conditioning, at the lowest cost. DIAMOND: frame stacking beat its early cross-attention trial; its limitations name
  a transformer over time for memory and scalability.
- Craftax cell information (eda_cell_information.py, eda_cell_appearance_residual.py; 300 dev episodes): a map cell's
  meaning (terrain, entity, player facing) takes 44 values, 2.7 bits on average (player cell 3.3). At full daylight and
  not sleeping, pixels carry 0.003 bits beyond meaning (arrow direction aside); the 1.5-bit residual over all daylight
  frames is viewing conditions (light level, sleep darkening). HUD cells: icon + a ~4 x 4 px count glyph; up to 3.3
  bits per cell, mean 1.6; no night static on the HUD.
- Pooling each cell to r x r (eda_pooled_separation.py; RMS distance between prototypes of different meaning, 0-255):
  full res: map median 86 (min 13.9), HUD values median of minima 32.6; 3 x 3: map median 69 (min 4.8), HUD 1.0 (some
  values identical); 1 x 1: map 61.5, HUD 0. With P1 (10-25): 3 x 3 keeps day entity detection at AP 1.00 for all four
  kinds; night zombies 0.86, cows 0.85, skeletons 0.64, arrows 0.78 (full res 0.97 / 0.95 / 0.85 / 0.91).
- A continuous token's width does not bound its information (only noise or quantisation does); Craftax cells carry at
  most ~5.5 bits of meaning, so any working width holds a cell's meaning by day.
