# Perception proposal v4 (2026-10-26)

v1 was withdrawn after the user's correction (10-26): it special-cased the HUD with its own always-on encoder
(contradicting the agreed dynamic fovea, chosen precisely to avoid hard-coding the HUD), it called the fovea's evidence
role small (conflating full-resolution INPUT with full-information TOKENS), and it chose Craftax-shaped structure
(categorical per-cell tokens, cell-local encoder) on Craftax evidence. Principle: the goal is a general architecture
that answers the thesis, not a perfect Craftax model; Craftax-specific structure only with a measured need.

## Agreed or near-agreed
- Periphery = bounded-capacity latent tokens over the WHOLE frame, map and HUD rows treated identically (patches; 7 px
  in Craftax as a convenience). Less precise by design: the user's "blurry counterpart" to the fovea.
- Fovea = selective high-fidelity perception of a chosen crop anywhere on the screen, HUD included: a higher-capacity
  encoder whose tokens enter the belief as evidence, and whose pixels are the TSP target (3x3 cells, working size).
  Its evidence role is the capacity gap between periphery tokens and the fovea, a design parameter. A10 measured the
  readability a classifier achieved on raw cells; it is neither an upper bound nor what periphery tokens will hold.
- Fixation chosen for decision-relevant information (consequence-prediction gain offline, value of information in
  imagination).
- Tokens are CONTINUOUS (user, 10-26). Multimodality lives in the generator's categorical chance codes (DreamerV3's
  discrete stochastic state is used in continuous-control and 3D domains, so not Craftax-specific) and in the
  observation stage below. Categorical perception tokens shelved. Precedents for continuous percepts in long
  imagination: Dreamer 4 (tanh bottleneck latents, offline Minecraft), GameNGen (noise-augmented continuous latents).
  Dedieu's collapse of continuous patches was MSE reconstruction of raw patches, not a matched comparison.
- Anti-collapse: SIGReg on the continuous tokens (alternative: R2-Dreamer's Barlow-style term against a detached
  embedding).

## Viewing-conditions latent (lighting; AGREED, user 10-26)
Brain: (1) adaptation and gain control at the sensor (divisive normalisation, Carandini & Heeger 2012); (2) constancy:
the illuminant is estimated and discounted to recover surface properties (Adelson 2000; Gilchrist et al. 1999
anchoring); (3) reliability: dim light makes sensory evidence noisier, perception leans on priors and filling-in
(Weiss, Simoncelli & Adelson 2002), and attention as precision of prediction errors (Feldman & Friston 2010).
Gloriani & Schutz 2019 (observers trust inferred central content over real peripheral input at night) is a FAILURE
mode to guard against (overconfidence in inferred content), not evidence that the latent calibrates trust (Codex).
Design: one latent estimated from the whole frame, three uses:
- gain: modulates the encoder (normalisation / FiLM), so content tokens vary less with illumination;
- constancy and composition: perception discounts it, the generator composes appearance = f(content, conditions);
  our history: ~60% of day drift was frame-common lighting and ~64% of the old token-regression target was nuisance
  (R2, R55), so discounting lighting removes most of the old nuisance;
- precision: sets how much the update trusts detections versus tracking predictions, and the observation-noise scale.
Global first (Craftax's light is global); a coarse spatial field (shadows, local lights) is the general extension.
In imagination it comes from the belief's global stream (time of day, sleep).

## Observation stage (agreed in principle, 10-26; specifics v1)
The generator's final stochastic stage, not a module with its own target.
- Form (engression-style): percept = O(content, conditions, eps), eps ~ N(0, I) entering the network (pre-additive
  noise), one draw per frame per location, SHARED by the periphery head and the glimpse head (same frame = same
  realisation; fresh each frame, as real sensor noise and Craftax static are).
- Target: actual percepts (encoder outputs on real frames). Content is internal, never separately supervised.
- What separates world randomness from sensory noise (Codex: blocking inputs alone does not): (1) TIME: eps is fresh
  every frame while world content persists through codes -> records -> content; (2) LOCALITY (periphery head: one
  location plus conditions; the fovea head reads its crop's content and the shared noise field, Codex); (3) CAPACITY.
  An inductive bias, NOT a guarantee (Codex): a false percept can be accepted by the update, become a record and be
  re-rendered (a feedback loop), and the stage can misread the same persistent background repeatedly. Tracking
  analogue: detection probability p_D(conditions) + clutter independent over time (PMBM). The goal is coherent
  trajectories and appropriately uncertain beliefs, not a uniquely correct decomposition of every uncertainty source.
- Loss (written out): ES_1(F, y) = E||X - y|| - 0.5 E||X - X'|| (beta = 1; distances, not squared: beta = 2 only scores
  the mean), m >= 2 samples, unbiased. Applied as a PATCHED energy score over local windows of locations (captures
  local joint structure: adjacent entities, footprints; best in Pacchiardi et al. on spatial weather data) PLUS a
  global energy score over the whole percept (keeps the sum strictly proper, Pacchiardi Lemma 3.4). Periphery and
  glimpse are scored JOINTLY (the fixated window's periphery tokens concatenated with the glimpse tokens), so
  contradictory pairs are penalised, not only wrong marginals (Codex).
- Scope (Codex): under posterior codes the score trains the CONDITIONAL observation distribution (rendering + O);
  under prior codes on exogenous windows it trains the full predictive, and then the m samples must draw the chance
  codes independently too (sharing a chance draw drops between-world diversity from the pairwise term); computation
  is shared only where it does not depend on that draw. Both needed. The energy score is weak on
  dependence errors (Pinson & Tastu; Scheuerer & Hamill), so coherence is also checked outside the loss. It cannot
  recover entities the encoder omitted (entity retention stays a separate learning claim).
- Checks: classifier two-sample tests (Lopez-Paz & Oquab) between real and generated percepts per light level and
  position, under posterior and prior codes, on single percepts, periphery / glimpse pairs, and short trajectories
  conditioned on histories and actions (matched bins can coexist with wrong dynamics); entity probes on generated vs
  real percepts (existence, location, kind separately); the full noise-belief loop: spurious-percept lifetime,
  spurious-RECORD lifetime and false confirmation through imagination; the update's trust at night against evaluation
  labels (overconfidence check).
- Cost: conditional case, m = 2 samples of the final stage; prior-predictive case, m independent chance draws.

## TSP noise model
Pixel likelihood with location from (belief, conditions) — conditions also shift the mean (Craftax tints, desaturates
and blends at night, not only adds variance) — and scale from conditions and pixel position only.

## Dropped from v1
- HUD encoder and always-on HUD path (the fovea handles the HUD).
- Categorical per-cell tokens (shelved).
- Clean-target corruption consistency (would teach overconfidence under obscured evidence; Codex). Generic corruption
  stays for the label-free change-map teacher.

## Cross-patch context: A LOCKED for the first build (user, 10-26); B and C kept for the full-model comparison
Decision: A = localisable patch percepts; within-frame context is applied by the update (a shared module reading real
or generated percepts alike) and by the fovea (integrates its crop). Not established as the winner (Codex): a
predeclared A-vs-C comparison on the full model (matched data, capacity, budget, downstream modules; judged on
percepts, beliefs, recursive coherence AND decisions) remains planned.
- A: tokens p_x = e(patch_x; v), shared patch encoder, no absolute screen position in the token (position = token
  index, used explicitly by the update). The viewing-conditions latent v is the only cross-patch path into a token, so
  it must stay low-dimensional and computed from pooled frame statistics, or it becomes a channel for other entities
  and layout (Codex). Equivariance holds for patch-aligned shifts under fixed conditioning only (whole-tile camera
  moves in Craftax; sub-patch shifts in general need interpolation in the update).
- B: global contextual encoder (ViT, D4MJ's Raw; Dreamer 4).
- C: bounded soft context (ConViT-style gated local attention, initialised local; equivariant; registers). ConViT
  keeps C credible (heads can escape locality), it does not establish A's sample-efficiency advantage (Codex).
Why A first (interfaces, not guarantees): store traces, record assignment, per-location observation noise and the
generator are SIMPLER with localisable percepts. Context does not necessarily destroy localisation (SuperGlue
contextualises descriptors before localised matching), and locality does not guarantee object / background
separation (a local token can still conflate its entity with the terrain under it).
Evidence and its limits:
- D4MJ (R2): its contextual ViT kept entity content linearly local at every layer; its drift was lighting. Backbone
  review: patch SIGReg opposed camera-aligned transport at all 8 checkpoint means; absolute position embeddings were
  NOT isolated as the cause, and transport improved over part of training while semantic readout deteriorated.
- P1: little consistent detection gain from neighbours in this classifier setup, with exceptions (2x2 night arrows
  0.116 -> 0.226).
- A11: another visible entity within one cell in 14-36% of entity-steps; proximity, not measured crowding or loss of
  identity.
- Cost of A: joint cross-patch information may be lost to bounded per-patch compression; not measured anywhere (no
  "zero cost" claim); the fovea recovers only what it chooses to inspect.
- Brain: pooled periphery + crowding resolved by foveation (Freeman & Simoncelli; Whitney & Levi); grouping via
  horizontal connections and feedback (Lamme & Roelfsema).
