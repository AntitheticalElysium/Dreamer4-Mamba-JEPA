# Design decisions (system-design discussion from 2026-10-23)

One line each, with the reason. Literature and data facts live in READING.md; memory exploration in OPTION3_NOTES.md;
SPEC_v0.md is one sketch with unchosen assumptions, not a spec. Moved here 10-27 (package `saccade`); the EDA scripts,
superseded drafts (SPEC_v0, LEARNING_SKETCH, OPTION3_NOTES) and T1/T2 protocols named in these files stay in
artifacts/experiments/20261023_system_design/, the S0 results in artifacts/experiments/20261027_s0_harness/.

The first build, M0, is defined in STEP_SPEC §7; entries below marked [superseded] are history.

## Active experiment ownership (2026-11-02)
At the user's request, Codex owns the P1-F1.5 readiness and emergent-fovea experiment segment. The living evidence
map is [P2_READINESS.md](P2_READINESS.md); its bounded diagnostic protocol is
`artifacts/experiments/20261102_p1f15_readiness/PLAN.md`. This does not authorise a new architecture or loss by itself.
Decision 2 remains under discussion; the user asked to decide the controller's lasting action-contingency bias
after the diagnostics. P1-F2 is still intended as the clean architectural baseline plus the agreed chooser.
The user-approved bounded campaign is now reported in
`artifacts/experiments/20261102_p1f15_readiness/UPTAKE_RESULTS.md`: weak native evidence use, positive exploratory
gaze allocation, and a substantial online collection cost. These are discussion inputs, not new gates or a
controller decision. Self-feeding is P2's training target; outcome/query consistency and sampler tuning are parked.
P2 must reconcile detached-outcome credit and terminal-inclusive admission before implementation. P1's corpus
was not exogenous-filtered. No overnight launch or architecture/loss/corpus change was made by the campaign.

## Agreed
- Offline only; the actor trains in imagination. (The point of imagination training.)
- No full-frame pixel reconstruction and no pixel-equivalent persistent state; limited pixel grounding is intended:
  a fixed-budget fovea whose pixels are predicted. (Reason about latents; fidelity where attention goes.)
- Fovea placement is learned, driven by future consequences (value of information); tests: anticipation vs
  reaction, robustness to night static.
- Encoder trained jointly with the world. (A frozen encoder silently defined our targets.)
- Grounded targets: reward, termination, fovea pixels.
- "The model doesn't know" is part of the system (offline: nothing corrects the world where data is thin).
- Memory = working state (Mamba) + location-keyed episodic store read by attention over (location, content) pairs;
  pruning and parametric memory later if needed.
- Entities as object records (beliefs) in the working state, place traces in the store; K = 8 is a flexible cap.
- Self-location: not a focus, but coherent with noisy motion later (predict / correct / uncertainty). 2D now, 3D in
  mind.
- Mamba as the working state; the actor recipe as drafted (BC prior inside the actor). (User, 10-24.)
- The actor stage does not shape the world model's training for now: no BC head, no value-aware losses. (Generality.)
- Consequential entities stay representable whether or not they affect reward; the actor may ignore them.
- The learning objective splits into PERCEPTION (what the state holds) and DYNAMICS (distributional prediction of
  the next perception state). (Every working system separates them; ours merged them into one regression.)
- Weak coupling from dynamics back into perception (DreamerV3-style), kept flexible. (User, 10-24.)
- [Superseded 10-27: periphery acuity 3 x 3, full resolution the comparison] Periphery input stays full resolution initially; a degraded periphery is a planned later ablation (user, Codex, me,
  10-25). EDA P1 (eda_periphery_degradation.py): entity presence AP day / night from a pooled cell: zombies 1x1
  0.97 / 0.59, 3x3 1.00 / 0.86, full 1.00 / 0.97; skeletons and arrows collapse at night below 3x3. Degradation costs
  most at night, when threats matter most.
- Trans-saccadic prediction (same-frame, pre-glimpse) is the leading candidate for grounding the latent periphery;
  pixel vs latent target left open (pixels natural for Craftax sprites; DINOSAUR favours feature targets on natural
  images). It does not by itself solve memory compatibility or lighting separation (Codex).

- Dynamics direction: simulate entities, compose cells. Entities are flexible, exchangeable records (no type slots);
  cells are rendered from terrain + records + global context instead of moving contextual tokens (user's F warning).
  Details open. (User, Codex, me, 10-25.)

- The fovea has two roles (user, 10-25): an evidence stream (a high-capacity glimpse that updates the belief with
  selective high-quality information: HUD, surprising or dangerous events; pixel prediction only there) and a training
  signal (TSP grounds the periphery on fixated regions). Fixation favours decision-relevant information (user, 10-25):
  bootstrap offline with consequence-prediction gain (a proxy), then optimise value of information in imagination.
  Fixation adds fidelity; it must not be the only thing keeping unfixated entities represented (Codex).
- Correspondence becomes flow parsing (call delegated to me by the user, 10-25): self-motion predicted from the action
  and checked by global alignment; static content aligned by that shared shift (no per-cell terrain targets); the
  residual "what changed independently" map is a pixel-grounded target, reliability-weighted at night; no per-cell
  displacement labels (A4: terrain ill-posed, entity matching 72-89% right; A5: change map AUC 0.98 day / 0.87 night).
- Record belief: hypothesis sets are the candidate; their number (like K = 8 records) is a working number, tuned on
  the full model (user, 10-25). Dormant / revivable entities are one implementation of a requirement (predict DO
  revivals faithfully), not the requirement itself (Codex).
- "The model doesn't know": shelved until the complete model exists (user, 10-25); keep it cheap to add later (modular
  prior heads, reserved held-out mechanism splits, a truncation hook in the actor stage).
- Change-map target = the soft, observation-derived probability of a detector trained on generically corrupted day
  pairs with label-free targets (A9b; measured reliability, not ground truth); a grounded target beside reward /
  termination / fovea pixels (user, 10-26).
- TSP target = pixels of a 3x3-cell glimpse (user, 10-26; working size, recalibrate on data).
- No special-cased screen regions: the HUD is perceived like any region and read at high fidelity through the dynamic
  fovea (user, reaffirmed 10-26; earlier: a dynamic fovea "avoids hardcoding the HUD"). The fovea is an evidence
  stream, higher-capacity than the bounded-capacity periphery tokens.
- Perception tokens are continuous; categorical tokens shelved (user, 10-26). [Superseded 10-27 by generator v3:
  "chance codes stay categorical"; v3 has no chance codes.]
- Viewing-conditions (lighting) latent, CANONICAL (user, 10-26): one latent from the whole frame (a register) with three
  uses: gain on the encoder, constancy (perception discounts it, the generator composes appearance from content +
  conditions), evidence precision (how much the update trusts detections; observation-noise scale). Global first; a
  coarse spatial field is the general extension. Overconfident filling-in is a failure mode to check.
- [Superseded 10-27 by generator v3, which denoises the percept itself, noise included; the checks listed here
  carry over to the denoiser.] Observation stage = the generator's final stochastic stage targeting actual percepts, with per-frame noise shared by
  periphery and glimpse (user agreed, 10-26). Specifics (PERCEPTION.md v4): engression-style noise input; time /
  locality / capacity as an inductive bias (not a guarantee); patched + global energy score (beta = 1); joint
  periphery-glimpse scoring; independent chance draws for prior-predictive scoring; noise-belief loop checks.
- Cross-patch context: A (localisable patch percepts; context in the update and the fovea) LOCKED for the first build
  (user, 10-26); B (global) and C (bounded soft) kept in the spec for a matched comparison on the full model.

- M0 build settled 10-27 (user; Codex reviewed): denoiser = Dreamer-4-style space-time transformer with memory tokens
  (STEP_SPEC 7; the grid-conv proposal and its memory map dropped); JEDI switching dropped for M0 and kept as the FIRST
  ablation if self-feeding proves insufficient (CURRICULUM 2b); periphery acuity 3 x 3 by a declared rule (PERCEPTION;
  the 10-25 full-resolution choice becomes the planned comparison); every number in saccade/m0.yaml; code plan in
  M0_CODE.md; weak coupling = JEDI's encoder lr x 0.3 (LOSSES, reconciled); S1.2 probes and the acuity-relative S3
  perception blocker specified before training.

- Collapse prevention decided 10-27 (user asked for a dedicated investigation; my call): SIGReg dropped from M0.
  Routing (every denoiser frame input and target stop-gradient, memory tokens differentiable; REPA-E, JEDI) + a
  fixed-target owner per token type (adds the fovea's own reconstruction, the agreed limited pixel grounding) + scale
  by per-token layer norm without affine (I-JEPA, V-JEPA 2-AC) then JEDI's tanh clamp (revised after Codex's audit:
  BatchNorm over a chunk was not causal and broke the bound). Decision record:
  COLLAPSE.md. SIGReg kept as a planned comparison and for richer per-patch content later.

- M0 build (10-27): code in saccade/ (data.py, model.py, train.py, tests/test_m0.py); every check in M0_CODE passes
  (artifacts/saccade_m0/checks/CHECKS.md). One spec change from a measured implementation failure: the reward head is
  a categorical over the 256 symlog bins, not a discretised logistic mixture (LOSSES G3: its tail gradients took the
  clipped update and stalled the other terms; removing the term restored them). P1 not launched: Codex reviews first.

## Reserve ideas (keep; don't build until a measured need appears)
- Coarse peripheral target: predict a per-cell colour / contrast summary everywhere ("peripheral vision"). Cheap,
  biology-inspired; risk: zombies are the lowest-contrast sprite (09-26: 2.5 vs 2.6-6.5), so colour alone may miss
  them; whether it counts as full-frame reconstruction must be measured, not argued. (User, 10-24.)
  P1 adds: a 1x1 colour summary finds zombies at night with AP 0.59 only (skeletons 0.32), so as a target it would be
  weak exactly where threats matter.

## Open, in discussion
- Records design v3 (RECORDS.md), AGREED in direction (user, 10-26): one coherent association per step, per-object
  position hypotheses, layered ownership, presence vs visibility (no dormant state), births = inactive records
  activating, amortised correction (observation stage unchanged), imagination samples one world while the actor reads
  its belief. Added: revisability by deferred commitment when the association is split (A14b: 7% of re-entries).
  An approximation chosen for cost; scene particles documented as the upper end. A14 v1 retracted (A14b).
- [Superseded by v3, next entry] Generator v2 (GENERATOR.md): typed tokens (records, locations, global), one transition network, per-token
  categorical latents decoded JOINTLY by masked iterative decoding (EMERALD / Genie / MaskGIT), S = 3.
- Generator v3 APPROVED (user, 10-27): denoise the next percept from the predicted belief (belief-MDP; JEDI recipe:
  EDM, x-prediction, ~3 Euler steps; RAE width / noise-shift rules); no hidden world, no chance codes. Reward and
  termination are observations the update reads (Craftax HUD does not show achievements). Conservation and revisit
  consistency are learned properties to demonstrate.
- Build minimal first (user, 10-27): M0 = recent-percept window + Mamba state, no records / store / self-location
  (STEP_SPEC v1.1 §7); increments I1 (store + self-location) and I2 (records) enter through the belief and the
  denoiser's conditioning, on measured triggers (TEST_SPEC v0).
- Staged test v1 (TEST_SPEC.md): S0 harness with evaluator-power checks, S1 one-step, S2 recursion, S3 decisions;
  increments as diagnosed comparisons. Curriculum v0 (CURRICULUM.md): TBTT / state passing for long beliefs, context
  corruption + self-conditioning from the start, multi-step distributional post-training with a horizon schedule,
  imagination H = 16. Outcome models: reward as a samplable density (not two-hot), survival-conditioned termination.
  AGREED (user, 10-27); contracts per Codex recorded (CURRICULUM v1, LOSSES v1.1, STEP_SPEC v1.3, TEST_SPEC v1.1).
- S0 complete, reviewed by Codex and fixed 10-27 (artifacts/experiments/20261027_s0_harness/RESULTS.md): exact dev
  labels; 51 one-step gates (real half 0 rejections, seven corruptions rejected; zero-variance rule, cardinal-motion
  gates); revisit references (terrain almost static), aged from the compared sighting; multi-step power against a
  simulator oracle with the operational event gates on rollouts (all seven corruptions rejected, delayed consequences
  narrowly; the energy score never pass / fail); per-cell probe definition with its pixel reference (achieved
  readability); outcome evaluator with finite-learner references and common reward bins (beating R2 does not show
  memory; the S1.2 controls and probes do). References are recomputed through the frozen reader after P1.
- Losses (LOSSES.md) and step specification (STEP_SPEC.md): agreed 10-27 as v1.1 / v1.3 (see the staged-test entry).
- Self-state supervision (body / inventory) beyond reward / termination / TSP — separate from removing the HUD input
  path (Codex, 10-26). Health enters reward (0.1 x health change).
- Perception objective (wording pre-v3; under v3 the next-percept term is the denoising loss): latent next-perception prediction + soft change map + TSP at fixations +
  grounded consequences + store revisit prediction; static, unfixated, inconsequential entities rely on appearance
  shared with moving instances (a learning claim).
- Action confounding: future recorded actions reveal outcomes (multi-step sampled scoring) and the expert saw clean
  symbolic observations. Exogenous data: random-branch steps regenerated exactly from the collector's seeds (A8:
  740,693 / 740,693 match); eps = 1.0 episodes for long windows.
- Generator: how real observations enter the state = per-frame perception, one update for both regimes (10-25;
  settled by v3). Code families / births by source belong to v1-v2 and to I2.
- Rollout robustness (training on the model's own rollouts) and its link to consequence horizons.
- Loss units and weights: likelihood units, then a harmoniser; only after the above.
- Alignment during training before self-location is reliable.

## Implementation audit closure (2026-10-27)
User authorised verifying/fixing the audit findings and committing before the next-turn production P1 launch.
M0's architecture, budgets and held-out split remain as declared. Corrections:
- Keep the encoder's LayerNorm+tanh; sampled clean estimates use an idempotent range clamp, preserving a valid
  zero-loss target (LOSSES principle 7). V-JEPA normalises predictions inside its loss and is not evidence for applying
  this encoder map again after a different diffusion loss.
- S1 uses threshold-grouped AP, complete recent conditioning plus beliefs, equal outcome precision, observable
  kind-level sighting history and last-visible side, the health-adjusted unlock event, and a saved checkpoint-bound
  reader. C2ST uses the existing MLP probe (train-only standardisation, equal input-group energy): raw-coordinate trees
  failed a matched-marginal dependence control and two smoke power checks. Missing blocker exposures are explicitly
  unmeasured. These repair measurements, not scientific failures.
- Merge the duplicate YAML key; pin LightGBM 4.6.0 in the project venv; bind resume to configuration/seed/source/data;
  record bounded gradient/raw-token diagnostics. Long cache and metric regression tests cover the repaired contracts.
- Correct the launch working directory and the reward-gradient explanation: clipping factor is not Adam update size.
- Original audit results remain in artifacts/experiments/20261027_m0_implementation_audit; correction protocol/results
  are in artifacts/saccade_m0/checks/review_fixes. P2/P3 and their gradient checks remain deferred.


### 2026-11-02 status: conditional fine-evidence follow-up (no new design choice)

See [campaign results](../../artifacts/experiments/20261102_foveal_information/RESULTS.md)
and P2_READINESS §8. Raw fine tokens add entity information beyond full coarse history,
prior emitted belief and action. Much less consistent advantage is accessible through
whole-crop mixing and existing emitted memory; negative carry probing does not prove
erasure. No unique causal training diagnosis or repair selected. Cheap endogenous
self-distillation/direct expectation heads are research candidates, with batching and
proper NOOP-prefix teacher contracts explicit. No P1-F2 launch authorized by these tests.

### 2026-11-02: approved P1-F1.6 spatial fusion experiment
The user approved a from-scratch F1.5 comparison: every fine-to-belief input passes
through a shared local88->128->64 GELU joint coarse/fine/light transform. Its output
uses the existing token out_map; the raw fine bypass is removed, the ordinary coarse
path and256D projection remain. This supersedes the untrained optional16D residual
prototype. No new chooser, loss, sampler, corpus or budget. The observation targets,
TSP timeline and short fine-context cache are unchanged; this is a belief-interface
experiment, not a demonstrated global architectural repair. Protocol, papers,
pre-launch proof and one-shot training/evaluation orchestration:
`artifacts/experiments/20261102_p1f16/`. Seed1 only; matched F1.5 battery follows
automatically on the checkpoint selected by the existing monitor rule.

### 2026-11-03: F1.6 evaluated; no new architecture selected
Training25.6M frames and the full matched battery completed; checkpoint10 by the
original monitor rule. All48 pinned sources match; F1.5's repeated actor-interface
readings reproduce exactly. Fusion improves several consequences and real-pixel
reconstruction but worsens risky generated damage/death and cows. The intended
useful-look repair is not demonstrated: the original whole-map reading and a
matched58-scene source-local/native-TSP follow-up remain negligible, including
the40-scene common fine-readable subset. Connected sensitivity is not semantic
retention. Detailed evidence and limits:
[review](../../artifacts/experiments/20261103_p1f16_review/RESULTS.md).
No automatic adoption as a global upgrade, no new repair or P2/actor launch.


### 2026-11-03 — P1-F2 single-change preparation

User specifies F1.6 base, changing only fixation policy; explicitly retains the
original combined-score softmax with equal normalized C/S weights and20% uniform
exploration. Original8-frame101584-param coarse-only Head, not E3's matched
ablation architecture. No belief-conditioned chooser, new world loss, corpus
change or foveal-route repair. Experiment contract, online-clock working choice,
isolation/resume checks and paired automatic evaluation:
`artifacts/experiments/20261103_p1f2/PLAN.md`. Preparation/smokes authorized;
overnight training remains unlaunched until requested. This is a first online
acquisition test, not an adopted proven replacement for P2/P3 VoI.

### 2026-11-03 — P1-F2 launch authorized

User authorized the prepared P1-F2 and its automatic post-training evaluation;
mixer diagnosis is on hold pending those results. Launched22:38AEDT,
`saccade-p1f2.service`, invocation`08ad625927a347fd927753069220097f`.
No recipe amendment. Frozen pre-launch PLAN remains historical; launch identity,
source archive and live status reside in`artifacts/experiments/20261103_p1f2/`.
