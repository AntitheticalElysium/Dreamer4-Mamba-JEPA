# Generator — CURRENT: v3 (section below, 2026-10-27). v2 sections kept above it for the record (superseded).

Status 2026-11-02: decision 1's factorized v3 amendment is implemented in P1-F1.5. It generates physical percepts
then conditional glimpses, not the separate concrete world of the historical v0-v2 summary below. Its current
evidence and P2 limitations are in [P2_READINESS.md](P2_READINESS.md).

The generator advances one concrete world by a step and renders its percept. It is the prior inside the belief update
(propagating records' hypotheses) and the simulator in imagination. v0 had six named code families; v1 replaced them
with three token types sampled in a fixed order (records -> locations -> global), which Codex showed does not couple
two RECORD events (a skeleton's shot and the arrow it creates are both record events). v2 is about how the next step's
random choices are drawn JOINTLY.

## The question: how to sample many coupled random choices in one step
Couplings that must hold within a step: parent -> child (shot -> arrow), exclusion (two mobs into one cell),
neighbouring first-seen terrain (a lake edge continues), self-state following the sampled interaction (damage after
an attack, inventory after mining).

| design | how coupling is obtained | evidence | cost / step |
|---|---|---|---|
| A. one global latent (DreamerV3, STORM, TWM) | one draw for the whole world | works on Crafter; our E22: one channel is consumed by entering terrain | 1 pass |
| B. per-token latents sampled independently (Dedieu block teacher forcing) | none within a step | our S5 / S6: own-noise coin flips make content appear / vanish; EMERALD: one-pass parallel decoding gives incoherent "hallucinations" on Crafter | 1 pass |
| C. per-token latents, fixed type order (v1) | across types only | misses within-type couplings (shot -> arrow, adjacent terrain) | 3-4 passes |
| D. per-token latents, full autoregressive order (IRIS, Delta-IRIS) | every dependency, in a fixed order | accurate; slow, order is arbitrary | one pass per token |
| E. per-token latents, masked iterative decoding (MaskGIT; Genie's dynamics; EMERALD) | learned conditionals among all tokens; confident tokens fixed first, the rest resampled conditioned on them | EMERALD (Crafter, 10M steps): score 51.6% without MaskGIT, 53.8% with one decoding step, 58.1% with 3, 55.1% with 8; Genie's dynamics model is a MaskGIT transformer | S passes (3) |
| F. diffusion / flow over the next state (Dreamer 4, DIAMOND) | joint refinement of continuous latents | strong long simulation; continuous targets and several denoising steps | 4+ passes |

## Recommendation: E on our typed tokens
- Tokens: records (K; active or inactive), locations (in view), one global token (self-state, time, illumination).
  Typing keeps entity randomness and terrain randomness in separate latents (it encourages, not guarantees, the split;
  our E22 failure is the reason to type).
- One transition network reads the previous world (attention across tokens + the action; per-token recurrent state =
  slots of the Mamba working state) and outputs a categorical latent distribution per token.
- Prior training (EMERALD's form): KL(sg q(c) || p(c_masked | world, a, c_unmasked)) with a random mask ratio, so the
  prior learns every conditional among tokens; plus the agreed weak coupling KL(q || sg p) with a small weight and free
  bits. The posterior q per token reads (world, action, next percept) in local windows, as before.
- Sampling: S = 3 iterative decoding steps (working number): all tokens predicted, the most confident kept, the rest
  re-masked and re-predicted conditioned on them. A deterministic parent event (a ready skeleton shooting) is decoded
  early and its child conditions on it; terrain flows outward from confident cells; conflicts are resolved by the
  conditionals rather than post hoc.
- Births are inactive records activating; first-seen content is a location latent, sampled once per branch and written
  to the branch store; the action target's outcome is the target location's latent; self-state is the global latent.
  No named event families.

## Rendering
Background from the store + edits -> front-most active record per location -> viewing conditions -> observation stage
(energy-score trained, shared per-frame noise; the glimpse from the same content).

## Precedents and limits
EMERALD's evidence is a spatial categorical latent over pixels (reconstruction-based), not typed object / location
tokens without reconstruction; OC-STORM's object + scene tokens rely on a pretrained segmenter (Codex). The joint
decoding mechanism transfers; object formation stays our learning claim.

## Open
Latent sizes per type; S; mask schedule; whether record motion should use continuous codes in general; how an
activating record picks its location (attention over locations); per-token "doesn't know" heads later.

## v3 (2026-10-27): continuous next-percept denoising from the belief (user chose B)
The fork's question "what is denoised" (Codex) is answered by the belief-MDP construction (Kaelbling, Littman &
Cassandra 1998): a partially observed world is simulated from the agent's side by sampling the next OBSERVATION from the
belief's predictive distribution p(o' | b, a) and updating the belief b' = tau(b, a, o'). No hidden concrete world is
needed; with an exact belief this gives the same distribution over observation sequences as sampling a hidden world.
- What is denoised: the next percept (lighting latent, periphery tokens, glimpse tokens at the fixation), i.e. OBSERVED,
  stop-gradient targets, as JEDI denoises next encoder latents. Persistent content (records, store, Mamba state) lives
  in the belief and is never a denoising target; it is trained by having to predict real percepts.
- Conditioning: the PREDICTED belief B- (the update's deterministic predict step: self-location from the action,
  records' predicted hypotheses, store reads, Mamba state). B- plays the afterstate's role; the denoiser supplies the
  chance (Codex: continuous innovations keep the afterstate-plus-chance principle).
- Backbone (10-27): Dreamer-4-style space-time transformer over percept and memory tokens (STEP_SPEC 7, m0.yaml).
- Sampler: start with JEDI's recipe (EDM noise levels, x-prediction, ~3 Euler steps); shortcut forcing (Dreamer 4) is a
  later speed adaptation, not part of the precedent. Context corruption / own-sample conditioning for robustness
  (Dreamer 4 tau_ctx; JEDI's random switching).
- Precedents: JEDI (reconstruction-free latent diffusion; online; 3 Euler steps; context 4; Atari100k, stochastic Atari,
  Craftium; its Atari table takes the better of two batch sizes per game); Long-Context SSM video world models (Po et
  al. 2025: Mamba state + diffusion forcing); Dreamer 4 (offline imagination on generated latents).
- Folded away: posterior chance codes, masked prior, KL pair and free bits, the generator's world transition and typed
  record / global tokens, the world sample, the separate observation-noise head. The OBSERVATION / PERSISTENT-CONTENT
  distinction is kept (Codex): the belief is the persistent content, the percept is incomplete noisy evidence, and
  periphery + glimpse are denoised jointly so they stay correlated. Amended by decision 1 (agreed 11-01; learned-fixation
  TEACHING 10 / 10a; `denoiser.factorised`, off for the P1 / P1-F worlds): the correlation is kept by the chain rule
  instead, periphery and lighting never reading the glimpse, the glimpse sampled given them (cascade).
- What moves into the belief update: records' predicted motion becomes the update's own deterministic predict step
  (propagating each record's hypothesis distribution); outcomes (which way the zombie went, an arrival) appear in the
  generated percept and are absorbed by the update's correct step. Records SUPPORT conservation through the conditioning
  and association; the generated trajectories must DEMONSTRATE it (Codex).
- Thesis wording: chance is implicit (entity-structured through the belief's records), no longer an explicit compact
  code; the four-row separation stays (nuisance: lighting latent + observation noise; memory: Mamba + store;
  aleatoric: the sampler; epistemic content: written to the store once generated).

## Superseded fork table (2026-10-27, kept for the record)
## Open fork (2026-10-27): how chance is represented — categorical codes + masked decoding, or continuous denoising
Why categorical in the first place (WHITEBOARD §3 / §8, 10-12): an explicit chance variable sized to the aleatoric part
(Stochastic MuZero afterstate + categorical chance), so the deterministic completion takes a point loss and the chance is
measurable in bits (~1.37 bits per visible mob); categorical > Gaussian latents in DreamerV2 (42 / 55 Atari games).
The relation the user spotted: MaskGIT is the practical few-step form of masked (absorbing-state) discrete diffusion
(MDLM, Sahoo et al. NeurIPS 2024: the ELBO is a weighted sum of masked cross-entropies); shortcut forcing (Dreamer 4)
is diffusion forcing (per-frame noise levels) + shortcut models (Frans et al. ICLR 2025: the denoiser also takes the step
size, so 1-4 steps suffice). Both are iterative joint samplers; the difference is WHAT is sampled.

| | A. categorical codes + masked decoding (v2) | B. continuous typed tokens + shortcut / flow denoising |
|---|---|---|
| what is sampled | small per-token chance codes; the next state is a deterministic completion | the next state itself (record states, location percepts, global) |
| reconstruction-free spatial precedent in RL | none (EMERALD's MaskGIT latents are reconstruction-trained) | JEDI (2605.13013): 16x8x8 latent, next-latent diffusion, no decoder; Atari100k mean HNS 1.450 vs DIAMOND 1.459; beats DreamerV3 on stochastic Atari; Craftium tasks; 38 A100-h, 13.5M params |
| parts needed | posterior q, masked prior, KL (dyn / rep), free bits, deterministic completion, observation-noise head | one denoiser (history + action conditioned), noise / step-size schedule; the observation stage folds into it |
| chance made explicit and measurable | yes (codes, bits, attribution to entities) | implicit in the noise (per-token, so still entity-structured) |
| continuous randomness (general / 3D) | discretised (DreamerV3 does this on control) | native |
| known vs to-sample content (store recall, first-seen terrain) | unmasked vs masked tokens | per-token noise level 0 vs 1 (diffusion forcing) |
| robustness to own outputs | posterior-conditioned unrolls | context corruption (Dreamer 4 tau_ctx 0.1), JEDI's random switching, self forcing |
| our history | E22MS (explicit chance + completion) failed for measured reasons (terrain absorbed the channel; completion without conservation) | untested by us |
Cost per imagined step is similar (S decoding passes vs S denoising steps over the same tokens); neither realises
"compute proportional to chance entropy" in practice. Decision belongs to the user (it changes the thesis wording from
"explicit compact chance" to "entity-structured stochastic generation").
