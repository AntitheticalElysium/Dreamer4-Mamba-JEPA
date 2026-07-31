# Decisions and function plan

Companion to `ARCHITECTURE.md`. That file says what the system is; this one says
what is settled, what is open, and exactly which functions exist.

The contract: **no new files, and no function that is not listed here.** Private
stateless tensor helpers are permitted and are listed like everything else — the
defence is planning every signature in advance, not forcing arithmetic to be
duplicated inside oversized functions. Anything absent below is a design change,
not an implementation detail. If the plan is wrong, fix the plan.

## Settled

| # | Decision | Basis |
|---|---|---|
| S1 | Patch size 7, no resize, 81 tokens — one token per rendered tile | Craftax-Classic obs is 63×63 with `BLOCK_PIXEL_SIZE_AGENT = 7`; 63 = 9×7. Divisors of 63 are {1,3,7,9,21,63}, of which only {7,21} are tile-aligned and non-degenerate. 7 is the finest. `[R0-CHOICE]` — no genericity claim is made |
| S2 | `Z*` = D4 `tanh` bottleneck, **unnormalized**; direct readout is `tanh`-bounded | Keeps the flow anchor paper-faithful and Stage A genuinely fixed-representation. Matches the readout's codomain to its target's, which is the principle behind V-JEPA's `normalize_reps` for unbounded features. Bounds range only — **does not** address the scale contraction toward the conditional mean, which S35 shows is a proven consequence of determinism rather than an open question |
| S3 | MTP `L = 8`, leads `n = 0..7`; policy lead 0 = outgoing `a_t`, reward lead 0 = incoming `r_t` | §3.3 states `L = 8`. Module defaults agree in both reproductions, but edwhu's eval script runs `L: int = 2` and `nicklashansen/dreamer4` has no MTP heads at all, so the paper is the basis and the reproductions only fail to contradict it. Eq. (9)'s inclusive `Σ_{n=0}^{L}` reads as nine terms and is recorded as ambiguous |
| S4 | Reward and value: 255 bins, symlog support ±20 | D4 §3.3 "Following Dreamer 3"; D3 `rewhead`/`value` `bins: 255` with centres `symexp(linspace(-20,0))` mirrored. Two-hot is linear interpolation between neighbouring centres, so bin count sets grid density, not a quantization floor; the wide support is miscalibration headroom, not wasted resolution |
| S5 | `time_every = 4` | D4 §3.4. Reproductions ship 2 and 1 — not inherited |
| S6 | λ-returns in next-index form, `G_t = r_{t+1} + γc_{t+1}[(1−λ)v_{t+1} + λG_{t+1}]` | Eq. (10) prints same-index reward, continuation and value, which is not self-consistent with §3.3's shared-index annotation. `dreamerv3/agent.py:487` indexes `rew[:, 1:]` and `boot[:, 1:]`, expanding to exactly this form |
| S7 | λ = 0.95 | `[DESIGN]`. D4 is silent; DreamerV3 `lam: 0.95` is implementation precedent only |
| S8 | Temporal state is a per-layer, per-slot pair; the conditioning slot is retained in every arm | `ARCHITECTURE.md` Boxes 3 and 4 |
| S9 | One generic `evaluate` covers every path | The occupying latent and its conditioning are per-call arguments, not state. Flow runs N candidate rungs then one commit; direct runs a predictor head then one commit (S34) |
| S10 | The signal table has exactly `k_max` rows; every context index is clamped to `k_max−1` | Eq. (4)'s grid tops out at `1−d`, so τ = 1 is never trained. MMBench2 sizes `signal_embed` at `k_max+1` and labels uncorrupted context with index `k_max` — a row its own sampler cannot reach; `plan_cem.py:189,545` pass `tau_ctx=0.0`, so its planner runs entirely on that untrained row, and `round(0.9·k_max) = k_max` also fires at `k_max = 4`. Same pathology as the predecessor's unreachable shortcut rows, in the upstream source |
| S16 | τ_ctx = **0.1 noise / 0.9 signal**, snapped to the nearest trained bin | `train_dynamics.py:576` states it verbatim: "tau_ctx is the noise fraction: 0 = fully clean, 1 = fully noisy". The paper's "signal level τ_ctx = 0.1" is a slip — 0.1 *signal* contradicts its own word "slightly". No longer open |
| ~~S17~~ | **Superseded by S34.** The in-block query is untrainable in one pass, which S17 did not see: a position cannot hold the real latent for later blocks to attend to *and* the query to be predicted from. Direct now predicts from committed world features. The firewall objection S17 raised against that is answered by reading spatial and register outputs, which the dynamics mask makes agent-free | |
| S18 | Invariant 5 is `[DESIGN]`, not `[D4-ENTAILED]` | No audited source persists dynamics state across accepted frames — MMBench2 re-draws context noise per frame (`z0_ctx = torch.randn_like(...)` inside the per-frame call) and re-prefills. Persistence is our deviation and the `[MAMBA]` efficiency hypothesis rests on it |
| S11 | Context corruption is applied once at commit, never at read | Fresh noise per read changes the prefix every frame, invalidating any KV cache or SSM state and contradicting invariant 5 |
| S12 | Agent slots exist from Phase 1B, last in the layout, masked both ways until Phase 2 | Fixes `S`, mask shape, stream count and state shapes once, so no Phase-1B gate is invalidated when the agent modality activates |
| S13 | `depth % time_every == 0` and at least two time layers | At `depth = 4` the entire Mamba blast radius would be one module and parameter matching would be dominated by it |
| S14 | Anchor drops GQA and attention-logit soft capping, keeps RoPE and QKNorm | Table 2 shows GQA at FVD 70 → 71, adopted for KV bandwidth at 2B parameters — no benefit at our scale. RoPE and QKNorm have executable precedent in pinned MMBench2 attention; GQA and capping have none anywhere. Anchor is described as paper-constrained *modulo declared omissions* |
| S19 | Flow is **5 backbone passes/frame**, direct is **1** plus a predictor head | Flow's commit is a real extra pass: S11 puts corruption at commit, and the final rung's input is the running Euler iterate, not a fresh corruption of the accepted latent. Direct needs no candidate pass at all under S34. 5-vs-1 is an *evaluation-count* ratio; `diagnostics.cost` measures whether it is a throughput ratio |
| S20 | Causal temporal tokenizer, as D4 §3.1 | The primary model follows Dreamer 4. Both arms share one encoder, so the T-vs-M comparison stays fair even though the encoder carries history; a frame-only encoder is a later ablation, not a fork |
| S21 | `k_max ≥ 8`, declared in `Config` alongside `K = 4` | `round(0.9·k_max)` hits the untrained top row exactly at `k_max = 4`, and S10's clamp then drops τ_ctx to 0.75. Both failures vanish at `k_max ≥ 8` (τ_ctx = 0.875). k_max sets the training noise grid; K is the generation rung count — they are independent and neither was registered |
| S22 | The reward and continuation caused by `a_t` are read at lead 0 of `h_{t+1}`, never `h_t` | S3 defines reward lead 0 as the reward *arriving*. `a_t` is chosen at `h_t`; its consequence arrives with `o_{t+1}`. Reading lead 0 at `h_t` returns the previous action's reward and shifts every return by one step — the predecessor's `reward_logits[:, 0, 0]` is correct only under this reading. Asserted by `gates.alignment` |
| S23 | `Z*` is defined at MAE probability 0 | Masking is a Phase-1A training mechanism. The predecessor trained with masking silently disabled while advertising 0.9; the mirror failure is emitting cached targets under a random mask, which makes the same frame yield different `Z*` |
| ~~S24~~ | **Superseded by S34.** With no candidate pass there is nothing for a `{candidate, commit}` table to distinguish, so direct's conditioning slot carries a single reachable embedding | |
| S25 | Action table has `n_actions + 1` rows, the extra being BOS | Reachable at every true episode start; the predecessor shipped 18 rows undocumented. The query-shape half is void under S34, which has no query |
| S26 | Committed and observed blocks carry the finest step index `d_min` | The signal bin is fixed by S16; nothing fixed `d`. MMBench2 labels context blocks with `d_min`; training samples `d` per block, so a committed block needs a declared value rather than an inherited one |
| S27 | Bounded encoder context `W`, part of `C*` — `z_t = Z*(x_{t−W+1..t})` everywhere | Phase 1A windows carry a `W−1` burn-in that is encoded but not scored; once frozen, each episode is scanned once and cached **under the same `W` limit** — an unbounded full-episode scan would produce a different `Z*` from deployment. `W` is not a capacity number: it defines the representation, so changing it changes every `z_t` and it must be frozen before the final encoder trains |
| S28 | Match total **deployed** parameters within a declared tolerance while holding `d_model`, depth, token layout and shared interfaces fixed | Primary knob is Mamba's `d_state`, the only one that moves M-arm parameters without touching the shared backbone. Parameter counts move discretely, so `d_state` alone may not reach tolerance at a sane state size — any additional knob must be declared before training. Report the unmatched residual, FLOPs, memory, recurrent-state size and measured throughput regardless. The predecessor called arms matched in a comment while the temporal module differed by 29.6% |
| S29 | Archived replay is for debugging and smoke tests only; regenerated replay backs every reported number | Its expert has no byte-level provenance and its terminal-window support is 58 windows. Keeping both uses named stops the old set from quietly becoming the final dataset, and keeps `expert.py` exercised |
| S30 | The frozen latent cache lives on `Episode` as `latents` + `latent_digest`; `train_representation` writes it at the Phase-1A boundary and `load_episodes` verifies it | The digest covers exactly `C*`: encoder checkpoint, exported copy, `W`, bottleneck and packing, `p_mae = 0`, patch size. Without it a cache built under a different encoder or `W` is silently reusable, which is the one place a wrong number contaminates every downstream result at once |
| S31 | `Batch` names its regions explicitly: `burn_in` (int prefix), `valid` (per-position target validity). The MAE mask is **not** a loader output — the encoder generates it | "masks" was ambiguous across five different things. Burn-in is always a prefix, so an integer beats a mask. Burn-in frames update encoder memory and score no loss; scored frames follow immediately under that memory. `patches` and `latents` are phase-determined: Phase 1A carries pixels, Phase 1B onward carries cached latents |
| S32 | `Encoder.forward(patches, memory, p_mask, rng) -> (z, memory, patch_mask)` | One signature serves all four uses: Phase-1A window training, frozen episode scanning, recurrent execution, and diagnostics. The caller supplies and receives the bounded-`W` memory, so batched scanning and frame-by-frame execution are identical by construction; `p_mask = 0` on every `Z*` path per S23; burn-in is the caller slicing `z[:, burn_in:]`, which keeps it out of the encoder |
| S33 | `scan_step_parity` also covers the encoder; `alignment` also asserts the `W` horizon and `p_mae = 0` | Batched `W`-context scan ≡ frame-by-frame recurrence ≡ the cached `Z*`; frames older than `W` cannot change `z_t`; burn-in + scored window ≡ episode caching; reset clears encoder memory. Folded into existing gates — no new function |
| S15 | Windows never cross an episode boundary; `evaluate` takes no reset mask | Keeps a reset a fresh construction. Asserted by `gates.alignment`; relaxing it changes the signature |

## Design decisions taken on evidence

Three questions that shape the experiment rather than the code. Each was settled
against `third_party/`, not by preference.

### S34 — Direct predicts from an external head over committed world features

**Rejected:** an in-block query trained in one teacher-forced pass. Position `t`
cannot simultaneously hold the real latent, so later positions can attend to it,
*and* hold the query, so it can be predicted from. Filling every position with a
query resolves that by deleting the task: measured, the prediction is bitwise
identical for two different latent sequences, so the model is asked to predict
`z_t` from the action history alone.

**Also rejected:** a two-pass teacher-forced loss (commit pass to build memory,
query pass against it). Correct, but doubles Direct's training cost for a
mechanism no source uses.

**Taken:** `ẑ_{t+1} = P(world features of committed block t, a_t)`, one pass, the
predictor reading spatial and register outputs. This is what V-JEPA 2-AC does --
its predictor is a separate module over the interleaved `(a_k, s_k, z_k)`
sequence -- and what DINO-WM does. Box 4 rejected it over the §3.3 firewall, but
that objection only bites if the predictor pools the *agent* slot; spatial and
register outputs are agent-free by induction over depth, which `gates.firewall`
already asserts.

Dreamer 4 can share its backbone because a corrupted latent at `τ > 0` still
carries signal, so one block is both context and query. A query token carries
none, so the same trick does not transfer.

### S35 — Deterministic Direct is `K = 1`; the stochastic extension is best-of-K

Craftax is stochastic enough for this to matter. Measured on the installed
environment, 64 draws per `(s, a)` over 72 pairs: **83% branch**, mean 4.2
distinct successors, and top-mode mass as low as 0.19. The distribution is
heavy-tailed -- most states are near-deterministic, a minority are strongly
multimodal, and those are the mob-spawn and combat states that decide episodes.

MoP-JEPA (`2607.05238`, Prop. 1) proves the consequence: under squared loss the
optimal single predictor is `E[z'|c] = Σ w_m μ_m`, error lower-bounded by the
between-mode variance, and for separated modes the optimum lies far from *every*
mode. Under cosine loss with normalised targets the same holds. Prop. 2 shows a
gated weighted-sum mixture does not escape it -- it still emits one vector.

The fix is Prop. 3: best-of-K regression, `L = E[min_k ‖g_k(c) − z'‖²]`, which is
the per-context K-means distortion, so every optimum assigns a head per mode.
Plus a router trained on the winning index and a load-balance term.

**Why this settles the roadmap rather than the code:** at `K = 1` the MoP loss
*reduces exactly to the dense loss*. So deterministic Direct is not a separate
design -- it is the `K = 1` case of the extension. Stage A runs `K = 1`; if it
collapses, `K > 1` is a strict generalisation of the same head, not a new arm.

**Consequence for the plan:** `diagnostics.multistep_error` is a mean error, and
the mean error is *minimised* by the collapsed predictor, so it cannot detect the
failure it exists to detect. It needs MoP-JEPA's measurement alongside: distance
from the prediction to the nearest true successor mode, against distance to their
mean, at high-branching contexts.

### S36 — Dynamics context `C = 3·T_short`, with `T_long = 4·T_short`

Neither cache is currently bounded, and `Config` had no dynamics-context field at
all. Appendix A gives all three of D4's configurations: Minecraft `C=192,
T1=64, T2=256`; SOAR and Epic Kitchens `C=96, T1=32, T2=128`. Every one satisfies
`C = 3·T_short` and `T_long = 4·T_short` -- an invariant across every
configuration the paper reports, not a single data point.

Taking `T_short = 16` gives `C = 48`, `T_long = 64`. This also satisfies §3.4's
requirement that the long batch exceed the context.

## Open, with the functions each one constrains

Nothing here blocks writing the listed signatures. Each blocks a *body* or a
config value, and each must be closed before the phase named.

| Question | Close before | Constrains |
|---|---|---|
| **SIGReg on a projector vs on `z`** — projector keeps `Z*` and gives clean attribution; on `z` forks `Z*` from the anchor and tests the stronger claim | Stage B | `representation.Projector`, `representation.representation_loss` |
| **EMA views / masks / loss; faithful LeJEPA vs anti-collapse ablation** — is the EMA arm learning spatial invariance, temporal predictability, or both? | Stage B | `data.views`, `representation.representation_loss`, `representation.update_target` |
| **Generated-prefix contract** — not just a length: the one-step term, the autoregressive term, rollout length, whether gradients pass the first prediction, relative weighting, whether the real predict-then-commit path is used, and how running-RMS treats the composite. Likely first version: one-step teacher-forced + two-step autoregressive, with the first prediction committed through the runtime path so the loss actually exercises `advance` | **Before Stage-A direct training** | `transition.transition_loss` |
| **Go/no-go threshold *numbers*** — formulas exist now; scales come from the anchor or a pilot; numbers freeze **before any experimental cell is inspected**. Choosing them after all four cells train is not preregistration, whatever the intent | After the anchor, before inspecting Direct/Mamba cells | `Config` |
| **Capacity**: `n_latents`, `d_bottleneck`, `n_spatial`, `k`, `depth`, `d_model`, `W`, `k_max`, sequence lengths — needs a 6 GB probe over the real worst cases (tokenizer training with decoder and gradients, long-sequence flow-Transformer training, Mamba training and parity, optimizer state and EMA copies), not forward inference. Invariant `n_latents = n_spatial × k` is a consequence of D4's packing, not a law | Phase 1A | `Config` fields only |
| **Imagination horizon** — set from measured multi-step accuracy, not inherited | Phase 3 | `Config` field, gated by `diagnostics.multistep_error` |
| **Matching tolerance, final Mamba dimensions, and any declared fallback knob** — the *rule* is settled (S28); these are its numbers | **Before building the Stage-A models** | `diagnostics.cost`, `Config` |
| **Executed-control metric definition** — primary score, seed set, episode count, paired comparison, sampled vs deterministic policy, control of the flow arm's deployment-corruption draw, BC and random controls, aggregation and intervals. Defined **before** Stage A; *run* at its exit. Defining it afterwards leaves room to pick whichever metric flatters the preferred arm | **Before Stage A** | `execution.run_episode`, `Result` |
| **Expert regeneration settings, acceptance threshold and provenance record** — the *policy* is settled (S29); these are its parameters | Phase 1A | `expert.train_expert`, `expert.collect` |

## Function plan

Modules in architecture order. `Type` means a dataclass or `nn.Module`;
everything else is a function. Every entry maps to a box in `ARCHITECTURE.md`.

### Contracts

**`config.py`** — `Config` (Type, frozen). Every constant, plus
`transition ∈ {flow, direct}` and `time_mixer ∈ {attention, mamba}`. The four
Stage-A arms are four `Config` values; there is no arm factory.

**`sources.py`** — `source_digests(config)`, `verify_sources(recorded, config)`.

**`state.py`** — `WorldState` (Type: latent, memory, step), `RealState` (Type:
encoder_memory, world). No functions: a reset is a fresh construction, and
non-mutation is a property of the time mixer rather than a caller duty.

### Environment and data — Box 1

**`env.py`** — `reset(key)`, `step(state, action)`. `step` returns `terminated`
and `truncated` as separate raw fields; continuation is derived in `agent.py`.

**`expert.py`** — `train_expert(config)`, `collect(params, config)`.

**`data.py`** — `Episode` (Type, unshifted storage: `observations`,
`actions_taken`, `rewards`, `terminated`, `truncated`), `Batch` (Type, block
arrays: `patches`, `led_to_action`, targets, masks), `patchify(frames, patch)`,
`episode_splits(n, seed)`, `sample_batch(episodes, rng, config)`,
`save_episodes(path, episodes)`, `load_episodes(path)`,
`views(batch, rng, config)`.

`Episode` and `Batch` are the two index conventions, deliberately named apart.

### Representation — Box 2

**`representation.py`** — `Encoder` (Type), `Decoder` (Type), `Projector` (Type),
`pack(z, n_spatial, k)`, `reconstruction_loss(pred, target, mask)`,
`representation_loss(online, target, projector, views, config)`,
`update_target(online, target, momentum)`.

`Encoder.forward` is all of `C* ∘ E*`: patch projection, MAE replacement, latent
tokens, backbone, bottleneck, `tanh`. `Decoder` is diagnostic-only after freeze.

### Backbone and time mixer — Box 3

**`backbone.py`** — `Layout` (Type: segment order, slices, modality ids),
`space_mask(layout, mode)`, `rope(x, positions)`, `Attention` (Type),
`SwiGLU` (Type), `Block` (Type), `Backbone` (Type).

`mode ∈ {encoder, decoder, dynamics}` is the only place the three masks differ.
`Block` is pre-RMSNorm → space → optional time → MLP, matching the source layer
exactly. `Backbone.forward(x, memory) -> (x, memory)`.

**`time_mixer.py`** — `TimeAttention` (Type), `TimeMamba` (Type),
`time_mixer(config)`.

Both types share `forward(x, memory) -> (y, memory)` and never mutate `memory`
in place. This module is the entire Mamba blast radius.

### Transition — Box 4

**`transition.py`** — `World` (Type), `flow_conditioning(rng, shape, config)`,
`observe(world, encoder, state, led_to_action, patches, rng, config)`,
`advance(world, state, led_to_action, rng, config)`,
`transition_loss(world, batch, rng, config)`.

`World.forward(state, led_to_action, latent, conditioning) -> (latent_out,
agent_out, state_out)` is the generic `evaluate`. `advance` is *N* read-only
candidate evaluations plus exactly one commit evaluation — flow N=4, direct N=1
per S17 — and always reads `h` from the commit pass. `observe` is the one place a real frame becomes `(e_t, z_t, m_t, h_t)`. It takes
`rng` because the flow arm corrupts the committed latent at τ_ctx while the direct
arm commits it clean; it exists because training,
imagination context construction and execution would otherwise each inline
encode-then-commit, which is the accretion this contract forbids.
`transition_loss` covers shortcut-flow and direct feature prediction, including
the rollout schedule, selected by `config.transition`.

### Agent — Box 5

**`agent.py`** — `Heads` (Type: policy, reward, continuation, value),
`twohot(values, centers)`, `head_targets(batch, config)`,
`head_loss(predictions, targets, config)`.

`value` exists from construction but enters no optimizer before Phase 3.
`head_targets` is where MTP lead alignment and the terminal/truncation split are
realised.

### Imagination and improvement — Boxes 6, 7

**`imagination.py`** — `Trajectory` (Type),
`imagine(world, heads, state, agent, rng, config)`.

The caller — `train_actor` — builds the starting state by repeated `observe`;
`imagine` receives it complete and owns no encoder. Box 6's "encode and scan the
context" describes the caller's work, and that boundary is where the
predecessor's context re-slice bug lived.

`Trajectory` carries **soft** `continuation` — the head's probability, not a
boolean — because imagination has no ground-truth termination. DreamerV3 passes
`self.con(inp, 2).prob(1)` and sets `term = 1 - con`, so the probability enters
the discount directly (`agent.py:206,404`). It carries no boundary mask: that
repo sets `last = jnp.zeros_like(con)` for imagination, and the horizon end is
the recursion's initial condition `R_T = v_T`, not a mask. Hard `terminated` and
`truncated` stay in `Batch`, where they are the continuation head's targets.

**`actor_critic.py`** — `lambda_returns(trajectory, config)`,
`actor_loss(logits, actions, returns, values, prior_logits, config)`,
`critic_loss(logits, returns, centers)`.

`actor_loss` is PMPO plus the reverse KL to the frozen prior.

### Execution — Box 8

**`execution.py`** — `Result` (Type),
`run_episode(world, encoder, heads, seed, config)`.

### Orchestration

**`train.py`** — `optimizer(modules, config)`, `train_representation(config)`,
`train_dynamics(config)`, `train_agent(config)`, `train_actor(config)`.

One driver per phase, in phase order. `optimizer` is the only place parameter
groups are built, and the only place upstream `_no_weight_decay` is honoured.

**`checkpoint.py`** — `save(path, config, **state)`, `load(path, config)`.

**`__main__.py`** — `main()`. **`__init__.py`** — exports only.

### Validation — the Stage-A gate list

**`gates.py`** — `alignment(config)`, `scan_step_parity(config)`,
`reset_parity(config)`, `firewall(config)`, `branch_nonmutation(config)`,
`recurrent_carry(config)`.

`scan_step_parity` covers two things, not one: scan versus recurrent step, **and**
teacher-forced windowed forward versus the equivalent `evaluate` reconstructed
from the prefix. Nothing else exercises `advance`, so without the second half the
most bug-prone function in the system ships untested until Stage-A results are
already contaminated.

`alignment` carries the Box-1 fixtures: length invariants, no-action-leak,
future-observation leakage (with MAE masking disabled or seeded), reward shift,
window-start action identity, and that the reward and continuation caused by
`a_t` are read at lead 0 of `h_{t+1}` (S22).

**`diagnostics.py`** — `multistep_error(world, batch, config)`,
`latent_stats(world, batch, config)`, `head_calibration(heads, agent, batch, config)`,
`cost(modules, config)`.

`latent_stats` reports range *and* scale, so S2's residual — contraction toward
the conditional mean — is measured rather than assumed away. `cost` reports
deployed parameters, training-only parameters, FLOPs, memory, throughput, and
`e_t`/`m_t` state sizes separately, per invariant 5, which is why it takes the
whole module set rather than the world alone.

## Totals

19 types, 49 public functions and 18 private helpers, across 20 modules.

## Signatures corrected during implementation

Each was forced by a contract already in this document, and each is a defect the
plan would otherwise have shipped.

| Signature | Why |
|---|---|
| `observe` takes and returns `RealState`, not `WorldState` | It has to carry the encoder's bounded-window memory, which `WorldState` does not hold. Merging them is how a rollout starts from zero memory while looking correct |
| `imagine` takes the starting `agent` readout | Recomputing it would ingest the same latent twice, against the rule that `m_t` already covers block `t` |
| `Trajectory` carries `agent` | The frozen prior and the critic must be evaluated where the actions were chosen, not at the first state alone |
| `diagnostics.cost` takes the module set and the world | It cannot separate deployed from training-only parameters given the world alone |
| `World.forward` returns memory, not a `WorldState` | `WorldState.latent` is defined as the *accepted* latent, which only a commit site can supply; a candidate has no accepted latent to put there |

`representation_loss` and `expert.train_expert` raise `NotImplementedError`
naming their open decision rather than guessing a default.

## S37 — `window` bounds state, not receptive field

Measured: with two encoder time layers at `window = 4`, perturbing frames more
than 4 back still moves `z` by 1.8e-3, and beyond 8 back moves it by **exactly
zero**. Influence travels one window per time layer, so the reach is their
product while each layer's cache stays bounded to `window`.

The state bound is the load-bearing half -- it is what makes the cache and the
deployed rollout produce the same latent for the same frame. `Config.receptive_field`
names the other half so no claim confuses them.

Also settled by execution: the tokenizer always mixes time with attention, in
every arm. Mamba's state summarises all history rather than a window, so a Mamba
encoder cannot honour the bound that makes `Z*` well defined -- and keeping the
encoder common is what confines the substitution to the dynamics.

## Verified defects, in fix order

Two independent audits; every entry below reproduced by execution before being
accepted. Nothing here reopens the architecture -- these are the implementation
failing to be what this document already says.

**Before any training run:**

| # | Defect | Evidence |
|---|---|---|
| 1 | Direct's transition loss is observation-free | Prediction bitwise identical for two different latent sequences. Fixed by S34 |
| 2 | Flow is diffusion forcing, not shortcut forcing | Eq. (7)'s bootstrap branch and its two half-step evaluations are absent, so the step token is inert and the arm is the paper's own ablation: Table 2 puts it at FVD 875 against 329 |
| 3 | `tanh` at inference but not in training | `_direct_candidate` squashes, `transition_loss` does not. A unit that learned 0.900 emits 0.716, decaying to 0.431 over six recursive steps toward a fixed point of 0 |
| 4 | Encoder window not enforced | `‖z(t=19 \| 20 frames) − z(t=19 \| 4)‖ = 1.7e-2`. The declared bound is documentation only |
| 5 | Phase 1A never writes the latent cache | `train_dynamics` fails on `batch.latents is None` |
| 6 | Multi-block decode against a cache is non-causal | New blocks attend bidirectionally; batched-vs-sequential differs by 9.2e-2. Attention corrupts silently, Mamba crashes -- divergent failure across the compared axis |
| 7 | Committed content and its label disagree | Mixes at 0.9 signal, labels bin 7/8 = 0.875 |
| 8 | No gate reaches `transition_loss` or `advance` | All six gates pass on the arm defeated by defect 1 |

**All fixed.** Plus, from the same pass: LPIPS at weight 0.2 with `p ~ U(0, 0.9)`
sampled per image; running-RMS loss balancing; the short/long batch curriculum;
seeded construction and device-matched generators; learned agent tokens;
diagnostics that commit before predicting.

Two things the fixes added rather than repaired. `transition.initial` commits a
known latent and returns the state it produces -- gates, diagnostics and Phase 3
all needed it, and inlining it three times is how the uncommitted-start defect
appeared in the first place. `data.unpatchify` is required by the perceptual
term. Both are in the inventory.

Still deferred, unchanged: `representation_loss` (Stage B) and
`expert.train_expert` (regeneration settings), each raising rather than guessing.
