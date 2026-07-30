# Decisions and function plan

Companion to `ARCHITECTURE.md`. That file says what the system is; this one says
what is settled, what is open, and exactly which functions exist.

The contract: **no new files, no new functions.** Anything not listed below is a
design change, not an implementation detail. If the plan is wrong, fix the plan.

## Settled

| # | Decision | Basis |
|---|---|---|
| S1 | Patch size 7, no resize, 81 tokens — one token per rendered tile | Craftax-Classic obs is 63×63 with `BLOCK_PIXEL_SIZE_AGENT = 7`; 63 = 9×7. Divisors of 63 are {1,3,7,9,21,63}, of which only {7,21} are tile-aligned and non-degenerate. 7 is the finest. `[R0-CHOICE]` — no genericity claim is made |
| S2 | `Z*` = D4 `tanh` bottleneck, **unnormalized**; direct readout is `tanh`-bounded | Keeps the flow anchor paper-faithful and Stage A genuinely fixed-representation. Matches the readout's codomain to its target's, which is the principle behind V-JEPA's `normalize_reps` for unbounded features. Bounds range only — **does not** address the scale contraction toward the conditional mean, which stays open as a failure mode |
| S3 | MTP `L = 8`, leads `n = 0..7`; policy lead 0 = outgoing `a_t`, reward lead 0 = incoming `r_t` | §3.3 states `L = 8`. Module defaults agree in both reproductions, but edwhu's eval script runs `L: int = 2` and `nicklashansen/dreamer4` has no MTP heads at all, so the paper is the basis and the reproductions only fail to contradict it. Eq. (9)'s inclusive `Σ_{n=0}^{L}` reads as nine terms and is recorded as ambiguous |
| S4 | Reward and value: 255 bins, symlog support ±20 | D4 §3.3 "Following Dreamer 3"; D3 `rewhead`/`value` `bins: 255` with centres `symexp(linspace(-20,0))` mirrored. Two-hot is linear interpolation between neighbouring centres, so bin count sets grid density, not a quantization floor; the wide support is miscalibration headroom, not wasted resolution |
| S5 | `time_every = 4` | D4 §3.4. Reproductions ship 2 and 1 — not inherited |
| S6 | λ-returns in next-index form, `G_t = r_{t+1} + γc_{t+1}[(1−λ)v_{t+1} + λG_{t+1}]` | Eq. (10) prints same-index reward, continuation and value, which is not self-consistent with §3.3's shared-index annotation. `dreamerv3/agent.py:487` indexes `rew[:, 1:]` and `boot[:, 1:]`, expanding to exactly this form |
| S7 | λ = 0.95 | `[DESIGN]`. D4 is silent; DreamerV3 `lam: 0.95` is implementation precedent only |
| S8 | Temporal state is a per-layer, per-slot pair; the conditioning slot is retained in every arm | `ARCHITECTURE.md` Boxes 3 and 4 |
| S9 | One generic `evaluate`; the A/B commit choice is a **`Config` field**, identical across the Stage-A cells | One signature covers every path; but if cells could differ in commit semantics the factorial is invalid |
| S10 | The signal table has exactly `k_max` rows, and τ_ctx must be a trained grid bin | Eq. (4)'s grid tops out at `1−d`, so τ = 1 is never trained. MMBench2 sizes `signal_embed` at `k_max+1` and labels uncorrupted context with index `k_max` — a row its own sampler cannot reach. Same pathology as the predecessor's unreachable shortcut rows, in the upstream source |
| S11 | Context corruption is applied once at commit, never at read | Fresh noise per read changes the prefix every frame, invalidating any KV cache or SSM state and contradicting invariant 5 |
| S12 | Agent slots exist from Phase 1B, last in the layout, masked both ways until Phase 2 | Fixes `S`, mask shape, stream count and state shapes once, so no Phase-1B gate is invalidated when the agent modality activates |
| S13 | `depth % time_every == 0` and at least two time layers | At `depth = 4` the entire Mamba blast radius would be one module and parameter matching would be dominated by it |
| S14 | Anchor drops GQA and attention-logit soft capping, keeps RoPE and QKNorm | Table 2 shows GQA at FVD 70 → 71, adopted for KV bandwidth at 2B parameters — no benefit at our scale. RoPE and QKNorm have executable precedent in pinned MMBench2 attention; GQA and capping have none anywhere. Anchor is described as paper-constrained *modulo declared omissions* |
| S15 | Windows never cross an episode boundary; `evaluate` takes no reset mask | Keeps a reset a fresh construction. Asserted by `gates.alignment`; relaxing it changes the signature |

## Open, with the functions each one constrains

Nothing here blocks writing the listed signatures. Each blocks a *body* or a
config value, and each must be closed before the phase named.

| Question | Close before | Constrains |
|---|---|---|
| **SIGReg on a projector vs on `z`** — projector keeps `Z*` and gives clean attribution; on `z` forks `Z*` from the anchor and tests the stronger claim | Stage B | `representation.Projector`, `representation.representation_loss` |
| **EMA views / masks / loss; faithful LeJEPA vs anti-collapse ablation** — is the EMA arm learning spatial invariance, temporal predictability, or both? | Stage B | `data.views`, `representation.representation_loss`, `representation.update_target` |
| **Flow commit A vs B** — eq. (4) samples τ over a grid ending at `1 − 1/d`, so **the paper itself never trains at τ = 1**; B's final-rung read (τ = 0.75 at K = 4) is therefore inside the trained conditions and A's clean commit is outside them. Against that, real `Observe` commits clean latents regardless, so under B the imagination and observation paths commit different conditions while A keeps them identical. Genuine trade-off, not source-decidable | Phase 3 | `transition.advance`, call site only |
| **τ_ctx value** — narrowed by S10/S11 to a trained grid bin applied at commit; which bin is still open. `1 − 1/k_max` is attractive (edwhu uses `k_max − 1` and its `k_max = 8` makes the noise fraction 0.125 ≈ 0.1) but rests on one repo's default | Phase 1B | `transition.flow_conditioning`, `transition.advance` |
| **Encoder temporal causality** — §3.1 makes the tokenizer causal, so `z_t = Z*(o_≤t)` and every arm needs burn-in or a declared bounded context. A frame-wise encoder would delete `e_t`, make `Z*` trivially cacheable and move all temporal modelling into the dynamics where the Mamba intervention lives; it is also the single largest D4 deviation in the design and removes the tokenizer's temporal compression | **Before any encoder code** | `representation.Encoder`, `state.RealState`, `data.sample_batch`, whether Phase 1B caches `Z*` targets |
| **Generated-prefix robustness and go/no-go thresholds** | Stage A exit | `diagnostics.multistep_error`; thresholds in `Config`, frozen before any arm is seen |
| **Capacity**: `n_latents`, `d_bottleneck`, `n_spatial`, `k`, `depth`, `d_model` — needs a 6 GB probe. Invariant `n_latents = n_spatial × k` is a consequence of D4's packing, not a law | Phase 1A | `Config` fields only |
| **Imagination horizon** — set from measured multi-step accuracy, not inherited | Phase 3 | `Config` field, gated by `diagnostics.multistep_error` |
| **Parameter/FLOP matching rule; official executed-control metric** | Stage A exit | `diagnostics.cost`, `execution.run_episode` |
| **Replay reuse vs regeneration** — the archived replay has no expert provenance and 58 distinct terminal windows. It can support debugging; it is weak as final evidence | Phase 1A | Whether `expert.train_expert` / `expert.collect` are populated, or the module is dead |

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
`advance(world, state, led_to_action, rng, config)`,
`transition_loss(world, batch, rng, config)`.

`World.forward(state, led_to_action, latent, conditioning) -> (latent_out,
agent_out, state_out)` is the generic `evaluate`. `advance` is one semantic
transition: K rungs for flow, one call for direct, and the A/B commit choice.
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
`imagine(world, heads, state, rng, config)`.

**`actor_critic.py`** — `lambda_returns(rewards, values, continues, config)`,
`actor_loss(logits, actions, returns, values, prior_logits, config)`,
`critic_loss(logits, returns, centers)`.

`actor_loss` is PMPO plus the reverse KL to the frozen prior.

### Execution — Box 8

**`execution.py`** — `Result` (Type), `run_episode(world, heads, seed, config)`.

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

`alignment` carries the Box-1 fixtures: length invariants, no-action-leak,
future-observation leakage (with MAE masking disabled or seeded), reward shift,
and window-start action identity.

**`diagnostics.py`** — `multistep_error(world, batch, config)`,
`latent_stats(world, batch, config)`, `head_calibration(heads, batch, config)`,
`cost(world, config)`.

`latent_stats` reports range *and* scale, so S2's residual — contraction toward
the conditional mean — is measured rather than assumed away. `cost` reports
deployed parameters, training-only parameters, FLOPs, memory, throughput, and
`e_t`/`m_t` state sizes separately, per invariant 5.

## Totals

19 types, 48 functions, 20 code modules. Every entry is required by a box; none
is a helper. Adding one re-opens this document.
