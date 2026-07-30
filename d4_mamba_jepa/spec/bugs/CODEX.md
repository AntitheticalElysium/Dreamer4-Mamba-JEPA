# CODEX — Architecture and source-fidelity audit

Audit date: 2026-07-30

Audited repository commit: `d26d360fb2d4aa343eaf915be49baa5c0d59eeac`

Primary target: `d4_mamba_jepa/spec/ARCHITECTURE.md`

Scope: current `d4_mamba_jepa/` implementation, its live
`craftax_jepa_config()`, cited code under `third_party/`, installed pinned
Craftax/Mamba sources, and the registered expert replay.

This is a read-only forensic review, not an architecture proposal. A statement
is marked unverified when the required source or lineage record is absent; no
missing fact is inferred from comments or filenames.

## Executive verdict

`ARCHITECTURE.md` is reliable for most live tensor shapes and broad dataflow,
but it is not a complete source-deviation ledger. Several lines for which
`diff` is omitted are materially different from their cited implementations,
and several source/provenance claims are false.

The highest-risk gaps are:

1. The expert replay cites `Craftax_Baselines@7ce36fa`, but that source is
   absent from `third_party`, and the artifact does not preserve the PPO
   training configuration needed to reconstruct its lineage.
2. The continuation and categorical BC heads are local implementations, not
   unchanged MMBench2 heads.
3. Mamba uses a non-default execution path, and the local optimizer applies
   weight decay to parameters that upstream marks as exempt.
4. The Dreamer-CDP predictor and I-JEPA/V-JEPA2 loss descriptions are
   inaccurate.
5. The statement that every checkpoint contains digest-pinned sources, strict
   atomic saving, and full RNG state is false.
6. The transition convention is wrong for replay windows sampled from the
   middle of an episode.
7. Several `untested` or `unexamined on Craftax` statements are stale relative
   to `ABLATIONS.md`.

The most urgent corrections are therefore components 2, 7–9, 11–13, and 16,
plus the transition convention and stale source manifest.

## Audit method and executable checks

- Read every substantive line of `spec/ARCHITECTURE.md`.
- Traced each component through the live local call path.
- Compared each claimed analogue to the pinned `third_party` checkout when
  present.
- Verified the registered MMBench2, Mamba, Gymnasium, Craftax, and LeJEPA
  digests through `source.py`.
- Verified the relevant source checkout commits.
- Verified the expert replay SHA-256 and manifest values.
- Instantiated both live arms from an equal seed and compared common state-dict
  tensors.
- Ran `.venv/bin/pytest -q d4_mamba_jepa/tests`.

Observed test result:

```text
98 passed, 7 skipped, 1 warning in 31.59s
```

The seven skips are the six module-level CUDA Mamba tests in
`tests/test_mamba_temporal.py` and the cached Mamba rollout test in
`tests/test_rollout.py`. The audited runtime reported:

```json
{
  "torch": "2.13.0+cu130",
  "cuda_available": false,
  "cuda_version": "13.0",
  "device": null
}
```

The installed Mamba source digest was independently verified against the pinned
official file, but the Mamba forward, cache, mixed-precision, and gradient paths
were not executable on this runtime. A source digest check is not a substitute
for those runtime tests.

## Component-by-component findings

### 1. Environment — ARCHITECTURE.md:22–26

Status: implementation verified; provenance incomplete.

Verified locally:

- `craftax_env.py:39-43` fixes the native size at 63, target size at 64, action
  count at 17, achievement count at 22, and environment name at
  `Craftax-Classic-Pixels-v1`.
- `craftax_env.py:85-111` converts native HWC float pixels to CHW uint8 and
  zero-pads only the bottom and right edges.
- `craftax_env.py:179-198` treats a timeout as a truncation with continuation
  1, and death/lava as absorbing with continuation 0.

Verified against installed Craftax 1.6.1:

- `craftax_classic/constants.py:14-20` gives the 7-by-9 map observation,
  two-row inventory display, and seven-pixel agent tiles. The renderer therefore
  produces a 9-by-9 tile image, or 63 by 63 pixels.
- `constants.py:47-64` defines action IDs 0 through 16.
- `constants.py:108-131` defines achievement IDs 0 through 21.
- `game_logic.py:5-13` terminates on timeout, lava, or death.
- `game_logic.py:1694-1700` computes reward as newly unlocked achievements plus
  0.1 times the health delta.

The continuation convention is a real analogue of Crafter
`third_party/sources/danijar__crafter/crafter/env.py:83-118`, where
`discount = 1 - dead`.

Deviation/provenance gap:

- There is no Craftax source checkout under `third_party`; the implementation is
  pinned through installed-package version and file digests instead.
- `spec/SOURCE_MANIFEST.md:16` still identifies `danijar/crafter` as the live
  environment and `SOURCE_MANIFEST.md:51-55` discusses installed Crafter rather
  than Craftax. The source manifest is stale after the Craftax migration.

### 2. Replay — ARCHITECTURE.md:28–32

Status: artifact values verified; source fidelity cannot be verified.

Verified:

- `artifacts/expert/craftax_expert_v1.pt.manifest.json` records:
  - 320 episodes;
  - 696,746 transitions;
  - replay SHA-256
    `7e5cdfc8b8cc813e0b51113f0c959c2c3ddcf3877a9ff0e1777ccfd7d4e0155b`;
  - parameters SHA-256
    `585f7e31b9f952c7fe40fb72456293d441c24e21707f0cac594123a387258638`;
  - stochastic rollout (`greedy=false`);
  - 252 capped episodes;
  - `max_steps=2500`.
- The file digest matches the registered replay digest.
- `data.py:184-201` computes the deterministic 80/10/10 whole-episode split.
  For 320 episodes this is 256 train, 32 development, and 32 sealed.
- Existing run reports record the exact split produced with seed 20260727.
- `data.py:215-277` hashes and loads the complete replay; it now refuses a
  capacity that would silently evict part of the dataset.

Corrections and gaps:

- The 80/10/10 split is a runtime pipeline property, not a property stored in
  the replay itself.
- No checkout of `MichaelTMatthews/Craftax_Baselines`, no `ppo_rnn.py`, and no
  matching `wrappers.py` exists under `third_party`. The repository therefore
  cannot verify the local expert implementation against the source named at
  `expert/ppo_expert.py:8-20` and in the replay manifest.
- The present local trainer contains a `DeathPenaltyWrapper` and defaults
  `DEATH_PENALTY` to 10 at `expert/ppo_expert.py:128-153` and `:273-295`.
  The replay manifest does not record whether the stored
  `ppo_expert_v2.msgpack` was trained with this default.
- The artifact also omits the PPO training budget, optimizer schedule,
  minibatching, layer size, reset ratio, reward-shaping configuration, and a
  trainer implementation digest. The parameters are hash-pinned, but their
  training lineage is not reconstructable.
- `diff episodes capped at max_steps=2500` is true but is not the only potential
  deviation from the unavailable cited source.

### 3. Sampler — ARCHITECTURE.md:34–38

Status: verified.

- `common.py:130-200` computes
  `round(batch_size * terminal_fraction)`, samples that many terminal episodes
  uniformly, forces their final valid window, and shuffles all rows.
- Remaining rows call `EpisodeReplay.sample`.
- `data.py:74-111` first samples an eligible episode uniformly and then samples
  a valid start uniformly within that episode. It is episode-uniform, not
  transition-uniform.
- The documented batch size, sequence length, and terminal fraction match the
  live runner.

No upstream sampler analogue is cited. This is a local replay contract.

### 4. Encoder — ARCHITECTURE.md:40–48

Status: live path verified; comparison needs narrowing.

- `model.py:190-223` instantiates the exact pinned MMBench2 `Encoder` and
  `Decoder` classes.
- `model.py:243` calls the tokenizer with `training_mask=False`.
- `model.py:193-208` consequently passes `mae_p_min=mae_p_max=0` to the
  encoder.
- Upstream `third_party/sources/nicklashansen__mmbench2/src/model.py:114-146`
  confirms that this takes the deterministic no-MAE path.
- Upstream `model.py:508-561` confirms the patch projection, latent
  concatenation, block-causal transformer, bottleneck projection, and `tanh`.
- The documented live shapes and geometry are correct.
- `model.py:325` removes the decoder in the JEPA arm, and
  `model.py:383-386` makes the online encoder trainable.

Corrections:

- `Encoder, unmodified` is true only at the imported class-source level. Its
  initialization, optimization, masking regime, and decoder-free JEPA objective
  are not the upstream training regime.
- Dreamer 4's 512 bottleneck tokens are the Minecraft paper configuration in
  `third_party/sources/edwhu__dreamer4-jax/docs/appendix.txt:14-15`.
  The available JAX code analogue defaults to 16 latents in
  `scripts/train_policy.py:107-120`. The document is comparing to a paper-scale
  configuration, not to the cited code default.
- The live default remains full encoder LR, but `ABLATIONS.md` now records
  Craftax tests of bottleneck width, latent count, and encoder learning rate.
  They are no longer an entirely unexamined surface.

### 5. Packing — ARCHITECTURE.md:50–55

Status: verified.

- `model.py:397-416` uses the upstream patchification and packing operations.
- `third_party/sources/nicklashansen__mmbench2/src/model.py:626-633` verifies
  the unchanged reshape.
- Sixteen 16-dimensional bottleneck tokens pack into four 64-dimensional
  spatial tokens, or 256 floats per frame.

### 6. Dynamics — ARCHITECTURE.md:57–63

Status: upstream class verified; local instance changes are incompletely stated.

- `model.py:247-261` instantiates the pinned MMBench2 `Dynamics` class with the
  documented live dimensions.
- `model.py:262-264` replaces its action encoder.
- `third_party/sources/nicklashansen__mmbench2/src/model.py:885-1029`
  confirms the action, shortcut, spatial, register, and agent-token layout.

Additional deviations/clarifications:

- `DiscreteActionEncoder` has `n_actions + 1 = 18` embeddings: one for each of
  the 17 real actions and one for `-1` start/unlabelled. Calling it simply
  `17-way` omits this dedicated state.
- Local dynamics is constructed with `lang_dim=0`, disabling the source's task
  projection and task-conditioned agent-token initialization.
- The shortcut step/signal embeddings and zero-initialized flow output head
  remain in the module. The live JEPA objective does not optimize a flow loss,
  and its predictor rollout does not use shortcut denoising.
- The class source is unchanged, but the resulting live system is not an
  unchanged MMBench2 dynamics training regime.

### 7. Temporal operator — ARCHITECTURE.md:65–71

Status: core replacement verified; two important deviations are omitted.

Verified:

- `temporal.py:113-135` replaces only dynamics layers whose time module is the
  exact upstream `TimeSelfAttention` type.
- The live depth/time cadence replaces exactly two dynamics temporal modules.
- The swap occurs after all shared JEPA modules are built.
- Equal-seed live construction produced 227 common state-dict entries and zero
  common-tensor mismatches.
- With `d_model=64`, `expand=1`, and `headdim=64`, official Mamba computes
  `d_inner=64` and one SSM head, as shown in
  `third_party/sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py:78-85`.
- Exact current parameter counts:
  - Transformer world: 986,348 total; 718,452 trainable.
  - Mamba world: 996,202 total; 728,306 trainable.
  - One dynamics attention module: 16,644.
  - One dynamics Mamba module: 21,571.
  - Difference: 4,927 per replaced module and 9,854 for the world.

Undocumented deviations:

1. `temporal.py:43-51` explicitly sets `use_mem_eff_path=False`. Official
   Mamba defaults it to `True` at
   `third_party/sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py:58-60`.
   This remains official Mamba code, but it selects a different execution path
   with potential performance and numerical consequences.
2. Official Mamba marks `dt_bias`, `A_log`, and `D` with
   `_no_weight_decay=True` at `mamba2.py:127-140`. The local world runner sends
   every trainable parameter to an undifferentiated AdamW group with
   `weight_decay=1e-2` at `craftax_runners.py:136-166`. The upstream
   no-weight-decay contract is ignored.

Runtime boundary:

- Source identity and construction verified on CPU.
- Official Mamba forward, recurrence/cache equivalence, mixed precision, and
  backward were skipped because CUDA was unavailable.

### 8. Predictor — ARCHITECTURE.md:73–79

Status: local shapes verified; cited source description is materially wrong.

Local implementation:

- `model.py:85-160` mean-pools two 64-dimensional agent tokens.
- It concatenates that 64-dimensional state with an explicit 64-dimensional
  next-action token.
- Its MLP maps 128 to 64 to 256 and reshapes the result to `[B,T,4,64]`.

Dreamer-CDP analogue:

- `third_party/sources/fmi-basel__Dreamer-CDP/dreamerv3/configs.yaml:91-100`
  configures an 8192-dimensional deterministic RSSM state.
- `dreamerv3/rssm.py:63-68` maps that state to a single flattened encoder
  representation.
- Action conditioning has already entered the RSSM core at `rssm.py:84-100`;
  the predictor does not concatenate an explicit next-action token.
- `rssm.py:140-143` computes cosine distance over the final flattened feature
  axis. It is not a per-spatial-token cosine loss.

The local predictor is therefore only broadly inspired by Dreamer-CDP's
two-layer predictor and stopped target. It is a different input state, action
interface, output representation, and rollout mechanism.

### 9. Self-prediction loss — ARCHITECTURE.md:81–91

Status: local operator verified; source correspondence is overstated.

Verified locally:

- `objectives.py:278-338` uses `K=jepa_jumps=5` and
  `context=sequence_length-K=11`.
- It autoregressively predicts exactly the final five real future states.
- Each predicted packed frame `[4,64]` is flattened to 256, projected to 64,
  L2-normalized, and compared by summed MSE.
- The target encoder and target projector are stop-gradient EMA copies.
- EMA retention ramps from 0.99 to 0.999 over the configured world schedule.
- Visual augmentation is absent.
- Self-prediction is the primary encoder objective, accompanied only by reward
  and continuation losses.

SPR comparison:

- `third_party/sources/mila-iqia__spr/src/models.py:287-305` verifies the
  normalized-MSE global loss.
- SPR default flags enable global and disable local SPR at
  `scripts/run.py:100-107`.
- However, SPR `jumps=5` produces t0 plus five transitioned predictions:
  six loss positions, as shown by `models.py:438-469` and its repeated
  `self.jumps + 1` reshapes. Local `jepa_jumps=5` means exactly five future
  positions and omits the t0 loss.
- SPR's default classifier is `q_l1` at `scripts/run.py:110`, not the optional
  MLP classifier whose shape the local `JepaProjector` copies.
- SPR's EMA helper is imported from external `rlpyt.models.utils`. That helper
  is not present in the pinned SPR checkout or local environment, so the claim
  that local parameter-and-buffer updates exactly match SPR cannot be verified
  from the repository. Local code EMA-updates parameters and directly copies
  BatchNorm buffers at `model.py:532-553`.

I-JEPA/V-JEPA2 correction:

- I-JEPA target features are feature-wise layer-normalized, then compared using
  Smooth-L1 at
  `third_party/sources/facebookresearch__ijepa/src/train.py:295-313`.
- V-JEPA2 target features are layer-normalized and compared by an Lp loss over
  masked tokens at
  `third_party/sources/facebookresearch__vjepa2/app/vjepa/train.py:429-450`.
- They do score masked token positions rather than one global frame vector, but
  they do not use the local SPR-style L2-normalize-then-cosine/MSE operator.
- I-JEPA's published EMA endpoints `[0.996,1.0]` are verified in
  `configs/in1k_vith16-448_ep300.yaml:42-44`.

SIGReg:

- The sliced-normality and Epps-Pulley source files are digest-pinned and
  imported unchanged.
- The surrounding use of SIGReg on action-predicted, projected Craftax tokens is
  a local integration, not an unchanged LeJEPA training algorithm.

### 10. Loss composition — ARCHITECTURE.md:93–97

Status: verified, subject to the Mamba optimizer deviation above.

- `training.py:125-190` implements:

  ```text
  jepa_weight * jepa
  + reward_weight * normalized_reward
  + continuation_weight * normalized_continuation
  ```

- `WorldLossNormalizer` registers only flow, reward, continuation, CDP, and
  reconstruction terms at `training.py:28-46`; JEPA is not normalized.
- `craftax_runners.py:95-248` verifies 20,000 updates, batch 8, LR `1e-4`,
  AdamW WD `1e-2`, gradient clip 1, warmup 1,000, and the optional separate
  encoder parameter group.
- There is no post-warmup decay; LR is constant after warmup.
- The optimizer does not honor Mamba's parameter-level no-WD markers.

### 11. Task heads — ARCHITECTURE.md:99–105

Status: routing verified; source attribution is false for continuation.

Reward:

- `model.py:266-275` instantiates the exact MMBench2 `RewardHeadMTP`.
- Upstream `third_party/sources/nicklashansen__mmbench2/src/model.py:736-807`
  verifies attention pooling, the MLP, MTP output, and symlog centers.
- Local overrides are horizon 8, 255 bins, and range `[-10,10]`.

Continuation:

- `model.py:62-82` defines a local `ContinuationHeadMTP`.
- It mean-pools agent tokens rather than using the reward head's attention pool.
- MMBench2 contains no continuation head.

Therefore `src MMBench2 MTP heads` is incorrect in the plural. Only the reward
head is an upstream MMBench2 head; continuation must be registered as a local
head.

Routing:

- `training.py:151-168` trains the heads from post-transition agent tokens of
  the five imagined JEPA positions.
- MTP targets are restricted to those positions. Earlier context positions can
  still receive gradients indirectly through the encoder/dynamics/predictor;
  the restriction refers to direct task-head inputs and targets.

### 12. BC policy — ARCHITECTURE.md:107–113

Status: major undeclared port and understated Dreamer-4 difference.

Local/source comparison:

- `common.py:78-107` copies the MMBench2 attention pool, MLP ratio, and small
  output initialization.
- Upstream `PolicyHeadMTP` at
  `third_party/sources/nicklashansen__mmbench2/src/model.py:810-882` outputs
  `L x act_dim_max` continuous, tanh-squashed action means.
- Local `BCPolicy` outputs one 17-way categorical distribution.

Replacing continuous eight-distance MTP output with single-distance categorical
classification is a material source deviation and is missing from
`ARCHITECTURE.md`.

Training details omitted from the knobs line:

- warmup 250;
- AdamW weight decay `1e-2`;
- gradient clip 1;
- cross-entropy of positions `[:-1]` against led-to actions `[1:]`.

Dreamer-4 JAX comparison:

- The source packs `dyn`, `task`, `pi`, and `rew` into the optimized parameter
  tree at
  `third_party/sources/edwhu__dreamer4-jax/scripts/train_bc_rew_heads.py:791-799`.
- Its loss combines shortcut, policy, and reward objectives at
  `train_bc_rew_heads.py:450-470`.
- The difference is not merely that source updates `p["dyn"]`: source jointly
  updates dynamics, task embedding, policy, and reward under the continuing
  world objective. Local Craftax BC freezes the entire world and trains only the
  categorical head.

### 13. Imagination — ARCHITECTURE.md:115–122

Status: live algorithm verified; comparison and cache wording need correction.

Verified local behavior:

- Actor is initialized as a copy of BC.
- Frozen BC serves as the behavioral prior.
- World, encoder, reward/continuation heads, and prior are frozen.
- One actor rollout is generated per replay context.
- PMPO balances positive/negative advantage partitions and adds reverse
  `KL(actor || BC)` at `imagination_actor_critic.py:294-342`.
- TD-lambda includes gamma 0.997, lambda 0.95, and predicted continuation at
  `imagination_actor_critic.py:259-291`.
- Value learning uses symexp-twohot targets.
- Live budgets are 500 updates, batch 64, context 8, and horizon 32.

Corrections:

- `imagine_trajectory` does not carry a recurrent Mamba cache. Nevertheless,
  generated latents are appended and the most recent eight states are rescanned
  at every step (`imagination_actor_critic.py:472-513`). The M arm therefore
  accumulates information through generated latent state inside a bounded
  window; it does not accumulate an unbounded recurrent hidden/cache state.
  `never accumulates state` is too strong.
- The cited JAX reproduction itself concatenates generated latents/actions,
  slices the latest context, and rescans dynamics at
  `third_party/sources/edwhu__dreamer4-jax/dreamer/imagination.py:367-424`.
  Sliding-window rescanning is not a deviation from that source code.
- Dreamer 4 uses context 192 for Minecraft, but 96 for SOAR and Epic Kitchens
  (`docs/appendix.txt:14-45`). The available JAX implementation defaults to
  context 16 (`scripts/train_policy.py:143-150`). The document must state which
  reference it compares against.
- Local value support is 255 bins over `[-10,10]`; the JAX analogue defaults to
  101 bins over `[-3,3]`.
- Local TD-lambda uses predicted continuation. The inspected JAX reproduction
  omits continuation from its return recursion; local comments correctly cite
  the paper rather than that code for this equation.
- No task tokens and no context corruption are genuine differences from the
  Dreamer-4 paper. The paper specifies context corruption at signal level 0.1
  in `docs/main.txt:259-261`.

### 14. Executed evaluation — ARCHITECTURE.md:124–126

Status: score and pairing verified; CI interpretation is inaccurate.

- `craftax_achievement.py:24-96` executes frozen random, BC, and actor policies
  directly in live Craftax and samples categorical policies at temperature 1.
- Environment seeds are shared across conditions.
- `craftax_achievement.py:99-116` resamples seed identifiers and recomputes the
  nonlinear aggregate score for both conditions on the same resampled seed
  list.
- `executed_control.py:60-86` matches the official Crafter formula at
  `third_party/sources/danijar__crafter/analysis/common.py:47-55`.

Correction:

- Policy RNG seed is a deterministic function of environment seed, with a
  separate offset for random. Resampling environment seed IDs therefore also
  resamples the associated policy-sampling streams.
- The interval covers variation across a fixed set of joint
  environment/policy-seed pairs. It does not isolate environment-seed variance,
  and it contains no training-seed uncertainty or alternative policy-sampling
  schedule uncertainty.
- `exploratory`, `one training seed`, and tier status are experiment-history
  statements. They conflict with the document's own rule that dated/result
  statements belong in `ABLATIONS.md`.

### 15. Oracle — ARCHITECTURE.md:128–131

Status: broad description accurate; uncertainty coverage is incomplete.

- Continuous targets use episode-disjoint train/validation/test splits.
- Constant, timestep, raw-pixel, and latent features receive independently
  validation-selected ridge probes.
- Nonlinear latent extraction uses an MLP.
- Nonlinear pixel extraction uses a CNN, not an MLP.
- `preserved` means the best latent probe is within the configured margin of
  the best pixel probe, after checking that the target is visible beyond the
  constant/timestep floor.
- The self-audit covers perfect, constant, random/misaligned, and one-step
  shifted relationships.
- `achievement_group` has no constant, timestep, or pixel baseline, as the
  architecture document acknowledges.

Additional limitation:

- `craftax_oracle.py:341-418` gives episode-bootstrap CIs only to the
  latent-linear continuous R².
- The nonlinear latent/pixel estimates have no CIs.
- `achievement_group` reports AUROC, average precision, and Brier without
  bootstrap CIs.
- Thus the module-level statement that every target is scored with an
  episode-level CI is not true for all reported probes/targets.

The oracle is a local diagnostic instrument and has no cited upstream
architectural analogue.

### 16. Provenance — ARCHITECTURE.md:133–136

Status: line 134 is materially false; line 135 is correct; line 136 is only
partly contextualized.

World checkpoints:

- `checkpoint.py:88-127` saves atomically.
- They store source report, implementation hash, world, normalizer, step, and
  config.
- Optimizer and RNG state are included only when an optimizer is supplied.
- Captured RNG consists of Torch CPU, Torch CUDA states when available, and the
  explicitly supplied NumPy generator.

Other artifacts:

- Tokenizer checkpoints are atomic and source-pinned but contain no optimizer or
  RNG state.
- BC and live Craftax actor checkpoints contain only format, paired world hash,
  head config, and head weights (`craftax_runners.py:80-93`).
- Craftax value checkpoints use direct non-atomic `torch.save` and contain only
  the value state dict (`craftax_runners.py:379-385`).
- `imagination_actor_critic.py` defines a provenance-checking combined
  actor/critic loader, but the live Craftax production runner does not write that
  format.

Implementation coverage:

- `checkpoint.py:22-37` hashes a fixed file allowlist.
- Active code omitted from that allowlist includes `craftax_env.py`,
  `craftax_achievement.py`, `craftax_oracle.py`, `oracle_metrics.py`,
  `expert/ppo_expert.py`, and `expert/generate.py`.
- A world checkpoint can therefore pass implementation provenance while these
  active environment/evaluation/data-generation files differ.

Source coverage:

- `source.py:263-283` includes MMBench2, Mamba2, and Gymnasium CartPole only.
- Craftax has a separate `craftax_source_report`.
- LeJEPA has a separate `lejepa_source_report`.
- SPR, I-JEPA, V-JEPA2, Dreamer-CDP, Dreamer 4, and DreamerV3 are not part of
  the world checkpoint's core `source_report`.
- `ARCHITECTURE.md:135` correctly reports this limitation.

Identifiers:

- `common.py:28` defines
  `POLICY_FORMAT = "d4_lite_cartpole_bc_policy_v1"`.
- `imagination_actor_critic.py:52-53` defines CartPole-named imagination and
  evaluation formats.
- Live Craftax actor files use the old BC policy format.
- The world format is now `d4_mamba_jepa_world_v1` and does not contain
  `cartpole`.
- The stricter imagination format is currently loader/test infrastructure, not
  the format written by the Craftax production runner.

## Cross-cutting document defects

### Inherited-value summary — ARCHITECTURE.md:140–147

The origin labels may remain historically correct for unchanged live defaults,
but the descriptions `untested surface`, `carried over unexamined`, and
`Craftax-native choices exist only at the boundaries` are stale.

`ABLATIONS.md` records Craftax evaluations of:

- `d_bottleneck` at row 4;
- `n_latents` at row 5;
- terminal sampling at rows 12–15;
- predictor context at row 14;
- encoder LR at rows 16–19;
- SIGReg versus EMA at row 18.

The document should distinguish:

1. where the live value originally came from;
2. whether that value has subsequently been tested on Craftax;
3. whether a Craftax result actually selected a replacement.

The present single `inherit` field conflates these questions.

### Transition convention — ARCHITECTURE.md:149–154

The general transition relation is correct:

```text
(obs_t, action_t) -> (obs_{t+1}, reward_{t+1}, continue_{t+1})
```

The statement `Position 0 gets the start action` is not correct for sampled
subsequences:

- `data.py:98-101` stores `episode.actions[start - 1]` at position zero whenever
  `start > 0`.
- `common.py:110-126` does the same for forced terminal windows.
- Only a window beginning at the true episode start receives `-1`.

Position zero's reward/continuation is always marked invalid because the
transition outcome lies outside the sampled window, even when the preceding
action is known.

Correct convention:

> Position zero gets `-1` only at a true episode start; otherwise it gets the
> preceding real action. Outcome validity at position zero remains false.

### “Not reproductions” — ARCHITECTURE.md:156–160

Correct:

- the live Craftax JEPA arm has no RSSM latent;
- it is offline rather than online joint world/control training;
- its live world objective has no pixel reconstruction or RSSM KL.

Too broad or internally contradictory:

- The live actor explicitly uses reverse `KL(actor || BC)` in component 13, so
  `KL terms ... absent` is false when applied to the whole live system.
- Transformer KV-cache and Mamba recurrent-cache code exist in `rollout.py` and
  `temporal.py`; the live JEPA imagination path simply does not use those
  caches.
- The package also contains optional CDP reconstruction-anchor and generative
  rollout paths, even though they are inactive in `craftax_jepa_config()`.

The section should be scoped explicitly to the live Craftax JEPA world-model
objective and live JEPA imagination path.

## Independent validation section

This section is for a second research agent. The validating agent must not treat
the conclusions above as evidence. It should independently read the local code
and source analogue for each claim, record exact paths and lines, and classify
each claim as:

- `CONFIRMED` — all material parts follow directly from inspected code/artifact;
- `PARTIAL` — some parts hold but the claim needs narrower wording;
- `REJECTED` — inspected evidence contradicts the claim;
- `UNVERIFIABLE` — required source, artifact, or lineage record is absent.

The validator should record the current commit, worktree status, source
checkout commits, environment, and test results before making decisions. It
must not modify training data, checkpoints, source pins, or experimental
artifacts during validation.

### Required validation preflight

Run from the repository root:

```bash
git rev-parse HEAD
git status --short
find d4_mamba_jepa/spec -maxdepth 2 -type f -print | sort
find third_party -maxdepth 3 \( -iname 'ppo_rnn.py' -o -iname 'wrappers.py' \) -print
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'
.venv/bin/python -c 'from d4_mamba_jepa.source import source_report, craftax_source_report, lejepa_source_report; print(source_report()); print(craftax_source_report()); print(lejepa_source_report())'
.venv/bin/pytest -q d4_mamba_jepa/tests
```

For each relevant source checkout, also record:

```bash
git -C <third_party/source/path> rev-parse HEAD
git -C <third_party/source/path> status --short
```

Untracked bytecode caches do not invalidate a pinned source file when its
registered digest still matches, but they must be reported separately from a
clean-checkout claim.

### Claim validation worksheet

| ID | Claim to validate independently | Minimum required evidence | Agent decision | Evidence / correction |
|---|---|---|---|---|
| C01 | Live Craftax observations are 63x63 padded to 64x64 CHW uint8; 17 actions and 22 achievements. | Local adapter plus installed pinned Craftax constants/renderer. |  |  |
| C02 | Timeout keeps continuation 1 while death/lava set it to 0. | Local `step`, Craftax termination source, Crafter discount analogue. |  |  |
| C03 | `SOURCE_MANIFEST.md` is stale because it still registers Crafter as the live environment. | Compare manifest to `craftax_env.py` and `source.py`. |  |  |
| C04 | Replay counts, hashes, cap, and stochastic rollout match the artifact manifest. | Recompute artifact/parameter hashes and inspect JSON. |  |  |
| C05 | The 80/10/10 split is computed by the runner and is not embedded in the replay artifact. | Inspect `whole_episode_splits`, runner, and serialized replay keys. |  |  |
| C06 | The cited `Craftax_Baselines@7ce36fa` source is absent from `third_party`, preventing a source diff. | Exhaustive filename/repository search under `third_party`. |  |  |
| C07 | Expert-policy training lineage is incomplete because the manifest omits PPO config and trainer digest. | Compare manifest schema to `ppo_expert.default_config()` and training code. |  |  |
| C08 | Sampler remainder is episode-uniform, not transition-uniform. | Trace `sample_sequences` into `EpisodeReplay.sample`. |  |  |
| C09 | MMBench2 encoder and packing classes/functions are imported unchanged, but their JEPA training regime is local. | Compare constructors and live objective to pinned MMBench2 training path. |  |  |
| C10 | Local dynamics uses an 18-entry action embedding and disables language/task projection. | Inspect `DiscreteActionEncoder`, live config, and upstream `Dynamics`. |  |  |
| C11 | Equal-seed Transformer/Mamba worlds share bit-identical common initialization. | Reinstantiate both arms and compare all intersecting state-dict tensors. |  |  |
| C12 | The Mamba arm is exactly 9,854 parameters larger at the audited config. | Count live total and per-temporal-module parameters. |  |  |
| C13 | Local Mamba explicitly disables upstream's default memory-efficient path. | Compare adapter constructor to official `Mamba2.__init__`. |  |  |
| C14 | Local AdamW applies weight decay to Mamba parameters marked `_no_weight_decay`. | Inspect optimizer groups and upstream parameter attributes. |  |  |
| C15 | CUDA-unavailable tests do not runtime-validate Mamba forward/cache/backward. | Inspect skip predicates and rerun with `-rs`; run on CUDA if available. |  |  |
| C16 | Local predictor is 128→64→256 with explicit next action; Dreamer-CDP is 8192→encoder-vector with indirect action conditioning. | Compare local predictor, CDP config, `_core`, `predictor`, and loss. |  |  |
| C17 | Dreamer-CDP's cited loss is global flattened-vector cosine, not per-spatial-token cosine. | Inspect tensor flattening and `cosine_distance(axis=-1)`. |  |  |
| C18 | Local `jepa_jumps=5` yields five future losses while SPR `jumps=5` yields t0 plus five transitions. | Trace both rollout loops and loss reshapes. |  |  |
| C19 | Local projector copies an optional SPR MLP shape, not SPR's default `q_l1` classifier. | Inspect SPR CLI defaults and classifier construction. |  |  |
| C20 | Exact SPR EMA buffer semantics are not verifiable because `rlpyt.models.utils.update_state_dict` is absent. | Locate the imported helper in checkout/environment or establish absence. |  |  |
| C21 | I-JEPA and V-JEPA2 use LayerNorm plus Smooth-L1/Lp, not SPR-style L2-normalized cosine/MSE. | Inspect pinned training loss functions. |  |  |
| C22 | JEPA loss is unnormalized while reward and continuation are RMS-normalized. | Trace `_jepa_world_loss` and `WorldLossNormalizer`. |  |  |
| C23 | Continuation is a local mean-pooled head; no MMBench2 continuation head exists. | Search pinned MMBench2 and inspect local class. |  |  |
| C24 | Local BC is single-step categorical, unlike MMBench2's continuous MTP head. | Compare output shapes, activation, and loss. |  |  |
| C25 | Dreamer-4 JAX BC jointly optimizes dynamics/task/policy/reward and retains shortcut/reward losses. | Inspect source parameter tree and combined loss. |  |  |
| C26 | Live JEPA imagination rescans a sliding eight-latent window without recurrent cache, but still carries generated latent state. | Trace `imagine_trajectory` and `_sample_next_jepa`. |  |  |
| C27 | The cited JAX reproduction also uses sliding-context rescanning. | Inspect its imagination scan body and context update. |  |  |
| C28 | Dreamer-4 context 192 is Minecraft-specific; real-world paper configs use 96 and JAX code defaults to 16. | Inspect appendix and `RLConfig`. |  |  |
| C29 | Local value support and continuation-aware return differ from the JAX reproduction. | Compare local value head/TD-lambda to JAX defaults and recursion. |  |  |
| C30 | Executed score matches official Crafter geometric-mean scoring. | Compare formulas and test with identical synthetic success rates. |  |  |
| C31 | Evaluation CI covers fixed joint environment/policy-seed pairs, not environment variance alone. | Trace policy-seed construction and bootstrap resampling unit. |  |  |
| C32 | Oracle bootstrap CIs cover only latent-linear continuous R², not nonlinear or achievement metrics. | Enumerate every `episode_bootstrap_ci` call and output field. |  |  |
| C33 | World checkpoints are atomic/source-bearing, but tokenizer, BC, actor, and value artifacts do not all contain full RNG/provenance. | Inspect every production writer and corresponding loader. |  |  |
| C34 | Live Craftax value saving is non-atomic and unprovenanced. | Inspect `train_craftax_imagination` output block. |  |  |
| C35 | `implementation_sha256` omits active environment/evaluation/oracle/expert files. | Compare allowlist to live call graph. |  |  |
| C36 | Core `source_report` omits Craftax and all JEPA/actor sources. | Execute and inspect all separate source-report functions. |  |  |
| C37 | CartPole-named policy identifiers remain, but the world format no longer contains CartPole. | Inspect constants and live production writers. |  |  |
| C38 | `untested/unexamined on Craftax` is stale for the knobs listed in current ablations. | Map `ABLATIONS.md` rows to architecture components and live values. |  |  |
| C39 | Position zero uses `-1` only at a true episode start; mid-episode windows use the preceding action. | Construct starts 0 and >0 through both sampler paths. |  |  |
| C40 | `KL terms and recurrent caches are absent` is too broad for the whole live/package system. | Locate actor KL, cache implementations, and active/inactive routes. |  |  |

### Validator output requirements

The validating agent should add a dated subsection below this heading, without
rewriting the CODEX findings above:

```markdown
## Independent validation — <agent/name>, <date>, <commit>

Environment:
- repository commit:
- worktree state:
- Python/PyTorch/CUDA:
- source checkout identities:
- tests:

Decisions:
- C01: CONFIRMED/PARTIAL/REJECTED/UNVERIFIABLE
  - evidence:
  - correction, if any:
...
- C40: ...

Overall:
- claims confirmed:
- claims partial:
- claims rejected:
- claims unverifiable:
- highest-risk disagreement:
```

Every `CONFIRMED` or `REJECTED` decision must cite executable code or an
artifact with exact path and line/key. Prose in this report, commit messages,
and prior review summaries are not independent evidence.

## Post-audit addendum — validation of `CLAUDE.md`

Validated independently on 2026-07-30 at the same repository commit. The full
claim-by-claim evidence is in `spec/bugs/VALIDATION.md`. This addendum is part
of the findings list and supersedes any conflicting statement in either earlier
audit.

### Corrections established by the cross-audit

- `CLAUDE.md` A1, A4, A5’s provenance gap, A6, A8, B1, and B4-B8 are
  materially confirmed.
- A2 is narrower than stated: JEPA task-head leads 5-7 have zero loss gradient,
  but their rows belong to dense output tensors and therefore move under AdamW
  weight decay. Base/CDP trains them, and live deployment reads lead 0.
- A3 is narrower than stated for the same reason: only step row 2 and signal row
  4 receive loss gradient, but the unreachable embedding rows decay rather than
  remaining byte-identical to initialization.
- A7 is not a simple train-at-16/deploy-at-8 mismatch. Causal BC training on a
  16-frame window trains context lengths 1-15, including 8. The real mismatch
  is the context-length distribution: executed control uses a sliding length 8
  after warm-up while 7/15 BC positions train with more than eight frames.
- B2 is a live-Craftax contextual warning, not a false calculation: the
  parameter-matching comment is correct for the dataclass’s 16/32 defaults but
  stale for the 64/64 Craftax override.
- B3 is an ambiguity, not a material contradiction; the specifically named
  policy/actor format identifiers do contain `cartpole`.
- **B9 is refuted.** Pinned LeJEPA `MINIMAL.md:172-174` contains exactly the
  convex invariance/SIGReg loss form. The local temporal-prediction use remains
  a substantial integration deviation, but absence of the loss form is false.
- `CLAUDE.md` section C is wrong that Dreamer-CDP uses per-spatial-token cosine.
  `dreamerv3/rssm.py:252-260` flattens the convolutional encoder output and
  `:140-143` compares one complete vector per batch/time item.
- Its claim of exact SPR EMA equivalence is unverified. SPR’s
  `rlpyt.models.utils.update_state_dict` implementation is absent from the
  pinned checkout and environment.
- Its CI wording is too narrow: evaluation bootstraps fixed joint
  environment/policy-seed outcomes, not environment variance alone.

### Added bug: JEPA imagination discards Mamba recurrent state

This implementation fact is confirmed and already visible in the architecture
at `ARCHITECTURE.md:121`; its consequences needed stronger classification.

- `imagine_trajectory` passes `use_cache=True`, but the JEPA early return in
  `rollout.py:109-116` ignores that argument.
- `_sample_next_jepa` returns only a generated latent and an agent token; it
  neither accepts nor returns the `MambaTemporalState` defined in
  `temporal.py:13-21`.
- `imagination_actor_critic.py:512-513` retains only the newest eight latent
  states/actions. Each uncached Mamba call reaches `self.mamba(flat)` at
  `temporal.py:109`, starting both temporal SSMs from fresh state and rescanning
  only that finite window.
- Generated latents remain in the window, so the M arm is not memoryless. It
  has eight-state latent memory but no SSM carry or information older than the
  window across the 32-step horizon.
- World training rescans contexts of length 11-15, with post-transition passes
  up to 16, whereas actor imagination is capped at 8. This is an undocumented
  train/deployment context regime change for both backends and prevents the
  experiment from testing Mamba’s long-memory advantage.

The proposed causal interpretation remains **ambiguous**:

- Existing results do show full/slow M actor-minus-BC values of `-0.656` and
  `-0.262`, versus `-0.124` and `+2.042` for T.
- Every completed T/M comparison is `INIT* 1SEED`; the fixed-initialization run
  was aborted at T 3k/20k (`ABLATIONS.md:10-12,16,32-34`).
- The state-loss mechanism cannot explain M’s higher BC score by itself because
  BC precedes imagination and both executed BC policies use the same
  sliding-eight protocol. It can plausibly impair M’s later actor training.

Do not attribute the T/M difference to recurrence until a fixed-checkpoint,
fixed-context, fixed-action/RNG ablation compares current sliding-eight rescan,
growing full context, persistent Mamba state, and a matched Transformer
history/cache control.

### Additional bugs and hidden defaults found after both audits

1. **Generic JEPA optimizer crash.** `objectives.optimizer_groups()` calls
   `world.decoder.parameters()` unconditionally at `objectives.py:448`, but a
   JEPA world sets `decoder=None`. Reproduced:
   `AttributeError: 'NoneType' object has no attribute 'parameters'`.
   The Craftax production runner bypasses this helper, so existing runs are not
   affected.

2. **Expert `env_seed` is not replayable.** `expert/generate.py:120-123`
   generates reset keys by splitting a batch key, then `:193` records an
   arithmetic scalar seed that does not reconstruct that key. The replay hash
   fixes the data, but the per-episode seed field does not reproduce its
   trajectory.

3. **Timeout/death collision is classified as truncation.** Pinned Craftax done
   is timeout OR lava OR death. `craftax_env.py:185-188` defines absorbing
   terminal as `done and not timeout`, so death/lava on the exact native timeout
   transition receives continuation 1. `expert/generate.py` has the same
   timestep-only inference. Current 2,500-step caps cannot reach the native
   10,000-step timeout, so audited artifacts are unaffected.

4. **Core provenance has irrelevant hard dependencies.** Every checkpoint calls
   unconditional `source_report()`, which verifies installed Mamba and
   Gymnasium CartPole even for Transformer-only Craftax checkpoints, while
   omitting active Craftax and JEPA sources. Source reporting should be
   config-conditional.

5. **LeJEPA verifies less than it imports.** The loader hashes three files but
   imports through package paths whose `__init__.py` files import many
   additional modules. Those transitive files are outside
   `lejepa_source_report()`. The checkout is currently at the pinned commit, but
   the claimed runtime digest closure is incomplete.

6. **Expert post-done masking is documentation only.**
   `expert/generate.py:7-10` says completed vector slots are masked, but the scan
   still steps them; `done_prev` only resets the RNN input. First-done slicing
   preserves stored episodes, so this is not current replay corruption.
   `truncated_episodes` also excludes native timeouts reached inside the cap,
   though that edge is unreachable in the current 2,500-step artifact.

### Updated highest-risk order

1. Recurrent-state/context semantics plus the incomplete fixed-init T/M control.
2. Missing expert/source provenance and non-replayable episode seed lineage.
3. Unreachable JEPA modules/output rows and Mamba-specific AdamW decay.
4. Generic JEPA optimizer failure and checkpoint source-report composition.
5. Timeout and vector-rollout edge cases, which do not explain current results.
