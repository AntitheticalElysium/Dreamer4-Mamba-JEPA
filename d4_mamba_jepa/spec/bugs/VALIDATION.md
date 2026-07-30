# Independent validation of `CLAUDE.md`

Audit date: 2026-07-30
Repository: `craftax-clean-baseline` at
`d26d360fb2d4aa343eaf915be49baa5c0d59eeac`
Validator: Codex

This is an independent source-and-runtime audit of every A/B finding in
`CLAUDE.md`, plus the material claims in its “verified accurate” section. A
`CONFIRMED` verdict means the material claim follows from current code, a pinned
source, or a recomputed artifact. `PARTIAL` means that the mechanism exists but
the wording or consequence is too broad. `REFUTED` means the inspected source
contradicts the claim. `UNVERIFIABLE` is reserved for a source comparison whose
required bytes are absent.

No model, replay, checkpoint, source pin, or experiment artifact was modified.

## Preflight

- Python 3.12.12, PyTorch 2.13.0+cu130, CUDA unavailable.
- Current test result:
  `98 passed, 7 skipped, 1 warning in 35.70s`.
- The seven skips are the CUDA-dependent Mamba tests. This audit verifies the
  Mamba routing statically but does not claim a current CUDA recurrence test.
- Relevant source heads:
  - MMBench2 `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`
  - Mamba `f577286d052741c35d39cd43bdc3fad27120f22c`
  - SPR `0b9dd4e7b9bbdfaecdf9a3713bf5931fb54ab0ca`
  - Dreamer-CDP `a851fa3e3d70b624b094ee1810ad4bb602346092`
  - DreamerV3 `e3f02248693a79dc8b0ebd62c93683888ddaccfe`
  - dreamer4-jax `8144b940d801971f12ec5633553b95001e555949`
  - LeJEPA `c293d291ca87cd4fddee9d3fffe4e914c7272052`
- Those checkouts match the registered commits. The LeJEPA checkout contains
  untracked Python bytecode caches; no tracked source modification was found.
- The root worktree already contained untracked experiment reports and both
  audit files. They were treated as evidence, not rewritten artifacts.

## A claims

| ID | Verdict | Deciding evidence | Correction or boundary |
|---|---|---|---|
| A1 | **CONFIRMED** | Upstream zero-initializes `flow_x_head` in `third_party/sources/nicklashansen__mmbench2/src/model.py:964-966`. The live JEPA loss never uses its spatial output: `training.py:125-190`, `objectives.py:278-374`, and default `CDPPredictor` pooling at `model.py:141-143`. A fresh live backward gave `grad is None` for both flow-head tensors and `encoder.mae.mask_token`; the spatial output was exactly zero. | 4,160 flow-head and 64 mask-token parameters are truly immobile even under AdamW because their gradients are `None`. This is not merely a small gradient. |
| A2 | **PARTIAL** | `training.py:151-168` feeds task heads only the five `rollout_agents`; `_mtp_scalar_targets` and `continuation_mtp_loss` taper supervision by lead. Recomputed gradient mass was nonzero for leads 0-4 and exactly zero for 5-7, with effective position counts 5:4:3:2:1. | “Never receive loss gradient” is correct for the live JEPA arm. “Never move” or “ship as noise” is too broad: the complete output tensors have dense gradient objects, so AdamW weight decay changes the zero-gradient rows. Inactive reward rows remain near their zero-reward bias prior; inactive continuation rows decay toward zero logits. Base/CDP with `T=16` trains all eight leads, and deployment reads lead 0 only. |
| A3 | **PARTIAL** | Every live JEPA pass supplies `max_step_index` and `k_max`; see `objectives.py:304-308`, `rollout.py:71-75`, and `imagination_actor_critic.py:447-464`. Recomputed row gradients were nonzero only for step row 2 and signal row 4. | Six rows are functionally unreachable, but they do not stay byte-identical to initialization: embedding gradients are dense tensors and AdamW decays the unused rows. The two reachable rows are constant conditioning vectors, not literally one parameter vector. |
| A4 | **CONFIRMED** | Local heads are `256→64→64` and `64→64→64` at `model.py:163-180,352-356`. SPR’s optional global MLP uses hidden 512/output 256 and its optional final MLP uses hidden `2s`; `third_party/sources/mila-iqia__spr/src/models.py:210-215,237-247`. SPR CLI defaults are `q_l1` plus a linear final classifier at `scripts/run.py:110-111`. | This is a source-default deviation, not proof that the local smaller MLP is harmful. |
| A5 | **CONFIRMED** for the provenance failure; **UNVERIFIABLE** for fidelity | An exhaustive repository/`third_party` search found no `Craftax_Baselines` checkout or `ppo_rnn.py`; `third_party/SOURCES.lock` has no entry. `expert/ppo_expert.py:1-16` describes a reimplementation and its substitutions. | The string `Craftax_Baselines@7ce36fa` is not byte-level provenance. Without the cited bytes, the reimplementation cannot be source-diffed and should not be called faithful. |
| A6 | **CONFIRMED** | Recomputed from the hash-pinned replay: 320 episodes, 696,746 transitions, 68 terminal episodes overall, 58 in the train split, and 252 cap-truncated episodes. `common.py:138-166` selects four terminal rows per eight-row batch, always from the last 16-frame window. At 20,000 updates that is exactly 80,000 draws from 58 possible windows. | `80,000/58 = 1,379.31` is the exact mean draw count per support element, not a claim that every window is sampled equally often. |
| A7 | **PARTIAL** | Executed evaluation slices to the latest eight frames at `craftax_achievement.py:29-41`. BC uses `logits[:, :-1]` against `led_to_actions[:, 1:]` at `craftax_runners.py:294-300`. Because dynamics is causal, one 16-frame BC batch trains decisions at context lengths 1 through 15, including length 8. | “BC was trained at 16” is false as a categorical statement. There is still a distribution mismatch: after warm-up, execution always uses a sliding length-8 window, whereas BC training gives equal positional weight to lengths 1-15 (8/15 positions use at most eight frames; 7/15 use more). |
| A8 | **CONFIRMED** | `checkpoint.py:22-38` hashes a fixed core allowlist and omits `craftax_env.py`, `craftax_data.py`, `craftax_achievement.py`, `craftax_oracle.py`, `oracle_metrics.py`, `craftax_resolution.py`, `executed_control.py`, and `expert/`. | These files can change without invalidating an existing world checkpoint’s implementation digest. |

## B claims

| ID | Verdict | Deciding evidence | Correction or boundary |
|---|---|---|---|
| B1 | **CONFIRMED** | Dreamer-CDP’s default configuration has `deter: 8192` at `third_party/sources/fmi-basel__Dreamer-CDP/dreamerv3/configs.yaml:93`; 4096 is a size preset at line 140. Local `model.py:107-109` says 4096 and mislabels `rssm.py:140`. | The live command documented by Dreamer-CDP uses the 8192 default. |
| B2 | **PARTIAL** | Current Craftax counts reproduce exactly: Transformer world 986,348; Mamba world 996,202; attention module 16,644; Mamba module 21,571. `craftax_runners.py:64-67` overrides Mamba state/head dimension to 64/64. | The `config.py:52-57` arithmetic is correct for the dataclass’s 16/32 defaults. It is stale only when read as rationale for the live Craftax override; it is not mathematically false in its stated default context. |
| B3 | **PARTIAL**, clarity only | `common.POLICY_FORMAT` and `imagination_actor_critic.FORMAT`/`EVALUATION_FORMAT` contain `cartpole`; the world/checkpoint/Craftax data formats do not. | `ARCHITECTURE.md:136` names the particular symbols and does not explicitly say every variable named `FORMAT` is CartPole-named. The sentence is ambiguous, not a material code contradiction. |
| B4 | **CONFIRMED** | DreamerV3 reward `bins: 255` is at `third_party/sources/danijar__dreamerv3/dreamerv3/configs.yaml:98`; line 101 is the value head. | The number is correct; only the reward citation is wrong. |
| B5 | **CONFIRMED** | SPR local/global defaults are in `third_party/sources/mila-iqia__spr/scripts/run.py:106-107`. | Path precision error only. |
| B6 | **CONFIRMED** | The oracle self-audit includes a shifted-label test at `craftax_oracle.py:447-478`. The nonlinear pixel ceiling is `_cnn_predict`, while the latent nonlinear ceiling is `_mlp_predict`, at `craftax_oracle.py:385-391`. | The architecture text’s “ridge + MLP probes” is incomplete. |
| B7 | **CONFIRMED** | `SOURCE_MANIFEST.md` still registers Crafter and old cross-track files and contains no Craftax, SPR, LeJEPA, or Craftax-Baselines row. | It is stale provenance documentation. |
| B8 | **CONFIRMED** | The JEPA branch returns at `rollout.py:109-116`; the following triple-quoted expression at `:117-122` is unreachable and cannot be the function docstring. | Code-quality/documentation defect; no runtime semantic effect. |
| B9 | **REFUTED** | The pinned and tracked `third_party/sources/rbalestr-lab__lejepa/MINIMAL.md:172-174` explicitly computes invariance loss, SIGReg loss, and `lambda*sigreg + (1-lambda)*invariance`. | The missing `scripts/je.py` prevents a diff against that launcher implementation, but not verification of the claimed loss form. Also, 1024 slices is in the pinned README/ablations, 64 projector dimensions is in the projection ablation, and lambda 0.05 is in upstream sweeps. Treating 1000/512 as the one authoritative default is unjustified. The local temporal-prediction integration still differs substantially from LeJEPA’s multi-view invariance setup. |

Summary of A/B decisions:

- confirmed: A1, A4, A5 provenance, A6, A8, B1, B4-B8;
- partial: A2, A3, A7, B2, B3;
- refuted: B9;
- unverifiable: byte-level Craftax-Baselines fidelity in A5.

## Material corrections to `CLAUDE.md` section C

1. **Dreamer-CDP is not per-spatial-token cosine.** Its encoder flattens the
   convolutional output to one vector per batch/time item at
   `third_party/sources/fmi-basel__Dreamer-CDP/dreamerv3/rssm.py:252-260`.
   Its predictor produces one matching encoder vector and cosine distance is
   taken across that final feature axis at `rssm.py:140-143`. Section C’s
   “per-token cosine confirmed” statement is refuted.

2. **Exact SPR EMA equivalence is not source-verified.** SPR imports
   `update_state_dict` from `rlpyt.models.utils` at
   `third_party/sources/mila-iqia__spr/src/models.py:5` and calls it at
   `:354-366`. The helper is absent from both the pinned checkout and the
   current environment. The reciprocal interpretation of `momentum_tau=0.01`
   is plausible but not independently provable from the available source.

3. **The confidence interval does not isolate environment randomness.**
   `craftax_achievement.py:128-133` deterministically derives a policy RNG seed
   from each environment seed. Bootstrap resampling therefore resamples fixed
   joint environment/policy-seed outcomes. It contains neither alternative
   policy-sampling draws nor training-seed uncertainty.

4. **The MTP and shortcut rows need AdamW qualification.** Zero loss gradient
   is not equivalent to frozen parameters when the containing tensor receives a
   dense gradient. The inactive MTP/embedding rows decay; only parameters with
   `grad is None` (A1) are completely skipped by AdamW.

Section C component verdicts after rechecking its remaining statements:

| CLAUDE section | Verdict | Boundary |
|---|---|---|
| C §1 environment | **PARTIAL** | Dimensions, actions, achievements, reward, ordinary timeout, death, and lava behavior are confirmed. Simultaneous timeout plus death/lava is misclassified locally; see N3. |
| C §2 replay | **CONFIRMED** | Counts, cap, stochastic policy, and whole-episode split were checked against the artifact and runner. |
| C §3 sampler | **CONFIRMED** | Forced-terminal and episode-uniform remainder mechanisms match code; A6 adds the exact support size. |
| C §4/§5 encoder and packing | **CONFIRMED** | Constructor, MAE-off path, shapes, tanh bottleneck, and 16→4 packing match the pinned MMBench2 implementation. |
| C §6 dynamics | **CONFIRMED with registered local-instance deviations** | Token shapes and categorical action slot are correct. The 18th start embedding, `lang_dim=0`, dead flow output, and local training regime remain deviations. |
| C §7 temporal operator | **CONFIRMED** | Live dimensions, parameter counts, one Mamba head, late swap, and shared initialization were independently reproduced. CUDA execution remains untested in this runtime. |
| C §8 predictor | **PARTIAL** | Local predictor shape is correct; the Dreamer-CDP “per-token” characterization is refuted. |
| C §9 self-prediction | **PARTIAL** | Local normalized MSE, global flattening, source loss shapes, and I-/V-JEPA comparisons are confirmed. Exact SPR EMA semantics are unverified. |
| C §10 loss composition | **CONFIRMED** | JEPA is unnormalized; reward and continuation are RMS-normalized; optimizer and schedule values match the runner. |
| C §11 task heads | **PARTIAL** | Shapes and source bins match. A2 narrows the effective JEPA MTP horizon and AdamW behavior. |
| C §12 BC policy | **CONFIRMED with source deviations** | Head architecture and gradient isolation match; it is a local single-step categorical port, not upstream continuous MTP behavior. |
| C §13 imagination | **CONFIRMED** | Budgets, return/PMPO parameters, sliding-eight rescan, generated-latent carry, and absent recurrent state match code. |
| C §14 executed evaluation | **PARTIAL** | Score and seed pairing are correct; “environment-seed variance only” is too narrow because policy RNG is coupled to environment seed. |
| C §15 oracle | **CONFIRMED** | Score formula, oracle ceilings, and missing achievement controls match code, with B6’s omitted shifted/CNN details. |
| C §16 provenance | **CONFIRMED with N4/N5 additions** | Core report keys and omissions match. The unconditional irrelevant dependencies and incomplete LeJEPA import closure were missed. |

## The Mamba recurrence question

### Confirmed implementation fact

The live JEPA imagination actor does not preserve Mamba recurrent state.

- `imagine_trajectory` calls `sample_next_packed(..., use_cache=True)` at
  `imagination_actor_critic.py:490-497`.
- The JEPA early return at `rollout.py:109-116` ignores `use_cache`, the
  denoising schedule, and the generator. `_sample_next_jepa` calls
  `world.forward_dynamics` on a complete finite sequence and returns only a
  latent and agent token. It never returns a `MambaTemporalState`.
- `imagination_actor_critic.py:512-513` truncates latent and action history to the
  newest eight positions after every imagined step.
- With no cache, `MambaTimeMixer.forward` calls `self.mamba(flat)` at
  `temporal.py:109`. Official Mamba initializes `conv_state` and `ssm_state` to
  `None` on each such forward (`mamba2.py:169-176`), so both temporal Mamba
  layers rescan the current eight-step window from fresh zero state.

The generated latent is fed back into the window, so M-JEPA is not memoryless.
It has an eight-latent finite history. What is absent is state older than that
window and any continuous SSM carry across the 32 imagined transitions.

Training is also a different context regime. With `sequence_length=16` and
`jepa_jumps=5`, world self-prediction begins with 11 real states and rescans
sequences of length 11 through 15 (plus post-transition passes up to 16) at
`objectives.py:278-335`. Actor imagination begins at eight and remains capped
at eight. This is an unregistered train/deployment context deviation for both
backends and specifically prevents testing Mamba’s long-memory thesis.

### Ambiguous causal claim

The missing carry is a credible explanation for M-JEPA failing to improve its
BC, but the current results do not establish that causal link.

- The observed exploratory scores do match the user’s pattern:

  | arm | BC | actor | actor minus BC, paired 95% CI |
  |---|---:|---:|---:|
  | full M | 3.604 | 2.948 | -0.656 [-1.316, 0.120] |
  | full T | 2.606 | 2.481 | -0.124 [-0.862, 0.456] |
  | slow M | 3.554 | 3.292 | -0.262 [-1.177, 0.626] |
  | slow T | 2.353 | 4.395 | +2.042 [0.960, 2.623] |

- All completed T/M rows are marked `INIT* 1SEED` in
  `spec/ABLATIONS.md:10-12,16,32-33`. The pre-fix constructor consumed RNG in
  backend-specific order, producing 16 mismatched shared JEPA tensors, and
  phase heads were also not cleanly matched.
- The fixed-initialization T/M rerun was aborted at T world step 3,000/20,000
  (`ABLATIONS.md:34`), so no completed matched result exists.
- Missing recurrent carry cannot explain why the M **BC** itself scores higher:
  BC is trained before imagination and both executed BC policies use the same
  finite sliding-eight evaluation protocol. It can plausibly affect the later
  M actor update because its imagined horizon cannot use older SSM state.

Therefore:

- **confirmed bug/invalidated thesis test:** Mamba long recurrence is not
  exercised during live actor imagination;
- **not confirmed:** that this mechanism caused the T/M actor-vs-BC difference.

A discriminating ablation must hold checkpoint, replay contexts, actions, and
policy RNG fixed while comparing:

1. current sliding-eight full rescan;
2. growing-context full rescan for both T and M;
3. persistent Mamba state with an explicitly defined reset policy;
4. a matched Transformer cache/history control.

Persistent state and sliding-window semantics are different models. Once old
states are dropped, an exact state representing only the retained window cannot
be recovered by subtracting their contribution; the defined choices are to
keep unbounded state or reset and rescan the retained window.

## Findings missed by `CLAUDE.md`

### N1 — generic JEPA optimizer helper crashes

`objectives.optimizer_groups()` unconditionally executes
`world.decoder.parameters()` at `objectives.py:448`. A JEPA world deliberately
sets `decoder=None` at `model.py:328`. Reproduced on the live Transformer JEPA
config:

```text
AttributeError: 'NoneType' object has no attribute 'parameters'
```

The production Craftax JEPA runner builds its own optimizer, so current runs do
not hit this. The public core helper and any generic checkpoint/training path do.

### N2 — expert record `env_seed` is not the reset key that generated the episode

`expert/generate.py:120-123` derives vector reset keys by splitting a
batch-level key, but `:193` records `seed + batch*num_envs + slot` as
`env_seed`. That integer does not recreate the split key through
`CraftaxPixelEnv(seed=env_seed)`. For seed 0/slot 0, the actual reset key is
`[346279018, 360566543]`; the reset key derived from recorded `env_seed=0` is
`[928981903, 3453687069]`.

The replay bytes remain fixed by their file hash, but per-episode environment
reproduction from the serialized lineage field is impossible. The manifest
must either store the actual JAX keys or label this field as a non-replayable
ordinal.

### N3 — simultaneous timeout and death/lava is misclassified

Pinned Craftax defines done as timeout OR lava OR death at
installed `craftax_classic/game_logic.py:5-13`. The local adapter uses
`terminal = done and not timeout` at `craftax_env.py:185-188`. If death or lava
occurs on the exact timeout transition, timeout wins and continuation is set to
1, contrary to the stated `1-dead` contract. `expert/generate.py:173-180` uses
the same timestep-only inference.

The current expert and evaluation caps are 2,500, below Craftax’s 10,000-step
timeout, so this does not affect the audited artifacts. It is a real latent
adapter bug for runs reaching the native horizon.

### N4 — core checkpoint provenance imposes irrelevant dependencies

Every world/tokenizer save and load calls the unconditional `source_report()`
at `checkpoint.py:104-111,141-147,172-175,218-222`.
`source_report()` always imports/verifies installed Mamba and Gymnasium
CartPole at `source.py:263-283`, including for a Transformer Craftax-only
checkpoint, while omitting the active Craftax and JEPA sources.

Consequences:

- a Transformer-only Craftax run cannot save/load without an installed,
  byte-matching Mamba package and CartPole;
- the provenance gate can fail on irrelevant source drift;
- it still does not fail on active Craftax/SPR/LeJEPA drift.

This is both a coverage gap and a source-report composition bug. Reports should
be conditional on the checkpoint’s active config.

### N5 — LeJEPA runtime import exceeds its verified file set

`source.load_lejepa_sigreg()` verifies only `slicing.py`,
`epps_pulley.py`, and `univariate/base.py`, then imports through Python package
names. Importing `lejepa.multivariate.slicing` and
`lejepa.univariate.epps_pulley` first executes both package `__init__.py` files,
which import many additional modules. Those transitive files and the
`__init__.py` files are not digest-checked by `lejepa_source_report()`.

The checkout commit currently matches, so no live drift was found. The
checkpoint-level claim “digest-pinned LeJEPA implementation” is nevertheless
too broad; either pin the transitive import closure or load the verified files
in isolation.

### N6 — the expert rollout says “masked after done” but is not masked

`expert/generate.py:7-10` says completed vector slots keep stepping “masked”.
The scan at `:124-139` forwards `done_prev` only to the RNN reset input; it
still steps every environment and records its outputs. Slicing at the first
`done` (`:169-181`) keeps the saved episode boundary correct, so this is not
artifact corruption. It is a documentation/compute-path discrepancy.

Also, `truncated_episodes` increments only when no `done` occurs within
`max_steps`; a native timeout reached inside the cap is not counted. The
current 2,500-step artifact cannot reach the 10,000-step native timeout, so its
reported 252 count is unaffected.

## Priority

1. Treat current T/M control conclusions as unidentifiable until the recurrent
   context axis and fixed initialization are both controlled.
2. Register the inactive JEPA parameters and either remove them or exclude
   their unreachable rows/modules from optimization.
3. Fix source composition and expert seed lineage before claiming reproducible
   source/data provenance.
4. Repair `optimizer_groups()` before using the generic JEPA training API.
5. Keep N3/N6 as tested edge-case fixes; they do not explain the existing
   artifact results.
