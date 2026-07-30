# ARCHITECTURE.md audit — undocumented deviations and doc/code contradictions

Audit date: 2026-07-30
Branch: `craftax-clean-baseline` @ `d26d360`
Auditor: Claude (Opus 5), reviewing `spec/ARCHITECTURE.md` against `d4_mamba_jepa/`
and the pinned checkouts in `third_party/`.

Scope: every component and line of `ARCHITECTURE.md` was checked against the
live code and, where the spec cites a source, against that source's bytes in
`third_party/sources/`. Numeric claims marked *measured* were produced by
executing the live `craftax_jepa_config()` model in `.venv` on the RTX 3060, not
read off the code.

**Verdict.** The spec is accurate on essentially every claim it makes (§C). What
follows is what it does not say: deviations from the pinned sources whose
consequences are invisible from the spec (§A), and places where the spec and the
code disagree with each other (§B).

Nothing in this file has been fixed. It is a findings list.

---

## A. Undocumented deviations from source

### A1 — `dynamics.flow_x_head` is dead weight in the JEPA arm

Upstream zero-initializes `flow_x_head` (`nn.init.zeros_` on both weight and
bias, `third_party/sources/nicklashansen__mmbench2/src/model.py:964-966`), on the
assumption that the flow loss will move it. The JEPA arm has no flow loss, and
with `jepa_predictor_context="pooled_agent"` the `CDPPredictor` discards its
`spatial_tokens` argument entirely (`model.py:141-142`), so nothing consumes the
`Dynamics` spatial output at all.

Measured, after one full `world_loss(...).backward()` on the live JEPA config:

```
dynamics.flow_x_head.weight: grad=None
dynamics.flow_x_head.bias:   grad=None
weight.abs().sum() == 0.0
params with requires_grad=True and zero/None grad: 3
  encoder.mae.mask_token
  dynamics.flow_x_head.weight
  dynamics.flow_x_head.bias
```

4,160 parameters sit in the AdamW `trainable` list and can never move (PyTorch
skips `grad is None`, so weight decay does not touch them either), plus 64 more
in the dead MAE mask token. §6 records the dynamics output as "tokens, agent
tokens"; the "tokens" half is an identically-zero tensor at every step of
training and deployment.

### A2 — reward and continuation heads: leads 5–7 of 8 never receive gradient

Measured per-lead gradient magnitude at `B=8, T=16` on the live config:

```
reward head        (L=8): [1.636, 1.302, 0.968, 0.663, 0.333, 0.0, 0.0, 0.0]
continuation head  (L=8): [0.692, 0.561, 0.432, 0.292, 0.139, 0.0, 0.0, 0.0]
```

Cause: `_mtp_scalar_targets` (`common.py:92-97`) and `continuation_mtp_loss`
(`objectives.py:212-217`) both iterate `usable = T - lead` and `break` once it
reaches zero. In `_jepa_world_loss` the heads are fed `rollout_agents`, whose
time axis is `K = jepa_jumps = 5`, not `sequence_length = 16`. So leads ≥ 5 are
fully masked out, and the surviving leads are trained from 5, 4, 3, 2 and 1
positions respectively.

§11 lists `reward_horizon=8 · continuation_horizon=8`, which reads as the
trained horizon. The effective trained horizon is 5, with a 5:4:3:2:1 count
taper. §11's `diff` line states that the heads see `jepa_jumps` positions but
does not state that three of eight output slots per head are therefore untrained
weights. Deployment only ever reads lead 0
(`heads["reward_logits"][:, 0, 0]`, `imagination_actor_critic.py:499`,
`rollout.py:294`), so this does not break inference — it makes the declared
horizon misleading and ships 3/8 of each head's output layer as noise.

**NARROWED after validation.** "Ships as noise" overstates it and I should not
have written it that way. Leads 5-7 are rows of `out.weight`/`out.bias`, which
receive a *dense* gradient tensor, so AdamW's decoupled weight decay does move
them — they decay toward their initialization prior rather than staying put.
Only parameters whose `.grad` is `None` (A1) are skipped by the optimizer
entirely. Two further boundaries: the base/CDP arms feed all `T=16` positions
and do train all eight leads, and deployment reads lead 0 only.

### A3 — shortcut conditioning collapses to two constant bias vectors

Every JEPA code path fills the conditioning indices with constants:
`step_indices = max_step_index` and `signal_indices = k_max`
(`model.py:488-493`, `objectives.py:305-306`, `common.py:209-220`,
`rollout.py:71-72`, `craftax_achievement.py:36-41`).

Measured per-row gradient of the embedding tables:

```
dynamics.step_embed.weight   (3, 32): [0.0, 0.0, 0.132]
dynamics.signal_embed.weight (5, 32): [0.0, 0.0, 0.0, 0.0, 0.112]
```

Six of the eight rows (192 parameters) are functionally unreachable; the two
live rows are concatenated and added identically at every position, i.e. they
act as a constant conditioning bias. `SOURCE_MANIFEST.md` lists "shortcut signal
and step embeddings" among the upstream contracts that are *not* locally
reinterpreted; in this arm they are degenerate. Not recorded in `ARCHITECTURE.md`.

**NARROWED after validation**, same reason as A2: "stay at initialization
forever" is wrong. The embedding weight is a single dense tensor, so the
unreachable rows get a zero *loss* gradient but still decay under AdamW.
Unreachable is the accurate word; frozen is not.

### A4 — `JepaProjector` width deviates from SPR, and from SPR's default head

`model.py:166-170` claims fidelity to `mila-iqia/spr` `src/models.py`
`global_classifier` (`Linear -> BatchNorm1d -> ReLU -> Linear`). Live shapes:

```
jepa_projection: Linear(256, 64) -> BatchNorm1d(64) -> ReLU -> Linear(64, 64)
jepa_prediction: Linear(64,  64) -> BatchNorm1d(64) -> ReLU -> Linear(64, 64)
```

Two deviations, neither in §9:

1. **Hidden width.** Both SPR MLP variants use hidden = 2× output.
   `third_party/sources/mila-iqia__spr/src/models.py:210-215` is
   `Flatten -> Linear(pixels*hidden, 512) -> BatchNorm1d(512) -> ReLU -> Linear(512, 256)`
   (hidden 512, out 256); `:239-243` is `Linear(s, 2s) -> BatchNorm1d(2s) -> ReLU -> Linear(2s, s)`.
   Ours uses hidden = out.
2. **Wrong branch.** SPR's *default* is `--classifier q_l1`
   (`scripts/run.py:111`), i.e. a `QL1Head` built off the Q head, not the MLP
   branch at all; and `--final-classifier linear` (`scripts/run.py:112`), which
   makes the prediction head a single `nn.Linear` (`src/models.py:245-247`).
   Our prediction head is a BatchNorm MLP where SPR's default is one linear layer.

The loss itself (`F.normalize(p=2, dim=-1, eps=1e-3)` then
`mse_loss(...).sum(-1)`) *is* faithful — see §C.

### A5 — the PPO expert has no byte-level provenance

§2's `src` line cites `Craftax_Baselines@7ce36fa ppo_rnn.py` in the same form as
the digest-pinned sources. It is **not** present in `third_party/sources/` and
**not** listed in `third_party/SOURCES.lock`. It exists only as a docstring
reference at `expert/ppo_expert.py:9-11`, above a file that documents itself as a
re-implementation: distrax replaced by a native categorical, chex by
`flax.struct`, orbax by `flax.serialization`, wandb/logz dropped, the target env
changed to `Craftax-Classic-Symbolic-v1`.

The component that generated the entire 696,746-transition training corpus is the
one component in the pipeline with no verifiable upstream identity. Everything
downstream inherits that.

### A6 — terminal-window sampling has a support of 58 windows

`outputs/d4_mamba_jepa/craftax_expert_v1/report.json:538` records
`train_terminal_episodes: 58`. Cross-checked against
`d4_mamba_jepa/artifacts/expert/craftax_expert_v1.pt.manifest.json`:
`truncated_episodes: 252` of 320 hit the 2500-step cap, so only 68 episodes ever
reach `continues[-1] == 0`; 58 survive the 80% train split. (Timeout at
`max_timesteps = 10000` is unreachable under a 2500 cap, so all 68 are genuine
death/lava terminals.)

§3's `terminal_fraction=0.5` forces `round(8 × 0.5) = 4` of every 8 rows to be
*the last window* of one of those 58 episodes (`common.py:156-164`, start index
`len(episode.obs) - sequence_length`). Over 20,000 updates that is 80,000 draws
from 58 fixed 16-frame windows — roughly 1,380 repetitions each. §3 documents
the mechanism accurately but not that half of every batch comes from a
58-element set.

### A7 — executed evaluation runs at context 8; BC was trained at 16

`reviews/artifacts/craftax_achievement_run.py:49` defaults `--context 8`, and
`_policy_action` slices `observations[-context:]` (`craftax_achievement.py:30`).
§12 trains BC on `sequence_length = 16` windows. The dynamics is block-causal, so
the agent token at the last position of an 8-frame window is not the token at
the last position of a 16-frame window. §14's `diff` covers seeds and sampling
temperature but not this context change between training and executed
evaluation.

**NARROWED after validation.** "BC was trained at 16" is false as a categorical
statement, and the correction matters. BC's loss is `logits[:, :-1]` against
`led_to_actions[:, 1:]` (`craftax_runners.py:294-300`), and the dynamics is
block-causal, so one 16-frame batch trains decisions at context lengths 1
through 15 — including 8. The real mismatch is distributional, not categorical:
BC weights lengths 1-15 uniformly (7/15 positions see more than eight frames),
while execution uses length 8 for every post-warm-up step.

### A8 — `implementation_sha256()` does not cover the Craftax boundary

`IMPLEMENTATION_FILES` (`checkpoint.py:22-37`) omits `craftax_env.py`,
`craftax_data.py`, `craftax_achievement.py`, `craftax_oracle.py`,
`oracle_metrics.py`, `craftax_resolution.py`, `executed_control.py` and all of
`expert/`.

§16 says "digest-pinned sources recorded into every checkpoint", which is true of
the upstream pins. But the implementation digest that gates checkpoint reload
(`checkpoint.py:211-217`) covers only the training core. The environment
adapter, the achievement evaluator, the oracle and the expert generator can all
change without tripping drift detection on any existing checkpoint.

---

## B. Doc/code contradictions

### B1 — `model.py:109` contradicts §8 and is wrong

The comment says Dreamer-CDP "predicts from a 4096-d deterministic state
(`rssm.py:140`)". Verified in `third_party/sources/fmi-basel__Dreamer-CDP`:

- `dreamerv3/configs.yaml:93` is `rssm: {deter: 8192, ...}` — the live default.
- `deter: 4096` appears at `configs.yaml:140`, a smaller size preset.
- `dreamerv3/rssm.py:140` is `pred_enc = self.predictor(feat['deter'])`.

§8's "8192-d deterministic state (`configs.yaml:93`)" is correct. The code
comment states the wrong width and conflates two different files' line 140.

### B2 — `config.py:54-56` parameter-matching rationale is stale for Craftax

The comment justifies `expand=1` as "15,014 parameters at d_model=64 versus
16,644 for the upstream temporal attention module. expand=2 would confound the
backend comparison with a 67% larger temporal module." That arithmetic holds only
at the `D4LiteConfig` defaults `mamba_d_state=16, mamba_headdim=32`, which
`craftax_jepa_config` overrides to 64/64 (`craftax_runners.py:66-67`).

Measured at the live Craftax config:

```
T arm: TimeSelfAttention  16,644 params x2 = 33,288
M arm: MambaTimeMixer     21,571 params x2 = 43,142   (+29.6% per module)
world totals: 986,348 (T) vs 996,202 (M)
```

§7's "arms are not parameter-matched" is correct. The config comment reads as
though they still are, and is the more likely thing to be read at the call site.

### B3 — §16's "`POLICY_FORMAT`/`FORMAT` still carry `cartpole`" is partial

Carrying `cartpole`:
`common.POLICY_FORMAT` (`= "d4_lite_cartpole_bc_policy_v1"`),
`imagination_actor_critic.FORMAT`, `imagination_actor_critic.EVALUATION_FORMAT`.

Not carrying it: `checkpoint.FORMAT`, `checkpoint.TOKENIZER_FORMAT`,
`craftax_run.FORMAT`, `craftax_data.FORMAT`, `expert/generate.FORMAT` — all
`d4_mamba_jepa_*`. The bare name `FORMAT` is ambiguous across eight constants and
the sentence reads as though it applies to all of them.

### B4 — §11's `configs.yaml:101` points at DreamerV3's value head

In `third_party/sources/danijar__dreamerv3/dreamerv3/configs.yaml`, the reward
head's `bins: 255` is line 98; line 101 is `value: {..., bins: 255}`. Both are
255, so the value is right; the citation is filed under "Task heads → reward
bins" while pointing at the value head. (Our `ValueHead` does reuse
`cfg.reward_bins`, so the reference is defensible — just mis-placed.)

### B5 — §9's `run.py:106` is `scripts/run.py:106`

Content confirmed: `--local-spr` default `0` at
`third_party/sources/mila-iqia__spr/scripts/run.py:106`, `--global-spr` default
`1` at `:107`. Only the path is truncated.

### B6 — §15 understates the oracle

`craftax_oracle.py` does two things the spec omits:

- the self-audit also runs a **timestep-shifted (off-by-one)** input, not just
  perfect/constant/misaligned (`:447`, `:476`);
- the non-linear pixel ceiling is a **CNN**, not an MLP (`:385`); the MLP is
  applied only to the latent (`:388`). The spec says "ridge + MLP probes".

### B7 — `SOURCE_MANIFEST.md` is pre-migration

Audit date 2026-07-20. It still lists `danijar/crafter` as "Environment"
(`:16`), references `m3_hjwm_compact/data.py` and `m3_hjwm_compact/checkpoint.py`
as byte-level identities (`:38-39`), asserts "Crafter action IDs, reward,
termination" as authoritative (`:70`), and has no row for Craftax, SPR, I-JEPA,
LeJEPA or Craftax_Baselines. `ARCHITECTURE.md` does not cite it, but it is the
sibling provenance document and is now describing a superseded environment.

### B8 — `rollout.py:117-122` is unreachable

The JEPA branch returns at `:114`; the triple-quoted block at `:117-122` is a
no-op string expression, not a docstring. `sample_next_packed` has no docstring.

### B9 — RETRACTED. The SIGReg loss form IS in the pinned source.

I claimed the convex combination `(1-λ)*sim + λ*sigreg` could not be verified
because `scripts/je.py` is absent from the checkout. That was wrong: I searched
for that launcher and never looked at the tracked `MINIMAL.md`. Re-verified at
`third_party/sources/rbalestr-lab__lejepa/MINIMAL.md:171-174`, git-tracked at
`c293d291`:

```python
inv_loss = (proj.mean(0) - proj).square().mean()
sigreg_loss = sigreg(proj)
lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
```

Exactly the reference form. The claim is withdrawn.

What survives is a *larger* deviation I failed to state: LeJEPA's invariance
term is a **multi-view** variance-to-the-mean across `V` augmented views of one
image; ours is a **temporal** next-step prediction error. Same slot in the
objective, different quantity. Now registered in `objectives.py` and §9.

`bstat_lambda=0.05` ✓ and `EppsPulley(n_points=17)` ✓ remain in range.
`num_slices=1024` vs 1000 and `projector_dim=64` vs 512 come from an ablation
sweep, not an authoritative default — calling them departures was also too
strong.

---

## C. Verified accurate

Recorded so a re-audit does not redo this work.

**§1 Environment.** Craftax `1.6.1` installed; all three `source.py` digests
(`game_logic.py`, `constants.py`, `renderer.py`) match. `len(Action) == 17`,
`len(Achievement) == 22`, `BLOCK_PIXEL_SIZE_AGENT == 7`,
`observation_space.shape == (63, 63, 3)`, `max_timesteps == 10000`. Reward
formula confirmed at `game_logic.py:1695-1700`
(`achievement_reward + 0.1 * health_delta`); `is_game_over = done_steps | in_lava | is_dead`.
`continues = 1 - dead` with timeout → 1 confirmed at `craftax_env.py:186-188`.

**§2 Replay.** Manifest: 320 episodes, 696,746 transitions, `max_steps: 2500`,
`num_envs: 32`, `greedy: false`. `SPLIT_SEED = 20260727`, 80/10/10 whole-episode
via `whole_episode_splits`.

**§3 Sampler.** `int(round(batch_size * terminal_fraction))` at `common.py:142`;
forced last window at `:159`. The remainder goes through
`EpisodeReplay.sample`, which draws `valid[randint(len(valid))]` then a uniform
start — episode-uniform, not transition-uniform, exactly as the spec states
(`data.py:92-93`).

**§4/§5 Encoder + packing.** `build_tokenizer(cfg, training_mask=False)` at
`model.py:243`; measured live `encoder.mae.p_min == p_max == 0.0` while
`cfg.mae_p_max == 0.9`, so `MAEReplacer` takes its documented fast path
(`mmbench2 src/model.py:133-137`). Measured shapes: `n_patches=64`,
`patch_dim=192`, bottleneck `(B,16,16,16)` in `[-0.72, 0.54]` (tanh), packed
`(B,16,4,64)`. Dreamer-CDP `enc_lr: 6e-6` / `dyn_lr: 4e-4` confirmed at
`configs.yaml:88-89`.

**§6 Dynamics.** Measured agent tokens `(B,16,2,64)`. Upstream
`ActionEncoder(action_dim=16)` is continuous, and `DiscreteActionEncoder`
replaces it in the same token slot with matching init constants (`base` std
0.02, delta std 1e-3 mirroring upstream `fc2`).

**§7 Temporal operator.** `mamba_d_state=64, mamba_headdim=64` set at
`craftax_runners.py:66-67`; measured `d_inner=64, nheads=1, headdim=64,
d_state=64, ngroups=1, d_conv=4` — exactly one head, as stated. The swap runs at
`model.py:317`, after `_build_jepa()` at `:310`.

*The shared-init claim was verified empirically*: constructing both arms under
`torch.manual_seed(0)` gives 227 shared state-dict tensors, **all 227
bit-identical**; only the 10 transformer-only / 16 mamba-only temporal tensors
differ. The D037 contract holds.

**§8 Predictor.** Live `jepa_predictor.net` = `Linear(128, 64) -> LayerNorm(64)
-> SiLU -> Linear(64, 256)`; context is the mean over 2 agent tokens (64) plus
the next action token (64) = 128 in, `n_spatial * d_spatial = 256` out.
Dreamer-CDP's per-token cosine confirmed at `rssm.py:141`
(`optax.losses.cosine_distance(sg(slow_tokens), pred_enc, axis=-1)`).

**§9 Self-prediction loss.** SPR `spr_loss` (`src/models.py:287-293`) is
`F.normalize(p=2., dim=-1, eps=1e-3)` on both sides then
`F.mse_loss(reduction="none").sum(-1)` — reproduced exactly at
`objectives.py:344-349`. `--local-spr` default 0 confirmed. SPR
`momentum_tau=0.01` (`scripts/run.py:82`) matches our `jepa_ema_tau=0.99` under
the reciprocal convention documented at `model.py:534-542`. I-JEPA's linear
momentum ramp is at `src/train.py:228`
(`ema[0] + i*(ema[1]-ema[0])/(...)`) with published `ema: [0.996, 1.0]`
(`configs/in1k_vith14_ep300.yaml:42-44`). I-JEPA scores per token via
`F.layer_norm` + `smooth_l1_loss` (`train.py:297, 311`); V-JEPA 2 per token via
L1 (`third_party/vjepa2/app/vjepa/train.py:447`) — neither has a global variant,
as claimed. SPR's aux-beside-Q-learning composition is at `algos.py:131-136`.
Our global flatten `[B,K,4,64] -> [B,K,256] -> 64` confirmed at
`model.py:519-523`.

**§10 Loss composition.** `WorldLossNormalizer` registers exactly
`("flow", "reward", "continuation", "cdp", "reconstruction")` — no `"jepa"`
(`training.py:34-39`), so the JEPA term is unnormalized as stated. AdamW `lr=1e-4,
weight_decay=1e-2`, `clip_grad_norm_(1.0)`, `warmup=1000` (linear, constant
after), `WORLD_STEPS=20_000`, `WORLD_BATCH=8`. EMA ramp `0.99 -> 0.999` linear
over `ema_schedule_steps = world_steps` (`craftax_runners.py:197-200`).
The `encoder_learning_rate` param-group split behaves as described
(`craftax_runners.py:143-169`).

**§11 Task heads.** Measured `reward_head.out: Linear(64, 2040)` = 8 × 255;
`centers_log = linspace(-10, 10, 255)`; `continuation_head.out: Linear(64, 8)`.
`bins: 255` present in DreamerV3 `configs.yaml` (see B4 for the line).

**§12 BC policy.** `BCPolicy` reproduces `PolicyHeadMTP`'s attention pool
(`pool_query = randn * 0.02`, `pool_kv = Linear(d, 2d, bias=False)`,
`scores / sqrt(D)`, `softmax(dim=-1)`), the `MLP(mlp_ratio=2.0)` projector, and
the `normal_(std=0.01)` / `zeros_` output init. World frozen via
`freeze_module(world)`. *Not noted in §12*: the `L = 8` multi-token output is
dropped to a single step, and tanh-squashed means become categorical logits —
both reasonable ports, but they are deviations the spec does not list.

**§13 Imagination.** `gamma=0.997, lambda_=0.95, alpha=0.5, beta=0.3`
(`craftax_runners.py:334-337`); `ACTOR_STEPS=500, ACTOR_BATCH=64,
ACTOR_CONTEXT=8, ACTOR_HORIZON=32`. The `[:, -context:]` re-slice is at exactly
`imagination_actor_critic.py:512`. `_sample_next_jepa` carries only an agent
token, never a conv/SSM state (`rollout.py:41-94`), and `imagine_trajectory`
does not pass even that (`:490-497`) — so `MambaTimeMixer.forward` takes its
`self.mamba(flat)` full-scan path on the sliding 8-state window at every one of
the 32 steps. Exactly as stated.

**§14/§15/§16.** `_crafter_score` matches `danijar/crafter`
`analysis/common.py:47-55` (`exp(mean(log(1 + percents))) - 1`). Policy sampling
seed is a deterministic function of the env seed
(`craftax_achievement.py:132`), so CIs cover environment-seed variance only, as
stated. `preserved` requires `latent_exists >= exists - margin` where `exists` is
the pixel ceiling (`craftax_oracle.py:405-406`); `achievement_group` uses only a
latent ridge with no constant/timestep/pixel reference (`:421-441`).
`source_report()` returns exactly `{mmbench2_model, mamba2, gymnasium_cartpole}`
— no Craftax, no JEPA source. (`craftax_source_report()` and
`lejepa_source_report()` exist but are not called by `craftax_run.py`, which uses
`source_report()` at `:351`.)

---

## D. Validation task for a second agent

You are re-auditing the claims above **independently**. Do not accept them. Your
job is to mark each of A1–A8 and B1–B9 as `CONFIRMED`, `REFUTED`, or
`PARTIAL`, with the evidence that decided it. §C is provided as context; you may
spot-check it but it is not the task.

Ground rules:

- **Do not trust this file's line numbers.** They were correct at `d26d360`;
  re-derive them. If a line number has drifted, that is a finding, not a refutation.
- **Do not trust this file's measurements.** Every claim tagged *measured* must
  be reproduced by executing code, not by reading it. Use the project venv:
  `PYTHONPATH=. .venv/bin/python …`. Mamba-2 requires CUDA — its Triton kernels
  fail on CPU tensors, so move the model to `cuda` before any `mamba2` forward.
- **Read the primary source, not a summary of it.** Every `third_party/sources/*`
  claim must be checked against the file's actual bytes at the pinned commit.
- **A deviation is only a finding if it is absent from `ARCHITECTURE.md`.** The
  spec explicitly records many deviations; re-reporting one of those is a false
  positive. Check `ABLATIONS.md` too — history belongs there by the spec's own
  rule, so a finding already logged there is not undocumented.
- **Distinguish "wrong" from "imprecise".** B3, B4, B5 are citation-precision
  issues, not errors of fact. Say which you think each one is.

Specific things worth attacking:

1. **A1/A2/A3 are the load-bearing measurements.** All three claim parameters
   receive no gradient. Reproduce each with an independent script — construct
   the world, run one `world_loss` backward on synthetic data, and inspect
   `.grad` directly. If any of the three is an artifact of synthetic data
   (e.g. all-ones `led_to_continues` masking a branch), say so.
2. **A2's consequence.** Confirm or refute that lead slots 5–7 are dead *for the
   whole training run*, not just for one batch — i.e. that nothing else in
   `craftax_runners.train_craftax_jepa_world` ever calls the heads with
   `T > jepa_jumps`. Check the CDP/base path too: does `world_loss`'s
   non-JEPA branch feed them full-length `T`? If so, the deadness is
   JEPA-arm-specific and should be stated that way.
3. **A6's arithmetic.** I derived 68 terminal episodes from
   `truncated_episodes: 252` in the manifest plus the claim that a 2500-step cap
   makes the 10000-step timeout unreachable. The `58` is read from a run report.
   Verify both independently — ideally by counting
   `sum(ep.continues[-1] == 0.0 for ep in replay.episodes)` on the real replay
   after `whole_episode_splits(320, seed=20260727)`. The replay is ~8.5 GB;
   load it with `weights_only=False` and count without materializing `obs` if
   memory is tight.
4. **A5's negative claim.** Prove the absence properly: search the whole repo
   (including `.gitignore`d paths and any sibling checkouts) for a
   `Craftax_Baselines` working tree, and diff `expert/ppo_expert.py` against the
   real upstream `ppo_rnn.py` + `wrappers.py` at `7ce36fa` if you can fetch
   them. If the port is faithful modulo the documented removals, A5 downgrades
   from "no provenance" to "provenance not pinned".
5. **A7's severity.** I claim a train/deploy mismatch. Quantify it if you can:
   does the agent token at position 7 of an 8-frame window actually differ
   materially from position 15 of a 16-frame window, given block-causal
   attention and a frozen encoder? Measure a cosine, do not argue from the
   architecture.
6. **A4's "wrong branch" claim.** Confirm that `q_l1` really is SPR's shipped
   default and that `QL1Head` is structurally unlike our `JepaProjector`. Also
   check whether the SPR *paper* (not the repo default) uses the MLP branch —
   if it does, A4's second point weakens to a repo-default-vs-paper distinction.
7. **B1.** Confirm `configs.yaml:93` vs `:140` and `rssm.py:140` at the pinned
   commit, and confirm which preset Dreamer-CDP actually trains with. If the
   paper's reported runs use a 4096 preset, the code comment is right and §8 is
   wrong — the reverse of my finding.

Deliverable: a `VALIDATION.md` in this directory with one row per claim
(`ID | verdict | evidence | notes`), and an explicit list of anything I missed.
Findings I got wrong matter more than findings I got right — say so plainly.

---

# Round 2 — audit of `CODEX.md`, and new findings

Audit date: 2026-07-30 (same commit `d26d360`).
This round audits `bugs/CODEX.md` and then extends the list.

CODEX ran on a **CPU-only** runtime (`cuda_available: false`) and reported
`98 passed, 7 skipped`. On this machine (RTX 3060, CUDA 13.0) the same suite is
**`105 passed, 0 skipped, 1 warning in 49.61s`**. Every Mamba forward, cache,
mixed-precision and gradient test that CODEX had to leave unvalidated executes
and passes. Findings below that required CUDA are new to this round.

## E. Verdict on CODEX.md

### E.1 — Correct, and independently found in §A/§B above

No re-verification needed; two independent audits agree.

| CODEX | Same as | Note |
|---|---|---|
| §2 / C06 Craftax_Baselines absent | A5 | |
| §1 / C03 `SOURCE_MANIFEST.md` stale | B7 | |
| C08 sampler remainder episode-uniform | §C | |
| C11 equal-seed arms share init | §C | both measured 227 identical tensors |
| C12 parameter counts | B2 | identical numbers: 16,644 / 21,571; 986,348 / 996,202 |
| C35 `implementation_sha256` omissions | A8 | |
| C36 `source_report` omissions | §C | |
| C37 CartPole identifiers only partly renamed | B3 | |
| C32 / §15 oracle pixel ceiling is a CNN | B6 | |
| C22 JEPA term unnormalized; LR constant after warmup | §C | |

### E.2 — Correct, and I missed them. Verified this round.

These are real. They are added to the findings list.

**A9 — Mamba runs off its default execution path.**
`temporal.py:50` passes `use_mem_eff_path=False`. Upstream default is `True`
(`third_party/sources/state-spaces__mamba/mamba_ssm/modules/mamba2.py:59`).
Still official Mamba code, different kernel path and different numerics.
Undocumented. *(CODEX C13 — CONFIRMED.)*

**A10 — the optimizer overrides Mamba's no-weight-decay contract, and only in
the M arm.** Upstream marks three tensors exempt:
`self.dt_bias._no_weight_decay = True` (`mamba2.py:130`),
`self.A_log._no_weight_decay` (`:136`), `self.D._no_weight_decay` (`:140`).
`craftax_runners.py:136-166` puts every trainable parameter into one AdamW group
with `weight_decay=1e-2` and never reads that attribute. Decoupled AdamW shrinks
them by `lr*wd = 1e-6` per step, ≈ 2% over 20,000 updates. The T arm has **no**
such tensors, so this is an asymmetry applied to exactly one arm of the single
research axis: `A_log→0` drags `A=-exp(A_log)` toward `-1`, `dt_bias→0` toward
`softplus(0)=0.693`, `D→0`. Undocumented. *(CODEX C14 — CONFIRMED, and it is an
arm asymmetry, which CODEX did not note.)*

**A11 — §11's `src MMBench2 MTP heads` is false for the continuation head.**
Grepping the pinned `src/model.py` for continuation/termination/discount returns
nothing: MMBench2 has **no** continuation head. `ContinuationHeadMTP`
(`model.py:62-82`) is local, and it **mean-pools** agent tokens where the
upstream `RewardHeadMTP` uses a learned attention pool (`pool_agent="attn"`,
`model.py:274`). Two heads reading the same tokens with different poolers, one
attributed to an upstream source that does not contain it. *(CODEX C23 —
CONFIRMED.)*

**A12 — §8's "per-token cosine" is wrong, and I wrongly confirmed it.**
Dreamer-CDP's `dyn_deter` is a **single global cosine per timestep**, not
per-spatial-token. Chain: `rssm.py:141`
`cosine_distance(sg(slow_tokens), pred_enc, axis=-1)`; `pred_enc =
self.predictor(feat['deter'])` whose output width is `self.enc_output`
(`rssm.py:65,67`), a scalar dim set from `self.enc_output_dim =
calculate_encoder_output_dim(...)` (`agent.py:44`); and `_observe` flattens
tokens to one vector per timestep (`rssm.py:99`). So `axis=-1` spans the whole
flattened representation.

In my round-1 §C I listed this as verified. **I was wrong** — I read the
`axis=-1` and did not follow `enc_output` to its definition. CODEX is right.

This matters beyond a citation fix: §9's `diff` presents "scores ONE global
vector per frame" as our deviation from the JEPA family, while §8 presents CDP as
per-token. Since CDP is *global*, our global scoring is **faithful to the source
we took the predictor from**. The two spec lines point in opposite directions and
one of them is factually false. *(CODEX C17 — CONFIRMED; supersedes my §C entry.)*

**A13 — SPR's `jumps=5` gives six loss positions; ours gives five.**
`models.py:449` appends the t0 latent before the jump loop at `:453-457`, so
`len(pred_latents) = jumps+1 = 6`; `algos.py:296-298` splits `spr_loss[0]` (t0,
weight `t0_spr_loss_weight=1.0`) from `spr_loss[1:].mean(0)`. Our
`jepa_self_prediction_loss` produces exactly `K=5` transitioned positions and
**no t0 term**. SPR's t0 term is its augmentation-invariance objective — the
BYOL-like half of the anti-collapse pressure. We dropped the augmentation (§9
records that) *and* the t0 loss (§9 does not). Dropping t0 without augmentation
is arguably forced, but the consequence — weaker anti-collapse than SPR — is
undocumented. *(CODEX C18 — CONFIRMED.)*

**A14 — the SPR EMA-equivalence claim is unverifiable from this repo.**
`third_party/sources/mila-iqia__spr/src/models.py:5` imports `update_state_dict`
from `rlpyt.models.utils`. `rlpyt` is neither vendored under `third_party` nor
installed in `.venv`. `model.py:534-542` asserts "This is the same EMA as SPR's
`update_state_dict` (rlpyt) up to the reciprocal tau naming" — that cannot be
checked here. *(CODEX C20 — CONFIRMED.)*

**A15 — §12 understates the Dreamer-4 BC difference.**
`third_party/sources/edwhu__dreamer4-jax/scripts/train_bc_rew_heads.py:791-799`
builds the optimized tree as `{"dyn", "task", "pi", "rew"}` — the reproduction
jointly updates dynamics, task embedding, policy **and** reward head under a
combined loss. §12's diff says only "optimizes `p["dyn"]` during BC".
*(CODEX C25 — CONFIRMED.)*

**A16 — §13's sliding window is not a deviation from the cited source.**
`third_party/sources/edwhu__dreamer4-jax/dreamer/imagination.py:409-415` does
exactly `jnp.concatenate([z_ctx_clean_t, z_clean_pred], axis=1)[:,
-context_length:, :, :]` and the same for actions. The cited reproduction slides
and rescans too. The genuine deviations are the **absent recurrent cache** and
the **context length**, not the sliding itself. §13's `diff` conflates them.
*(CODEX C27 — CONFIRMED.)*

**A17 — the Dreamer-4 comparison constants in §4 and §13 are cherry-picked.**
- §13 "reduced vs Dreamer 4's 192-frame context": `docs/appendix.txt:15` gives
  `C = 192` for **Minecraft only**; `:37` and `:45` give `C = 96` for SOAR and
  Epic Kitchens; and the cited JAX code defaults to `context_length: int = 16`
  (`scripts/train_policy.py:147`). Our 8 is close to the code default, not 24×
  smaller than "Dreamer 4".
- §4 "`n_latents=16` vs Dreamer 4's 512 latent tokens": `appendix.txt:14` does
  say `N_b = 512, D_b = 16`, but `train_policy.py:109-110` defaults
  `enc_n_latents=16, enc_d_bottleneck=32`. So our `n_latents=16` **matches the
  reference implementation's own default**, and our `d_bottleneck=16` matches the
  paper while differing from the code's 32. The spec compares to the paper for
  one number and never mentions that the code agrees on the other.
Neither line says which reference it is comparing against. *(CODEX §4, C28 —
CONFIRMED.)*

**A18 — the value head's bin count is an undocumented deviation.**
`ValueHead` defaults to 255 bins over `[-10,10]` and `craftax_runners.py:354-357`
passes `num_bins=cfg.reward_bins=255`. The cited JAX reproduction defaults to
`num_reward_bins: int = 101` and `num_value_bins: int = 101`
(`train_policy.py:136,139`). §13's knobs line does not mention value support at
all. *(CODEX C29 — CONFIRMED.)*

**A19 — the transition-convention sentence is imprecise.**
`ARCHITECTURE.md:153` says "Position 0 gets the start action". In fact
`data.py:98-101` and `common.py:117-120` write `previous[0] =
episode.actions[start-1]` whenever `start > 0`; only a window at a true episode
start gets `-1`. `outcome_valid[:,0]` is `False` either way. The **code is
correct**; the sentence is wrong. *(CODEX C39 — CONFIRMED as a doc defect, not a
code defect; see E.3.)*

**A20 — non-atomic, unprovenanced artifact writers.**
`craftax_runners.py:382-385` writes `value.pt` with a bare `torch.save` — no
temp-file-and-rename, no format tag, no world pairing, no provenance, unlike
every other writer in the package. `save_policy_checkpoint` (`:80-93`) is atomic
but stores no RNG or source report. §16's "strict atomic saves with full RNG" is
true of `checkpoint.py` only. *(CODEX C33/C34 — CONFIRMED.)*

**A21 — two smaller ones, both correct.**
- `DiscreteActionEncoder` holds `n_actions + 1 = 18` embeddings (measured shape
  `(18, 64)`); §6's "17-way" omits the dedicated start/unlabelled slot.
  *(CODEX C10.)*
- `Dynamics` is built with `lang_dim=0` (`model.py:260`), so upstream's
  `task_proj` is `None` and agent tokens initialize to **zeros** every forward
  (`mmbench2 src/model.py:1000-1004`). The task-conditioning pathway the upstream
  agent tokens exist for is disabled. Undocumented. *(CODEX §6.)*

### E.3 — Ambiguous or overstated

**CODEX C15 / "Runtime boundary" — misleading as written.** CODEX says the Mamba
forward/cache/mixed-precision/backward paths "were not executable on this
runtime" and that "a source digest check is not a substitute for those runtime
tests". True of its own CPU-only environment; **not a property of the repo**. On
CUDA all 105 tests pass with zero skips. The skip predicates are correct
CUDA-guards, not missing coverage. This should not be carried forward as a
finding.

**CODEX C26 / "`never accumulates state` is too strong" — pedantic, and it
buries the real issue.** §13 plainly means recurrent SSM/KV state, and that
reading is correct: `MambaTimeMixer.forward` takes its stateless
`self.mamba(flat)` path on every imagination step. CODEX's point that generated
latents still carry information forward inside the window is true but trivial.
The consequential fact — what the rollout *does* once that window is entirely
synthetic — is in §F below, and CODEX did not look.

**CODEX C31 / evaluation CI.** Fair refinement: because the policy seed is a
deterministic function of the env seed (`craftax_achievement.py:132`), resampling
env seeds resamples policy streams too, so the interval is over joint pairs.
But the spec's intent — "this excludes training-seed variance" — is correct, and
CODEX's own sentence concedes it. Wording nit, not a defect. **PARTIAL.**

**CODEX C38 / stale `untested`.** Split verdict. The summary sentence "Nothing in
components 4-13 was **selected** on Craftax" is **literally true**: `ABLATIONS.md`
rows 4, 5 and 14 all record *rejected* or *no effect*, and rows 16-17 tested
encoder LR but the live default is still `encoder_learning_rate=None`. However
"the untested surface" and "carried over unexamined" **are** stale for
`d_bottleneck`, `n_latents`, predictor context and `terminal_fraction`, which
have Craftax rows. CODEX's proposed three-way split (origin / tested / selected)
is a good fix. **PARTIAL.**

**CODEX C21 / I-JEPA-V-JEPA2.** CODEX corrects a claim the spec does not make.
§9 says they "normalize and score per token, and have no global variant" — all
three parts are true (`ijepa/src/train.py:297` `F.layer_norm`; `:311`
`smooth_l1_loss`; `vjepa2/app/vjepa/train.py:447` per-token L1). The bare word
"normalize" is loose (LayerNorm, not L2), which is worth a word. **Not a defect.**

### E.4 — Wrong

**Executive verdict item 6: "The transition convention is wrong for replay
windows sampled from the middle of an episode."** The convention and the code are
correct; one *sentence* of prose is imprecise (A19). Calling the convention wrong
would send someone to change working code.

**Executive verdict item 3, second half, as scoped.** "the local optimizer applies
weight decay to parameters that upstream marks as exempt" is right (A10), but it
is listed under a heading implying it affects both arms. It affects only the M
arm — which makes it more serious, not less.

## F. New findings — the imagination pathology chain

The user's hypothesis was that the M arm's imagination fails because Mamba does
not carry recurrent state. **The premise is right, the executed numbers confirm
the asymmetry, and the mechanism is real — but it is not the one named, and the
attribution to the backend is not yet licensed by a clean experiment.** All
measurements below are on the trained checkpoints in
`outputs/d4_mamba_jepa/craftax_expert_v1/`, loaded with
`strict_implementation=False`.

First, the premise. `dev_action_accuracy` shows T ≥ M (0.1594 vs 0.1557), but the
**executed** Crafter score of the BC policy shows the opposite, and that is the
metric that matters (`reviews/artifacts/craftax_lr_experiment_summary.json`):

| cond | BC score | actor score | actor − BC | 95% CI |
|---|---|---|---|---|
| full_m | 3.604 | 2.948 | **−0.656** | [−1.32, 0.12] |
| full_t | 2.606 | 2.481 | −0.124 | [−0.86, 0.46] |
| slow_m | 3.554 | 3.292 | **−0.262** | [−1.18, 0.63] |
| slow_t | 2.353 | **4.395** | **+2.042** | [0.96, 2.62] |

M has the better world model *and* the better BC; only T's imagination improves
on its own BC.

**F1 — the imagined rollout leaves the trained support after 8 of 32 steps.**
Structural, from the code, arm-independent:
- Training (`objectives.py:300-318`): `past` starts at `context = T−K = 11` **all
  real** latents and grows to 15. At the last jump 4 of 15 entries are synthetic
  — a maximum synthetic fraction of **27%**, and the real window start is always
  in view.
- Deployment (`imagination_actor_critic.py:512`): the window is pinned at
  `context = 8`. Synthetic count is `min(step+1, 8)`, so from **step 7 the window
  is 100% synthetic** and stays that way for the remaining 24 steps.

**24 of 32 imagined steps (75%) run on a context containing zero real
observations — a regime that occurs at no point in world training.** §13's diff
records the sliding window and the missing cache; it does not record this, and
this is the load-bearing gap. (Note A16: the sliding itself matches the cited
source. The trained rollout length does not.)

**F2 — predicted latents are unbounded; real latents are `tanh`-bounded.**
`Encoder.forward` ends in `torch.tanh(self.bottleneck_proj(...))` (`mmbench2
src/model.py:560`), so every real packed latent lies strictly inside `(−1,1)`.
`CDPPredictor.net` ends in a bare `nn.Linear` (`model.py:128`) with no squashing,
and the training loss is a cosine in a *projected* space, which constrains
direction but not scale. Measured on one real context:

| arm | real min/max | real std | pred min/max | pred std | pred frac outside (−1,1) |
|---|---|---|---|---|---|
| T | −0.990 / +0.982 | 0.574 | −1.225 / +1.048 | 0.348 | 0.22% |
| M | −0.992 / +0.998 | 0.657 | −1.433 / +1.406 | 0.420 | **1.76%** |

Predicted latents carry ~60% of the real scale and escape the encoder's range,
and the M arm escapes 8× more often. They are fed straight back into
`Dynamics.spatial_proj`, which has only ever seen `tanh`-range inputs.
Undocumented.

**F3 — the rollout is a dead fixed point in T and destabilizes in M.**
32 live `_sample_next_jepa` steps from a real 8-context, batch 16, random actions:

| step | T `cos(z_t,z_{t−1})` | T p(cont) | M `cos(z_t,z_{t−1})` | M p(cont) |
|---|---|---|---|---|
| 2 | 0.994 | 1.0000 | 0.982 | 1.0000 |
| 7 | 0.993 | 1.0000 | 0.979 | 1.0000 |
| 15 | 0.995 | 1.0000 | 0.968 | 0.9771 |
| 23 | 0.995 | 1.0000 | **0.475** | 0.6990 |
| 31 | 0.996 | 1.0000 | **0.397** | 0.3895 |

T freezes: successive imagined latents are 0.99+ cosine-identical for all 32
steps, so reward and continuation are constant and the trajectory carries no
information. M tracks T until ~step 15 — the point at which the window has been
fully synthetic for 8 steps — then diverges, with latent std climbing
0.291 → 0.358 and p(continue) decaying to 0.39. **Neither arm produces a usable
imagined trajectory; they fail in opposite directions.**

**F4 — the PMPO advantage signal is at or below the value head's own error.**
From the run reports:

| arm | `return_std` | `value_mae` | ratio | pos/neg split |
|---|---|---|---|---|
| T | 0.0162 | 0.00247 | 15% | 1030 / 1018 |
| M | 0.0611 | 0.0359 | **59%** | 1177 / 871 |

PMPO partitions on the **sign** of `returns − values`. For T the returns are
nearly constant (F3), so the split is a 50.3/49.7 coin flip; for M the value
error is 59% of the entire return spread, so the sign is mostly critic noise.
`mean_advantage` is `+6.4e-4` (T) and `−1.7e-4` (M). With `beta=0.3`,
`KL(actor‖BC)` is the only non-noise term in the actor loss. Undocumented.

**F5 — the continuation head is uncalibrated and its imagined value swings per
run.** `mean_continue` across the four archived arms: **1.0000** (expert_v1 T),
0.8440 (expert_v1 M), 0.9908 (slowenc M), 0.6395 (slowenc T). It enters TD-λ as a
per-step multiplier (`imagination_actor_critic.py:283`), so the effective horizon
ranges from 32 (at 1.0) to ~2 (at 0.64) depending on the run. A head trained
with `terminal_weight=8` on 50%-terminal batches (§3, §11) is being read on
latents it never saw. Undocumented.

**F6 — BC is barely above the uniform prior, so imagination starts from noise.**
BC cross-entropy moves 2.833 → 2.378 (M) / 2.410 (T); `ln 17 = 2.833`. Held-out
action accuracy is 0.156 / 0.159 against a `1/17 = 0.059` floor, and the actor's
final entropy is 2.68 / 2.73 against a 2.833 maximum. The actor is initialized
from, and KL-anchored to, a policy that is ~85% of the way to uniform.
`ABLATIONS.md:17` already refutes under-training as the cause.

**F7 — attribution caveat, and it is the important one.**
The T-vs-M executed gap above comes from `ABLATIONS.md` rows 1 and 17, both
flagged **`INIT*`** ("pre-`ba3ae1e`, so T-vs-M shared init differed") and
**`1SEED`**. Row 19 — the fixed-init paired re-run at `enc_lr=6e-6` that would
remove the confound — is recorded as **ABORTED at T world 3k/20k; no result**.
The checkpoints I measured F2/F3 on are from row 1, also `INIT*`.

So: F1, F2 and the *existence* of the fixed-point/divergence failure are
architectural and hold regardless. The claim that the **backend** causes the
executed gap is **not currently supported by any clean experiment**, and should
not be written down as if it were. The decisive run is row 19, re-run to
completion.

## G. Added validation items

Append to the §D worksheet, same rules (do not trust line numbers or
measurements; reproduce with `PYTHONPATH=. .venv/bin/python`; CUDA required for
every Mamba path and for all of §F).

| ID | Claim | Minimum evidence |
|---|---|---|
| A9 | `use_mem_eff_path=False` departs from upstream's `True` default. | `temporal.py` vs `mamba2.Mamba2.__init__` signature. |
| A10 | AdamW decays `dt_bias`/`A_log`/`D` despite `_no_weight_decay`, in the M arm only. | Upstream attributes; enumerate optimizer groups; confirm T has no such tensors. |
| A11 | MMBench2 has no continuation head; ours mean-pools where the reward head attn-pools. | Exhaustive grep of pinned `src/model.py`; compare both local heads. |
| A12 | Dreamer-CDP's `dyn_deter` cosine is global, not per-token. | Follow `enc_output` from `agent.py:44` into `rssm.py:65-67` and the `axis=-1` reduction. **I got this wrong in round 1 — check it hardest.** |
| A13 | SPR `jumps=5` yields 6 loss positions including t0; ours yields 5. | `models.py` rollout append order + `algos.py` `spr_loss[0]` split. |
| A14 | `rlpyt.models.utils.update_state_dict` is absent from repo and venv. | Import line + filesystem search. |
| A15-A18 | Dreamer-4 BC param tree; JAX imagination slides; context 192/96/16; value bins 101 vs 255. | The four cited JAX/appendix locations. |
| A19 | Position 0 carries the preceding real action for mid-episode windows. | Construct `start=0` and `start>0` through both sampler paths. |
| A20 | `value.pt` is written non-atomically with no provenance. | Compare every writer in the package. |
| A21 | 18 action embeddings; `lang_dim=0` zeroes agent-token init. | Measure embedding shape; trace upstream `task_proj is None` branch. |
| F1 | Training tops out at 27% synthetic context; deployment is 100% synthetic for steps 8-31. | Derive from `objectives.py` loop bounds and `imagination_actor_critic.py:512`. Arithmetic, not opinion. |
| F2 | Predicted latents escape the encoder's `tanh` range; M escapes 8× more than T. | Reproduce the min/max/std table on the trained checkpoints. |
| F3 | T's rollout is a fixed point; M's diverges after ~step 15. | Re-run the 32-step diagnostic. Vary the action policy (random vs actor) and batch — **if the divergence disappears under the real actor, F3 is an artifact and must be marked so.** |
| F4 | Advantage magnitude is at/below `value_mae`; PMPO splits on noise. | Recompute from `imagination_report.json`; better, instrument a live `actor_critic_update`. |
| F5 | `mean_continue` ranges 0.64-1.00 across archived arms and scales TD-λ. | Read all four reports; trace the multiplier. |
| F6 | BC sits ~85% of the way to uniform. | Compare to `ln 17`; cross-check `ABLATIONS.md:17`. |
| F7 | The T/M executed gap is confounded (`INIT*`, `1SEED`) and the clean run aborted. | `ABLATIONS.md` rows 1, 17, 19 and their flag definitions. **If any of F1-F6 is presented as explaining the backend gap, reject it.** |

Highest-value disagreement to chase: **A12** (I confirmed a false claim in round
1) and **F3** (the only finding here whose mechanism is inferred from one
diagnostic rather than from the code alone).

---

# Round 3 — fixes applied

Applied 2026-07-30 on top of `81d3466`, after cross-validation by
`VALIDATION.md` and the `CODEX.md` addendum. Suite: **119 passed, 0 skipped**
on CUDA (was 105 before the new tests; the 7 skips other audits saw were their
CPU-only runtimes, not repo defects).

## Behaviour changes

| id | change | blast radius |
|---|---|---|
| N1 | `optimizer_groups` guards `world.decoder is None`. It crashed with `AttributeError` on every JEPA world; the CDP decoder-leak check it wraps is unchanged and still fires. | Generic API only — the Craftax runner builds its own optimizer, so no archived run is affected. |
| N3 | `craftax_env.is_dead(state)` reads the two absorbing disjuncts of `game_logic.is_game_over` (`in_lava | player_health <= 0`) instead of inferring `done and not timeout`. Same fix, vmapped, in `expert/generate.py`, plus an assertion that a non-dead `done` really is at the native horizon. | Latent: unreachable under the 2,500-step cap, so no archived artifact changes. |
| A10 | Parameters carrying upstream's `_no_weight_decay` go in a `weight_decay=0.0` group. | **M arm only.** Verified: T with `enc_lr=None` still builds exactly one group `(159 params, wd=0.01, lr=1e-4)` — byte-identical to before. `ABLATIONS.md` rows 1/17/18 now carry the new `WD*` flag. |
| A20 | `value.pt` is written through `_atomic_torch_save` with `VALUE_FORMAT`, its config, and the paired world digest; `load_value_checkpoint` verifies both. | New field `value_checkpoint_sha256` in `imagination_report.json`. Old `value.pt` files predate the format and will not load. |
| N4 | `source_report(cfg)` is config-conditional; `verify_recorded_sources` re-verifies exactly what a checkpoint recorded instead of comparing whole reports. | A T-arm world no longer needs an installed Mamba or CartPole to save/load. **Verified both archived `craftax_expert_v1` checkpoints still load.** `source_report()` with no argument is unchanged. |
| N5 | The three LeJEPA files execute under isolated module names with synthetic parent packages. | Executed set now equals verified set; no `lejepa.*` enters `sys.modules`. Statistic unchanged. |
| N2 | Records store `reset_key` (the uint32 JAX key the slot was actually reset with) plus `batch_seed`/`env_slot`/`record_index`/`final_timestep`. The old `env_seed` ordinal is gone. | New artifacts only. The archived replay is hash-pinned and untouched; its per-episode lineage stays unrecoverable. |
| N6 | Manifest gains `timed_out_episodes`, disjoint from `truncated_episodes`. | New artifacts only. |

## Documentation-only corrections

- `model.py` — CDP is 8192-d (`configs.yaml:93`), not "4096-d (`rssm.py:140`)";
  the SPR EMA-equivalence claim is downgraded to unverifiable (A14);
  `JepaProjector` records both SPR deviations (A4); `ContinuationHeadMTP` is
  marked local, with its mean-vs-attention pooling asymmetry (A11).
- `objectives.py` — the missing t0 term is registered (A13); the LeJEPA comment
  now cites the verified `MINIMAL.md:174` and names the real deviation
  (multi-view invariance vs temporal prediction).
- `config.py` — the parameter-matching comment is marked stale for the live
  64/64 Craftax override (B2). `temporal.py` registers `use_mem_eff_path=False`
  (A9). `rollout.py`'s unreachable docstring is now the function's (B8).
  `checkpoint.py` states the implementation-digest coverage boundary and why
  widening it would be worse (A8).
- `ARCHITECTURE.md` — every validated deviation registered, plus a new `tested`
  field separating origin / tested-on-Craftax / selected-on-Craftax (C38).
- `SOURCE_MANIFEST.md` — rewritten: imported-and-verified vs read-reference vs
  **cited but absent** (Craftax_Baselines, rlpyt, LeJEPA's `je.py`).

## Retractions

- **B9 is withdrawn** — the convex SIGReg form is in the tracked
  `MINIMAL.md:171-174`. I searched for `scripts/je.py` and never opened the
  markdown.
- **A12 stands against my own round-1 §C** — Dreamer-CDP's cosine is global. I
  confirmed the spec's "per-token" claim without following `enc_output` to its
  definition.
- **A2, A3, A7 narrowed** — zero *loss* gradient is not frozen: rows of a dense
  gradient tensor still decay under AdamW. Only `grad is None` (A1) is skipped
  entirely. And causal BC on a 16-frame window does train context length 8; the
  A7 mismatch is distributional, not categorical.

## Deliberately NOT changed

- The `flow_x_head`, dead MAE mask token and degenerate shortcut embeddings
  (A1/A3) are registered, not removed — deleting them would fork the
  "unmodified MMBench2 `Dynamics`" claim and invalidate every checkpoint.
- `ContinuationHeadMTP`'s mean pooling (A11) and the `JepaProjector` widths
  (A4) are registered, not aligned — changing either invalidates head weights
  and is an experiment, not a fix.
- F1-F6 (the imagination pathology) are diagnosis, not defects with an obvious
  correct value. `horizon=32` vs a 5-step trained rollout, `context=8`, and the
  unbounded predictor output are all live design choices; changing them is an
  ablation. **F7 still stands: nothing here licenses attributing the T/M
  executed gap to the backend. `ABLATIONS.md` row 19 must be re-run to
  completion, now also clearing `WD*`.**
