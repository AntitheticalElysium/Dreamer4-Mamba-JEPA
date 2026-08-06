# Independent audit: Stage-1 head adaptation and next-stage routing

Date: 2026-07-18  
Committed repository audited: `b48618ed5cb8896bfe10182d2b3597f830473f0f`  
Status: **audit complete; generated-state head supervision validated; H2 and
full-world Stage 2 remain HOLD; planner remains NO-GO**

This report is the copy-ready response to the companion. It distinguishes:

1. whether the committed experiment is correct and reproducible;
2. what its fixed-arm results literally show;
3. what the missing equal-update controls show causally;
4. whether H2 is calibrated enough to become the default;
5. whether full-world K2 retraining is now the smallest justified next step.

## Copy-ready consensus response

I concur with the central diagnosis, but not with the breadth of the filed
claim or the immediate Stage-2 route.

1. **The committed result is real and reproducible.** Commit chronology is
   valid, all three fresh evaluation artifacts were pinned before fitting,
   all six H2 evaluations reproduce exactly, two independently repeated H2
   fits are tensor-bit-exact, and H1/H3 planner heads are bit-identical in all
   six cases. I found no transition-indexing leak, target-encoder update, stale
   recurrent state, or non-head parameter drift.
2. **Generated-state task supervision is a causal mechanism, especially for
   deep task prediction.** The missing equal-update all-real factorial now
   exists. Relative to its matched all-real control, generated supervision
   improves Mamba K8 reward AUROC in 3/3 seeds, terminal AUROC in 3/3, and
   ranking advantage in 3/3. GRU improves K8 reward Pearson, reward sign
   AUROC, terminal AUROC, and event magnitude in 3/3, but its reward AUROC and
   planner-ranking effects are heterogeneous. The defensible claim is
   “generated-state exposure repairs deep deployed task prediction”; it is not
   “generated-state exposure caused every planner-ranking pass.”
3. **The absolute H2 ranking claim is exact but is not the causal contrast.**
   H2 advantage is positive and its environment-cluster CI excludes zero in
   all six fixed runs. Against equal-update event-sampled real controls,
   however, the paired ranking improvement excludes zero in only 2/3 Mamba
   seeds and 1/3 GRU seeds; one GRU seed is exactly unchanged. Against H1, no
   per-seed H2 ranking delta has a strictly positive paired CI.
4. **“H2 wins the full acceptance list, both backends, all seeds” is
   refuted.** Versus H0, H2 improves 137/168 registered reward comparisons and
   82/96 continuation comparisons, but worsens 31 and 14 respectively. It
   also worsens GRU-606 ranking and regret. Most discrimination regressions
   occur at K1.
5. **H2 demonstrates a reward-amplitude/false-reward tradeoff, not calibrated
   sparse-event recovery.** At K8 it raises event magnitude and Pearson, but
   family-mean zero-reward MAE rises about 13x for Mamba and 8x for GRU versus
   H0. On fork suffixes whose actual return is exactly zero, mean absolute
   predicted eight-step reward rises from .024 to .136 for Mamba and .023 to
   .173 for GRU versus H1. Every H2-minus-H1 seed has a cluster CI excluding
   zero for this false-reward increase; none has a strictly positive paired
   ranking CI. H2 is therefore a useful diagnostic arm, not a selected
   planner head.
6. **The event intervention does not isolate reward class imbalance.** Its
   realized mixture is about 54% event-containing, not exactly 50%, because
   the uniform half can also contain events. Across all nine supervised labels
   it raises event frequency from about 3.8% to 10.9%, terminal frequency by
   roughly sevenfold, and reduces unique sampled windows from about 18.2K to
   12.8K. The same event schedule trains the continuation head, so reward
   coverage and terminal coverage are coupled.
7. **The calibration caveat should be strengthened.** H2 continuation Brier
   skill is merely around zero and remains negative in 4/6 runs. The fresh
   reward set's true absolute event magnitude is `.3865`, not `.46`; H2 reaches
   11.4% of it for Mamba and 14.5% for GRU. Mamba's mean decoded prediction on
   truly negative events is still positive at K8 (`+.0144`).
8. **Full-world Stage 2 is not yet the uniquely justified smallest lever.**
   A registered K2-versus-per-step-K8 head-only diagnostic shows that deeper
   supervision still improves K8 reward Pearson, magnitude, sign separation,
   and Mamba ranking. It also increases false reward, and GRU significantly
   loses K1 event AUROC/Pearson. Thus depth mismatch is still binding, while a
   single shared head exhibits cross-depth conflict.
9. **Primary sources provide the missing next control.** SPR applies one
   shared reward predictor at every recurrent jump. Dreamer 4 instead uses
   multi-token reward prediction of length 8 with a separate output layer per
   forecast distance. The earlier local probes were also independently fitted
   per depth. A small depth-indexed/MTP task-head control, plus natural
   recalibration, is therefore source-backed and cheaper than retraining the
   whole world.
10. **If Stage 2 follows, fix its loss routing.** Apply latent/dynamics loss
    only to uniform replay. Keep reward-event sampling as a separate reward
    loss arm and do not apply it implicitly to continuation. Construct
    terminal-aligned continuation examples per generated depth and mask after
    episode boundaries. Run GRU primary and Mamba matched; do not swap
    backends at deployment.
11. **Planner and online policy remain NO-GO.** The Stage-1 bundle is now
    spent for selection. A new planner gate needs naturally calibrated reward
    and continuation, a fresh hash-pinned execution set, and executed
    planner-versus-random episodes. Reliability remains shadow-only.

This is meaningful progress: the evidence now locates the immediate failure in
task-head distribution/depth/calibration, not in a missing action signal or a
proven-bad temporal core. It does not yet establish a runnable agent.

## 1. Source and commit control

| Source | Exact identity used | What it establishes here |
|---|---|---|
| Local Stage-1 registration | `eeca3ab` | H0/H1/H2/H3 and acceptance list existed before fresh collection/fitting |
| Fresh Stage-1 bundles | `c557166` | natural 940-955, terminal 916-931, fork 135-142 pinned before fit |
| Stage-1 runner | `d6fd700` | executable head-only implementation |
| Stage-1 outcome | `b48618e` | fixed artifacts and filed claims |
| SPR paper | `2007.05929v4.pdf`, SHA-256 `77ea8bcaf2a484982ac91031d66de43c07b8c1057023a9d1c7754e762dfdc151` | recurrent shared reward prediction and latent self-prediction at every jump |
| SPR source | `mila-iqia__spr` commit `0b9dd4e7b9bbdfaecdf9a3713bf5931fb54ab0ca` | `models.py:438-469` unrolls all jumps; `algos.py:269-303` supervises all jump rewards and masks SPR after terminals |
| V-JEPA 2 paper | `2506.09985v1.pdf`, SHA-256 `9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226` | frozen encoder, teacher forcing, T=2 autoregressive rollout objective |
| V-JEPA 2 official source | `facebookresearch__vjepa2` commit `204698b45b3712590f06245fbfba32d3be539812` | `app/vjepa_droid/train.py:425-447` computes teacher-forced and autoregressive predictions and scores the autoregressive sequence |
| Dreamer 4 paper | `2509.24527v1.pdf`, SHA-256 `8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb` | L=8 MTP reward/policy heads; 50/50 uniform/relevant task mixture; dynamics loss only on uniform data |
| Dreamer 4 PyTorch source | `nicklashansen__dreamer4` commit `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6` | explicitly unofficial/incomplete; not ground truth for the paper's task mixture |
| Dreamer 4 JAX source | `edwhu__dreamer4-jax` commit `8144b940d801971f12ec5633553b95001e555949` | explicitly unofficial; corroborates MTP head shape, but its local default L=2 differs from the paper's L=8 |
| Official Mamba source pin | `state-spaces__mamba` commit `f577286d052741c35d39cd43bdc3fad27120f22c` | recurrence/cache authority from the earlier backend gate |
| Installed Mamba-2 files used by Stage 1 | `mamba2.py` SHA-256 `605e4439...8707`; `ssd_combined.py` SHA-256 `0b7c4cfa...919d` | matches the recorded runtime source identities |
| Crafter source pin | `danijar__crafter` commit `e04542a2159f1aad3d4c5ad52e8185717380ee3a` | canonical collection implementation and environment semantics |

Important source interpretation:

- Calling H2 **Dreamer-4-inspired** is fair and was correctly disclosed.
  Calling it faithful would be false. Dreamer 4's relevant sequences
  accomplish annotated Minecraft tasks, its reward is task-conditioned, and
  its dynamics loss is restricted to uniform data. H2 selects any signed local
  reward in two positions and jointly updates reward and continuation.
- A future local per-step shared-head objective is **SPR-shaped**, not a
  reproduction: SPR jointly trains an RL agent, convolutional transition
  model, reward predictor, and representation targets.
- A depth-indexed head is **Dreamer-4/MTP-inspired** unless it matches the
  paper's task conditioning and MTP semantics. The source still supplies a
  concrete control invariant: forecast distance has its own output layer.

## 2. Commit, artifact, and regeneration verification

### 2.1 Chronology

The chronology is valid:

1. `eeca3ab` registers Stage 1.
2. `c557166` commits fresh bundle tensors and hashes.
3. `d6fd700` commits the runner.
4. `b48618e` commits heads, report, and outcomes.

`stage1_head_adaptation.py` is byte-identical from `d6fd700` through the
result commit (SHA-256
`4bed7aeb5305d31c88f4ad3c1d061c952f96e4fc72dd0f14d51c75ac20d5f7a9`).
The report records `head=d6fd700` and the correct tracked-source digest.

The working tree was clean at the companion handoff. The present uncommitted
changes are this independent audit, its tests, controls, and the exact-resume
repair.

### 2.2 Fresh artifacts

| Artifact | SHA-256 | Independent check |
|---|---|---|
| natural 940-955 | `f10af55a5c1c29f632f5c21e73af470f6bc7d9593f2ed1c1b1be0c953435d629` | exact semantic regeneration |
| terminal 916-931 | `718f7b9ad7bd451ae5479329e10aaa8c6e16126c601b3c78bbb26c8b3e721153` | exact semantic regeneration |
| fork bundle 135-142 | `08c48e44e9e67ce82ebe9f7303fc7d8072bd0d2b78b9905afcd7fc4641f185f2` | exact regeneration with repeat verification |
| training replay | `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad` | report/control match |

The manifest is underspecified: it records absolute paths, hashes, counts, and
one global `canonical=true`, but not collector arguments, source SHA, exact
seed lists, caps, or repeat digests. That is a provenance weakness, not a
result invalidation, because all three tensors independently regenerate
exactly under the repaired canonical collectors.

Fresh natural same-target evaluation contains 2,372 rows and 133 reward events:

- true mean absolute event magnitude: `.386466`;
- true positive mean: `+.605263`;
- true negative mean: `-.222368`.

Fresh terminal evaluation contains 2,426 rows and 16 terminal targets.
The fork bundle contains 48 anchors, of which 24 have reward-differing
candidate suffixes.

### 2.3 Exact result reproduction

- Every scalar in the six reported H2 evaluations was rerun from the committed
  base+head states. Maximum absolute discrepancy: `0.0`.
- Mamba-505 H2 and GRU-606 H2 were independently refitted from their base
  checkpoints. Reward and continuation head tensors are bit-identical to the
  committed files; non-head state digests remain exact.
- H1 and H3 reward/continuation state digests are identical in all six
  base/seed cases. This confirms the companion's vacuous-H3 diagnosis and the
  paired schedule.
- The original head files contain CUDA-tagged tensors and require
  `map_location` for CPU loading. The new control checkpoints store task heads
  on CPU.

## 3. Code audit

### 3.1 Correct indexing and no leakage

The replay convention is:

`(o_t, a_t) -> (o_{t+1}, r_t, c_t)`.

For a window beginning at `start`:

- observing `obs[t]` uses `action[t-1]` as the previous action;
- the post-observation context at local `t>=1` predicts
  `reward[t-1], continue[t-1]`;
- after real observations 0...7, imagining `action[7]` predicts the context
  and task targets for transition 7 (K1);
- imagining `action[8]` predicts transition 8 (K2).

The H2 event index checks exactly transitions 7 and 8. Synthetic regression
tests now pin these alignments and the K8 extension. Training reads only the
pinned replay; none of the fresh evaluation tensors enters fitting.

Each training batch starts from a new recurrent state. Each fork candidate
starts from a cloned recurrent state. The existing recurrent cache-isolation
and sequence/step tests pass. No stale-cache path was found.

### 3.2 Freeze contract

`freeze_world_except_heads()` disables every world parameter, reenables only
reward/continuation, and invokes the executable frozen-encoder contract.
Independent retraining and both new controls show bit-identical non-head state
digests. The world contains no mutable BatchNorm running state on this path.
There is no target-encoder EMA update.

### 3.3 H3

H3's auxiliary event/sign heads share no trainable parameter with the planner
heads under the frozen trunk. Their optimizer states are independent; the
global gradient clip never bound in the observed run. Exact H1/H3 planner-head
identity confirms that H3 is not evidence for or against an auxiliary
objective.

A meaningful auxiliary test requires:

- a shared trainable adapter/trunk;
- the same adapter without auxiliary losses as its matched control;
- fresh or explicitly diagnostic evaluation.

### 3.4 Original provenance gaps

The original runner:

- writes no raw target predictions or ranking rows;
- writes no per-result script/protocol/checkpoint/schedule hashes;
- records no trainable-name list, non-head digest, wall time, or peak VRAM;
- saves GPU-tagged head tensors;
- skips an existing result tag without verifying that its source, base, and
  head artifact match, so an interrupted rerun could mix stale entries.

These do not overturn this run because exact refits/evaluations succeeded.
They should be repaired before reuse. The Stage-1b/1c artifacts implement the
stronger provenance contract.

### 3.5 Checkpoint resumption defect

`save_world_checkpoint()` saved an explicit NumPy `Generator` state, but
`restore_optimizer_and_rng()` restored only Torch CPU/CUDA state. Thus a
resumed replay schedule could diverge while the helper claimed exact
resumption.

The helper now:

- requires the corresponding NumPy generator when its state is present;
- restores its bit-generator state;
- rejects a saved CUDA RNG state when CUDA is unavailable;
- validates these requirements before mutating optimizer or Torch RNG state.

A regression test pins both exact restoration and failure atomicity.

## 4. Severity-ranked findings

### BLOCKER 1 — H2 is not a calibrated planner-head selection

H2 improves event NLL and magnitude by placing more reward mass away from
zero. The decoded reward used by the planner also becomes falsely nonzero on
ordinary transitions and zero-return suffixes.

Family means at K8:

| family | arm | reward AUC | Pearson | event magnitude | zero MAE | overall MAE | overall NLL | ranking advantage |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Mamba | H0 | .586 | .031 | .0044 | .0018 | .0232 | .548 | .0769 |
| Mamba | H1 | .633 | .063 | .0079 | .0034 | .0247 | .360 | .1481 |
| Mamba | H2 | .654 | .144 | .0442 | .0229 | .0423 | .305 | .1463 |
| GRU | H0 | .588 | .146 | .0137 | .0037 | .0245 | .488 | .1176 |
| GRU | H1 | .611 | .166 | .0118 | .0042 | .0251 | .326 | .1440 |
| GRU | H2 | .642 | .182 | .0562 | .0296 | .0474 | .292 | .1477 |

The NLL improvement is real; the decoded-expectation error used for planning
gets worse. H2-minus-H1 mean absolute predicted return on truly zero-return
fork suffixes increases by `.112` for Mamba and `.150` for GRU. Every seed's
cluster interval excludes zero.

**Required decision:** H2 remains an ablation/shadow checkpoint. It is not the
Stage-2 default and cannot grant planner GO.

### BLOCKER 2 — the proposed Stage-2 sampling route is not source-correct

Dreamer 4 uses the relevance mixture to amplify task losses and applies its
dynamics loss only on uniform sequences to avoid optimistic generation.
Applying a full local latent+dynamics+reward+continuation objective to H2
batches would:

- train dynamics disproportionately on event/terminal neighborhoods;
- entangle sparse reward supervision with continuation;
- reduce replay diversity;
- make attribution to per-step targets impossible.

A corrected schematic objective is:

`L_uniform = L_latent_per_step + lambda_r L_reward + lambda_c L_continue`

`L_total = E_uniform[L_uniform] + alpha E_event[L_reward_only]`

The event term is a registered factorial, not an unconditional default. A
final natural-distribution calibration phase or calibrated head must still be
tested.

### HIGH 1 — filed acceptance and causal language exceed the evidence

Literal H-arm-versus-H0 direction audit:

| arm | reward improved/equal/worsened | continuation improved/equal/worsened | ranking improved/equal/worsened |
|---|---:|---:|---:|
| H1 | 134 / 0 / 34 | 82 / 0 / 14 | 10 / 2 / 0 |
| H2 | 137 / 0 / 31 | 82 / 0 / 14 | 10 / 0 / 2 |

H2's 31 reward regressions are concentrated at shallow depth: 17 occur at K1.
They are discrimination/correlation regressions, not event NLL/magnitude
regressions.

The honest claim is “broad partial recovery, strongest at deep task readouts.”

### HIGH 2 — H0 was not an equal-update mechanism control

H0 receives no new updates. H1/H2 receive 3,000 updates, and seven of their
nine targets are teacher-forced real contexts. H1-H0 therefore cannot alone
identify generated exposure.

Stage-1b completes the 2x2:

| sampling | all-real, equal updates | 7 real + K1/K2 generated |
|---|---|---|
| natural | R1 | H1 |
| event-focused | R2 | H2 |

Family-mean K8 contrasts:

| family | contrast | reward AUC | Pearson | terminal AUROC | ranking advantage |
|---|---|---:|---:|---:|---:|
| Mamba | H1-R1 | +.061 | +.029 | +.196 | +.073 |
| Mamba | H2-R2 | +.088 | +.109 | +.200 | +.067 |
| GRU | H1-R1 | +.011 | +.034 | +.087 | +.031 |
| GRU | H2-R2 | +.043 | +.030 | +.026 | +.056 |

The point signs support generated-state exposure. Cluster intervals show the
effect is strongest and most consistent on Mamba/deep terminal metrics:

- H1-R1 Mamba terminal-AUROC intervals exclude zero in 3/3; reward-AUROC in
  2/3; ranking in 1/3.
- H2-R2 Mamba terminal and reward-AUROC intervals exclude zero in 3/3;
  ranking in 2/3.
- H1-R1 GRU terminal intervals exclude zero in 2/3; reward-AUROC in 1/3;
  ranking in 1/3.
- H2-R2 GRU terminal, reward-AUROC, and ranking exclude zero in 1/3 each; one
  ranking delta is exactly zero.

Thus the mechanism is validated for deployed task prediction, not uniformly
for action ranking.

### HIGH 3 — event sampling and continuation are confounded

The H2 sampling pool contains:

- 3,282 event windows among 40,867 uniform-eligible windows (`8.03%`);
- approximately equal incidence of positive-event and negative-event windows;
- terminal transitions in `6.58%` of event-pool windows.

Realized schedules:

| arm | event-containing windows | unique windows | event fraction over all 9 labels | terminal fraction over all 9 labels |
|---|---:|---:|---:|---:|
| H1 | 7.9-8.3% | 18,163-18,203 | 3.74-3.82% | .054-.058% |
| H2 | 53.95-54.14% | 12,739-12,860 | 10.87-10.95% | .369-.408% |

Because continuation uses the same sampled windows, H2 does not isolate
reward coverage. In Stage 2, reward and continuation sampling must be
separate.

### HIGH 4 — a single shared head has measurable cross-depth conflict

The Stage-1c diagnostic starts from H0, uses a common natural schedule, and
compares shared-head supervision through K2 versus every generated step
through K8 on seed 505.

Paired D8-minus-D2 findings:

| family | readout | delta | episode/env-cluster 95% CI |
|---|---|---:|---:|
| Mamba | K8 Pearson | +.0629 | [.0286, .1134] |
| Mamba | K8 event magnitude | +.0083 | [.0061, .0108] |
| Mamba | ranking advantage | +.0361 | [.0080, .0667] |
| Mamba | zero-suffix absolute predicted reward | +.0258 | [.0175, .0333] |
| GRU | K8 Pearson | +.0611 | [.0211, .0959] |
| GRU | K8 event magnitude | +.0075 | [.0051, .0099] |
| GRU | ranking advantage | +.0111 | [-.0077, .0435] |
| GRU | zero-suffix absolute predicted reward | +.0295 | [.0201, .0378] |
| GRU | K1 event AUROC | -.0312 | [-.0604, -.0024] |
| GRU | K1 Pearson | -.0263 | [-.0623, -.0002] |

Absolute K8 event magnitude remains only `.0112` for Mamba and `.0140` for
GRU, respectively 2.9% and 3.6% of truth. D8 helps deep mapping but does not
calibrate it, and GRU trades shallow for deep behavior.

This is exactly the regime in which the source-backed horizon-indexed/MTP head
control is warranted.

### HIGH 5 — continuation-depth training needs its own boundary design

The common 16-observation Stage-1c schedule structurally contains:

- D2: zero terminal labels;
- D8: 131 terminal labels, all at generated depth 8.

An episode ends at its terminal transition, so a common unpadded K8-eligible
window cannot place that terminal at K1/K2. D8-D2 continuation therefore
mixes depth coverage with first terminal exposure and is not a clean
horizon-only contrast.

Required continuation curriculum:

1. choose terminal transitions explicitly;
2. construct a separate prefix so the same terminal type is the target at
   each K;
3. include matched natural non-terminal contexts;
4. mask every loss after the terminal boundary;
5. evaluate on the untouched natural terminal distribution.

Reward-event sampling must not silently stand in for this curriculum.

### MODERATE 1 — absolute CIs were read as intervention CIs

The reported H2 CI asks whether H2's absolute chosen-minus-uniform-candidate
advantage is positive. It does not ask whether H2 improved over H0/H1/R2.
Several H0 and all-real control runs already have positive absolute intervals.

The equal-update report retains raw paired rows and uses environment-cluster
bootstrap deltas. Model-training seed uncertainty remains only three points;
family claims should retain per-seed signs rather than invoke asymptotic
training-seed inference.

### MODERATE 2 — magnitude denominator and signed failure

The filed “10-12% of actual” uses the older `.46` magnitude. On the fresh
Stage-1 target rows:

- actual absolute event mean: `.3865`;
- H2 Mamba: `.0442` = 11.4%;
- H2 GRU: `.0562` = 14.5%.

Absolute magnitude also hides sign. Mamba H2's K8 negative-event conditional
mean is `+.0144`, not negative.

### LOW — “regret down ~33%” is backend-specific

Relative to H0:

- Mamba family regret falls about 33.2%;
- GRU family regret falls about 17.9%;
- combined reduction is about 26%.

The qualitative improvement remains; the unqualified number does not.

## 5. New controlled artifacts

The audit-local protocols were written before their implementations/runs and
their hashes were captured by the outputs. They were not committed before
execution, so this is weaker preregistration evidence than the original
Stage-1 commit chronology. They are diagnostics on spent data and cannot
grant planner GO.

### 5.1 Equal-update factorial

| Artifact | SHA-256 |
|---|---|
| protocol | `9cdaabb63336c26b80cb3a64ceb1feee6574971fa25abde7fbbeb77ad4dc259b` |
| runner | `7313956cc9d85b9077b0b9a6702664e57303bfe1ab23bde18f82539e221cb0d2` |
| report | `13cb0d216ada42fcff3205f6e50782cb6eff5305bd0d78d7327c73d2dfdb7931` |
| raw paired rows | `5ff39d1fc28d44e493abb7a01289dc99f5b06c09190bcc6b49961b8d9a716a75` |
| paired analysis | `7b6885309a3d5d48ffe05fc0703e9d286d01e5d4ac48f105bdec92740f3c1c59` |

All 12 R checkpoints have bit-identical pre/post non-head digests and verified
head-file hashes.

### 5.2 Head-depth ceiling

| Artifact | SHA-256 |
|---|---|
| protocol | `d4c13f654bfbb2e761cff53aaee69ae994c78a379b920c69877ed090c1c211ed` |
| runner | `93e7b39f04ca7b6f5b0926b9d4e91b197fbcab8549767d24083fcb318d250a6d` |
| report | `b7240bd951158b0796558654395f32275c93ac46c6fa2e6b64a3a837c25511ee` |
| raw paired rows | `b579a4ce10a7e1c03f663b1b6aba730640163ef43ffc2757ed8bc9c07108c655` |
| paired analysis | `08fac1ee2a1ee84f5edb32b24682a08a310bc393bb14083dfd24df4eb09fc5e4` |

All four D checkpoints have bit-identical non-head digests and verified
head-file hashes.

## 6. Tests and measured memory

Tests after the audit changes:

- compact CPU/default suite: **92 passed, 19 CUDA-skipped**;
- targeted CUDA production/cache/checkpoint/Stage-1 suite:
  **13 passed**;
- compact end-to-end smoke test: **passed**;
- new Stage-1 indexing/sampling/freeze/resume/depth tests:
  **5 passed** within the suites above;
- `git diff --check`: **passed**.

The one existing warning is the known conversion of a grad-bearing SSL test
loss to `float`; it does not affect this audit.

Measured peak VRAM during head training:

| phase | backend | peak allocated | peak reserved |
|---|---|---:|---:|
| Stage-1b R1/R2 | Mamba-2 | 39.73 MiB | 56 MiB |
| Stage-1b R1/R2 | GRU | 37.59 MiB | 56 MiB |
| Stage-1c D2 | Mamba-2 | 39.55 MiB | 54 MiB |
| Stage-1c D8 | Mamba-2 | 39.58 MiB | 54 MiB |
| Stage-1c D2 | GRU | 37.41 MiB | 54 MiB |
| Stage-1c D8 | GRU | 37.44 MiB | 54 MiB |

Training wall time on the recorded RTX 3060 Laptop GPU:

- Stage-1b Mamba: about 100-101 seconds per 3K-head run;
- Stage-1b GRU: about 70-72 seconds;
- Stage-1c D8: 148 seconds Mamba, 102 seconds GRU.

This phase is nowhere near the 6 GB memory limit. Mamba remains slower for
these short recurrent head-adaptation sequences.

## 7. Architecture and objective corrections

### 7.1 Retain the stable world; change the immediate hypothesis

The frozen representation is not the present binding failure:

- the fork oracle showed substantial unused predictive headroom;
- fixed depth-specific probes recover task information at K8;
- equal-update controls show generated-state supervision repairs deployed
  task metrics;
- D8 head-only exposure yields further deep improvement.

Do not restart topology search. The next hypothesis is:

> task mapping is depth-dependent and poorly calibrated under a single shared
> head and biased event sampling.

### 7.2 Minimal source-backed head matrix

Use one fixed seed per backend first, then replicate only a discriminator:

| arm | head | sampling | purpose |
|---|---|---|---|
| C0 | current shared head, per-step K8 | natural | reproduced Stage-1c reference |
| C1 | depth-indexed output layers or depth embedding, per-step K8 | natural | test shared-head cross-depth conflict; MTP-inspired |
| C2 | C1 + reward-event auxiliary batch | event term on reward only | test magnitude without continuation confound |
| C3 | C2 followed by natural head calibration | natural final phase | test false-reward repair |

Acceptance must include both benefit and harm:

- K1 and K8 event AUROC/AP and signed correlation;
- positive and negative conditional decoded means;
- event and zero MAE/NLL;
- cumulative predicted reward on truly zero-return suffixes;
- continuation evaluated separately;
- paired ranking and regret.

If C1/C3 cannot use the probe headroom, full-world Stage 2 becomes justified.
If they can, preserve the world and proceed sooner to the planner gate.

### 7.3 Correct Stage-2 contract, if needed

- encoder remains frozen and target EMA disabled;
- uniform replay supplies every latent/dynamics target;
- every generated step receives a boundary-masked latent target;
- natural replay supplies shared reward/continuation task loss;
- event-focused reward is a separate factorial term;
- terminal-aligned continuation is a separate stratified term;
- no loss is applied after termination;
- GRU is the primary sprint path; Mamba-2 is the matched thesis backend;
- no training-backend/deployment-backend swap;
- keep checkpoints separately runnable.

K=2 is a smallest full-world implementation check, not evidence that K=2 is
the right deployed horizon. SPR's longer per-jump supervision and the present
D8 result justify a registered K2-versus-longer control after the K2 code path
passes correctness and memory tests.

## 8. Explicit go/no-go decisions

| Decision | Ruling | Reason |
|---|---|---|
| Stage-1 code/result correctness | **GO** | chronology, indexing, fresh data, exact eval/refit checks pass |
| Generated-state task supervision | **GO** | equal-update controls validate deep task-prediction benefit |
| “H2 wins full acceptance list” | **REFUTE** | 31 reward, 14 continuation, and 2 ranking comparisons worsen |
| H2 as default sampler/head | **NO-GO / HOLD** | false reward, terminal confound, no paired H2-vs-H1 ranking pass |
| Natural shared H1 as diagnostic | **GO** | conservative operational repair; not calibrated planner approval |
| Depth-indexed/MTP head control | **GO, next** | primary-source backed, cheap, directly matches depth-specific probe evidence |
| Full-world Stage 2 as immediate next run | **HOLD** | not yet uniquely identified; loss routing needs correction |
| GRU primary sprint path | **GO** | faster, stronger continuation; still requires the same gates |
| Mamba-2 matched thesis path | **GO** | retained; Stage-1 effect is real and often stronger vs equal-update controls |
| Predictor mixture | **DEFER** | no new evidence that mode diversity estimates uncertainty |
| Reliability weighting | **NO-GO; shadow-only** | no held-out calibration change |
| Planner execution gate | **NO-GO** | reward/continuation calibration and fresh execution evidence absent |
| Full online actor/critic training | **NO-GO** | no executed planner-vs-random pass |

## 9. Required handoff to the next researcher

1. Review the new control code/artifacts and reproduce the paired summaries.
2. Amend the architecture evidence ledger with:
   - generated-state head supervision validated for deep task prediction;
   - H2 false-reward/terminal-sampling tradeoff;
   - shared cross-depth head conflict;
   - planner still NO-GO.
3. Register the small C0/C1/C2/C3 head matrix before implementation.
4. Keep Dreamer-4/MTP-inspired and SPR-shaped variants explicitly labelled.
5. If Stage 2 proceeds, enforce the corrected uniform/event/continuation loss
   separation and retain GRU/Mamba as separately runnable checkpoints.
6. Reserve and hash-pin a new evaluation/execution bundle before fitting the
   selected intervention.
7. Do not use any Stage-1/1b/1c bundle to grant planner GO.
