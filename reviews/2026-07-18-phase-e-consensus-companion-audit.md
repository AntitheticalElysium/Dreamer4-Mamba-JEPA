# Phase-E consensus audit: task supervision, deployment routing, and sprint gate

Date: 2026-07-18
Repository HEAD audited: `24d5a7d`
Status: **independent review complete; planner remains NO-GO**

This review answers the companion's consensus request after independently
checking the committed Phase-E code, artifacts, checkpoint/data provenance,
the six fixed full-grid models, the relevant pinned source repositories, and
the local papers. It also reports three pre-registered fixed-model diagnostics
run after the original Phase-E result.

## Copy-ready consensus response

I concur with the broad route but not with the proposed first intervention as
stated.

1. **Concur:** Phase E is a genuine planner NO-GO for both backends. The next
   work should diagnose and repair task supervision under generated states,
   not restart topology search.
2. **Refute:** the original result does not establish “Mamba fails at h1 while
   GRU fails at h8.” Its 56-window horizon subsets contain respectively
   1/0/2/1 reward events at h1/h2/h4/h8, and each horizon scores a different
   target. A common-target evaluation over 3,262 transitions shows smooth
   degradation for every checkpoint: Mamba event AUROC .877 to .637 and GRU
   .888 to .648 from K0 to K8. The backend-specific story was a small-sample
   artifact; the planner failure was not.
3. **Refine:** reward events are rare by count (4.04% held out, 4.25% train),
   but zero steps do not dominate the converged reward objective. Events
   contribute 89.6% of Mamba and 84.1% of GRU held-out K0 reward NLL; the
   exact training replay schedules expose an event in roughly 83% of
   minibatches. Plain class-weighting is therefore not the clearest
   mechanism-matched first fix, and none of the inspected Dreamer-CDP,
   DRAMA, SPR, Dreamer 4, or Horizon Imagination implementations provides a
   canonical class-weighted planner-reward precedent.
4. **New positive diagnosis:** generated contexts are degraded but retain
   task information. Fresh frozen-context probes recover K8 event AUROC
   .707/.734, reward-sign AUROC .708/.745, and terminal AUROC .830/.858 for
   Mamba/GRU; all 18 K8 checkpoint-level confidence intervals exclude .5.
   The deployed heads, not a task-blind state, are the immediate binding
   failure.
5. **Approve as the first controlled intervention:** expose reward and
   continuation heads to every generated step, initially by head-only
   adaptation of the six frozen checkpoints at K=2. Run a separate factorial
   event-focused sampling or auxiliary event/sign arm. Evaluate the deployed
   naturally distributed reward probabilities without class weights.
6. **Do not merely change `rollout_steps` from 2 to 5.** The local bridge
   supervises only its final latent and never applies task heads to generated
   states. SPR supervises latent and reward predictions at every jump; Dreamer
   4 uses horizon-specific multi-token reward outputs. K=2 versus K=5 becomes
   meaningful only after the local objective has per-step latent and task
   targets.
7. **Backend:** use GRU as the primary path to the first planner episode and
   keep Mamba-2 as a matched research/thesis candidate. GRU is modestly better
   through intermediate imagined depths and its recurrent inference is about
   1.57x faster in the measured B=128 planner microbenchmark. This is not a
   license to train Mamba and deploy GRU: their learned states and recurrent
   caches are not interchangeable. A hybrid would require explicit
   distillation and complete revalidation.
8. **Fresh evaluation is mandatory.** Seeds 131-134 are now diagnostic/spent.
   A trained intervention cannot earn planner GO on them. Reserve a new
   canonical fork bundle and terminal set before fitting, and hash-pin them
   before looking at intervention results.
9. **Do not start the online policy sprint yet.** The production path still
   lacks a runner/planner, the “frozen” sprint config does not enforce encoder
   freezing, and the current checkpoint loader does not verify the supplied
   encoder identity or restore optimizer/RNG state. These are release
   blockers, not cosmetic issues.

The proposed reward-reweighting run is therefore **HOLD as a sole lever** and
**GO only as a separately labelled factorial control** alongside generated-
state per-step task supervision.

## 1. What was independently verified

### 1.1 Commit chronology and immutability

- `1295807` contains the Phase-E protocol and evaluator before terminal data
  collection or evaluation.
- `24d5a7d` adds the terminal cache and results.
- The evaluator file is byte-identical between the registered and result
  commits (SHA-256
  `fecc7ab275fd807971b1aa6b622b595f0b11654af7937f578d4590880edb71264`).
- Its recorded source digest reconstructs the tracked compact Python state at
  execution. The evaluator report correctly records `head=1295807`.
- Every original reported checkpoint scalar was recomputed. Maximum absolute
  discrepancy was `0.0`.

One protocol/implementation divergence remains: the text registers a pooled
environment-seed CI for G-E1, while the evaluator's initial Boolean gate asks
that all three per-training-seed CIs exclude zero. The result commit discloses
this and manually records both readings. Neither reading grants GO, so this
does not change the decision.

### 1.2 Pinned artifacts

| artifact | SHA-256 |
|---|---|
| held-out 20 episodes | `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad` |
| training replay | `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad` |
| terminal set 900-915 | `66ce53c2b7dd4439b5237dbc8f419db791fac91f92c546d9735703e6ddeee071` |
| fork bundle 131-134 | `3b45ac6b7360fd1f4cd3310d15ff517ccfc6f3f0bbdf2df62c151d980ccf4138` |
| canonical Step-1 encoder checkpoint | `0b37a2466c8915c6c700e6dca0cf000aa319eaca10ced970dc1beff8439e567f` |
| GRU 505 | `3aa852c22b21cdd87456eeab71c0e0b98e0ade34074a88a7f92fb8b08da420b7` |
| GRU 606 | `76bab037bfa8a7b01f579c3d5f654479253cd9ff152485e6e6c7bea99308164b` |
| GRU 707 | `ca366b1e0308848a75a62de7b20b5dbff9db2a28667fbdf9daed885c7d06a5a5` |
| Mamba-2 505 | `6524eff4abc286b85556686e0bd76c848a04993532e46adf730005e2a02a989b` |
| Mamba-2 606 | `9e29e16e41b237d911278f611ff61b2ab4c391741523d45aee70aefd672bbbaf` |
| Mamba-2 707 | `4fc0c8a89dfd9673e7f7b875834cedaea158fecdf4ef5a89761756ae6b1e78ec` |

All six online and target encoder tensors were compared directly with the
canonical Step-1 target encoder and are bit-identical. This is stronger than
trusting checkpoint metadata. The Phase-E report should nevertheless have
stored the held-out and checkpoint hashes itself.

### 1.3 G-E1 recomputation

Only 11 anchors across four environment seeds have reward-differing suffixes.
The reported pooled values reproduce exactly:

| backend | mean advantage | env-cluster 95% CI | two-level training+env 95% CI |
|---|---:|---:|---:|
| Mamba-2 | +.0222 | [-.0090, .0469] | [-.0292, .0782] |
| GRU | +.0545 | [-.0049, .1150] | [-.0198, .1285] |

The wider two-level interval is the appropriate warning when generalizing
across both learned models and environments. GRU is a credible near-miss, not
a pass. “Random” in `chosen-minus-random` means a uniform choice among the four
archived candidate suffixes; it is not a separately executed random-policy
episode and should be labelled accordingly.

### 1.4 G-E3 recomputation

The terminal cache contains 16 terminal episodes and the reported AUROC/Brier
values reproduce. The registered teacher-forced gate passes. It is weaker
than the prose implied:

- Mamba checkpoint terminal mean probabilities are .360/.383/.216 with
  recall at `P(term)>=.5` of .313/.375/.188.
- GRU values are .114/.127/.080 with recall .000/.063/.063.
- Brier skill over the empirical constant predictor is positive for all six
  models, but the registered absolute threshold of .20 is far looser than the
  approximately .032 terminal climatology Brier.

Thus G-E3 licenses “real-state terminal ranking is useful,” not “planner
continuation is calibrated.”

## 2. Severity-ranked audit

### BLOCKER 1 — task heads are never trained on generated states

In `model.py:1193-1198`, reward and continuation losses consume only
teacher-forced `context[:, 1:]`. In `model.py:1102-1152`, the rollout bridge
feeds back predicted latents but supervises only the final JEPA target. No
reward or continuation loss is applied inside that loop.

This matches the measured failure precisely: the heads work on real contexts,
then shrink reward and termination toward zero after generated transitions.
It also means increasing the existing rollout length is not the registered
SPR-like control.

### BLOCKER 2 — the frozen-encoder sprint contract is documented, not enforced

`sprint_candidate_config()` says “frozen encoder usage,” but `ModelConfig`
contains no freeze flag and a newly constructed candidate retains 321,504
trainable online-encoder parameters. The generic `world_update()` defaults to
`online_hybrid_recipe()`, backpropagates through that encoder, calls
`mark_parameters_updated()`, and EMA-updates the target.

Any runner using the advertised candidate config plus the generic update path
silently voids the Step-1 certificate and reopens the representation gates.
The runner must require an explicit phase contract, set encoder parameters
non-trainable, assert zero encoder gradients/optimizer membership, and disable
target EMA during the frozen phase.

### BLOCKER 3 — no runnable planner/collection/evaluation loop exists

`train.py` provides isolated world and actor-critic update functions, but no
collect/train/act/evaluate orchestration and no random-shooting/CEM planner.
Phase E evaluates archived suffixes; it does not execute a planner in Crafter.
The first planner episode remains a future milestone.

### BLOCKER 4 — new Crafter collection is not process-reproducible

`collect_terminal_enriched()` and `CrafterAdapter.reset()` omit the repository's
`crafter_canonical.canonicalize(env)` repair. Pinned Crafter resets the world
with `hash((seed, episode))`; the world also constructs some material/entity
sets whose iteration order depends on the Python process hash seed. I observed
different seed-903 episode lengths/hashes in fresh raw processes (239 versus
151 steps), while two canonicalized processes matched exactly (149 steps,
reward 4.1, identical digest).

The committed terminal tensor is hash-pinned, so the filed Phase-E numbers are
reproducible from that artifact. The claim that its seed alone regenerates the
same data is false. Existing replay/held-out caches should be treated as
artifact-defined and never silently regenerated. Every new train/evaluation
collector must canonicalize after every reset, verify a repeat digest in a
fresh process, and record the artifact hash.

### HIGH 1 — the original imagined-horizon diagnosis was underpowered

The 56 original windows contain one h1 event, zero h2 events, two h4 events,
and one h8 event. Different horizons score different reward transitions. That
sample supports a conservative fail when a metric is undefined/unstable; it
cannot identify backend-specific depth behavior.

### HIGH 2 — real-state continuation was mistaken for deployment evidence

G-E3 never evaluates the continuation head after generated transitions. The
same-target continuation diagnostic below shows catastrophic probability
collapse after one generated step for both families. Planner scoring
multiplies future rewards by these probabilities, so this is a direct
deployment failure.

### HIGH 3 — checkpoint “strict production/resumption” is overstated

`save_world_checkpoint()` stores a caller-supplied `encoder_sha256` without
deriving it from the actual encoder state. `load_world_checkpoint()` verifies
neither that value nor the stored model-source hash against the current
environment. It returns, but does not restore, optimizer/RNG state. Existing
tests even accept a fabricated 64-character encoder hash. Exact state loading
is strict; provenance and resumption are not yet strict.

The verification checkpoints also retain only a scalar JEPA `loss_history`;
component histories needed to diagnose reward optimization are absent.

### HIGH 4 — a backend cannot be swapped at deployment

Mamba-2 cache entries are the official `(conv_state, ssm_state)` tuples; GRU
uses hidden vectors. Their transition parameters and generated-state
distributions are learned jointly with their heads. “Mamba for training
throughput, GRU as deployment head” is not a configuration—it is an untrained
cross-model state transfer. Distillation could be studied, but it is a new
architecture/training experiment and must pass every task/planner gate.

### MODERATE — evidence documents now lag Phase E

The architecture ledger still says predictive quality is uncalibrated and the
specification banner overstates gate openness. It should record:

- real-state reward/continuation separation;
- generated-state reward and continuation collapse;
- retained task information under frozen probes;
- planner NO-GO;
- the local bridge's final-only divergence from official V-JEPA-2-AC code.

## 3. The discriminating experiments

All diagnostics use the six fixed checkpoints and cannot turn the old gate
into a GO. The complete registration, indexing contract, hashes, and outcomes
are in `reviews/2026-07-18-phase-e-depth-diagnostic-protocol.md`.

### 3.1 Same target, increasing generated suffix

Every K scores the same 3,262 targets (140 events, 20 episode clusters). A
fixed eight-real-frame base and fixed action sequence take 16 temporal updates;
only the final K transitions are replaced by generated states.

| family | metric | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | event AUROC | .877 | .780 | .701 | .662 | .637 |
| Mamba-2 | event AP | .508 | .280 | .135 | .093 | .074 |
| Mamba-2 | signed Pearson | .663 | .514 | .164 | .096 | .048 |
| GRU | event AUROC | .888 | .812 | .754 | .724 | .648 |
| GRU | event AP | .435 | .269 | .151 | .132 | .098 |
| GRU | signed Pearson | .634 | .457 | .207 | .174 | .142 |

Every checkpoint degrades. The original Mamba-707 h1 inversion disappears
(.754 on common K1 targets), and the original GRU h8 chance result becomes
.648 at the family level. Paired GRU-minus-Mamba AUROC is unresolved at K0
and K8, but modestly favors GRU at K1/K2/K4 (approximately +.032/+.053/+.062;
two-level CIs exclude zero at those depths).

The signed/magnitude failure is more severe than event AUROC alone reveals:
actual event magnitude averages .463, while mean absolute decoded event reward
falls from .229/.227 at K0 to .0048/.0193 at K8 for Mamba/GRU. An absolute
event score also cannot tell positive from negative reward, so it is
insufficient for planning.

### 3.2 Continuation on the same targets

The held-out cache has 14 recorded terminal transitions; six episodes were
capped at 200 transitions with continuation still one and were correctly not
relabelled.

| family | metric | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | terminal AUROC | .954 | .873 | .850 | .694 | .568 |
| Mamba-2 | Brier skill | .330 | .012 | .003 | -.003 | -.004 |
| Mamba-2 | mean P(term), terminal rows | .277 | .009 | .004 | .0006 | .00009 |
| GRU | terminal AUROC | .946 | .950 | .944 | .932 | .669 |
| GRU | Brier skill | .129 | .037 | .031 | .008 | -.004 |
| GRU | mean P(term), terminal rows | .086 | .027 | .022 | .0067 | .00010 |

Both backends are unusably overconfident in continuation under imagination.
GRU preserves rank longer, but by K8 both have negative skill against a
constant predictor.

### 3.3 Does the generated state still know the task label?

For each checkpoint and K, the world is frozen and a fresh depth-specific MLP
probe is trained only on the training replay. Held-out evaluation uses the
same common targets.

| family | probe | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | reward event AUROC | .909 | .800 | .753 | .711 | .707 |
| Mamba-2 | reward sign AUROC | .810 | .791 | .736 | .719 | .708 |
| Mamba-2 | terminal AUROC | .968 | .878 | .871 | .878 | .830 |
| GRU | reward event AUROC | .915 | .827 | .805 | .790 | .734 |
| GRU | reward sign AUROC | .824 | .787 | .794 | .757 | .745 |
| GRU | terminal AUROC | .958 | .945 | .937 | .932 | .858 |

All six checkpoints retain above-chance K8 separation for all three tasks with
episode-cluster intervals excluding .5. This is strong evidence for head
covariate shift and against “the generated representation has already erased
all task information.”

It is an upper bound, not a solved planner: the probes are depth-specific,
class-balanced, and discriminative rather than naturally calibrated. The
first control must determine whether one shared head can recover across depths.

## 4. Reward imbalance: what the evidence does and does not say

### Facts

- Held-out: 143/3,542 reward events = 4.04%.
- Training replay: 1,827/42,979 = 4.25%.
- Terminal data: 117/2,877 = 4.07%.
- Eligible same-target held-out rows: 140/3,262 = 4.29%.
- In exact B=4, 16,000-step replay schedules, 82.8-84.2% of minibatches
  contain at least one event.
- On 64 exact replay minibatches, reward is the largest measured loss
  component for both backends.
- Event rows contribute most of the reward NLL after training, despite their
  rarity.

### Consequence

There is a sparse-event **coverage/diversity** problem, especially for
positive/negative types and task strata. There is not evidence that the
unweighted objective is numerically dominated by zeros. Class-dependent
weighting of the only planner reward distribution also changes the population
optimum and can bias decoded expected rewards unless it is corrected and
recalibrated under the natural data distribution.

The safe control is therefore one of:

1. sample 50% uniform and 50% event-containing sequences for the **task-head
   update only**, retaining uniform dynamics data and natural-distribution
   evaluation; or
2. retain the unweighted two-hot planner head and add an auxiliary event and
   reward-sign objective.

Both must be compared with uniform generated-state task supervision. “Weighted
loss wins training NLL” is not a gate; calibrated signed reward and actual
ranking are. The 50/50 arm is **Dreamer-4-inspired, not source-faithful**:
Dreamer 4 defines relevance by accomplishment of registered Minecraft tasks,
whereas this Crafter control would define an event-containing sequence from
the replay labels.

## 5. Primary-source audit

All listed repositories are at the recorded commit with no tracked
modifications. Horizon Imagination has only untracked `__pycache__`
directories.

| source | exact commit | directly relevant finding |
|---|---|---|
| official Mamba | `f577286d052741c35d39cd43bdc3fad27120f22c` | `Mamba2.step()` mutates and returns the real convolution and SSM states; `allocate_inference_cache()` returns `(conv_state, ssm_state)`. Local recurrent use matches this API. |
| Dreamer-CDP | `a851fa3e3d70b624b094ee1810ad4bb602346092` | `dreamerv3/agent.py:229-234` applies ordinary reward-distribution and continuation losses on model features. No event class weights. |
| DRAMA | `a50bd54c34e77d1d13e988a031733a47817098e2` | `sub_models/world_models.py:658-669` uses unweighted symlog two-hot reward, BCE termination, and prior/posterior alignment. |
| SPR | `0b9dd4e7b9bbdfaecdf9a3713bf5931fb54ab0ca` | `src/models.py:449-467` predicts a reward at every jump; `src/algos.py:276-300` supervises every reward/latent jump and masks reset crossings. Paper default K=5. |
| official V-JEPA 2 | `204698b45b3712590f06245fbfba32d3be539812` | `app/vjepa_droid/train.py:425-449` computes teacher-forced and autoregressive losses over all generated positions; config uses `auto_steps: 2`. The local final-only bridge is a labelled divergence, not a close source reimplementation. |
| Dreamer-4 JAX reproduction | `8144b940d801971f12ec5633553b95001e555949` | `train_bc_rew_heads.py:424-455` uses horizon-indexed MTP reward outputs and unweighted symexp-two-hot CE over all valid horizons. This is a reproduction, not an official release. |
| Dreamer-4 PyTorch reproduction | `b8abafbf4da72c59b6aa09f8499ccde0d6a37fd6` | Provides the same broad MTP/symexp design family; also not an official Dreamer-4 release. |
| Horizon Imagination | `c79ec5e2450be22711c7d717e49326edf77061f2` | Reward/done heads train on clean trajectory segments while the generative model is trained for inference-like noisy/prefix conditions. No class-weighted reward precedent. |
| NE-Dreamer | `11cd3a978b83743f795cbfa81c2e095344912c17` | Uses explicit prior/posterior KL alignment. It highlights a mechanism absent from the deterministic local bridge; it does not justify reward reweighting. |
| Crafter | `e04542a2159f1aad3d4c5ad52e8185717380ee3a` | Installed package files are byte-identical to this source. `env.py:74` uses Python `hash()` in reset seeding, which makes canonicalized collection essential. |

Paper checks:

- Dreamer 4, `2509.24527v1.pdf`, SHA-256
  `8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb`,
  Eq. 9 uses MTP length 8 with one output layer per distance and symexp-two-hot
  rewards. For sparse Minecraft tasks it mixes 50% uniform and 50% relevant
  sequences; dynamics loss remains on uniform sequences to avoid optimistic
  generations. This supports event-focused sampling as a control, not blind
  class weighting.
- SPR, `2007.05929v4.pdf`, SHA-256
  `77ea8bcaf2a484982ac91031d66de43c07b8c1057023a9d1c7754e762dfdc151`,
  uses K=5 and truncates at episode boundaries. Its source confirms per-jump
  latent and reward targets.
- V-JEPA 2, `2506.09985v1.pdf`, SHA-256
  `9cfcfde5fb0d9730637da5b9e7317825c3f3d09e91f3553e22eeba42c74d2226`,
  uses a two-step rollout loss and a frozen encoder, but its action predictor
  and objective are not the local temporal/predictor split.
- On Training in Imagination, `2605.06732v2.pdf`, SHA-256
  `d5090958c236509a195949febba05be65e00413a54c92dffe616f812f1e02d00`,
  separates dynamics and reward errors and warns that systematic reward bias
  can survive imagination training. It reinforces the need for
  natural-distribution calibration; it does not license a weighted planner
  reward distribution.

## 6. Minimal controlled intervention matrix

### Stage 0 — correctness repairs, before fitting

1. Make “frozen phase” an executable invariant:
   - encoder excluded from optimizer;
   - `requires_grad=False`;
   - target EMA disabled;
   - assertions/tests for zero encoder gradients and bit identity.
2. Fix checkpoint provenance:
   - derive encoder-state digest rather than accept a string;
   - verify source/config/encoder digests on load;
   - provide an explicit optimizer/RNG restore operation;
   - save per-component loss histories.
3. Canonicalize Crafter after every reset and add a fresh-process repeat test.
4. Pre-register and collect fresh canonical evaluation bundles before any
   intervention result is inspected.

### Stage 1 — smallest mechanism test on fixed worlds

Freeze each of the six complete world models except reward/continuation heads.
Use training replay only, identical update counts and sampled rows.

| arm | generated-state task loss | event treatment | purpose |
|---|---|---|---|
| H0 | none; existing head | natural | fixed baseline |
| H1 | every K=1..2 generated step, shared head | natural | direct covariate-shift repair |
| H2 | same as H1 | 50/50 uniform/event-containing task batches | sparse-coverage control |
| H3 optional | same as H1 | auxiliary event+sign BCE, unweighted two-hot planner head | separate detection from magnitude |

Run both GRU and Mamba-2; pair replay schedules within training seed. Do not
update temporal, predictor, or encoder parameters. This is cheap enough to
answer the mechanism before another full world retrain.

Acceptance requires improvement on a fresh natural-distribution set in:

- event AP and AUROC;
- signed reward Pearson/Spearman;
- positive and negative event MAE/NLL and decoded means;
- continuation Brier skill, terminal AP/AUROC, and terminal probabilities
  after K=1/2/4/8 generated suffixes;
- direct suffix ranking and regret.

No arm may be selected solely on the probe set, old 131-134 bundle, training
loss, or one training seed.

### Stage 2 — if a shared head can recover

Retrain the full frozen-encoder world at K=2 with:

- latent target at every generated step;
- reward and continuation target at every generated step;
- boundary masking;
- task sampling arm selected only by Stage 1;
- GRU primary and Mamba-2 matched.

Only after this works should K=2 versus K=5 be tested. That comparison then
measures rollout depth rather than silently retaining a final-only loss.

### Stage 3 — if the shared head cannot recover

The depth-specific probe result makes Dreamer-4-style horizon-indexed reward
and continuation heads the next source-backed control. Compare a shared head
against K-indexed MTP heads under identical frozen contexts and data. Do not
change topology simultaneously.

Posterior/prior alignment or noise-conditioned state training is a later
control if per-step supervision still fails. It is not the smallest current
intervention.

## 7. Revised planner gate

The old gate should remain in the historical record but not be reused as the
only release criterion.

1. Use identical target transitions at every K.
2. Report AP as well as AUROC for sparse reward/terminal labels.
3. Reward gate must include signed correlation, positive/negative strata,
   event MAE/NLL, decoded calibration, and direct return ranking.
4. Continuation gate must use Brier **skill relative to climatology**, terminal
   AP/AUROC, and terminal probability/recall at imagined K—not only
   teacher-forced real states.
5. Use hierarchical uncertainty over environment and training seeds.
6. Pre-register the fresh bundle and thresholds before intervention results.
7. A planner GO ultimately requires planner-versus-random executed episodes,
   not only archived-suffix ranking.

## 8. Explicit GO / NO-GO table

| decision | ruling | reason |
|---|---|---|
| Task-supervision diagnosis | **GO** | common-target degradation plus recoverable frozen-context signal |
| Plain class-weighted planner reward as sole first lever | **NO-GO** | mechanism mismatch, calibration risk, no inspected canonical precedent |
| Event-focused sampling/auxiliary as factorial control | **GO** | sparse coverage is real; Dreamer-4 relevant-sequence mixture provides precedent |
| Head-only per-step generated-state task adaptation | **GO** | smallest discriminating, directly mechanism-matched experiment |
| Immediate K=2 to K=5 with current final-only bridge | **NO-GO** | not SPR-like and does not supervise deployed heads |
| GRU route to first planner episode | **GO, conditional on repaired gates** | stronger intermediate-depth task behavior and faster/lighter step inference |
| Mamba-2 matched thesis/research route | **GO** | retains K8 task signal; sequence-training throughput remains useful |
| Mamba-trained / GRU-deployed hybrid | **NO-GO** | incompatible learned state/cache; requires distillation |
| Predictor mixtures | **NO-GO / deferred** | no evidence this failure is multimodal uncertainty |
| Reliability weighting | **NO-GO / shadow-only** | still uncalibrated against held-out real rollout errors |
| Full online policy training | **NO-GO** | imagined task calibration and production runner invariants fail |

This preserves the larger project direction: Mamba remains an experimentally
serious thesis component, but it is not allowed to block the first honest
planner episode or inherit claims from a GRU deployment. The immediate
scientific question is now narrow and testable: can naturally calibrated task
heads learn the task mapping on the generated state distribution that the
fixed worlds already preserve?

## 9. Tests and new artifacts

Commands completed on the actual RTX 3060 environment:

```text
.venv/bin/python -m pytest -q m3_hjwm_compact/tests
105 passed, 1 warning in 61.94s
```

The warning is pre-existing (`float()` conversion of a gradient-bearing test
tensor). New indexing/provenance tests pass, and all filed aggregate metrics
were recomputed from raw rows.

New fixed-model diagnostics:

| artifact | SHA-256 |
|---|---|
| `phase_e_same_target_depth.json` | `d3dbd243b814fb3495a9bb77812de1da8480ba4a772beafc810ecafb9cb96fb8` |
| `phase_e_same_target_rows.json` | `2f3bb307f9a05d7acf0018829e08141af35c144d5290b2bff4deffd777091f8c` |
| `phase_e_same_target_continuation.json` | `e0a470d7b5d2b5e4d93893bd0b6bc868f269d7fb7b0470e71f84f008c22ea762` |
| `phase_e_same_target_continuation_rows.json` | `21be653be57c6b05c6d1e36243d8ac35b6e4380982a02f6938c9a76ad624bc18` |
| `phase_e_context_probe.json` | `c676a3fe84e173e25ef9f58a587fb633a48a997e7c0fb07af254af0f0eeaf7ed` |
| `phase_e_context_probe_rows.json` | `323b6717a683f8ce50a572ef45a2ce7af3fa890cf10f3a1f125884ceb0a63943` |

Measured peak CUDA memory for the diagnostics was 94-155 MiB allocated
(136-212 MiB reserved), depending on backend/probe. A separate full
`imagine_step` microbenchmark at B=128 measured:

| backend | step time | peak allocated | peak reserved |
|---|---:|---:|---:|
| Mamba-2 | 2.601 ms | 83.37 MiB | 102 MiB |
| GRU | 1.655 ms | 48.29 MiB | 70 MiB |

These measurements do not overturn the prior sequence-training result:
Mamba-2 trains the full-grid sequence approximately 1.4x faster in the
existing 16K runs. They do overturn any assumption that it is also the faster
recurrent planner backend at this scale.
