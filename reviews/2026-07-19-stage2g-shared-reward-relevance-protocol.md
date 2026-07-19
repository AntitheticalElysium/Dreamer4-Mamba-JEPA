# Stage-2G protocol: shared-path sparse-reward relevance factorial

Date: 2026-07-19

Status: **pre-registered before implementation, gradient-scale measurement,
smoke training, full training, or evaluation**

## Question

Stage-2C established a trade-off:

- generated latent supervision alone (C-L) improves latent rollout but loses
  counterfactual action ranking;
- adding generated reward NLL through the shared world trunk (C-LR) restores
  ranking and deep reward discrimination, but creates unsafe false reward.

Stages 2D-2F show that frozen-head adaptation, two global calibration scalars,
and a source-aligned categorical operator do not resolve this trade-off.

Does sparse reward **relevance** supervision on the exact generated state
consumed by the planner reward head improve the shared representation without
placing event-biased NLL pressure on the planner reward logits?

This stage repairs the previously vacuous Stage-1 H3 control. H3's event/sign
heads saw a frozen trunk and therefore shared no trainable parameter with the
deployed reward path. Stage-2G lets event/sign gradients cross the generated
planner-state input into the action embedding, future predictor, and temporal
core. The planner reward head remains unchanged.

## Primary-source selection

The following pinned papers and repositories were checked before selecting
the control.

| Source | Pinned identity | What it actually licenses here |
|---|---|---|
| DeepMDP | `1906.02736v1`, SHA-256 `8d378562...e2e33` | Its two losses are reward prediction and latent-transition prediction. The local world already has both through a shared trainable trunk; DeepMDP does not supply a missing third loss |
| DBC | `2006.10742v2`, SHA-256 `8322d722...b952`; official code `facebookresearch/deep_bisim4control@5967b6d0ccfc1032837cbe542f7bc5a96dc02cbb` | Reward differences can shape a control-relevant representation. Its actual Eq. 4/source loss jointly trains an observation encoder against a learned probabilistic transition metric; that is not the local frozen-encoder/post-transition setting |
| TACO | `2306.13229v3`, SHA-256 `93e21eba...33c`; official code `FrankZheng2022/TACO@84c38e34f4f9dfd2b059fb6d1356757e8d40712e` | The source uses a batch-matched BxB InfoNCE between state-plus-action-sequence and future-state representations, jointly learning state/action encoders. Its reported implementation uses very large contrastive batches; a four-row local true-vs-wrong-action loss would be TACO-inspired and gate-adjacent, not faithful |
| BYOL-AC | `2406.02035v1`, SHA-256 `38c8e0a7...ab27` | Eq. 5 uses a distinct predictor per action and a stop-gradient future target. The local predictor already receives action directly, and literal per-action local predictors would be a large architecture intervention. Its theorem assumes, among other things, a uniform policy and symmetric per-action dynamics |

Consequently:

- do **not** call the selected control DeepMDP, DBC, TACO, or BYOL-AC;
- do **not** transplant DBC's metric recursion into a frozen visual encoder;
- do **not** use a small-batch same-anchor contrast while claiming faithful
  TACO;
- do **not** introduce per-action predictor capacity while the immediate
  diagnosed failure is reward safety rather than absence of action response.

The selected intervention is a deliberately local mechanistic control:
**shared-path sparse-reward relevance auxiliary**. DeepMDP/DBC motivate the
question that reward-relevant geometry matters, but the event/sign
factorization and sampling scheme are local.

## Factorial arms

The two existing Stage-2C checkpoints are immutable references:

| Arm | Generated latent | Generated planner reward NLL | Relevance auxiliary |
|---|---:|---:|---:|
| C-L | `1.0` | `0` | `0` |
| C-LR | `1.0` | `.10` | `0` |
| **G-LA** | `1.0` | `0` | `lambda_aux` |
| **G-LRA** | `1.0` | `.10` | `lambda_aux` |

Reference hashes:

- C-L:
  `227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1`;
- C-LR:
  `60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae`.

This is a two-factor comparison:

- C-L versus G-LA asks whether relevance supervision can replace generated
  planner-reward NLL as the representation-shaping channel;
- C-LR versus G-LRA asks whether it helps when added to the current
  discriminator;
- G-LA versus G-LRA retains the generated-reward factor under the same
  auxiliary.

No baseline is mutated and no result may be attributed to “more supervision”
without reporting both factorial contrasts.

## Immutable base-world contract

- backend/topology: full-grid GRU, no bypass;
- seed: 505;
- exact frozen Step-1 encoder and disabled target EMA;
- local `local_symlog` planner reward operator with historical initialization;
- replay:
  `data/replay_40k_v1.pt`,
  SHA-256
  `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad`;
- updates: 16,000 from the exact Stage-2C initialization;
- base batch: 4, window: 16 observations;
- optimizer: AdamW, learning rate `1e-4`;
- clipping: world global norm 100;
- exact uniform base schedule:
  `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0`;
- base loss: unchanged `frozen_dynamics_recipe()`;
- generated K1/K2 latent target weight `1.0`;
- generated planner reward NLL factor as shown in the arm table;
- generated continuation `0`;
- no terminal pool, calibration, depth head, reliability weighting, or
  categorical change;
- bfloat16 autocast and update ordering unchanged.

The base batch remains uniform for every loss above. Event-balanced windows
are consumed by the auxiliary only.

## Relevance auxiliary

### Input and gradient path

For a separate 10-observation auxiliary window:

1. observe eight real observations;
2. imagine K1 and K2 using the recorded actions and deterministic predictor;
3. take `world.pool(state.tokens)` after each generated transition—the exact
   64-dimensional input passed to the planner reward head;
4. apply two independent linear training-only heads:
   - event logit: `reward != 0`;
   - sign logit, evaluated only on event rows: `reward > 0`.

The auxiliary loss is:

`L_aux = BCE(event_logit, event) + BCE(sign_logit[event], positive[event])`.

The auxiliary heads are discarded at evaluation. They are not a replacement
reward decoder and cannot directly change planner outputs. Their only
deployment-relevant path is through shared world parameters.

Required gradient invariants:

- nonzero gradients in `action_input`, `future`, and `temporal`;
- exactly zero auxiliary-only gradients in the planner reward head,
  continuation head, online encoder, and target encoder;
- detaching the planner-state input makes every world gradient exactly zero
  while leaving auxiliary-head gradients nonzero.

### Auxiliary sampling

Construct all valid 10-observation windows and inspect only generated
transition indices 7 and 8. Exclude any window with a terminal at either
transition.

Partition into:

- `zero`: both rewards are exactly zero;
- `positive`: at least one positive reward and no negative reward;
- `negative`: at least one negative reward and no positive reward.

Mixed-sign windows are excluded from the auxiliary schedule. Every auxiliary
batch contains two zero windows, one positive window, and one negative
window. Thus exactly half of windows are event-containing and sign is
balanced at the window level. The scalar planner reward and continuation
losses never consume this batch.

A deterministic auxiliary schedule and a disjoint fixed auxiliary probe are
constructed from a separate NumPy generator. Their seeds, counts, byte
digests, realized row-level event/sign frequencies, and overlap checks must be
recorded before training.

### Gradient-scale registration

`lambda_aux` is fixed mechanically before smoke training; it is not selected
from a performance metric.

On the first 16 pinned base and auxiliary batches at the common fresh
initialization:

- measure the RMS L2 gradient of raw generated reward NLL in
  `action_input + future + temporal`;
- measure the RMS L2 gradient of raw `L_aux` in the same modules;
- set
  `lambda_aux = .10 * rms_grad_generated_reward / rms_grad_auxiliary`.

This matches the initial shared-gradient scale of the already registered
C-LR generated reward term. Record full component gradients. Stop if either
gradient is non-finite/zero or if `lambda_aux` lies outside `[.01, 10]`.
Once computed and sealed in the preflight artifact, the coefficient cannot be
changed.

The same coefficient is used in G-LA and G-LRA. Auxiliary-head parameters use
a separate AdamW optimizer at `1e-4` and a separate norm-100 clip, so their
gradient norm cannot alter world clipping.

## Correctness tests before smoke

- exact event/sign transition indexing on synthetic trajectories;
- exact exclusion of terminal and mixed-sign windows;
- deterministic, balanced auxiliary schedule and disjoint probe;
- both heads see the generated K1/K2 planner-state inputs, not frozen target
  latents or real teacher-forced contexts;
- auxiliary-only gradient routing passes all invariants above;
- a detached/vacuous auxiliary update leaves the world update bit-identical
  to the no-aux reference;
- auxiliary initialization does not consume or change the base-world RNG;
- C-L/C-LR default training behavior and checkpoint compatibility remain
  unchanged;
- encoder digest, transition convention, stale-state revision, and
  boundary-mask tests continue to pass.

## Smallest discriminating preflight

Before either full run:

1. reproduce the existing 64-update C-LR history/state fingerprint exactly;
2. pin source, replay, base schedule, auxiliary schedule, probe, fresh-world,
   and auxiliary-head hashes;
3. compute and seal `lambda_aux` by the registered gradient formula;
4. run G-LA and G-LRA for 256 updates from the common initialization;
5. require finite losses, gradients, parameters, logits, and decoded planner
   rewards throughout;
6. require nonzero world-state divergence from the corresponding no-aux arm;
7. require fixed auxiliary-probe total BCE to fall from update 0 to 256 and
   event/sign AUROC point estimates to exceed `.55`;
8. require no encoder drift and peak reserved VRAM below 5,500 MiB;
9. require absolute decoded planner reward below 100 on the fixed probe.

This probe is a trainability/safety screen, not evidence that the planner
improved. Failure stops the full run without inspecting DEV.

## Split and evaluation

Training and preflight code must not import any evaluation artifact. After
the two full checkpoints and training reports are committed and hash-pinned,
evaluate only the already-spent Stage-2 DEV tier:

- natural:
  `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e`;
- terminal:
  `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf`;
- fork:
  `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8`.

FINAL must not be indexed or deserialized. Re-evaluate A, C-L, and C-LR with
the same evaluator and require their complete raw blocks to reproduce the
committed Stage-2C references exactly before interpreting either new arm.

Report K0/K1/K2/K4/K8 reward, continuation, and latent metrics; positive and
negative conditional reward; event and zero MAE/NLL; zero-return fork
suffixes; fork ranking/regret; training/probe curves; parameter and gradient
digests; changed fork choices; source/checkpoint/data hashes; wall time; and
peak VRAM.

## Decision gates

All paired intervals use episode/environment-cluster bootstrap with the same
registered directionality as Stages 2C-2F.

### Validity

Every source/split/hash/indexing/gradient/freeze/checkpoint invariant must
pass, and A/C-L/C-LR raw references must reproduce exactly. Otherwise repair
and rerun without interpreting DEV.

### Auxiliary mechanism

Evaluate both contrasts separately.

For G-LA versus C-L and G-LRA versus C-LR, a mechanism pass requires:

1. K8 event AUROC and Pearson point estimates improve and neither is
   significantly lower;
2. K8 event MAE is not significantly higher;
3. absolute zero-suffix predicted return is not significantly higher and its
   point delta is at most `+.02`;
4. chosen-minus-random improves as a point and ranking/regret are not
   significantly worse;
5. latent and continuation metrics show no significant harm at
   K1/K2/K4/K8.

Failure of one contrast does not erase the other, but no aggregate
“auxiliary helps” claim is allowed.

### Operational candidate

Assess G-LA and G-LRA separately. A candidate passes only if all hold:

1. absolute zero-suffix return delta versus A and its CI upper bound are
   `<= +.02`;
2. chosen-minus-random is positive, is not significantly below A or C-LR,
   and its point is at least the lower of A and C-LR;
3. K8 AUROC, AP, Pearson, and event MAE point estimates preserve C-LR;
4. K8 AUROC/Pearson/event-MAE show no significant harm versus C-LR;
5. K0/K1 reward discrimination and zero-MAE show no significant harm versus
   A, with zero-MAE point deltas `<= +.005`;
6. latent and continuation safety hold versus the arm's no-aux factorial
   reference at every registered depth.

Even an operational pass is DEV-only. It requires matched-seed replication
and a fresh evaluation tier before planner execution.

## Outcome-independent routing

- **Invalid implementation/preflight:** repair; do not train or evaluate.
- **Neither mechanism contrast passes:** reject this local event/sign route.
  Revisit a separately registered stronger-action-conditioning or adequately
  batched TACO-inspired control; do not tune auxiliary thresholds on DEV.
- **One mechanism contrast passes but no operational candidate passes:**
  record the shared relevance effect as causal but insufficient. No planner.
- **Operational candidate passes:** replicate on matched seeds and a fresh
  tier before any planner protocol.

Mamba transfer, predictor mixtures, reliability weighting, FINAL, planner
execution, actor/critic, and online policy training remain **NO-GO**
throughout Stage-2G.

## Preflight outcome — 2026-07-19

Status: **PASS; full matched G-LA/G-LRA training authorized**

Three invalid preflight attempts were blocked before full training:

1. auxiliary initialization restored CPU but not CUDA RNG state;
2. the fixed probe moved logits to CPU before applying a CUDA mask;
3. class-ordered probe chunks computed sign BCE on empty all-zero chunks,
   producing `NaN`.

The first two attempts stopped before any smoke update. The third completed
bounded smoke work, but its probe gate was mathematically invalid. The repair
aggregates all probe logits and labels before computing either BCE; it does
not change an arm, coefficient, threshold, dataset, or outcome route. CUDA
regressions now pin both failure classes.

The valid clean run records:

- preflight SHA-256:
  `5551ead595a0d1ae71d4e479918176439e1a1405cbcdb11b07d9159919f5b97d`;
- exact historical 64-update C-LR state/history fingerprints;
- base schedule SHA-256:
  `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0`;
- auxiliary schedule SHA-256:
  `d109da9a1c8950ec929dd5dcdf5873e871f78c40a26cf0b5a5413e22d1550f1b`;
- disjoint probe SHA-256:
  `9c4c2b80017e6b4e687fc3c44c91e954021a4a2ef828e1522a55f3eebe5d0fae`;
- raw generated-reward shared-gradient RMS `21.69279`;
- raw auxiliary shared-gradient RMS `11.00702`;
- mechanically fixed
  `lambda_aux = 0.19708130570134666`.

Every one of the 16 auxiliary gradient checks reaches action input, future
predictor, temporal core, and auxiliary heads while planner reward,
continuation, online encoder, and target encoder gradients remain exactly
zero. The detached control has exactly zero shared-world gradient.

Held-out auxiliary-probe changes after 256 updates:

| Arm | BCE | Event AUROC | Sign AUROC |
|---|---:|---:|---:|
| G-LA, u0 | `1.45734` | `.53434` | `.41016` |
| G-LA, u256 | `1.14777` | `.72754` | `.80469` |
| G-LRA, u0 | `1.45734` | `.53434` | `.41016` |
| G-LRA, u256 | `1.16020` | `.73893` | `.80078` |

Both auxiliary worlds differ from their exact 256-update no-aux references.
Both remain finite, preserve the frozen encoder, reserve 174 MiB peak VRAM,
and keep fixed-probe decoded reward below `.009` absolute.

This is only a trainability, routing, and short-horizon safety pass. It is not
reward, ranking, or planner evidence. It authorizes the two registered
16,000-update GRU arms and nothing larger.

## Full-training seal — 2026-07-19

Status: **both registered arms complete; DEV evaluation authorized only after
this seal is committed and hash-pinned**

| Field | G-LA | G-LRA |
|---|---:|---:|
| checkpoint SHA-256 | `c7c90965...36e5` | `40cdbf59...7823` |
| final world digest | `f0ebb034...56d1a` | `a0e2cb78...94cb8` |
| final auxiliary digest | `9c5d3287...25d8` | `7e98d18d...e002f` |
| training time | `2225.89 s` | `2199.54 s` |
| peak allocated VRAM | `123.19 MiB` | `123.90 MiB` |
| peak reserved VRAM | `174 MiB` | `174 MiB` |
| final-probe event AUROC | `.84391` | `.88151` |
| final-probe sign AUROC | `.95703` | `.97656` |

Training report SHA-256:
`4cc81e774c9d7ab21fa667b03ce12d47ec9ef20a4a82c35f0a90184c5f2e8e60`.

Raw history SHA-256:
`87637ab2ed4df4d77f06f661d6449c2bf87b3aef5b868dff68a62bf8c7290876`.

Every component history contains exactly 16,000 finite values. Both
checkpoints strict-load with the registered GRU config, reproduce their full
world digests, and restore their training-only auxiliary state to the exact
recorded digest. Their frozen encoder digests are identical.

The probe improvement confirms that the auxiliary remains trainable at full
budget. It is not evidence of reward safety or control usefulness. No DEV or
FINAL artifact was imported by the training modules.

## DEV outcome — 2026-07-19

Status: **VALID NEGATIVE; reject the local event/sign auxiliary**

The evaluator and gates were sealed at
`0a0e7904aa5aa436f46e1e0e8e866048f94945d3` before DEV access. A placeholder
tautology in an uncommitted validity draft was found and replaced before that
seal; the committed gate pins the exact initialization, schedules,
coefficient, checkpoints, encoder, and full state digests.

A, C-L, and C-LR reproduce every committed Stage-2C raw block exactly. Every
world remains bit-identical during evaluation. The analysis is deterministic
and reproduced byte-for-byte.

| Arm | K8 AUROC | K8 Pearson | K8 latent error | K8 terminal AUROC | Zero-suffix abs return | Chosen-random |
|---|---:|---:|---:|---:|---:|---:|
| A | `.6711` | `.1615` | `.0560` | `.8701` | `.00947` | `.2770` |
| C-L | `.6775` | `.0552` | `.0519` | `.9186` | `.00638` | `.1056` |
| C-LR | `.7359` | `.1892` | `.0534` | `.9139` | `.06404` | `.2754` |
| G-LA | `.7252` | `.0982` | `.0637` | `.8214` | `.04656` | `.0627` |
| G-LRA | `.7197` | `.1007` | `.0638` | `.8220` | `.04743` | `.1611` |

G-LA improves K8 event AUROC versus C-L by
`+.04776 [+.01083,+.08356]`, but increases false reward by
`+.04017 [+.01611,+.06689]`, does not improve ranking, and significantly
harms latent error and terminal AUROC at every depth.

G-LRA does not improve K8 AUROC/Pearson or ranking versus C-LR and also
significantly harms latent error and terminal AUROC at every depth. Its
false-reward point improves versus C-LR, but the complete mechanism still
fails.

Both candidates exceed the A-relative false-return budget:

- G-LA: `+.03709 [+.01981,+.05456]`;
- G-LRA: `+.03796 [+.01557,+.05885]`.

Registered decisions:

- G-LA mechanism: **FAIL**;
- G-LRA mechanism: **FAIL**;
- G-LA operational: **FAIL**;
- G-LRA operational: **FAIL**;
- route: **REJECT_LOCAL_EVENT_SIGN_AUXILIARY**;
- planner/Mamba/FINAL/online policy: **NO-GO**.

Artifacts:

- evaluation report:
  `8a294c59836d3515ffc6a5d680fa3de7fcc605080966e9f4a5f2a61bb6790f37`;
- evaluation raw:
  `ebfc2cbe0e04ee3e579b80d2eda7686e5a4a10eea7f555889d6de866c480f574`;
- analysis:
  `5036e11d2a0b5c30d6e417a10826df085826bf0be757be630110226cf0edac57`.

The auxiliary probe learned event/sign labels strongly, but those labels did
not imply useful action identity. The full independent audit and source
boundary are in
`reviews/2026-07-19-stage2g-outcome-and-independent-review.md`.
