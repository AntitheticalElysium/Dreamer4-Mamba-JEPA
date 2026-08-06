# Stage-2F outcome and independent review

Date: 2026-07-19

Status: **VALID SPLIT VERDICT — operator mechanism passes; operational
candidate fails**

## Executive verdict

The reward distribution matters causally, but changing to the pinned
DreamerV3/CDP distribution does not produce a deployable world model.

Against the matched zero-initialized local arm, the DreamerV3/CDP
reward-distribution arm:

- significantly reduces absolute false return on zero-return suffixes by
  `-.02903 [-.03760, -.02146]`;
- improves deep latent cosine error, significantly at K2/K4/K8;
- raises K8 reward AUROC by `.01086` and signed Pearson by `.04700`, although
  both intervals include zero;
- worsens fork chosen-minus-random by `.06032` as a point, with a wide
  interval `[-.22124, +.10588]`;
- worsens event MAE by `.00979` as a point, also unresolved by its interval.

That satisfies the pre-registered, deliberately permissive **operator
mechanism** gate: false reward improves significantly and none of the
registered harm intervals resolves against the operator. It does **not** prove
general no-harm; ranking and event-MAE points move in the wrong direction and
DEV has low power.

The operational gate fails. Versus the current C-LR reference and A, F-DZ
misses the zero-suffix CI ceiling, K8 point preservation, K0 Pearson, K1
zero-MAE, and K4 continuation requirements. Its ranking point is also worse.

The correct decision is:

- retain the current C-LR checkpoint only as a research reference, not a
  deployable planner model;
- do not promote F-DZ;
- stop categorical-operator/calibration search on spent DEV;
- route next to a separately registered reward-relevant
  representation/action-conditioning control.

FINAL, planner execution, Mamba transfer, reliability weighting, actor/critic,
and online policy training remain **NO-GO**.

## 1. Primary-source verification

The implementation was checked against clean pinned repositories:

| Source | Commit | Exact fact used |
|---|---|---|
| Dreamer-CDP / DreamerV3 code | `a851fa3e3d70b624b094ee1810ad4bb602346092` | symexp-spaced original-reward support, original-space target interpolation, symmetric `E[reward]`, reward output `outscale: 0.0` |
| DRAMA | `a50bd54c34e77d1d13e988a031733a47817098e2` | local uniform-symlog target and `symexp(E[symlog])` decode |
| unofficial Dreamer-4 JAX | `8144b940d801971f12ec5633553b95001e555949` | corroborates the local symlog-space target/decode family; not official Dreamer-4 source |

The DreamerV3/CDP equations are:

1. build a half-grid `linspace(-20, 0, 128)`;
2. apply symexp and mirror it to 255 strictly increasing original-reward bins;
3. interpolate the scalar target between neighboring bins in original reward
   space;
4. train two-hot cross entropy in float32;
5. decode the probability-weighted original-reward support with paired
   symmetric summation.

The local model still uses its own LayerNorm/MLP, pooled world state, frozen
JEPA encoder, action conditioning, data, and optimizer. “DreamerV3/CDP
reward-distribution-aligned” is the maximum source claim. This is not a
DreamerV3, CDP, or Dreamer-4 reproduction.

## 2. Why the initialization factorial was necessary

DreamerV3's reward support reaches approximately `±4.85165184e8`. Its source
also sets the categorical output kernel scale to zero and initializes bias to
zero.

Applying that decoder to the local randomly initialized head produced fresh
rewards between roughly `3.9e5` and `1.6e6`, while the local decoder produced
`.071`–`.243` on the same logits. A one-arm operator swap would therefore
have combined the new equations with an initialization their source
explicitly avoids.

The registered arms separated this:

| Arm | Operator | reward output init |
|---|---|---|
| F-R | local | historical PyTorch default; existing C-LR |
| F-LZ | local | zero |
| F-DZ | DreamerV3/CDP | zero |

F-LZ versus F-R measures initialization. F-DZ versus F-LZ measures the
operator. F-DZ versus F-R measures the combined source-aligned candidate
against the current reference.

This control proved material. Zero initialization alone significantly lowers
K8 event AUROC:

`F-LZ - F-R = -.03977 [-.07455, -.00400]`.

Without F-LZ, that loss would have been incorrectly attributed to the
DreamerV3 distribution.

## 3. Correctness and chain of custody

### Implementation checks

- the reward operator is an explicit serialized `ModelConfig` axis;
- legacy checkpoints normalize to `local_symlog`;
- operator mismatch is rejected during strict checkpoint load;
- the support is nonpersistent and does not alter parameter/state keys;
- F-LZ and F-DZ have identical initial state dicts and parameter names;
- zero initialization changes only `reward.net.3.weight` and
  `reward.net.3.bias`;
- original-space targets match an independent searchsorted/interpolation
  implementation;
- support ordering, symmetry, center zero, endpoints, target normalization,
  endpoint clipping, loss, and symmetric decode are regression-tested;
- uniform DreamerV3 logits decode exactly zero;
- the historical local loss/decode remains bit-identical;
- transition indexing remains
  `(obs_t, action_t) -> (obs_{t+1}, reward_t, continue_t)`.

The local first-64-update fingerprint reproduced exactly after the new axis:

| Field | SHA/digest |
|---|---|
| schedule | `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0` |
| initial state | `55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260` |
| state after 64 updates | `92048e7311cda13ff178ed921b129ad5c85f95c56fba9c3046bc8c8d00b17415` |
| full history bytes | `ce25a57adab01d26bcf516ba929bfc6f81617b9748cf03c786d810d7d23a1a3b` |

### Split order

1. protocol: `154594a`;
2. source-aligned implementation/tests: `b133779`;
3. deterministic training-only preflight artifact: `79a5eb5`;
4. preflight hash pin: `6ad8f2d`;
5. two full training runs, with no evaluation imports;
6. checkpoint/training artifact seal: `87a0afa`;
7. evaluator with exact checkpoint/report hashes: `54e4c0d`;
8. DEV evaluation and paired analysis.

FINAL was never indexed or deserialized.

### Artifact hashes

| Artifact | SHA-256 |
|---|---|
| preflight | `9ece53e398d21547e0dee25f4b3147db90eacd2360d833a23bbde17990e86a00` |
| F-LZ checkpoint | `e6b448b1cfa6415080ee0148618c84846916250107f5a6f6f6a44a1530511743` |
| F-DZ checkpoint | `171c3826f6f9c5791b3ef03476fc5e8014fd99437dd20aed7bd63706a73671cb` |
| training report | `a602155d14badfc370a94cc922cc584d7fe1093f789b2e58de3b7f64928d4f08` |
| training raw histories | `9fdb9f318bbb847fafd18852ea7a43bdc96ff28abbb1339b1474fe8631759c87` |
| evaluation report | `93d9abb86a41cd13d3b52c157a6f387cd31bd75d4042ccd12110c7659ea244b4` |
| evaluation raw rows | `ec11705526282638492722dd85b24a3c1d5ce68a8265055e2dc67a4349d18b93` |
| paired analysis | `b70b0325158c786fcc56c08360c54fee4b6fb716dcce58fa8f97d62a1db2193b` |

F-R reproduces the complete committed C-LR raw block exactly: natural reward,
continuation, latent, and fork rows. Every evaluated state digest is unchanged
before/after DEV.

## 4. Preflight and training

F-LZ and F-DZ share zero-initialized state digest
`0dcd7b3fd433b0a20f5708723be35404d336a72678980041d9acca6786f80b27`.

At the 64-update preflight:

- F-LZ decoded generated reward grows from historical near-zero numerical
  residual (`<6.8e-8`) to an absolute maximum `.00480`;
- F-DZ grows from exact zero to `.00700`;
- every loss, gradient, parameter, and decode is finite;
- both reserve 170 MiB peak VRAM.

Full training uses the same 16,000 updates, schedule, batch, replay, optimizer,
base loss, K1/K2 latent target, and `.10` generated-reward coefficient:

| Last-500 metric | F-LZ | F-DZ |
|---|---:|---:|
| total loss | `.157996` | `.158703` |
| JEPA | `.020614` | `.020716` |
| base reward NLL | `.071839` | `.072206` |
| generated latent | `.023050` | `.023119` |
| generated reward NLL | `.150866` | `.150095` |
| generated weighted | `.038137` | `.038128` |
| training time | `28.91 min` | `29.11 min` |
| peak reserved VRAM | `170 MiB` | `170 MiB` |

Optimization scale and compute are closely matched. Both checkpoints
strict-load with their serialized operator and reproduce their final state
digests exactly. Their frozen encoder digests are identical.

## 5. DEV point results

### K8 reward

| Metric | A | F-R | F-LZ | F-DZ |
|---|---:|---:|---:|---:|
| event AUROC | `.67114` | `.73594` | `.69616` | `.70703` |
| average precision | `.11889` | `.12368` | `.10357` | `.12214` |
| signed Pearson | `.16146` | `.18915` | `.17140` | `.21840` |
| event MAE | `.45912` | `.43483` | `.43943` | `.44922` |
| decoded event magnitude | `.00570` | `.03369` | `.02658` | `.01719` |
| zero-row MAE | `.00082` | `.01123` | `.01077` | `.00509` |

### Fork behavior

| Metric | A | F-R | F-LZ | F-DZ |
|---|---:|---:|---:|---:|
| chosen-minus-random | `.27698` | `.27540` | `.25317` | `.19286` |
| regret | `.12857` | `.13016` | `.15238` | `.21270` |
| absolute zero-suffix return | `.00947` | `.06404` | `.05644` | `.02741` |
| absolute gated zero-suffix return | `.00944` | `.06319` | `.05570` | `.02700` |

F-DZ changes seven informative fork choices versus F-LZ: two improve and five
worsen. Versus F-R, ten choices change: four improve and six worsen. The
environment-clustered ranking intervals remain wide, so these point losses
are concerning but not statistically resolved.

## 6. Matched operator effect: F-DZ versus F-LZ

The causal operator contrast is:

| Metric | Delta `F-DZ - F-LZ` | paired 95% CI |
|---|---:|---:|
| abs. zero-suffix return | `-.02903` | `[-.03760, -.02146]` |
| K8 event AUROC | `+.01086` | `[-.02170, +.04109]` |
| K8 Pearson | `+.04700` | `[-.09196, +.18375]` |
| K8 event MAE | `+.00979` | `[-.00414, +.02333]` |
| K8 event magnitude | `-.00939` | `[-.02349, +.00516]` |
| chosen-minus-random | `-.06032` | `[-.22124, +.10588]` |
| regret | `+.06032` | `[-.10588, +.22124]` |

Latent cosine error improves:

- K2 `-.000323 [-.000526, -.000081]`;
- K4 `-.000976 [-.001335, -.000591]`;
- K8 `-.002155 [-.002773, -.001467]`.

No continuation contrast versus F-LZ significantly worsens.

The registered mechanism gate therefore passes. The correct narrow statement
is:

> Under matched zero initialization and training, the
> DreamerV3/CDP distribution causally reduces false reward and changes the
> learned shared representation, with no statistically resolved harm in the
> registered low-power mechanism screen.

It is **not** licensed to say “DreamerV3 reward heads are better” or “ranking
is preserved.” The ranking and event-MAE point estimates explicitly caution
against those claims.

## 7. Operational failures

F-DZ fails five operational areas.

### False-return CI

Against A, absolute zero-suffix return changes
`+.01794 [.01179, .02309]`.

The point is within `+.02`, but the CI upper bound is not. This is a near miss,
not a pass.

### K8 point preservation

Versus F-R:

- AUROC `.70703 < .73594`;
- AP `.12214 < .12368`;
- Pearson `.21840 > .18915`;
- event MAE `.44922 > .43483`.

The required conjunction fails. Paired K8 harm intervals remain unresolved,
but point preservation was separately pre-registered.

### Shallow reward

At K0, Pearson is significantly below A:

`-.08612 [-.15780, -.01999]`.

At K1, zero-reward MAE is significantly higher than A:

`+.000922 [.000091, .001832]`.

### Continuation

K4 terminal AUROC is significantly below F-R:

`-.02494 [-.05205, -.00229]`.

K8 Brier skill improves, but the all-depth safety conjunction correctly
fails.

### Ranking point

F-DZ versus F-R chosen-minus-random is
`-.08254 [-.25687, +.11671]`.

This is not statistically resolved, but it is the wrong point direction and
does not support planner readiness.

## 8. Bigger-picture interpretation

This experiment rules out two tempting shortcuts:

1. **“The reward issue is just post-hoc calibration.”** Stage-2E refutes
   that.
2. **“Use the DreamerV3 categorical equations and the problem is solved.”**
   Stage-2F refutes that operationally.

It also gives positive architectural information:

- the reward loss is changing the shared dynamics/predictor representation,
  not merely the scalar head—deep latent errors differ significantly under
  matched initialization;
- original-space interpolation/expectation suppresses false reward much more
  effectively than local global calibration;
- output initialization is itself a meaningful training intervention;
- the unresolved trade-off is between sparse reward relevance, decoded
  magnitude/calibration, continuation, and action ranking—not an isolated
  softmax temperature defect.

The project has not been “stuck on metrics” in this sequence. Stages 2C–2F
have eliminated, with controlled evidence:

- latent-only generated supervision as a deployable solution;
- frozen-trunk reward-head adaptation;
- global logit calibration;
- categorical-operator replacement as a complete solution.

What remains is a representation/objective question.

## 9. Decisions and next boundary

| Decision | Ruling |
|---|---|
| Promote F-DZ | **NO-GO** |
| Keep F-R/C-LR as research reference | **GO**, diagnostic only |
| More categorical temperatures, biases, bins, decoder swaps, or DEV thresholds | **NO-GO** |
| Claim DreamerV3/CDP distribution causally reduces matched false reward | **GO**, narrow mechanism claim |
| Claim no ranking/event harm | **NO-GO**, intervals are underpowered and points worsen |
| Mamba transfer | **NO-GO** |
| Planner / FINAL / online policy | **NO-GO** |

The next smallest discriminating experiment should leave the planner scalar
reward head and natural replay objective unchanged while adding a
reward-event/action-relevance signal that actually shares trainable
representation parameters. This repairs the vacuity of the earlier H3 arm,
whose auxiliaries shared no trainable trunk with the planner head.

Before implementation, the next protocol should verify and choose between:

1. a DeepMDP/DBC-motivated reward-relevant shared representation auxiliary;
2. a TACO-inspired counterfactual action/reward contrast;
3. a BYOL-AC-motivated stronger action-conditioning adapter.

The smallest arm should branch from the F-R/C-LR recipe, use uniform replay,
keep the scalar reward NLL unweighted, apply any event/relevance mixture only
to the auxiliary, and retain A/F-R plus a vacuous-head control. This targets
representation geometry without directly teaching the planner head to
hallucinate sparse events.

That choice should be source-verified and preregistered separately. Stage-2F
does not authorize architecture search, Mamba transfer, or planner execution.
