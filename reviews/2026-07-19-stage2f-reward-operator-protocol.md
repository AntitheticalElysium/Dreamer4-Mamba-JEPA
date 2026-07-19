# Stage-2F protocol: matched reward-distribution operator control

Status: **pre-registered before operator implementation, smoke training, full
training, or evaluation**

Date: 2026-07-19

## Question

Stage-2E shows that two post-hoc global scalars cannot separate C-LR's useful
action-conditioned reward signal from its false-return and event-magnitude
failures.

Does the exact categorical reward distribution contribute to that trade-off?
Specifically, under otherwise matched C-LR training, compare:

1. the current local symlog-space target and
   `symexp(E[symlog])` decoder; and
2. the pinned DreamerV3/CDP original-reward-space target and `E[reward]`
   decoder.

This is a reward-**distribution** control, not a DreamerV3, Dreamer-CDP, or
Dreamer-4 reproduction.

## Primary-source ground truth

### Current local / DRAMA-shaped operator

Local `m3_hjwm_compact/model.py` and DRAMA commit
`a50bd54c34e77d1d13e988a031733a47817098e2`,
`sub_models/functions_losses.py:25-55`:

- make 255 uniformly spaced centers in symlog coordinates from `-20` to
  `20`;
- apply symlog to the scalar target and interpolate between neighboring
  symlog centers;
- optimize two-hot cross entropy;
- decode `symexp(sum softmax(logits) * symlog_centers)`.

The inspected unofficial Dreamer-4 JAX repository at commit
`8144b940d801971f12ec5633553b95001e555949` uses the same target/decode
family. It is not official Dreamer-4 source.

### DreamerV3/CDP operator

Pinned Dreamer-CDP commit
`a851fa3e3d70b624b094ee1810ad4bb602346092`:

- `dreamerv3/configs.yaml:100` selects `symexp_twohot`, 255 bins, and
  `outscale: 0.0` for the reward head;
- `embodied/jax/heads.py:132-144` constructs original-reward support by
  applying symexp to a half-grid from `-20` to `0` and mirroring it;
- `embodied/jax/outs.py:273-330` interpolates the scalar target in original
  reward space and decodes the probability-weighted original reward support;
- `outs.py:285-309` uses a symmetric summation so uniform symmetric
  probabilities decode exactly to zero;
- `embodied/jax/nets.py:230-251` confirms `outscale: 0.0` zeros the output
  kernel; its bias initializer is also zero.

The local head MLP, LayerNorm, trunk, training data, action conditioning,
latent objective, and optimizer remain local. The new arm is therefore
labelled **DreamerV3/CDP reward-distribution-aligned**, not
source-faithful DreamerV3 as a system.

## Why output initialization is a required control

A pre-registration calculation exposed a structural interaction. Applying the
DreamerV3 decoder to the local randomly initialized reward output gives:

- DreamerV3 support endpoints approximately `±4.85165184e8`;
- four fresh first-step decoded values between `3.93e5` and `1.56e6`;
- the same local logits decode between `.071` and `.243` under the local
  operator.

Thus a single “change only the equations” arm would combine a new operator
with an initialization that its source explicitly avoids. It would be neither
a fair operator attribution nor a source-aligned control.

## Arms

| Arm | Distribution | Final reward-output initialization | Role |
|---|---|---|---|
| **F-R** | local | PyTorch default | existing C-LR reference; no retraining |
| **F-LZ** | local | exact zero weight and bias | initialization control |
| **F-DZ** | DreamerV3/CDP | exact zero weight and bias | operator candidate |

F-LZ versus F-R isolates output initialization within the local operator.
F-DZ versus F-LZ isolates the reward distribution under identical
source-compatible zero initialization. F-DZ versus F-R answers whether the
combined source-aligned distribution/init is operationally preferable to the
current candidate.

The invalid DreamerV3-operator/default-initialization combination receives no
full run.

## Immutable training contract

- backend/topology: full-grid GRU, no bypass;
- seed: 505;
- frozen step-1 encoder and exact encoder checkpoint already used by
  Stage-2C;
- replay: `data/replay_40k_v1.pt`, existing pinned hash;
- updates: 16,000;
- batch: 4;
- window: 16 observations;
- optimizer: AdamW, learning rate `1e-4`;
- clipping: global norm 100;
- exact uniform schedule SHA-256:
  `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0`;
- base objective: unchanged `frozen_dynamics_recipe()`;
- generated objective: K1/K2 latent weight `1.0`, reward weight `.10`,
  continuation weight `0`;
- no event/terminal pool, event weighting, calibration, depth head, or
  auxiliary classifier;
- same bfloat16 autocast and update order as Stage-2C.

The `.10` reward coefficient is held fixed. Any loss/gradient scale difference
is part of the distribution operator; it will be reported rather than tuned
after seeing an outcome.

F-R is the committed C-LR checkpoint:
`60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae`.
Its full-state digest is
`93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49`.

## Pre-change local regression fingerprint

Before adding an operator axis, the first 64 C-LR updates under the exact
contract produced:

- initial state digest:
  `55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260`;
- final state digest:
  `92048e7311cda13ff178ed921b129ad5c85f95c56fba9c3046bc8c8d00b17415`;
- float64 history-byte digest over total/base/generated component rows:
  `ce25a57adab01d26bcf516ba929bfc6f81617b9748cf03c786d810d7d23a1a3b`.

After implementation, the default local path must reproduce all three exactly
before any operator smoke or full run.

## Implementation requirements

1. Add an explicit serialized reward-operator axis; never infer an operator
   from a filename.
2. Normalize legacy checkpoint configs to the local default without changing
   their state dict or behavior.
3. Preserve existing local target, loss, decode, parameter names, and random
   initialization bit-for-bit.
4. Construct the DreamerV3 support exactly as the pinned odd-bin source.
5. Interpolate DreamerV3 targets in original reward space, including exact
   bins and endpoint clipping.
6. Decode DreamerV3 predictions with the pinned symmetric expectation.
7. The support is a nonpersistent constant; it must not alter checkpoint
   state or optimizer membership.
8. A DreamerV3 checkpoint must strict-load through the ordinary checkpoint
   loader and restore its operator from serialized config.
9. Zero initialization changes only the last reward linear's weight and bias.
10. F-LZ and F-DZ must have identical initial state dicts and optimizer
    parameter names.

## Required unit and source-alignment tests

- support is strictly increasing, symmetric, has exact center zero, and
  matches an independent implementation of the pinned equations within
  backend float32 transcendental precision;
- original-space two-hot targets are nonnegative, sum to one, interpolate the
  known Crafter rewards correctly, and clip endpoints;
- loss and target match an independent searchsorted reference;
- uniform symmetric logits decode exactly zero;
- asymmetric uncertain logits make local and DreamerV3 decoders diverge;
- zero output initialization preserves the historical local near-zero decode
  exactly and the DreamerV3 symmetric decoder returns exact zero;
- local default output remains bit-identical to the pre-change path;
- legacy local checkpoint strict-load and prediction identity pass;
- new DreamerV3 checkpoint round-trip restores the operator;
- config drift between operators is rejected;
- transition indexing remains
  `(obs_t, action_t) -> (obs_{t+1}, reward_t, continue_t)`;
- focused/full CUDA suites, lint, compileall, and diff checks pass.

## Smallest-discriminating preflight

Before either full run:

1. reproduce the 64-update local fingerprint above;
2. prove F-LZ/F-DZ initial state equality;
3. run 64 training updates for F-LZ and F-DZ;
4. require finite losses, gradients, parameters, and decoded rewards at
   updates 0/1/16/64;
5. require no encoder drift and peak reserved VRAM below 5,500 MiB;
6. record base/generated component histories and decode ranges.

If either zero-initialized arm is non-finite or produces absolute decoded
training reward above `100` during this smoke, stop before full training. Do
not rescue it by clipping support or tuning coefficients.

## Evaluation split

Only the already-spent Stage-2 DEV tier may be evaluated:

- natural:
  `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e`;
- terminal:
  `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf`;
- fork:
  `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8`.

FINAL must not be indexed or deserialized. DEV is mechanism evidence only.
All three arms are evaluated with their own registered loss/decode pair. A
post-hoc cross-decoder matrix may be reported as a diagnostic but cannot
select an arm or support a deployment claim.

Report K0/K1/K2/K4/K8 reward, continuation, and latent readouts; fork ranking
and zero-return safety; paired episode/environment-cluster contrasts; training
losses; parameter/state digests; source/checkpoint/data hashes; wall time; and
peak VRAM.

## Decision hierarchy

### Validity

Both new arms must satisfy every implementation/preflight invariant, share
the registered zero-initialized state, use the exact schedule, keep the
encoder frozen, and strict-load with the correct operator. Otherwise repair
and rerun without interpreting DEV.

### Operator mechanism: F-DZ versus F-LZ

Call the DreamerV3/CDP distribution a useful causal improvement only if:

1. zero-suffix absolute predicted return is lower and its paired CI upper
   bound is below zero;
2. K8 event AUROC and Pearson are not significantly lower;
3. K8 event MAE is not significantly higher;
4. fork chosen-minus-random/regret are not significantly worse;
5. latent and continuation metrics show no significant harm at K1/K2/K4/K8.

### Operational candidate: F-DZ versus F-R and A

F-DZ reaches an operational pass only if all hold:

1. absolute zero-suffix return delta versus A and its CI upper bound are
   `<= +.02`;
2. ranking is not significantly worse than both F-R and A;
3. K8 AUROC, average precision, Pearson, and event MAE point estimates
   preserve F-R;
4. K8 AUROC/Pearson/event-MAE show no significant harm versus F-R;
5. K0/K1 AUROC/Pearson and zero-MAE show no significant harm versus A;
6. latent and continuation safety hold versus F-R at every registered depth.

Even a pass requires a fresh evaluation tier and additional matched seeds
before planner execution.

## Outcome-independent routing

- **Invalid smoke/implementation:** repair; do not train or evaluate.
- **F-LZ materially changes the result:** attribute that change to output
  initialization, not the operator.
- **F-DZ fails the operator mechanism:** stop reward-operator search. Route to
  a separately registered reward-relevant representation/action-conditioning
  control; do not add more categorical knobs.
- **Mechanism passes but operational gate fails:** record the operator as
  causal but insufficient. No planner; decide the next control without DEV
  threshold tuning.
- **Operational gate passes:** require fresh seeds/tier before any planner
  protocol.

Mamba transfer, reliability weighting, FINAL, planner execution, actor/critic,
and online policy training remain **NO-GO** throughout Stage-2F.

## Pre-training clarification — 2026-07-19

The first executable source-alignment tests corrected two over-exact protocol
phrases before smoke training:

1. PyTorch and independent NumPy float32 `expm1` differ by at most 3 reward
   units in multi-million-magnitude tail bins (maximum relative difference
   below `9.6e-7`). Formula equality, strict ordering, exact symmetry, exact
   center zero, and known endpoints are the source-alignment requirements;
   cross-library transcendental bit identity is not.
2. The historical local decoder uses a naïve left-to-right reduction and maps
   uniform logits to approximately `-1.15e-7`. DreamerV3/CDP explicitly uses
   symmetric summation and returns exact zero. The local result must remain
   bit-identical rather than being silently “fixed”; only the DreamerV3 arm
   requires exact zero.

Neither correction uses training or DEV results and neither changes an arm,
coefficient, gate, or routing decision.

## Pre-training gate clarification — 2026-07-19

Before full training, “not significantly worse” is made executable as follows:

- for higher-is-better reward, continuation, and chosen-minus-random metrics,
  the paired 95% CI upper bound must be `>= 0`;
- for lower-is-better event MAE, latent cosine error, zero-reward MAE, and
  regret, the paired 95% CI lower bound must be `<= 0`;
- the K0/K1 zero-MAE point delta versus A must additionally be `<= +.005`,
  matching the preceding Stage-2E safety contract;
- operator-mechanism zero-suffix improvement requires both a negative point
  delta and a paired CI upper bound `< 0`;
- operational zero-suffix safety retains the already registered point and CI
  upper-bound ceiling of `+.02` versus A.

This clarification resolves directionality only. It does not change a metric,
threshold, arm, dataset, or outcome route.
