# Stage-2D outcome and independent review

Date: 2026-07-18

Reviewer: Codex companion

Status: **valid clean negative; stop reward-head adaptation on C-L**

## Verdict

Stage-2D is valid and both reward-head candidates fail.

The isolation invariant passes exactly: in both arms only the six
`reward.*` tensors changed, the non-reward state digest stayed bit-identical,
and every latent and continuation prediction at every evaluated depth is
elementwise identical to C-L.

The result separates two claims:

1. **Generated-state covariate shift remains real.** Relative to equal-update
   real-state fitting, D-G improves K8 reward Pearson by
   `+.04879 [+.01245, +.12829]`, event magnitude, event MAE, sign accuracy,
   and sign AUROC.
2. **A reward head on the fixed C-L representation cannot recover control.**
   D-R leaves aggregate fork ranking exactly at C-L; D-G makes it worse.
   Neither arm approaches A's ranking, and neither passes the registered
   reward-head candidate gate.

This refines the Stage-2C diagnosis. C-LR's ranking recovery cannot be
replicated by refitting only the reward decoder on the better-latent C-L
trunk. Reward gradients through the shared dynamics changed reward-relevant,
action-conditioned geometry—not merely output scale. Unfortunately that
shared change also created C-LR's false reward.

The next problem is therefore a constrained representation/calibration
problem: preserve C-LR's action ranking while repairing its reward baseline.
Do not run more reward-head adaptation on C-L and do not reopen broad
full-world or Mamba training.

## Provenance and implementation audit

Pre-outcome commits:

- protocol: `b546e64`;
- implementation and tests: `0bb9ed5`;
- execution HEAD:
  `0bb9ed5ddfe81e5c3bbdfe97c62d581c8edf11ac`.

Pinned inputs:

| Input | SHA-256 |
|---|---|
| C-L base checkpoint | `227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1` |
| Stage-2C raw rows | `e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5` |
| Stage-2C report | `b73360a52bb137ef939a45c55f247fd0091011273fd6b1c1b8594201101706fc` |
| replay | `c55257feb2f903d32806b2694dd35e049fcd48397d3525b505c9dd715c455dad` |
| schedule | `d8ed746758296f365282823eba8595751b407d616c96b93e8f8417904126fc4c` |

Outcome artifacts:

| Artifact | SHA-256 |
|---|---|
| `stage2d_dr_s505.pt` | `1a5c991a71feefdcaf463fe149b5302edca01471ada98eb9ab7444944be8f5d4` |
| `stage2d_dg_s505.pt` | `cae7234b6d8094b4e17b881fd5646d7c5f2e1d461d8f7fb500775ce1140cba93` |
| `stage2d_report.json` | `a3375a0b8ecec5c608c28f22de48e6b23c59a65d163a2456052776904a7a9dab` |
| `stage2d_raw.json` | `90cb29bdb1833d386b2cedaafbb2b34e358b85fc75e66330eb74cf6e060277fb` |
| `stage2d_analysis.json` | `fa91948947fc0506845409f4577a0c709320cc7612da2a5bc1aa180de40e9825` |

Both arms:

- branch from reward-head digest
  `091e08894efc407b7b8d1cd2b4af375adadf340c4f603c53ff3e23a9fa8ac7f3`;
- retain non-reward digest
  `c44815c4236b748fb4f95d0f82a14671aa7ddbc462d6cd3c62cf388e5686c6c5`;
- consume the exact same 24,000 natural windows and nine labels per window;
- use 3,000 updates, batch 8, AdamW `1e-3`, and no event/terminal pool;
- train exactly 41,343 reward-head parameters;
- never access the FINAL tier.

Training cost:

| Arm | Wall time | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| D-R | 70.23 s | 37.15 MiB | 54.00 MiB |
| D-G | 68.71 s | 37.15 MiB | 54.00 MiB |

Before fitting, the complete CUDA suite passed **130/130** with one known
PyTorch warning. The focused clean-commit contract passed **9/9**. These tests
cover transition alignment, matched labels/schedules, real-prefix identity,
reward-only gradients and optimizer state, exact checkpoint digests,
non-reward bit identity, FINAL non-access, paired gate direction, and false
reward rejection.

No indexing, recurrent-cache, target-encoder, frozen-parameter, schedule, or
checkpoint defect was found.

## Results

All intervals are paired 95% cluster-bootstrap intervals.

### Direct reward readouts at K8

| Metric | A | C-L | D-R | D-G |
|---|---:|---:|---:|---:|
| event AUROC | .67114 | .67748 | .67493 | .71002 |
| average precision | .11889 | .11355 | .11246 | .12640 |
| signed Pearson | .16146 | .05523 | .00836 | .05715 |
| event magnitude | .00570 | .00294 | .00309 | .01016 |
| event MAE | .45912 | .46245 | .46277 | .45812 |
| zero MAE | .00082 | .00071 | .00124 | .00349 |

D-G versus D-R:

- event AUROC: `+.03509 [-.01969, +.08741]`;
- signed Pearson: `+.04879 [+.01245, +.12829]`;
- zero MAE: `+.00225 [+.00185, +.00265]` (worse);
- event MAE: `-.00465 [-.00856, -.00175]` (better);
- event magnitude: `+.00707 [+.00417, +.01067]`;
- sign AUROC: `+.20543 [+.07917, +.36635]`.

This is credible evidence that generated-state exposure affects the intended
deep readout. It is not enough for action selection.

### Fork ranking

| Arm | within-anchor Pearson | chosen-minus-random | regret |
|---|---:|---:|---:|
| A | .4011 | .27698 | .12857 |
| C-L | .2034 | .10556 | .30000 |
| D-R | .2046 | .10556 | .30000 |
| D-G | .0138 | .06270 | .34286 |
| C-LR, for context | .4884 | .27540 | .13016 |

D-R changes two of 21 discrete fork choices relative to C-L, but those swaps
have equal aggregate realized value: its chosen-minus-random and regret are
exactly unchanged. D-G worsens chosen-minus-random by
`-.04286 [-.10000, .00000]` versus D-R and is significantly worse than A:
`-.21429 [-.45686, -.03968]`.

This is another example of why across-episode event AUROC cannot stand in for
within-anchor action ranking.

### False reward

Absolute predicted return on truly zero-return suffixes:

| Arm | Absolute sum |
|---|---:|
| A | .00947 |
| C-L | .00638 |
| D-R | .00922 |
| D-G | .02181 |

D-G remains narrowly within the absolute A + `.02` candidate budget:
delta `+.01234 [+.00703, +.01924]`. However it is significantly worse than
D-R by `+.01260 [+.00949, +.01681]`, so the registered generated-state
mechanism gate fails. It also significantly worsens K0 and K1 zero-reward MAE
versus A.

### Gates

- I, exact isolation: **PASS**.
- M, generated-state mechanism including safety: **FAIL** on incremental
  false reward.
- C, D-R candidate: **FAIL**.
- C, D-G candidate: **FAIL**.
- Registered route: **STOP_HEAD_ADAPTATION; diagnose reward parameterization
  and calibration before more world training**.

## What this does and does not prove

Supported:

- Stage-2D's implementation is not responsible through an accidental update;
  it enforces the intended intervention exactly.
- Generated-state reward supervision improves some deep task statistics even
  with a frozen representation.
- The fixed C-L representation plus the current shared two-hot decoder does
  not recover fork ranking under either matched head schedule.
- C-LR's shared-trunk reward gradient is functionally important to its
  action-conditioned ranking.

Not supported:

- “The two-hot reward distribution is definitely the sole problem.” A fixed
  head may fail because the C-L state lacks linearly/MLP-decodable
  reward-relevant branch geometry, because one shared decoder aliases depth,
  because sparse likelihood optimization suppresses events, or because of
  their interaction.
- “Generated-state adaptation never works.” The paired K8 Pearson and event
  effects are positive; this one C-L candidate fails deployment.
- “C-LR is ready after a threshold.” No calibration rule has yet been fitted
  on independent data, and thresholding can change cumulative branch order.
- Any conclusion about Mamba. Stage-2D uses only GRU-505.

## Next smallest investigation

Do not retry C-L with more updates, event oversampling, a larger shared head,
or the already-rejected C-matrix depth-indexed control. Instead use C-LR—the
only arm that has the required branch ordering—as a frozen starting point and
ask whether its reward output can be calibrated without changing that order.

The next diagnostic should be calibration-only and split-safe:

1. freeze all C-LR parameters;
2. collect per-step decoded reward/logits and actual rewards on a calibration
   split not used for evaluation;
3. compare identity, global affine/temperature calibration, and a
   zero-aware event-probability calibration or dead-zone mapping;
4. fit every scalar on the calibration split only;
5. evaluate unchanged on the spent DEV as a mechanism diagnostic;
6. require the zero-suffix CI budget, K0/K1 calibration, K8 AP/Pearson, and
   fork ranking simultaneously.

A simple positive affine scaling cannot change equal-length suffix ordering;
it can only test amplitude. A dead-zone or event-probability gate can change
ordering and therefore must be evaluated, not assumed safe.

If no low-capacity calibration preserves C-LR ranking while repairing false
reward, the remaining supported route is a separately registered
reward-relevant representation objective/control—not another decoder sweep.
That route must keep the C-L latent arm and A as independently runnable
controls.

## Decisions

- D-R: **REJECT**.
- D-G: **REJECT**.
- More reward-head fitting on C-L: **NO-GO**.
- C-LR low-capacity frozen-output calibration diagnostic: **GO**.
- New full-world objective, Mamba transfer, additional seeds, FINAL,
  planner execution, actor/critic, and online policy: **NO-GO**.
