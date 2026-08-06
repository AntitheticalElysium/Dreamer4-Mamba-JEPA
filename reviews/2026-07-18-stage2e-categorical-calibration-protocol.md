# Stage-2E protocol: frozen C-LR categorical calibration

Status: **pre-registered before implementation, calibration fitting, or
evaluation**

Date: 2026-07-18

## Question

C-LR is the only Stage-2 arm that preserves/improves latent accuracy,
continuation, and fork ranking together, but it hallucinates cumulative reward
on truly zero-return suffixes. Stage-2D shows that fitting the reward head on
the C-L trunk cannot reproduce C-LR's action ranking.

Can two global categorical calibration scalars remove C-LR's false-reward
bias while preserving its branch ordering and reward discrimination?

This is a calibration-only diagnostic. It does not test a new world model,
Mamba, DreamerV3's alternative two-hot operator, true Dreamer-4 MTP, or
online control.

## Source boundary

`reviews/2026-07-18-reward-head-source-correction.md` records the exact
operator distinction:

- local/DRAMA/unofficial-D4-JAX:
  `symexp(sum softmax(logits) * symlog_bins)`;
- DreamerV3/CDP:
  `sum softmax(logits) * symexp(symlog_bins)`, with different target
  interpolation.

Stage-2E retains the local operator. Temperature and zero-bin bias are local
post-hoc calibration controls with no claimed source authority.

## Immutable model

- Base: C-LR checkpoint `reviews/artifacts/stage2c_clr_s505.pt`.
- SHA-256:
  `60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae`.
- Full-state digest:
  `93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49`.
- Every model parameter and buffer remains frozen and bit-identical.
- The calibrator operates only on already-produced reward logits.
- Continuation logits, latent predictions, recurrent state, and action
  ranking inputs other than calibrated reward are unchanged.

## Strict calibration/evaluation split

### CAL, used to fit and select two scalars

- `data/heldout_20ep_v1.pt`;
- SHA-256:
  `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad`;
- 20 episodes, 3,262 eligible targets, 140 reward events;
- same-target K0/K1/K2/K4/K8 logits, yielding 16,310 categorical examples
  with the target repeated once per state depth.

The CAL set is historically spent for earlier representation diagnostics. It
is used here only to fit calibration parameters. It is disjoint from the
Stage-2 DEV artifacts and cannot evaluate the calibrator.

### DEV, used once after the CAL artifact is committed

- natural seeds 960-975:
  `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e`;
- terminal seeds 932-947:
  `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf`;
- fork seeds 143-150:
  `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8`.

DEV is already spent and licenses only a mechanism decision. The evaluator
must not run until the fitted CAL parameter artifact has its own commit.

The FINAL tier must not be indexed, deserialized, or evaluated.

## Calibration arms

Let \(l_i\) be the frozen C-LR logits, \(i_0=127\) the exact zero bin,
\(T=\exp(t)>0\), and \(b\) a zero-bin offset:

\[
l'_i = l_i/T + b\,\mathbf 1[i=i_0].
\]

Fit four fixed arms:

- **E-I:** identity, \(t=0,b=0\);
- **E-T:** temperature only, fit \(t\), fix \(b=0\);
- **E-Z:** zero-bin bias only, fix \(t=0\), fit \(b\);
- **E-TZ:** fit both \(t,b\).

Fitting minimizes ordinary unweighted local two-hot cross-entropy over every
CAL example. Use deterministic full-batch LBFGS in float64 from the identity
initialization. No event weighting, depth weighting, reward threshold, DEV
metric, or manual parameter choice is allowed.

Select the arm with the lowest finite CAL NLL; ties within `1e-10` select the
lower-capacity arm in order E-I, E-T, E-Z, E-TZ. Selection and parameters must
be saved and committed before DEV evaluation.

## Required executable checks

1. Zero reward maps exactly to bin 127 and every two-hot target sums to one.
2. E-I logits and decoded rewards are exactly the uncalibrated values.
3. Temperature is strictly positive; disabled parameters remain exact.
4. Synthetic logits show zero bias changes only bin 127.
5. LBFGS reduces or preserves CAL NLL and is deterministic across repeats.
6. The fit script contains no DEV manifest/path and cannot access FINAL.
7. The evaluator loads only the committed calibrator artifact and DEV tier.
8. The world state digest is unchanged before/after CAL collection and DEV
   evaluation.
9. E-I reproduces committed C-LR natural predictions and fork rows exactly.
10. Paired gates reject a false-reward improvement that harms ranking or
    shallow reward.
11. Focused/full CUDA suites, compileall, lint, and diff checks pass.

## DEV evaluation

Evaluate E-I, E-T, E-Z, and E-TZ for transparency, but apply the registered
candidate gate only to the CAL-selected arm. Reusing DEV to choose a different
arm is forbidden.

Report:

- reward metrics at K0/K1/K2/K4/K8;
- raw and continuation-gated fork returns, within-anchor correlations,
  chosen-minus-random, and regret;
- cumulative absolute reward on truly zero-return suffixes;
- paired episode/environment-cluster contrasts against C-LR and A;
- CAL parameters/NLL, checkpoint/data/script/commit hashes, state digests,
  wall time, and peak VRAM.

Continuation and latent rows are reused only after exact identity assertions.

## Candidate gate

The CAL-selected arm must satisfy all conditions:

1. CAL NLL is strictly lower than E-I by at least `1e-6`.
2. Absolute zero-suffix predicted-return delta versus A and its paired 95% CI
   upper bound are both `<= +.02`.
3. Fork chosen-minus-random and regret are not significantly worse than
   either C-LR or A.
4. K8 event AUROC, average precision, and signed Pearson point estimates are
   not below C-LR; K8 event MAE does not increase as a point.
5. K8 AUROC/Pearson/event-MAE paired contrasts do not show significant harm
   versus C-LR.
6. K0 and K1 AUROC/Pearson do not significantly worsen versus A.
7. K0 and K1 zero-reward MAE deltas are each `<= +.005` versus A and do not
   significantly worsen.
8. Model digest, continuation predictions, and latent predictions remain
   exact.

## Outcome-independent routing

- **Fit or isolation invalid:** repair and rerun the same protocol.
- **Selected arm fails:** reject global temperature/zero-bin calibration.
  Do not sweep thresholds on DEV. The next architectural control, if
  pursued, is a separately registered matched retrain of the local versus
  DreamerV3/CDP two-hot operators or a reward-relevant representation
  objective.
- **Selected arm passes:** confirm the frozen calibrator on a genuinely fresh
  evaluation tier and matched additional world seeds. Only a replicated pass
  may open a separate planner-execution protocol.

Mamba transfer, FINAL, planner execution, actor/critic, reliability weighting,
and online policy training remain NO-GO throughout Stage-2E.

## Outcome appendix — 2026-07-19

Status: **VALID NEGATIVE; selected E-TZ rejected**

The CAL artifact was committed before DEV. Unweighted CAL NLL selected E-TZ
with `T=1.4998668`, zero-bin bias `1.2552725`, and NLL `.187375935`
versus identity `.195777025`.

The first DEV invocation stopped at the exact E-I control. The new evaluator
had decoded CPU-copied logits on CPU, whereas the canonical Stage-2C path
decoded float32 logits on CUDA; differences were at most `3.58e-7`. Commit
`e2b40ca` restored canonical CUDA decode and added a regression. The original
Stage-2C evaluator and checkpoint reproduced the committed rows exactly, and
the repaired E-I path then passed natural-prediction and fork-row identity.
No non-identity DEV result was observed before this outcome-independent
repair.

E-TZ reduced absolute predicted return on zero-return suffixes from C-LR
`.06404` to `.04177`, but its A-relative delta remained
`+.03230 [.02432,.04041]`, above the registered `+.02` ceiling. K8 Pearson
improved `+.01634 [.00456,.02642]`, while event MAE significantly worsened
`+.01334 [.00554,.02294]` and event magnitude fell. Four of eleven gate
conditions failed.

Decision: reject global temperature/zero-bin calibration and do not select
another arm or sweep a threshold on spent DEV. FINAL, planner, Mamba transfer,
and online training remain NO-GO. Full audit:
`reviews/2026-07-19-stage2e-outcome-and-independent-review.md`.
