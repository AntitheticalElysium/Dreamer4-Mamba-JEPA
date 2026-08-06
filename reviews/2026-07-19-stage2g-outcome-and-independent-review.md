# Stage-2G outcome and independent review

Date: 2026-07-19

## Verdict

Stage-2G is **valid and negative**.

- Validity: **PASS**.
- G-LA versus C-L auxiliary mechanism: **FAIL**.
- G-LRA versus C-LR auxiliary mechanism: **FAIL**.
- G-LA operational candidate: **FAIL**.
- G-LRA operational candidate: **FAIL**.
- Registered route:
  **REJECT_LOCAL_EVENT_SIGN_AUXILIARY**.
- Planner, FINAL, Mamba transfer, actor/critic, and online policy:
  **NO-GO**.

The negative result is not explained by a conventional implementation
failure. The exact initialization, base schedule, generated-state indexing,
planner-state input, gradient routes, frozen encoder, checkpoint state, and
historical evaluation references all reproduce.

The narrow causal conclusion is:

> At seed 505, with the registered event-balanced auxiliary data and
> mechanically fixed coefficient, adding actual-action event/sign
> classification to the shared generated world path makes event/sign labels
> easier to linearly decode, but does not improve counterfactual action
> selection and harms latent/continuation fidelity.

This does not reject reward-relevant representation learning, TACO, or
BYOL-AC. Stage-2G implements none of those methods faithfully.

## Sealed provenance

The DEV evaluator and gates were committed before DEV access:

- evaluator commit:
  `0a0e7904aa5aa436f46e1e0e8e866048f94945d3`;
- evaluator SHA-256:
  `d77f197d574586da700e92e9e6a87cd227cdcf73c1f3d68f7485a7dd9a5b0bd2`;
- analysis SHA-256:
  `fb77071dd2361e7e048da8867f07285eb64350b6d25a18c247ecc63975387555`.

Outcome artifacts:

| Artifact | SHA-256 |
|---|---|
| `stage2g_eval_report.json` | `8a294c59836d3515ffc6a5d680fa3de7fcc605080966e9f4a5f2a61bb6790f37` |
| `stage2g_eval_raw.json` | `ebfc2cbe0e04ee3e579b80d2eda7686e5a4a10eea7f555889d6de866c480f574` |
| `stage2g_analysis.json` | `5036e11d2a0b5c30d6e417a10826df085826bf0be757be630110226cf0edac57` |

Upstream chain:

- preflight:
  `5551ead595a0d1ae71d4e479918176439e1a1405cbcdb11b07d9159919f5b97d`;
- training report:
  `4cc81e774c9d7ab21fa667b03ce12d47ec9ef20a4a82c35f0a90184c5f2e8e60`;
- training raw:
  `87637ab2ed4df4d77f06f661d6449c2bf87b3aef5b868dff68a62bf8c7290876`;
- G-LA checkpoint:
  `c7c909654b6eda45149e080417da2c1fb0637120b9c725b3e0ff2482392336e5`;
- G-LRA checkpoint:
  `40cdbf59b23b9878e2ec1660e795babf3b0254d99dbcd889939135f84c0f7823`.

Only the spent DEV tier was used:

- natural:
  `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e`;
- terminal:
  `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf`;
- fork:
  `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8`.

FINAL data were not opened.

## Pre-DEV gate repair

The first draft of `stage2g_analysis.py` contained a placeholder expression
that made the shared-initialization validity check tautologically true. It
was found before DEV, removed, and never used to interpret an outcome.

The committed evaluator/gate instead fails closed on:

- exact fresh-world and auxiliary-head initial digests;
- replay, base schedule, auxiliary schedule, and probe hashes;
- exact mechanically registered `lambda_aux`;
- update count, batch size, and generated-reward weights;
- checkpoint metadata, full world digest, auxiliary digest, and encoder
  provenance;
- no state mutation during evaluation;
- exact reconstruction of all four raw blocks for A, C-L, and C-LR.

Adversarial tests alter initialization, schedule, reference rows, checkpoint
metadata, false reward, and individual factorial outcomes and require the
appropriate gate to fail.

## Implementation audit

### Transition and label indexing

The code follows:

`(obs_t, action_t) -> (obs_{t+1}, reward_t, continue_t)`.

After observing indices `0..7`, generated K1 consumes `action[7]` and targets
`reward[7]`; K2 consumes `action[8]` and targets `reward[8]`. Synthetic tests
pin both action and reward indices.

Terminal exclusion inspects the same two continuation indices. Mixed-sign and
terminal windows are excluded before scheduling. The sealed realized
auxiliary schedule contains:

- 50% event-containing windows;
- 25% positive-only windows;
- 25% negative-only windows;
- no terminal or mixed-sign windows;
- no overlap with the fixed probe.

### Exact deployment input

`generated_planner_states()` takes `world.pool(state.tokens)` after each
generated transition. `M3HJWM.imagine_step()` passes that same pooled tensor
to `world.reward`.

A permanent forward-hook regression now records the actual reward-head input
at K1/K2 and requires it to be bit-identical to the auxiliary input. The
auxiliary is therefore not trained on target-encoder latents, real
teacher-forced contexts, or the wrong side of a transition.

### Gradient and optimizer routing

The sealed 16-batch registration and current tests show:

- nonzero auxiliary gradients in action input, future predictor, temporal
  core, and the two auxiliary heads;
- exactly zero auxiliary-only gradients in planner reward, continuation,
  online encoder, and target encoder;
- detaching the auxiliary input gives exactly zero world gradient;
- the detached auxiliary leaves a full world update bit-identical to its
  no-aux reference;
- auxiliary-head initialization preserves CPU and CUDA RNG streams;
- world and auxiliary parameters use separate optimizers and separate
  clipping.

The coefficient was not tuned:

`lambda_aux = .1 * 21.6927876 / 11.0070245 = .1970813057`.

This matches the initial shared-gradient RMS of the registered `.10`
generated-reward term. It does not guarantee that relative gradients stay
matched later in training; that is a limitation of the registered design,
not an implementation discrepancy.

### Checkpoints and frozen state

All five checkpoints strict-load with the registered GRU configuration and
reconstruct their expected full state digests:

- A: `6467e319...e0e4`;
- C-L: `a0cf4ec1...c1c9`;
- C-LR: `93509072...e49`;
- G-LA: `f0ebb034...56d1a`;
- G-LRA: `a0e2cb78...94cb8`.

Every state digest is unchanged after DEV evaluation. Both new checkpoints
carry encoder SHA-256
`3cc79446d18aaeea3f8c022e20f8d2b63db1bf33f5e7f7f3bf9ef759d3f825cc`.

### Historical reconstruction

A, C-L, and C-LR reproduce the committed Stage-2C
`reward_predictions`, `continuation_predictions`, `latent_errors`, and
`ranking_rows` exactly. This is stronger than metric-level agreement and
rules out evaluator drift as an explanation.

## Results

### Main deployment readouts

| Arm | K8 event AUROC | K8 AP | K8 Pearson | K8 event MAE | K8 latent error | K8 terminal AUROC | Zero-suffix abs return | Chosen-random | Regret |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | .6711 | .1189 | .1615 | .4591 | .0560 | .8701 | .00947 | .2770 | .1286 |
| C-L | .6775 | .1136 | .0552 | .4625 | .0519 | .9186 | .00638 | .1056 | .3000 |
| C-LR | .7359 | .1237 | .1892 | .4348 | .0534 | .9139 | .06404 | .2754 | .1302 |
| G-LA | .7252 | .1210 | .0982 | .4540 | .0637 | .8214 | .04656 | .0627 | .3429 |
| G-LRA | .7197 | .1279 | .1007 | .4505 | .0638 | .8220 | .04743 | .1611 | .2444 |

G-LRA's absolute chosen-minus-random interval is positive, but that is not a
causal intervention interval. It fails the registered preservation tests
against A and C-LR.

### Training-only auxiliary probe

| Arm | Probe BCE, u0 -> u16000 | Event AUROC, final | Sign AUROC, final |
|---|---:|---:|---:|
| G-LA | 1.4573 -> .6903 | .8439 | .9570 |
| G-LRA | 1.4573 -> .6781 | .8815 | .9766 |

The auxiliary clearly learned its own labels. That success did not transfer
to the planner readouts.

### G-LA versus C-L

Favorable changes:

- K8 event AUROC:
  `+.04776 [+.01083,+.08356]`;
- K8 Pearson point:
  `+.04295 [-.03966,+.11325]`;
- K8 event MAE:
  `-.00849 [-.01491,-.00214]`.

Registered failures:

- zero-suffix absolute return:
  `+.04017 [+.01611,+.06689]`, exceeding the `+.02` mechanism budget;
- chosen-minus-random:
  `-.04286 [-.16508,+.12698]`, a worse point rather than an improvement;
- latent error is significantly worse at K1/K2/K4/K8;
- terminal AUROC is significantly worse at K1/K2/K4/K8.

The K8 reward discriminator improvement is real within this DEV evaluation,
but the complete auxiliary mechanism does not pass.

### G-LRA versus C-LR

- K8 event AUROC point:
  `-.01620 [-.06160,+.02624]`;
- K8 Pearson point:
  `-.08849 [-.19810,+.00859]`;
- zero-suffix absolute return:
  `-.01661 [-.04403,+.00718]`;
- chosen-minus-random:
  `-.11429 [-.30004,+.03733]`;
- latent error is significantly worse at every depth;
- terminal AUROC is significantly worse at every depth.

Adding the auxiliary to C-LR reduces false-reward magnitude as a point, but
it does not preserve the reward/ranking mechanism and causes broad temporal
harm. The registered mechanism therefore fails.

### Operational safety versus A

- G-LA zero-suffix delta:
  `+.03709 [+.01981,+.05456]`;
- G-LRA zero-suffix delta:
  `+.03796 [+.01557,+.05885]`.

Both CI upper bounds exceed the allowed `+.02`. Both ranking points are below
A and C-LR. G-LA also has significantly worse K8 event MAE than C-LR.
G-LRA significantly loses K0 AUROC/Pearson and K1 Pearson versus A.

Neither arm is close to an operational pass.

### Fork choices

The auxiliary materially changes decisions rather than merely rescaling
returns:

| Comparison | Changed choices, all 48 | Changed choices, 21 task-differing |
|---|---:|---:|
| G-LA vs C-L | 33 | 12 |
| G-LRA vs C-LR | 26 | 11 |
| G-LA vs A | 26 | 11 |
| G-LRA vs A | 28 | 11 |

Those changed choices are not better on average under the registered
advantage/regret readouts.

## Independent numerical audit

A separate raw-row reconstruction imported no project metric or gate module.
It independently recomputed:

- reward AUROC, Pearson, zero/event MAE at every depth for every arm;
- continuation terminal AUROC and Brier skill at every depth;
- latent means at every registered depth;
- fork advantage, regret, and zero-suffix returns;
- the key episode/environment-cluster bootstrap intervals using the sealed
  seeds and 2,000 draws.

Every point and checked interval matches `stage2g_analysis.json` exactly. The
analysis was also executed twice and produced byte-identical output.

The interval unit remains important: there is one trained seed. Cluster
bootstrap quantifies held-out episode/environment variation, not
training-seed variation. It cannot license a population-level architecture
claim. This limitation cannot rescue either arm because both already fail
the within-seed registered gates.

## Memory and runtime

| Phase | Peak reserved VRAM |
|---|---:|
| Valid preflight/smokes | 174 MiB |
| G-LA full training | 174 MiB |
| G-LRA full training | 174 MiB |
| DEV A/C-L/C-LR | 166 MiB |
| DEV G-LA | 168 MiB |
| DEV G-LRA | 166 MiB |

Full training took `2225.9 s` for G-LA and `2199.5 s` for G-LRA. Each DEV arm
took approximately 41-42 seconds. VRAM is not the failure mechanism.

## Source-faithfulness interpretation

Stage-2G was correctly labelled as a local control.

The official TACO source at pinned commit
`84c38e34f4f9dfd2b059fb6d1356757e8d40712e`:

- concatenates the state representation and encoded action sequence;
- computes a `B x B` state-action/future matching matrix;
- uses identity labels with cross-entropy;
- jointly optimizes its representation and action encoder.

BYOL-AC Eq. 5 uses a distinct predictor `P_a` per action and a stop-gradient
future target. Its theory additionally assumes a uniform policy and symmetric
per-action dynamics.

Stage-2G instead classifies event/sign from only the recorded actual-action
generated state. It has no same-state alternative-action contrast, no
batch-matched future negatives, and no per-action predictor. Therefore:

- the probe can succeed by learning state/context/event regularities without
  learning useful action identity;
- its failure does not refute TACO or BYOL-AC;
- its success on event/sign labels cannot be called evidence of
  action-relevant geometry.

The divergence is now empirically consequential, not merely terminological.

## Is the implementation responsible?

No conventional defect was found. Within the exact seed-505 factorial, the
registered auxiliary intervention itself is the supported cause of the
changes. That conclusion is bounded to this coefficient, sampling contract,
capacity, and training seed.

The most plausible mechanism is objective shortcut/interference:

1. actual-action event/sign classification is solvable from contextual state
   features and does not require separating alternative actions;
2. its shared gradients successfully make those labels linearly separable;
3. the same gradients significantly degrade latent prediction and terminal
   ranking at every depth;
4. fork action ranking does not improve.

This mechanism is an inference from the combined evidence, not a directly
isolated theorem. A coefficient sweep on the spent DEV tier would not
distinguish shortcut from dose and would introduce adaptive selection, so it
is not authorized.

## Wider project ruling and next step

The project has now tested:

- generated latent supervision;
- generated planner-reward NLL;
- frozen-head generated-state adaptation;
- two-scalar post-hoc calibration;
- a matched DreamerV3/CDP reward-distribution operator;
- shared-path event/sign relevance.

None is deployable. Continuing to tune reward losses on the same seed and
spent DEV tier is no longer a disciplined search.

If work continues, the next protocol should change the question from
“can reward events be decoded?” to “does action identity determine the
predicted future?” It should:

1. use a training-only diagnostic and a newly sealed evaluation tier, not
   Stage-2 DEV or FINAL;
2. compare separate, matched arms rather than combining architecture and
   loss changes;
3. retain the present shared action-token baseline;
4. test either:
   - an adequately batched, explicitly labelled TACO-style
     state/action-sequence-to-future contrast; or
   - a matched-capacity action-modulated predictor
     (FiLM/AdaLN or small action-specific modulation), explicitly labelled
     BYOL-AC-motivated rather than BYOL-AC-faithful;
5. include action-shuffled and same-prefix wrong-action controls so that a
   state-only shortcut cannot pass;
6. run the smallest train-only gradient/overfit/VRAM screen before any full
   world training;
7. require matched training seeds and a fresh held-out tier before any
   backend transfer or planner episode.

A literal 17-predictor BYOL-AC transplant needs a parameter-matched control;
the previously estimated approximately 1.8M extra parameters would swamp the
current roughly 240k comparator.

Mamba remains part of the eventual thesis, but Stage-2G is backend-agnostic
negative evidence about the supervision signal. Moving this failed objective
to Mamba would confound the question. GRU remains the diagnostic backend until
an action-identity objective passes; then GRU/Mamba should be compared under
the same validated contract.

## Verification status

Before DEV:

- focused Stage-2G tests: 23 passed;
- full compact suite: 186 passed, one pre-existing BF16 warning;
- new files: clean under Ruff and `py_compile`;
- repository-wide Ruff remained red on 39 pre-existing findings outside
  Stage-2G and was not silently represented as passing.

After the outcome:

- permanent artifact-chain/rejection tests were added;
- exact reward-head-input regression was added;
- focused Stage-2G suite: 26 passed.

The full suite is rerun in the outcome commit.
