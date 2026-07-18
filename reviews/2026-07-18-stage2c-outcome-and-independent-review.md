# Stage-2C outcome and independent review

Date: 2026-07-18

Reviewer: Codex companion

Status: **complete; both registered deployment gates fail**

## Executive verdict

Stage-2C is valid and materially changes the diagnosis.

There is no evidence that a transition-indexing, target-encoder, recurrent
cache, boundary-mask, checkpoint, or evaluation bug caused the Stage-2
results. The clean factorial shows instead that:

1. **Per-step generated latent supervision works for its direct target.**
   C-L improves held-out latent cosine error at K1, K2, K4, and K8, with all
   four paired confidence intervals below zero.
2. **That representation gain is not sufficient for planning.** C-L
   significantly worsens fork action ranking and weakens the deployed reward
   readout. A better latent cosine score is therefore not a valid proxy for a
   better control model.
3. **Generated reward gradients through the shared dynamics are causal.**
   Adding only the registered `0.10` generated-reward term restores the lost
   action ranking and improves deep reward discrimination, magnitude, and
   continuation calibration.
4. **The same reward term is not deployable.** It creates a statistically
   clear false-reward bias on truly zero-return suffixes and gives back part of
   the C-L latent gain. Continuation is not the failure in this clean arm.

The correct result is therefore neither “per-step supervision failed” nor
“Stage 2 repaired the world model.” It is:

> Per-step generated latent targets improve latent rollout fidelity, but
> reward gradients through the shared dynamics trade that fidelity for task
> discrimination and induce unacceptable false reward. The representation
> and task objectives must now be separated.

Do not transfer this objective to Mamba, do not replicate it at more seeds,
and do not execute the planner yet.

## Contract and provenance audit

The run respected the pre-outcome protocol
`reviews/2026-07-18-stage2c-decoupled-protocol.md`:

- baseline A checkpoint SHA-256:
  `fcbc9407a36faf59e32ec1425c2fbee7a5e5a21ea73cb13170a828e4e9c6d1f2`;
- required common initializer:
  `55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260`;
- required uniform schedule:
  `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0`;
- C-L weights: latent `1.0`, reward `0.0`, continuation `0.0`;
- C-LR weights: latent `1.0`, reward `0.10`, continuation `0.0`;
- both arms used the same 16,000-update schedule and computed the same
  components;
- neither arm used a terminal/event pool;
- both encoder-freeze assertions passed;
- the evaluator never loads or indexes `manifest["final"]`.

Artifacts:

| Artifact | SHA-256 |
|---|---|
| `stage2c_cl_s505.pt` | `227479107568901e8ed1945c31de17fba2c0f2d197541f9b3a3ee8d554a06aa1` |
| `stage2c_clr_s505.pt` | `60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae` |
| `stage2c_report.json` | `b73360a52bb137ef939a45c55f247fd0091011273fd6b1c1b8594201101706fc` |
| `stage2c_raw.json` | `e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5` |
| `stage2c_analysis.json` | `39e4c81a1ac13b8fc5f1144724a16a832a74d63ed5e3515faaddd86dbe2b6c0b` |

All checkpoint tensors are finite. Both checkpoints strictly instantiate
`FullGridGRUTemporal`, carry the exact initializer digest, and contain 16,000
entries for each recorded objective component.

Training was matched between the two new arms:

| Arm | Minutes | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| C-L | 28.60 | 138.71 MiB | 176.00 MiB |
| C-LR | 28.63 | 138.71 MiB | 176.00 MiB |

The full CUDA suite passed before fitting: **121 passed**, with one known
PyTorch warning and no skips. The focused Stage-2 objective/evaluator suite
passed **10/10**, including K1/K2 indexing, post-terminal masking, equivalence
to the old combined loss, gradient isolation, schedule exclusion, cluster
resampling, gate direction, and a regression test that forbids access to the
FINAL manifest tier.

## Registered results

All intervals below are paired 95% cluster-bootstrap intervals. Reward,
continuation, and latent rows resample episodes; ranking and zero-suffix rows
resample environment seeds.

### G1: generated-latent arm

The direct latent mechanism passes:

| Depth | A error | C-L error | C-L - A (95% CI) |
|---|---:|---:|---:|
| K1 | .02269 | .02086 | -.00183 `[-.00214, -.00155]` |
| K2 | .02845 | .02617 | -.00228 `[-.00265, -.00190]` |
| K4 | .03871 | .03571 | -.00299 `[-.00392, -.00211]` |
| K8 | .05598 | .05186 | -.00412 `[-.00618, -.00210]` |

But the deployment-safety gate fails:

- chosen-minus-random falls `.27698 -> .10556`;
- paired ranking delta is `-.17143 [-.41592, -.01245]`;
- regret rises `.12857 -> .30000`, with the paired CI also excluding zero;
- K8 reward Pearson falls `.16146 -> .05523`;
- K8 decoded event magnitude falls `.00570 -> .00294`;
- K1 continuation Brier skill worsens by
  `-.00731 [-.01409, -.00185]`.

The continuation change is numerically small in absolute probability
(`P(terminal | nonterminal)=.00073`) and deep continuation improves, but the
registered safety rule is correctly strict. False reward does not explain the
ranking loss: C-L reduces absolute zero-suffix prediction
`.00947 -> .00638`.

**G1 deployment gate: FAIL. Direct latent mechanism: PASS.**

### G2: generated-reward increment

C-LR improves every registered K8 reward point estimate versus A:

| Metric | A | C-LR | Paired delta where registered |
|---|---:|---:|---:|
| event AUROC | .67114 | .73594 | +.06480 `[+.01407, +.12673]` |
| average precision | .11889 | .12368 | +.00479 |
| signed Pearson | .16146 | .18915 | +.02769 |
| decoded event magnitude | .00570 | .03369 | +.02799 |

The incremental comparison against C-L is also positive: K8 AUROC rises
`.67748 -> .73594` and event magnitude `.00294 -> .03369`. Ranking is
restored to baseline (`.27540` versus `.27698`), and the C-LR-versus-A ranking
CI comfortably spans zero.

Continuation does **not** reproduce old Arm B's collapse:

- K8 terminal AUROC: `.87009 -> .91388`;
- K8 Brier skill: `-.00590 -> +.00088`;
- paired K8 Brier-skill delta:
  `+.00679 [+.00416, +.00976]`;
- K8 nonterminal termination probability remains `.00071`.

Two safety conditions fail:

1. Absolute zero-suffix predicted return grows
   `.00947 -> .06404`, a paired delta of
   `+.05457 [+.03682, +.07243]`, well beyond the registered `+.02` ceiling.
   Continuation gating barely changes it (`.06319`), proving that a
   continuation fix cannot remove this error.
2. C-LR gives back a significant fraction of C-L's latent gain at K1-K4
   (`+.00081`, `+.00109`, `+.00134` versus C-L). It nevertheless remains
   significantly better than A at every depth, so this is an objective
   trade-off rather than a total latent failure.

**G2: FAIL.**

## What was responsible

The implementation is responsible in the scientifically relevant sense, but
not because it is conventionally broken.

- Generated latent gradients improve the autoregressive latent path.
- Generated reward gradients reach the same temporal/predictor trunk and
  alter that representation. The C-L/C-LR factorial shows they cause the
  reward/ranking recovery, the false-reward bias, and part of the latent
  regression.
- The old Arm B continuation collapse was not intrinsic to per-step reward
  supervision. It arose from the old combined objective and terminal/event
  distribution. Removing generated continuation and the terminal pool repairs
  continuation.
- The failure is not explained by Mamba, because all arms here use the same
  GRU backend.
- The failure is not explained by frozen encoder insufficiency: the
  intervention improves latent prediction while degrading task deployment.
  The conflict occurs after the encoder.

This is an objective-routing problem: one shared trunk is being asked to
optimize cosine latent fidelity and a sparse, heteroscedastic reward
likelihood, and the latter can improve discrimination by shifting the reward
baseline on abundant zero-reward states.

## Gate-quality review

The gates checked the right deployment properties and prevented a misleading
selection on K8 AUROC alone. One interpretation correction is necessary:

- G1 intentionally combines a **mechanism test** with **deployment safety**.
  Its overall failure must not erase the four-depth latent result. Future
  records should report “latent mechanism pass / deployment gate fail.”
- Average precision and false reward are indispensable because AUROC is
  insensitive to probability scale and class prevalence. C-LR demonstrates
  exactly why AUROC alone is unsafe here.
- Planner ranking is necessary but not sufficient. C-LR matches A's ranking
  while hallucinating reward, so it is still not executable.

The DEV tier has now been used for several decisions. These results are
diagnostic and cannot license a final claim. The FINAL tier remains sealed.

## Required next step

Follow the registered “separated task adapter/head objective” route. The
smallest discriminating control is:

1. start from the C-L checkpoint, which has the best latent rollout;
2. freeze encoder, temporal core, future predictor, continuation head, and all
   non-reward parameters;
3. compare equal-update reward-head-only adaptation on:
   - teacher-forced real states at all matched target positions; and
   - the same schedule with K1/K2 replaced by generated states;
4. use natural uniform replay only—no event or terminal pool;
5. require bit-identical latent and continuation outputs;
6. require reward/ranking recovery **and** the existing zero-suffix budget.

This local factorial is justified by Stage-1's equal-update evidence and
Stage-1C's registered natural-recalibration route. It is not a literature
reproduction. If it fails, stop optimizing the present scalar reward decoder
and revisit reward parameterization/calibration before any larger world-model
or Mamba run.

## Decisions

- C-L as deployable world model: **NO-GO**.
- C-LR as deployable world model: **NO-GO**.
- Per-step generated latent supervision as a measured mechanism: **YES,
  retained as diagnostic evidence**.
- Generated reward through the shared world-model trunk: **NO-GO**.
- Generated continuation / terminal curriculum: **NO-GO**.
- Mamba transfer or replication: **NO-GO**.
- Planner execution and online policy learning: **NO-GO**.
- Small frozen-trunk, reward-head-only factorial: **GO on spent DEV as a
  diagnostic only**.
