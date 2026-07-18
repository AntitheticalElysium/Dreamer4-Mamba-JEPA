# Stage-2C protocol: uniform generated-latent/reward factorial

Status: **pre-registered before implementation fitting or outcome inspection**  
Date: 2026-07-18  
Authority: user accepted the independent Stage-2 audit and authorized the
corrective implementation, tests, training, and committed review.

## Question

Stage-2 Arm B improved one deep reward discriminator while worsening latent
accuracy, continuation calibration, and false reward. Its generated objective
and terminal pool changed too many mechanisms together.

This experiment asks two smaller questions:

1. Does supervising both generated K1 and K2 latents improve autoregressive
   latent accuracy beyond the existing final-K2 rollout bridge?
2. Conditional on that latent path, can a gradient-balanced generated reward
   term improve deep reward without recreating the false-reward, continuation,
   or latent failures?

It does **not** test terminal calibration, Mamba, online policy learning, or a
source-faithful reproduction.

## Source boundary

- SPR source `mila-iqia__spr@0b9dd4e7b9bbdfaecdf9a3713bf5931fb54ab0ca`
  predicts latent state and reward at every recurrent jump
  (`src/models.py:438-469`) and masks latent SPR losses after terminals
  (`src/algos.py:269-303`).
- Official V-JEPA 2 source
  `facebookresearch__vjepa2@204698b45b3712590f06245fbfba32d3be539812`
  scores every element of an autoregressive latent sequence
  (`app/vjepa_droid/train.py:425-449`).

The local loss is **SPR/V-JEPA-2-shaped**. It is not a reproduction: the local
model has a frozen convolutional encoder, separate temporal and future
predictor modules, cosine targets, Crafter rewards, and no joint Q-learning.
Neither source supplies our reward coefficient.

## Fixed baseline and data

- Reuse Stage-2 Arm A checkpoint
  `reviews/artifacts/stage2_armA_s505.pt`, SHA-256
  `fcbc9407a36faf59e32ec1425c2fbee7a5e5a21ea73cb13170a828e4e9c6d1f2`.
- Require initial-state digest
  `55e31261de2ced792bab1754d9060cefefb682d4964324fbca5643da8d2c7260`.
- Require main schedule digest
  `427eb8a311ac9a99ec7f5fd529added9035777a1146864c4ab53d68c2c1295d0`.
- Training replay and DEV hashes must match
  `reviews/artifacts/stage2_eval_bundles.manifest.json`.
- DEV remains the already-spent diagnostic set: natural seeds 960-975,
  terminal seeds 932-947, and fork seeds 143-150.
- The pinned FINAL tier—natural 976-991, terminal 948-963, fork 151-158—must
  not be loaded, deserialized, evaluated, or used for selection.

## Matched arms

All new arms use GRU seed 505, the exact Stage-2 fresh initializer, identical
uniform 16-observation replay windows, 16,000 optimizer updates, batch 4,
AdamW `1e-4`, gradient clipping 100, bf16, prefix 8, and two generated steps.
The encoder stays frozen. The unchanged base objective includes teacher-forced
JEPA, reward, and continuation plus its existing final-K2 rollout bridge.

For a generated step \(k\), with post-terminal mask \(m_k\):

\[
L_{\mathrm{lat},k}
=m_k\left[1-\cos(\hat z_{t+k},z^-_{t+k})\right],
\qquad
L_{\mathrm{rew},k}
=m_k\,\mathrm{NLL}(\hat r_{t+k},r_{t+k}).
\]

The arms are:

- **A — pinned baseline:** base objective only; no retraining.
- **C-L — latent factorial:** base plus the mean K1/K2 generated latent loss.
- **C-LR — reward factorial:** identical to C-L plus `0.10` times the mean
  K1/K2 generated reward NLL.

Both new arms compute the same generated latent and reward components; only
the registered scalar reward coefficient differs. Generated continuation is
exactly zero in both. No terminal pool, event pool, relevant-sequence mixture,
or auxiliary batch is allowed.

The reward coefficient `0.10` is fixed before fitting. It is the rounded
gradient-balance value from the Stage-2 audit: on the fixed initial batch,
unweighted shared-dynamics norms were latent `2.52` and reward `23.54`
(`2.52/23.54 = .107`). This is a local scale correction, not a
source-derived hyperparameter. The runner records the same pre-training
diagnostic but may not change the coefficient from its result.

## Required executable checks before fitting

1. Synthetic observation/action/reward/continuation indexing at K1 and K2.
2. Post-terminal masking: a terminal at K1 excludes K2 for every component.
3. Component equivalence with the committed Stage-2 combined loss at weights
   `1/1/1`.
4. Generated latent-only backward gives zero reward- and continuation-head
   gradients.
5. Generated latent+reward backward gives zero continuation-head gradients.
6. C-L and C-LR consume exactly the pinned uniform schedule and no terminal
   pool.
7. Fresh initialization, encoder identity, baseline checkpoint, replay,
   manifest, and schedule hashes are asserted.
8. Evaluator tests cover paired cluster resampling, zero-suffix reward,
   continuation calibration, latent error, and gate direction.
9. Full existing CPU/GPU suite and compileall pass.

## Raw outputs and paired uncertainty

The runner must save, for A/C-L/C-LR:

- complete strict checkpoints and component histories;
- reward predictions at K0/1/2/4/8;
- continuation predictions at K0/1/2/4/8;
- per-target last-predictor latent cosine error at K1/2/4/8;
- complete fork ranking rows, including raw and continuation-gated predicted
  returns;
- training time, peak VRAM, source/script/config/init/schedule/checkpoint
  identities.

All comparisons use the identical targets. Reward, continuation, and latent
intervals resample episodes; ranking and zero-suffix intervals resample
environment seeds. Selection uses paired contrasts, not overlap of marginal
confidence intervals.

## Gates

This is a one-seed discriminator. A pass licenses fresh-seed replication, not
an architectural claim or planner execution.

### G1 — generated-latent mechanism, C-L versus A

- K1 and K2 latent cosine-error point deltas must be negative, with at least
  one paired 95% CI strictly below zero.
- K4 and K8 latent error must not show significant harm; neither point delta
  may exceed `+.005`.
- Continuation Brier skill must not significantly worsen at any depth, and
  nonterminal predicted-termination probability may not exceed
  `max(A + .01, .02)`.
- Ranking advantage/regret and zero-suffix false reward must not
  significantly worsen.

### G2 — generated-reward increment, C-LR versus C-L and A

- Versus A, K8 event AUROC, average precision, signed Pearson, and decoded
  event magnitude must all improve as points; the paired AUROC CI must exclude
  zero.
- Versus C-L, K8 event AUROC and decoded event magnitude must improve as
  points, demonstrating that the added reward term—not latent supervision
  alone—contributes to its registered target.
- K1 event AUROC and Pearson must not significantly worsen.
- Raw cumulative prediction on truly zero-return suffixes must be at most
  `A + .02`; the paired CI upper bound must also be `<= .02`.
- Latent error must not significantly worsen against C-L at any depth; versus
  A, no point delta may exceed `+.005`.
- Continuation Brier skill and nonterminal termination must satisfy the G1
  safety rule.
- Ranking advantage and regret must not significantly worsen versus A.

### Overall routing

- **G1 fail:** stop full-world generated-objective expansion. Diagnose why the
  already-present final-K2 rollout bridge and per-step latent target conflict;
  do not add reward, Mamba, or terminal curricula.
- **G1 pass, G2 fail:** retain C-L only as a latent diagnostic. Generated
  reward through shared dynamics remains rejected; next consider a separated
  task adapter/head objective, not a larger recurrent backend.
- **G1 and G2 pass:** replicate C-LR on GRU seeds 606/707 and matched Mamba
  before touching FINAL. A replicated pass licenses a separately registered
  planner-vs-random execution gate.

Actor/critic, reliability weighting, predictor mixtures, FINAL evaluation, and
online policy training remain NO-GO throughout this experiment.
