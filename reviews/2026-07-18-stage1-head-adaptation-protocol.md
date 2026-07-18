# Stage-1 protocol: head-only generated-state task adaptation

Status: **pre-registered before fresh-bundle collection and before any
fitting** (2026-07-18; executes the companion's Phase-E consensus audit §6
Stage 1 — its GO items only). Purpose: the smallest discriminating,
mechanism-matched test of whether naturally calibrated task heads can learn
the task mapping on the generated-state distribution that the fixed worlds
already preserve (frozen-context probes: K8 event AUROC .707/.734).

## Fixed worlds, adapted heads

The six committed full-grid checkpoints (X-FLG/X-FLM x 505/606/707). ALL
parameters frozen except the reward and continuation heads (H3 additionally
trains two tiny auxiliary linear heads). Encoder frozen under the executable
contract (enforce_frozen_encoder + assert_encoder_frozen at completion).
Training data: the pinned training replay ONLY. B=8, 3,000 head updates,
AdamW lr 1e-3, bf16, windows of 10 observations = 8-real-step prefix + 2
transitions.

| arm | generated-state task loss | sampling / objective |
|---|---|---|
| H0 | none (untouched checkpoints) | baseline |
| H1 | reward+continuation supervised at BOTH real prefix steps and the K=1,2 generated steps (shared heads) | natural window sampling |
| H2 | as H1 | 50% uniform / 50% event-containing windows for the task update (Dreamer-4-INSPIRED relevance mixture, not source-faithful) |
| H3 | as H1 + auxiliary event and sign BCE heads (planner reward head stays unweighted two-hot) | natural sampling |

Pairing: identical replay window schedule per (checkpoint, arm) for H1/H3;
H2's schedule differs by design (recorded). No temporal/predictor/encoder
updates anywhere; parameter-set assertions enforced.

## Fresh evaluation data (hash-pinned BEFORE any fitting; canonicalized
## collection with the repaired collectors)

- Natural depth set: random-policy episodes, env seeds 940-955, cap 400
  steps — same-target reward evaluation at K in {0,1,2,4,8}.
- Terminal set: seeds 916-931 run to termination — continuation at depth.
- Ranking bundle: canonical fork bundle, seeds 135-142 (8 seeds, 4 day/2
  night, 3 common-RNG branches) — suffix ranking + chosen-vs-random.
Seeds 79-94 and 115-130 remain untouched; 131-134/900-915 are not reused
for acceptance.

## Acceptance (companion's list; H-arm vs H0, same fresh data, both
## backends, per training seed)

Improvement required on: event AP and AUROC; signed reward Pearson/Spearman;
event MAE/NLL and decoded event means (magnitude collapse is the known
failure: .229 -> .005/.019 by K8); continuation Brier SKILL vs climatology,
terminal AP/AUROC and P(term) at K=1/2/4/8; suffix ranking advantage and
regret. No arm may be selected on training loss, the old 131-134 bundle, the
probe set, or a single training seed. Natural-distribution evaluation only —
no class weights at evaluation.

## What this cannot do

No planner GO by itself (a planner GO ultimately requires executed
planner-vs-random episodes on a further fresh bundle under the revised gate
of the consensus audit §7). No architecture/topology changes; no K=5; no
full-world retraining (that is Stage 2, conditional on this result); no
policy training.
