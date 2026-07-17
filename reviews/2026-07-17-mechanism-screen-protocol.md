# Mechanism screen protocol: which factor drives the full-grid effect?

Status: **EXPLORATORY, pre-registered before training**, executing the
companion's HOLD condition (2026-07-17 audit: "Do not proceed directly to
fresh-seed confirmation... isolate capacity, flattening, bypass, and
recurrence using already-spent seeds"). Uses ONLY spent resources: training
seeds 505/606, monitor bundle 131-134 (sha 3b45ac6b). Seeds 79-94, 115-130
remain untouched. Nothing here changes defaults or gates.

## Question

The exploratory full-grid arm moved five factors at once (capacity 29k->3.03M
temporal, flattening vs pooling, learned input/output projections, mixing,
bypass removal). Which factor(s) carry the 2-3x separation gain?

## Arms (identical validated contract: T=16, B=4, 16k updates, frozen
## encoder, frozen_dynamics_recipe, AdamW 1e-4, clip 100, bf16; GRU-only —
## the backend was at parity inside both topologies, recorded limitation)

| arm | isolates | construction |
|---|---|---|
| MS-PC | capacity | pooled + bypass topology (step-4 shape: pool -> in_proj -> 2xGRU -> out_proj -> broadcast + dense bypass), hidden mechanically matched to the full-grid arm's ~3.03M temporal params |
| MS-FB | bypass | the exact FlattenedGRUTemporal with the dense residual bypass ADDED BACK (out = x + proj(state)); nothing else changes |
| MS-FF | recurrence | full-grid feedforward: same stem/out projections, parameter-matched, NO temporal state (context is a pure function of current tokens) |

Seeds 505 and 606 per arm (6 runs). Same per-seed shared non-temporal
reference and replay-stream digests as the exploratory screen. Existing
shuffled controls from the exploratory screen serve as the chance reference;
no new controls are trained (screen compares architectures, not causal
gates).

## Pre-registered readout (computed by the runner)

Primary metric: mean separation_all (the metric carrying the H-T effect),
averaged over the two seeds, compared against two anchors taken from the
COMMITTED exploratory report: pooled-64 baseline mean (3 step-4 seeds) and
the full-grid mean (registered seeds 505/606, both backends).

Ordering rule per question, with gap = fullgrid_mean - pooled_mean:
- an arm "reaches the full-grid anchor" if it covers >= 75% of the gap;
- it "stays at the pooled anchor" if it covers <= 25%;
- otherwise "unresolved at screen scale".

R1 capacity (MS-PC): reaches -> capacity explains the effect (the full-grid
   story collapses to "bigger temporal module"); stays -> capacity alone
   refuted (consistent with 4b's large pooled arms, now at matched contract).
R2 bypass (MS-FB): reaches -> bypass removal NOT required; stays -> bypass
   removal load-bearing.
R3 recurrence (MS-FF): reaches -> recurrence not required, the large learned
   projections/mixing suffice (a "spatial-capacity" reinterpretation);
   stays -> temporal state load-bearing.

Secondary records per arm: legacy + tie-aware retrieval (tie-aware is the
preregistered primary for any future confirmation), changed-mask retrieval,
day/night + pixel/task-effective strata, peak allocated AND reserved VRAM,
train minutes, loss tails.

## Consequences (pre-stated)

- The fresh-seed confirmation design depends on which factors survive; the
  confirmation itself remains HELD until user+companion consensus.
- If MS-FF reaches the anchor, the "temporal topology" framing is wrong and
  the next question becomes spatial mixing capacity in the predictor vs the
  temporal path.
- If several arms land "unresolved", the screen scale (24 anchors, 2 seeds)
  was insufficient and the confirmation must carry the factor arms.
