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

## OUTCOMES (appended 2026-07-18; results committed at 5d344e2; companion
## verification + corrections adopted)

All three registered ordering calls: UNRESOLVED at the 75/25 rule.
Independent companion recomputation matches exactly:

| arm | mean separation | gap recovered |
|---|---:|---:|
| pooled-64 anchor | 0.003981 | 0% |
| MS-PC (3M pooled+bypass) | 0.007198 | 52.97% |
| MS-FB (flattened+bypass) | 0.006029 | 33.72% |
| MS-FF (full-grid, no recurrent cache) | 0.007191 | 52.86% |
| full-grid recurrent/no-bypass anchor | 0.010055 | 100% |

Pairwise bypass result (the only clean single-factor pair): restoring the
unit-strength bypass hurt both matched runs (505: 0.006929 -> 0.004659;
606: 0.012439 -> 0.007399).

LICENSED CONCLUSION (companion wording adopted): "the full-grid
recurrent/no-bypass combination remains the best tested candidate, while its
gain appears distributed across capacity, global spatial mixing, explicit
recurrence, and bypass removal. No single mechanism was identified at the
registered resolution." NOT licensed: any single factor proven necessary.

Interpretation corrections on record:
1. MS-FF removes the RECURRENT CACHE, not all temporal state — the world
   model remains autoregressively recursive through generated tokens. Label:
   "explicit recurrent-cache control". It is also not operator-exact (width
   332 + residual GELU vs width 261 + GRU cells); a cleaner future control
   resets the exact flattened GRU's cache every step.
2. The anchors are not fully paired (low anchor = pooled seeds 101/202/303;
   high anchor averages GRU+Mamba full-grid arms; mechanism arms are
   GRU-only, seeds 505/606). Percentages are screening heuristics, not
   paired effect sizes.
3. The bypass is a FIXED unit-scale residual; its negative result may
   involve scale/normalization as well as information bypass. Any revisit
   should use a normalized or learnably gated residual control.

CONSEQUENCE (2026-07-18 tri-party directive): pivot to the one-week
vertical-slice sprint — full-grid/no-bypass Mamba-2 as SPRINT CANDIDATE
(not confirmed default), identically shaped GRU as first-class control,
architecture moratorium during assembly, Phase E task-head validation as
the binding gate before any planner. Confirmation seeds 115-130 remain
reserved; fresh-seed topology confirmation deferred until after the sprint.
