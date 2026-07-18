# C-matrix protocol: depth-indexed task heads (registered before implementation)

Status: pre-registered 2026-07-18 per the companion Stage-1 audit section 7.2,
BEFORE any implementation or fitting. Purpose: determine whether task heads
alone can reach the depth-specific probe headroom (K8 event AUROC ~.71/.73),
deciding between "heads suffice" (-> planner gate sooner) and "full-world
Stage 2 justified".

## Arms (one fixed seed per backend first — 505; replicate only a
## discriminator; base = committed X-FLM_s505 / X-FLG_s505; all-frozen
## worlds, heads-only training, Stage-1c natural per-step-K8 schedule)

| arm | head | sampling | purpose |
|---|---|---|---|
| C0 | shared head, per-step K8 | natural | reproduced Stage-1c reference (D8) |
| C1 | DEPTH-INDEXED output layers (one per forecast distance 1..8, shared trunk; Dreamer-4/MTP-INSPIRED — paper Eq. 9 one output layer per distance; NOT faithful: no task conditioning) | natural | shared-head cross-depth conflict |
| C2 | C1 + reward-EVENT auxiliary batches applied to the REWARD loss ONLY (continuation stays natural — decouples the Stage-1 confound) | event term on reward only | magnitude without terminal confound |
| C3 | C2 followed by a natural-distribution head calibration phase | natural final phase | false-reward repair |

## Acceptance (benefit AND harm, fresh-natural + fork-135-142 evaluation;
## these bundles are Stage-1-spent and CANNOT grant planner GO — this matrix
## selects a head design only; the selected design must then be confirmed on
## a newly reserved bundle before any planner work)

- K1 AND K8 event AUROC/AP + signed Pearson/Spearman (C1 must not trade K1
  for K8 — the GRU D8 regression is the failure to beat);
- positive AND negative conditional decoded means (sign correctness);
- event and zero MAE/NLL; cumulative predicted reward on truly zero-return
  suffixes (the H2 false-reward failure, must NOT regress vs C0);
- continuation evaluated separately (natural schedule; no event coupling);
- paired ranking advantage + regret vs C0 (paired per-anchor, env-clustered).

Decision rule: C1 (or C3) "reaches headroom" if K8 event AUROC >= .70 with
K1 not significantly below C0 and zero-suffix false reward <= C0 + .02.
Reaching -> head-design selected, Stage-2 HOLD converts to "not needed for
heads"; not reaching -> full-world Stage 2 justified with the audit's 7.3
loss routing.
