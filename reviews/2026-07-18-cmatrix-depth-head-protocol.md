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

## OUTCOMES (appended 2026-07-18; companion audit verified — its table
## reproduces exactly from the committed report)

CLEAN PREREGISTERED NEGATIVE: C1, C2, C3 all REJECTED by the selection rule
(K8 AUROC >= .70 never reached: best .682; C2/C3 exceed the false-reward
budget: zero-suffix predicted return up to +.13 over C0; Mamba C1
significantly WORSENS ranking [-.0689,-.0071]; GRU C3 damages K1
[-.1239,-.0251]). No replication seed needed for this candidate screen; one
seed does NOT license "depth-indexed heads can never help".

FLAWS ON RECORD (companion findings, all conceded):
1. CONTINUATION POSITION SHORTCUT: terminal labels occur ONLY at depth slot
   8 in the training windows (131/131), so depth-indexed continuation for
   depths 1-7 trained purely on "continue" — K1/K2/K4 terminal AUROC =
   exactly .500; the strong K8 value is substantially a window-position
   artifact. Depth-indexed continuation is INVALID as deployment evidence;
   future continuation training needs terminal + matched non-terminal
   targets at EVERY supervised depth + post-terminal masking.
2. C0 is not an exact Stage-1c D8 reproduction (schedule off-by-one, fixed
   in the runner; max drift .0022 AUROC, identical rankings).
3. NOT Dreamer-4 MTP (relabelled "depth-indexed generated-state readout
   control"); this screen does NOT disprove MTP.
4. C1 capacity confound recorded (heads 6.29x params).
5. C2 coefficient / C3 calibration-budget were implementation choices, not
   protocol-specified; the negative applies to this implementation.

CONSEQUENCE (user + companion consensus): Stage-2 HOLD LIFTED for one
controlled A/B — reviews/2026-07-18-stage2-ab-protocol.md.
