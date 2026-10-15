# Inferred-chance vs energy-score training of the reassignment operator: preregistration

2026-10-15 afternoon. Supersedes the draft `TRANSPORT_INFERRED_PREREG.md` (kept unchanged as history). Written after the
design checks below and before any IC training. Governs `transport_ic_train.py`, `transport_eval.py` and
`transport_recursion.py` on these runs. The user authorized this comparison and asked that uncertain event assignments be
addressed.

## Question

Does training the identical reassignment operator by inferred chance outperform energy-score training, where the
deterministic afterstate takes a point loss given a label-free inferred reassignment and the proposal head takes a
likelihood? The comparison is on coherence, movement, static fidelity and recursive imagination.

The mechanism under test (model-free, OBJECTIVE INVESTIGATION): the two-sample energy score rewards spurious reassignments
as a hedge of deterministic afterstate error. A point loss given the chance outcome cannot be lowered by randomizing, and
a log-score gives rare false events a non-vanishing gradient.

## Design corrections over the draft, each measured before this document (`ic_posterior.py`)

1. **Canonical one-endpoint labels.** The draft labelled a swap at both endpoints. The operator's sampler (independent
   Gumbel-max proposals plus the greedy matching) then accepts swaps 1.59-1.96× too often (synthetic isolated pairs,
   p = .02-.4). Labelling only the up/left endpoint, as DOWN or RIGHT, is calibrated (0.92-1.03×).
2. **Uncertain assignments.** On 400 training windows (6,000 transitions, 98% day), true interior mob moves are classified
   by what the deterministic afterstate already holds:

   | afterstate | PREDICTED (already moved by the deterministic part) | GHOST | STAY (genuine chance events) |
   |---|---|---|---|
   | trained operator | 31% (zombie chase: 500 / 805 zombie moves) | 13% | 56% (626 / 630 cow moves) |
   | D parent | 27% | 22% | 51% |

   The posterior should capture STAY events and leave PREDICTED ones to the deterministic part. Share of STAY events
   captured:

   | rule | trained afterstate | D-parent initialization | precision (D parent / trained) |
   |---|---|---|---|
   | gain > 0 (ML) | 92% | 90% | 0.60 / 0.84 |
   | Bayes MAP (measured scale + prior) | 88% | **54%** | 0.96 / 0.96 |

   Gain > 0's false positives at initialization are mostly same-terrain (0.09 per frame), which are near-identity for a
   point loss. Different-terrain false positives are 0.0002 per frame.

   **Decision:**
   - The deterministic point loss uses the ML assignment (gain > 0). A missed genuine event teaches blur, so misses are the
     costly error there.
   - The chance head uses Bayesian soft targets q = sigmoid(gain / s − logit π) on the ML matching. s is the frame's
     measured mean residual (a Laplace scale); π is a running, label-free rate of accepted swaps per pair. Uncertain
     events get partial credit instead of a hard 0 / 1.
   - No threshold is tuned on labels. The only constants are 0 and the running statistics.
3. **Zombie chase is mostly deterministic.** The afterstate already predicts 62% of zombie moves, so the posterior
   correctly leaves those to the deterministic part. Zombie movement is not a chance-head gate.

## Arms (both seeds; both arms in one process per seed; identical streams)

- **operator (ES):** the Stage 3 energy-score operator, an in-process replicate and the matched baseline.
- **ic:** the identical model, initialization, data, noise streams, optimizer groups and budget. Only the loss differs
  (`transport_ic_train.infer`):
  - point loss = the parent's D loss on LN(T̂·b) vs the observed successor over 81 tokens;
  - plus the canonical soft cross-entropy on the proposal head, weight 1;
  - the proposal head's trunk receives the first Gaussian draw;
  - the sampler at evaluation is unchanged (Gumbel-max proposals plus the exact matching).
- Seed sets 1 and 2 as before (init 7 / 17; windows 20261014 / 20261016; noise 2026101401 / 2026101501).
- Exact D parent, batch 16, L16, 3,000 updates. Decay groups asserted identical by name.

## Readings at 3,000 updates

Both readers; the 104-root evaluation and frozen evaluator of TRANSPORT_COST_PREREG. free_s is the seed's free-synthesis
arm (Stage 3 seed 1, seed 2 run).

**Gates, per seed (IC passes a seed if all hold under both readers):**
- **G1:** all-source valid movement ≥ 15%; cow valid movement ≥ 20.5%; distinct valid outcomes ≥ 2.5;
- **G2:** valid ≥ 90%; duplicate ≤ 5%; absent ≤ 5%;
- **G4:** scored static corruption ≤ max(free_s, 0.5%);
- **G5:** scored joint ES ≤ free_s + 0.002;
- **N, non-inferiority to the in-process ES operator:** valid ≥ ES − 2pp; all-source valid movement ≥ ES − 5pp; cow ≥ ES −
  5pp; distinct ≥ ES − 0.5; joint ES ≤ ES + 0.002.

**Verdicts:**
- **"Inferred chance established (exploratory)"** = IC passes at both seeds.
- **Mechanism falsified** if IC's mean P(move | no-mob afterstate content) is not below the ES operator's at both seeds
  (`transport_diag.py content`).
- **"Cost bought with movement"** if IC passes G4 but fails G1 or N.

**Preregistered secondary readings (recursive imagination, `transport_recursion.py`, all 200 DEV roots, 8 samples, H16).**
Directional predictions, each reported with the ES replicate as reference:
- **R1:** bright terrain error on seen cells at depth 16 is lower for IC than for ES (ES: 7.5-8.1%; deterministic
  afterstate: 1.1%).
- **R2:** bright cow census ratio at depth 16 is closer to 1 for IC than for ES (ES 0.47-0.49).
- **R3:** phantoms per frame stay ≤ 0.01 (transport cannot invent mobs).

**Diagnostics (not gates):**
- posterior swaps per frame, scale s and π during training;
- the posterior's precision / recall against T* on DEV and training windows at u500 / 1,500 / 3,000;
- swap classes;
- P(move) by content;
- cross-terrain endpoint-terrain error (a representation limit, expected unchanged);
- payload substitution.

## Known risks, stated before running

- **EM fixed points.** The posterior uses the reference it trains. Ghosting at initialization (22% of moves) could
  self-reinforce. Watched via posterior swaps per frame, s, and DEV precision at milestones.
- **The canonical convention changes which endpoint proposes.** P(move | content) is compared through accepted swaps
  (swap classes), not through raw proposals.
- **Births, deaths, border crossings and arrows** stay inexpressible by transport (≈3% of interior terrestrial mob changes).
- **Cross-terrain moves** carry their background (5.2% of real moves; representation limit).
- **One budget, no extension, no tuning.** All checkpoints are kept under `transport/ic_seed1`, `transport/ic_seed2`. The
  fresh confirmation set stays sealed.

## Addendum v2 (2026-10-15, 15:1x): implementation bug + one design correction; v1 runs stopped at update 900

**Bug.** The soft target was specified and coded as q = sigmoid(gain/s − logit π). The Bayesian posterior this design
claims is sigmoid(gain/s + logit π).
- With π ≈ 0.01 the flipped sign gives q ≈ 0.99 to every accepted pair, so v1 taught the chance head every candidate swap
  at full strength.
- The MAP rule measured in `ic_posterior.py` (P1) had the correct sign, so the design checks are unaffected.

**Design correction (one change), driven by v1's trajectory and a synthetic test:**
- In v1 the gain > 0 posterior grew from 0.3-0.5 to 2.1-3.6 inferred swaps per frame by update 400-600 (real moves ≈ 0.5).
- `20261015_whiteboard/em_toy.py` (ring world, soft-copy afterstate, point loss through the inferred reassignment, 2 seeds):

  | rule | inferred swaps per step (true 1.2-1.6) | precision | recall |
  |---|---|---|---|
  | gain > 0 | 5.0-5.5 | 0.25-0.31 | 1.00 |
  | MAP, gain/s > log((1−π)/π) | ≈ true | 0.99-1.00 | 0.98-1.00 |
  | oracle | = true | 1.00 | 1.00 |

- Maximum likelihood has no preference for identity among look-alike cells. The point loss cannot correct it, because
  swapping look-alikes costs nothing.

**v2:**
- T̂ = candidates whose posterior odds exceed 1 (MAP: a swap must pay its description cost s·log((1−π)/π)).
- q = sigmoid(gain/s + logit π) on all candidates.
- π = running posterior mass per pair (initial 0.01).
- Everything else, including gates, verdict script, seeds, budget and evaluation, is unchanged.

The v1 artifacts are preserved (`transport/ic_v1_bug_seed{1,2}`, `levers_logs/ic_v1_bug_seed*_train.log`, u500
checkpoints) as the record of the gain > 0 + flipped-sign variant.

**New risk (stated before running):** with the D-parent afterstate, MAP captures 54% of genuine chance moves at
initialization (88% with a trained afterstate). Missed moves teach the afterstate to blur. Watched via posterior swaps per
frame (expected about 0.15-0.4) and, at milestones, the posterior's recall on DEV.

## Addendum v3 (2026-10-15, ~15:35): v2 stopped at update 600 (collapse toward "always stay"); one change

**v2 trajectory.** Posterior swaps per frame went 0.22 → 0.11 → 0.03 (seed 1) and 0.13 (seed 2) by update 600, with the
global prior π falling to 0.0019. Sampled swaps: 0.017 / 0.13 per frame vs about 0.3 genuine chance moves.

**Synthetic test 2** (`20261015_whiteboard/em_toy2.py`; hard regime, where evidence for genuine moves sits near the MDL
threshold, as for the D-parent afterstate):

| rule (2 seeds) | recall | precision | head's swap probability at mob pairs (true 0.19-0.24) |
|---|---|---|---|
| oracle | 1.00 | 1.00 | 0.23-0.24 |
| global prior (v2) | **0.00 (complete collapse)** | — | 0.0002 |
| model's own context prior | 1.00 | 0.94-1.00 | 0.27-0.31 |

- **Mechanism:** a global prior self-tightens. Fewer acceptances lower π, which raises the threshold, which accepts fewer.
- The proper Bayesian form uses the model's own prior for each pair: posterior ∝ p_head(pair | context) × likelihood. Near
  mobs the prior rises; on terrain it falls. Look-alike terrain swaps are rejected by a low prior, not by a global
  threshold.

**v3 (one change):** q = sigmoid(gain/s + logit p_head(j,k)), with p_head the chance head's own (detached) probability of
the canonical proposal (RIGHT for (j, j+1), DOWN for (j, j+9)). T̂ = candidates with q > 0.5. Everything else, including
gates, verdict script, seeds, budget and evaluation, is unchanged.

**Stop rule (fixed now, before v3 runs):** v3 is the last posterior variant in this study. If v3 fails, the result is
recorded and diagnosed; no v4 is launched within TRANSPORT_IC.

**New risk:** the synthetic test shows mild over-dispersion (0.27-0.31 vs 0.19-0.24), so a self-reinforcing prior could
over-accept.

**Artifacts preserved:** v2 at `transport/ic_v2_globalprior_seed{1,2}` (u500 checkpoints, logs).
