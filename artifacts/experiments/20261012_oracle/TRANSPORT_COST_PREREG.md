# Transition cost and second-seed replication: preregistration

2026-10-15. Written after the user's second transport mandate. Fixed before the cost probe, before any cost-trained
model, and before any second-seed training. Governs `transport_cost_diag.py probe`, `transport_train.py` with
`--seed-set/--arms/--cost`, and the evaluation of those runs.

## Principle and fit with the existing architecture

**Principle:** preserve what persists, learn what changes, sample only what is genuinely unpredictable.

In our world model, deterministic CorrT/CorrG already computes a soft copy correspondence over (self, 4 neighbours) plus
a generated candidate. The F1b control showed that this deterministic reference becomes a clean pre-chance afterstate
once a stochastic operator owns the moves.

The operator under test samples an exact one-to-one correspondence over the same (self, 4 neighbours) neighbourhood on
top of that afterstate. It is a stochastic correspondence, with no object slots, no terrain rules and no new decoder.

The **transition cost** prices departures from identity: λ · E[number of tiles proposing a change per frame], that is
λ · Σ_j (1 − P_j(stay)). It is the minimum-description-length prior "every sampled change must pay for itself in the
predictive score". It treats all content alike and uses no labels. It adds no parameter; the cost arm is the identical
model with one extra loss term.

**Limit stated in advance:** the operator's proposal logits come from its own head on the noise trunk, not from CorrT's
routing scores. Deriving the stochastic correspondence from CorrT itself would be more unified. It is not done here,
because it would change the architecture inside a matched comparison.

## Preconditions measured (NOTEBOOK, transport cost section)

1. **Lineage is clean** (`transport_lineage.py`):
   - Stage 3's pinned inputs contain no layouts, F1b code or checkpoints, fan or panel.
   - Both arms' u0 weights equal the D parent exactly (145 tensors) and differ from every F1b checkpoint.
   - The D parent is the oracle study's control arm, with identically zero oracle weights.
   - F1b was strictly an oracle positive control: evaluated only, never loaded for training.
   - Disclosure: the exploration prior β ("~1 proposal per frame") was chosen after seeing the census rate of 0.5 moves
     per step. It is one label-free scalar.
2. **The cross-terrain contextual-token problem cannot be fixed by a cost:**
   - 5.2% of real in-view mob moves cross terrain classes.
   - Swapping the TRUE tokens decodes 100% of those endpoints with the wrong terrain, against 0% for same-terrain moves.
   - The seed-1 operator moves across terrain in 10.8% of its mob moves.
   - This stays a documented representation limit (content bound to background) and is not a gate here.
3. **What the score pays per unit of move probability** (straight-through, 48 stationary training windows, label-free
   tile classes; seed-1 operator at 500 / 1,500 / 3,000):

   | content | value |
   |---|---|
   | static | +4.5e-5 / +9.8e-5 / +2.9e-5, window SE 5.7–7.4e-5 (indistinguishable from 0) |
   | changing, along the realized move | +5.2 to +6.5e-3 |
   | changing, other directions at moving steps | +3.5e-3 |
   | changing, at staying steps | −1.1 to −2.3e-4 |

   The expected value of moving content that changes at rate r per direction is about 1.65–1.78e-2 · r − 1.1 to
   2.3e-4.

## The single cost value (rule fixed here, not tuned)

**λ = 1e-4.** It is the smallest one-significant-figure value above the largest measured mean value the score pays for
moving static content (9.8e-5).

With the measured conditional values, it keeps a positive expected incentive for content that changes in more than
about 1.2% (u3000), 1.4% (u1500) or 2.0% (u500) of steps per direction. Real cows change in about 9–12% per direction.

The margin is thin: about 2–3× at the level of all labelled mob tiles. The stay-still risk is real and is tested below.

## Go / no-go probe (before any cost training from scratch)

**Setup:**
- From the seed-1 operator at 3,000 updates, model and optimizer state restored from Stage 3's state.
- Two matched 1,000-update continuations: no cost vs cost λ.
- Identical fresh windows (seed 20261030) and noise (seed 2026103001); phase LR (1e-4).
- Measured on the same 48 stationary training windows: mean P(move) for label-free static tiles, and for labelled
  cow / zombie / skeleton tiles (labels used diagnostically only).

**Pass** only if, relative to the no-cost twin:
- P(move | static) is reduced by ≥ 30%, **and**
- P(move | cow) is reduced by ≤ 15%.

**Fail → stop:** record and report. The cost experiment, and the cost-dependent part of this mandate, is not launched,
and no λ is retried.

## Experiment (launched only if the probe passes)

**Runs:**
- Every run starts from the exact D parent. Same backbone, reference, operator, data ledger construction, phase AdamW
  (decay groups asserted identical by name on shared tensors), energy objective, batch 16, L16, **3,000 updates**.
- Arms within one process share windows, Gaussian noise and Gumbel draws.
- **Seed set 1** (completion init 7, windows 20261014, noise 2026101401, the original Stage 3 streams): one process
  trains {operator, operator_cost}.
  - The in-process operator is a numerical replicate of Stage 3's seed-1 operator and the matched control for the cost.
  - Stage 3's seed-1 free arm remains the free reference for seed 1.
- **Seed set 2** (completion init 17, windows 20261016, noise 2026101501): one process trains {free, operator,
  operator_cost}.

**Evaluation:** identical to Stage 3 (104 bright DEV roots × 64 samples, 14 strict sources, both frozen readers, the
frozen evaluator for static corruption, 2,000 root-cluster bootstrap draws).

**Readings and gates.** Every gate holds under BOTH readers, at 3,000 updates.

- **R, architecture replication (seed 2):** operator − free lower 95% bound > 0 for valid AND for all-source valid
  movement. Seed-2 operator's G1/G2/G4/G5 are reported. G4 is expected to fail without the cost.
- **C, cost success (per seed, s ∈ {1, 2}).** operator_cost passes all of:
  - G1: all-source valid movement ≥ 15%; cow valid movement ≥ 20.5%; distinct valid outcomes ≥ 2.5;
  - G2: valid ≥ 90%; duplicate ≤ 5%; absent ≤ 5%;
  - G4: scored static corruption ≤ max(free_s, 0.5%);
  - G5: scored joint ES ≤ free_s + 0.002;
  - N, non-inferiority to the same seed's in-process unpenalized operator: valid ≥ op − 2pp; all-source valid movement
    ≥ op − 5pp; cow valid movement ≥ op − 5pp; distinct valid ≥ op − 0.5; ES ≤ op + 0.002.

**Verdicts:**
- **"Transition cost established (exploratory)"** = C at both seeds AND R.
- C at one seed only = **"seed-dependent"**.
- Passing G4 while failing G1 or N = **"cost buys static fidelity with movement"**, which is a failure.
- Fresh confirmation (base seed 2026101400) stays sealed unless "established". Even then, it is the next step only on
  the user's approval.

**Diagnostics (not gates):**
- swaps per frame by class (mob ↔ passable / solid / mob, terrain ↔ same / different);
- P(move) by content;
- static corruption of samples vs the reference alone;
- cross-terrain share of mob moves;
- payload substitution;
- the 500 / 1,500 trajectory;
- the seed-1 numerical-replicate difference.

## Stop rules

- No λ sweep, no extension, no further repair inside this study.
- Bugs are fixed only if they are bugs, and recorded.
- All earlier checkpoints and results are preserved. New runs write to new directories (`transport/seed1_cost`,
  `transport/seed2`).

## Out of scope (kept visible)

This addresses WHITEBOARD row 3, coherent stochastic transitions. It does not test:
- nuisance invariance;
- recurrent memory sufficiency;
- persistent epistemic state;
- recursive imagination;
- actor improvement;
- sparse stochastic computation. No efficiency claim is made from the architecture.
