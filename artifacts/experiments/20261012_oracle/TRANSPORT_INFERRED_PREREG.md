# Inferred-chance training vs energy-score training of the reassignment operator: preregistration (NOT launched)

2026-10-15. Written after the model-free objective investigation (NOTEBOOK, "OBJECTIVE INVESTIGATION"), before any
training of this comparison. The user asked for a proposal only. This document fixes the comparison and its
falsification criteria. Nothing here has run.

## Why this comparison, and not localized scoring

The model-free investigation (`transport_objective.py`) found:

1. **Local joint energy scores do not help.**
   - Exact real-fan mixtures: a spurious different-terrain swap's penalty relative to the value of a genuine move rises
     only 0.077 → 0.091 (global → 5-cell crosses).
   - Dependence sensitivity falls with locality (global 4-8% of the movement margin; crosses 2-3%; pairs 1-2%; per-token
     0).
   - Through the actual operators' straight-through head, local rules *reward* spurious static swaps (+0.9 to +1.4e-4 per
     unit probability) and *discourage* labelled mob movement. The global rule is ≈ 0 on static content and positive on
     mobs.
2. **Mechanism: spurious reassignments hedge deterministic error.**
   - The two-sample first-power energy score with locally identical draws never penalizes a one-draw perturbation unless
     the prediction is unbiased (triangle inequality).
   - Dose-response through the frozen operators: static-tile push −8.8e-5 (t −3.8) with zero static error (privileged
     substitution) → ≈ 0 with the learned soft afterstate (bias 0.014) → +7e-5 to +2e-4 with a hard-copy afterstate (bias
     0.028). Seed 2 replicates (oracle t −3.1).
   - Genuine-move value is unchanged in every case.
3. **The deterministic error at static cells is contextual drift of the tokens**, not unpredictable nuisance (bright
   views):
   - real-future spread at static cells 0.0018;
   - learned afterstate bias 0.014;
   - an exact copy of the current token 0.048.
   Persistent content does not have a persistent token.
4. **The static gate is not limited by the representation under correct reassignment.** Every reference swapped with the
   true assignments scores 0.27-0.30% (true tokens 0.18%). The excess comes entirely from wrong reassignments.
5. **The chance outcome is recoverable from observation alone** (label-free local search, gain > 0, greedy one-to-one).
   With the learned unprivileged afterstates in bright views: recall 0.78-0.80, precision 0.95-0.99, ≈ 0 false terrain
   swaps per frame. It is poor in darker views (precision 0.24-0.28: night nuisance).

So the energy score is not shown unsuitable in general: given an accurate afterstate, it penalizes spurious swaps (point 2).
In our target space, though, the afterstate's unavoidable drift error is converted into rewarded spread. Stochastic MuZero's
decomposition (WHITEBOARD §3) removes that incentive by construction:
- the deterministic part takes a point loss *given* the chance outcome, and a point loss cannot be lowered by randomizing;
- the chance variable takes a likelihood (log-score) loss, whose gradient on rare false events does not vanish.

The chance outcome comes from a posterior; point 5 says a parameter-free one is feasible in bright views.

## Arms (identical model; only the training objective differs)

- **ES:** the existing operator objective (TRANSPORT_PREREG / TRANSPORT_COST_PREREG): two Gumbel-max samples, the exact
  matching, straight-through, first-power energy score on the 35 scored tokens.
- **IC, inferred chance.** Per update, with the current reference a = LN(b) detached:
  - **Posterior:** gain(j, d) = |a_j − z_j| + |a_k − z_k| − |a_k − z_j| − |a_j − z_k| for each tile j and direction d to
    neighbour k = j + d (interior map tiles, no player), against the observed successor z. Choice = argmax_d gain, kept
    only if gain > 0. The SAME parallel greedy matching as the sampler (`TransportCompletion.matching`) resolves conflicts,
    giving T̂. Nothing is learned, and there is no threshold other than 0.
  - **Deterministic loss:** the parent's own D loss (MSE + 0.1·L1 + 0.01·max) on LN(T̂·b) against z, over all 81 tokens,
    as in F1b but with T̂ inferred instead of the simulator's T*.
  - **Chance loss:** cross-entropy of the proposal logits (masked as in sampling) toward each map tile's T̂ class (stay, or
    the direction of its swap partner, labelled at both endpoints), averaged over map tiles and frames. Weight 1, not
    tuned.
  - Sampling and evaluation are unchanged: Gumbel-max proposals plus the exact matching.

**Shared:** the exact D parent; backbone, reference and trunk identical at initialization; ledger, noise and Gumbel
streams; phase AdamW with decay groups asserted identical; batch 16, L16, 3,000 updates. Seed sets 1 and 2, both arms in
one process per seed.

## Readings and gates (both readers, 3,000 updates, the same 104-root evaluation and frozen evaluator)

IC passes at a seed if it satisfies:
- **G1:** all-source valid movement ≥ 15%; cow valid movement ≥ 20.5%; distinct valid outcomes ≥ 2.5;
- **G2:** valid ≥ 90%; duplicate ≤ 5%; absent ≤ 5%;
- **G4:** scored static corruption ≤ max(free_s, 0.5%);
- **G5:** scored joint occupancy ES ≤ free_s + 0.002;
- **N, non-inferiority to the same seed's in-process ES arm:** valid ≥ ES − 2pp; all-source valid movement ≥ ES − 5pp;
  cow ≥ ES − 5pp; distinct ≥ ES − 0.5; joint ES ≤ ES + 0.002.

**"Inferred chance established (exploratory)"** = IC passes at both seeds.

**Falsification:**
- **The mechanism is falsified** if IC's P(move | no mob) is not below the ES arm's at both seeds. Removing the hedging
  incentive would then not suppress spurious reassignment.
- **The approach is falsified for this representation** if IC passes G4 but fails G1/N: the posterior's ~20% misses
  under-train movement.
- **Posterior collapse:** if the posterior's precision on DEV (scored against T*, diagnostic) falls below 0.9 during
  training, IC is reported as failing through self-reinforcing posterior error, not through the objective.

**Diagnostics (not gates):**
- posterior recall and precision during training (DEV, bright / dark);
- afterstate bias at static cells;
- P(move) by content;
- swap classes;
- the cross-terrain endpoint-terrain error (expected unchanged: a representation limit).

## Known limits stated before any run

- The posterior search is greedy and local. It misses ~20% of true moves in bright views, which can under-train
  movement.
- In darker views it is poor (nuisance), so false same-terrain swaps can teach the chance head terrain swaps.
- Whole-token transport still carries the wrong background on cross-terrain moves (5.2% of real moves). That is invisible
  to the static gate and is a representation limit no objective fixes.
- No claim on recursion, decisions, nuisance invariance, memory or efficiency.
- The fresh confirmation set stays sealed. Historical results and checkpoints are untouched.
