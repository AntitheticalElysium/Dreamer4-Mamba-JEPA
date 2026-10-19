# Transport vs free synthesis: matched two-arm preregistration

2026-10-15. Written before any Stage 3 optimization and before the F1b fitting control was evaluated. Governs
`transport_model.py`, `transport_train.py` and `transport_eval.py`. Authorized by the user's 10-15 mandate: one bounded,
preregistered, matched comparison, conditional on feasibility. No H16, actor, reconstruction or simulator labels in
training. No third arm, extension or hyperparameter search.

## Question

Does stochastic reassignment of retained content provide a better learning mechanism for coherent transitions than
independently generating stochastic content?

## Prerequisite (F1b, `transport_stage2.py fit|fiteval`, rule fixed before its evaluation)

Stage 2 F1a showed that swapping *true* afterstate tokens expresses real moves: 100% of strict move keys and 99.7% of
panel moves, under both readers. The available references, however, are post-chance means, not afterstates.

F1b continues the D parent for 1,500 updates with the parent's own D loss, through LN(T*·b) with the TRUE assignment
supplied. This is diagnostic only. A matched identity continuation runs alongside it.

**Stage 3 launches only if** the F1b-assigned reference, at 1,500 updates and under BOTH readers, reaches all of:
- ≥ 90% correct strict move keys;
- ≥ 90% correct strict stay keys;
- ≥ 90% panel moves with both endpoints correct.

Otherwise the failure is diagnosed, recorded and the study stops. The identity continuation is reported for attribution
only. It cannot block or rescue the launch.

## Arms (identical except the stochastic head)

**Shared by both arms:**
- the D parent's trainable backbone and deterministic reference b (`PairedWorld.reference`, pre-LN);
- the noise trunk: `direct_model.Completion`, with 4 local + 4 shared Gaussian dimensions, spatial attention and FF;
  identical initialization (seed 7, asserted);
- identity carrier and renderer: tile content is its afterstate token. F1a showed this expresses moves at reader level;
  a learned renderer would add a context-reconstruction path;
- the training ledger: uniform eligible windows, seed 20261014;
- noise seed 2026101401. Two independent draws per update: Gaussian, then Gumbel from the same generator in both arms;
- the first-power energy objective on the 35 scored tokens (`direct_model.losses`);
- phase AdamW, LR schedule, grad clip 1, batch 16, L16, **3,000 updates**;
- weight-decay groups asserted identical by parameter name on every shared tensor. This fixes the earlier mismatch:
  fresh construction plus `load_state_dict` keeps the Mamba no-decay flags in both arms.

**The two heads:**
- **free:** ẑ_j = LN(b_j + G(e_j)). G is Linear(256 → 192), weight std 0.001, bias 0.
- **operator:** ẑ = LN(T·b).
  - Proposals: P is Linear(256 → 5), weight std 0.001, bias [β, 0, 0, 0, 0], with β = ln(4·62) for about one proposed
    move per frame at initialization (no labels). Each map tile proposes stay / up / down / left / right by Gumbel-max.
  - Matching: a parallel greedy matching accepts a proposal when its perturbed margin beats every live proposal touching
    either cell. T is an exact permutation of disjoint adjacent swaps; the HUD and the player never move.
  - Gradients: the reference takes its gradient through the exact permutation. The proposals use straight-through
    Gumbel-softmax (τ = 1) through the linear swap relaxation. The same hard operator is used in training and
    evaluation.
  - Unit checks: `transport_model_test.py`.

**Known, documented differences:**
- completion-head parameters: G has 49,344, P has 1,285;
- the operator alone uses the Gumbel draws and a straight-through estimator;
- runtime.

The operator cannot express births, deaths, border crossings or arrows (Stage 1: about 3% of terrestrial mob changes
in the scored interior). Those stay deterministic. No stochastic creation path is added.

## Evaluation

- **Checkpoints:** 3,000 updates is primary. 0 / 500 / 1,500 are descriptive only and select nothing.
- **Panel:** 104 bright DEV roots × 64 samples, batch 4, sampling seed 2026101402 + start.
- **Physics:** on the 14 strict sources (8 cow) of audit S5, under the real ridge and bank readers.
- **Frozen evaluator:** `direct_coherence.analyze` also runs, for static corruption and history.
- **Intervals:** 2,000 root-cluster bootstrap draws, seed 20261015, on operator − free.

**Gates. All of them must hold under BOTH readers for "transport better (exploratory)":**

| gate | condition (operator arm unless stated) |
|---|---|
| G1 movement is learned, not stay-only | all-source valid movement ≥ 15%; cow valid movement ≥ 20.5% (half the real 41.02%); distinct valid outcomes per source ≥ 2.5 |
| G2 coherence | valid ≥ 90%; duplicate ≤ 5%; absent ≤ 5% |
| G3 comparison | operator − free lower 95% bound > 0 for BOTH valid movement (all sources) AND valid |
| G4 static | scored static corruption ≤ max(free's, 0.5%) |
| G5 joint law | scored joint occupancy ES ≤ free's + 0.002 |

**Fixed readings:**
- G2 without G1 is **"coherent but static"**, a failure. Fewer duplicates paid for by absence, a lower energy score,
  or sharper proposals alone do not count.
- If G3 fails while both arms fail G1, the verdict is **"neither learns movement"**.
- The reference's own validity (T = I) is reported. A pass that depends on reference drift is flagged.

**Diagnostics (not gates):**
- swaps per frame, and the share of swaps touching a decoded mob;
- proposal probabilities at mob vs non-mob tiles;
- the 500 / 1,500 trajectory;
- a numerical payload-substitution check. With an identity carrier, provenance holds by construction: the destination
  receives exactly the source's token.

**Confirmation:** reserved fresh roots (base seed 2026101400) stay sealed unless every gate passes. A pass is one seed
on an exploratory panel. Training-run variance is unmeasured; numerical divergence alone gave a 2.1pp spread between two
direct runs.

## Stop rules

- Fixed budget: no extension, no rerun with changed settings, no inference-only substitute.
- A failed gate is diagnosed with inexpensive frozen-model probes and recorded. The mechanism is not repaired within this
  study.
- A numerical or execution failure is fixed only if it is a bug. It is recorded either way.
