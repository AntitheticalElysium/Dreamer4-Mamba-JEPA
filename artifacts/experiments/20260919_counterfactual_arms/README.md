# Counterfactual supervision arms — 2026-09-19

Tests whether LeWM's generated states fail the gate because the world was never trained to
distinguish different actions from the **same** state. Direct was; LeWM was not. Direct also
trains on a fork corpus LeWM has never seen, so "new data" and "sibling contrast" are entangled
and a two-arm test cannot separate them.

**Result: NULL, and trending negative. Counterfactual supervision with MSE does not close the gap.**

## Design

Three arms continue from the same Raw-10k world for 10,000 updates, encoder **frozen**:

| arm | extra targets/update | spread over |
|---|---:|---|
| A control | 0 | — |
| A′ data | 68 | **68 independent** fork roots (factual action only) |
| B sibling | 68 | **4** fork roots × 17 siblings |

A′ and B receive an identical extra-target budget and differ only in whether those targets are
independent roots or siblings of one root. Loss `0.8·factual + 0.2·extra`, per-root averaged.

Pinned identical: frozen encoder and pre-encoded latents (40,960 factual windows from the gate's
own sampler; 15,016 fork roots at seeds 15000–16504, leakage-asserted against the sealed
13000–14511 range), world init, optimizer, LR, schedule, updates, RNG, factual batches, MSE,
context. The root+action control is computed from the frozen encoder, so it is **the same bar for
every arm**. Predictor-projector BatchNorm statistics are frozen in all three arms, so branch
batches cannot move them in A′/B but not in A.

## Result

100-root historical confirmation panel, rank readout, three probe seeds:

| arm | generated | control | paired diff | frozen-head AUC cost |
|---|---:|---:|---|---:|
| A control | **53.0** | 55.67 | −0.03 [−0.129, +0.065] | 0.161 |
| A′ data | 50.0 | 55.67 | −0.05 [−0.147, +0.040] | 0.147 |
| **B sibling** | **48.67** | 55.67 | −0.08 [−0.168, +0.010] | **0.114** |

DEV-36: 16.67 / 15.0 / 16.0 against a control of 19.0.

- **No arm beats its control**, and none approaches the predeclared +10 threshold.
- Ordering is A > A′ > B: adding the fork corpus slightly hurt, sibling structure slightly more.
- **Factual held-out MSE is flat** (0.01565 / 0.01546 / 0.01570) — nothing was traded away.
- Both A′ and B fit their extra targets equally (final extra loss 0.0130 vs 0.0129), so the null
  is not one arm failing to learn.
- **Action sensitivity went *down*** — frozen-head derangement cost 0.161 → 0.147 → 0.114.
  Training explicitly on 17 branches made generated states *less* action-discriminative.

## Mechanism — suggested, not established

Averaging MSE over 17 branch targets may reward predicting their centroid:

| | A control | A′ data | B sibling |
|---|---:|---:|---:|
| branch spread / ‖z‖ | 0.0822 | 0.0778 | 0.0765 |
| distance to branch mean | 0.850 | 0.808 | 0.810 |

Directionally consistent but **modest** — spread contracted ~7% while the action-AUC cost fell
~29%. Suggestive only.

## Where the improvement went — the allocation test

Effect fidelity improved a lot (R² .518 → .780) while every semantic AUC stayed flat. Two
explanations survive that, and they imply different next experiments:

- **ALLOCATION** — the semantic directions occupy a tiny share of δz variance, so coordinate-uniform
  MSE spends its capacity elsewhere. The gain landed *outside* the semantic subspace, and no
  exposure regime fixes it.
- **DOWNSTREAM** — the gain landed inside the semantic subspace too, and the failure is somewhere
  after δz geometry entirely.

`allocation.py` separates them. Fit one linear direction per outcome on the **real** effect
(`successor z − root z`) on TRAIN, take their span as the semantic subspace S, and decompose each
arm's *predicted* effect on DEV into S and its complement.

**Control first** — a subspace that predicted nothing would also show a world failing inside it,
which would make the whole decomposition unfalsifiable. On held-out real effects those directions
are genuinely predictive:

| direction | held-out AUC | positives / 2176 |
|---|---:|---:|
| death | **0.881** | 986 |
| damage | **0.876** | 1011 |
| reward_positive | **0.843** | 83 |
| achievement_event | **0.820** | 31 |
| inventory_changed | 0.691 | 103 |
| tile_changed | 0.693 | 91 |

*(An earlier control run reported nonsense — death AUC 0.119. That was my bug, not a finding:
`semantic_directions` returns a QR-orthonormalized basis whose column i is a Gram-Schmidt residual,
not the i-th outcome's direction. The per-outcome control must use the raw directions. The R²
decomposition below is unaffected — projection onto a span is basis-invariant.)*

**Variance share.** Those six directions hold **0.179%** of true δz variance. Isotropic expectation
for six of 192 dimensions is 3.125%. The semantic subspace is **~17× under-represented** in the
quantity MSE optimizes.

**Decomposition.**

| arm | R²_total | R²_inside S | R²_outside S |
|---|---:|---:|---:|
| A_control | 0.064 | **−0.579** | 0.065 |
| Ap_data | 0.138 | **−0.050** | 0.139 |
| B_sibling | 0.166 | **−0.091** | 0.167 |
| **B − A** | **+0.103** | (+0.488, still negative) | **+0.102** |

Every point of improvement landed outside S: `ΔR²_out` +0.102 against `ΔR²_total` +0.103. And in S
*no arm beats a constant predictor* — all three R² are **negative** after 10k updates, including
the two trained directly on counterfactual branch targets.

**ALLOCATION is confirmed.** MSE over 192 equally-weighted coordinates spends ~99.8% of its
capacity on directions that carry no outcome semantics, and the 0.18% that does carry them is
never fit at all. This is the mechanism behind the flat semantic AUCs — not a data deficit, not a
supervision-structure deficit.

## What this rules out, and what it points at

**Rules out:** "just add counterfactual data and supervise it with MSE." Both the data (A′) and the
sibling structure (B) were tested at matched budget; neither helped, and the allocation test says
why — their gains could not have reached the semantic directions.

**Rules out (newly):** a from-scratch paired retrain under the same MSE objective. It would buy more
of the same out-of-subspace fidelity. The user's conditional was to proceed to from-scratch *if the
allocation result was negative*; it is positive, so from-scratch is **not** launched.

**Points at:** the **loss form**. Any objective that reweights toward the directions that carry
outcomes — an action-centred / delta / contrastive loss, a whitened or semantically-weighted target
metric — is now the indicated next move, with a concrete target to aim at: R²_inside S, currently
negative for every arm.

**TC was not run.** The instruction was to extend to TC only if Raw came back positive. It did not.
TC also still carries its independent CLS→z export defect, which this treatment does not address.

## Limits

One model seed per arm (three probe seeds); 10k continuation updates rather than a fresh joint
retrain; one hyperparameter point (mass 0.2, 4 fork roots/update); frozen encoder throughout.
Intervals are wide — "null" here means **no detectable improvement**, not proven equivalence.

## Reproduction

```
python encode_pool.py      # frozen-encoder factual + fork pools
python train_arms.py --arm {A_control,Ap_data,B_sibling}
python evaluate_arms.py    # evidence/arms_eval.json, evidence/verdict.json
python decompose.py        # evidence/decompose.json -- effect fidelity, 20-draw derangement, semantics
python allocation.py       # evidence/allocation.json -- control + in/out-of-subspace decomposition
```
