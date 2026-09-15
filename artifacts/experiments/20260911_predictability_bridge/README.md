# Predictability bridge: can the frozen world transition a spatial stream?

Status: **complete**, 2026-09-10. Verdict: the pooled spatial stream is geometrically
predictable, but its semantic gain over a persistence baseline is ~0.02 AUC. The ladder's
observation headroom does not survive the transition. Result in
[`evidence/bridge.json`](evidence/bridge.json).

## Question

The [feature ladder](../20260910_feature_ladder/README.md) showed LeWM's ViT patch grid
retains state the CLS/z export discards — Raw gains +0.103 successor AUC, TC +0.091.
That is a claim about **observation only**. The trained world predicts projected `z` and
cannot generate patch tokens, so patch retention and `z`'s transition-preservation
advantage are not yet combinable, and none of it reaches imagined rollouts.

This asks whether they *could* be, retraining nothing. For every root and candidate
action the frozen world already produces `h_next` (from consuming the completed pair)
and a generated `z_next`. Does either predict the observed successor's patch
representation — and does a predicted patch vector still carry the semantics that made
patch tokens attractive?

This is the test that decides whether a two-component state contract is viable at all.
It should run before any retraining or checkpoint ladder.

## Decision map

| Reading | Conclusion |
|---|---|
| Predictable from `h` alone | A lightweight spatial/persistent readout may be viable |
| Predictable only with generated `z` | The spatial stream rides on `z`, not on memory |
| Not predictable in either arm | Phase-1's objective must be retrained to carry the stream |
| Only Raw predictable | Stop TC; develop the Raw spatial export |
| Predictable but semantics fail | Readout/probe transfer, not missing dynamics |

## Protocol

Two stages, both fitted on TRAIN, evaluated on DEV.

**Stage 1 — geometry.** Fit a probe from the world state to the observed successor patch
vector. Conditions:

| Condition | Input |
|---|---|
| `h_only` | `[0, h_next]` |
| `joint_generated` | `[z_generated, h_next]` — the imagination-time state |
| `joint_observed_ceiling` | `[z_observed, h_next]` — handed the real successor encoding |
| `action_only_floor` | one-hot action; a Mamba state can re-encode the candidate action |
| `persistence_floor` | the root's own patch vector; predicting no change |

Total and action-effect R² are reported separately, because predicting a root's mean
successor is easy and predicting how the seventeen actions differ is the part
imagination needs. Uncertainty bootstraps roots.

**Stage 2 — semantics.** Read the *predicted* patch vector with the decoder fitted on
*observed* patch vectors — the M03 observed-to-generated contract, one decoder and one
TRAIN normalization for both. A prediction that is close in MSE but semantically empty
fails here. `observed_ceiling` reads the real observed patch through that same decoder.

Every condition is zero-filled into one 192+256 slot layout, so all enter a probe of
identical width and parameter count — the device `memory_view` uses for its z/h
ablations. Targets are `patch_pca192` and `patch_mean`.

## Target basis

Targets are TRAIN-whitened to unit per-coordinate variance, with numerically null
directions dropped and the retained count recorded. This is not cosmetic: raw PCA
components have per-coordinate TRAIN std spanning 0 to ~49, and the near-null tail meets
`_fit_probe_many`'s `clamp_min(1e-6)` in stage 2, amplifying pure noise by ~10⁶ — the
first smoke returned R² ≈ −1.7e12 from exactly that. Whitening also makes stage-1 R² an
average over coordinates rather than a report on the leading component. The ladder's own
basis is left untouched, so its recorded result is unaffected.

## Data identity

Source run `artifacts/lewm_gates_20260906/m03_bootstrap/evaluation_v2`. Roots, all-action
successor pixels and labels from its sealed sidecar; `h_next`, generated `z` and observed
`z` from its published `memory/features/{raw,tc}.primary_{train,dev}.pt`, at the
four-frame `c4` context the arms were trained on. Patch features are recomputed through
[`ladder._encode_rungs`](../20260910_feature_ladder/ladder.py), whose SHA256 is recorded
in the report — the bridge depends on that file, so its identity is pinned.

## Outcome

Stage 2's ceiling is informative at full scale: `patch_pca192` observed ceiling reaches
semantic AUC 0.799 / R² +0.345 (raw, MLP), matching the ladder's 0.805 / 0.320. The
64-root smoke's negative ceiling was sample size, as expected.

MLP, DEV. Geometry is R² against the observed successor patch vector; semantics is that
vector read by the decoder fitted on observed patch vectors.

| arm | condition | total R² | effect R² | sem AUC | sem R² |
|---|---|---:|---:|---:|---:|
| raw | h_only | +0.728 | **+0.705** | 0.695 | +0.112 |
| raw | joint_generated | +0.701 | +0.717 | 0.692 | +0.101 |
| raw | joint_observed (ceiling) | +0.880 | +0.874 | 0.727 | +0.154 |
| raw | persistence floor | **+0.762** | +0.000 | **0.671** | +0.138 |
| raw | action-only floor | −0.002 | +0.147 | 0.527 | −1.653 |
| raw | observed patch (stage-2 ceiling) | — | — | 0.731 | +0.202 |
| tc | h_only | +0.539 | +0.707 | 0.605 | −0.170 |
| tc | joint_generated | +0.543 | +0.704 | 0.596 | −0.173 |
| tc | joint_observed (ceiling) | +0.800 | +0.820 | 0.652 | −0.005 |
| tc | persistence floor | +0.664 | +0.000 | 0.646 | +0.079 |
| tc | action-only floor | +0.030 | +0.356 | 0.529 | −1.712 |
| tc | observed patch (stage-2 ceiling) | — | — | 0.713 | +0.201 |

**1. The action-effect component is genuinely predictable — from `h` alone.** Effect R²
0.705 (raw) and 0.707 (tc) against action-only floors of 0.147 and 0.356. `h_only` and
`joint_generated` are indistinguishable throughout, so generated `z` adds nothing the
Mamba state does not already carry. This is the "predictable from `h`" branch, and it
rules out "predictable only with generated `z`".

**2. But the semantic payoff over persistence is ~0.02 AUC.** The persistence floor —
predicting that the patch vector does not change — reaches 0.671 semantic AUC against
`h_only`'s 0.695 and an observed ceiling of 0.731. Persistence also beats `h_only` on
*total* R² (0.762 vs 0.728), because total variance is dominated by root identity, which
persistence gets for free; its effect R² is 0.000 by construction. The disaggregation is
what makes this readable: the world predicts *how the successor differs by action*, and
that skill buys almost nothing once the prediction is decoded semantically.

For scale: the ladder showed patch tokens carry **+0.103** successor AUC over `z` in
observation. Through the transition that headroom becomes **+0.024** over persistence.

**3. Fine spatial detail is not predictable at all.** `patch_pca192` gives effect R²
−0.014 (raw) and −0.145 (tc) — at or below zero. The world predicts the pooled,
low-frequency summary (`patch_mean`) and not the whitened spatial detail. Whitening
deliberately equalises all 192 PCA directions, which upweights exactly the low-variance,
high-frequency directions; this is therefore a harsh target and the contrast between the
two targets should be read as pooled-versus-detailed, not as a single verdict.

**4. Raw is ahead of TC on every semantic row.** `h_only` 0.695 vs 0.605, ceiling 0.731
vs 0.713, and TC's continuous R² is negative wherever Raw's is positive. TC's higher
action-only floor (0.356 vs 0.147) is consistent with the larger action share M03 and the
ladder both measured in its latent.

### Reading

Closest to the "predictable but semantics fail" branch. A two-component state contract is
**not clearly viable as-is**: the stream that is predictable is the pooled summary, and
predicting it is worth ~0.02 AUC over assuming it does not change. Adding a spatial
readout to the current frozen world would not recover the ladder's observation headroom.

**Scope correction (2026-09-15).** This does not bear on TC-LeWM's own patch pathway.
The paper freezes the encoder for policy learning and never regularizes or predicts patch
tokens — its policy reads *observed* tokens. So a negative result here forecloses bolting
an *imagined* spatial stream onto the trained world; it says nothing about whether a
policy benefits from observed patch tokens. That is tested separately in
[patch_token_policy](../20260915_patch_token_policy/README.md).

That is a result about *this* frozen world, not about the design. Phase-1 never trained
anything to carry a spatial stream, so the honest next question is whether an objective
that supervises one produces a transitionable stream — not whether this one already does.

## What this cannot establish

One-step transition and retention, never control. A predictable patch stream is not a
policy, not a rollout, and not grounds to authorize M4. `c4` only: the memory panel
already showed context interventions are flat, so longer prefixes are not expected to
change this, but they are not tested here. One training seed, one panel, and the M03
coverage gap is unchanged — sparse binary targets will report `insufficient_coverage`
rather than a value.

The stage-2 continuous half came out positive at the ceiling (+0.345 raw, +0.201 tc), so
both halves are supported. Linear-family continuous R² is negative throughout, as in the
ladder and M03; only the fixed MLP recovers these scalars.

## Run

```bash
TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu .venv/bin/python \
  artifacts/experiments/20260911_predictability_bridge/bridge.py --device cuda
```

Output is immutable: `evidence/bridge.json` is never overwritten. `--limit N` writes
`bridge.smoke.json` and stamps `structural_smoke_not_a_result`.
