# Predictability bridge: can the frozen world transition a spatial stream?

Status: ready to run. Structural smoke passed on CUDA at 12 and 64 roots; no result recorded.

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

## What this cannot establish

One-step transition and retention, never control. A predictable patch stream is not a
policy, not a rollout, and not grounds to authorize M4. `c4` only: the memory panel
already showed context interventions are flat, so longer prefixes are not expected to
change this, but they are not tested here. One training seed, one panel, and the M03
coverage gap is unchanged — sparse binary targets will report `insufficient_coverage`
rather than a value.

Stage 2's continuous half may be weak even at the ceiling: the ladder's equivalent
decoder reached R² +0.320 at 256 TRAIN roots, but the 64-root smoke ceiling was negative.
If the full run's `observed_ceiling` continuous R² is not positive, read the binary half
only and treat the continuous half as unsupported.

## Run

```bash
TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu .venv/bin/python \
  artifacts/experiments/20260911_predictability_bridge/bridge.py --device cuda
```

Output is immutable: `evidence/bridge.json` is never overwritten. `--limit N` writes
`bridge.smoke.json` and stamps `structural_smoke_not_a_result`.
