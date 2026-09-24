# Matched 10k `z→z` vs `u→u` through the M03 suite

Status: **complete**, 2026-09-17. Primary panels only (`--skip-history`). Runner:
[`matched_10k.py`](matched_10k.py); worlds and cache in [`evidence/`](evidence/); suite output
at `artifacts/lewm_gates_20260918/m03_matched/`.

**`u` moves the state rows substantially and the capability rows not at all.** That was the
4k reading on a one-step panel; it holds at full budget on the gate's own rows.

## Setup

Both worlds train to 10,000 updates on the **same** cached 25,600 windows, initialization
seed, batch order, history length (3 action-pairs) and unchanged MSE loss — the cache is
symlinked from the 4k run and loaded rather than rebuilt, so the inputs are byte-identical.
`u` is the fixed, TRAIN-fitted, label-free PCA of the frozen pooled patch grid.

### The adapter

`u` cannot be expressed as a LeWM checkpoint: `z` is the projection of CLS while `u` comes
off the pooled patch grid, so no encoder state dict represents it, and pointing the gate at
the encoder checkpoint would silently evaluate the original export. The adapter is an
**encoder shim** presenting the ordinary interface — `projected_and_cls` and `__call__` —
while emitting `u`. Every gate encode path (`_encode_lewm`, `encode_memory`) then runs
unchanged; nothing about the evaluation is reimplemented for this arm. `load_m03_bundle` is
patched to build (frozen 10k encoder [+shim], fresh 10k world) from a *world* file, and the
gate keys its feature cache on `_sha256(checkpoint_path)`, so the two arms never share cache
entries.

Verified in the output: `raw_cls` and `tc_cls` are both **0.7517**, identical — same frozen
encoder, CLS untouched, only the export swapped — and that figure matches the original gate's
TC observed static of 0.752.

## State rows: `u` is a large, real improvement

mlp, mean binary AUC:

| panel | `z→z` | `u→u` | Direct-M |
|---|---|---|---|
| static, observed | 0.7433 | **0.7824** | 0.8484 |
| successor state, observed | 0.6580 | **0.7826** | 0.8476 |
| successor state, generated | 0.6201 | **0.7077** | 0.7518 |
| memory `[z,h]`, generated successor | 0.6200 | **0.7440** | — |

On the gate's generated successor-state row, `u→u` reaches **0.708** against Direct's 0.752,
and sits above the original sealed gate's Raw (0.682) and TC (0.626). Through the memory
supplement the joint `[z,h]` readout reaches **0.744**, above the original Raw's 0.691 and TC's
0.616. Every `u` component improves: `z`-slot 0.599 → 0.695, `h` 0.631 → 0.736.

## Capability rows: unmoved

| row | `z→z` | `u→u` | Direct-M |
|---|---|---|---|
| coarse outcomes, generated | 0.6407 | 0.6443 | 0.5686 |
| *action-only floor* | *0.6825* | *0.6825* | *0.6825* |
| fatal-safe ranking | **0.3333** | **0.3333** | **0.8611** |
| memory outcomes `[z,h]`, generated | 0.629 | 0.632 | — |
| *memory action-only floor* | *0.6626* | *0.6626* | *0.6626* |

The two arms land on **exactly the same** fatal-safe rate, 0.3333, intervals containing
uniform. Neither clears the action-only floor on coarse outcomes, in the primary panel or
through memory. Direct stays at 0.8611 on ranking.

In the memory panel `u`'s outcome readout is if anything slightly *worse* than `z`'s on `h`
(0.683 against 0.708), while its successor-state readout is far better. The split between the
two target families is clean.

## Reading

The observed-state deficit is real, and `u` fixes a large part of it — measured here on the
gate's own rows rather than a substitute metric, at full budget, with everything matched. It
does not propagate to outcomes or to action choice. Whatever limits those rows is downstream
of the state representation and is not addressed by improving it.

## What this is not

**Not a sealed gate result.** The patched loader bypasses the checkpoint identity, recipe and
source checks that make a run sealed. `suite_complete` is recorded as `false`,
`m03_capability` is `measured_pending_capability_review`, and `m4_authorized` remains false.

**Not a Raw-versus-TC comparison.** Both arms share one TC-consecutive encoder
(`paired_window/raw` is `variant: tc`) and their worlds train on MSE alone.

**Historical panels were not run.** `exact961`, `policy104`, `hazard5402` and `legacy751` are
absent by choice, and those are where the original weakness was shown to replicate. One seed,
frozen encoder, and coverage is unchanged at 13/25 static and 18/25 successor binary targets —
no amount of training addresses missing labels.
