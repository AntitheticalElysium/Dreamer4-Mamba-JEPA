# Centering-window ablation: was it the window, or the horizon?

Status: **ready to run**. Implemented and tested; not yet trained.

## Question

The [stride-4 retrain](../20260915_frame_skip_retrain/README.md) moved TC's effective rank
5.68 → 16.73 and cut its full-latent variance ratio 5.10× → 2.30×. But it changed **three**
things at once: the centering timescale, the prediction horizon (1 → 4 native steps) and
the action conditioning (1 → 4 stacked actions). It cannot attribute the recovery.

This isolates the window. Dynamics stay one-step on consecutive frames with a single
outgoing action; only the index set entering SIGReg changes.

## Why it is genuinely window-only

Both arms **encode identical frames**. The window is the union of the prediction frames and
the centering frames, so at `centering_stride = 4`:

| | encoded native offsets | prediction indices | centering indices |
|---|---|---|---|
| `consecutive` (control) | 0,1,2,3,4,8,12 | 0,1,2,3 | **0,1,2,3** |
| `strided` (treatment) | 0,1,2,3,4,8,12 | 0,1,2,3 | **0,4,5,6** |

Seven frames, span 13, in both. That controls the two confounds that would otherwise ruin
the comparison:

- **Projector BatchNorm** sees flattened `B*T` samples, so an arm encoding 4 frames and one
  encoding 7 would differ in normalization as well as window. Identical encoded frames make
  the BN batch identical.
- **Sampling** eligibility depends on span, so differing spans would shift the start
  distribution. Identical spans make the eligible ranges identical.

Pinned by tests: same seed gives byte-identical `frames` and `actions` in both arms, and
with shared weights the **prediction loss is bit-identical** while only the regularization
term differs. That is the operational definition of window-only.

## Implementation

Two config fields, both digest-safe at their defaults under schema v1 (`recipe_dict` omits
them, so every existing checkpoint's `recipe_id` is untouched):

- `joint.centering_stride` — spacing of the centering set, independent of the prediction
  pairs. `> 1` requires schema v2 and `stride == 1`, so the ablation cannot be silently
  combined with the frame-skip recipe.
- `joint.centering` — `consecutive` or `strided`, selecting which index set SIGReg centers.

`window_layout()` returns the encoded offsets and both index sets, and is the single source
of truth for the sampler, the loss and the G1 screen. At the defaults it returns the four
consecutive frames of the v1 recipe.

**The pair contract is relaxed** to accept this: `run_joint_pair` previously required the
two arms to be `raw` and `tc` differing only in `variant`. It now requires a pair to differ
in exactly one declared axis — `variant` *or* `joint.centering` — so a TC/TC pair is legal
and a two-axis or identical pair is still refused. The arm directories remain `raw`/`tc`
for tooling compatibility; `pair_axis.json` records what each slot actually holds.

## Arms

| slot | recipe | variant | centering | digest |
|---|---|---|---|---|
| `raw/` | `lewm_mamba_tc_window_consecutive.json` | tc | consecutive | `58f31b403ab1…` |
| `tc/` | `lewm_mamba_tc_window_strided.json` | tc | strided | `26ceec5f30f1…` |

Both arms are TC. The directory names are an artifact of the pair keys, not the treatment —
read `pair_axis.json`.

## Readings

| outcome | conclusion |
|---|---|
| strided recovers rank toward the stride-4 figure | the window was the cause; horizon and action stacking were incidental |
| strided ≈ consecutive | the window was **not** the cause; the stride-4 recovery came from the 4-step horizon or the stacked actions |
| strided recovers rank but loses proxy retention | the trade TC already shows is intrinsic to centering timescale, not to our recipe |

The matched proxy comparison (TC-minus-Raw MLP AUC) is not available here, since both arms
are TC. Against Raw the references are the existing stride-1 and stride-4 runs.

## Run

```bash
TRITON_F32_DEFAULT=ieee .venv/bin/python -m d4mj paired-run \
  --dataset artifacts/craftax_support_v2 \
  --raw-recipe d4mj/recipes/lewm_mamba_tc_window_consecutive.json \
  --tc-recipe  d4mj/recipes/lewm_mamba_tc_window_strided.json \
  --out artifacts/lewm_gates_20260916/paired_window
```

Seven encoded frames rather than four, so expect ~1.75× the encoder cost per update against
the stride-4 run's ~0.5 s/update.

## Also built

`LeWMEncoder.export()` returns projected `z`, unprojected CLS and the 4×4 pooled patch grid
from one pass, so the observation and policy interfaces are first-class rather than captured
by a forward hook. It adds no training objective: patch tokens remain unregularized and
unpredicted, as in the paper. `projected_and_cls` is bitwise unchanged.

The [BC probe](../20260915_patch_token_policy/README.md) now takes `--checkpoint NAME=PATH`,
so both observation interfaces can be evaluated on any pair — including these arms once
trained.
