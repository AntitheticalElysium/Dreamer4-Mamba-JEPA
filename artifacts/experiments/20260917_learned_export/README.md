# Is 192-D too few, or did joint training choose a poor mapping into it?

Status: **complete**, 2026-09-17. Positive control over frozen checkpoints; no world model
trained, no gate artifact changed, nothing authorized. Runner:
[`learned_export.py`](learned_export.py), evidence in [`evidence/export.json`](evidence/export.json).

## Why

The [readout refit](../20260917_generated_readout/README.md) separated two failures. Outcome
decoding from generated latents was a decoder artifact and recovers when refitted. Successor
**state** decoding does not: refitting rescues Direct and does nothing for either TC arm, and
the gap is already present on *observed* successor `z` -- 0.658 and 0.681 against Direct's
0.848. So the weakness is upstream of any decoder, and the diagnostics cannot say whether
192 dimensions are simply too few or the mapping into them is poor.

## Design

The ViT stays **frozen**. Every export is exactly **192-D**, so width is held constant and
only the mapping varies. Each learned export is trained on TRAIN roots to predict successor
labels from the root plus the action, then **frozen**, after which the standard M03 probe and
metrics run on it unchanged -- so these numbers sit on the same scale as the readout table.

| export | mapping |
|---|---|
| `current` | the joint-trained projection `z`, frozen |
| `cls_learned` | trained 192 → 192 from the frozen CLS |
| `patch_learned` | trained 16×192 → 192 from the frozen 4×4 pooled grid |

## Result: the width is not the bottleneck; CLS is

mlp probes, successor-state targets:

| arm | export | binary AUC | continuous R² |
|---|---|---|---|
| consecutive | current | 0.6183 | −0.086 |
| consecutive | cls_learned | 0.6411 | −0.016 |
| consecutive | **patch_learned** | **0.7635** | **+0.203** |
| strided | current | 0.6266 | −0.277 |
| strided | cls_learned | 0.6393 | +0.006 |
| strided | **patch_learned** | **0.7261** | **+0.133** |
| *Direct-Mamba (reference)* | *its own export* | *0.8476* | *+0.075* |
| *Direct-Attention (reference)* | *its own export* | *0.8476* | *+0.089* |

Three things follow.

**192 dimensions are enough.** At identical width, a learned patch summary reaches 0.7635
against the current export's 0.6183, and turns a negative continuous R² into +0.203. Whatever
is missing is not capacity in the export.

**Relearning from CLS barely helps** -- 0.6183 → 0.6411, and 0.6266 → 0.6393. So joint
training did not merely pick a poor map *out of CLS*; CLS is itself the lossy step.

**The information is in the patch tokens.** `patch_learned` beats `cls_learned` by +0.12
binary AUC in both arms, and on continuous state it exceeds both Direct anchors. The current
export cannot reach it by construction: `z` is the projection of CLS, and the patch tokens
are never regularized by SIGReg nor predicted by the world.

## What it does not establish

This is an upper bound, and deliberately so. The learned exports hold extra trainable
parameters in the mapping and are trained **supervised on the evaluation task's label family**
before being frozen; DEV roots are held out throughout, but nothing here shows a SIGReg or
JEPA objective would *find* this mapping without label supervision. It is evidence that the
interface can carry the information, not that joint training can be made to put it there.

It also does not make patch tokens "the answer". It localises the loss to the CLS bottleneck
and rules out export width, which is what a control is for. Whether richer **inputs**, richer
**prediction targets**, or both are what matter is the next question, and is not answered
here -- that needs the four matched one-step arms (`z→z`, `u→z`, `z→u`, `u→u`), which require
retraining and have not been run.

Single dataset, single split, 256 TRAIN and 128 DEV roots over 17 actions, one seed per
export. M03 status is unchanged and `m4_authorized` remains false.
