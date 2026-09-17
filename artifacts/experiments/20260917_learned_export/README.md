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

mlp probes, successor-state targets. `*_pca` are **unsupervised** -- no labels touch them --
and bound how much of the learned gain is supervision. Learned exports are the mean over
three seeds, with half-spread:

| arm | export | binary AUC | continuous R² |
|---|---|---|---|
| consecutive | current | 0.6183 | −0.086 |
| consecutive | cls_pca *(unsup.)* | 0.6001 | unreliable |
| consecutive | **patch_pca** *(unsup.)* | **0.7173** | −0.177 |
| consecutive | cls_learned | 0.6443 ±0.0024 | −0.041 ±0.004 |
| consecutive | **patch_learned** | **0.7631 ±0.0054** | **+0.152 ±0.015** |
| strided | current | 0.6266 | −0.277 |
| strided | cls_pca *(unsup.)* | 0.5651 | unreliable |
| strided | patch_pca *(unsup.)* | 0.6377 | −0.796 |
| strided | cls_learned | 0.6379 ±0.0049 | +0.007 ±0.014 |
| strided | **patch_learned** | **0.7204 ±0.0055** | **+0.105 ±0.010** |
| *Direct, current-state+action* | *(comparable baseline)* | *0.7571* | — |

Seed spread is ±0.005 AUC, so the gains are stable and were not an artifact of the
unseeded initialization the audit found.

**192 dimensions are enough.** At identical width the patch route reaches 0.763 against the
current export's 0.618. Whatever is missing is not capacity.

**It is not merely supervision.** On the consecutive arm the *unsupervised* patch PCA alone
reaches 0.7173, +0.099 over the current export with no labels at all. The information is
genuinely present in the patch tokens and linearly accessible.

**But the two arms differ sharply, and that is new.** On the strided arm `patch_pca` gains
only +0.011 (0.6377 against 0.6266), so most of that arm's learned gain is supervision. The
widened centering window appears to have made the patch information **less linearly
accessible**, which no earlier panel showed.

**Relearning from CLS barely helps, and an unsupervised CLS basis is worse than the current
export** (0.600 and 0.565 against 0.618 and 0.627). So the joint-trained projector is not
badly fitted -- CLS is the lossy step, and the projector is making reasonable use of what
reaches it.

**It matches Direct rather than beating it.** The right comparison is Direct's
current-state-plus-action score, **0.7571** -- not the 0.8476 quoted in an earlier revision
of this file, which is Direct decoding the *actual observed successor* and so answers a
different question. The consecutive patch export reaches 0.763 against that 0.757: level,
with extra task supervision, not ahead.

### A measurement caveat on the PCA controls

`cls_pca`'s continuous R² is **not trustworthy** and is omitted above. `_fit_probe_many`
standardizes every input dimension with a `clamp_min`, which stretches a near-zero principal
direction to unit variance and injects noise; truncating the null tail (187 and 164 of 192
components retained) moved R² from −164 to −9.4 but did not repair it. The binary AUC, which
carries the argument, is unaffected. `patch_pca` compresses 3072 dimensions into 192 and has
no comparable null tail.

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
