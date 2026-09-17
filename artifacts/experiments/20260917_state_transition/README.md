# Can the world preserve and predict a richer 192-D state?

Status: **complete**, 2026-09-17. Frozen encoder, four matched one-step arms, diagnostic —
not a gate run and not an architecture decision. Runner:
[`state_transition.py`](state_transition.py), evidence in
[`evidence/evaluation.json`](evidence/evaluation.json) and
[`evidence/training.json`](evidence/training.json).

## Setup

`u` is the **fixed, TRAIN-fitted, label-free PCA** of the frozen 4×4 pooled patch grid
(20,000 samples, full rank 192 retained); its mean and basis are persisted, so every arm and
every later re-scoring shares one map. `z` is the joint-trained export. Both are 192-D.

The encoder is forwarded **once** over a fixed pool of 25,600 windows (119.6 s) and all four
arms train off the cached tensors. Same Mamba size, same initialization seed, same batch
order, same 4,000-step budget everywhere. Windows are sliced to the prediction frames via
`window_layout`, since the consecutive recipe encodes seven and rolls out four.

Scored on the M03 roots against successor-state labels, with **native TRAIN-fitted** readouts
alongside **observed-to-generated transfer**, on binary and continuous targets.

## Result: the world can carry the richer state, and needs both ends

mlp probes; action-only floor 0.5739:

| arm | generated, native | generated, transfer | hidden state | gen. R² | hidden R² |
|---|---|---|---|---|---|
| `z→z` *(baseline)* | 0.6476 | 0.6303 | 0.6584 | −0.026 | −0.142 |
| `u→z` *(richer input)* | 0.6392 | 0.6332 | **0.7002** | +0.022 | **+0.127** |
| `z→u` *(richer target)* | 0.6538 | 0.6305 | 0.6718 | −0.172 | −0.081 |
| **`u→u` *(both)*** | **0.7050** | **0.6920** | **0.7460** | +0.022 | **+0.168** |

Linear probes agree in ordering: `u→u` leads at 0.7002 generated and 0.7378 hidden against the
baseline's 0.6487 and 0.6638.

**`u→u` is best on every measure.** The richer state is transitionable: one step of prediction
reaches 0.7050, against 0.7173 for reading the *root* `u` directly in the export control. The
world preserves most of what the representation carries rather than discarding it.

**Both ends are needed, and they do different things.** On the hidden state, richer input alone
buys +0.042 (0.6584 → 0.7002) and richer target alone +0.013 (→ 0.6718), while both together
buy +0.088 — more than the sum, so they are not independent contributions. On the *generated*
readout, richer input alone buys nothing (0.6392 against 0.6476); only the pair moves it.

**The continuous targets follow the input.** Every `u`-input arm turns the baseline's negative
R² positive on the hidden state (−0.142 → +0.127 and +0.168), while `z→u` stays negative. This
is the one place the earlier export control looked bad for `u` — its root-level continuous R²
*worsened* — and the transition reverses that.

**The transfer gap is small here.** Native minus observed-fit transfer is 0.013 to 0.017 across
arms, against the 0.12–0.15 gap that the M03 decoder artifact produced. Freshly trained arms
read on their own distribution do not reproduce it, which is consistent with that artifact
being a property of the fitted decoder rather than of generated latents.

## What this does not establish

Training loss is **not** comparable across arms with different targets: `z→u` at 0.994 and
`z→z` at 0.315 differ in target scale, not quality. Only `z→z` against `u→z` (0.315 / 0.328)
and `z→u` against `u→u` (0.994 / 0.322) share a target.

This is a 4,000-step diagnostic over a fixed 25,600-window pool, one seed per arm, on the
consecutive checkpoint only — **not** the sealed 10,000-update recipe, and not comparable to
the gate runs. The strided arm was deliberately excluded: its patch information is not
linearly accessible (unsupervised PCA gained only +0.011 there), so it is a poor host.

It does **not** show that a JEPA/SIGReg objective would learn `u`, only that a world model can
transition it once it exists. It does not test closed-loop multi-step rollout. And it does not
adjudicate the alternative explanations for the original weakness — predictor capacity, the
paper's per-layer AdaLN action conditioning, optimization, or the objective — all of which
remain untested, and any of which could improve the `z` route during joint training.
