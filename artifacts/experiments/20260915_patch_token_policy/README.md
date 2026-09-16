# Patch-token BC: does token-preserving cross-attention beat pooled export?

Status: **complete**, 2026-09-16. Verdict: token-preserving cross-attention beats pooled
export decisively in both arms, and TC beats Raw at every condition. Result in
[`evidence/policy.json`](evidence/policy.json).

## Question

TC-LeWM's downstream policy keeps, per camera, the CLS token **and a 4×4 spatially
pooled grid of patch tokens**, and lets action queries cross-attend to that sequence
(paper Table 4: "34 visual tokens via cross-attention + task embedding"). §A.1: *"Patch
tokens are not directly regularized by SIGReg but are retained for downstream policy
learning."*

Our export is projected CLS only, and [`agent.py`](../../../d4mj/agent.py) mean-pools
before an MLP. Neither the [ladder](../20260910_feature_ladder/README.md) (which
flattened the grid to one vector) nor the [bridge](../20260911_predictability_bridge/README.md)
(which asked the world to *transition* the grid — something the paper never does) tested
that pathway.

## Why this is not a retrain

**The paper never trains on patch tokens.** The encoder is frozen before policy learning,
and patch tokens are neither SIGReg-regularized nor predicted by the world model. A
faithful patch implementation is therefore an *export and policy-interface* change, not a
world-model change — so the pathway can be tested on the existing frozen checkpoints at
full fidelity. Frame skip is the part that genuinely needs a retrain; it is specified
below and not implemented here.

## This is real behaviour cloning

`support_v2` carries no BC-eligible episodes (0 of 10,080 — its rollouts are ε-noisy).
`craftax_expert_v1` does: **320 PPO-expert episodes, 696,746 transitions, 20.6/22 mean
achievements.** [`expert.load_archive`](../../../d4mj/expert.py) converts it exactly —
the 64×64 frames are Craftax's native 63×63 plus a zero-padded row and column, so the
crop is lossless and the CHW→HWC permute is a view (spec S49). Verified: the discarded
row and column contain only zeros. Splits are whole-episode 80/10/10 via `episode_splits`.

## Conditions

One parameter-identical head; only the sequence length changes. Tokens are zero-padded
to width 192 so even the input projection is shared.

| Condition | Tokens | What it isolates |
|---|---:|---|
| `z` | 1 | today's export |
| `cls` | 1 | the projector's cost |
| `patch16` | 16 | the 4×4 grid alone |
| `cls_patch16` | 17 | the paper-shaped condition |
| `patch16_mean` | 1 | the 16 tokens averaged — patch-only pooling control |
| `cls_patch16_mean` | 1 | the same 17 tokens averaged — the paper-shaped pooling control |
| `direct` | 32 | Direct's native 32×32 spatial latent (64×16 bottleneck repacked at `packing=2`), at its native 64-frame context |

Direct's TRAIN encodings are indexed from `artifacts/eda/latent_cache_64` rather than
recomputed: its `cache_digest` equals the anchor's `encoder_digest`
(`665c0df7757fbc91`), and cache episode *i* is expert TRAIN slot `train[i]` with
byte-identical actions. That reuses 256 of 288 episodes. Reuse is never on trust — each
span is checked against a fresh contextual encode of the prefix plus the two checked
positions (the receptive field is 31, so they carry full context), which keeps the saving
instead of re-encoding the span it was meant to avoid. Fails closed beyond 1e-4.

Head: one learned action query, 4-head cross-attention at width 128, LayerNorm + MLP,
linear to 17 logits, cross-entropy. Fixed recipe and seed across every condition.
Uncertainty bootstraps episodes; a majority-action floor is reported.

**`cls_patch16` − `cls_patch16_mean` is the decisive contrast**: identical features,
tokens kept versus pooled. `cls_patch16` − `patch16_mean` would *not* be — it also adds
CLS, confounding token structure with CLS presence. `patch16` − `patch16_mean` is the
patch-only control.

Every condition predicts the same DEV frames, so differences are reported as **paired
episode-bootstrap intervals**, not as two separate per-condition intervals:
`cls_patch16 − cls_patch16_mean`, `patch16 − patch16_mean`, `cls − z`, `cls_patch16 − cls`.

## Outcome

250 TRAIN / 32 DEV expert episodes (six were shorter than `frames + PREFIX`), 64,000
TRAIN and 8,192 DEV frames. Majority-action floor **0.1467** [0.1329, 0.1621]. Direct's
TRAIN encodings were reused for all 250 episodes, each validated against a fresh
contextual encode at max-abs 3.8e-05.

Top-1 expert-action accuracy, DEV:

| arm | condition | tokens | top-1 | 95% interval |
|---|---|---:|---:|---|
| raw | z | 1 | 0.1625 | [0.1473, 0.1780] |
| raw | cls | 1 | 0.1678 | [0.1500, 0.1859] |
| raw | **patch16** | 16 | **0.2419** | [0.2180, 0.2690] |
| raw | cls_patch16 | 17 | 0.2406 | [0.2135, 0.2658] |
| raw | patch16_mean | 1 | 0.1729 | [0.1523, 0.1952] |
| raw | cls_patch16_mean | 1 | 0.1747 | [0.1554, 0.1975] |
| tc | z | 1 | 0.2009 | [0.1749, 0.2272] |
| tc | cls | 1 | 0.2173 | [0.1910, 0.2439] |
| tc | **patch16** | 16 | **0.2610** | [0.2308, 0.2938] |
| tc | cls_patch16 | 17 | 0.2593 | [0.2272, 0.2902] |
| tc | patch16_mean | 1 | 0.2175 | [0.1904, 0.2418] |
| tc | cls_patch16_mean | 1 | 0.2234 | [0.1987, 0.2496] |
| direct (anchor) | direct | 32 | 0.2275 | [0.2019, 0.2567] |

Paired episode-bootstrap differences on the same DEV frames:

| arm | contrast | difference | 95% interval |
|---|---|---:|---|
| raw | **cls_patch16 − cls_patch16_mean** | **+0.0659** | [+0.0497, +0.0820] |
| raw | patch16 − patch16_mean | +0.0691 | [+0.0519, +0.0883] |
| raw | cls_patch16 − cls | +0.0728 | [+0.0539, +0.0922] |
| raw | cls − z | +0.0054 | [−0.0032, +0.0148] |
| tc | **cls_patch16 − cls_patch16_mean** | **+0.0359** | [+0.0266, +0.0455] |
| tc | patch16 − patch16_mean | +0.0435 | [+0.0333, +0.0544] |
| tc | cls_patch16 − cls | +0.0420 | [+0.0305, +0.0536] |
| tc | cls − z | +0.0164 | [+0.0094, +0.0228] |

**1. Keeping the tokens is worth more than anything else measured here.** The decisive
contrast — identical features, tokens versus pooled — is **+0.066** (raw) and **+0.036**
(tc), both intervals well clear of zero. The patch-only control agrees (+0.069, +0.044).
Pooling the 4×4 grid to one vector throws away most of what the grid carries: every
pooled condition sits within noise of `cls`, while the token conditions are 4–7 points
above it. Our `agent.py` mean-pools, so this is the interface we are currently using.

**2. TC beats Raw at every single condition.** z 0.2009 vs 0.1625, cls 0.2173 vs 0.1678,
patch16 0.2610 vs 0.2419. That reverses every prior result in this project — and it is
mechanistically consistent rather than surprising. The ladder measured TC's
action+interaction share at 0.29–0.38 against Raw's 0.02–0.04; predicting *which action
the expert took* is exactly the task that variation serves. TC's centering is not
destroying information so much as re-allocating it toward action-relevant structure, which
costs absolute state and pays for action inference.

**3. The projector costs TC but not measurably Raw.** `cls − z` is +0.0164 [+0.0094,
+0.0228] for TC and +0.0054 [−0.0032, +0.0148] for Raw. The CLS→z bottleneck the ladder
found in Raw's *state* retention does not show up in Raw's *action* prediction.

**4. Direct, the flagged anchor, is beaten by TC's patch grid.** 0.2275 against TC's
0.2610 and above Raw's `z`. It is not a matched comparison — Direct trained on this
archive, carries 1.46× the encoder parameters, and reads a 64-frame context against
LeWM's single frame — but it is no longer the ceiling it was on state retention.

### What this does and does not establish

It establishes that the frozen encoders support substantially better
current-observation action prediction than our CLS-only export delivers, and that the
paper's token-preserving interface is what unlocks it. It says nothing about imagined
rollouts: the paper never transitions patch tokens, and the
[bridge](../20260911_predictability_bridge/README.md) showed our world cannot. Absolute
accuracies are low because the target is a 17-way action from a **stochastic** recurrent
PPO expert — the manifest records `greedy: false`, and these episodes carry no epsilon at
all — so even a perfect model cannot match a sampled policy exactly. The contrasts, not
the levels, are the result. One seed, one archive.

This probes the **stride-1** frozen encoders. The stride-4 retrain postdates it, and its
checkpoints have not been run through this experiment.

## How we differ from TC-LeWM

Same ViT-Tiny family and the same 192-D embedding, `history_size = 3`, batch 128,
J = 1024, λ = 0.09, W = 4. The departures:

| | TC-LeWM (base LeWM config, "architectures unchanged") | ours |
|---|---|---|
| ViT input | 224×224, **patch 14 → 256 patch tokens** | 63×63, patch 7 → **81 patch tokens** |
| predictor | ARPredictor: 6-layer transformer, 16 heads, `dim_head` 64, MLP 2048 → **~9.46M** | 6× Mamba-2 width 256 → **2.50M** |
| frame skip | **4** | **1** |

Encoder *parameters* are within ~2% (both ViT-Tiny; only the patch-embedding conv and
position table differ), so the encoder is not under-parameterised — but it sees **3.2×
fewer patch tokens**, and their predictor is **~3.8× ours**. The Mamba swap is this
project's thesis and not a defect, but the capacity gap was never stated.

Direct is a flagged anchor, not a matched comparison: it trained on this archive and the
LeWM arms did not, it keeps its native 64-frame context while LeWM sees one frame, and it
carries 1.46× the encoder and 3.90× the world parameters.

## The frame-skip retrain

Implemented and recorded separately in
[frame_skip_retrain](../20260915_frame_skip_retrain/README.md): `joint.stride` under a
versioned recipe schema that keeps every existing v1 checkpoint digest bit-identical,
with stride-4 recipes for both arms. The v2/v3 paper discrepancy over stacked actions,
predictor context and window size is resolved there in favour of the pinned v3.

## What this cannot establish

One-step action prediction from a single observed frame. Not control, not a rollout. A
positive result shows the frozen encoder supports better current-observation control and
that CLS-only export hides it; it says nothing about imagined rollouts, because the paper
never transitions the patch stream either.

## Run

```bash
TRITON_F32_DEFAULT=ieee JAX_PLATFORMS=cpu .venv/bin/python \
  artifacts/experiments/20260915_patch_token_policy/patch_policy.py --device cuda
```

The report pins the archive SHA256, the script SHA256, the frozen-evaluation proof, the
per-arm checkpoint identities, and every sampled episode's slot and start index.

Immutable output at `evidence/policy.json`. `--limit` writes `policy.smoke.json`.

Requires `--frozen-eval-proof` (defaulted to `d4mj/m03/frozen_eval_compat.json`) because
the stride work moved the source manifest; see [`d4mj/m03/README.md`](../../../d4mj/m03/README.md).
