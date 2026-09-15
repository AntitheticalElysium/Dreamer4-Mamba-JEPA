# Patch-token BC: does token-preserving cross-attention beat pooled export?

Status: ready to run. Structural smoke passed on CUDA; no result recorded.

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
| `patch16_mean` | 1 | the 16 tokens averaged — separates cross-attention over tokens from the information in them; this is what `agent.py` does today |
| `direct` | 32 | Direct's native 32×32 spatial latent (64×16 bottleneck repacked at `packing=2`), at its native 64-frame context |

Head: one learned action query, 4-head cross-attention at width 128, LayerNorm + MLP,
linear to 17 logits, cross-entropy. Fixed recipe and seed across every condition.
Uncertainty bootstraps episodes; a majority-action floor is reported.

`cls_patch16` − `patch16_mean` is the decisive contrast: same features, tokens kept
versus pooled.

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

## The frame-skip retrain, specified but not implemented

The paper trains at frame skip 4, so a four-frame window spans 16 environment steps;
`joint.frames = 4` samples consecutive Craftax steps. Measured on support_v2, consecutive
frames differ in **19.0%** of pixels against **45.0%** at lag 4 — our centering window
carries roughly a third of the within-window variation TC's regularizer was designed
around. That is a candidate mechanism for TC's rank collapse (39.2 → 5.7) and it is a
recipe gap, not an architecture one.

Implementing it requires care, and the hazard is concrete: `recipe_digest` is
`sha256(canonical_json(asdict(config)))`, and `load_m03_bundle` rejects a mismatch. Adding
a `skip` field to `LeWMConfig` changes the digest for **every** config, so the existing
raw/TC checkpoints would stop loading and M03, the ladder, the bridge and this experiment
would all break. A frame-skip run therefore needs a versioned recipe schema that keeps
old digests intact, not a new field on the current dataclass.

Two further decisions a Craftax frame-skip recipe must settle, neither of which LIBERO
faces: with skip 4 there are four actions between retained frames, so the predictor's
single outgoing action is only the first of them, and the remaining three are unobserved
causes of the transition. And `window_weights` counts windows by frame count, not span.

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

Immutable output at `evidence/policy.json`. `--limit` writes `policy.smoke.json`.
