# Step 1 protocol (pre-registered): faithful same-frame I-JEPA representation

Committed BEFORE the first run, per the consensus matrix (re-audit §"Minimal next
experiment matrix", step 1) and the concurrence addition that pass thresholds be
numeric and fixed in advance.

## Objective under test

Same-frame I-JEPA, ported from the pinned source
(`facebookresearch__ijepa @ 52c1ae9`, `src/masks/multiblock.py`,
`src/models/vision_transformer.py::VisionTransformerPredictor`,
`src/train.py::forward_target/forward_context/loss_fn`, config
`in1k_vith14_ep300.yaml`), **isolated from action prediction** (no temporal
model, no task heads, no next-frame term).

Faithful elements:
- 4 target blocks/image, scale 0.15–0.2 each, aspect 0.75–1.5; one context
  region, scale 0.85–1.0, minus target overlap (`allow_overlap=False`);
  per-batch uniform index counts via the official min-trim; `min_keep=4`
  (scaled from 10 at 16×16-grid to our 8×8 grid).
- Context encoder processes ONLY visible tokens (token dropping, not
  mask-token substitution).
- Predictor: fixed 2D-sincos positional embeddings; learned mask token +
  target-position embedding as queries; per-target-block prediction passes
  (context repeated per block, as in the official forward); post-norm +
  projection.
- Targets: EMA encoder on the full frame, **LayerNorm over the feature dim**,
  gathered at target positions; loss = smooth L1 at masked positions only.
- EMA decay 0.996.

Labelled deviations (each with reason):
1. Conv stem instead of ViT patchify → masked-patch pixels leak into visible
   tokens through the receptive field. Direction of risk: makes the pretext
   task *easier* (weaker representation), cannot manufacture a metric pass.
   Fallback if gates fail: CNN-JEPA-style masked convolutions (SparK), to be
   pinned from its official repo before implementing.
2. 2 register tokens ride along in the context pass (needed downstream for
   HUD/inventory); they receive no positional queries and no loss.
3. Predictor depth 2, width d=64, no dim narrowing (theirs: depth 6+, 384 on
   768) — scale-appropriate.
4. Optimizer: AdamW lr 3e-4, wd 1e-4, no warmup/cosine, bf16 autocast — kept
   identical to every previous arm so cross-arm rows remain comparable
   (official uses lr 1e-3 + schedules at batch 2048).
5. No VICReg/SIGReg term in this arm: the experiment tests whether the true
   I-JEPA asymmetry alone prevents collapse at our scale. (The streamwise
   term exists in the hybrid control arm.)

## Arms (identical seeded data; replay RNG `np.random.default_rng(2027)`;
torch seed 101; training frames from the shared cache; probe set = seed-2
collect(400))

- `ijepa300`: objective above, 300 updates, batch 64 frames.
- `hybrid300`: current compact objective (streamwise-VICReg temporal hybrid,
  unmasked, LossConfig defaults, rollout off), 300 updates, batch 4×T16 (its
  native shape) — the incumbent control.
- `untrained`: same init, 0 updates — the reference row for every gate.

## Pre-registered gates (final row vs the SAME-RUN untrained row; measured with
`verification/representation_control.py` corrected functions on the held-out
probe set)

| gate | metric | pass bar |
|---|---|---|
| G1a | `target_observation_variance_fraction` | ≥ untrained − 0.05 |
| G1b | `target_same_stream_unrelated_cosine` | ≥ 0.80 × untrained |
| G2a | `target_stream_effective_rank_mean` | ≥ 0.90 × untrained |
| G2b | `target_patch_pool_covariance_rank` | ≥ 0.90 × untrained |
| G3 | semantic probe test accuracy (sane per train/test majorities) | ≥ untrained − 0.02 |
| G4 | inventory ridge R² (corrected split/standardization) | ≥ untrained − 0.02 |
| G5 | SSL loss, last-100 mean vs first-100 mean | ≤ 0.70 × first (the arm must actually learn; G1–G4 alone are passable by doing nothing) |

Curve logging every 25 updates: `target_observation_variance_fraction`,
`target_stream_effective_rank_mean`, loss. Abort rule: observation variance
fraction < 0.30 at any checkpoint (untrained ≈ 0.43) → stop, record failure.

Decision rule: pass all G1–G5 → step 2 (extend the SAME arm to 500 then 1000
updates; gate = no late reversal of G1–G4 and loss still ≤ its 300-update
value). Fail → CNN-JEPA masked-conv fallback (deviation 1) or official
LeJEPA/SIGReg image-level objective; no other knobs turned first.

Step 3/4 (frozen-encoder temporal + backend attribution) follow the matrix as
written in the re-audit; their thresholds will be pre-registered in their own
protocol file after step 2 fixes the encoder.

## Amendment 1c (pre-registered before the 1c run; after v1/v2 failures)

- v1 (token-dropping I-JEPA): failed G2a/G2b/G3 — conv-stem leak (deviation 1)
  made the task trivial (loss →0.009). As pre-registered, fallback applied.
- v2 (leak-free SparK/CNN-JEPA sparse stem, pinned b7b246c): G3 recovered;
  G2a/G2b still fail — I-JEPA asymmetry alone does not preserve per-stream
  diversity at this scale. Protocol fallback #2 applies.
- **Arm 1c `lejepa`**: identical to v2 plus SIGReg, the published LeJEPA
  composition (pinned rbalestr-lab/lejepa @ c293d29, MINIMAL.md):
  `loss = (1-λ)·L_ijepa + λ·L_SIGReg`, λ = 0.02 (their example default; official
  grid 0.01-0.1). SIGReg ported verbatim (17-knot quadrature on [0,3], 256
  fresh random unit projections, ECF match to standard Gaussian, ×N scaling)
  with two labelled adaptations: (a) sample axis = observations per stream
  (the corrected axis semantics; LeJEPA's axis is the image batch), statistic
  averaged over streams; (b) applied directly to a dense online-encoder pass
  with NO projector — the 2026-07-13 re-audit measured that a projector hides
  encoder collapse (its VICReg-projector arm: flat rank 3.5), and the gates
  measure encoder tokens.
- Gates G1-G5 unchanged. Same data, seeds, budget.
