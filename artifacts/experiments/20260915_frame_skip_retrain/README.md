# Frame-skip retrain: is TC's centering window physically too short?

Status: recipes ready, not yet trained. Schema and sampler implemented and tested;
existing checkpoints verified still loadable.

## The finding that motivates it

TC centres over a window of `joint.frames = 4` **consecutive** Craftax steps. TC-LeWM
trains at frame skip 4, so its four-frame window spans 13 native steps. Measured on
support_v2, consecutive frames differ in **19.0%** of pixels against **45.0%** at lag 4 —
our centering window carries roughly a third of the within-window variation the
regularizer was designed around. `r_t = z_t − z̄_t` over a window where little happens is
a small, noise-dominated residual, which is a candidate mechanism for TC's rank collapse
(39.2 → 5.7) and its CLS action-concentration. This is a **recipe** gap, not an
architecture one.

## Verified: the paper changed between v2 and v3

PAPERS.lock pins **v3**. The two versions disagree on all three numbers that matter here,
so the recipe must name which it follows:

| | v2 (arXiv HTML) | v3 (pinned PDF) |
|---|---|---|
| predictor action input | **"Stacked actions, frame_gap×7"** | Eq (8) `ẑ_{t+1} = g_ψ(z_{t−h+1:t}, a_{t−h+1:t})` — one action per retained frame |
| predictor temporal context | **"8-frame sequence with learned temporal embeddings"** | Table 4: "Predictor context h=3" |
| centering window | **"W=8 frames, ≈1.4 s on LIBERO"** | Table 4: "W = 4" |

Both agree on frame skip 4. v3 is followed here, because it is what PAPERS.lock pins and
what our architecture already matches (4 frames / 3 outgoing actions ≡ h=3, W=4).

The v2 "≈1.4 s" is worth keeping: at LIBERO's ~20 Hz, 8 *native* frames is 0.4 s, so
1.4 s implies 8 **retained** frames at the frame gap (8 × 4 / 22 Hz ≈ 1.45 s). That
confirms the centering window counts retained frames and its physical span is
`gap × W` native steps — the point this experiment turns on.

The loader that would settle whether the *implementation* stacks actions
(`stable_worldmodel`'s dataset, which supplies the base config's `frameskip: 5` and the
action `input_dim: ???`) is neither vendored nor installed, so the v2/v3 difference cannot
be resolved from local sources. It is recorded, not guessed.

## What was implemented

`joint.stride` on `JointSettings`: native steps between retained frames. A four-frame
window at stride 4 spans 13 native steps, and the retained outgoing action is the **first
of the four** inside each transition — three intermediate actions are unobserved causes,
exactly what v3's Eq (8) specifies and a departure LIBERO does not face, since its
actions are continuous and v2 stacks them.

Frame count, action count and batch are unchanged, so the objective and the statistical
batch are untouched; only the window's physical span changes.

### The compatibility guarantee

`recipe_digest` is `sha256(canonical_json(asdict(config)))` and `load_m03_bundle` rejects
a mismatch, so a naive new field would have stopped every existing raw/TC checkpoint from
loading and broken M03, the ladder, the bridge and the BC probe. Instead:

- `recipe_dict` omits `joint.stride` at its default under schema `v1`, so **every existing
  v1 digest is bit-identical**. Verified directly: both checkpoints' stored `recipe_id`
  still equals `recipe_digest(parsed config)`.
- `stride != 1` requires schema `d4mj_lewm_recipe_v2`, so no v1 digest can ever denote a
  strided run, and v2 retains the field so digests cannot collide.
- Two regression tests in `test_joint_data.py` pin both properties. Full suite: 261 passed.

## Design: which retrain to run

Two options, both served by this schema. They answer different questions.

**B — faithful v3 (what the recipes encode).** Stride the whole window: 4 retained frames
at stride 4, one action per retained transition. Faithful, and it changes *two* things at
once — the dynamics horizon becomes 4 native steps **and** the TC window spans 13. If TC
improves, the cause is ambiguous.

**A — TC timescale only (a controlled ablation).** Keep 1-step dynamics and stride only
the latents entering SIGReg. This isolates "our TC window is physically too short"
without turning one-action dynamics into four-step dynamics.

A is sound and is the cleaner *first* test, but it needs a second field and costs more per
step than first appears: to hold the prediction objective fixed at 3 consecutive pairs
*and* span the SIGReg window over 13 steps, a window must encode 7 frames — the 4
consecutive prediction frames plus 3 more at stride 4 — about 1.75× the encoder cost. It
is also explicitly **not** faithful: it regularizes latents that are never prediction
targets, which the source never does.

Recommendation: run **B** first, because it is the faithful recipe and the pinned source's
actual configuration, then **A** only if B moves TC — at which point A attributes the gain
between horizon and timescale. Running A first inverts the usual order: it spends the
cheaper controlled experiment before knowing there is an effect to attribute.

## Cost and what it cannot establish

Two arms × 10,000 updates at ~0.30 s/step ≈ 50 min each on the recorded RTX 3060 profile,
plus G1 at update 2,000. Paired seeds, identical initialization, one seed — as before, no
uncertainty across training seeds.

A stride-4 run is a **new** training run with new checkpoints and a new recipe identity.
It does not re-validate M0–M3, does not authorize M4, and its G1 screen must use the same
span (`screen_windows` still counts consecutive frames and needs the same treatment
before a v2 paired run — that is the one remaining implementation gap).

## Run

```bash
TRITON_F32_DEFAULT=ieee .venv/bin/python -m d4mj preflight \
  --recipe d4mj/recipes/lewm_mamba_tc_stride4.json \
  --dataset artifacts/craftax_support_v2 --out <fresh-run-directory>
```

Recipes: `lewm_mamba_raw_stride4.json` (digest `40c6f93d290d…`),
`lewm_mamba_tc_stride4.json` (digest `79895890026a…`). Both schema v2, stride 4, span 13.
