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

`joint.stride` on `JointSettings`: native steps between retained frames. Four retained
frames at stride 4 occupy **13 observation indices** and cover **12 environment
transitions** — three retained transitions of four native steps each.

**Actions are stacked, not subsampled.** TC-LeWM's predictor takes
`"Stacked actions, frame_gap × 7"` (v2 Table 4), so each retained transition carries
*every* native action inside it. `LeWMWorld` widens only `pair_projection` to
`latent_dim + stride × action_dim`; every other parameter shape is untouched, and at
stride 1 the stack is one action and all shapes are byte-identical to the v1 recipe.

This matters far more in Craftax than in LIBERO. Measured on the expert archive over
692,906 eligible stride-4 windows, **99.76%** contain at least one action change that
first-action conditioning would discard, and only **3.38%** of retained transitions have
their four native actions internally constant. Subsampling would have meant the predictor
almost always sees one action while predicting a transition caused by four different
ones. Stacking removes that aliasing entirely.

### Which paper version this follows

v3 is pinned, but v3's Table 4 has **no action-input row at all** and the word "stack"
appears nowhere in it; Eq (8)'s `a_{t−h+1:t}` is generic notation that cannot settle how
four native actions reduce to one symbol. v2 states stacking explicitly. First-action
conditioning is stated in *neither* version. So stacking is the only documented design and
is what we implement. The caveat: v3 also changed `h` (8→3) and `W` (8→4) in that table,
so we cannot be certain only the action row was dropped for space.

### The compatibility guarantee

`recipe_digest` is `sha256(canonical_json(asdict(config)))` and `load_m03_bundle` rejects
a mismatch, so a naive new field would have stopped every existing v1 checkpoint loading.
`recipe_dict` omits `joint.stride` at its default under schema `v1`, leaving every v1
digest bit-identical — verified directly against both checkpoints' stored `recipe_id` —
and `stride != 1` requires schema `v2`.

Changing `data.py`, `config.py`, `lewm_config.py` and `lewm.py` does still move the
*source manifest*, which is a separate identity. That is handled by the measured
frozen-evaluation proof in [`d4mj/m03/`](../../../d4mj/m03/README.md): stride-1 inference
across the delta agrees to 5.96e-07, equal to the kernel's own within-tree run-to-run
envelope, with some runs bitwise identical. Training resume stays strict.

G1 samples the same span and aggregates each retained transition's label over its four
native steps (total reward, any event or termination) — `screen_windows` and
`audit_episodes` both count span. Regression tests pin span, stacking and label
aggregation against the unstrided case.

## What this cannot establish

A stride-4 run is a **new** training run with new checkpoints and a new recipe identity.
It does not re-validate M0–M3 and does not authorize M4. It still changes two things at
once — the dynamics horizon becomes four native steps *and* the centering window spans 13
indices — so a TC improvement would need the SIGReg-stride-only ablation to attribute
between horizon and timescale.

One open consequence: M03's all-action fork evaluation forks 17 *single* actions. Against
a stacked-action world that is ill-posed — a candidate is now a four-action sequence — so
evaluating stride-4 checkpoints needs an explicit forking convention (fork the first
action holding the rest, or fork sequences) before those panels mean anything.

## Run

```bash
TRITON_F32_DEFAULT=ieee .venv/bin/python -m d4mj paired-run \
  --dataset artifacts/craftax_support_v2 \
  --raw-recipe d4mj/recipes/lewm_mamba_raw_stride4.json \
  --tc-recipe  d4mj/recipes/lewm_mamba_tc_stride4.json \
  --out artifacts/lewm_gates_20260916/paired_stride4
```

Two arms x 10,000 updates at ~0.30 s/step on the recorded RTX 3060 profile, plus G1 at
update 2,000. Paired seeds, identical initialization, one seed.
