# Frame-skip retrain: is TC's centering window physically too short?

Status: **complete**, 2026-09-16. Both arms finished the 10,000-update budget in 2h52m.
G1 passed; the completed-budget audit passed all eight components. Verdict: the longer
centering window recovers most of TC's rank collapse and all of its scale inflation,
without curing either fully. Result in [`evidence/completed/`](evidence/completed/).

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

The gate path is stride-aware too, which unit tests over the new code did not catch:
`recurrence_audit` built its own scalar actions and `screen_retention` one-hot encoded a
flattened action tensor, so `paired-run` would have died in preflight before training
began. Both now take the stack, and a regression test runs the LeWM gates on a stride-4
bundle rather than only the sampler and the world.

G1 samples the same span and aggregates each retained transition's label over its four
native steps (total reward, any event or termination) — `screen_windows` and
`audit_episodes` both count span. Regression tests pin span, stacking and label
aggregation against the unstrided case.

## G1 at update 2,000

Paired stride-4 run `artifacts/lewm_gates_20260916/paired_stride4`, against the sealed
stride-1 screen from [20260906_lewm_paired](../20260906_lewm_paired/evidence/g1_screen.json).
Same screen recipe, same 256 TRAIN / 128 DEV episodes, same fixed windows.

| metric | raw s1 | raw s4 | tc s1 | tc s4 |
|---|---:|---:|---:|---:|
| DEV effective rank | 23.10 | 20.48 | **3.08** | **7.82** |
| coordinate variance | 0.818 | 0.722 | 2.623 | 0.652 |
| mean norm | 1.921 | 2.008 | 4.105 | 3.168 |
| normalized prediction MSE | 0.0653 | 0.0745 | 0.0646 | 0.6125 |

**1. TC's rank collapse is partly a timescale artifact — but only partly.** Effective rank
rises **3.08 → 7.82**, a 2.5× recovery, which is direct support for "our centering window
was physically too short". It remains 2.6× below Raw's 20.48, so the longer window
mitigates rather than cures it.

**2. TC's scale inflation looks *mostly* like a timescale artifact.** Coordinate variance
falls 2.62 → 0.65 and is now comparable to Raw's 0.72, where at stride 1 it was 3.2×
Raw's. The inflated energy M03 and the ladder both measured (TC total energy 4.8–6.5×
Raw's) is substantially a consequence of centering over four native steps, not something
intrinsic to centering.

**3. Raw's prediction beats persistence for the first time.** At stride 1 it was
*unresolved* — −0.0015 [−0.0072, +0.0043], straddling zero. At stride 4 it is
−0.0191 [−0.0244, −0.0144]. A four-step physical horizon makes persistence a much weaker
baseline, and the model clears it cleanly.

**4. But TC's prediction task got much harder**: normalized MSE 0.0646 → 0.6125, against
Raw's 0.0653 → 0.0745. Predicting four native steps ahead costs TC an order of magnitude
where it costs Raw almost nothing.

### A normalization caution

`prediction_minus_persistence` and `prediction_minus_permuted_actions` are raw-MSE
differences, and the two arms' latent scales differ by ~4× *and changed between strides*,
so those numbers are not comparable across arms or across strides as reported. Dividing by
each arm's own coordinate variance — a derived quantity, not one the evaluator computes:

| contrast, ÷ own variance | raw s1 | raw s4 | tc s1 | tc s4 |
|---|---:|---:|---:|---:|
| prediction − persistence | −0.002 | −0.027 | −1.109 | −1.382 |
| prediction − permuted actions | −0.027 | −0.040 | −1.759 | −0.631 |

Read that way TC's margin over persistence *improves* (−1.11 → −1.38) while its margin
over permuted actions *degrades sharply* (−1.76 → −0.63). With four stacked actions per
transition, action identity is more diffuse and shuffling costs the model less. That is
worth watching at 10,000 updates; it is the one reading here that points against the
stride change.

These are update-2,000 screen numbers on one seed, not the completed budget, and G1 is a
screen rather than a capability result.

## Completed budget (update 10,000)

Run `artifacts/lewm_gates_20260916/paired_stride4`, audited by the same read-only driver
used for the stride-1 pair. All eight components pass: pair identity, dataset and windows,
and normalization / recurrence / projection-retention for both arms.

**Effective rank — the number this experiment was built to move:**

| spectrum | raw s1 | raw s4 | tc s1 | tc s4 |
|---|---:|---:|---:|---:|
| raw | 39.231 | 34.088 | **5.684** | **16.729** |
| residual | 25.951 | 19.283 | 5.735 | 16.315 |
| persistent | 38.942 | 33.845 | 5.474 | 9.002 |

TC's raw-latent rank recovers **5.68 → 16.73, a 2.9× gain**, and the raw/TC gap narrows
from 6.9× to 2.04×. The residual spectrum — the one SIGReg actually regularizes under
centering — moves almost identically (5.74 → 16.32). The persistent spectrum gains least
(5.47 → 9.00), which is coherent: centering never constrained the time-constant component,
so a longer window helps it least. Raw loses some rank throughout (39.23 → 34.09).

**Scale inflation is gone, and slightly reversed.** TC's coordinate variance falls
2.76 → 0.553, now *below* Raw's 0.755, where at stride 1 it was 3.2× above. The inflated
energy M03 and the ladder both measured was an artifact of centering over four native
steps, not a property of the objective.

**Prediction, stride-4 only.** Both arms clear persistence and permuted actions:

| arm | normalized MSE | pred − persistence | pred − permuted actions | permuted ÷ own variance |
|---|---:|---:|---:|---:|
| raw | 0.0342 | −0.0527 [−0.0599, −0.0458] | −0.0246 [−0.0299, −0.0198] | −0.033 |
| tc | 0.0988 | −3.1091 [−3.3770, −2.8489] | −0.2572 [−0.3084, −0.2121] | −0.465 |

There is no stride-1 counterpart: that audit **stopped** its prediction diagnostics at the
failed TF32 recurrence gate, so this comparison exists only at G1. Normalized by each arm's
own variance, TC remains an order of magnitude more action-sensitive than Raw (−0.465 vs
−0.033), though it continued the degradation seen at G1 (−1.76 at s1 → −0.63 at s4/G1 →
−0.465 at s4/10k). Stacking four actions per transition makes action identity more
diffuse, and that cost grows through training.

**Retention is unchanged.** `projection_stop` is False for both arms in both runs and
`critical_semantic_retention` remains `not_evaluated` — the two-probe destruction rule did
not fire at stride 4 any more than at stride 1. The longer window did not change the
projection-boundary verdict either way.

**One incidental improvement:** the stride-4 checkpoints pass `tc_recurrence` and
`raw_recurrence` natively. The stride-1 pair failed the full-stack SSM tolerance under its
original TF32 environment and needed the separate IEEE diagnostic; this run used explicit
IEEE throughout, so its recurrence contract holds without an after-the-fact remedy.

### Reading

The timescale hypothesis is **substantially confirmed and partially insufficient**. Our
four-native-step centering window was the dominant cause of TC's scale inflation and a
majority of its rank collapse. It was not the whole story: at 16.73 against Raw's 34.09,
TC's representation is still markedly lower-rank, so something beyond window length is
compressing it.

What this does **not** establish is that the stride-4 arms are better *for control*. Rank
and prediction are representation diagnostics; the semantic panels that would settle it —
M03's all-action forks — are ill-posed against a stacked-action world until the macro-fork
convention is chosen. That remains the gating question, deferred deliberately.

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
