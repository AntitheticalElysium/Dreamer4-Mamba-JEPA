# Centering-window ablation: was it the window, or the horizon?

Status: **complete**, 2026-09-17. Both arms reached 10,000 updates; the completed-budget
audit passed all eight components. **The centering window accounts for the whole stride-4
rank recovery.**

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

Two index spaces are in play, and they must not be confused: **native offsets** are frame
positions in the episode, while **tensor positions** index the encoded window, whose frames
are the sorted union of both sets.

| | encoded window (native offsets) | prediction (native) | centering (native) | centering (tensor positions) |
|---|---|---|---|---|
| `consecutive` (control) | 0,1,2,3,4,8,12 | 0,1,2,3 | **0,1,2,3** | 0,1,2,3 |
| `strided` (treatment) | 0,1,2,3,4,8,12 | 0,1,2,3 | **0,4,8,12** | 0,4,5,6 |

The strided arm centers over frames four native steps apart — offsets 0, 4, 8, 12, spanning
13 steps. Those sit at tensor positions 0, 4, 5, 6 because native 8 and 12 are the sixth and
seventh frames of the seven-frame window; there is no native offset 5 or 6.

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

### On what basis the spectra are read

The screen and the audit encode seven frames but roll out four, and every reading --
the outgoing actions, the labels, the prediction target, the retention probes -- is
indexed on the three prediction transitions. So the headline spectra are read on the
four prediction frames, native offsets 0-3, in **both** arms.

That makes the within-ablation contrast clean, and exactly matches the basis the
stride-1 run used. It does **not** match the stride-4 run, which read its spectra across
frames 4 native steps apart, spanning 13. Comparing a number from this ablation directly
against TC s4's 16.73 therefore compares two different measurement bases, and the
comparison is indicative rather than decisive.

For a matched cross-reference the completed audit also records `window_spectra`: the same
three spectra over all seven encoded frames, which span the same 13 native steps as the
stride-4 window and are still measured identically in both arms. Read the primary table
for the treatment effect and `window_spectra` for the comparison to stride-4.

| outcome | conclusion |
|---|---|
| strided recovers rank toward the stride-4 figure | the window was the cause; horizon and action stacking were incidental |
| strided ≈ consecutive | the window was **not** the cause; the stride-4 recovery came from the 4-step horizon or the stacked actions |
| strided recovers rank but loses proxy retention | the trade TC already shows is intrinsic to centering timescale, not to our recipe |

The matched proxy comparison (TC-minus-Raw MLP AUC) is not available here, since both arms
are TC. Against Raw the references are the existing stride-1 and stride-4 runs.

## What had to be fixed first

A TC/TC pair broke four places that had quietly assumed either that the arm slot names
the declared variant, or that the encoded window is the prediction window. The first
cost a 2,000-update run; the rest were found by reading ahead of the job.

| where | assumption | consequence |
|---|---|---|
| `screen_joint_pair` | the `tc/` arm declares `variant: tc` | G1 stopped on `pair_identity` after 44 min |
| `screen_features` | encoded window == prediction window | would have failed G1: 7 frames, 3 actions |
| `require_joint_screen` | `arms[config.variant]` names one arm | would have refused each arm its own parent |
| `evaluate_completed` | slot == variant; one log per campaign | would have failed after the full 10,000 updates |

All four now go through one definition. `pair_axis()` returns the single declared axis a
pair differs on, and the screen, the launcher and the audit all call it; arms are
resolved by sealed recipe digest rather than by the variant label; the log is run-local.

Verified before relaunching, rather than by running into them: the centering-pair case
now drives the whole G1 screen to a decision and both arms through continuation in
`d4mj/tests/test_joint_screen.py`, and a toy centering pair was driven end to end through
the completed-budget audit -- all eight components, both heartbeat slots, `window_spectra`
recorded. Each fix was also checked to fail on the code it replaced, so the new assertions
are known to bite rather than assumed to.

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

## Result

Completed budget, 10,000 updates per arm, one seed. Audit decision
`joint_budget_audited_m4_blocked`, all eight components pass, 200 heartbeats verified,
`pair_axis: joint.centering`. M4 remains blocked.

Spectra on the four prediction frames (native offsets 0-3), identical basis in both arms:

| arm | rank_raw | rank_residual | rank_persistent | coord_var | mean_norm | norm_pred_mse |
|---|---|---|---|---|---|---|
| TC consecutive (control) | 5.1445 | 5.6881 | 4.5931 | 3.8623 | 3.6738 | 0.0728 |
| TC strided (treatment) | **19.1643** | **18.4537** | **18.2254** | 1.8579 | 5.4527 | 0.0421 |
| effect | **+14.02** | +12.77 | +13.63 | −2.00 | +1.78 | −0.031 |

Matched-span cross-reference, spectra over all seven encoded frames (offsets
0,1,2,3,4,8,12, spanning the same 13 native steps as the stride-4 window):
**5.1908 → 19.4573, +14.27**. The two bases agree, so the effect is not an artifact of
which frames are measured.

### The window was the cause

| run | TC rank_raw | basis |
|---|---|---|
| stride-1 TC | 5.6842 | offsets 0-3 |
| **this control (consecutive)** | **5.1445** | offsets 0-3 |
| stride-4 TC | 16.7285 | offsets 0,4,8,12 |
| **this treatment (strided)** | **19.1643** | offsets 0-3 |

Two things follow. The control reproduces the stride-1 collapse (5.14 against 5.68)
*while encoding the same seven frames as the treatment*, so the seven-frame window and
the BatchNorm batch are not what moves rank. And the treatment, which changes only which
index set SIGReg centers, clears the stride-4 figure outright (19.16 against 16.73).

The stride-4 retrain moved TC rank by +11.04 (5.68 → 16.73) while changing three things at
once. Changing the centering window **alone** moves it +14.02. So the window is not merely
the dominant term — it is the whole effect, and the 4-step horizon and the stacked actions
contributed nothing positive to the rank recovery. This answers the question the experiment
was built to ask, in the first row of the Readings table.

### Scale inflation is only half addressed

Coordinate variance falls 3.86 → 1.86, essentially onto the stride-4 value (1.89). But the
mean norm *rises*, 3.67 → 5.45, against 3.03 at stride-4 and 3.39 at stride-1. Whatever the
widened window fixes about the spread of the latent, it does not fix the offset, and on that
axis it is worse than either earlier run. "Scale inflation gone" would be wrong here.

### Semantic retention did not follow the rank

`projection_stop` is false in both arms, so neither is blocked. The two probe families
disagree about which arm loses more in the projection, and each is significant in only one
arm:

| probe | TC consecutive | TC strided |
|---|---|---|
| linear | −0.0193, CI [−0.0434, +0.0054] | −0.0461, CI [−0.0812, −0.0097] |
| MLP | −0.0367, CI [−0.0593, −0.0141] | −0.0101, CI [−0.0384, +0.0159] |

(projected AUC − CLS AUC; negative means the projection discards what CLS retains.)

Because they point opposite ways, **this run does not support a claim that the widened
window trades retention for rank, nor that it improves retention.** The absolute MLP
projected AUCs are within about 0.02 of each other across arms (reward 0.720/0.723,
negative reward 0.525/0.532, achievement 0.793/0.772). The honest reading is that a
threefold change in latent rank moved these short-future semantic proxies very little in
either direction — which is itself worth knowing, because it means rank recovery is not
by itself evidence of recovered semantics.

### Did the recovered rank buy any capability? No.

Run through [`d4mj/m03/`](../../../d4mj/m03/) against the shared 20260906 cache, stopped
after the two informative panels. Evidence in [`evidence/m03/`](evidence/m03/).

**The M03 verdict could not have depended on these checkpoints.** `m03_capability` is
gated by `critical_coverage`, which counts positive/negative label support and seed
clusters in the corpus; no model output enters it. Computed from this run's own stages it
reports `insufficient_coverage` over a missing set of 116 entries that is *identical* to
the 20260906 run's.

That settles the formal **status** and nothing else. It was first used here as a reason to
stop the run, which was a mistake: predetermined coverage predicts the gate verdict, not
the metric values, and certainly not hidden-state retention. Gate eligibility is not
diagnosis. The run was stopped during `historical_encoding`, before the Mamba-state memory
supplement, which is the only panel that probes `h` at all; it was measured afterwards by
`--baseline` over this run's published replay.

Static retention, mean AUC over the 13 binary targets with adequate support:

| source | linear | mlp |
|---|---|---|
| TC consecutive (control) | 0.7268 | 0.7433 |
| **TC strided (treatment)** | **0.7516** | **0.7616** |
| TC stride-1 | 0.7305 | 0.7522 |
| Raw stride-1 | 0.7880 | 0.7882 |
| Direct-Mamba anchor | 0.8299 | 0.8484 |

All-action one-step outcomes, strided minus consecutive: **no significant positive
difference on any target.** Three significant negatives -- `inventory_changed` −0.031
(linear), `achievement_event` −0.079 and `inventory_changed` −0.040 (mlp).

So a 3.7x change in effective rank (5.14 → 19.16) bought about +0.02 static-retention AUC,
left the arm below Raw stride-1 and far below the Direct anchor, and slightly *hurt*
outcome prediction on this panel.

The supported claim is narrower than "the window fixes geometry, not capability". Every
number above, and every number in the BC probe below, reads the **observed encoder
pathway** -- projected `z`, CLS, patch tokens. None of them reads the Mamba hidden state,
and the older TC checkpoint showed that `h` is exactly where the asymmetry lives:
generated-outcome AUC `z` 0.502 against `h` 0.695, while generated successor state stayed
flat at `z` 0.610 against `h` 0.611. So observed-feature panels agreeing with each other is
not three independent confirmations; it is one pathway measured three ways. What rank
recovery does to `h` is a separate question, answered by the memory supplement below.

The Direct-Mamba anchor scores 0.8299 / 0.8484 in both runs, to every printed digit. That
is an independent check that the bridged feature cache returned the same bytes, since those
encodings were reused rather than recomputed.

### The projection collapses to the action prior

Expert behaviour cloning over both arms, via the
[patch-token probe](../20260915_patch_token_policy/patch_policy.py); evidence in
[`evidence/patch_policy/`](evidence/patch_policy/). 250 train / 32 dev episodes,
64,000 samples, top-1 accuracy, episode-clustered intervals.

Majority-action floor **0.1467**, interval [0.1329, 0.1621].

| interface | TC consecutive | TC strided |
|---|---|---|
| projected `z` | 0.2039 [0.1804, 0.2290] | **0.1420 [0.1268, 0.1597]** |
| `cls` | 0.2356 [0.2107, 0.2611] | 0.1663 [0.1489, 0.1865] |
| `patch16` | 0.2584 [0.2296, 0.2896] | 0.2476 [0.2212, 0.2747] |
| `cls_patch16` | 0.2543 [0.2222, 0.2848] | 0.2499 [0.2204, 0.2797] |
| Direct-attention | 0.2275 [0.2019, 0.2567] | -- |

The strided arm's projected `z` is **not distinguishable from the majority-action floor**:
0.1420 against 0.1467, with intervals that overlap almost entirely. The consecutive arm's
`z`, at 0.2039, is clearly above it. So widening the centering window drove the projection
from carrying real action-relevant structure to carrying approximately none, while tripling
its effective rank.

The patch tokens, which SIGReg never regularizes and the world never predicts, barely move:
0.2584 against 0.2476, intervals overlapping. The damage is confined to exactly the pathway
the treatment acts on.

Tokens beat pooled patches in both arms, and by more in the strided one -- `patch16` minus
`patch16_mean` is +0.058 [0.045, 0.072] consecutive and +0.076 [0.061, 0.093] strided, both
significant. `cls` minus `z` is positive in both, +0.032 and +0.024: the projection discards
what its own CLS retains, which the M03 static panel showed too.

This sharpens the picture for the observed-encoder pathway. Effective rank rose 3.7x on the
axis TC-LeWM is built around, and the projection simultaneously became useless for
predicting the expert's next action. On this pathway a higher-rank latent is not a better
latent, and the centering objective can raise rank by spreading the representation in
directions carrying nothing a policy can use.

Two limits on that reading. This probe consumes **observed** encoder features, so it says
nothing about the Mamba state or about imagined successors -- the pathway where the older
checkpoint's `h` beat `z` by 0.19 AUC on generated outcomes. And a widened window weakening
BC through `z` and CLS is not evidence that widening it improves BC anywhere; the honest
downstream win here is the patch interface, not the window.

Direct-attention reused 250 cached spans, agreeing with recomputation to 3.8e-5.

### The Mamba state: h survives, but loses to knowing the action

The supplement the first pass never reached, run by `--baseline` over this run's published
replay. Evidence in [`evidence/m03/memory_primary.json`](evidence/m03/memory_primary.json).
Primary panel, context 4, generated successors, mlp probes.

| arm | outcomes z | outcomes h | outcomes [z,h] | successor z | successor h |
|---|---|---|---|---|---|
| TC consecutive | 0.524 | **0.715** | 0.600 | 0.612 | 0.644 |
| TC strided | 0.499 | **0.702** | **0.525** | 0.607 | 0.622 |
| TC stride-1 | 0.502 | 0.695 | 0.614 | 0.610 | 0.611 |
| Raw stride-1 | 0.558 | 0.636 | 0.571 | 0.664 | **0.702** |

Two facts replicate on the new checkpoints. `h` beats `z` by about 0.20 AUC on generated
outcomes in every TC arm, and `h` does **not** rescue successor state, where it sits within
0.02 of `z`. Raw stride-1 still leads on successor state outright.

The widened window does not help the hidden state. `h` drifts down slightly, 0.715 → 0.702,
and the joint readout collapses: 0.600 → **0.525**, far below `h` alone. In the strided arm,
concatenating `z` with `h` drags a 0.70 readout most of the way to chance, which is the
memory-side signature of the same `z` degradation the BC probe found.

**The action-only control, read correctly.** These probes are fitted once on
`train_observed_successor` and then applied to observed, generated and reset latents alike
(`gate.py:1784`). The action-only baseline is fitted natively. So a generated-condition
comparison against it pits a transfer-handicapped score against an in-domain one, and the
two conditions must be reported separately. Action-only floor 0.6626; primary panel, c4,
mlp:

| readout | observed | generated | obs − action | gen − action |
|---|---|---|---|---|
| `[z,h]` consecutive | 0.7551 | 0.5995 | **+0.092** | −0.063 |
| `[z,h]` strided | 0.7615 | 0.5250 | **+0.099** | −0.138 |
| `h` consecutive | 0.7155 | 0.7155 | **+0.053** | **+0.053** |
| `h` strided | 0.7023 | 0.7023 | **+0.040** | **+0.040** |

In domain, `[z,h]` beats action-only by about +0.09 in **both** arms, and the strided arm is
marginally the better of the two. `h` alone beats it by +0.04 to +0.05 and carries **no
transfer penalty at all**: its observed and generated scores are identical, because `h` is
built from context that does not differ between the conditions.

The deficit is therefore specific to **generated `z`**, which drags the joint readout below
the floor. Whether that is missing information or a decoder fitted in the wrong subspace is
not settled by this panel: the generated-condition numbers come from an observed-fit decoder,
so they confound the two. `h` is the built-in control -- a deficit that appears for `z` and
the joint but not for `h` is localised to the successor latent, not to the readout.

The paired generated-condition differences are still recorded in
[`evidence/m03/memory_primary.json`](evidence/m03/memory_primary.json), and remain valid as
what they are: a transfer comparison, not a statement that memory fails to beat the action
prior.

### What this does not establish

One seed, one dataset, no Raw arm. The cross-run comparisons to stride-1 and stride-4
involve different recipes and, between those two, different measurement bases; only the
within-ablation contrast is fully controlled. Nothing here is an M03 capability result:
the semantic panels still depend on the deferred macro-fork convention, and prediction MSE
is not comparable across arms, since each arm predicts its own encoder's latents and the
persistence baselines differ by an order of magnitude (3.37 against 0.31).

## Also built

`LeWMEncoder.export()` returns projected `z`, unprojected CLS and the 4×4 pooled patch grid
from one pass, so the observation and policy interfaces are first-class rather than captured
by a forward hook. It adds no training objective: patch tokens remain unregularized and
unpredicted, as in the paper. `projected_and_cls` is bitwise unchanged.

The [BC probe](../20260915_patch_token_policy/README.md) now takes `--checkpoint NAME=PATH`,
so both observation interfaces can be evaluated on any pair — including these arms once
trained.
