# Foveal information route: discussion before another training run

2026-11-02. Discussion and proposed diagnostics only. No experiment or training
run is launched by this note. P1-F2's learned chooser is deferred by the user.
The question is whether an overnight run should instead test a repair to fine
evidence entering the persistent belief.

## 1. What the completed measurements establish

See [F1 RESULTS](../20261102_foveal_prediction/RESULTS.md). In the declared
night/entity-containing/no-scroll population, whole raw current glimpse tokens
reduce next-fine prediction error by 39–44% over the coarse/history baseline.
At the current entity's cell, the secondary reading is approximately 26% in
both diagnosis and disjoint confirmation cohorts. The native fine contribution
at the belief input gives only approximately 2–3% on that cell; the exported
memory gives approximately zero. These are bounded fitted-reader results, not
information-theoretic measurements or native generator improvements.

Two qualifications materially affect the next experiment:

1. F1's future fine target is the NEXT frame at the CURRENT crop's screen
   position. The predictor does not receive next-frame coarse tokens. This
   proves some history-to-future predictive value, but does not establish an
   incremental benefit after the future periphery is supplied. It also does
   not measure the actual random-next-fixation distribution.
2. F1's input supplement is the isolated fine contribution `W_g G`, not the
   sum of coarse and fine contributions. See
   `20261102_foveal_information/conditional.py:34-37`, reused by F1. Its weak
   readout cannot, by itself, identify interference from ADDING coarse content.

The coarse-target result is also narrower: raw fine has no reproducible large
advantage across both reader seeds in this population. Memory does improve the
coarse-target instrument, but that can include reformatting coarse history;
it does not establish fine-specific uptake.

## 2. The actual native routes

`saccade/model.py`, `Belief.percept_embed`, places the nine 64D fine tokens
at their correct locations on an 81-cell canvas and includes the fixation mask.
It then concatenates coarse tokens, the canvas, mask and light and applies one
linear map to 256 dimensions:

```
e_t = W_p vec(P_t) + W_g vec(G_canvas,t) + W_m mask_t + W_v v_t + b
```

So the input is already location-aware. What it lacks is a content-dependent
interaction between corresponding coarse and fine evidence BEFORE compression.
The subsequent Mamba is nonlinear. Its full internal carry is much larger than
its 256D output; four emitted memory tokens are projections of that output.
Calling the entire memory a 256D store would be inaccurate.

At a fixed fixation, 576 active fine coordinates enter a 256D linear map.
Some ambient-space directions must be discarded. That is not proof that the
natural fine representations contain 576 independent, task-relevant dimensions,
or that the discarded directions are useful. The map also varies by screen
location, making a shared bounded reader's task different from reading raw
crop-relative tokens. Scale, conditioning and sample sharing remain explanations.

The factorised denoiser has different information sets for its two outputs:

```
current fine ──> global projection ──> Mamba ──> memory ──> future physical
      │                                                    │
      └──────── past-glimpse temporal context ───────────────┼─> future fine
                                                           │
future physical (coarse/light) ─────────────────────────────┘
```

Physical spatial attention excludes glimpse keys/values, and temporal attention
is per token position. Thus past fine content reaches physical predictions
through the belief route. Glimpse queries, however, also have direct temporal
access to past glimpse tokens in the recent context. The future-glimpse decoder
is therefore not forced to recover recent fine evidence from Mamba.

This suggests a possible training-pressure problem: coarse predictions can use
coarse history, and fine predictions can use future coarse plus cached fine
history. Neither necessarily strongly rewards retention of recent fine content
in the recurrent belief. This is a plausible mechanism, not an identified cause.
It is particularly important to distinguish short-context fine prediction from
longer-term belief retention.

## 3. First proposed check: usefulness with the native conditioning available

The primary question is whether past fine evidence improves the next fine
distribution AFTER supplying next periphery/light and the older context.
The frozen checkpoint remains unchanged.

Use two complementary instruments; neither substitutes for the other:

- Matched conditional prediction/denoising readers, with an explicit information
  set. Hold older coarse/fine history, action, outcomes, previous belief and
  future query fixed; compare no current raw glimpse, current raw glimpse, and
  the actual native memory after that glimpse. Supply future physical tokens.
- A frozen-native paired-look test: take two genuine crops from the SAME frame
  following the same prior history, an informative crop and a matched control
  crop. Recompute the belief AND temporal cache consistently for each history.
  Hold the next target crop and real next physical tokens fixed across arms,
  use paired noise draws, and compare next-fine denoising and generated semantics.
  This tests the trained model's use of two real observation histories; arbitrary
  zeroed glimpses or incompatible memory/cache combinations are not equivalent.

Condition on clean real next physical tokens for the question as posed, then on
the actual cascade's conditioning augmentation (sigma 0.1) and representative
physical noise levels used in training. Report both repeated-region queries and
the actual next-fixation distribution. Use conditional denoising scores over
native noise levels as well as MSE: a mean-only readout can miss information
about uncertainty. Report episode-paired intervals and both existing diagnosis
and confirmation cohorts without using the final split.

Interpretation:

- Raw fine helps but memory does not: useful conditional information is available
  but not easily accessible through the exported belief. The direct cache may
  already supply part of this information to the native fine decoder.
- Raw fine's gain largely disappears: the earlier large gain is mostly redundant
  with the future periphery for this target/population. That would weaken the
  case for forcing memory to improve this particular next-fine target; it does
  not settle usefulness to action selection, hidden-state inference, long gaps,
  or other future physical targets.
- The native model benefits from informative looks through its direct cache but
  not persistent belief: the question becomes durable storage/inference, rather
  than merely getting information into the next glimpse decoder.

This test is not yet implemented or run. Its final protocol should specify
matched control selection, cohorts, noise levels, reader capacity and budget
before fitting, rather than select them after seeing improvements.

## 4. Second proposed check: projection versus accessibility

Audit the frozen input before assuming global compression is responsible:

1. Compare raw fine, isolated native `W_g G`, and an invertibly standardized or
   whitened version of `W_g G`, with matched prediction targets and fitted-reader
   budgets. Rescue by an invertible transformation demonstrates that useful
   information survived and its accessibility/conditioning mattered. Failure to
   rescue is not proof of deletion.
2. At fixed fixations, use the projection's SVD to separate retained and null
   components. Report ranks/singular values and predictive value, not rank alone.
   Natural-data redundancy may make a null component recoverable from retained
   components; finite-reader failures cannot establish its impossibility.
3. Compare isolated branches with the actual full sum, with the coarse input
   otherwise supplied identically to the instrument. This is needed to evaluate
   addition interference rather than reuse the isolated-branch result as if it
   measured interference.
4. Audit actual runtime BF16 accumulation versus FP32 with equivalent quantized
   inputs/weights. F1 computed the isolated contribution in FP32 and stored FP16;
   this is not the actual native mixed accumulation. Do not assume rounding is
   important before measuring it.

Compression, entanglement and pressure are overlapping hypotheses. These checks
can establish particular recoverability and numerical effects. They cannot
identify a unique universal root cause from a failed probe.

## 5. The user's spatial-fusion proposal

The substantive change would be a learned local, content-dependent integration
of corresponding coarse/fine evidence before global compression:

```
coarse cell + aligned fine evidence + light/mask
                  │
      shared local nonlinear fusion
                  │
       corrected spatial evidence
                  │
             256D projection -> Mamba
```

This provides local correspondence and shared learning across screen locations
before location-specific weights mix the evidence globally. It could improve
which features survive without increasing the recurrent emission width.

Two linear layers in succession alone are just another linear map, possibly
with a factorization constraint. The proposal needs a nonlinearity or
content-dependent gate/attention to add the interaction that is currently absent.
Immediately squeezing each 64D fine token into a 16D coarse replacement could
introduce another bottleneck; preserve the fine residual while testing the idea.

A minimal function-preserving pilot is:

```
P'_i = P_i + mask_i * phi(P_i, G_i, v)
e' = W_p vec(P') + W_g vec(G_canvas) + W_m mask + W_v v + b
```

`phi` is a small shared nonlinear module whose final layer is zero-initialized.
At initialization the native outputs are unchanged; the original fine path is
preserved. It tests whether local interactions make the existing 256D channel
more useful. A subsequent clean architecture need not retain both paths, but
removing the residual in the first pilot would confound the mechanism.

This belongs inside the belief update. It does not change the history-free
observed percepts or denoising targets. TSP still predicts before reading the
glimpse; the physical prediction never sees its target-frame fine evidence.
Real and imagined updates use identical fusion, location conventions and resets.
The actor continues to read belief, not a hidden world. Because `percept_embed`
also feeds outcome heads, better current-outcome prediction alone would not
establish a future-memory repair.

## 6. Short learned comparison before an overnight run

If the frozen tests show relevant conditional information, a bounded joint
continuation is the direct acquisition test. Compare:

- Unchanged continuation.
- Local nonlinear residual fusion before compression.
- A roughly parameter/compute-matched nonlinear alternative without the same
  local correspondence bias, if we want to attribute success to LOCAL fusion
  rather than simply extra nonlinear capacity.

Use the full P1 objective and train encoder, belief and denoiser jointly. Keep
starting checkpoint, optimizer state, learning-rate schedule, clipping, batches,
uniform fixations and existing loss recipe matched. Initialize the new route
without changing the old RNG stream. Log update norms; restored-learning-rate
effects must not be attributed to the new route. No guided curriculum, new
chooser, loss switch, wider state, corpus change or sampler change in this test.

Evaluate real future predictive effects, native high-noise denoising and sampled
semantics, alongside belief accessibility and the usual regressions. Improved
reconstruction/probe loss alone is insufficient. A paired informative-look
benefit after training is more diagnostic than a generic lower global loss.

An unchanged-versus-local comparison can establish benefit of the package; the
nonlinear control is needed for a stronger claim about spatial fusion's cause.
Neither a short null nor one successful seed establishes that the architecture
is universally sufficient. Fix the continuation budget before training and
return for discussion rather than extend it automatically.

If qualified, the overnight run should change the route only, keeping F1.5's
uniform gaze, objectives, sampler, corpus and declared budget. The learned
fixation chooser remains a separate future comparison.

## 7. Brain and machine-learning evidence: support and limits

- Ganmor, Landy and Simoncelli (2015) measure precision-dependent integration of
  peripheral and foveal orientation evidence across saccades, with systematic
  foveal overweighting. This supports correspondence-aware integration of
  coarse and fine evidence. Their sequential-saccade experiment does not dictate
  our same-frame implementation. Our two views share pixels/noise, so treating
  them as independent sensors would double-count evidence.
  <https://www.cns.nyu.edu/pub/eero/ganmor15.pdf>
- Williams et al. (2008) find task-dependent peripheral-object information in
  foveal retinotopic cortex, associated with discrimination performance and
  interpreted as feedback. This supports recurrent detail processing, not a
  proven single privileged memory route or the exact proposed fusion module.
  <https://web.mit.edu/bcs/nklab/media/pdfs/Williams.etal.N2008.pdf>
- FiLM demonstrates useful content-dependent feature modulation. A shared
  location-aligned application here is our adaptation; its visual-question
  answering experiments do not validate Saccade's memory mechanism.
  <https://arxiv.org/html/1709.07871v2>

The biological evidence makes the proposal reasonable inspiration. The native
conditional test and controlled learned route comparison determine whether it
addresses our measured failure.
