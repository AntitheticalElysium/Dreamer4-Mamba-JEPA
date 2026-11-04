# Test 2: much of the apparent input loss is reversible accessibility

2026-11-02. Completed authorized frozen audit. No native-model training or
architecture change, no Test 1, no chooser or overnight launch. Protocol:
[PLAN.md](PLAN.md); code: [audit.py](audit.py), [analyze.py](analyze.py).

```
Prediction benefit at current entity cells, disjoint confirmation cohort:

Raw fine evidence                         26.4 / 25.9%
          |
          v  native576->256 fine projection
Projected fine vector                      2.7 /  2.8%
          |
          v  reversible per-fixation recoding, NO null component restored
Retained evidence in crop coordinates     19.0 / 22.0%

Actual BF16 full mixed input               0.8 / -0.9%
          |
          v  subtract KNOWN coarse contribution, decode retained component
Recovered from actual BF16 mixture         18.5 / 22.3%
```

Numbers are percentage future-fine MSE reductions versus the identical
coarse/history baseline, two fitted-reader seeds. They are NOT percentages of
mutual information or native-model success. Future targets are next frame at
the CURRENT fixation, as in F1; future periphery is not supplied.

**Finding:** a substantial part of the raw-to-projection prediction gap can be
recovered without restoring any discarded information. It is therefore wrong
to interpret that entire gap as irreversible loss at the fine projection.
The representation/reader interface demonstrably matters. The native Mamba's
acquisition/retention failure remains unresolved: it receives the full mixture,
and does not have the instrument's separate known coarse tokens for subtraction.

## What ran and why the comparison is informative

- Frozen F1.5ck10 SHA
  `95fa2cc82cb9dc3cae11cbf9f44f880aff17e250de639911f39d159564978128`.
- Exact F1 train400/dev400 observations, targets, natural-frame minibatches,
  two seeds611701/611702,2,400 updates, batch512, AdamW lr.001/wd.0001,
  clip1, same fine-head initialization. Eight supplements x two seeds=16 fits.
- Fit first360 train episodes, validate next40, excluding world-monitor
  episodes. Diagnosis dev0:250; disjoint confirmation250:400. Final split unused.
- Primary population: night, entity in current crop, no scroll on next step.
  Diagnosis2,628 frames/164 episodes; confirmation1,824/102. Source-cell and
  neighborhood readings are predeclared secondary checks against background-only
  improvement. Simulator labels select reporting masks, never optimization targets.
- All readers have the same architecture and1024 padded supplement dimensions.
  Only current representation changes; no future information enters an input.
- Complete numerical/fit/evaluation process262.6s; peak2.89GiB. Mechanical smoke
  and CPU bootstrap analysis additional. World checkpoint hash unchanged.

The base and raw-fine control predictions reproduce F1 **bit for bit** for BOTH
seeds: maximum per-cell error-array difference0.0, not just equal printed means.
The native arm here uses FP32 contributions; F1's input arm used cached FP16
contributions. Those two arms are not claimed bit-identical.

The recoding is `r=A_f^+(A_f g)`, using the exact frozen fine projection at
fixation f. It outputs576 crop coordinates but has only256 independent degrees
of freedom. Since A_f has full row rank, `A_f r=A_f g`: the recoding is invertible
on the projected representation. It adds no discarded null component, target
information or fitted semantic labels. Unlike the native projection coordinates,
it restores alignment with crop-relative fine coordinates.

This is a strong SURVIVAL control: positive recovery proves information survives.
It is not merely a second failed probe. It does hard-code a fixation-dependent
inverse; the result does not separately identify scale conditioning, sample
sharing, or the learned reader's difficulty with location-dependent functions.

## Full results, including residual deficits

All entries percentage MSE reduction relative to base; seed1 / seed2.

| Supplement | Diagnosis whole crop (primary) | Confirmation whole crop (primary) | Diagnosis entity cell (secondary) | Confirmation entity cell (secondary) |
|---|---:|---:|---:|---:|
| Raw fine |40.52 /43.87|38.90 /42.19|26.35 /26.90|26.41 /25.94|
| Native projected fine |5.24 /7.85|4.55 /6.85|3.24 /3.84|2.72 /2.82|
| Recode retained component |35.41 /40.44|33.25 /38.53|20.21 /22.85|19.03 /21.98|
| Discarded null component alone |27.59 /29.66|25.70 /27.07|13.32 /13.28|12.57 /12.15|
| Shared random rank256, crop coordinates |35.35 /40.09|33.28 /37.86|23.23 /24.88|21.99 /23.78|
| Actual BF16 mixed input |−0.39 /1.71|−0.20 /1.01|1.50 /−0.19|0.78 /−0.85|
| Recover retained from actual BF16 mixture |35.49 /39.40|33.45 /37.49|19.47 /22.60|18.53 /22.31|

Paired episode-bootstrap95% intervals, confirmation entity-cell reading:

- Raw:26.41%[24.54,28.19],25.94%[23.21,28.25].
- Native projected:2.72%[1.72,3.61],2.82%[2.11,3.59].
- Retained recoding:19.03%[17.64,20.23],21.98%[20.61,23.26].
- Recovered BF16:18.53%[17.16,19.71],22.31%[20.86,23.68].

Recoding improves error OVER the native projected reader by16.77%[15.43,18.11]
and19.71%[18.37,20.97] at these cells. This improvement confirms in both reader
seeds and disjoint cohorts; it is not sampling noise in one selected scene.

### The remaining raw-versus-retained gap is real in these instruments

Raw still beats retained in both seeds/cohorts. Confirmation entity-cell error
is9.11%[7.50,10.90] and5.08%[2.71,7.26] lower with raw than retained. The raw-fine
gain versus base exceeds retained by7.37 and3.96 percentage points, respectively.
This remaining gap is compatible with discarded useful directions and/or finite
readout limitations; the experiment does not establish irreversible loss as its
unique explanation.

The null component alone predicts well, but retained and null can carry redundant
information on the natural data manifold. Its13-point benefit must not be added
to retained's benefit or called13% unique lost information.

The random256-dimensional subspace also retains useful forecasting capability.
This weakens an assertion that256 linear measurements cannot carry the tested
fine information. It does NOT prove that one256D vector is sufficient to carry
all coarse/fine/self-state content or long-term memory, and random versus native
also changes projection directions and whether the projection varies by fixation.

## Numerical audit

All49 legal fixation slices rank256; condition numbers12.92–14.55. Synthetic
double-precision right-inverse error max2.89e-15; symmetry8.67e-16;
idempotence1.39e-15; projected null residual3.51e-15. Shapes, current-crop ordering,
whole-crop supply, current-only transformations and target exclusion passed.

On natural primary frames, retained and null components each contain approximately
50% of raw fine squared norm. This is energy, not semantic information content.

Confirmation numerical errors:

| Measurement | RMS relative to indicated fine scale |
|---|---:|
| FP32 full sum minus known coarse contribution versus isolated fine |0.0000655% of fine-branch RMS|
| Actual BF16 mixed sum versus FP32 sum |1.95% of fine-branch RMS|
| Actual BF16 versus FP32 accumulation with quantized weights/inputs |1.85% of fine-branch RMS|
| Recovered BF16 component versus FP32 retained component |5.29% of retained RMS|
| Cached FP16 fine contribution versus fresh FP32 contribution |0.0227% of fine-branch RMS|

Precision is not literally lossless. More importantly, recovery from the actual
BF16 mixture still gives18.5–22.3% entity-cell improvement versus19.0–22.0% from
the FP32 retained component. It cannot explain the much larger collapse to the
native vector's approximately3% accessible gain on this test. On the primary
whole-crop score, BF16 recovery differs from FP32 retained by+0.20/−1.04 baseline
percentage points in confirmation; the seed2 penalty is statistically detectable.
Do not turn this into an exact no-effect/equivalence claim. This audit does not
test rounding amplification throughout Mamba's full recurrent trajectory.

## What this says about a repair

Demonstrated: the native fine projection contains substantially more usable
predictive information than the previous reader exposed. The input geometry
and location-dependent encoding are a real obstacle for the bounded reader.
The full mixed input poses an additional acquisition burden even when our
reader has coarse tokens separately. With known subtraction and decoding, much
of that burden is removable. Native Mamba does not receive those separate tokens.

Unresolved: whether native Mamba's weak fine uptake is mainly joint compression,
learning an appropriate local interaction from the mixed vector, weak future-loss
pressure, or a combination. The audit did not train a replacement route and cannot
claim that spatial fusion already repairs the world model.

The evidence supports testing shared local nonlinear fusion BEFORE compression:
it can build a content-dependent, spatially aligned correction while both views
remain explicit, avoiding the need to discover that interaction from a changing
global code. Preserve the original fine residual during a function-preserving
pilot. Do not implement the pseudoinverse as a production repair: it is a frozen
diagnostic using the current weights, not a justified joint-training architecture.

We do not have evidence that merely increasing256D width or switching precision
addresses the dominant measured gap. The next acquisition test, if authorized,
is a short matched full-objective joint continuation of unchanged versus local
fusion, with a matched nonlinear control if attributing benefit to locality.
No such continuation was launched automatically. Return to discussion here.

Artifacts: [results.json](results.json) contains all populations/intervals and
numeric readings; [checks.json](checks.json), [identity.json](identity.json),
fit curves, all16 predictor states and per-cell errors are retained. The base/raw
replication checks are included in results.json. A manifest binds sources/results.
