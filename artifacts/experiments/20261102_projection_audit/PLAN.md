# Frozen projection audit (Test 2)

2026-11-02. Authorized by the user after the spatial-fusion discussion. Test 1
(future-periphery conditioning) is deferred. No world, encoder, Mamba, sampler,
corpus or fixation-policy update. No architecture continuation or overnight run
is authorized by this protocol. Fix the protocol before fitting/readings.

## Questions and inference boundaries

The observed gap is raw fine -> weak accessible prediction through the fine
projection and exported memory. Distinguish projection rank loss, location/scale
geometry, coarse-plus-fine addition, and BF16 rounding. A successful invertible
recoding demonstrates information SURVIVED; a failed finite reader cannot prove
it was erased. Null-space energy alone does not establish task-relevant loss.

At a fixed fixation f, A_f is the 256x576 slice of percept_in for its nine fine
tokens in their exact canvas order. The complete input is e=A_f g+c(p,v,mask).
Given p,v,mask, subtracting c recovers A_f g in exact arithmetic. Thus addition
alone is not irreversible conditional on the baseline's known physical inputs.
Native Mamba, unlike this conditional instrument, receives only the summed input,
not a separate copy of the current coarse tokens. The subtraction control is
therefore a generous recoverability test, NOT a claim that native Mamba can
perform it or that joint coarse/fine compression is harmless. This inference
clarification was added after mechanical smoke, before prediction results.

## A. Exact mechanics and numerical audit

Use the frozen F1.5 ck10 SHA already pinned by F1. Hash all inputs and scripts.
For all 49 legal fixations:

1. SVD A_f, rank/condition/singular values and P_f=A_f^+ A_f. Check symmetry,
   idempotence, A_f P_f=A_f and reconstruction on synthetic retained/null vectors.
2. Report raw fine energy in P_f g and (I-P_f)g on natural train/dev frames.
   Values describe subspaces, not numbers of entities retained.
3. Compute full FP32 embedding and c with the actual percept_in and observed
   physical/fine inputs. Verify subtraction recovers A_f g numerically. Measure
   actual BF16/autocast embedding error against FP32, and separately against FP32
   accumulation with inputs/weights first quantized to BF16. This separates input
   quantization from accumulation/output rounding. Report error relative to the
   fine branch and on the decoded retained component, by fixed populations.
4. Compare cached FP32->FP16 fine contributions against newly computed FP32.
   Numerical perturbations do not establish semantic harm by magnitude alone.

Native coarse/fine cached observations originated under the declared runtime;
this is an input-interface audit, not an encoder-precision experiment.

## B. Matched held-out future-fine instruments

Reuse F1's exact train400/dev400 cached observations and next-fine targets,
baseline, two seeds (611701/611702), fit/validation/diagnosis/confirmation
partitions, natural frame sampling, 2,400 updates, batch512, AdamW lr.001,
wd.0001, clip1. Final weights only; no extensions after nulls. Future periphery
does not become a predictor input. Fine-target heads only, keeping the original
F1 fine-body/head initialization and optimizer dynamics for exact replication
where possible. Every supplement is padded to1024, same reader architecture.

Eight supplements:

- base: zeros.
- fine: whole raw9x64 crop in native crop coordinates.
- native: FP32 A_f g in256 coordinates.
- retained: A_f^+(A_f g), a576-coordinate recoding of EXACTLY the information in
  native (full row rank qualified mechanically). No targets/data are used to
  build this map. The lost null component is not restored.
- null: g-retained, complementary directions; usefulness here alone does not
  imply unique information unavailable from retained on the natural manifold.
- random_retained: a fixed shared seeded random rank256 orthoprojector applied
  to g, then represented in the same576 crop coordinates. A dimension-control,
  not a matched alternative trained architecture or proof of global sufficiency.
- mixed: actual BF16 mixed embedding, including coarse/mask/light/bias. Baseline
  knows the coarse/mask/light, so this is information-equivalent to native apart
  from numerical precision, but presents subtraction as a learned task.
- recovered_bf16: subtract exact known BF16-quantized offset from mixed, then
  apply A_f^+ (using the identically quantized weight/input convention). This
  tests whether actual numerical loss prevents useful prediction after decoding.

Keep FP32 supplements in FP32 for these instruments; do not silently compress
the audit's recovered values into FP16. Do not scale using development targets.
No new predictive module learns simulator labels. If mechanical qualification
fails, fix the mechanical bug before fitting and preserve its log.

Primary population = F1's night/entity-in-current-crop/no-next-scroll population.
Primary measure = next fine MSE over the whole crop. Secondary, declared before
fitting: next fine at current entity cells and their one-ring neighborhood;
day/entity, night/entity, night/empty, scroll and all-frame strata. Predictions
refer to the next frame at the CURRENT fixation, as in F1.

Use paired episode-bootstrap95% intervals, two fitted-reader seeds separately,
on diagnosis episodes0:250 and disjoint confirmation250:400. Report all arms,
including negative results. Compare each to base and compare retained/native,
raw/retained, recovered_bf16/retained, random_retained/retained. No final split.
Existing semantic probes remain earlier evidence; future latent forecasts here
do not establish actor benefit or prove a repaired native model.

## C. Predetermined readings and stopping boundary

- Retained rescues native's prediction deficit: information survived; a reversible
  location-aware change in representation makes it accessible. Compression alone
  cannot explain the rescued portion. This does not establish fine uptake by Mamba.
- Raw remains better than retained: an accessible residual advantage remains.
  This is compatible with rank loss or bounded-reader limitations; do not call
  it uniquely irreversible erasure. Null utility may also reflect redundancy.
- Random_retained works:256 linear measurements can support this tested capability
  with that representation. This is not proof that Mamba's entire256D mixed input
  has enough capacity for every task/history.
- Mixed worse but recovered succeeds: subtraction/geometry is an acquisition
  burden; it is not information-theoretic erasure by addition.
- BF16 recovery matches FP32 retained prediction: no demonstrated predictive
  harm from its numerical loss on this measurement. A small norm error alone
  would not suffice for this conclusion.

Stop after this finite battery, analyze, and return to the user to discuss a
repair. No learned route change, budget ladder or overnight launch follows.
