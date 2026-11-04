# Proposed test: coarse-history redundancy versus native-belief utility

2026-11-03. User approved proceeding, including measuring1B's real performance cost
alongside the ablations. No production input/controller adopted. P1-F1.6 performance
remains outside this comparison.

**Question.** Does native belief add more when the cheap head lacks its direct
eight-frame history? A positive interaction supports redundancy/substitutability.
It does not by itself prove that the eight-frame head was using information
inefficiently, that belief is sufficient for all required predictions, or that
the one-frame version is the better chooser.

**Four matched arms, fixed F1.5 checkpoint10.**

| Direct coarse history | Capacity-matched coarse-only auxiliary control | Native belief auxiliary input |
|---|---|---|
| Eight frames | 8C | 8B |
| One frame | 1C | 1B |

Use E3's118032-parameter AuxHead and input shapes for all four arms. In1C/1B mask
the older seven slots of BOTH spatial coarse tokens and context (light, prior
actions, rewards, termination and validity) to zero before the head. Keep the
last-frame copy residual identical. Older raw context must not quietly survive
the visual-history ablation. This keeps shapes, parameter initialization and
arithmetic comparable; it does not claim equal effective information or usable
parameter count across history lengths. A compact one-slot implementation and
its timing would be a separate deployment check, not a scientific input change.

The control auxiliary vector is the SAME fixed256D random sketch of the last
coarse grid used by E3. It provides no extra historical information and keeps the
global route active in both arms. The belief input is E3's native corrected
b[t-1] from the full real prefix with seeded uniform glimpses. Train-only channel
standardization and256->64 global broadcast remain unchanged. Thus the primary
contrast is the E3 capacity-matched baseline, not an unlabelled extra-parameter
comparison against the smaller original101584-parameter head.

**Timing.** Predict p[t] from p[t-1] (or p[t-8:t]), prior context and queried
a[t-1]. Belief must be BEFORE that queried action. The actual and NOOP forecasts
receive identical belief/history. No p[t], reward[t], termination[t] or fine crop[t]
enters forecasting. Only surprise subsequently compares the forecast with arrived
coarse p[t], before fixation. One-frame does not mean giving the forecast its target.

World/encoder/belief are frozen, auxiliary features detached. No head-loss gradient
trains Mamba, so a bypass here cannot itself damage the main belief representation.
It can reduce this readout's incentive to use that representation.

**Proposed budget.** Two head seeds611601/611602, all four arms, same initialization
and per-example minibatch draws within seed;12000 updates each, B256, existing
12000-step warmup/cosine/AdamW/clip recipe, natural training frequencies and original
monitor exclusion. Snapshots0,1000,4000,8000,12000, final weights primary. Run all
arms to the fixed end, regardless of intermediate signs. Eight fits should be
roughly25-35 minutes based on prior head-only runs; this is an estimate requiring
a smoke benchmark, not a promised ETA. Original E3's4k fits have a different LR
schedule and cannot stand in for the new8C/8B controls at12k.

**Readings.** Reuse identical saved and disjoint confirmation contexts, plus their
natural random-frame subsets. For each seed and cohort report:

* Conditional value of belief: L(1C)-L(1B) and L(8C)-L(8B), and their paired
  interaction. Report absolute MSE changes as well as relative changes; a larger
  percentage improvement on a much worse baseline is not sufficient.
* Practical replacement:1B versus8C and8B, not only1B versus1C.
* Factual error at actual counter cells, first/later increments, moving/nearby
  entities and edited/entering terrain; sparse outcomes retain exposure counts.
* Combined-policy crop coverage for drink/wood/stone/coal/iron/sapling, contact
  sites, entities by kind, entering content and day/night. Individual C/S diagnose
  the combined policy; perfect immediate C is not an adoption requirement.
* All fixed existing policies remain available for descriptive comparison. No
  C/S weights, exploration rates, temperatures, seeds or checkpoints are selected
  on development performance. Head competence and main-world competence remain
  separate; broader entity targeting is not closed-loop teaching evidence.

Use paired episode bootstrap intervals, keep seeds separate, and preserve negative
results. Do not average away a lost coal/iron or entity capability. Inspect the
learning trajectory because earlier4k acquisition was inadequate in some fits;
do not silently extend a negative12000-update budget.

**Interpretation before results.**

*1B beats1C but loses to8C*: belief compensates for missing history, but replacing
the current head is not supported. *1B preserves8C while1C regresses*: belief is a
useful substitute under this fit. *8B also benefits with the longer recipe*: the
earlier4k result was not the final test of belief utility; short training and
redundancy are distinct explanations. *No belief gain*: this route/training recipe
remains a negative, not proof that no useful information exists in Mamba. This
design does not separate long history, fine evidence and representation accessibility.

**Checks before any fit.** Identical initialized weights/minibatches/target samples;
masking covers all older context channels; changing masked head inputs cannot
change1C/1B outputs at fixed auxiliary vector; no current/future leakage; prefix and
reset alignment; actual NOOP C=0; gradients confined to the head; immutable world/
cache/checkpoint hashes; snapshots do not consume optimizer RNG. Save forecasts,
scores, counters, metadata, weights and optimizer states to avoid replaying fits.

**Runtime and production boundary.** Native b[t-1] contains the previous selected
fine glimpse. A fully fresh belief-conditioned chooser creates the dependency
gaze[t-1] -> glimpse[t-1] -> belief[t-1] -> gaze[t], even with a tiny readout. The
older native-collector benchmark separated this sequencing cost from forecast
cost (../20261102_foveal_information/CHOOSER_RESEARCH.md). Do not attach its full
22.5-hour native-scoring estimate to this new head; its actual cost is unmeasured.
Likewise, do not claim that using cached uniform-prefix beliefs in this diagnostic
solves production own-history collection. If useful, benchmark the real path before
adoption. A block-start belief anchor is a previously documented approximation,
not a transparent optimization or an automatically adopted fallback.

**Architectural context.** A shared belief with small downstream heads is coherent:
DreamerV3 eq1 reads reward/continuation/observation predictions from its shared
model state (https://arxiv.org/html/2301.04104v2). That precedent does not guarantee
Saccade's detached auxiliary readout agrees with its diffusion generator. Sharing
the information source reduces duplicated state estimation; the cheap head still
learns its own prediction mapping and can disagree about action effects or errors.

**Approval/implementation note, before fitting.** User approved proceeding and
suggested extending8B. Inspection confirms original E3 files save weights/auxiliary
normalization only, no optimizer/RNG state; their4k-decay schedule differs. Preserve
those runs as historical evidence. Fit ALL four arms on the common12k clock rather
than call a mismatched continuation a controlled comparison. Save optimizer/RNG
states at every new snapshot. No overnight world training is authorized by this test.

**Timing protocol, before measurements.** Measure compact one-frame inference
(algebraically folding away the masked columns), then the REAL fresh-belief loop:
batched coarse perception cache -> head actual/NOOP -> selected fine crop -> native
two-token Mamba update. Batch across streams, never loop individual streams and
never include native denoiser scoring.1C chooses fixations in a batch and requires
no extra collection-time Mamba pass.1B refreshes belief at EVERY frame; no stale
anchor, uniform-history replacement, thinning or reduced frame budget. Test native
reference/Triton implementations and a bounded compiled reference step if feasible;
verify numerical equivalence before treating it as an optimization. Report head
milliseconds separately from complete collector wall time. Use actual P1 shapes
16x64 and4x256 and asynchronous episode resets. Compare ordinary P1 forward/backward
under each collected chunk as well as collector-only time; no optimizer/world update
or checkpoint is retained. Preserve equal data, reset/carry and main training code.
Report medians/range over fixed repeat counts, warmup/compilation separately, and
extrapolate added hours explicitly from measured per-frame overhead, not the older
expensive-chooser estimate. Optimizations may change arithmetic only, not inputs or
when fine evidence arrives. Failure/timeout of compilation is recorded, not evidence
that further optimization is impossible. Total one-shot wall cap45min; no other job
is killed. Discuss completed readings before any further campaign or world launch.
