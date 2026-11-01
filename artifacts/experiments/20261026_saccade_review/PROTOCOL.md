# Independent saccade S0 review

Reviewed checkout: `4c615dc4`. Preserve the package, committed results and pre-existing work. All new outputs go here; disposable runtime dependencies go in `/tmp`.

## Questions declared before experiments

1. Reproduce the six S0 modules from their current package entry points using identical saved inputs, into separate files. Compare numerical arrays/JSON, including floating-point differences. Independently verify replay against stored pixels on a stratified sample.
2. Test statistical rejection at zero estimated variance. A deterministic difference must not silently pass.
3. Test the claimed memory interpretation of the reward information ladder: hold input information and data splits fixed, change only the capacity of the memory-free learner. Improvement below the published R2 value without memory refutes the stated implication; no improvement does not establish a Bayes limit.
4. Check whether supported metrics distinguish trajectories with identical counts/centroids but different spatial arrangements, and whether the operational pass/fail families reject each declared corruption. These tests evaluate harness sensitivity, not world-model learnability.
5. Check observation ownership in revisit comparisons, train/test episode grouping in the two-sample classifier, probe calibration and source/result provenance.

## Boundaries

- No model training for the proposed architecture, final-split access, changes to canonical design, or edits to the reviewed harness.
- Existing dev/train diagnostic data only. Simulator state remains evaluation-only.
- GPU probe reproduction is authorized; measure actual runtime/memory. Avoid expanding into unrelated baseline campaigns.
- Record negative and inconclusive evidence. Reproduction is separate from correctness and architectural success.

## Bounded follow-up after the capacity check

The predeclared 127-leaf memory-free model did not improve on the 31-leaf reference (dev NLL 0.02557 vs 0.02434). Preserve that negative result. Test one additional, explicitly different axis: add counts from the entire visible current frame pair to R2, keeping the 31-leaf learner, split and stopping rule fixed. Only observable entity kinds and unoccluded terrain are counted; no achievements or older history are supplied. This checks whether the published current-input reference covers the information in the model's percept, rather than tuning the first comparison until it improves.
