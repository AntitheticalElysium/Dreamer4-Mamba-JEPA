# Bounded matched continuation: 3,000 to 6,000

2026-10-14. User-authorized budget extension of the completed paired-event study.
Saved before continuation implementation or optimization. Exactly two existing
arms: direct energy generator and paired event editor. No new scientific arm.

Restore the complete shared3,000-update serialized state: both worlds, Adam
moments and groups, window/noise generators, global torch/CUDA RNG states,
ledger, history and global update. Keep frozen encoder, original TRAIN pool,
architecture, first-power energy objective, batch16/L16, precision, data/noise
matching, optimizer and current learning-rate rule unchanged. Global updates
3,000..5,999 retain LR1e-4. No warmup restart, loss change, normalization fix,
source routing change, new masks, reward labels or simulator training labels.

The entering-cell fault remains independently diagnosed and unmodified. Excluded
final outputs have no scored loss gradients in this teacher-forced recipe.

## Endpoint and evaluation

- Hard stop at6,000 total updates; never extend automatically.
- Save model/optimizer/RNG/ledger atomically every25 updates, and snapshots at
  4,500 and6,000. Evaluation at4,500 is descriptive;6,000 is the fixed endpoint.
- Preserve the original3k contract, all source/checkpoint/evaluation bytes and
  its`coherent_completion_not_established` result. Use a separate continuation
  artifact subtree and contract, binding original state and source hashes.
- Use exactly the same exploratory cohort,64 samples/root,batch4, original
  matched noise seed rule,32 real futures, two frozen readers and2,000-draw
  paired episode/root intervals. Keep invalid draws in all denominators.
- Preserve original numerical coherence/movement/joint-law/static/reader margins.
  A derived criteria document changes the final checkpoint to6k and declares
  lineage, while retaining every original numeric rule and original criterion hash.
- Report at both milestones: validity, movement overall and by species, source
  disappearance, source-retained duplication, multiple destinations, camera/static
  fidelity, joint occupancy and source/destination event distributions, valid
  semantic diversity, reader disagreement and marginal-preserving128-shuffle
  dependence. Compare to3k and across arms with the same paired root intervals.
- Retain sampled latent variance/selectivity scope; no adaptive-compute or
  four-way uncertainty separation claim. No H16 or actor run.

## Interpretation and stop rule

The question is whether more unchanged optimization improves compatible sampled
transitions. Lower energy alone is insufficient. Declining duplication offset by
more disappearance is incomplete coherence. Require movement and joint-law gains
to coexist with fewer invalid outcomes; report both readers and small-root limits.

At6k explicitly report each original coherence condition, what improved and what
deteriorated. If negative, preserve it and return to broader representation,
objective and transition principles; do not extend this decoder campaign again.
Partial learning is not a successful repair. A negative does not establish
architectural impossibility or uniquely isolate its upstream cause.

No automatic second seed, fresh-data collection or architectural follow-up is
authorized by this launcher. A promising exploratory result is reported for
review; the existing fresh reservation remains unopened.

Before main continuation, verify a small split-resume against uninterrupted
optimization from the full3k state, exact sampler/noise streams and same model/
optimizer endpoints within the original execution tolerance. Verify new evaluation
reproduces a frozen3k batch at identical shapes and noise. Logs go exclusively to
`artifacts/eda/levers_logs`; agents record concise experimental status in NOTEBOOK.
