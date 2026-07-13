# Pre-registered: oracle repair + counterfactual action-fit microtest

Consensus basis: companion review of the fork-oracle run (all findings adopted;
senior concurrence with two additions — a non-privileged shift-copy baseline,
and a framing note that the "dynamics stack" claim named the action path and
predictor, not the backend alone). Companion's ordering adopted over mine:
microtest BEFORE budget probe. Step 4 (Mamba) deferred until the shared
action/predictor path passes minimal viability; if run earlier, it is labelled
a diagnostic of the failing stack, not a thesis verdict (user directive to test
Mamba stands and is best served unconfounded).

## Part 1 — oracle repair and archive (companion spec, verbatim)

Re-run the fork oracle with:
- raw bundle saved and hashed BEFORE aggregation: anchors (frames), suffixes,
  all true/alternative branch frames, task outcomes, per-branch raw-RGB masks;
  snapshots pickled where the simulator permits;
- stratified anchors: day AND night (~1/3 night, matching daylight<0.5), plus
  an interaction-tagged subset (anchors where the live rollout's next 8 steps
  changed inventory/health/achievements);
- per-branch S3-style changed masks (union reported as secondary);
- leave-one-branch-out oracle estimator;
- both estimands reported (mean-of-ratios and ratio-of-means) with cluster
  intervals labelled screening-grade (4-6 env seeds);
- all-pairs task divergence plus per-anchor incidence;
- instrument tests: synthetic oracle formula check, RNG replay bit-exactness,
  action-suffix indexing, mask aggregation.

## Part 2 — counterfactual action-fit microtest (the discriminating experiment)

Dataset: from ~32 snapshot anchors (fresh, saved raw), pair each identical
prefix with 4 distinct action suffixes × 3 RNG branches each (fork machinery).
Split anchors 24 train / 8 held-out.

Train the CURRENT GRU+predictor stack (frozen step-1 encoder, standard losses,
rollout=1) on the training anchors' windows only; evaluate:
- training-anchor fit (can the stack overfit known futures?);
- held-out-anchor prediction;
- correct-vs-shuffled suffix loss separation on both splits;
- k=1 and k=8 copy margins, uniform-token and changed-token readouts.

Pre-registered interpretation table:
- cannot overfit training anchors → implementation/action-injection/capacity
  failure (proceed to predictor-controls matrix, skip budget);
- overfits, does not generalize → data/regularization/capacity (budget probe
  de-prioritized; predictor matrix with regularization arms);
- generalizes AND separates correct-from-shuffled → ordinary budget becomes
  the leading hypothesis (staged budget probe: resume seed-101 rollout-1
  checkpoint from saved optimizer/RNG state, evaluate at 8k and 16k with
  action-attribution logging; promote to 3 seeds only on material movement).

## Part 3 — shift-copy baseline (senior addition; non-privileged reachability)

Predict the k=1 latent by programmatically scrolling the CURRENT frame
according to movement-action geometry (7-px tile shift in the action
direction; leading-edge column/row left as-is), encode with the frozen
encoder. No simulator access. Report copy margin on changed patches overall
and on move-succeeded anchors (validated offline against branch data).
Readout: if shift-copy clears the ≥5% copy margin at k=1, a deterministic
function of current observation + action clears S3-A's per-step bar without
privilege — closing the "current inputs may not suffice" objection at the
gate's level. (k=8 composition left to the model.)

## Also in this round

- m3_hjwm/ARCHITECTURE_SPEC.md status header reconciled (step-1 pass; rollout
  bridge status) — flagged twice by the companion.
- Motion-weighted loss stays LAST, correctly labelled (novel adaptation, no
  Dreamer-4 precedent), uniform control mandatory, HUD/register regression
  guarded — only after the above.
