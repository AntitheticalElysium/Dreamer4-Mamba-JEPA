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

## Results

### Part 1 — repaired oracle (fork_oracle_v2.json; raw bundle archived, sha256 6e1a3add…)

R-A survives every correction: per-branch masks + LOO estimator, mean-of-ratios
88.6% (cluster 95% screening [80.5, 95.2]), ratio-of-means 91.0%; night stratum
(16/48 anchors) 89.5/92.9% — night does not rescue the ceiling; interaction
stratum 91.4%. Registers: copy 0.089 vs LOO oracle 0.0022. All-pairs task
divergence with per-anchor incidence: **50% of anchors show some reward
divergence**, 31% health, 25% inventory, 21% achievements (mean all-pairs
rates 6.5–15.5%) — consequential stochasticity has substantial incidence,
correcting the v1 "modest" framing.

### Part 3 addition — shift-copy baseline (non-privileged)

Scrolling the current frame by the movement action's tile geometry and encoding
with the frozen encoder: **+30.3% relative improvement over copy at k=1** on
move anchors (n=9, screening-grade; 7/9 all-branches-moved). A deterministic
function of current observation + action clears the 5% bar six-fold without
simulator privilege — the "current inputs may not suffice" objection is closed
at the per-step level.

### Part 2 — counterfactual action-fit microtest (microtest_v1.json)

24 train / 8 held anchors, 4 suffixes × 3 branches each, 2,000 updates,
frozen encoder, rollout=1, reward/continuation weights 0:

| readout | train anchors | held anchors |
|---|---|---|
| teacher-forced JEPA | 0.0366 | 0.0395 (tiny gap) |
| open-loop k=1 copy margin | **+0.0030** | −0.0196 |
| open-loop k=8 copy margin | **+0.0140 (≈12% relative)** | −0.0094 |
| action separation k=8 (shuffled − correct) | **+0.0043** | +0.0016 |

**FILED INTERPRETATION WITHDRAWN (2026-07-13 companion audit; decisive control
verified by senior reproduction).** My "middle cell / diversity-vs-memorization"
diagnosis was refuted by the control I failed to run: evaluating the existing
full-scale S3 checkpoints (which never saw fork seeds 21-24) on the identical
evaluator. Verified to four decimals: S3 seed-303 scores **+0.0289** on the
microtest's own training anchors and **+0.0052** held-out — better than the
microtest model on both. Full checkpoints also sit at ≈chance (25-29%) on
four-way same-anchor suffix retrieval, as does the microtest model. The
replacement conclusion (companion wording, adopted):

> Current models learn some above-copy latent evolution, but neither the
> microtest nor the full-scale runs demonstrate held-out counterfactual action
> selection.

Additional companion findings, all accepted: the microtest silently discarded
every night anchor (split took the first 6+2 of each seed's day-ordered 12);
chronological within-trajectory splits; only the `true` suffix evaluated;
rollout_steps=2 trains a 2-step bridge while the eval is 8-step (its one-anchor
control shows the action path CAN fit a partial suffix mapping when horizon-
matched: 50% 4-way retrieval, +0.055 separation); "budget de-prioritized" was
unsupported (exposure confound: 30.5 vs 5.6 replay-equivalents; Dreamer-CDP
trains at 1.1M steps, ratio 512); no pre-registered numeric definition of
"overfits". The shift-copy claim is downgraded to its defensible form: a
Crafter-specific action transform beats copy on a small movement subset (5/9
movement anchors improve; k=1 only).

**Oracle v2 numbers corrected** (LOO errors were branch-averaged before
per-branch masks; per-branch fix verified, companion's numbers reproduced
exactly): mean-of-ratios **85.4%** [76.0, 94.2], ratio-of-means **88.1%** —
conclusion unchanged (bar reachable), filed estimates superseded.

## THE GATE PROBLEM (root cause of apparent non-progress, both agents concur)

S3-A measures dynamics fidelity (beat static copy), which an action-agnostic
predictor can pass. The unresolved question is narrower and now well-defined:

> **Can an architecture learn the correct same-state counterfactual future
> from the action sequence, out of sample?**

This becomes the next gate (Dreamer 4's own source logs an action-shuffle loss
ratio — the faithful precedent: nicklashansen__dreamer4 train_dynamics.py:831).
Copy margin is retained as the dynamics-fidelity gate alongside it.

## Agreed next sequence (companion's, senior-concurred; supersedes my matrix)

1. Oracle report corrected and reissued (DONE above; per-branch masking test
   added; snapshot archiving + RNG-replay/suffix-index regressions to follow).
2. Matched causal-action probe replacing the microtest: all four same-anchor
   suffixes; environment-seed splits; day/night + action-effective strata;
   horizon-matched training (explicit k=8 arm); 4-way retrieval + true-vs-wrong
   ranking + matched separation; trained no-action and shuffled-action
   controls; several inits; cluster uncertainty; numeric definitions
   pre-registered BEFORE running.
3. Re-evaluate existing checkpoints under the causal metrics first (nearly
   free).
4. Staged 4k → 8k → 16k continuation tracking updates AND replay-equivalent
   exposure; stop if causal metrics stay at chance.
5. Smaller architecture matrix WITH an early pooled/global causal baseline
   (Dreamer-CDP/LeWM-shaped) — the 66-independent-recurrent-streams temporal
   factorization is itself a major untested hypothesis no cited system uses;
   AdaLN-zero as a replacement (parameter-matched), depth control, compact
   unified feature/action causal arm if VRAM permits.
6. GRU vs Mamba-2 on the first arm with reproducible held-out action selection
   (user directive standing).
