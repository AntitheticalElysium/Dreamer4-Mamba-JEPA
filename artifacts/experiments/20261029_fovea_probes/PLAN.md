# Fovea probes on the frozen P1 world (2026-10-29, before any run)

## What can and cannot be observed
Nothing in M0 chooses where to look. P1 trained with fixations from the uniform prior over the 49 window centres
(data.fixation_prior; STEP_SPEC 7); P2 keeps them (CURRICULUM: "fixations come from the random prior"); P3 uses the
same random controller for actor and BC (m0.yaml p3.fixation; TEST_SPEC S3.1); learned fixation is the stage after S3
(TEST_SPEC: "anticipation vs reaction; night robustness"; DECISIONS: placement learned from value of information).
So no fixation BEHAVIOUR can have emerged. What the frozen world can show is whether it offers what a learned
fixation would need: (F1) it keeps what the fovea saw; (F2) where looking would pay off is state-dependent and tied to
consequences rather than to pixel noise. Both are readings of the world's own heads and loss, descriptive, and select
no architecture.

The random prior makes P1 data a randomised experiment: the fixation at each frame is drawn independently of the
state and of the agent's action (the collector acts on symbolic observations), so contrasts between fixated and
unfixated regions are not confounded by content.

Sources: value of looking = expected reduction of task-relevant uncertainty (Sprague & Ballard, NIPS 2003; Najemnik &
Geisler, Nature 2005; Williams 2505.01249: expected information gain); gaze follows task goals rather than salience
(Hayhoe & Ballard, TICS 2005); prediction-error seeking is captured by unpredictable noise ("noisy TV", Burda et al.,
ICLR 2019); trans-saccadic prediction of foveal input from the periphery (Herwig & Schneider 2014). Measured: at the
3 x 3 periphery acuity, presence / kind / terrain / HUD values are near-exact by day, the fovea's added input is at
night (PERCEPTION, acuity table); 99.8% of immediate consequence cells are the player's 4 neighbours + HUD
(READING, consequence_locality).

## F1. Does the world keep what the fovea saw? (trans-saccadic retention)
- Claim: the belief retains glimpse evidence after the eye moves: the TSP head (glimpse pixels predicted from the
  predicted belief, the periphery and the lighting latent BEFORE the glimpse is read) is better on world cells that
  were inside the glimpse k frames earlier than on cells that were not.
- Competing: no retention (equal NLL); or a positional artefact (window coverage depends on screen position).
- Control: within screen position x day / night strata, fixated-at-(t - k) vs not, both seen by the periphery; the
  camera shift from the player's movement is undone (map cells keyed by world position, HUD cells by screen position).
  Input ceiling: the fovea reconstruction head on the glimpse itself (what the glimpse holds beyond TSP's inputs).
- Readout: retention gain (nats per pixel) by k = 1..8, day / night, content (entity / terrain / HUD); episode-bootstrap
  SE.
- Outcomes: gain > 0 beyond 3 SE -> retained, with its decay; ~0 while the input ceiling shows information in the
  glimpse -> the belief does not keep it; both ~0 -> the glimpse adds little beyond the periphery at this acuity.

## F2. Where would looking pay off, according to the world? (value of looking)
- Claim: the value of a fixation for the world's own one-step prediction depends on the state (more than on the
  position) and concentrates on consequential content.
- Competing: (a) no value anywhere; (b) a fixed best position (state-independent); (c) value follows pixel
  unpredictability (night static) rather than consequences.
- Measure: for a frame t, the 49 counterfactual glimpses of the SAME real frame (only frame t's fixation changes; the
  history keeps its recorded random fixations) enter the belief and the denoiser's context; value(L) = mean over L' of
  the one-step denoising loss of the real frame t + 1 minus that loss under L. Loss = the G1 objective on the
  PERIPHERY and lighting tokens of t + 1 (the glimpse tokens of t + 1 are excluded: overlap with the next fixation
  would reward looking where the eye goes next), at the 8 monitor noise levels, common random numbers across the 49
  fixations, two independent noise sets A / B.
- Readouts: (R1) mean value map over the 49 centres, day / night; (R2) cross-validated state dependence: the best
  fixation per frame chosen on A, scored on B, minus the best fixed centre chosen on half the episodes and scored on
  the other half with set B (and the reverse); (R3) value by window content (zombie / cow / skeleton / arrow / none,
  HUD, the faced cell) x day / night, position-matched; (R4) anticipation vs reaction: windows holding a zombie that
  attacks at t + 1 vs one that attacked at t (counts reported, likely small); (R5) value vs the glimpse's input
  information (TSP NLL - reconstruction NLL at L), correlated across L within frames, in night windows with no entity
  (static only) vs with an entity.
- Samples: random dev frames plus a separately reported night-with-a-mob-in-view set (never pooled into averages).
- Outcomes: (a) value ~0 everywhere -> the world gains nothing from where the fovea looks (with F1: not kept, or
  nothing to add); (b) value > 0 but R2 ~0 -> a fixed position would do, nothing to learn; R2 > 0 beyond 3 SE with
  value on consequential content -> the substrate for learned fixation exists in this world; R5 high on static night
  windows -> value tracks pixel noise (the noisy-TV failure).
- Not established: that a learned controller would find it, or that the world's value matches value in the
  environment (it is the world's own, the quantity imagination-trained fixation would optimise).

## Addendum (10-29, after a smoke test of analyse.py on the first 50 episodes, before the full run ended)
The smoke test showed a NEGATIVE lag-1 retention gain (TSP worse on cells fixated one frame earlier: day -0.10 (0.05),
night -0.06 (0.02) nats per pixel). Before reading the full data, a descriptive split is added: static camera between
t - 1 and t with the cell's content unchanged (no entity at that world cell at t - 1 or t; HUD values unchanged), and
camera moved. A negative gain on static unchanged cells cannot be stale memory of changed content; a negative gain
only when content changed or the camera moved can. Neither split changes the declared F1 readouts.
