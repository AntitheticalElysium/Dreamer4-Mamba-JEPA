# Independent audit of the F4 health-output claim

User request, 2026-11-07: establish why zombie damage is under-produced; adversarially audit RESULTS 12, without
interrupting Claude. No training run or canonical code change. This is one bounded diagnostic, not a repair search.

## Claims to separate

1. A hit classifier at the final residual establishes accessible predictive information, not a correctly calibrated
   health-transition representation. AP differences and ratios are not information quantities.
2. `head` is final RMSNorm, action/noise-dependent affine modulation, then a shared linear projection. Test each.
3. K=4 uses step size 1/4, coarser than d_min=1/8: its training target bootstraps two model steps. Its x-prediction
   is not directly supervised as the data conditional mean. Compare that actual teacher with d_min and data.
4. A projection on one realised transition equals event probability only for the appropriate two-endpoint mixture.
   Multiple damage sizes, digit geometry, noise, and other visual changes violate that assumption.
5. A successful frozen readout fit establishes a locally available alternative under that fit's data/objective;
   failure cannot distinguish interference from incomplete features, regularisation, domain shift or target errors.

## Measurements, declared before GPU use

- DEV histories, t>=8, four situation-defined strata: awake/asleep crossed with adjacent/nonadjacent zombie.
  Sample up to 6000 awake-adjacent, all asleep-adjacent, 6000 awake-nonadjacent and 4000 asleep-nonadjacent. Inverse
  inclusion weights recover their natural DEV frequencies for regression; never select by the recorded outcome.
  Five episode folds. This diagnostic adapts readouts on DEV folds; it does not create a new evaluated world.
- Reuse the production encoded prefixes and eight lightly corrupted cached frames, action, belief, and native K4
  physical sampling. Skipping the independent glimpse stage is checked bitwise against the complete sampler.
- Record raw last-block health residual, final-normalised residual, fully modulated pre-linear feature, and output;
  one native future per history. Hit AP at each, plus output with current-digit interactions. Pure noise negative
  control. Save states, metadata and source pins.
- At the same initial noise/context: finest-step data-trained estimate, both finer calls forming the K4 bootstrap
  target, and K4's estimate. Compare row-specific lean and full token error without calling lean probability.
- One additional finest-step call per row at a uniformly drawn training-grid signal. Fit an affine residual
  correction to the frozen output weights using raw next-percept targets and squared error, across fitting folds.
  Two fits: the existing shared projection (all 81 cells), and health-only (same affine family, applied only at
  health as a diagnostic upper bound). Natural state and cell frequencies; ridge 1e-4 times average feature second
  moment; intercept unpenalised. This uses native lightly noised context, NOT the full training context distribution.
- Held-out native 4-step sampling under the untouched head and the two frozen fits, four independent futures,
  identical initial/context randomness. Measure hit/miss damage, natural overall false changes, fair ensemble Brier,
  awake/asleep, and global latent sample distance (descriptive, not a proper probabilistic quality metric).
- Frozen squared-loss gradient contributions at the linear output, health versus other cells, to distinguish
  local conflict from weak aggregate drive. These are objective derivatives, not actual Adam training updates.

## Interpretation

- AP loss specifically at norm/modulation revises 'last linear layer'. Retained AP after all those stages weakens
  complete erasure, but does not prove the feature is in an appropriate form for the intended output.
- A biased fine teacher and a matching biased K4 target imply the coarse head inherits a target problem; cannot
  call the K4 head alone under-fitted from mismatch to data/event probability.
- A shared affine correction that improves held-out damage and the actual regression objective without false-drop
  costs establishes a readout lever in these fixed features. It does NOT identify how joint training arrived here.
- Health-only improvement with shared-fit failure indicates a local shared-objective/capacity tradeoff in this
  frozen representation, not proof that the architecture must use a separate health head.
- Neither fitting result identifies a training repair without a matched training comparison. No automatic followup.

Research: Dreamer 4 eq.3,6-8 (2509.24527); Shortcut Models (2410.12557); Hewitt & Liang (D19-1275), Geiger et al.
(PMLR 162/geiger22a). Source code and saved results take precedence over narrative interpretations.

Before any full-fit results: add a semantic entity-count guard to native refit readings, using the checkpoint-bound
S1 reader. Also report rises, not just drops. A head intervention cannot count as a globally harmless lever based
on next-token distance alone.

Post-hoc CPU control, after seeing stage AP and the affine regression fit (before native fit outcomes): refit a
single 64-hidden-unit residual GELU readout on the exact SAME finest-step pure-noise health features and real-token
targets. Five episode folds, 50 epochs, one fixed seed, 10% fitting episodes for best validation epoch, natural
inverse-inclusion weights. Compare with an affine pure-noise fit and the original fine output. This tests whether
nonlinear target translation is accessible; it does not test original shared-head optimisation or native sampling,
and a negative result is not proof that all nonlinear readers fail. No architecture/training repair is adopted.

Final post-hoc CPU control: fit one natural-frequency weighted logistic health-drop probe across ALL four situation
strata, five episode folds, rather than a different probe for each awake/asleep adjacent cohort. Report within-cohort
AP and probabilities beside all-row calibration. This tests the cohort side-information confound in interpreting
the original specialised classifiers as a ready-to-use global health predictor. Same frozen pre-linear features;
no new data collection, model changes or followup generation.
