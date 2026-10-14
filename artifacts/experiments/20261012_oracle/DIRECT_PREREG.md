# Direct conditional completion: matched two-arm preregistration

2026-10-14. Written before any direct-generator optimization. This is one study with two training arms, not a
posterior/codebook/prior repair. No H16, actor, adaptive-compute mechanism, third training arm or unconditional seed
replication is authorized by this protocol.

## Question and intervention

Can a reconstruction-free continuous JEPA/Mamba world learn physically coherent alternative one-step outcomes directly
from recorded observations/actions when its completed successors receive a proper distribution objective?

Both arms use the same unconditioned D parent, frozen Raw 81×192 tokens, trainable six-layer corrt/fmamba backbone,
completion head, initialization seed7, data/noise draws, optimizer/schedule, batch16 and **3,000 updates**. Each L16
window supplies15 next-state targets:720,000 targets per arm. Mamba features are computed once with gradients, then
reused for two independent completion samples. No future is fed to a chance encoder or generator. Causal teacher
context is checked explicitly. Fresh Gaussian noise has4 local dimensions per token and4 shared dimensions per frame;
a small spatial mixer modulates both generation and correspondence logits. This is a parameterization, not a bit-rate
claim. Completion retains the parent's soft copy candidates; no untested hard assignment is added.

The only trained contrast is `point` versus `energy`:

`point = .5*d(X1,Y) + .5*d(X2,Y)`

`energy = point - .5*d(X1,X2)`

Distance has exponent1: .5global RMS Euclidean norm on35 interior tokens plus .5mean RMS Euclidean norm on nine fixed
overlapping3×3 interiors. There is no squared-distance diversity objective, finite-distance smoothing, event mask,
temperature, rate/sparsity penalty or simulator label. The metric is strictly proper on its35-token projection only.
HEAD outputs affect the entire frame, so unscored HUD/edge fidelity and variation are reported separately.

## Data and scope, fixed before optimization

The eligible TRAIN ledger contains367,619 of969,661 possible windows, from6,109 TRAIN episodes. All17 actions remain.
Select only from the first observed RGB frame: map-pixel75th percentile of max(R,G,B)≥150. Selection is measurable
from the input history for every target; no actual future movement, scroll, hit or outcome controls inclusion.
Uniform eligible-window sampling replaces terminal oversampling. The pixel rule defines **bright-view**, not certified
daylight:81.77% of later frames meet the same operational threshold, reported without filtering those frames.

The scored interior is rows1–5, columns1–7 of the7×9 map. Its immediate copy sources are visible for a one-cell shift.
HUD and unseen entering terrain are excluded from the training distance. This controls those confounds; it does not
learn nuisance invariance, infer hidden timer phases, preserve epistemic terrain or validate control.

The existing200-root/32-key fan is exploratory. The same RGB-history rule selects104 roots;69 have variable interior
mob layouts (56 cow,16 zombie,7 skeleton,1 arrow; overlap allowed). Physical conservation's strict source stratum has
14 independent variable roots (8 cow,3 zombie,3 skeleton). Sources are isolated, camera-stationary, non-DO, near the
player where new spawning cannot contaminate their immediate neighborhood, and all five destinations lie in the
scored interior. No generated sample is dropped because it is misregistered or incoherent.

Only real-trained frozen readers are used. The primary real-ridge reader is exact on all104×32 real mob layouts.
The independent bank reader is a sensitivity check, with primary clean-source selection fixed from real data only.
Source-level physical claims require≥95% real-reader event agreement, agreement of the frozen readers on the relevant
source stratum, and camera/nearest-real checks. A nearest-real projection cannot turn an invalid sample into success.

## Readings and predeclared decision

Evaluate64 independent completed samples per root at the final3,000-update checkpoint. Milestones0/500/1,500 describe
optimization and noise dependence; they cannot select the primary checkpoint. Use paired root/episode bootstrap95%
intervals (2,000 draws, fixed seed20261014). Real-to-real32-key splits calibrate finite-fan/readout floors before training.
Their split Monte Carlo interval is not a population-confidence interval. Floor rules and numeric margins are saved
atomically in `direct_generator/criteria.json` before launch and pinned into the run's evidence contract.

- **Primary distribution:** joint decoded zombie/cow/skeleton layout energy on the scored35 tokens, using the primary
  real-state reader. Read complete samples before averaging. Require the upper95% bound of energy-minus-point<0.
  Report exact-truth and independent-reader sensitivity; no improvement headline if it disappears on reader-valid
  strata or is explained solely by threshold disagreement. Cow source/destination Brier is the cleanest conditional
  law sensitivity; hidden zombie/skeleton phases can affect movement, so their exact full-state probabilities are not
  asserted to equal the model's partially observed conditional law.
- **Physical coherence is binding:** jointly measure source clearing and one destination, absence, multiple destinations,
  source-plus-destination duplicates, count distributions, collisions and camera consistency. On strict valid sources,
  candidate absence+multiple and duplication upper95% bounds must be within the real-reference floor plus finite-draw
  resolution. Count/collision metrics outside the trained domain are diagnostic, not a hidden additional criterion.
- **Static fidelity is binding:** the upper95% paired increase in scored decoded static-fact corruption must not exceed
  the pre-training real-reader/fan-resolution margin. Report absolute corruption of both arms. Contextual latent
  variation on locally unchanged cells is not itself physical corruption; compare it with real-fan variation.
- **Actual alternatives:** require nonconstant valid source/destination outcomes across independent samples; token
  variance or code entropy cannot establish diversity. Its coverage must meet the independent-real split baseline
  within the fixed finite-draw margin. Record marginal and joint frequencies, local-nearest disagreement, output/noise
  sensitivity and where variation occurs. Never label a layout impossible merely because it was absent from32 keys.

**Exploratory promising** means the distribution contrast resolves and the physical/static conditions hold on the
available cohort. Its14-root strict stratum cannot establish a confirmed physical positive. A favorable but statistically
unresolved source-law reading remains exploratory; it cannot become an exact conditional-calibration claim.

**Confirmed success** requires the same criteria on genuinely fresh reserved seeds, with≥20 independent variable clean
source roots, including≥20 cow-source roots for the conditional movement claim. Failures of these criteria are valid
negatives for this implementation/budget after numerical and live-noise checks pass. Reader failures, inadequate
coverage or still-materially-improving held curves are recorded as inconclusive/budget-limited; they do not reject direct
stochastic dynamics universally. Distribution improvement with incoherent construction is explicitly a failure to
establish coherent completion and leads to diagnosis, not a larger generator.

## Fresh confirmation and efficiency

Reserve base seed2026101400 and the existing archived CPU expert/PPO observation collector before training. Fresh
outcomes cannot fit readers, tune thresholds, select checkpoints or affect either arm's optimization. Collect only after
an exploratory-promising result: target≥128 bright roots and the20/20 source/cow coverage above; continue if necessary
to at most512 bright roots/1,536 episodes. Stop at the first episode boundary meeting the declared coverage. Every root
receives32 independent evaluation-only futures at its recorded action. A separate128-episode historical DEV reserve
is campaign-held-out only, because prior global access is unknown; it cannot replace genuine fresh confirmation.

Measure synchronized backbone, two-head and optimizer times during training; inference backbone amortization and
incremental1/2/16/64-sample head costs; allocated/reserved peak memory and throughput. Compare sampled variance on
mob-uncertain, semantically stable, scored/unscored and HUD locations against the real fan. This measures allocation,
not learned adaptive compute. Enforce≤6GiB with a real-parent batch16 smoke before launch.

Full model/optimizer/window-noise sampler/Torch-CUDA RNG states and batch-noise ledgers are saved atomically every25
updates; milestone forecasts and evaluation batches resume independently. Validate interrupted2→4 versus uninterrupted4,
causal future isolation, output and gradients from both independent draws, routing/generation influence after LayerNorm,
finite norms, same inference batch shape, and source/data/preregistration hashes. Numerical deviations are reported.
Runtime logs go to `artifacts/eda/levers_logs`; only agents write NOTEBOOK experiment interpretations.
