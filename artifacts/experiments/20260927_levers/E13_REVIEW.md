# E13 independent review and literature pass — 2026-10-02

Reviewed attachment, NOTEBOOK at commit `9fabf964`, diagnostic source and saved JSON reports. No GPU use, training,
checkpoint changes, threshold changes, or new measurement populations. Calculations below recompute saved metrics on CPU.
This is a review, not an intervention predeclaration or a new experimental result.

## Verdict

E13 substantially strengthens the diagnosis of the **current per-tile rollout**: interaction tiles are wrong even from
training inputs; consequence classes and change signals are recoverable from the backbone; repairing content changes later
movement predictions. Token-level interaction supervision is the intervention most directly matched to this evidence.

It does not yet establish that gradient starvation is the specific optimizer mechanism, that 24% of entering terrain is
irreducibly unpredictable, or that these content pathways explain the sealed H16 action-selection deficit. Those stronger
claims should remain separate from the verified observations. This does not invalidate the substitutions or underfit result.

## What the saved evidence establishes

- `evals/consfit_corrt_raw_teacher_s7_u18000.json`: 19,428 alive training transitions, 870 strict changed DO/place faced tiles,
  1/870 caught (0.1149%), 869/870 copied. Held windows: 1/560 caught. These strict consequence tokens comprise 0.0553% of all
  81-token targets, and 0.8921% of the diagnostic's alive-transition L1 error. **Loss share is not gradient share.** The actual
  world teacher objective is uniform L1 (`tworld.rollout_losses`), not raw MSE. `consfit` excludes dead-next targets when
  computing this diagnostic share; the training teacher loss itself does not apply that mask.
- Seed 8 is an important exception, not a universal never-learning result: strict changed tiles caught 25.52% TRAIN / 27.14%
  held, with DO mining 47.44% / 47.94% and placement still zero in the adequately represented placement classes.
- `stageprobe_corrt_raw_teacher_s7_u18000.json`: a standardized linear class probe, changed examples weighted 5x, reads next
  class from backbone h at 86.33% on 556 changed held examples. The input linear reference reaches 98.38%. This is evidence
  that useful consequence-predictive features reach the output interface. It is not proof the original embedding-regression
  head has already learned to express a correct next embedding.
- `conscalib_corrt_raw_teacher_s7_u18000.json`: an **unweighted** change probe reads h at AUC 0.9446, mean P(change) 0.6341 on
  changes versus 0.1194 on nonchanges. This supports the output bottleneck without relying only on weighted class probes.
- The class-mass check (`check_catmass.py`, recorded result in its docstring/notebook) distinguishes a sharp copied class
  from a hidden correct mode: on changed examples, categorical mass is 0.067 on the true next class and 0.828 on the current
  class. Enabling sampling on that checkpoint cannot supply a correctly learned interaction distribution.
- Teacher s7 population substitution: position-failure rate 0.448795 base -> 0.282129 consequences -> 0.341365 entering
  content -> 0.121486 both. Individually these repairs remove 37.14% and 23.94% of failures; together 72.93%.
  Seed 8: 37.27%, 27.09%, and 63.75% respectively. These are interventions on oracle content, not expected gains of a trained
  loss or a sampler.
- Teacher s7 first-error cases: target-tile repair restores 60.36% of 439 decisions, random tile 2.96%, all map tiles except
  target 34.62%, entire current frame 98.18%. False scrolls alone: target repair restores 93.10% (58 cases). The target is a
  strong contributor; the rest of the frame also contributes. The predeclared `R4_target_tile_causal` conjunction remains
  **false** because all-but-target exceeds 30%; do not silently label that rule passed.

## Scope corrections

### Allocation mechanism

The established result is **underfit output despite decodable inputs/backbone features**. Uniform supervision, rare targets,
late seed-dependent learning, and a favorable weighted probe motivate an allocation intervention. They do not distinguish
low gradient magnitude, cancellation with background gradients, copy/write gate attenuation, regression geometry, and
optimization difficulty. Direct/categorical failures mean a copy gate alone cannot explain all arms.

A short same-head comparison can finish this part of the diagnosis: hold backbone features, head initialization, batches,
optimizer and dose fixed; compare uniform versus interaction-focused loss, measuring original head gradients/updates and
rare-versus-unchanged errors. Inspect `proj` and `choose` separately for mixture heads. Frozen-backbone success would isolate
head training without requiring another encoder or dynamics architecture. End-to-end continuation still needs its own
matched comparison, including the Mamba variant.

### Entering terrain

`check_enterpred.py` compares a finite MLP using **three edge tokens** (0.756 accuracy) with the world (0.760), edge copying
(0.714), and majority class (0.476). These are model performances, not a Bayes ceiling. The MLP does not explicitly use all
four frames, all visible cells, map history, or memory of reentries. Its split is held pool windows, not fresh episode seeds.
The same window split is used in the input/backbone consequence checks. Their TRAIN underfit conclusion remains strong.

E11's five futures start from the same full simulator root (`20260926_diagnosis/futures.py`), with the hidden map fixed.
The empirical variance measures future RNG variability conditional on that full state, on the retained all-samples-alive
population. It does not measure uncertainty over unseen maps conditional only on the images, nor is a finite-sample estimate
an exact pixel-conditioned floor. Thus its roughly 8-9% variance share does not rule out a distributional treatment of hidden
terrain, and the local MLP's roughly 24% classification error does not establish how much such a treatment can fix.

### Oracle decomposition

The 28/36/3/33 percentages are sequential oracle error reductions, not unique, independent causes. The `realign` intervention
also fills exposed cells with true tokens (teacher s7: 1,539 corrections / 12,769 filled cells). Effects interact: for teacher
s7, consequence-only and entering-only reduce excess by 11.91% and 32.03%, while both reduce it 55.33%, an 11.39-point excess
reduction beyond their sum. After position+entering repairs, the further consequence reduction is approximately 3.5% of base.
A small final residual contribution therefore does not imply consequences are unimportant.

## Literature checked and how it applies

- [CAER](https://arxiv.org/html/2608.30897), Sec. 2.3: unit-mean token weights redistribute coefficient mass. The first-order
  benefit depends on weight/gradient-utility covariance. This supports focused allocation; it does not prove that covariance
  or gradient cancellation in our model. Its model-derived action-response weights can miss our copied interaction tiles and
  respond strongly to camera scrolling. Prefer an input-known interaction region for the first causal test.
- [IMPACT](https://arxiv.org/html/2609.00161), Sec. 3: a text-attention region prior, detached local-error weighting, normalized
  loss, and separate gradient routing for the attention that supplies the region. It is more than global high-error mining.
  For our first test, the known faced tile substitutes for localization; weighting every high-error border or mob token could
  preferentially spend effort on uncertainty or token-context artifacts.
- [CGSReg](https://arxiv.org/html/2607.15142v1), Sec. 3/4: mask-area-normalized prediction loss improves frozen-world Pong
  transfer in four of five matched models. Scores improve from -21 to -11.9 (DreamerV3), -13.9 to -5.8 (DIAMOND), -21 to -1.9
  (TWISTER), and -15.8 to -4.1 (Simulus); STORM stays -21. This is useful intervention evidence, not complete repair.
- [EAWM](https://arxiv.org/html/2601.19336), Sec. 3.3 and App. B: event prediction with focal loss, event-boundary suppression,
  and event-aware observation weighting in the general/Dreamer formulation. The Simulus branch retains its observation loss.
  Reported Craftax improvement is approximately 10%. It is not evidence that any standalone event head suffices for our
  frozen-patch output bottleneck. Event definitions must separate camera scrolling from world-content changes.
- [Gradient Starvation](https://arxiv.org/abs/2011.09468): theory about cross-entropy feature discovery. Our h already carries
  much of the event signal and our main loss is L1, so call this an analogy until actual head gradients/interventions settle it.
- [Delta-IRIS](https://arxiv.org/html/2406.19320), Sec. 2.2: stochastic innovations separated from predictable changes, with
  autoregressive sampled deltas. Relevant to a hybrid future design; its trained local reconstruction tokenizer differs from
  our contextual ViT tokens. Do not equate sampling our existing k-means code head with this method.
- [Dreamer 4](https://arxiv.org/html/2509.24527v1), Sec. 4.1: dynamics use uniform sequences while policy BC uses relevant
  sequences, explicitly to avoid optimistic generations. Preserve failed interactions and prerequisites rather than training
  dynamics only on successful mining/placement.

## Recommended interpretation of the three options

1. **Interaction-focused token loss first.** Preserve factual next-token targets and the background loss. Initially focus the
   faced tile on **all** DO/place attempts, successful or failed, with region-normalized or bounded mean-normalized weights.
   An input/action-defined mask avoids using future success to alter outcome frequencies and avoids CAER's cold start. Do not
   simply weight all screen-space changes: camera movement would dominate. This changes allocation without an inference head.
2. **Event supervision second, coupled to what is generated.** An auxiliary event classifier reading only h has zero direct
   gradient to the current `proj`/`choose` output parameters. h already supports AUC ~0.94, so such a classifier may improve
   its own metric without correcting the emitted tile. Alternatives are a supervised copy/write gate actually used by the
   output, or a fixed factual-trained semantic decoder supervising the generated tile. Keep token fidelity and independent
   evaluation to detect decoder exploitation. Compare event-only, weighting-only, then combined if individually useful.
3. **Distributional hidden-content modeling separately.** Train calibrated conditional probabilities/innovations, retain
   spatially correlated border content, and persist a sampled map across revisits. Condition on available memory; previously
   observed terrain should not be freshly sampled. Evaluate probabilities/proper scores and expected decisions over multiple
   samples; sampling is not expected to beat the optimal point predictor on squared error or top-1 class accuracy. A targeted
   stochastic innovation model can retain the Mamba backbone, without immediately replacing the whole system with diffusion.

Watch coupled effects: correct mined/placed terrain must agree with inventory/reward/prerequisite changes; correct tile
classification alone can permit policy exploits. Fixing the current content chain also does not automatically repair
real-to-generated decision-head transfer. The paused H16 panel and subsequent actual actor-versus-BC result are needed to
connect the mechanism repair to the project's purpose. No new expensive gate family is proposed here.
