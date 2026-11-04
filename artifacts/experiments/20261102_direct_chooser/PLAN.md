# Direct spatial chooser feasibility

2026-11-02. User authorized this test after DISCUSSION.md. No foveal route, world
weights, corpus, production launcher or sampler is changed. Sparse distillation is not
launched automatically. Read the prior discussion in ../20261102_foveal_information/.

## Fixed first-stage design, before any fit

* Frozen F1.5 checkpoint 10. Collect detached native periphery/light from real frames.
* First 400 S0 train episodes; 0:360 fitting, 360:400 validation. Exclude P1's monitor
  episodes from both. Natural transition frequencies, uniformly sampled fitting frames.
  Include terminal successors; never cross episode boundaries or read beyond a target.
* Report the saved 400-dev-episode locus benchmark and fresh dev episodes 400:600.
  They are development comparisons; final evaluation is untouched. Confirmation episodes
  are disjoint from the locus bank and the previous 250-episode conditional-readout bank.
* One generic spatial forward model. Local eight-frame coarse history -> width 64;
  global history of lighting, actions, arrived reward/termination and validity -> width
  64; queried action and learned grid position embeddings; two spatial attention blocks
  (4 heads, MLP expansion 2); 16-D output delta added to previous coarse tokens.
  Global spatial attention permits remote effects. No semantic/HUD-specific pathway.
* Predict p[t] using frames through t-1 and a[t-1]. No current p/v/r/d, current crop,
  native fine-history belief or semantic labels in the predictor. Surprise may compare
  its prediction with arrived p[t] AFTER predicting, before choosing the crop.
* Mean squared error over all 81x16 coordinates, no event weights or curriculum.
  Two head seeds (611601/611602), 4,000 updates, batch 256, AdamW lr 1e-3, wd 1e-4,
  global clipping 1; fixed cosine decay to 1e-4 after 100 warmup updates. Read final
  weights, not best development weights. Validation loss is descriptive only.
  The shape/timing smoke uses separate weights/RNG and is not an arm.
* C = window sum of squared actual-action vs NOOP mean differences; S = window sum
  of squared arrived-periphery vs actual-action mean differences. Equal mean-normalized
  C/S, softmax T1, 20% uniform, exactly as the previous working policy.

## Primary retention gates

Fixed saved reference has 300 examples for each target class and 105–144 episode
clusters. Its per-class uncertainty is several percentage points; a paired 5-point
comparison can be conclusive when decisions agree but is not guaranteed to be powered.
No margin expansion or post-null sample/budget extension. Report inconclusive if needed.

For EACH head seed, paired episode-bootstrap 95% lower bound of candidate minus
reference must be >= -0.05 for all four remote component capabilities: drink C, stone C,
coal S, iron S. Also require the same noninferiority for actual mixed-policy probability
mass on each class's remote counter. Contact coverage/mass and C/S individually remain
separate readouts; a contact success cannot substitute for a remote failure.

Freeze mixture baselines in reference.json BEFORE fitting. Report argmax, probability
mass, and paired categorical draws (32 per context, shared uniforms). Means of draws are
secondary to exact probability mass. No replacing exact probabilities by a lucky draw.

Full adoption additionally requires fresh-cohort retention, native noise-seed robustness,
early-checkpoint/online adaptation, own-history behavior/night/noise checks and full-path
cost. If the primary gates clearly fail, stop at feasibility and discuss rather than
automatically run sparse teachers or a new P1. A positive checkpoint-10 test alone is
not an endogenous-from-scratch qualification or a world-learning improvement claim.

## Secondary diagnostic readings

* Held-out factual token MSE vs copy-previous, per class and on random transitions.
* Remote and contact C/S argmax, union coverage, actual mixed-policy mass and TV from
  the native policy. Report all kinds, no averaging away a failed iron/coal capability.
* On fresh dev 400:600, obtain matched native C/S for at most 128 contexts/class plus
  128 random contexts, two common-noise forecasts. Correct actual/NOOP belief forks.
  A fresh draw is not bit-identical to the saved locus benchmark; distinguish them.
* Report exogenous (epsilon=1) and mixed-policy action support. Predictive action
  contrasts are not automatically identified causal effects outside offline support.
* Save predictions, per-window scores/probabilities, sampled choices, episode/time keys,
  learning curves, checkpoint/data/code hashes, tests and elapsed time/memory.

## Instrument checks, before fitting

History excludes current/future target and never crosses episodes. Actual NOOP C is
exactly zero. Reset padding has validity flags and contributes zero content. Correct
action alignment against native data. Perturbing current p/r/d cannot alter the forward
mean. C/S policy matches the existing function numerically. Gradients reach only the
head. Native raw collection reproduces saved coarse tokens within bf16 rounding, with
the same head using full-precision stored tokens converted to float for its arithmetic.

Short negative fit is a negative for this bounded model/recipe, not proof that direct
spatial prediction in general cannot select gaze. Predicting the conditional mean is
intentional for ranking, not an imagined-world representation.
