# Dissection after E1/E2/E3, before further fitting

2026-11-03. E1/E2/E3 all complete. P1-F1.6's world results are explicitly out of scope.
This is post-result analysis, not a revised preregistration or a new adoption gate.

Observed: E1 endpoints exactly reproduce original head weights. In E2 some heads
acquire drink/stone contingency and reduce their factual counter error; seed/arm
rankings reverse. E3 changes are small. The following bounded work addresses concrete
interpretation risks, not a search for a better head or policy:

1. Replay ONLY the two already completed E2 FRESH checkpoint10 fits (12000 updates,
   unchanged schedule/seeds/minibatches). Add snapshots0,100,1000,2000,4000,6000,
   8000,10000,12000. The original code stored only their endpoints. Require exact
   endpoint prediction/C/S equality with original NPZ. This resolves whether the
   handover develops with a fixed encoder rather than needing representation changes.
   No longer fits or alternative LR, loss, head, seed or input. Hard wall cap10min.
2. From ALL archived forecasts, stratify actual counter changes by previous value0
   versus>0, success/failure, and day/night. Labels are evaluation only. Separate the
   target counter's own C energy from neighboring-cell contributions to its windows.
   This checks first-unit-only learning and window-overlap false interpretations.
3. Use paired episode-bootstrap intervals for E3 rich-minus-coarse and E2
   carry-minus-fresh endpoints, retaining each head seed separately. Check identical
   episode/action/reward/reset metadata across encoder caches and saved optimizer
   steps4000/8000/12000. Do not pool away seed disagreement.

Raw forecast coordinates are fp16 archives; direct counter MSE/C measurements in the
NPZ were computed in float32. Compare their aggregate agreement before interpreting
directional changes from fp16 vectors. NOOP forecasts have no paired true NOOP future:
distance to previous tokens is a persistence reference, not a counterfactual oracle.

No joint world training, reader fitting, hyperparameter search, source alteration,
automatic follow-up or reading of F1.6 performance. Discuss these findings afterward.
