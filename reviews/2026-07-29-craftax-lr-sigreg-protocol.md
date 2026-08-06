# Craftax encoder-LR, SIGReg, and executed-control protocol

Date: 2026-07-29
Status at registration: not yet launched
Parent evidence: `reviews/2026-07-29-craftax-encoder-anchor-outcome.md`

## Primary question

Does lowering only the world encoder learning rate from `1e-4` to `6e-6`
preserve the task state that full-rate JEPA training erases, while retaining
enough predictive learning to improve the actual Craftax pipeline?

BC and imagination do not receive the slow LR. They already freeze the world,
so changing their head optimizers would introduce a second intervention.

The existing full-LR 20,000-update baseline is the long-run control. It was
trained after the EMA/BatchNorm correction. Subsequent default-path changes are
declared bit-preserving, and a unit control requires the new two-group optimizer
at equal `1e-4` rates to reproduce the original one-group trajectory exactly.

## Registered order

1. Actual-recipe prefix transfer: Transformer, EMA, pooled predictor,
   `terminal_fraction=0.5`, JEPA plus reward plus continuation, 2,500 updates,
   full versus slow encoder LR, three paired seeds, EMA schedule pinned to
   20,000 updates.
2. Full slow-encoder T-JEPA and M-JEPA pipeline: 20,000 world updates, 3,000 BC
   updates, and 500 imagination updates, using the original seed and all
   original non-encoder hyperparameters.
3. Full representation oracle on both slow worlds.
4. SIGReg mechanism grid: clean JEPA-only setting, full versus slow encoder LR,
   three paired seeds. This asks whether higher-dimensional Craftax changes the
   CartPole failure mode; it is not a promotion experiment.
5. Narrow slow-LR spatial interaction: three slow spatial-predictor cells
   compared with the completed slow pooled cells. The rejected full spatial
   grid is not repeated.
6. One-seed actual-recipe SIGReg full-versus-slow interaction. It is
   diagnostic-only.
7. Executed Craftax evaluation for old full-LR and new slow-LR T/M policies on
   the identical fixed seeds `100000..100029`, 2,500 maximum steps, categorical
   policy sampling, with official Crafter score and paired actor-minus-BC and
   actor-minus-random intervals.

The exact executable order is
`reviews/artifacts/craftax_queue9.sh`.

## Decision predicates

Encoder-LR transfer is supported only if slow beats full on paired semantic
change in the actual-recipe prefixes without destroying held-out predictive
cosine.

The full pipeline is not rescued by an oracle proxy alone. A useful result must
also improve BC action accuracy and executed official score. The imagination
actor must be compared with its own BC and random policies on paired seeds; an
achievement-count delta alone is insufficient.

SIGReg is informative if it changes semantic retention relative to EMA in the
clean matched grid. One actual-recipe seed cannot establish an architecture
win. Any SIGReg result must report its prediction term, SIGReg term, and encoder
LR because the objective scale and optimizer timescale can interact.

Spatial context is reopened only if the slow spatial cells separate from the
already completed slow pooled cells by more than seed variation. Otherwise the
D045 rejection stands.

## Claim boundaries

- The fixed executed seeds are fresh and disjoint from replay seeds `0..319`
  and the probe seed family beginning at `90000`, but they are exploratory, not
  a sealed confirmatory tier.
- Slow LR can preserve a random representation without making it absolutely
  sufficient. All oracle targets begin degraded, so an absolute pass may
  require representation acquisition or pretraining before slow/frozen
  dynamics training.
- No result or outcome review is committed until the relevant job is complete,
  its artifact digests are present, and its oracle/evaluator audit passes.
