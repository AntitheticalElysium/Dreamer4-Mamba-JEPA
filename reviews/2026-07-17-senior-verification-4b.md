# Senior verification: Step-4b long-context scale screen — CONCUR

Date: 2026-07-17. Scope: independent verification of the companion's 4b
screen (commits 1476dab / 9cf965c / ef29246) and of every decision in
`reviews/2026-07-16-long-context-scale-companion-review.md`.

## Independent checks performed (all pass)

1. **Registered arithmetic reproduced exactly from the raw row files** (my
   own recomputation, not the report): k8 separations 0.001646 / 0.001230 /
   0.001479 / 0.001220; delta_small −0.00041697; delta_large −0.00025892;
   interaction +0.00015805; per-env delta_large positive in 1/4 (111 only);
   minimum-effect threshold 0.000148; tie retrieval 27.60/28.12/23.44/27.08%;
   per-horizon curve including the k2–7 post-hoc mean +0.000332 positive in
   all four env seeds; scale-vs-small deltas −0.000168 / −0.000010; the full
   rung table. Zero discrepancies.
2. **16/16 committed checkpoint hashes validate**; monitor bundle hash on
   disk matches the pinned manifest; shared-init and replay-stream digests
   identical across all four arms.
3. **Large-arm parameter match 0.119%** (3,372,004 vs 3,376,032) as claimed.
4. **84/84 tests pass** (including slow collector/environment regressions).
5. **Collector for seeds 111–114 canonicalizes the live env at both reset
   sites** and runs verify_repeat — the 2026-07-15 collector lesson held.
6. Gate evaluation is faithful to the preregistration: conditions 3 and 5
   fail, so "confirmatory replication not licensed" is the correct verdict —
   including the discipline of not letting the rung-2000 or k2–7 patterns
   override the registered endpoint.

## Concurrence

I adopt the companion's decisions without modification: Mamba-2 support GO;
research backend retained; large pooled Mamba as default NO-GO; "scale makes
Mamba better" refuted for this screen; no shuffled controls or fresh-seed
confirmation licensed; GRU-64 remains operational; online RL still NO-GO.
Seeds 115–130 remain unspent and reserved.

Two readings I want on the record as the senior reviewer:

- The screen's most decision-relevant negative is not Mamba-specific:
  **neither large core improved final-horizon counterfactual separation over
  its 30k-parameter counterpart** (LL-G −0.000168, LL-M −0.000010 vs small).
  Capacity+context scaling of the POOLED adapter is a dead end in both
  backends. That elevates the topology question above the backend question.
- The k2–7 intermediate-horizon Mamba advantage (positive in 4/4 env seeds,
  reversing only at k=8) plus Mamba's better retrieval alongside worse
  separation ("smoother, less action-discriminative at the endpoint") is a
  coherent, preregistrable hypothesis — not noise to discard, not a result
  to claim.

## Note on the user's scale/dataset skepticism (2026-07-17)

The user's instinct — "testing both in isolation doesn't reflect scaled
performance; a GRU is too simplistic for a Dreamer-4-class system" — is
partially aligned with what the evidence says, with two corrections:

1. What our experiments select is the backend for THIS compact pooled
   topology at THIS scale. Dreamer 4 does not use a pooled GRU **or** a
   pooled Mamba — it keeps hundreds of dense tokens and mixes them jointly
   in a block-causal transformer. At Dreamer-4 scale the pooled-vector
   design itself is the thing that would be replaced, so "GRU beat Mamba"
   does not transfer to that regime, and no claim we hold says it does.
2. The dataset point has a sharper form than "Crafter is too simple":
   replay episodes cap at ~200 transitions, so T=128 was the longest
   well-supported context and there is no 1,000+-step dependency structure
   for a state-space model to exploit — DRAMA's decisive GRU gap appears at
   1,664 tokens on a purpose-built memory task. On this data, at these
   horizons, backend parity is a PLAUSIBLE literature outcome (2026-07-17
   companion correction: DRAMA's synthetic result shows Mamba CAN win under
   deliberately long dependencies; it does not predict equality on Crafter,
   so "expected" overstated it), which the 4b review documents.

The testable version of the user's concern is therefore: **is the pooled
bottleneck (not the backend) what's limiting action-discriminative dynamics?**
That is exactly the long-registered "source-shaped flattened-latent global
arm" follow-up — screened this round (see 2026-07-17 exploratory protocol).
