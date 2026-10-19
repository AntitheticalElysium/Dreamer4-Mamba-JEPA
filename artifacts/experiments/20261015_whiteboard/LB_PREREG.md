# Selective persistence (PI) vs an ordinary evolving latent belief (LB): preregistration

2026-10-15 late. Written before LB has trained. The user's sharpened question, after R39 and ARCHITECTURE_RESEARCH.md (Codex):

> Does selective persistence offer a measurable advantage over an ordinary evolving latent belief, without sacrificing
> contextual reasoning?

## Arms

- **PI:** the trained PI worlds (`pi_seed{1,2}/pi_u3000.pt`): selective persistence of observation tokens (hard scroll, chance
  swaps, EM-gated edits). Imagination feeds its own frames back through the token backbone. Existing recursion samples and
  readings (R23, R24, R33).
- **LB:** `lb_world.py`. R2-Dreamer's RSSM unchanged (12M configuration, block-GRU deterministic path, one-hot categorical
  stochastic state, KL balancing with free bits, LaProp + AGC).
  - Observations enter only through the posterior. Imagination advances the latent belief by prior samples and never feeds
    predictions back.
  - Adaptations: an MLP embedding of the frozen JEPA tokens, and a token readout of the current frame trained with PI's token
    loss (no pixels).
  - Same data, windows and batch as PI; trained from scratch, seed sets 1 / 2.
- **Budget:** 30,000 updates.
  - **Declared extension rule:** if LB's token loss over updates 25k-30k is still more than 5% below its 20k-25k mean, training
    continues to 60,000 and LB is evaluated there.
  - PI's lineage had ~40k updates of backbone training (D parent) plus 3k; disclosed.

## Evaluation (identical protocols to PI)

- **Recursion:** `transport_recursion` protocol. 200 DEV panel roots, 15 real context frames observed, recorded actions, H16,
  8 samples, bright roots. Readings:
  - `transport_recursion.evaluate`;
  - `mob_fading`, `census_split`, `pursuit`, `decision_proxies`, `entering_memory`;
  - the R19 guards.
- **One-step:** `transport_eval` analyze on the exploratory cohort, 64 prior samples of the next frame, both readers.

## Readings and verdict (both seeds)

- **A (persistence):**
  - persisting-mob score at d16: PI − LB ≥ 0.10;
  - AND |cow census d16 − 1| smaller for PI than LB.
- **B (contextual reasoning not sacrificed), non-inferiority:** PI ≥ LB − 0.03 on each of:
  - zombie-adjacency AUC16;
  - cow-adjacency AUC16;
  - health-drop AUC16;
  - zombie pursuit approach at depth 6-16.
- **C (coherence, reported):** one-step valid / duplicate / absent; joint ES (both readers); phantoms against reality's
  far-mob rate.

**Verdict:**
- **"Selective persistence gives a measurable advantage without sacrificing contextual reasoning"** if A and B hold at both
  seeds.
- **"Persistence advantage with a contextual cost"** if A holds at both seeds and B fails at either.
- **"No persistence advantage over an ordinary latent belief"** if A fails at either seed.
- Readings where LB is better are reported in all cases.

**Secondary:**
- arrivals (`census_split`);
- revisited / new entering terrain error;
- vanish and move rates;
- decision rates;
- LB at 10k / 20k (learning curve);
- imagination compute per step.

## Known limits

- **Architectures differ beyond the persistence mechanism.** LB has a global latent and global decoding; PI has a spatial
  backbone, a chance operator, and a deeper training lineage.
  - A positive result is evidence for selective persistence *as implemented in PI* against a standard latent belief, not a
    pure ablation.
  - A matched spatial-latent ablation would be the follow-up.
- LB's stochasticity is the RSSM prior. It has no separate chance operator.
- Fresh confirmation stays sealed.

## Addendum (10-15 late, before any LB evaluation): a defect in the token objective's scale, corrected

**What happened.** The first launch scored the readout with PI's *mean* D token loss. Against R2-Dreamer's KL, which is summed
over latent dimensions, encoding the observation was worth less than its KL cost, and the posterior collapsed:
- posterior entropy 87.7 of a maximum of 88.7;
- KL at the free-bits floor;
- the readout predicted mean tokens (evaluation-pipeline smoke at ~2.7k updates: no mobs, 45-80% tile error);
- the same through 2.5k updates at both seeds.

**Correction.** R2-Dreamer's decoder likelihood sums over pixels, so LB now uses R2-Dreamer's own `MSEDist(agg='sum')`, summed
over the 81 × 192 token elements. This restores the reference's loss scale; it is not a tuning choice. PI's mean D loss is
logged for comparison.

Both seeds restart from scratch (`lb_v2`); the discarded runs are not used. Budget, extension rule, readings and verdict are
unchanged.
