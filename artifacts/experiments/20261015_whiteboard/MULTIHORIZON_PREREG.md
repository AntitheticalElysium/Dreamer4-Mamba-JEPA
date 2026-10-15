# Can an evolving belief represent graded consequences directly? Multi-horizon expected-consequence heads, preregistration (R54)

2026-10-16. Written before any code for this study ran.

## Why

- R51-R53: the action-dependent part of H16 risk is concentrated in near-term consequences.
  - The real first-step damage of one sampled future ranks the escape moves at 0.71 (within-root AUC against the 32-key P16
    order).
  - Later damage carries almost none of it: 0.46.
- A realistic imagined future accumulates action-independent damage that dilutes rollout risk read-outs. D's rollouts escape
  this only because its future goes quiet (zombies vanish, pursuit 0.05).
- PI's coherent scene is the most action-informative (adjacency ranking 0.65), but its MAP copy/edit gate under-calls the
  first-step hit (0.60).
- User direction (10-16): explore how an *evolving predictive belief* could represent consequences without regenerating every
  observation token.

## Hypothesis

A belief that predicts the **expected** future consequence channel at several horizons directly, without rolling out
observations, ranks actions at least as well as the best rollout. L2 regression targets the conditional mean, the graded
quantity a MAP gate discards.

Precedents:
- general value functions over observation features (Sutton et al., Horde, 2011);
- γ-models (Janner et al., NeurIPS 2020);
- the value / reward heads of MuZero and TD-MPC2.

## Design (label-free, reconstruction-free, head-only on frozen beliefs)

- **Backbones** (frozen): D parent; PI s1; PI s2.
- **Inputs:** the 8-frame context (frozen Raw tokens) with its actions, the last one being the first action under test. Read:
  - the belief at the 18 HUD cells of the root frame (index 7);
  - the action token.
- **Targets:** the observed HUD tokens (frozen Raw tokens, LN) of the training pool at root + k, for k ∈ {1, 2, 3, 4, 6, 8, 12,
  16}.
  - Pool trajectories are end-aligned: deaths fall at the last frame and are kept as observed frames.
  - Roots are seeded at s + 7 with s ≤ 40; 30,000 roots.
  - The continuation is the behaviour's, so the head predicts the expectation under the behaviour policy after the first
    action.
- **Heads:**
  - primary: closed-form ridge (λ = 1e-3 · N) per horizon, from [belief at cell, action token, cell one-hot] to the token at that
    cell;
  - secondary: one MLP shared across cells and horizons (hidden 512, AdamW 1e-3, 3,000 updates, batch 1,024).
- **Read-out:** the frozen bank HUD health reader is linear in the HUD tokens, so it gives the expected health at each horizon.
  - Risk of a first action = max over k of the expected health loss (h_root − h_k)⁺.
  - Also the per-horizon rankings.

## Evaluation (DV16's 200 roots, `dv16.load_roots`; nothing from them is used in fitting)

- Within-root AUC over the escape choice (noop + 4 moves) against the real P16 order, as R51.
- DV16 expected safe, 5-way and all 17 actions (sleep included).
- Paired bootstrap 95% intervals over roots (4,000 resamples).

## Readings

- **P1 (sanity):** the k = 1 head on D's backbone ranks within 0.03 of D's own first-step rollout ranking (0.711).
- **P2 (primary):** on D's backbone, the multi-horizon risk ranks at least as well as D's 16-step rollout (AUC 0.727): the paired
  difference's interval lies above −0.03.
- **P3:** on PI's backbone (s1 and s2), the multi-horizon risk ranks better than PI's own rollouts (0.580 / 0.626), with
  intervals excluding 0. This tests whether PI's belief holds the graded risk its gate discards.
- **Secondary:**
  - per-horizon rankings;
  - the MLP head;
  - sleep: the expected consequence of sleeping vs the real P16 0.86;
  - all-17 expected safe vs uniform 0.512 / one real future 0.610.

## Limits

- One evaluation protocol; heads only (the belief is not trained for this).
- The behaviour continuation is averaged, whereas DV16's labels use the recorded one.
- Health is read through a linear probe that was fitted with labels; the probe is used only for evaluation, never as a training
  target.
- Fresh confirmation data stays sealed. The judge block is not read.
