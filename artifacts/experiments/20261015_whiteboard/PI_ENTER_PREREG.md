# PI + sampled content at never-seen entering cells only (evaluation only), preregistration

2026-10-15 night, after R29-R31 and before any rollout of this variant. One change relative to PI.

## Question

Does drawing new content only where the uncertainty is epistemic (entering cells never seen before) restore arrivals without
R29's terrain noise?

## Evidence that motivated it

- **R24:** arrivals are missing because new content is the mean.
- **R29:** sampling content at every innovated cell restores arrivals but adds terrain noise.
- **R31:** that noise is on cells that never left the view, i.e. at gate-fired edits (×3 over all depths, ×5 at d16). It is
  not re-entry.
- **R30:** on real contexts, the head's entering-cell content is calibrated for cows (0.91 / 0.96) and over-predicts zombies
  ×1.6-2.0 (n = 28).

## Arms

Same PI u3000 checkpoints, head (`innov_u1500.pt`), roots, context, actions, H16, 8 samples and noise streams.
- **pi_s{1,2}:** the control.
- **pi_enter_s{1,2}:** content drawn from the head only at entering cells whose world position was never seen, in the real
  context or the imagined rollout (canvas bookkeeping; the canvas content is not used). Gate-fired edits and re-entering
  cells keep PI's point-predicted token.

## Readings and verdict

Exactly PI_INNOV_PREREG's A1 / A2 / H, with pi_enter in place of pi_innov:
- **A1:** cow arrivals at d16 in [0.25, 1.5] × real.
- **A2:** cow census d16 closer to 1 than pi's.
- **H:**
  - persisting ≥ pi − 0.05;
  - seen terrain ≤ pi + 0.5pp;
  - camera ≥ pi − 2pp;
  - phantoms ≤ the real far-mob rate at every depth.

Same verdict labels. **Prediction, written now:**
- A1 holds (cows calibrated at entering cells);
- seen terrain returns to PI's level (R31);
- zombie arrivals overshoot ×1.5-2 (R30).

**Secondary:** zombie arrivals, seen_split (stayed / re-entered), decision proxies, NEW entering tile error.

## Known limits

- Re-entering cells keep PI's mean token (the canvas is a separate tested variant, R26).
- Interior spawns are not drawn.
- Fresh confirmation stays sealed.
