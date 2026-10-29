# Completed diagnosis: read-only review

2026-10-29. Frozen outputs only; no model calls, training, fixes, or recipe changes.

## Checks declared before the additional analysis

1. Verify reported F2 relative improvements from the saved raw loss arrays.
2. Check whether the position-centred R3 HUD statistic must average to zero because HUD coverage is a fixed set of screen positions. A numerical zero here is not evidence that the HUD has no value.
3. F1's original exposure is most-recent glimpse at lag k versus no glimpse in eight frames. Camera motion changes a world cell's probability of appearing in those historical windows. Random fixation centres do not make these derived exposure groups identically distributed across motion histories.
4. As a descriptive robustness check, compare a cell glimpsed at t-1 with one not glimpsed at t-1, without conditioning on the earlier seven fixations. Use the known probability under the uniform 49-centre prior, computed from the target cell's screen position at t-1, and inverse-probability weighting. Restrict to cells with nonzero exposure probability. Report both Horvitz-Thompson and normalized-weight estimates with episode bootstrap standard errors, by day/night. This estimates the effect of a random crop that includes the cell versus one that excludes it, not an isolated one-cell intervention or all possible forms of memory.
5. Also report the one-step exposure contrast within the original static-camera/terrain subset and a stricter subset with the terrain code unchanged and the player cell excluded. This does not certify identical pixels: light and rendering noise may still change.

The original eight-frame contrast remains as recorded. These post-hoc checks are sensitivity analyses, not replacement predeclared gates or a test that information is absent from the belief.
