# PI + sampled innovation content: does drawing new content from a categorical restore arrivals? Preregistration

2026-10-15 night. Written before any code for this study exists. Track B; one change relative to the trained PI world.

## Question

PI never imagines arrivals: mobs walking in, new content at entering cells (R24). Is the cause that an innovated cell's
content is a point prediction (the mean), and does drawing it from a learned categorical distribution fix this without
harming persistence?

## Evidence that motivated it

- **R24:** arriving mobs per frame at d16 (type absent at the root) are real 0.116 zombies / 0.259 cows. PI draws
  0.006 / 0.023 and 0 / 0.002. Continuing mobs are kept at 61-80%.
- **R28:** sampling the change *decision* while the content is the mean token makes ghosts (phantoms ×4-5, vanish ×2.5).
- **R25:** K 4096 k-means codes over Raw tokens (E2's codebook) fit the current pool. Centroids keep 94 / 97 / 85% of
  zombies / cows / skeletons, with ≤ 0.04 false mobs per 1,000 cells and 99.4% tile agreement.
  - In PI, unchanged cells are copied and never quantized, so E2's flip failure cannot arise.
- **Dedieu et al. 2025** (arXiv 2502.01591, read 10-15): on our exact Craftax-Classic frames, all 81 per-patch tokens are
  predicted in parallel with independent categorical readouts and sampled. Continuous patch prediction collapses MBRL reward
  67.42 → 21.20.

## The change

- **Content head:** a linear head (256 → 4096 logits) on the frozen PI u3000 belief at each cell. Head-only training; PI is
  frozen.
- **Training:**
  - cross-entropy toward the code (nearest centroid) of the pre-chance target z_pre;
  - at PI's training-time innovation cells only: posterior edits (q > 0.5) and entering cells, computed exactly as in
    `pi_train.losses`;
  - seeded training-ledger windows (seed 20261020), batch 16, 1,500 updates, Adam 1e-3, zero init;
  - one run per PI seed.
- **Recursion:** exactly PI (MAP gate, exact copies, hard scroll, chance operator, same noise stream), except that every
  innovated cell (gate > 0, or entering) takes the centroid of a code drawn from softmax(head) at temperature 1, from its
  own generator (seed 20261021).
- **Name:** `pi_innov_s{1,2}`, in `pi_canvas.py`. Control: `pi_s{1,2}` (existing samples).

## Readings (bright, 8 samples, H16, both seeds)

- **A1 (primary, arrivals):** cow arrivals per frame at d16 (`census_split.py`, cow, type absent at root) lie in
  [0.25, 1.5] × real (real 0.259 → [0.065, 0.389]).
- **A2 (primary, census):** cow census d16 closer to 1 than pi's.
- **H (no harm):**
  - persisting-mob score d16 ≥ pi − 0.05;
  - seen-terrain error d16 ≤ pi + 0.5pp;
  - camera consistency d16 ≥ pi − 2pp;
  - phantoms per frame ≤ the real far-mob rate (`real_phantoms_per_frame_by_depth`) at every depth. Arrivals are "far" by
    definition, so the bar is reality's own rate, not 0.
- **Verdict:**
  - **"Sampled innovation restores arrivals"** if A1, A2 and H hold at both seeds.
  - **"Arrivals restored but the rollout is harmed"** if A1 holds at both seeds and H fails.
  - **"Not supported"** if A1 fails at either seed.
- **Secondary:**
  - zombie arrivals (real 0.116);
  - NEW entering tile error (sampling raises per-cell error against one real future; reported, not gated) and NEW entering
    tile-class distribution vs real;
  - zombie / cow census;
  - decision proxies (zombie-adj, cow-adj, health AUC16);
  - head cross-entropy and top-1 accuracy;
  - innovated cells per step.

## Known limits

- **Interior spawns are not addressed:** the MAP gate does not fire on unchanged-looking cells. Only arrivals through
  entering cells and content at gate-fired edits change.
- Codes are over contextual tokens. A centroid at an innovated cell is placed beside exact copies.
- Head-only, one budget, no tuning of K, temperature or budget. Fresh confirmation stays sealed.
