# M0 revision review, 2026-10-27

Read-only review at c1bfd2d0. No model training or architecture edits.

1. Reproduce the committed pooled-separation EDA on its original 300 dev episodes, and compare identical pairwise summaries for map and HUD. The published script reports the median of all map pair distances but the median of per-position HUD minimum distances. Record all distances and exact prototype collisions without interpreting these as learned readability or actor benefit.
2. Inspect the transformer timing stand-in and compute its temporal dependency reach. A four-frame attention mask repeated across several layers need not define a four-frame conditional model. This is a graph check, not a throughput or learned-performance test.
3. Distinguish the raw-pixel conditional entropy from information retained by the proposed encoder, and independently verify the paper-level claims used to change the backbone and curriculum.

Existing train/dev diagnostic data only; no final-split evaluation. Preserve the current design and all negative evidence.
