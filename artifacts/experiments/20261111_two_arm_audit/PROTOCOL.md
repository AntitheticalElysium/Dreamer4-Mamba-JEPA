Review only; no training. Replay the actual stream/horizon schedules and check final RNG states. Reproduce the
observed teacher split with a prior reset and a later genuine reset, compare continuous versus unrebased versus
rebased continuations, and compare the affected row to zero carry. This establishes whether the bug occurred and
how often, not how a corrected training run would score.

Native check: evaluate all three checkpoints on the existing 24 hunger histories with byte-identical recent
observations and opposite true outcomes. Same actions, fixations, sampling seeds and K=32 as the original test.
Fit a separate reader per checkpoint on other episodes, require >=99% held-out real accuracy and exact readings
on both real outcomes of every pair. Compare native event contrast and nearest-real-endpoint geometry. F4 is a
replication control. Existing DEV only; no tuning, no automatic experiments after this review.
