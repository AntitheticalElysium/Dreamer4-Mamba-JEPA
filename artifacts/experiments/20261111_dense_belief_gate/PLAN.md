# Frozen gate for dense teacher-belief MSE

One masked-action Mamba scan from the fixed early fork. Score post-mask states against the observed,
uninterrupted teacher's post-percept states at **every step**, ordinary equal-coordinate/time MSE.
Compare with scoring only H=1/2/4/8/16 at that same fork. Use the original MultiStep weights at checkpoints
1/5/10 (trained mask intact). No change detector. No teacher split or implementation changes.

Reproduce the prior exact-history bank through the 16-step fork; same actions and env keys. Compute
(1) optimal separate-versus-tied teacher target MSE (quarter squared difference), and (2) actual student's
MSE vs pair-mean control, including how much of the teacher's contrast it predicts. A pair-mean arithmetic
control is not a trained no-memory world. Record all times and masks, equal both time/coordinate reductions.
Compare with the previous mild exposure foils on the same physical future. Identical-pair sham must be zero;
oracle student reproducing teacher must get zero and pay exactly the oracle gap when tied.
Test the non-timer reward-only alias with identical vision and actual outcomes; no assertion of learned
achievement histories. Evaluation only; no model retrain or new objective weight.
