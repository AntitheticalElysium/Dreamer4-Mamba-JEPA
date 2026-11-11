# Short matched continuation, no full P1 run

Two 384-update arms, 16 streams x64, from the identical P1-HUD checkpoint10. Same seed and fresh AdamW,
2e-5 base LR (32-update ramp), same encoder LR ratio, clipping, harmoniser and G1/G2/G3. All weights
trainable. Preserve the parent belief parameters and add only the same independently initialized mask token.
The auxiliary is .1 in both arms, exactly as in the original multi-step recipe for this comparison only.
Dense takes the mean over H instead of using the final step. Shared horizons {2,4,8,16}; same draws and prefix.
The observed fork rebases segment IDs to preserve the carry. No other loss changes, labels, extra output heads,
or new change detector. Both arms replay the same saved HUD fixation trace and stream draws, verified by
target/frame indices, episode and uniform reference fields. The old gaze controller is not retrained here.

Budget is 393,216 frames per arm (~half an hour total training). This tests whether adding dense supervision
helps an already trained world on a short continuation. A null does not rule out from-scratch or longer learning.
Primary: due minus not-due native hunger rates on the fixed 24 paired scenes, K32, same RNG, qualified readers,
paired episode intervals and correct/false-decrement rates separately. Compare with parent, whose locked baseline
is in 20261111_two_arm_audit/PHASE_hud.json. Native factual guards: risky, adjacent, coal, iron on first120 DEV
episodes for parent and both arms, with class exposure and paired intervals. Final checkpoints only.
No overnight escalation. Save resume checkpoints and inference exports separately. Training assets immutable.
