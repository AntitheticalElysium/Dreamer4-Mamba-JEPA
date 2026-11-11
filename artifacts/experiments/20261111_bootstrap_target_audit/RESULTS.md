# Does blind self-prediction reward the needed distinction?

Frozen evaluation only, 2026-11-11. No production or historical source changes, no training.
Loaded original checkpoints 1, 5, 10 with their trained mask tokens. All positive endpoint and exact recent-input
checks pass. RAW.json retains every scene; SUMMARY.json includes background-bootstrap intervals.

## What the auxiliary can and cannot tell us

On 80 ordinary DEV windows (40 each at H8 and H16), the blind student uses older evidence to match its observed
teacher. At checkpoint 10, full carry vs carry rebuilt from the same eight recent inputs gives endpoint MSE
.1980 vs .3549 (H8), .2632 vs .4119 (H16); full carry wins 38/40 at each horizon. Rebuilding a recent-only carry
is a diagnostic intervention, not a native matched world. The benefit is in matching a learned latent target,
not established better prediction of consequences. Both routes receive the same recorded future actions.

On the exact 24 hunger pairs, future NOOP actions and the eight recent full-resolution inputs at the fork match.
At the endpoint a real decrement distinguishes the histories. Uninterrupted teacher endpoint difference
(median RMS) .12057 at checkpoint10; equalizing just the final percept input leaves .00846. The teacher target
is not blind to the outcome; it largely distinguishes the histories **after seeing it**.

For pair student outputs s+/s-, teacher outputs y+/y-, MSE = error of the two pair means plus
one quarter of the squared contrast error, averaging coordinates. Replacing both student outputs by their
pair mean is an exact arithmetic control, not a learned or deployable no-memory predictor.

At H8, checkpoint10 and correct teacher:
- Blind student difference projects .0970 of the teacher's endpoint difference in the required direction
  (scene-bootstrap95% [.0888,.1044], conditional on this model and paired bank). It was .0197 at checkpoint1.
  This is weak anticipatory alignment, not a probability, exact clock decoding or a causal auxiliary effect.
- Actual MSE .192625; equalizing the student's two outputs gives .193192, a **0.293%** difference.
- Even perfectly recovering the full teacher contrast at that same pair mean could save only .003484 MSE,
  1.80% of the pair-mean control's loss. These are conditional arithmetic contributions, not a general bound
  on training or achievable loss.
- The unrebased teacher counterfactual yields similarly weak alignment .0863 and only .323% MSE saving.
  Holding students fixed, fixing the teacher does not make the objective strongly discriminative here.
  This does not predict what retraining with the bug fixed would learn.

Taken with the previous native result (~zero food/thirst contrasts), this rules out neither memory loss nor
decoder underuse in general. It shows that the learned blind branch is not entirely history-insensitive,
that a small useful contrast survives there, and that optimizing the current average endpoint loss offers
little incentive to get that distinction right. There is no evidence that the auxiliary reduced native
generator reliance across the board. F4 and HUD lacked the multi-step reset bug and already failed those
native timing tests, so that bug is not a sufficient explanation for the broader world-memory problem.

## MSE and balancing

An MSE target predicts the conditional mean in the chosen latent coordinates. It can sensibly align a
deterministic shared component or an expected feature. It need not preserve which future is possible, or an
uncertain posterior over branches. More fundamentally, a jointly learned target can omit/attenuate distinctions
that matter to the deployed forecast. A different norm, EMA or stochastic consistency alone does not choose the
useful features. Stop-gradient theory has scope limits; see [Ni et al.](https://arxiv.org/html/2401.08898v3).
[Denoised MDPs](https://proceedings.mlr.press/v162/wang22c.html) is related motivation for grounding latent
dynamics by consequences, not evidence for this architecture or for reward-only grounding here.

The current separate auxiliary coefficient .1 already gives a shared-belief gradient comparable to the base
groups (previous trajectory audit). Bundling it into a HarmonyDream group multiplies both terms by the same
group weight; it does not change their coefficient ratio inside that group. A separate group can amplify it.
Using the implemented rectified formula, stationary sigma solves sigma² = mu(1+sigma).
For observed last-40-log-entry raw aux mean .17962, a separate group on raw loss gives coefficient 1.912;
a separate group on .1*loss gives effective raw coefficient .698. This illustrative equilibrium holds other
quantities fixed; it is not a forecast of optimization. Neither caps belief gradient magnitude.
[HarmonyDream, Appendix B](https://arxiv.org/html/2310.00344v3).

## Proposed contract, not an adopted implementation

Keep Mamba as the latent dynamics and inference component; let it advance action-conditioned beliefs through
missing observations and correct them when evidence arrives. Keep the deployed denoiser one step at a time
with its recent visual cache. Long latent rollouts can be trained without rendering every step. The state export
must be accountable to prediction of observable futures and outcomes, including where older evidence changes
them. Hindsight grounding and raw state-vector agreement do not enforce that contract.

Use real, causally available forecast targets as the anchor (observed percepts/change and outcomes), and latent
consistency as a supporting objective with a controlled weight. A predicted prior should be able to express
uncertainty rather than be forced to equal each realized posterior. Verify native one-step use of state; success
of a training-only head is insufficient. Fix the fork and matched initialization before any causal comparison.
No overnight run launched.
