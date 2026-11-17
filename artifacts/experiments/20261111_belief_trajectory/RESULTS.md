# Checkpoint trajectory and belief-gradient audit

2026-11-11. Frozen evaluation only. No world weights, original arm sources, canonical code, or queues changed.

## What the runs actually learned

Fixed bank: 57,576 states from 192 DEV episodes, four physical-episode folds, same true prefixes and fixations
at every checkpoint. Read both corrected belief b and post-action export bp. The recent baseline is a fixed
eight-step *semantic* summary (body, inventory, sleep, lighting, action and outcome), not the full visual cache.
Conditioning tests compare this baseline with the same features plus bp. Ridge and shared nonlinear readers are
descriptive; these are one-seed trajectories, not uncertainty intervals or a causal training comparison.

The awake-clock cohort is restricted to an old visible decrement (>8 steps), non-extreme digit values and awake
history since that decrement. This restriction depends on recorded observations, not successful latent decoding.
Counting from the last visible reset recovers exact replayed phase on all rows: hunger 17,428 (184 episodes),
thirst 14,483 (192), fatigue 18,329 (176). Time asleep >8 steps has 6,952 rows in 168 episodes. This guards against
a weak history proxy or impossible target. No world training uses the labels.

### Sleep duration beyond the recent window

Ridge held-out R²; both exact same-seed constructors were evaluated before training. Checkpoint 5 is the end of
the 64-frame phase; checkpoint 6 follows the transition to 256-frame chunks.

| Checkpoint | HUD b | Multi b | HUD bp | Multi bp |
|---|---:|---:|---:|---:|
| Untrained | .023 | .048 | -.050 | -.045 |
| 1 | .515 | .737 | .523 | .705 |
| 3 | .715 | .617 | .755 | .756 |
| 5 | .631 | .643 | .668 | .735 |
| 6 | .607 | .636 | .628 | .737 |
| 8 | .629 | .645 | .647 | .752 |
| 10 | .619 | .627 | .633 | .739 |

Recent-only ridge R² .298; MLP .344. Final recent+bp: ridge .655/.732 and nonlinear .427/.493 (HUD/Multi).
Both clearly acquire a useful long-duration signal. Multi's early advantage emerges during training rather than
already being a high untrained score. This does not rule out initialization changing learning speed; the trained
models differ in initialization, auxiliary, and split-state handling. The late gap is strongest at the export the
denoiser receives; corrected-belief readings are close. Neither readout certifies native sleep forecast benefit.

### Awake clocks and other history

Across all six trained checkpoints and both arms, best awake-clock R² is .00059 from b, .00071 from bp, -.02119
from recent+bp ridge, -.00306 from recent+bp MLP. No observed early acquisition then late disappearance on this
bank. The history reference is 1.0. These metrics do not prove phase information is absent in all possible
representations, and exact phase is not itself the deployed requirement. The prior independent native hunger and
thirst tests remain the evidence about failure to use the distinctions.

The older-zombie-side (last seen 9–64 steps ago) nonlinear reading with recent+bp lies .502–.525, versus .533 for
recent semantics alone. Already-earned achievement flags have high absolute AP but also strong recent-controls;
no consistent advantage of multi-step across the drink and zombie flags. Thus the sleep result is selective, not
a general statement about all memory or a test of useful foveal evidence.

## Was the bootstrap too weak?

At checkpoints 1, 5, 10, six fixed real training microbatches per checkpoint (three at 64, three at 256),
ordinary uniform fixation. Correct observed prefill constructs the incoming carry; training objective and weights
are original. Compare original vs rebase with same noise, start/horizon draws, weights and input. Measured weighted
gradient vectors on **shared belief parameters only**, excluding the mask token, before clipping and Adam.
This is a local frozen counterfactual, not a replay of historical updates or a population gradient estimate.

| Checkpoint | Median aux norm / combined G1+G2+G3 norm | Same, rebased | Median aux norm / G1 norm |
|---|---:|---:|---:|
| 1 | .835 | .835 | 89.6 |
| 5 | .544 | .491 | 36.0 |
| 10 | .993 | .992 | 50.4 |

Range of original aux/base across chunks .059–2.138; no claim of uniformly dominant supervision. Aux/G1 cosine
is close to zero on these chunks (roughly -.04 to +.05), not a demonstrated systematic conflict or aligned
predictive signal. Rebasing changes individual results on affected rows; untouched cases reproduce to numerical
precision. The auxiliary is a substantial source of pressure even when the immediate reset is corrected.
Its small scalar coefficient does not make it negligible. This answers magnitude, not whether those gradients
historically caused the learned representation.

## Implication for the design

Memory depth and forecast horizon are different. To predict a decrement next step, Mamba may need evidence from
20 steps earlier; prediction can still be one-step. If a recurrent representation is sufficient for conditional
next observations and rewards, longer-range predictions can be composed (Subramanian et al., AIS). Multi-step
auxiliaries are an optimization aid, not an architectural requirement. Even in PlaNet, overshooting helped some
dynamics models and slightly hurt its final RSSM. Predicting a fixed endpoint need not require the phase: one full
period gives the same next digit count regardless of which tick the decrement occurs on, away from saturation.

The user is right that a generic endpoint loss can still be dominated by easy coordinates. G1 does give a real
one-step incentive in the exact-history case: identical recent input, differing actual outcome, hidden future target.
But this reward competes with many easy forecasts, and high-signal denoising can read the answer. The natural
output need only distinguish consequences, not reconstruct named simulator clock variables.

Priority for a practical repair: keep the cache and strengthen the **actual next-percept forecasting** part of G1
where the target is least visible, before adding a blind rollout. Keep changes generic and the unchanged sampler,
data and outcome/grounding paths as controls. If later using delayed endpoints, choose tests that depend on the
information we want retained and validate native one-step forecasts. No evidence here that increasing the old
self-consistency coefficient or forcing longer forecasts will fix timing.

To identify causal value of the existing auxiliary: rebase the observed fork, preserve base initialization, and
compare on/off with identical fork arithmetic, chunks and random draws. Do not judge by the auxiliary loss,
readability or denoiser monitor alone. No historical checkpoint manipulation can reconstruct that training comparison.

Sources: [Approximate Information States](https://jmlr.org/papers/v23/20-1165.html),
[PlaNet Appendix D](https://arxiv.org/pdf/1811.04551), [Dreamer 4](https://arxiv.org/html/2509.24527).
Raw data: TRAJECTORY.json, UNTRAINED.json, GRADIENT.json. Figure: belief_trajectory.png.
