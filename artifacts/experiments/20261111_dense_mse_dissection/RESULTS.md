# Did the continuation start to learn hunger timing?

## The native forecast

Both final checkpoints drew a food decrement in 76/768 futures for each phase, in all 24 pairs with exactly equal pair counts. The simulator decrements on every early-phase member and none of the delayed-phase members at this fork. The continuation raised both arms' draw rates together from about 6.5% in the HUD parent to 9.9% at update 2,000. The real-frame digit reader qualified on the fixed bank. See [the completed-run audit](../20261111_dense_mse_2k_review/RESULTS.md).

The frozen replay looks below the reader threshold. The last eight percepts, the NOOP query action, gaze and random draws are identical within each history pair. Prior observations distinguish the phases. At four leads to the early deadline, project the forecasted food token along the difference between the two real successor tokens. One unit is the full real digit step for that scene and checkpoint. All five due-fork runs reproduce the saved full periphery outputs bitwise across all 7,680 draws. Intervals resample the 24 scenes, keeping each scene's 32 paired draws together.

| Checkpoint | First denoiser estimate at deadline, early minus delayed (% of digit step) | Final token at deadline, early minus delayed (% of digit step) |
|---|---:|---:|
| HUD parent | +0.001 [−0.020, +0.024] | −0.020 [−0.074, +0.027] |
| Endpoint 384 | −0.025 [−0.056, +0.007] | −0.021 [−0.069, +0.017] |
| Dense 384 | +0.005 [−0.024, +0.034] | −0.016 [−0.101, +0.043] |
| Endpoint 2,000 | +0.007 [−0.022, +0.035] | +0.013 [+0.0002, +0.029] |
| Dense 2,000 | +0.021 [−0.016, +0.064] | −0.027 [−0.080, +0.016] |

The dense arm has a tiny positive point estimate at the first call. Its change from update 384 is +0.016% of a digit step [−0.027%, +0.064%], and the sign is reversed in the final token. At 2,000 updates 9/24 scenes have a positive final mean in dense and 11/24 in endpoint. The endpoint's positive final mean is also tiny, exploratory and conditional on this bank. It does not change a digit on this test. The last two columns are token geometry, not probabilities. The differences establish that the native outputs are not bitwise insensitive to earlier history.

A shared pattern does appear. Averaging over phases, the final food token in dense 2,000 sits about 0, 0, 6 and 9% of a digit step toward the decrement at 8, 5, 2 and 1 steps before the early deadline. The HUD parent already had a similar pattern at about 0, 0, 3 and 6%. At the last fork dense predicts a 9.21% and 9.24% *mean token displacement* for the early and delayed histories respectively. The trajectory in both histories is consistent with an imminent drop cue shared by the pair. It gives no evidence that the model uses the older observation to time the decrement. Across all 20 checkpoint/fork combinations the mean signed final phase response is at most 0.042% of a digit step in absolute value.

## Other native forecasts

The same factual guard panel gives parent → 384 → 2,000 forecast rates. Both continuations train the full world with G1–G3 and the auxiliary. A comparison to the parent alone cannot attribute any improvement to the auxiliary.

| Guard | Recorded | Endpoint | Dense |
|---|---:|---|---|
| Risky termination | 14.71% | 8.12 → 6.03 → 11.34% | 8.12 → 5.84 → 9.61% |
| Adjacent negative reward | 15.70% | 8.41 → 7.55 → 7.94% | 8.41 → 7.48 → 6.53% |
| Coal on recorded unlock histories | 100% | 61.75 → 70.20 → 73.14% | 61.75 → 74.27 → 62.97% |
| Iron on recorded unlock histories | 100% | 16.45 → 17.18 → 23.66% | 16.45 → 16.81 → 22.43% |

Risky's closer marginal rate in dense does not establish better timing. AUROC of the saved coarse risk bins (assigned with the first two futures per history) is 0.701 parent, 0.735 endpoint 2,000 and 0.695 dense 2,000. Descriptive only, with 442 histories and a noisy score based on two draws. Dense coal's early lead over endpoint disappears. At 2,000, feeding real successor percepts gets the same heads to 97.4–97.6% for coal and 94.9–95.0% for iron, versus 63–73% and 22–24% under generation. Termination on the recorded successor remains essentially exact. This locates much of the guard gap in generated inputs. The heads on real successors also lost some coal/iron accuracy since the parent, so the entire change should not be ascribed to the sampler or denoiser. Mining readings are on realized unlock histories and do not measure false positives.

## What the auxiliary learned

The auxiliary test replays the same fixed fork and then rolls Mamba forward for 16 steps on actions and mask tokens. An uninterrupted teacher reads the actual percepts and outcomes. The two histories have their early and delayed decrements at steps 8 and 16. Only 23 scenes survive to the end, since episode 84 terminates. The measurements use each checkpoint's own teacher. The whole state changes during training, so the columns do not measure student progress against a fixed latent representation.

For each pair, average the two student forecasts and put that average back in both branches. This preserves what the two forecasts have in common and erases only their difference. A positive erasure cost means the branch-specific forecast helped match the teacher. The final column is the dot product of student and teacher contrasts divided by the teacher contrast's squared norm, aggregated across scenes and steps.

| Checkpoint | Whole blind-rollout MSE | Erasure cost | Erasure cost as % of tied MSE | Student contrast along teacher |
|---|---:|---:|---:|---:|
| Endpoint 384 | .5230 | .0000071 | 0.0014% | 0.40% |
| Dense 384 | .4634 | .0000114 | 0.0025% | 0.56% |
| Endpoint 2,000 | .3896 | .0000417 | 0.0107% | 1.97% |
| Dense 2,000 | .3484 | .0000517 | 0.0148% | 2.43% |

This is a small learned history-dependent effect in the blind branch. Dense matches the teacher's history contrast somewhat better by 2,000, with a scene-bootstrap interval of [.0000440, .0000586] for its erasure cost. The improvement is shared by both arms. By exact squared-error decomposition, 99.5% of the dense 384→2,000 reduction (.1150 total) is in matching the *pair mean*. The contrast-error reduction also reflects movement of the teacher's targets. The small erasure cost is an independent within-checkpoint measure of how much the student actually benefits from expressing the difference.

The time decomposition explains where the contrast helps. At dense 2,000 it helps on steps 8–15, when the observed food digits differ, by .000127 MSE on average. Over steps 1–7, tying the forecasts slightly *lowers* error by .000022. At step 16, after both have decremented, it also lowers error by .000030. Giving the teacher identical future inputs, while leaving its distinct prior histories intact, removes 99.3% of its average squared separation. That same-future-input check measures what generated the teacher target contrast. It leaves the student and native forecasts untouched. Dense latent matching mostly trains imitation of a belief that has already seen the evidence, with little reward for getting the timing right in advance.

## Held-out belief accessibility

Same 192 DEV episodes and 57,576 states as the earlier trajectory study, with episode-held-out folds. The awake timer labels require the last visible decrement to be older than eight steps. The elapsed-time-from-last-decrement rule recovers the exact simulator phase for every qualified hunger, thirst and fatigue state. This establishes that the full observable history contains the phase in these slices. The rows used for each timer are 17,428, 14,483 and 18,329 respectively.

| Belief readout, linear probe on post-action belief | Parent | Endpoint 384 | Dense 384 | Endpoint 2,000 | Dense 2,000 |
|---|---:|---:|---:|---:|---:|
| Hunger phase, R² | −.013 | −.013 | −.012 | −.012 | −.012 |
| Thirst phase, R² | −.018 | −.019 | −.018 | −.019 | −.018 |
| Fatigue phase, R² | −.011 | −.009 | −.008 | −.008 | −.009 |
| Elapsed sleep, R² | .633 | .662 | .673 | .606 | .604 |
| Drink already earned, AP | .970 | .972 | .972 | .971 | .970 |
| Zombie already defeated, AP | .955 | .957 | .958 | .955 | .953 |

Nonlinear readers given the post-action belief plus eight-step semantic summaries stay at or below R² zero for all awake timers. Elapsed sleep is a positive control. Its early rise in both arms disappears by 2,000; dense still exceeds the corresponding recent-only controls, whose R² is .298 linear and .344 nonlinear. The recent semantic baseline already reads much of the achievement information. For unseen-zombie side, post-action belief accuracy stays near .504, versus .516 from the recent linear baseline. These probes estimate accessibility to the fitted readers. The phase-pair intervention above measures native forecast sensitivity. The results do not locate all information in the internal recurrent carry or isolate its contribution from downstream readout.

## Decision

The implementation can learn a blind rollout and can begin to preserve a weak history-dependent latent contrast. Across these two continuations, applying the auxiliary at every step mostly improves generic latent fit. It does not develop a reliable phase-specific native forecast and has no consistent guard advantage over endpoint supervision. Use native history-conditioned forecasts as the next objective's acceptance test. Erasure should have a measurable cost on several consequences, including a case beyond HUD timers, while the objective itself remains label-free. These results are one continuation seed per arm. They do not compare against an equally long continuation with no auxiliary, establish that a different objective would succeed, or establish that Mamba cannot learn the relevant state.

Reproduction uses `python -m artifacts.experiments.20261111_dense_mse_dissection.native`, then `belief` and `summarize` in the same package. Use the repo environment with `MAMBA_DETERMINISTIC=1 JAX_PLATFORMS=cpu OMP_NUM_THREADS=2`. `NATIVE.json` and `BELIEF.json` contain checkpoint hashes, scene-level data and reader scores. `SUMMARY.json` contains the paired bootstrap and decompositions. No world weights were updated.

## Follow-up: next-step timer events

The exact-counter regression can miss threshold information. `due.py` fits linear and nonlinear binary
readers for whether food, drink or energy will decrement next, on the same episode-held-out folds.
The current digit is non-extreme, the last visible decrement is older than eight steps, and the agent has
remained awake. Exclude DO and SLEEP from the current action. Phase-plus-one equals the recorded decrement
indicator on every qualified row. Exact phase is only a label check. The inputs are recent semantics, or those
same semantics and the post-action belief. No model training. See `DUE_PLAN.md` and `DUE.json`.

| Next decrement | Events / rows | Recent linear AP | Parent linear AP | Endpoint 2,000 linear AP | Dense 2,000 linear AP | Recent MLP AP | Parent MLP AP | Endpoint 2,000 MLP AP | Dense 2,000 MLP AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Food | 652 / 13,927 | .04759 | .04789 | .04866 | .04869 | .04797 | .04736 | .04751 | .04718 |
| Drink | 830 / 11,580 | .07390 | .07295 | .07284 | .07344 | .07590 | .07417 | .07422 | .07482 |
| Energy while awake | 549 / 14,686 | .04283 | .04267 | .04260 | .04280 | .03985 | .04208 | .04172 | .04188 |

Linear hunger Brier improves against parent by .000073 endpoint and .000066 dense. Dense's paired episode
interval is [−.000111, −.000020]. Neither beats the linear recent-only Brier (.04520) or the episode-held-out
constant (.04463). The nonlinear hunger result does not improve. The thirst and fatigue Brier changes versus
parent include zero for both reader families and both final arms. Nonlinear thirst AP rises .00065 in dense
but remains below recent alone. These are conditional probe-fit comparisons, with one world seed and one
fixed fit per reader. Small score changes do not isolate causal timing information.

In the prior continuous-phase readouts, post-percept and post-action beliefs both remain below zero R²
for hunger, thirst and awake fatigue at all five snapshots. There are small movements within that range.
Post-action elapsed-sleep R² is .633 parent, .673 dense 384 and .604 dense 2,000. The post-percept reading
moves .619 → .652 → .520. Sleep duration was present before continuation and its early improvement fades
in both arms. Recovery and the periodic energy increase while asleep were measured in the separate full
MultiStep run, using a broader population. Its recovery AP moved .0460 → .0490 and energy-up-asleep AP
.1596 → .1669 relative to HUD, with confounded training lineage. Those are separate targets from sleep duration.

The native thirst replay used the earlier naturally occurring pair in DEV episode 376 (t658 and t654).
The eight recent frames, actions, outcome inputs and perceived tokens are exact matches. True next drink
is 4 versus 5 from a current digit of 5. HUD parent draws 4 once in 128 samples on each member.
Both 2,000-update exports draw 5 in all 128 samples on both members. The reader scores 1.0 on its
held-out real frames for all three models. This adds evidence on one known thirst pair. It leaves
native awake-energy timing and other thirst situations open. See `THIRST_endpoint.json` and
`THIRST_dense.json` for full counts and pinned export hashes.
