# Independent review: HUD grounding and multi-step Mamba

2026-11-11. Frozen checks and readout audit only. No arm source, training checkpoint, canonical code, or queue changed.
The tests answer behavior and mechanics separately. A checkpoint can be correctly evaluated even if training was confounded.

## Verdict

- Full-frame grounding remains the region-general design requirement. This implementation trades substantially better danger
  forecasts against much worse coal/iron and some ordinary inventory transitions. It does not fix timing.
- Multi-step is a mixed checkpoint and **not a valid isolated test** of the auxiliary. The normal observed recurrence
  silently resets at some artificial forks, and its initialization is different. Do not adopt or discard the idea from this run.
- Sleep duration is easier to read from the exported post-action belief, but no broad native memory repair is demonstrated.

## Implementation / provenance

1. **High severity: unrebased episode indices discard carry in the observed teacher.**
   MultiStep `trainer.py:42-43` splits after the percept and passes cumulative episode IDs unchanged into the suffix.
   Functional Mamba expects the incoming carry's segment to be zero. The suffix starts at nonzero whenever a real
   reset occurred earlier in the chunk. The student gets the genuine prefix carry, while the observed target and
   regular training suffix start with a spurious reset. The erroneous carry can also propagate into later chunks.
   On the actual BF16 CUDA kernel and trained weights, the affected row is bitwise equal to a zero-carry continuation.
   Unrebased versus contiguous suffix RMS .30431; rebased .00400. Unaffected row .00211 either way. Rebasing
   `seq[:,split:] - seq[:,split-1:split]` is appropriate because this fork is between a percept and its own action.
   A future true boundary is still a reset. Exact stream replay reproduces the saved final stream queue, current
   episodes, offsets, stream RNG and separate horizon RNG:
   
   | Chunk | Affected valid student targets | Valid student targets | Same-segment G1 frames after the split |
   |---|---:|---:|---:|
   | 64 | 17,577 | 194,728 | 463,268 / 12,800,000 |
   | 256 | 16,422 | 48,659 | 1,323,278 / 12,800,000 |
   | total | 33,999 (13.97%) | 243,387 | 1,786,546 / 25,600,000 (6.98%) |

   The count measures structural exposure, not error magnitude on every frame; it excludes downstream carry effects.
   The reported resume check tests repeatability, not continuity. The published preflight already reports unequal TSP/noise
   at update 1, although it only gates on G1 and nonzero auxiliary loss.
2. **Multi-step initialization is not matched.** `model_multistep.py:28` replaces the already constructed belief
   with a fresh instance. Same seed reproduces 43 different belief tensors, none elsewhere, plus the new mask parameter.
   Identical first-step G1 is weak evidence because the denoiser output is initialized to zero. Attach the mask parameter
   to the existing belief with a private RNG, and use a split/zero-weight sham before attributing any effect to the loss.
3. **HUD is a grounding-package comparison.** The expanded head and both correspondence targets change G2, and the
   outcome initialization shifts: first-step reward CE 5.60824 -> 5.47497 and continuation .67826 -> .71490, before
   any training. HUD and multi-step use identical saved configs, data manifest and targets. All multi-step source pins
   match; HUD's current `compare.py` and `consequences.py` differ from its launch pins. The raw population IDs, strata,
   recorded rewards and terminations agree across all arms. Summary JSONs bind to their actual checkpoint SHAs.
4. **Readout workaround needs clearer checkpoint handling.** The initial multi-step eval failed loading the extra
   mask parameter into M0. The recovered selected checkpoint and latest omit it. Compared the selected file with
   `ckpt_10.orig_with_mask.pt`: *every native model tensor is bitwise equal*, only `belief.mask_token` was removed.
   That does not invalidate normal inference; the stripped file is not a valid multi-step resume checkpoint.
   Future evaluator should instantiate the proper class or create a separate inference export. HUD STATUS has duplicate
   phase keys (first failed, last complete); recovered substage status and SHA-bound artifacts, not the headline, establish completion.
5. Timing and gradient meaning: actual auxiliary is one endpoint per microbatch, H=2/4/8/16, masked intermediate observations,
   detached same-network observed endpoint, starting carry attached. It has no direct denoiser loss. Near end median weighted
   scalar contributions: multi .01754, G1 .00878, G2 .10310, G3 .01492 (not gradients). Calling .1 a weak auxiliary
   is not justified by its coefficient alone. The normal diagnostic logs omit the auxiliary route.

## Matched checkpoint results

Truth and sampling IDs match for the consequence bank and recursion; episode paired bootstrap intervals describe
case uncertainty conditional on one trained seed per arm, not run-to-run variation. Monitor G1 uses each model's own
learned percept targets. Its weighted average is .003233 / .003077 / .003001, but pure-noise contribution is
.002162 / .002093 / .002098. Multi's monitor advantage over HUD is at nonzero signal. This is not a common-unit semantic score.

| Reading | F4 | HUD | Multi |
|---|---:|---:|---:|
| Risky generated termination (recorded .1277) | .0362 | .0810 | .0555 |
| Awake cooldown contrast, fraction of real | .159 | .189 | .294 |
| Sleeping damage contrast, fraction of real | .053 | .114 | .066 |
| Coal first unit | .986 | .545 | .821 |
| Coal increments at 1–3 | .912 | .109 | .057 |
| Iron first unit | .923 | .105 | .198 |
| Iron increments at 1–3 | .902 | .035 | .059 |
| Stone increments at 1–3 | .956 | .792 | .789 |
| Stone increments at 5–8 | .715 | .720 | .661 |
| Wood increments at 5–8 | .083 | .334 | .105 |
| Free-running entities step16 (real 1.273) | .861 | .617 | .756 |
| Conditional generated/real C2ST | .9989 | .9957 | .9985 |
| TSP observed-periphery pixel nats | 2.218 | 2.161 | 2.132 |

HUD's stone 5→6 gain is real on that slice, not a high-count or universal gain. For coal 1–3, HUD−F4 paired
interval [−.822,−.783]; multi−HUD [−.067,−.036]. For iron first HUD−F4 [−.841,−.795], multi−HUD [.069,.118].
For free entity count step16 HUD−F4 [−.360,−.132], multi−HUD [.030,.267]; count does not isolate persistence or identity.
Energy9→awake is 1.0 in all three. HUD's better danger contrasts still leave a large shortfall.
All C2STs easily separate real from generated; TSP on real coarse evidence is not a test of remembered fine evidence.

Raw HUD-token MLP exact digit accuracy is already **1.0 in all three**. The original report conflates raw tokens and global
embedding. Food embedding exact accuracy .417/.610/.518; health .923/1.000/.988. Body digits are more accessible in some
global/recurrent readouts, not all. Native ordinary periphery near-night entity recall is .966/.964/.968, broadly stable.
Post-action sleep duration R² .766/.797/.865; post-percept .779/.811/.812. The latter barely changes HUD→multi,
so the post-action gain alone does not establish better retention in the underlying carry. Asleep fatigue post-action
R² .139/.261/.301; awake hunger/thirst/fatigue post-action R² remain ≤.046. Awake cooldown/hit probe AP peaks in HUD,
not multi-step. A weak binned observer is not an oracle, and a probe does not establish use or absence of a variable.
S1 achievement macro AP from corrected belief .916/.917/.918 versus .929/.923/.928 from current percept controls.
Zombie last-seen-side accuracy .500/.488/.470, last-eight controls .502/.494/.472; no general retention gain.
Selective information loss is not excluded by periphery effective rank.

## Native memory contrasts

Natural thirst exact-history alias: F4 2/128 vs 2/128; HUD 1/128 vs 1/128; multi 2/128 vs 2/128. Not an estimated
population rate. Reused the qualified 24-pair hunger bank with matched recent observations, action and fixation histories;
all readers reach 100% held-out real accuracy and exactly decode both true endpoints of each pair. F4 reader results
replicate the original. From 1,536 futures per checkpoint, due/not-due: F4 5.99%/5.86%, HUD 6.51%/6.64%,
multi 5.34%/5.47%, versus truth 100%/0%. Nearest-real-endpoint geometry independently agrees, except one harmless
F4 sample at the boundary. The tiny differences are not meaningful recovery; see PHASE_SUMMARY.json.

## Design assessment

The user's proposal is a reasonable form of multi-step self-prediction. PlaNet's latent overshooting and SPR are
precedents, not exact equivalents: PlaNet trains predictive state distributions anchored by observation/reward
models, and SPR uses a slower target encoder plus RL. PlaNet reports help for some architectures and slight harm
for its final RSSM. Ni et al. distinguish expected self-prediction from distributional sufficiency and study
stop-gradient collapse avoidance under limited assumptions. A deterministic post-mask vector need not reproduce a
realized observation-conditioned posterior. Stop-gradient and good rank do not guarantee retention of a rare fact.

A teacher need not represent an explicit clock to teach it: a future teacher output that reliably represents a
food decrement can provide an indirect timing target. Conversely, teacher and student can agree on a representation
that discards that future distinction. The current experiment cannot establish either as the root cause.

Preferred stronger form: an occasional correctly forked, action-conditioned rollout of the recurrent state, scored
against a **real future percept through the same denoiser**. Supervise a sparse endpoint, not every intermediate frame,
and maintain ordinary training and cache use. No future observed memory/cache may leak into that auxiliary query.
Use the predicted post-action export, not a post-mask vector as if it were the observed posterior. Condition on an
explicit forecast horizon. Recorded future actions can disclose intermediate observations; use supported exogenous
windows for intervention claims, and report their coverage. Train on generic observed targets and reserve hidden
simulator state for evaluation. Keep uncertainty in the forecast.

Cheapest lever before extending horizons: increase weight on genuine low-signal flow targets (clean future most
hidden), keeping bootstrap/sampling and expected flow weight fixed. At tau0 F4 uses 1/8 of flow draws and ~2.53% of
flow coefficient mass, not 2.53% of its gradient. A global G1 multiplier boosts reconstruction too and the
harmoniser adjusts; a route-only multiplier changes gradient scale but not what can be learned from the target.
History corruption is a possible complementary lever, not mandatory. Masking a region can demand reconstruction
of visible details instead of timing; contextualized cached keys and neighboring frames can leak around a naive mask.

Success must be native forecast improvements on independent history contrasts (timing, danger, older sightings/re-entry,
fine evidence when qualified), with mining, false events, ordinary rendering and recursion guardrails. P2 still addresses
self-feeding separately. No new training scheduled.

Primary sources: [PlaNet](https://arxiv.org/pdf/1811.04551), [SPR](https://arxiv.org/abs/2007.05929),
[Ni et al.](https://arxiv.org/html/2401.08898v3), [Dreamer 4](https://arxiv.org/html/2509.24527),
[History-Guided Video Diffusion](https://arxiv.org/html/2502.06764v1).
