# Stage-A DEV investigations, 2026-08-02 to 2026-08-03

**Not a Stage-A result.** A DEV smoke to find out whether the pipeline learns and
whether the failure modes the register worries about appear. The preregistered
FINAL protocol (S52) was not run: 16 seeds rather than a sealed set, an 800-step
cap rather than Craftax's native 10000, one training seed per arm. The FINAL seeds
are untouched.

## 20k dynamics rerun, 2026-08-03

Setup: the same frozen tokenizer and data split; per arm 20,000 dynamics, 2,500
agent and 800 actor steps. This run includes the S67 row-wise shortcut schedule,
continued bootstrap clock, S68 direct-horizon cap and S69 widened BC sampling.
The raw report is `stage_a_olddesign/report_20k_preterminalfix.json`.

| arm | actor | BC | actor − BC (95% paired interval) | chosen h | h=4 rollout | reward MAE | terminal continuation |
|---|---:|---:|---:|---:|---:|---:|---:|
| flow-attention | 1.62 | 3.53 | −1.91 [−2.68, −0.50] | 16 | 0.087 | 0.065 | 0.992 |
| flow-mamba | 1.77 | 3.31 | −1.54 [−2.07, −0.31] | 8 | 0.137 | 0.056 | 0.924 |
| direct-attention | 4.41 | 5.46 | −1.05 [−4.32, 2.29] | 2 | 0.083 | 0.037 | 0.903 |
| direct-mamba | 3.19 | 4.84 | −1.65 [−2.91, 0.22] | 2 | 0.084 | 0.034 | 0.940 |

All actors are below their own BC prior; the two Flow failures are significant
on these 16 DEV seeds. All actor and BC episodes terminate, with mean actor
lengths 116–132. Longer dynamics training removes the former universal rollout
collapse at h=4, so that result does not explain the universal policy regression.

### Environment-fork localization

At 36 real states from eight BC trajectories, every one of Craftax's 17 actions
was executed from the same simulator state. The encoder preserves the resulting
variation and each transition model predicts the matched action better than an
off-action successor: effect cosine 0.67–0.74, matched MSE 0.075–0.081 against
0.117–0.133 off-action MSE.

The first universal failure is downstream. True immediate death probability
under the learned actors is 13.1–14.4%, while the continuation heads predict only
0.046–0.070% death even when given the **true encoded successor**. Reward
correlation across counterfactual actions is 0.02/0.13/0.34/0.35 on true
successors and degrades further on generated successors. Logged-action MAE did
not expose either failure. Raw measurements are in
`stage_a_olddesign/counterfactual_forks_preterminalfix.json`.

This locates the common failure in Phase-2 outcome supervision: the actor can
move away from the logged action support while its frozen reward and continuation
models treat those alternatives as rewarding and nonterminal. Phase 3 then
optimises the extrapolation error. Flow retains an additional shortcut-ladder
problem, but that cannot explain both Direct arms failing.

## 5k dynamics run, 2026-08-02

Setup: one tokenizer for all arms (S20), 3000 steps; per arm 5000 dynamics, 2500
agent, 800 actor. 344 train / 44 dev episodes, 260,081 transitions, 274 terminals,
76 BC-eligible. Phase 1A restored from checkpoint four separate times and
reproduced the cache digest `a62705fcfbace70c` every time.

## Results

| arm | actor | BC | random | actor − BC | horizon | rollout informative | γ·growth | reward MAE | zero | contraction |
|---|---|---|---|---|---|---|---|---|---|---|
| flow-attention | 2.18 | 2.33 | 1.17 | −0.15 | 4 | **no** | 1.085 | 0.1019 | 0.1013 | 1.421 |
| flow-mamba | 1.10 | 2.90 | 1.17 | −1.80 | 4 | **no** | 1.149 | 0.0878 | 0.1013 | 1.112 |
| direct-attention | 4.72 | 3.26 | 1.17 | +1.46 | 8 | yes | 1.235 | 0.0475 | 0.1013 | 0.959 |
| direct-mamba | 2.63 | 5.70 | 1.17 | −3.08 | 8 | yes | 1.265 | 0.0504 | 0.1013 | 0.957 |

**No arm beats its own BC prior.** Three are worse than the behaviour they cloned.
Across two runs of the same design direct-mamba moved from +0.76 to −3.08, so at
16 seeds and one training seed these gaps are noise-dominated and no ordering
between arms is established either.

## The finding that explains the rest

The rolled prediction error, against the marginal predictor (the constant mean
latent) that S63 compares to:

| arm | h=4 | h=8 | h=16 | h=32 | marginal |
|---|---|---|---|---|---|
| flow-attention | 1.189 | 1.302 | 1.405 | 1.507 | ~0.30 |
| flow-mamba | 0.760 | 0.821 | 0.836 | 0.825 | ~0.30 |
| direct-attention | 0.148 | 0.220 | 0.356 | 0.453 | ~0.30 |
| direct-mamba | 0.151 | 0.254 | 0.361 | 0.527 | ~0.30 |

**Both flow arms roll out worse than predicting the mean latent, at every
horizon.** Phase 3 for those arms imagines on trajectories carrying less
information than a constant, which is a sufficient explanation for RL not helping
there and needs no appeal to reward quality. Direct is informative to h≈8 and
crosses the marginal between 8 and 16, which is what S63 selects on.

The former claim that `γ·growth > 1` disproves Lemma 1's Lipschitz hypothesis is
withdrawn. Rolled-error accumulation is not a Lipschitz constant, and sampled
Flow dynamics also violates the lemma's deterministic premise. The values remain
descriptive only.

## Phase 2 damages the world model

Mean rolled error over 32 steps, same DEV batches, same world before and after
Phase 2:

| arm | after Phase 1B | after Phase 2 |
|---|---|---|
| flow-attention | 0.648 | **1.347** |
| flow-mamba | 0.683 | 0.828 |
| direct-attention | 0.323 | 0.327 |
| direct-mamba | 0.329 | 0.373 |

Every arm gets worse; flow-attention more than doubles. Phase 2 continues the
dynamics loss alongside three head losses, all normalised to unit RMS by
`_balance`, so dynamics carries roughly a quarter of the weight it had in Phase 1B
while head gradients reshape the world through the agent tokens' inputs. Joint
training is what D4 does; this balance is ours, and it is the next thing to
examine.

Note the flow arms are already above the marginal after Phase 1B alone (0.648,
0.683 against ~0.30), so Phase 2 compounds a problem it did not create.

## Corrections to earlier claims

**S61 is withdrawn.** I reported flow's reward models as carrying no information
against a 0.0795 zero baseline. That baseline was measured on *uncached* DEV
episodes with a burn-in of 30 and their own short/long mix, while the model MAE
came from *cached* DEV batches, and it ignored `reward_rows`. On the matched
baseline (0.1013) flow-mamba beats zero at 0.0878 and flow-attention is marginal
at 0.1019. The baseline is now computed inside `head_calibration` on exactly the
rows it scores, so the two cannot disagree again.

**"Three quarters of updates use short windows" is false.** Training passes
`total=steps`, so the long-only tail applies: the real split is 56.3% short,
43.7% long. The per-window BC coverage figures reproduce (10.9% short, 21.4%
long), but the blend is 15.5% reachable, so **84.5%** of expert behaviour can
never be a BC target, not 78.7% (S65).

**"Both substitutions help" was too strong** and is dropped. Arms now select
different horizons, so an actor comparison no longer varies only the transition
and the time mixer.

## What changed in the code

- Head output scales follow the pinned DreamerV3 config (S64): `reward` and
  `value` at 0.0, `policy` at 0.01, `continuation` at 1.0. We had shipped PyTorch
  defaults on all four.
- The horizon criterion is S63 above, replacing a rule that was *tighter* for a
  more accurate arm and degenerated to "no candidate qualified" on three arms.
- `latent_stats` predicted from one context block; it now uses the full context.
- `head_calibration` honours `reward_rows` and reports its own matched baseline.
- The runner keeps raw episode rows, which S52 requires.

## Incidents

Two Mamba Triton OOMs (in `cost`'s backward timing and in Phase 2's backward);
the first is caught, the second handled by one process per arm. A phase whose
length was not a multiple of `checkpoint_every` never saved its final model. One
run was killed by launching it and polling it inside the same shell invocation, so
a tool timeout killed the process group -- it is now launched under `setsid`.
Checkpoints were migrated twice for `Config` fields that provably affect only
Phase 3, asserted field-by-field rather than retraining the tokenizer.
