# P2 data preparation, 2026-11-02

P1-F1.5 used the full P1 training split (monitor excluded), not the exogenous-window
bank. The bank cannot explain its regressions. No corpus change has been made to F2.

## Why the exclusion exists

Future expert actions react to the recorded hidden state and stochastic outcome.
Reusing them as an externally imposed action sequence for a different imagined future
can leak information. The collector's uniform-random branch is independent of those
outcomes. This is the reason for the P2 trajectory-score filter; it is not a P1 filter.
See Ortega et al., [sequential intervention argument](https://arxiv.org/html/2110.10819v1).
An observationally trained predictor can still have causal confounding; identifying the
random branches does not magically repair its training on the rest of the corpus.

## Terminal-inclusive bank, using existing records

`terminal_audit.py` verifies the original full 2,500-step seed tree: all 705,646 random
actions agree with the stored actions, and all 2,432,119 training-step flags agree with
the saved exogeneity labels. The full key-tree length matters: splitting fewer keys
would generate a different experiment.

Admission is now prepared from a start's precommitted H random flags, including flags
after recorded death. Starts must be before recorded death/truncation and fit inside
the original 2,500-step external limit. Pad actual deaths, never time truncations.

| H | Complete-window starts | Terminal-inclusive starts | Added early-terminal starts |
|---|---:|---:|---:|
| 2 | 402,557 | 405,433 | 2,876 |
| 4 | 290,447 | 296,757 | 6,310 |
| 8 | 260,599 | 274,077 | 13,478 |
| 16 | 244,096 | 272,809 | 28,713 |

At H=16 each terminal position now has 1,914–1,915 eligible windows. Previously positions
1–15 had zero. This is an admission correction, not a learned-model result.
`terminal_inclusive_h16_bank.npz` contains loader-ready (shard, slot, start, recorded_T,
recorded_death) references. No raw corpus files or canonical P2 loader were changed.

Required loader contract:

* Regenerate the SAME precommitted random actions for the entire H-step horizon.
* A generated branch surviving past the recorded death continues those actions; do not
  switch it to NOOP because the recorded branch died.
* The recorded reference absorbs at actual termination: masked percepts, zero later
  reward, termination one. Each generated branch absorbs on its own sampled termination.
* Parent-episode train/dev/final separation stays unchanged. No start admission by
  future reward, survival, predicted confidence or event success.
* Before training, check the loader's windows against these independent counts and
  reject an intentionally survival-censored bank.

## Advanced-state support remains thin

The correction adds deaths but cannot manufacture events before them. At H=16 the
event union still contains one coal unlock, zero iron unlocks, zero zombie defeats.
At H=2 it contains 40 / 12 / 62 respectively. Using a descriptive advanced-start
criterion (five achievements already unlocked), only 5,961 H=16 starts qualify, with
the same one coal event and no iron/combat event. This criterion is not a competence
guarantee or a proposed training sampler.

If broad advanced-state P2 validation/training is wanted, prepare a separate collection:
select advanced expert prefixes BEFORE future outcomes, snapshot/replay-verify them,
then run precommitted random-action bursts with multiple independent environment seeds.
Split by parent episode, including sibling bursts. Keep natural and advanced-prefix
distributions explicit; do not silently install rare-event weighting. First count the
actual outcomes and report coverage before promising a training fix.

No new collection has been launched: that would change the corpus, and the current
bounded campaign ends with discussion. Existing indices and contracts are ready for
P2 implementation; advanced-prefix collection is a separate decision.
