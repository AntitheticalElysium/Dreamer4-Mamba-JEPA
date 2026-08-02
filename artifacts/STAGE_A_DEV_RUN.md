# Stage-A DEV run, 2026-08-02

**Not a Stage-A result.** A DEV smoke to find out whether the pipeline learns and
whether the failure modes the register worries about appear. The preregistered
FINAL protocol (S52) was not run: 16 seeds rather than a sealed set, an 800-step
cap rather than Craftax's native 10000, one training seed per arm. The FINAL
seeds are untouched.

## Setup

One tokenizer for all four arms (S20), 3000 steps; per arm 5000 dynamics, 2500
agent, 800 actor. Corpus: 96 archive episodes plus the support corpus -- 344
train / 44 dev episodes, 260,081 transitions, 274 terminals, 76 BC-eligible.
Phase 1A restored from checkpoint three separate times and reproduced the cache
digest `a62705fcfbace70c` every time.

## The BC control changed the conclusion

The first run compared the actor only against random, which S52 forbids. Adding
the control the protocol requires showed the opposite of what random suggested:

| design | actor | BC prior | random | actor − BC |
|---|---|---|---|---|
| shared critic trunk, batch 4, horizon fixed at 8 | 1.44 | **3.03** | 1.17 | **−1.59** |
| split critic, actor batch 16, horizon selected | 2.68 | 2.31 | 1.17 | +0.37 |

Imagination RL was destroying the cloned policy, and against a random baseline it
still looked like progress. Three structural faults were behind it, each verified
before being changed:

- **The critic shared the policy's trunk.** A value-only backward put gradient
  17654 into the body the policy reads, so the critic reshaped policy features
  outside PMPO and outside the prior KL. After the split: `None`.
- **Phase 3 inherited Phase 1A's batch of 4.** Phase 3 never runs the tokenizer
  and is nowhere near that memory ceiling; PMPO's sign-of-advantage estimate is
  over starting contexts. Now `actor_batch = 16`.
- **The horizon was never selected.** S54 requires DEV selection from candidates;
  the code silently used the default, and the multistep diagnostic it should have
  used could only measure one step because it ran on a short batch with context 15.

## Results, fixed design

| arm | actor | BC | random | actor − BC | 95% CI | horizon | separation | reward MAE |
|---|---|---|---|---|---|---|---|---|
| flow-attention | 2.68 | 2.31 | 1.17 | +0.37 | (−0.43, 1.03) | 32 | 0.0117 | 0.098 |
| flow-mamba | 2.58 | 2.95 | 1.17 | −0.37 | (−1.31, 0.98) | 4 | 0.0178 | 0.086 |
| direct-attention | 3.21 | 3.87 | 1.17 | −0.66 | (−1.61, 0.43) | 4 | 0.0626 | 0.049 |
| direct-mamba | **6.27** | 5.52 | 1.17 | +0.76 | (−1.12, 2.90) | 4 | 0.1451 | 0.043 |

**No arm beats its own BC prior.** Every interval straddles zero, so at 16 seeds
imagination training has not been shown to add anything over behaviour cloning.
That is the honest reading, and it is the claim S52 exists to adjudicate.

The arm *ordering* is unchanged from the uncontrolled run and both substitutions
still help, with Direct-Mamba highest. That ordering is a property of the whole
pipeline, not of imagination RL.

## Three things to distrust in this table

**Flow's reward model carries no information.** The zero-predictor MAE on DEV is
**0.0795**; flow-attention scores 0.098 and flow-mamba 0.086 -- both worse than
predicting zero. PMPO in Phase 3 is therefore optimising noise for the flow arms.
Direct is genuinely better at 0.043-0.049. Phase 3 should be gated on the reward
model beating that baseline before it runs at all.

**Direct's low one-step error is a warning, not a win.** 0.047 against flow's
0.25-0.37 at contraction ~0.95 is the conditional-mean collapse signature S35
predicts: under squared loss the collapsed solution minimises exactly that number.
Adjudicating it needs successor samples, which this run did not supply.

**The horizon rule degenerates.** "Largest candidate within 2x the one-step error,
else the smallest" gave 32 for flow-attention and 4 for the other three -- but for
flow-mamba, direct-attention and direct-mamba *no* candidate met the tolerance, so
4 is a fallback, not a selection. A tolerance relative to the one-step error is
tighter in absolute terms for an accurate arm, which is backwards. The rule needs
restating before it decides anything.

Continuation separation now rests on 25 terminal targets per arm rather than 3,
and orders the arms the same way the scores do. It is still a small sample.

## Incidents

Two CUDA OOMs inside Mamba's Triton autotuner, which benchmarks kernel configs
and needs contiguous headroom: once in `diagnostics.cost`'s backward timing, once
in Phase 2's backward. The first is now caught -- a measurement must never abort a
run -- and the second is handled by running one process per arm
(`artifacts/run_all_arms.sh`), since a long-lived process fragments the allocator
across arms. `flow-attention`'s Phase 3 was also lost once to a checkpoint that
only fired on the `checkpoint_every` modulus; phases now always save their final
step.

Phase 1A and 1B checkpoints predated two new `Config` fields
(`actor_batch`, `horizon_tolerance`). Both are read only by Phase 3 and the
selection rule, which was asserted field-by-field before migrating them rather
than retraining the tokenizer.
