# D4-lite imagination actor-critic protocol

Date: 2026-07-20

## Decision

The shooting planner is not the next optimization target. The next baseline is
the missing Dreamer-4 policy/value phase. Completion requires an
imagination-trained actor to execute directly in the real environment at
parity with the frozen BC positive control.

Mamba and CDP/JEPA remain disabled. The tokenizer, Transformer dynamics,
reward head, continuation head, and BC prior remain frozen.

## Sources read before implementation

1. Dreamer 4, `third_party/papers/2509.24527v1.pdf`, SHA-256
   `8655cce4bf12ce6210f6694f83c1a723c7acd7579214ca3ebc57c4394d0b1aeb`,
   especially Section 3.3 and equations 9-11.
2. `edwhu/dreamer4-jax` commit
   `8144b940d801971f12ec5633553b95001e555949`, especially
   `scripts/train_policy.py`, `dreamer/imagination.py`, and the policy,
   reward, and value heads in `dreamer/models.py`.
3. DreamerV3 paper, arXiv `2301.04104v2`, critic/actor equations 4-7 and
   the actor-critic hyperparameter table.
4. `danijar/dreamerv3` commit
   `e3f02248693a79dc8b0ebd62c93683888ddaccfe`, especially
   `dreamerv3/agent.py:imag_loss`, `lambda_return`, and
   `dreamerv3/configs.yaml`.
5. `fmi-basel/Dreamer-CDP` commit
   `a851fa3e3d70b624b094ee1810ad4bb602346092`, confirming that its
   actor/value path retains the DreamerV3 imagination algorithm.
6. `nicklashansen/mmbench2` commit
   `3dda6ea5bc60382ad9e1dcd1c6c3af67d69326a9`, for the actual PyTorch
   agent-token, reward-distribution, and shortcut-rollout interfaces used by
   the frozen local world.

The Nicklas Hansen Dreamer-4 checkout contains only tokenizer and dynamics
training. The Edward Hu checkout contains the full actor/value phase but is an
unofficial toy-environment reproduction. Neither is treated as canonical.

Two inspected reproduction choices are deliberately not copied:

- the Edward Hu runner initializes a new random actor rather than copying the
  BC actor described by the Dreamer-4 paper;
- its rollout averages agent tokens over the entire context where its own
  interface says to use the current state.

The local actor is initialized exactly from the verified BC checkpoint and
acts from the final causal state slot.

## Indexed algorithm

The local transition convention is:

```text
(state_t, action_t) -> (state_t+1, reward_t+1, continue_t+1)
```

For each replay context, one imagined trajectory is generated:

```text
s_0, a_0, r_1, c_1, s_1, ..., a_H-1, r_H, c_H, s_H
```

The frozen world generates each next latent with four shortcut steps. The
actor samples `a_t` from the current agent tokens. Reward and continuation are
read from the generated next-state agent tokens.

The categorical value head predicts `v_t` from `s_t`. With
`gamma=0.997`, `lambda=0.95`, and bootstrap `R_H = v_H`:

```text
R_t = r_t+1 + gamma * c_t+1 *
      ((1 - lambda) * v_t+1 + lambda * R_t+1)
A_t = stop_gradient(R_t - v_t)
```

The value loss is two-hot categorical cross-entropy on `symlog(R_t)`.

The actor uses Dreamer-4 PMPO exactly:

```text
(1-alpha) * mean(log pi(a|s) over A<0)
- alpha    * mean(log pi(a|s) over A>=0)
+ beta     * mean(KL(pi || frozen_BC))
```

with `alpha=0.5` and `beta=0.3`. There is no BC action loss during
imagination training and no gradient through sampled actions, world states,
rewards, continuations, or return targets.

## Registered reduced configuration

- context: 8 observations;
- imagination horizon: 32 decisions;
- batch: 16 distinct replay contexts, one rollout each;
- shortcut steps: 4;
- actor/value learning rate: `1e-4`;
- optimizer: Adam, no weight decay;
- global gradient norm clip: 1.0;
- value bins: 255 uniform symlog centers in `[-10, 10]`;
- actor initialization: exact frozen BC checkpoint;
- value output weight and bias: zero;
- training replay: the mixed random/noisy-balance world replay;
- development seeds: `960000:960012`;
- sealed parity seeds: `970000:970030`.

The first complete screen is 100 updates, or 51,200 imagined state-action
decisions (`100 * 16 * 32`). If it misses paired BC parity on development
seeds, the only registered budget extensions are 250, 500, and 1,000 updates,
each restarted from the same BC/value initialization and RNG seed. The
earliest development checkpoint satisfying paired mean parity is selected.
This fixed ladder separates a small-task stopping decision from arbitrary
post-outcome tuning. The sealed set is evaluated once after selection.

## Explicit divergences from large Dreamer 4

These differences are part of the reduced baseline, not silent claims of
full-scale reproduction:

- CartPole replaces multi-task Minecraft; task embeddings are omitted because
  the task identity is constant.
- The frozen 4-layer, width-64 local Transformer and 64x64 three-view pixel
  adapter replace the 1.6B-parameter transformer and 360p Minecraft tokenizer.
- Context is a sliding eight-state window rather than 192 frames.
- The local world was trained and validated with clean encoded context, so
  context latents are not corrupted to Dreamer 4's reported
  `tau_ctx=0.1`. Adding corruption at RL time would introduce an unseen world
  input distribution.
- Binary categorical actions and only policy distance zero replace Minecraft
  mouse/keyboard actions and eight policy MTP output layers.
- Reward and continuation are existing frozen local heads. Continuation is
  used in equation 10 explicitly because CartPole episodes terminate.
- Actor and value pool two agent tokens with the source-shaped local attention
  pooler; the paper's single task embedding interface is unavailable.
- Adam is restarted for the actor/value phase. Exact large-scale optimizer
  state, value-bin support, and clipping are not published.
- Evaluation is greedy to match the frozen BC positive-control protocol;
  imagination actions remain categorical samples as required by PMPO.
- Mamba and CDP/JEPA are absent. This run establishes the Transformer control
  baseline they must later modify.

## Mandatory pre-training tests

1. hand-computed TD-lambda and terminal indexing;
2. two-hot targets sum to one and interpolate adjacent bins;
3. PMPO increases sampled positive actions and decreases sampled negative
   actions;
4. reverse prior KL is zero at exact BC initialization;
5. actor and frozen prior are initially tensor-identical;
6. value output starts at exactly zero expectation;
7. only actor/value parameters receive gradients;
8. tokenizer, world, reward, continuation, and prior tensor hashes do not
   change across an update;
9. checkpoint load rejects world/BC/source/hash mismatches;
10. direct actor evaluation cannot call the shooting planner.

All ten contracts are executable regressions in
`d4_mamba_jepa/tests/test_imagination_actor_critic.py`; the pre-training suite
passes 12/12 tests.

## Completion gate

On the sealed 30 seeds, evaluated greedily from pixels with no planning and no
learning:

- imagination actor mean return is at least the paired frozen-BC mean;
- imagination actor mean return is at least the historical BC mean of 288.70;
- the episode-paired bootstrap 95% interval for actor minus BC has lower bound
  at least -25 return;
- the actor differs numerically from its BC initialization;
- the value head differs from zero initialization and has finite held-out
  imagined targets/loss;
- at least one positive and one negative PMPO set occurred during training;
- the frozen world and BC-prior tensor hashes are unchanged;
- an exact substantive evaluation repeat matches.

Random and privileged policies remain descriptive references. They are not
substitutes for the paired BC parity decision.

Outcome: the initial batch-16 ladder was rejected on its sealed set. The
source-aligned batch correction, fixed precision extension, passing result,
reproducibility evidence, and claim boundary are recorded in
`2026-07-20-d4-lite-imagination-actor-critic-outcome.md`.
