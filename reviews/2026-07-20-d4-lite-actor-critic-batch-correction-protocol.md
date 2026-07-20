# D4-lite actor-critic batch correction protocol

Date: 2026-07-20

## Prior result

The first complete Dreamer-4 actor/value implementation is mechanically
valid but its selected small-batch checkpoint is rejected operationally.

- batch 16, horizon 32, 1,000 updates;
- development: actor 245.75, paired BC 243.67;
- sealed seeds `970000:970030`: actor 208.60, paired BC 242.60;
- paired delta: -34.00, bootstrap 95% CI `[-71.8675, 3.7667]`;
- value loss changed from 5.49 in the first 25 updates to 2.13 in the last 25;
- actor and value changed, both PMPO sign sets occurred, and exact frozen
  world/prior tensor hashes matched before and after training.

This is preserved as a failed outcome. Development selection did not
generalize.

The earlier completion rule also required the actor to exceed the historical
BC mean of 288.70 from a different seed set. That is not a paired parity
test: the same frozen BC scored 242.60 on the first sealed set. The historical
number remains descriptive, but the corrected gate compares the actor to its
frozen BC on identical episodes. This correction does not alter the first
verdict because that actor also failed the paired comparison.

## Single correction

Increase the number of distinct replay contexts per optimizer update from 16
to 64. No loss, coefficient, model, data source, rollout horizon, shortcut
schedule, optimizer, learning rate, seed, or evaluation rule changes.

This is source alignment, not an outcome-selected coefficient:

- `edwhu/dreamer4-jax` commit `8144b940`, `scripts/train_policy.py`,
  `RLConfig.B = 64`;
- Dreamer 4 explicitly motivates one rollout per context to prioritize data
  diversity;
- the failed batch-16 run used only 512 imagined state-action rows per
  optimizer step, so action-independent positive-return regions produced
  visible finite-batch PMPO drift;
- measured peak VRAM was only 63 MB, invalidating the original assumption
  that 64 contexts would exceed the local 6 GB GPU.

The first screen remains 100 updates. If development paired mean parity
fails, the fixed extensions are 250, 500, and 1,000 updates, each restarted
from the exact BC/value initialization and RNG seed. The earliest passing
development checkpoint is selected.

## Constants

- actor initialized exactly from
  `c57f9bbf688e5d54cb6f50df7b2ea87110de58860a73ae901931970379ba80bd`;
- frozen world
  `a63bb1fe31b69f8b24e68534401fd18be50b36bb06fb3fedbac7f9231c32551d`;
- context 8, horizon 32, four shortcut steps;
- Adam `1e-4`, gradient clip 1;
- `gamma=0.997`, `lambda=0.95`, PMPO `alpha=0.5`, reverse-prior-KL
  `beta=0.3`;
- development seeds `960000:960012`;
- new sealed seeds `980000:980030`, unopened until selection;
- direct greedy execution, no planner and no evaluation-time learning.

`gamma` remains per local decision transition. The CartPole wrapper's action
repeat is already part of that induced MDP and Dreamer 3 likewise applies its
configured discount to agent decisions after action repeat; exponentiating
gamma post hoc would be a new objective, not a correction.

## Completion gate

On the new sealed 30 seeds:

- actor mean is at least the paired frozen-BC mean;
- paired actor-minus-BC bootstrap 95% CI lower bound is at least -25;
- actor mean exceeds paired random and that paired CI excludes zero;
- actor and value tensors changed from initialization;
- both PMPO sign sets occurred;
- frozen world and prior tensor hashes are exact;
- an exact substantive evaluation repeat matches.

The old 288.70 BC mean is reported as historical context only.

## Precision amendment after the fixed 30-seed evaluation

The selected batch-64, 250-update actor passed paired mean parity on the
registered 30 seeds:

- actor 236.0667;
- paired BC 231.2333;
- delta +4.8333;
- bootstrap 95% CI `[-35.3683, 45.4675]`.

Thus the effect direction passed but the registered noninferiority interval
remained unresolved. No model, checkpoint, hyperparameter, or margin changes.
Before evaluating another episode, the parity sample is fixed at 100 seeds,
`980000:980100`. The complete 100-seed run must be used regardless of interim
values. It reruns the first 30 seeds as an exact substantive reproduction and
adds the previously unopened `980030:980100` rows to narrow sampling error.
Completion is decided only from the pooled 100 paired episodes.

Outcome: **PASS**. The fixed actor scored 281.33 versus 249.32 paired BC,
delta +32.01 with bootstrap 95% CI `[6.66,57.94]`. The complete outcome,
negative batch-16 result, held-out value diagnostic, exact hashes, and claim
boundary are in
`2026-07-20-d4-lite-imagination-actor-critic-outcome.md`.
