# Deviation ledger

Every departure from a pinned primary implementation must be entered here
before training. A change is not allowed to hide behind a generic "D4-Mamba"
label.

Status values:

- `PLANNED`: documented, not implemented.
- `IMPLEMENTED`: code and unit tests exist.
- `SCREENED`: smallest discriminating runtime check passed.
- `REJECTED`: implementation retained only as evidence or removed.

| ID | Status | Local change | Source contract retained | Why required | Isolation / first failure suspect |
|---|---|---|---|---|---|
| D000 | SCREENED | Load MMBench2 `model.py` by exact path and reject digest drift | File contents and class definitions remain untouched | Its executable uses flat imports and is not packaged as a dependency | Source-loader test passes; a digest failure means source drift, not a model failure |
| D001 | SCREENED | Reduce image resolution and model width/depth to a named `D4LiteConfig` | Same tokenizer, block-causal layers, shortcut tokens, and flow head | Upstream training scale requires hardware far beyond 6 GB | Four-arm CUDA smoke, synthetic overfit, and real T-BASE preflight pass; remaining failure may still be under-capacity |
| D002 | SCREENED | Replace the continuous 16-D action MLP with a 17-way discrete embedding | Exactly one action token occupies the same token-layout slot | Crafter exposes 17 categorical actions | Led-to/indexing tests pass; after 5k T-BASE updates, paired action shuffling raises uniform-dev flow loss by 21.6% |
| D003 | SCREENED | Add a continuation MTP head beside the upstream reward MTP head | Head reads the same post-transition agent tokens | Crafter planning must distinguish terminal from continuing futures | Alignment and gradient tests pass, but efficacy fails: the 5k baseline predicts P(continue)=.998 on terminal-aligned generated states |
| D004 | SCREENED | Replace only `TimeSelfAttention` modules in dynamics with official Mamba-2 modules | Spatial attention, MLP, residual ordering, token layout, tokenizer, and heads stay unchanged | Tests the original long-context temporal-compute hypothesis | Source/mechanical checks pass; matched real-Crafter M1 passes all frozen feasibility gates and reproduces bit-exactly under the pinned official deterministic path; there is no short-context speed win |
| D005 | SCREENED | Branch a clean Mamba prefix state across shortcut denoising candidates and commit only the generated latent | Each denoising candidate sees one identical clean prefix | Mamba state is mutable whereas attention KV context is functionally shared | Cached/uncached equivalence and candidate-isolation tests pass; first suspect for any future rollout-only Mamba failure |
| D006 | SCREENED | Add a CDP-shaped cosine predictor from clean causal agent state plus next action to a stop-gradient future latent; update the encoder slowly and retain a frozen-decoder reconstruction anchor | Upstream flow loss remains present, receives detached tokenizer latents, and stays independently reportable | Tests predictive representation learning without letting future-target gradients or a moving decoder obscure attribution | Target-detach, encoder-gradient, decoder-freeze, optimizer-group, and CUDA tests pass; scientific efficacy remains untested |
| D007 | SCREENED | Reuse episode-bounded replay and canonical Crafter wrapping from `m3_hjwm_compact` through adapters | Observation/action/reward/continuation timing is unchanged | These utilities already passed indexing and reproducibility regressions | Adapter/source/replay hashes pass; real-data training and two bit-identical executed trajectories pass |
| D008 | SCREENED | Use categorical random shooting before implementing categorical CEM or PMPO | Upstream imagined reward scoring and first-action receding horizon remain | Produces executed behavior early with fewer new optimizer assumptions | Three-seed executed preflight and exact substantive repeat pass; it remains an evaluation instrument, not a learned Dreamer policy |
| D009 | SCREENED | Add strict atomic checkpoints around the composed model, full config, source digests, optimizer, and every RNG | Model state still loads with `strict=True`; no source checkpoint is rewritten | The upstream executable checkpoint format cannot reconstruct local adapters and prior work exposed partial RNG resume | Strict roundtrip, digest/config/source rejection, Torch/NumPy resume, tokenizer roundtrip, and failed-save preservation tests pass |
| D010 | SCREENED | Port only the tokenizer reconstruction, shortcut-flow, reward-alignment, and rollout equations needed around the unchanged upstream model classes | Operators, led-to convention, RMS normalization, K-grid, and Euler update follow the pinned scripts | MMBench2 is an executable-style project whose flat trainers cannot be safely imported as a library | Line-level provenance is in docstrings; synthetic overfit, paired action shuffle, cached rollout, and real preflight are the first parity checks |
| D011 | SCREENED | Add official Gymnasium `CartPole-v1` as a small control baseline with action repeat 2 and a three-channel pixel adapter: full foreground, pixel-localized zoom, and frame difference | Official dynamics, reward, termination, time limit, and RGB renderer are untouched; every derived view uses rendered pixels only | At 64×64, the raw mostly-white render let the reconstruction tokenizer ignore pole angle and action effects; the adapter makes task pixels measurable without exposing simulator state to the model | Source and installed-file hashes pass; held-out tokenizer probes recover cart position and pole angle; this task-specific observation adapter is the first suspect if transfer to raw imagery fails |
| D012 | SCREENED | Route reward and continuation heads through the agent tokens from the same noised shortcut-flow forward pass | This is the exact MMBench2 `dynamics_pretrain_loss` task-head route | The initial local port incorrectly made a second clean-latent task-head pass, creating a train/deployment mismatch | A regression forbids the second clean pass; the corrected world obtains an 18.2% paired action-shuffle penalty and positive executed planning |
| D013 | SCREENED | For the two-action control check, optionally enumerate all 256 horizon-8 plans and use common random numbers across candidates | Denoising schedule, imagined reward/continuation scoring, and first-action receding horizon are unchanged | Removes finite random-plan coverage and candidate-noise confounds from the smallest deterministic benchmark | On 30 fresh seeds, the frozen imagined planner scores 44.87 versus 19.00 random, paired delta +25.87 with 95% bootstrap CI [15.80, 36.73]; it misses the separately declared absolute mean-50 gate |
| D014 | SCREENED | Add a categorical version of the MMBench2 attention-pooled, gradient-isolated BC policy head and train it only on demonstration actions | Agent-token input, attention pooling, MLP shape, small output initialization, and head-only BC gradient boundary follow upstream `PolicyHeadMTP` | A source-shaped policy gives a positive executed baseline before Mamba or JEPA experiments and separates world competence from planner quality | Held-out action accuracy is 87.04%; on 30 fresh seeds the frozen pixel policy scores 288.70 versus 17.60 random, wins 30/30, and reproduces exactly |
| D015 | SCREENED | Serialize checkpoints through a binary file handle before atomic replacement | Payload, strict loading, and atomic replace semantics are unchanged | PyTorch otherwise embeds a random temporary basename in its ZIP archive | Identical model state now yields identical checkpoint bytes; policy weights reproduce bit-for-bit |
| D016 | SCREENED | Add the Dreamer-4 imagination phase as an actor/value-only update: initialize the actor from the exact BC head, freeze an exact BC prior, freeze the tokenizer/world/reward/continuation modules, train a zero-output symexp-twohot value head with TD-lambda targets, and train the actor with balanced PMPO plus reverse `KL(actor || BC)` | Dreamer 4 equations 10-11, `gamma=0.997`, `alpha=0.5`, `beta=0.3`, one rollout per replay context, direct policy execution, and the paper's frozen-transformer contract are retained | This is the missing phase that turns the working world-plus-BC stack into an imagination-trained agent instead of a shooting controller | Twelve pre-training regressions establish indexing, loss direction, exact initialization, gradient isolation, frozen tensors, checkpoint pairing/source rejection, and absence of planner routing; the selected actor clears executed BC parity on 100 fresh seeds |
| D017 | SCREENED | Use one CartPole task, the existing distance-zero categorical BC output, 32 imagined decisions, a sliding clean context of 8, and the existing pixel adapter instead of Minecraft task tokens, 192-frame context, `tau_ctx=0.1` corruption, and eight BC MTP distances | RL still acts from the final causal agent tokens, samples one action per state, rolls the source-pinned shortcut world, and applies the Dreamer-4 actor/value losses only to imagined data | Task embeddings are vacuous for one task; only distance zero is consumed by RL; adding unseen context corruption would mismatch the trained local world; the reduced benchmark cannot reproduce Minecraft scale | These scale/interface divergences run end to end, but remain first suspects for transfer; the selected local actor scores 281.33 versus 249.32 paired BC |
| D018 | SCREENED | Use a restarted Adam at `1e-4`, global norm clipping at 1.0, and symlog centers `[-10,10]` for the value head | Adam `1e-4` and a categorical symlog value head follow the inspected Dreamer-4 JAX reproduction; zero output initialization follows DreamerV3 and the Dreamer-4 paper | Dreamer 4 does not publish all optimizer/value-bin details, while the exact large-scale optimizer state is unavailable | Training remains finite and reproduces tensor/history exact; held-out value CE is 3.136, but value MAE 24.0 and imagined continuation near one remain explicit limitations |
| D019 | REJECTED | Select actor checkpoints only on a reserved development seed set and open a separate 30-seed parity set once; the first fixed budget ladder used batch 16 and 100, 250, 500, then 1,000 updates | Direct greedy actor execution without shooting or online learning matches the frozen policy deployment contract | The small benchmark needs a falsifiable completion rule and protection against repeated final-set tuning | The selected 1,000-step actor passed development mean parity but failed sealed parity: 208.60 versus 242.60 BC, paired -34.00 with CI [-71.87, 3.77]. The additional 288.70 cross-seed threshold was also invalid as a parity rule because the paired BC itself varied, but removing it cannot rescue this rejection |
| D020 | SCREENED | Increase the actor/value batch from 16 to the inspected Dreamer-4 reproduction's 64 distinct contexts while keeping one rollout per context and every algorithmic setting fixed; reserve new final seeds and make historical BC performance descriptive only | Dreamer 4's one-rollout-per-context diversity rule, PMPO equation, BC prior, and frozen-transformer boundary remain exact; the inspected actor runner uses `B=64` | The original 6 GB constraint assumption was false after measured peak use of only 63 MB, and small-batch PMPO noise moved the actor despite nearly action-constant imagined reward/continuation | Batch-64/250 updates reaches 281.33 versus 249.32 paired BC on fixed `980000:980100`, delta +32.01 [6.66,57.94]; retraining and full executed evaluation reproduce exactly |

`D010` is an administrative ledger correction made after the first preflight:
the port boundary was already disclosed in `README.md`, `SOURCE_MANIFEST.md`,
and source docstrings before execution, but it did not have its own numbered
row. No code or result was changed when the row was added.

`D012` records a real correction to that port. The flow equation was faithful,
but the first local composition trained task heads on a separate clean-latent
forward. Upstream MMBench2 trains them on the agent tokens returned by its
noised flow forward. All working CartPole checkpoints use the corrected route;
the older Crafter checkpoints remain historical evidence under their recorded
implementation hashes.

## Transition convention

All local adapters use:

```text
(observation_t, action_t) -> (observation_{t+1}, reward_{t+1}, continue_{t+1})
```

Within the block-causal sequence, the action token stored at state position
`t+1` is `action_t`, the action that led to that state. Position zero receives
the dedicated start action. Reward and continuation head zero predict the
transition outcome that led to their current state position.

## Forbidden silent changes

- No Mamba fallback to GRU or attention.
- No Mamba inside the tokenizer during the initial factorial.
- No removal of reconstruction in the initial CDP arm.
- No reward-bin, event-sampling, calibration, or planner-budget tuning that
  differs between factorial arms.
- No loading Transformer dynamics weights into the Mamba arm with
  `strict=False`.
- No claim of a Dreamer-4 PMPO agent until policy/value imagination is
  implemented and tested separately.

## Current first-suspect order

1. Continuation class imbalance and calibration: implementation alignment
   passes, but the baseline learns the dominant "continue" label.
2. Reward magnitude and generated-state deployment: event ranking improves,
   while event magnitude remains strongly underestimated.
3. Baseline scale and training budget: the 5k screen is before the upstream
   10k shortcut-bootstrap start and far below upstream training scale.
4. The categorical action adapter: it is measurably used, but correct-action
   benefit is small and heterogeneous.
5. CDP gradient routing and representation usefulness. It is mechanically
   screened but disabled in every real-Crafter result recorded so far.
6. Mamba cache/recurrent semantics only if future long-context or planner
   behavior diverges. Matched M1 training passes; Mamba is not an explanation
   for the failures shared with T-BASE.
