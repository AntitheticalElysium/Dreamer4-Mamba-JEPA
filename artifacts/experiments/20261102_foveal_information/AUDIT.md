# Contract audit before interpreting the new readers

Frozen checkpoint and world source are unchanged. `conditional.py` trains only disposable
labelled readout instruments. It does not train the world, change fixations used to replay
its history, alter the corpus, or tune a selector. The three-update smoke is not a result.

Checked before the full fit:

* S1's replay uses one deterministic uniform fixation stream per episode; whole-prefix
  Mamba replay. `bp[t]` is before the current percept; `b[t]` follows it; `bp[t+1]` follows
  its action. Current query excludes the endpoint T, so bp[t+1] is available without
  reading percept t+1. Baseline receives the same action and arrived outcomes as all arms.
* Cached train data: 128,564 frames, 111,804 in 360 fit episodes and 16,760 in 40 validation
  episodes. Kind counts over all map queries: 7,946,281 empty; 58,210 zombie; 61,361 cow;
  21,026 skeleton; 12,654 arrow. No simulator label enters world or chooser inputs.
* Every previous-frame index was checked exactly against its episode and time. Missing
  history is zero plus validity=0; no previous episode or future frame leaks in.
* Current full periphery, explicit local coarse token and full eight-frame physical
  history are supplied unchanged to every reader. The same padded supplement width,
  initial weights and minibatch generator seeds are used within each family/seed.
* The fine supplement is zero outside the selected crop. All arms receive the crop's
  identity and the availability bit. Inside/outside sampling is balanced per kind.
* Raw softmaxes are fitted on balanced classes, so are diagnostic scores, not calibrated
  natural event probabilities. Macro CE and class-conditional errors are appropriate
  comparisons under that declared query distribution; do not interpret event rates.
* Native supervised physical targets are detached percepts; the current fine crop has
  no direct path into a FUTURE physical query under the factorized mask. Its allowed
  path is through Mamba and memory projection. Train.py 127–131 preserves gradients
  from G1 through memory/belief. TSP sees the predicted belief and current coarse tokens;
  reconstruction sees current fine tokens. Grounding does not itself require the memory
  to transmit information when the coarse stream can satisfy its target.
* In Belief.forward (model.py 213), the percept and action steps have different residual
  inputs. Output RMS attenuation between them cannot be interpreted as information loss.
  The projected four memory tokens are a linear function of the same 256-D emitted state,
  not an independent larger state. Neither fact proves that fine information is absent.
* Paired source selection uses only current-frame type and conditional readers, not
  future outcomes. Same-frame actual crops, same earlier real history and row-matched
  entity-free controls. Native fixed-noise losses are reused; no new stochastic future is
  point-matched to the recording.

The initial smoke had an implementation error (`symlog` vs native `SYMLOG`). It stopped
before any reader fit, was corrected, and its log was preserved. The completed second
smoke exercises all 24 readouts, both splits and persistence; its three-update values
have no scientific interpretation.

Final interpretation checks:

* Raw-fine night recall improves in four fits; CE is not uniformly improved, arrow
  regressions retained, and downstream reader-family disagreement is explicit.
* The new strict paired cohort has eight cases, not 58. Old local masks do not match
  its source entities; new analysis omits local-loss claims rather than borrowing them.
* Query carry readout actually trained (weight RMS changes .0999/.0833), but train
  CE<.00025 and weak validation show overfitting. Its null is not proof of erasure.
* Native GPU contract suite passes 18/18. World model/data/train/YAML hashes unchanged.
* A direct forecast head must not see its arrived-frame target p[t]; a score student
  may read it before the fine crop. Previous-action timing and NOOP belief forks remain
  explicit in the chooser comparison; no production implementation exists yet.
