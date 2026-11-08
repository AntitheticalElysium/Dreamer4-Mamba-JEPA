# Cheap chooser development: policy, acquisition, adaptation and inputs

2026-11-03. User authorized all four readings, independently of the policy result.
No P1/P2 world training, production recipe changes, or learned-gaze overnight launch.
The completed `20261102_direct_chooser` experiment remains unchanged.

## Questions and limits fixed before reading these results

1. Does the existing head choose useful remote evidence when C/S are combined
   differently? This is a cached, paired policy ablation, not a head-training test.
2. Does remote prediction/contingency develop during the unchanged head fit?
3. Can one head/Adam state follow changing world representations (1 -> 5 -> 10)?
4. Does richer PRE-ACTION information improve selection under matched head fitting?

All use P1-F1.5 world seed1, with two head seeds611601/611602. Results concern this
world/recipe, not independent world-seed replication. The world and encoder are frozen
at each reading. No closed-loop acquisition benefit can be inferred: the worlds were
trained under uniform gaze. All experiments run even if question1 is negative.

### P0: cached policy ablation (CPU)

Both existing final heads, saved2617-context and disjoint978-context cohorts; the
same native C/S arrays are also scored. No prediction or refit. All49 possible3x3
windows, epsilon=.2 and equal C/S weights remain fixed. Let N(x)=x/mean(x), with
all-zero input yielding zero; U is uniform over49 windows.

* uniform: U.
* existing: .8 softmax((N(C)+N(S))/2)+.2U.
* C-only / S-only softmax: .8 softmax(N(C or S))+.2U.
* probability mixture: .8[softmax(N(C))+softmax(N(S))]/2+.2U.
* direct C-only / S-only: .8 x/sum(x)+.2U; zero sum falls back to U.
* direct probability mixture: .8[C/sum(C)+S/sum(S)]/2+.2U.

The last three isolate removing the softmax; they are not a temperature sweep.
Softmax probability mixing avoids the score blend's geometric-mean/product-like
combination. It does NOT remove component normalization or the relative influence
of a weak C signal. Direct normalization also loses absolute confidence. Neither
is assumed superior. No policy is selected on these development results.

Primary: exact probability mass covering each remote counter, each faced location,
and their union, per class (including failed attempts), head seed and cohort.
Also entropy, HUD allocation, random-frame day/night allocation, and episode-cluster
bootstrap95% intervals for paired differences against existing and uniform policies.
Class strata are evaluation labels only. Argmax is secondary; no lucky gaze draws.
Instrument must reproduce existing probabilities and reject deliberately reversed
score rankings; flat scores produce uniform, probability mixtures preserve the
arithmetic mean of component probabilities, and every policy preserves epsilon U.

### E1: unchanged fixed-encoder learning trajectory

Reuse exact checkpoint10 train400/dev600 detached caches. Same107091 train rows,
same16403 validation cohort as the old protocol (actual counts recorded), no monitor
episodes; no event reweighting. Exact original101584-parameter Head/Data,4000 updates,
B256, original AdamW/warmup/cosine/clip/RNG. Snapshot at initialization,100,500,1000,
2000,3000,4000. Evaluation must not consume fit RNG or change model/optimizer state.
Compare final weights/curves to the archived fit; report numerical differences.

At each snapshot report factual counter-cell squared error versus copy, actual
target frame-to-frame energy, C at the counter itself (rather than its neighboring
window), C/S remote and contact rankings, and all fixed P0 policy
probabilities. Global MSE is secondary. A drop in global loss alone is not emergence.
No extra updates after a null. No semantic-state decoder is assumed by token error.

### E2: carry the head through encoder checkpoints1,5,10

Re-encode the SAME train400/dev600 episodes under each checkpoint's current encoder;
identical reset/action alignment; representation snapshots are approximate online
tracking, not joint training or a continuous reconstruction of omitted checkpoints.
4000 updates per stage, total12000, two seeds. Carry weights AND Adam moments/step;
one global12000-step cosine schedule (100-step initial warmup), no LR restart at the
world switch. Same sampled transition-index stream throughout. Evaluate immediately
before/after switching targets, then at100,1000,4000 stage updates.

Fresh controls at checkpoints5 and10 get8000/12000 updates respectively, starting
from the same initialization and global schedule. Thus each endpoint comparison
matches cumulative updates/exposures; it changes whether earlier updates used older
representations. Checkpoint1 fresh control is the warm arm itself. Do not compare a
12000-update carried head to a4000-update fresh head as proof of adaptation benefit.

Report policy/remote-error trajectories, per-checkpoint target step/variance, and
paired endpoint differences. Raw loss across changing coordinates is not a semantic
progress measure. E1's fixed representation separates ordinary head acquisition from
representation tracking. Actual chooser-driven world feedback remains untested.

### E3: information contract, checkpoint10 only

Same two seeds,4000 updates, same minibatch draws/Adam schedule/objective. Two matched
auxiliary-input heads extend the original head with a256->64 auxiliary projection;
all other tensors and initial weights match. Both extra branches are active:

* coarse sketch: fixed random projection of the previous full coarse grid to256
  dimensions (a deterministic function of ALREADY supplied inputs).
* rich history: native corrected belief b[t-1] from the full real prefix, BEFORE
  a[t-1] and target frame t. Includes old coarse and fine/reward history.

Each auxiliary channel is standardized with fitting-only mean/std (episodes0:360,
monitor excluded), floor1e-3;
same floor/counts recorded, never fitted on dev. Same projection dimensions/parameter
count and per-example batch order. This controls extra head capacity and a new global
input route. A positive result supports utility of this richer history under this
reader, not the necessity of fine memory specifically; a negative fit cannot prove
no information exists. The native belief also changes representation accessibility:
improvement cannot uniquely distinguish additional information from easier use of
information already present in the coarse inputs. This is NOT a proposal to restore sequential full-belief
collection in production. The coarse sketch is fixed and label-free, seed611730.

Collect native beliefs under original seeded UNIFORM prefix fixations and current
checkpoint10 parameters. Preserve S.encode precision; b[t-1] never contains the
queried action. Verify coarse cache agreement, source/hash lineage, zero initial
pre-action state, reset alignment, and future/target exclusion. Compare actual/NOOP
using IDENTICAL history and auxiliary inputs. No target frame r/d/p/v or crop.

## Common reporting, correctness and scheduling

Saved and fresh development cohorts both reported; final split untouched. Reuse
their exact episode/time/class/window metadata. Grounded locations are primary;
factual local error and token drift are diagnostics, not proof of causal competence.
Action counterfactual support remains limited (old coverage report retained).
All policy probabilities and per-context errors saved; bootstrap is episode-clustered
with fixed seeds,2000 resamples. No multiple-comparison success claim from selecting
the best policy/epoch/class. No optimum weights/temperature/history length sweep.

CPU P0 can coexist with the existing evaluation at nice10 and <=2 threads. The one-shot
GPU worker waits until saccade-p1f16 is inactive AND its STATUS is complete, then checks
for other GPU compute clients. A failed/changed predecessor or occupied GPU stops the
worker with an explicit status; it never kills or restarts another job. No recurring
automation. Separate output paths; existing source pins are untouched.

Before full GPU work: CPU correctness tests and a small GPU smoke, including actual
native belief capture and one optimizer continuation. Fixed total GPU-worker wall
cap45min; infrastructure failures/timeout preserve partial outputs and stop, not
scientific extension. Record actual times, source/checkpoint/cache hashes and phase
status. All three E experiments authorized regardless of P0 outcome. No world-training
job follows. Stop for discussion after these results.
