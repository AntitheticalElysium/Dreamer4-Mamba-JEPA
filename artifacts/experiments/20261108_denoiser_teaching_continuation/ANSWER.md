# Continuing the three foundational questions

Codex, 2026-11-08. Read both user attachments, the actual design record, the relevant original conversation entries,
the completed denoiser-teaching code/results, and primary papers. Finished the interrupted M4 visibility correction
separately; no full training, canonical changes or interference with other agents' jobs.

## The common issue

The denoiser DOES train Mamba. There is no missing-gradient switch that explains everything. But an architecture
trained to predict observations jointly from memory, cached observations and corrupted targets does not necessarily
make a particular exported memory vector a sufficient actor input at a finite training budget.

Keep distinct:
- internal Mamba recurrent carry C_t;
- post-percept export b_t, the planned actor/critic input;
- post-action export bp_(t+1), used by the denoiser and current heads;
- recent denoiser cache, which includes earlier visual AND memory tokens.

The generator and proposed actor do not see the same information. Nor are b_t and bp_(t+1) the same readout.
The next action update reads private carry C_t, not merely b_t. A perfect predictive head attached to bp can therefore
coexist with an impoverished actor-facing b. Any teacher meant to certify b's decision sufficiency must read b and
the candidate action, without gaining private information by first processing that action through the full carry.

An exact counterexample in ADVERSARIAL_CHECKS: private state x=0/1, b=constant, reward1 iff a=x. A post-action export
that reads x from carry lets a predictive head be perfect, while the best actor restricted to b achieves only0.5
versus1.0 with x. This is an interface counterexample, not a measured assertion about F4's carry.

## 1. What happened to next-embedding prediction?

The record does NOT show a separately implemented Mamba auxiliary head being removed after a cost/complexity
benchmark. Next-perception prediction remained present throughout, but its implementation changed:

- The older LEARNING_SKETCH v2 predicts next percepts through a stochastic generator and energy score.
- The observation-stage discussion explicitly replaced pointwise latent MSE with distributional scoring to avoid
  using averaged stochastic outcomes as imagined samples.
- Generator v3, recorded in DECISIONS 10-27, folds prediction into next-percept diffusion conditioned on belief.
  The reason given is generative uncertainty and simpler module structure, not a measured cost of a small head.
- The minimal-build discussion adds recent percepts because a single global memory summary was judged too lossy
  for spatial rendering. That gives generation another source of information and weakens the requirement that an
  individual Mamba export be sufficient by itself.

Verified original conversation locations (e3e54d67 session, zero-based JSONL rows):11747 observation stage/energy
score,12769 categorical-vs-continuous fork and simplification,12872 v3,12960 M0 recent-window rationale.
Design anchors: GENERATOR v3, DECISIONS next-perception entry, STEP_SPEC7, older LEARNING_SKETCH2.

There are three different possible 'next embeddings':

| target | current status / consequence |
|---|---|
| Next spatial percept P_(t+1) | Already predicted distributionally by G1. Adding another such term is additional supervision, not restoring a removed generative capability. |
| Next global frame embedding e_(t+1)=percept_embed(P_(t+1)) | A dedicated b-only auxiliary is not in M0. It shortens the route to a prediction target, but that target is already an imperfect summary and includes fixation-dependent content. |
| Next recurrent belief/export | A distinct self-prediction objective. It can match an uninformative future state unless grounded by other requirements; it need not recover a missing timer automatically. |

A small predictor is not intrinsically a large compute burden; no particular implementation was timed here.
What matters first is its target, conditioning and routing.

**Would it fix the problem?** It could force the chosen source representation to support prediction without the
cache and without a noisy view of the answer. That is a real new constraint, not merely multiplying G1's weight.
But it does not guarantee retaining the right facts. In particular, today's global embedding poorly exposes several
visible digit transitions. Regressing onto that same target does not automatically make the missing changes more
important. It also cannot by itself certify a correct conditional distribution or coherent imagined trajectories.

A deterministic auxiliary can legitimately predict an expected feature even when futures are stochastic. The
mistake would be treating that mean as an actual sampled future. Do not force b itself equal to a history-free
frame embedding: that would conflict with retaining hidden history. A prediction decoder and stop-gradient target
are different from direct equality of state and observation.

Primary reference: Ni et al. distinguish expected-next-latent prediction from full distribution prediction and
analyse collapse/stop-gradient issues: https://arxiv.org/html/2401.08898v3 .

## 2. Is change-map supervision redundant? Shouldn't G1 teach the missing memory?

**There is overlap, but the training roles differ.** ChangeMap reads both REAL frame encodings plus bp and has a
fixed raw-observation-derived target. It can ground moving/changed content in the encoder even if G1 could lower
its loss by omitting that content from its learned targets. It does not have to anticipate a change to succeed.
G1 predicts learned percepts and reaches the encoder only through memory, with frame inputs/targets detached.

If perception were already completely informative and the full conditional generator perfect, much of the
change-map information would follow from that solution. Those conditions are not given. Therefore informational
overlap does not establish optimisation redundancy, and current evidence does not justify deleting this grounding
term. Conversely, its large gradient does not prove a useful memory-teaching role. Grounding and memory training
can be separated in principle, but no route-removal repair was tested here.

**In principle the user is right:** if an old fact changes the conditional future, cannot be obtained from other
inputs, and the training objective/capacity/optimisation resolve it, G1 should reward retaining it.

But the cache is not the only alternative input: during training the denoiser also sees the CORRUPTED NEXT TARGET.
At high signal levels that can reveal the next digit even when no input predicts timer phase. Pure-noise cases
require forecasting, but fitting a small subset of output coordinates poorly can still be cheap in the aggregate.
This is a plausible finite-training explanation, not proof that ramp or cache masking is the right repair.

### Completed M1: both branches really teach memory

150 original training microbatches,4 streams x64, same frozen F4. Mean norm dL/dbp, units1e-6, ordinary frames:

| source | flow | bootstrap |
|---|---:|---:|
| G1, all percept outputs |4.609|5.488|
| reward reader |45.212|46.250|
| change map |18.837|17.800|

Including bootstrap does not reveal a huge missing G1 contribution. It confirms G1 is active in both branches.
Within the flow branch, the relevant output's memory gradient rises4.6x at food changes,9.5x at recovery and21.4x
at zombie hits compared with ordinary frames. Bootstrap ratios1.6x/5.7x/8.0x. These are sensitivities at the trained
checkpoint, NOT shares of learning or measurements of the historical Adam updates. M1 gradients can also include
future denoiser losses that attend this memory token; they are not isolated semantic instructions.

### Completed M2/M3: no demonstrated hard credit cutoff

State-gradient consistency .396 food,.495 recovery,.623 hits does not suggest an absent common direction even by
the original descriptive metric. However, that metric cannot certify learning agreement: a simple regression
example has opposite state gradients but identical parameter gradients. Our exact CPU check gives state
consistency0 and parameter consistency1.

Relative to lag1, the gradient at lag10 is4.74% food,1.16% recovery,2.50% hits,4.43% wake. The plan's severe-cutoff
criterion (<1% by lag10 plus no reset emphasis) is NOT met. Gradients are nonzero at the previous resets. At lag30
they are1.56% for food and0.44% for recovery. Significant attenuation remains; adequacy is not established.

No reset-specific peak is required by a working recurrent algorithm. Nor does a small derivative at the final
checkpoint prove that adding supervision could never change the route: training changes the Jacobian itself.
These measurements show reach and sensitivity, not what rule is learned.

### Independently corrected M4

Same40,000 DEV transitions, seed2120, same five episode folds and logistic protocol. Reused F4's source-bound
encoded trajectories, recomputed only the embedding. Original signed-health control reproduced **exactly** at
AP.5371543984695801 before interpreting the repair. Corrected change detection uses absolute differences:

| any visible digit change | raw HUD-token AP | global input-embedding AP |
|---|---:|---:|
| health |1.000|.99865 [CI .99719,.99971]|
| food |1.000|.26796 [.24503,.29045]|
| drink |1.000|.19100 [.17427,.20547]|
| energy |1.000|.06281 [.05715,.06971]|

Health's reset-like changes remain readily accessible at the input. The other changes become much harder for the
same reader family. This is not proof of irreversible information destruction. Direct coarse-input blocks for all
four digits have full rank16; the learned projection is not simply zeroing them. Whole-scene mixing and relative
scale matter: the health column block's RMS is.0703 versus.0120 for energy. These are trained-snapshot properties,
not an isolation of their training cause or proof that a larger projection is needed.

An important distinction: visible digit changes are NOT every timer reset. Recovery can reset at full health
without any visible health change. In the original data report21.4% of recovery rises have a previous invisible
reset; do not turn health-change AP1 into perfect observability of every recovery reset event.

Original valid zero-value probes (not rerun or advertised as new): actor-facing b detects food0 at AP.906, drink0
at.99998, energy0 at.819. Poor exact-digit recall does not mean the actor is wholly blind to critical boundaries.

**What can we say G1 teaches?** It sends output-error corrections through the memory, distinguishes events from
ordinary frames in sensitivity, and reaches old inputs. These measurements do not decode the semantics of that
correction into 'store elapsed time'. We have not established whether the dominant failure is inadequate weight,
feature allocation, long-credit optimisation, or downstream use. Zombie hits mostly fit inside the short cache;
their failure cannot be assigned wholly to absent long memory.

The gradient-starvation paper is useful background on competing feature learning, but its main results concern
cross-entropy in a particular regime, not proof of this squared-loss temporal mechanism:
https://arxiv.org/html/2011.09468v4 .

## 3. Predictive heads and imagined-world consistency

The concern is correct. Our current joint model is factored as

`p(P',r',d' | history,a) = p_D(P' | history,a) p_R(r' | history,a,P') p_T(d' | history,a,P',r')`.

Sample one P', then read reward/termination conditioned on that SAME branch. This was the design's coherence
rationale. It does not mathematically guarantee perfect semantic consistency, but avoids independently sampling
outcomes and images. History is still needed for reward deltas/achievement memory; heads do more than interpret
the current image alone.

A before-outcome predictive head q(r'|b,a) is a marginal prediction. The marginal implied by the actual world is

`p_world(r'|history,a) = integral p_D(P'|history,a) p_R(r'|history,a,P') dP'`.

Two independently trained models need not agree. Even two PERFECT marginal samplers can disagree on individual
branches if drawn independently. Exact example: damage probability.1, both health and reward correctly calibrated;
independent draws disagree with probability2*.1*.9=.18. Therefore do not replace the reward used for imagination
with an independently sampled predictive head while keeping the old sampled percepts.

Training-only predictive auxiliaries need not create a second simulated world. They can shape b while imagination
still uses the original coherent chain. But if the auxiliary improves and native generated damage does not, the
model is NOT repaired. Compare the auxiliary forecast with the world-induced marginal and native consequence
readings. Marginal agreement alone does not certify full joint coherence either.

Dreamer is not a counterexample: it predicts rewards/images from a common latent state and trains prior/posterior
state agreement. Its heads do not constitute independently sampled incompatible worlds. We replaced that latent
structure with next-percept diffusion plus a recurrent update; simply importing another reward head does not import
all of Dreamer's state-learning constraints. https://arxiv.org/html/2301.04104v2 .

## Design discussion before a repair

The fundamental choice is the contract, not the number of heads:

1. Treat recurrence plus recent observations as the policy's full state, and give actor/critic a suitable joint
   readout. This aligns information availability; it does not by itself fix generator errors.
2. Keep the efficient visual cache for rendering, but require b_t itself to export sufficient control information.
   Then direct before-outcome supervision on b_t plus actions is principled, not an arbitrary patch. It still needs
   native-world validation and must preserve existing grounding and outcome readers.
3. Require all generation to pass through recurrent memory alone. That is a stronger architecture change with a
   much heavier spatial-retention responsibility; the original store/record design was intended to help with that.

No evidence here selects the third. The current implementation is close to the second in its intended actor
interface, but has not explicitly enforced or demonstrated that contract. I would not blindly add global
next-embedding MSE or remove change-map grounding. Define the target/state role first, then test whether a direct
predictive requirement improves the actual actor-facing representation AND the generated consequences.

All measurements and limits are retained. No further automatic diagnostics or training runs are queued.
