# Proposed chooser comparison and foveal routing diagnosis

2026-11-02. Discussion draft only. No experiment, fit, continuation or GPU job was
launched for this proposal. It does not amend the production recipe or the completed
campaign's protocol/results/manifest. The user asked to discuss before running.

## 1. Questions that must stay separate

1. Can we replace expensive native contingency/surprise scoring with a cheap endogenous
   chooser, without losing its remote consequence localization?
2. Does useful fine evidence reach future world predictions, and if not, why?

A good answer to (1) does not answer (2). A better eye cannot by itself establish a
working information route. The current P1-F1.5 diagnosis used real histories; this gap
is not explained by absence of P2 self-feeding training.

```
coarse spatial history ------------------------------> physical predictor
                  \                                      ^
fine crop -> 576 active fine values -> mixed 256-D input -> Mamba
                                                       -> 256-D emission
                                                       -> 4 memory tokens --^
```

Source: `saccade/model.py`, Belief.percept_embed/forward/memory and the factorized
denoiser's masks. Four memory tokens are a projection of one emission, not four
independent stores. Dimensions are not proofs of semantic information loss. The main
Mamba carry is larger than its emission and was not proven empty by our failed query
instrument. The current-query fine crop must remain excluded from physical prediction.

## 2. Correct reference for the chooser's remote requirement

Re-read native results, not just their union-coverage column:
`../20261101_p1f15/readings/locus/summary.json`, p1f15, 300 contexts per class.
The metric below is the fraction whose **argmax window contains the inventory counter**.
It is not the probability that the working mixed policy actually looks there.

| Class | Contingency: remote counter | Surprise: remote counter | Contingency: faced tile | Contingency: either |
|---|---:|---:|---:|---:|
| Drink | 77.3% | 44.3% | 7.7% | 77.3% |
| Stone | 39.3% | 33.3% | 57.7% | 97.0% |
| Coal | 13.0% | 78.0% | 84.3% | 97.3% |
| Iron | 9.7% | 75.3% | 59.0% | 68.7% |

Thus coal/stone's approximately 97% union coverage is NOT 97% remote localization.
Contingency and surprise have complementary strengths. A faced-tile-only replacement
could pass the old union metric while failing the user's requirement.

The known benchmark is already inspected development data, not a new untouched test.
Report it for retention and reserve a disjoint development episode cohort for
confirmation; the final evaluation split remains untouched.

## 3. Candidate A first: direct spatial forward head

One small shared action-conditioned model predicts the full next **coarse token grid**.
It needs cross-location communication (e.g. a small attention layer), not only independent
local convolutions: evidence at the faced tile may predict a distant counter change.
All grid locations are treated generically; no HUD, drink, coal or inventory output path.

At arrived frame t, choosing crop f[t], the legal timing is:

```
mu_taken = head(coarse history through t-1, previous actions/outcomes, a[t-1])
mu_noop  = head(the SAME pre-action history, previous actions/outcomes, NOOP)
C[t,w] = sum over cells in w of ||mu_taken - mu_noop||^2
S[t,w] = sum over cells in w of ||p[t] - mu_taken||^2
```

The mean predictor never reads p[t], arrived r[t]/d[t], or the realized fine crop.
The resulting chooser may read p[t] through surprise because p[t] exists before f[t].
Inputs/targets are detached; its loss trains the head, not the world encoder. Recorded
targets are sampled at natural frequencies. This is an expectation used for selection,
not a mean-valued replacement for stochastic imagined worlds.

Start with an eight-frame coarse history, matching the native explicit physical context.
This deliberately omits fine-history main-belief information and any longer implicit
memory not reconstructible from those inputs. Whether that omission loses localization
is part of the test, not an implementation detail. A full main-belief input each frame
would restore the costly sequential collection dependency.

Contingency from differences of conditional means is also a different estimator from
the native common-noise high-sigma denoiser contrast. The comparison must retain useful
behavior; it cannot be sold as algebraically identical. Unsupported offline action
counterfactuals remain unsupported even if the head's factual loss is small.

## 4. Candidate B: sparse endogenous self-distillation

A comparably small priority network predicts separate native C/S maps or their induced
policy from current p[t], past coarse history and a[t-1]. It may read p[t] because it
predicts a pre-crop priority, not a pre-outcome forward mean. Neither candidate sees g[t].
Occasional detached labels come from the current world itself, not a separately
pretrained reference. Distilling native scores retains their heuristic nature: neither
native C/S nor accurate student imitation is proof of learning progress or VoI.

Native teacher labels require **both** branches' correctly replayed pre-action belief
and context, plus the actual/NOOP forecasts with common noise. Existing training
activations do not make that whole computation free. Same-weight input/label generation,
refresh age, teacher noise averaging and reset/prefix handling must be explicit.

First time the complete sparse-label path after backward releases world activations,
including prefix replay and projection/context work. A proposed affordability cap is
10% extra end-to-end P1 wall time; this is for discussion, not an adopted limit or an ETA.
If even a useful sparse refresh cannot meet the agreed cap, do not build a long run
around it. A cheap head benchmark alone cannot qualify the pipeline.

## 5. Proposed bounded chooser test sequence

**First checkpoint-10 frozen feasibility, Candidate A only.** Fit on training episodes;
use episode-disjoint fitting, validation and reporting. Two instrument seeds, a fixed
budget declared before fitting, and no scientific-null extensions. Fix architecture,
history length and preprocessing before the first result. The fixed budget needs a
short timing preflight, not an invented throughput claim. Save factual forecasts, both
score maps and actual fixation probabilities, not only aggregate loss.

Acceptance has three necessary parts:

1. **Retain component localization.** Report remote-counter and faced-tile coverage
   separately for drink, stone, coal and iron, for C and S separately. Do not compensate
   lost iron with better stone or substitute contact coverage for remote coverage.
   Proposed noninferiority tolerance: five percentage points on each demonstrated
   remote capability (drink C, stone C, coal S, iron S), using paired episode-clustered
   intervals. Check whether available event/episode counts can resolve that margin
   BEFORE fitting; otherwise predeclare an inconclusive result rather than silently
   widening the margin. Recheck the native reference across forecast-noise draws.
2. **Retain the implemented policy's behavior.** Apply the same existing normalization,
   mixing, temperature and exploration floor: equal mean-normalized C/S, softmax T=1,
   20% uniform. Report probability mass on remote-containing windows, corresponding
   contact mass and actual sampled coverage with paired random draws. The mixture
   references/margins must be tabulated before fitting; component argmax retention is
   insufficient. Include the previously tested night-entity coverage, static-noise
   robustness, repeated-look and light-level checks. No recency mechanism is added here.
3. **Qualify timing, support and adaptation.** Exact NOOP C=0; no target/crop leakage;
   factual action strata and exogenous action-support strata reported separately.
   Replay complete prefixes with the chooser's own fixations, rather than scoring only
   fixed random histories. Measure the real collection + head training pipeline.

If A passes frozen feasibility, test it against earlier checkpoints (1 and 5) and in
a bounded head-only adaptation replay as encoder/world snapshots change. This checks
initial learning and moving representation tracking; an endpoint head does not establish
that an endogenous chooser can train from scratch. Reset head state only as the proposed
online recipe would, not separately retrain a perfect student for every snapshot.

**B comes second**, if affordable and useful after A's results. Use the same reporting
episodes, policy and acceptance criteria. Match student information and approximate
capacity; report the teacher-label cost separately and compare under an explicit compute
budget. Do not imply that output-map regression and forward prediction have identical
supervision. Failed A does not automatically authorize B or an overnight job: discuss
the failure and cost first. Neither proxy comparison alone proves greater world-learning
or actor benefit; that is a later matched joint-training claim.

## 6. What the brain papers actually support

Read primary papers and their qualification/discussion sections, not an architectural
analogy treated as evidence about Saccade.

* **Schira et al. 2009**, *The Foveal Confluence in Human Visual Cortex*:
  high-resolution fMRI mapped distinct foveal V1/V2/V3 representations. Very central
  V2/V3 representations occupy substantial cortex; this supports privileged processing
  capacity and maintained spatial organization. It does not identify a memory bypass.
  https://openaccess.city.ac.uk/id/eprint/11905/1/Schira_FovealConfluence_JNS2009.pdf
* **Ganmor, Landy & Simoncelli 2015**, *Near-optimal integration of orientation
  information across saccades*: peripheral and foveal evidence jointly improve orientation
  discrimination, with precision-dependent weights and some foveal overweighting. This
  was a particular task with contrast manipulation, not a rule to trust all fine input.
  https://www.cns.nyu.edu/pub/eero/ganmor15.pdf
* **Levy et al. 2001**, *Center-periphery organization of human object areas*:
  central/peripheral biases extend into higher-order object-selective cortex; face-related
  and building-related areas show different biases, including controls changing image size
  and feature distribution. Their proposed explanation concerns different resolution
  requirements. This strengthens the case for differentiated processing across stages,
  not merely a higher-resolution retinal input; the proposed explanation is not a direct
  anatomical demonstration of a dedicated memory route.
  https://www.nature.com/articles/nn0501_533
* **Stewart & Schütz 2018**, *Optimal trans-saccadic integration relies on visual
  working memory*: additional memory load impaired integration. Final analyses used ten
  subjects per feature condition after substantial matching/performance exclusions.
  Supports limited memory-mediated integration, not automatic accumulation of a detailed
  entire scene or one fixed neural storage design.
  https://doi.org/10.1016/j.visres.2018.10.002
* **Williams et al. 2008**, *Feedback of visual object information to foveal retinotopic
  cortex*: peripheral-object category information was decodable in foveal cortex, task
  dependent and related to performance. The authors infer feedback from higher areas
  but explicitly do not identify its anatomical route; BOLD need not mean spiking output.
  The direction is particularly important: peripheral/high-level information returning
  to foveal cortex, not proof of a privileged foveal feed-forward memory highway.
  https://web.mit.edu/bcs/nklab/media/pdfs/Williams.etal.N2008.pdf
* **Chambers et al. 2013**, *Is delayed foveal feedback critical for extra-foveal
  perception?*: the primary abstract reports that late TMS (350–400 ms) to foveal cortex
  impaired peripheral object discrimination. This gives causal task-specific support
  for recurrent processing. Only the abstract was available for this part of the pass.
  https://pubmed.ncbi.nlm.nih.gov/22503283/
* **Duhamel, Colby & Goldberg 1992**: primary abstract reports anticipatory receptive-field
  changes and updating of remembered stimulus locations across eye movements in parietal
  neurons. Supports action-aware spatial correspondence; it does not show that every
  feature or an exact world-coordinate map is copied across saccades.
  https://pubmed.ncbi.nlm.nih.gov/1553535/

Engineering inference, not a neuroscience result: preserve fine evidence with where/when
metadata long enough for learned, reliability-aware recurrent integration to use it.
A larger fine encoder followed immediately by one mixed global vector is not equivalent
to a spatial high-resolution representation remaining available across processing stages.

Also read Du et al. (ICML 2023), *On Uni-Modal Feature Learning in Supervised Multi-Modal
Learning*, sections 3–4: joint models can neglect useful harder features and lose to
separately trained streams; cross-modal interactions can also supply indispensable paired
features. Their supervised late-fusion experiments and simplified feature-learning theory
do NOT prove the same cause in our diffusion/Mamba learner, nor justify importing their
pretrained-teacher solution. The relevant warning is that connectivity and capacity alone
do not ensure a useful source is learned or used.
https://proceedings.mlr.press/v202/du23e/du23e.pdf

## 7. Proposed foveal-gap diagnosis: predictive value before a route repair

Existing results establish extra CURRENT entity information in fine tokens, not its
maximum value for P1's FUTURE targets. Existing crop interventions show little native
forecast benefit; repeating them without a new separating question would add little.

### Test F1: does the extra information help predict the actual future?

Keep the native encoder/checkpoint fixed, use real history, and fit equal-capacity
conditional prediction instruments with identical training scenes, targets and budgets:

* full coarse/history/action baseline;
* baseline + genuine current fine crop;
* baseline + its native mixed input contribution;
* baseline + native post-action emission/memory.

All instruments receive identical fixation/query/time metadata and matched padding.
The shared baseline includes arrived outcomes and the native belief BEFORE the current
fine observation, plus current coarse content and the next physical action, identically
in all arms. It must not accidentally include the post-fixation state being tested.
Past fine history remains matched; only access to the current fine content differs.
The fine arm must get raw content from the WHOLE crop, not only an oracle-selected source
token. The raw-fine arm is the necessary positive control. Do not select test cases on
future prediction improvement; include an unfiltered cohort and a predeclared night/fine-
informative stratum. Use episode splits and an independent confirmation cohort, not only
the eight strongest reader-selected cases. Simulator labels may define evaluation
strata/readouts but do not train the world or chooser.

Two target views are necessary, kept separate:

1. The actual next coarse P1 percept, including source-neighborhood and whole-frame
   scores; a current-kind probe is not a substitute.
2. The next fine evidence from a fixed, pre-outcome source-following query (and a separately
   reported location-independent query control). Current fine fixations must be identical
   across instrument arms. Future queries are evaluation targets from real frames, not
   information fed to the predictor. Use declared screen/action registration and report
   scrolling separately; do not assume a moving object stays in its old cell.

The fine forecast from past history and the cascade's conditional fine forecast GIVEN
the next coarse percept are different questions. Report them separately if both are
exercised; providing next-frame coarse tokens to a physical forecast would be leakage,
whereas the native glimpse stage legitimately conditions on them. Include terminal
successors, but never fabricate frames after the episode ends.

Begin with one-step conditional mean prediction. Squared error is appropriate for that
moment and can show a positive predictive contribution; it is not a stochastic-world
sampler. If the mean test is negative, do not conclude conditional distributions are
identical: information may concern variance/multimodality, the target may suppress it,
or the bounded instrument may fail. Report semantic future-event readouts as diagnostic
checks, not replacements for native target scores. Longer gaps are conditional follow-up
questions, not another unbounded probe ladder.

Interpretation:

| Result | Warranted reading |
|---|---|
| Raw fine helps next coarse prediction, native emission also helps an adapted reader | Information is accessible; native forecast use/training is implicated, not compulsory storage erasure |
| Raw fine helps next coarse prediction, mixed/emitted instruments fail | Localizes a recoverability/learning-difficulty gap; not yet proof of irreversible loss or which module caused it |
| Raw fine helps future fine/semantic targets, not coarse mean | Current physical target may weakly reward the distinction; distributional and instrument limits remain |
| Raw fine helps neither | No predictive positive control; do not claim a routing repair is necessary or that fine evidence has no value |

### Test F2: causal route comparison, only after predictive positive control

The smallest route candidate worth discussing is a bounded **past-fine evidence bank**:
retain the observed nine fine tokens with fixation and time, let future physical queries
attend to them. This tests maintaining spatially explicit evidence rather than forcing
all of it through the present mixed/emitted global interface. It is a learned feature
route, not sprite copying, simulator labels, a privileged death counter, or hard-coded
world coordinates. The physical query at t+1 may read only fine evidence acquired by t.

Proposed matched short joint continuations:

1. Original route, original objective (continued-training control).
2. Past-fine bank, original objective.
3. Identical added bank/attention capacity carrying coarse features of those same past
   crops and the same fixation/time metadata (capacity and gaze-metadata control).

All start from the same frozen base identity, with matched initialization of added
modules, data/fixations, optimizer recipe, clipping, noise and training budget. Perception,
Mamba and denoiser all train jointly under the full original P1 objective, unlike M0.5 v2.
No chooser change, loss rebalance, corpus fix or sampler change is bundled into this test.
The bank is explicitly part of the belief/history available in real AND imagined steps;
it is reset and advanced once, and generated fine evidence receives the same treatment.
It cannot become a hidden channel absent from the actor-side information contract.
Before fitting, test causal time masks (no g[t+1] in the physical query for t+1),
chunk/stream equivalence, resets, fixed-history crop controls and matched noise.

Read out held-out local future prediction, genuine informative-versus-control crop
effects, semantic consequences, night noise regressions and ordinary monitor losses.
Use the same source-defined regions across arms, independent of which arm wins. A
reliable fine-bank improvement beyond the capacity control shows a causal route change
can unlock useful fine evidence under the current objective. It does NOT establish
whether the old input projection, Mamba retention, or memory readout uniquely caused it.

A short negative continuation cannot reject the architecture if even its training
positive controls do not learn within budget. A positive continuation also does not
establish from-scratch or P2/actor benefit. Stop and discuss, rather than extend a failed
budget or automatically promote the bank into P1-F2.

If F1 instead implicates the target/learning pressure, discuss a separate pressure-only
comparison BEFORE coding it. For example, predicting future fine evidence at a past
fixation would supply a self-supervised temporal obligation; it also changes target
coverage and can encourage noise prediction. An auxiliary reconstruction objective or
coarse dropout is not a neutral proof of a routing cause. A later route x pressure
factorial would separate those interventions, but no such four-arm run is selected now.

## 8. Recommended next discussion

Run chooser A before spending on sparse teachers, with remote AND actual-policy gates.
For the primary information-gap question, first F1; only then choose a causal route test
or a pressure test with the user. No chooser-only overnight run is qualified as fixing
the fine-evidence gap. No architecture, acceptance margin, timing cap or launch is
canonized by this draft.
