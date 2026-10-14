# Reassessment: a bounded optimization control before another architecture

2026-10-14. Assessment only; no continuation or new architecture was launched.
The original3,000-update negative, scientific sources, thresholds, readers and
fresh-data reservation remain unchanged. CPU diagnostics are preserved under
`artifacts/eda/oracle_v1/paired_event/diagnostics/`; all original temporary execution
sources are retained byte-for-byte in `reassessment_preservation.json`.

The evidence supports one matched3k→6k budget extension before changing the
decoder, objective or representation. This is a test of whether additional
optimization can convert measured partial learning into coherence. It does not
make the paired editor the final architecture.

## Last available semantic learning interval

Saved semantic checkpoints are0/500/1,500/3,000. There is no finer measurement of
the last few hundred updates, nor an earlier single-packet intervention curve.
The table is the last available interval,1,500→3,000, with original episode/root
clusters and paired2,000-draw bootstrap intervals. No checkpoint is selected.

| Paired, primary reader | 1,500 | 3,000 | Change,95% interval |
|---|---:|---:|---:|
| Valid source outcome | 70.42% | 68.42% | −2.01pp[−8.82,+4.13] |
| Valid movement, all14 sources | 5.80% | 12.61% | **+6.81pp[+2.46,+11.16]** |
| Valid cow movement,8 sources | 6.25% | 9.57% | +3.32pp[−1.37,+7.23] |
| Source remains alongside another destination | 18.08% | 9.49% | **−8.59pp[−12.05,−5.36]** |
| Source absent | 3.01% | 10.38% | **+7.37pp[+4.13,+11.16]** |
| Joint occupancy energy score | 0.027167 | 0.024876 | **−0.002291[−0.004412,−0.000638]** |
| Dependence contribution to validity | 11.21pp | 10.27pp | −0.93pp[−5.65,+3.51] |

The second reader corroborates movement growth(+6.25pp[2.68,10.16]) and declining
duplication. Its cow-movement change resolves(+3.71pp[0.59,6.45]); the primary cow
interval does not. Cow joint-event Brier improves under both readers:
−0.04889[−0.08630,−0.01263] primary and−0.10168[−0.16669,−0.04724] second reader.
Temporal improvement in dependence remains unresolved under both readers.

There are61 additional valid movement draws:36 skeleton,17 cow and8 zombie.
Thus much of the aggregate gain comes from only3 skeleton roots. The direct
control's movement decreases over the same interval. Paired-minus-direct change
is+8.93pp[3.57,14.06] primary and+9.04pp[4.24,14.18] second reader.

It is incorrect to say only loss improves. Movement, duplication and event-law
learning improve, while source absence worsens and overall coherence remains
unrepaired. This supports a bounded optimization question, rather than an
assumption of convergence or a prediction that more training will solve it.

Evidence:`curve_reassessment.json`, with raw metrics and shuffle curves in
`curve_reassessment_raw.json` and`curve_reassessment_shuffle.json`.

## Does the actual objective notice the measured joint failure?

A CPU diagnostic declared its contract before computation at21:32:48 AEDT.
It uses the exact current`direct_model.distance`, with no fitting or parameter
updates. On each of the14 strict source roots, real branches0:16 predict
disjoint branches16:32. Each of the five source-neighborhood tokens is then
permuted independently across prediction draws, leaving every other token in
its original row. Every token's full192-D marginal is preserved exactly.
There are32 fixed shuffle repetitions/root, averaged before root-cluster intervals.

A common whole-frame permutation preserves score bit-for-bit. Two frozen readers
measure the resulting source/destination validity. The primary score excludes
prediction-prediction diagonal terms; a full empirical32-future score with
diagonals retained is reported separately.

| Exact continuous-token energy score | Coherent real ensemble | Endpoint-independent ensemble | Penalty,95% interval |
|---|---:|---:|---:|
| Primary16/16 split,all14 roots | 0.036499 | 0.038769 | **+0.002270[0.001341,0.003205]** |
| Same split,cow8 | 0.030517 | 0.033510 | **+0.002993[0.001743,0.004078]** |
| Full empirical32 sensitivity | 0.035763 | 0.038336 | **+0.002573[0.002048,0.003028]** |

Both readers give validity100%→52.59%(cow100%→50.42%). On the same targets,
the final direct/paired score gaps above the coherent split reference are
0.004753/0.005540. The shuffle penalty is47.76%/40.98% of those gaps; those
ratios provide scale, not an additive attribution of model error. Point mean
and coordinate median score0.06315/0.06150, versus coherent sampling0.03650.

The current objective recognizes this controlled loss of joint coherence and
prefers coherent alternatives over point statistics. A blanket claim that the
energy objective is blind to coordination is unsupported on this panel.
Scalar sensitivity does not prove effective parameter gradients, optimization
efficiency, correct weighting of rare consequences, or sufficiency of the target.

The five shuffled contextual tokens carry86.60% of scored real variance on
average,98.63% for cows. This is joint continuous completion, including contextual
re-encoding, rather than a pure permutation of binary occupancy. The usual
off-diagonal score is fair for independent ensembles, but without-replacement
permutations couple rows; no exact unbiased population-score claim is made for
that shuffled finite ensemble. The empirical32 sensitivity agrees. Fans still
condition on the full simulator state, rather than solely the model's history.

Evidence:`objective_sensitivity_{contract,scale}.json` and
`objective_sensitivity.json`; the exact executed source is retained.

## Entering normalization is an independent output-support fault

The frozen GPU result already localized164 entering cells with reference RMS<0.1,
329.27× pre/post-LN variance amplification and81.28% of full sampled variance.
The new CPU reference-backend recomputation identifies the composition of those
references: **93.793–99.164%** of copy weight goes to an exactly zero-padded
candidate, mean**97.850%**. The generation candidate averages0.937% weight.

This composition is CPU FP32 corroboration, not bit-exact BF16 CUDA replay.
Normalized-reference mean/max discrepancy is0.00746/0.05785. Low-reference sets
agree on103/104 roots, with one threshold-adjacent mismatch. Of73 entering cells
outside that low-reference stratum,72 enter at the bottom row and predominantly
choose STAY. Zero-padding explains the164-cell stratum, not every entering cell.

The exact code factorization and CPU autograd establish a further distinction:

- final LN operates independently over each token's192 features;
- the objective selects only35 interior outputs;
- training uses observed frames, with no generated-output feedback;
- actual saved-output gradients on all46 excluded tokens have max0 and
  **zero nonzero elements**;
- rescaling only unscored synthetic pre-LN inputs by1/1,000 changes the scored
  loss by0 and scored pre-LN gradients by0.

Thus numerical amplification through excluded final outputs cannot divert the
current training gradients or explain the interior deficit by self-feeding.
Earlier unscored packets can still affect scored outputs through attention and
neighbor writes; that is a separate trainable pathway.

Keep a numerical correction separate: quarantine excluded outputs or define
valid copy support, proving the35 scored outputs and their gradients unchanged.
That control is not learned unseen-terrain prediction. Full-frame recursion
eventually needs an actual supported generation/persistence treatment for those
cells. Changing numerical handling alongside a training continuation would
obscure the budget comparison without repairing its interior learning problem.

Evidence:`normalization_gradients_cpu.json`,
`normalization_composition_{all,summary}_cpu.json`.

## Requirements and implementation choices

WHITEBOARD describes required behavior. It does not determine a decoder.

| Required property | Evidence so far | Choices still open |
|---|---|---|
| Belief retains what history determines | Simulator64-frame counter tracker; learned same-slot memory | Mamba context/carry training, addressing, persistent hypotheses |
| Consequential content survives representation | Patch readouts and supplied-future expressibility | Joint JEPA learning, local content, consequence gradients, nuisance invariance |
| Irrelevant variability is discarded | Renderer nuisance measured; learned invariance unestablished | Relevance supervision, temporal/predictive abstraction, controlled observation augmentations |
| Conditional joint outcomes are calibrated | Direct noise learns alternatives; paired dependence increases | Direct noise, inferred continuous/categorical innovations, flow/diffusion |
| Individual transitions are compatible | Limited one-root paired semantic interventions; original coherence fails | Content transport, persistent entities, structured operators, learned joint generation |
| Sampled facts and hypotheses persist | Requirement supported by environment; learned branch persistence unestablished | Explicit memory or sufficiently trained recurrent belief |
| Outcomes support decisions efficiently | Simulator sampling has value; current small model fits6GB | Reuse deterministic work, compact stochastic computation, amortized risk/value |

These properties can in principle be learned; constraints can exclude particular
invalid solutions. A source-use constraint protects copied provenance, but does
not solve occlusion, creation, transformed appearance or consequences. No evidence
requires context-independent raw patches, conservation of feature sums, a
categorical chance variable, hard routes or a separate inverse encoder.

Observed transitions can teach outcome frequencies, content persistence and
consequences on supported actions; observable resets can teach deterministic
counter memory. Unseen facts remain uncertain until observed. Variability alone
cannot identify nuisance, since random mob motion matters while random rendering
does not. Reward/consequence evidence or other justified relevance assumptions
are needed. Architecture cannot supply missing action coverage or unobserved facts.

Observational distribution learning also does not uniquely identify physical
random coordinates: invertible reparameterizations of the innovation can describe
the same sampled law. Operational evidence should be compact interventions with
compatible departures, arrivals and downstream consequences, plus correct joint
frequencies and persistence. The current eight one-context changes fall short.

## Architectural alternatives, grounded in the literature

- **Correspondence/transport:** [ITC](https://arxiv.org/html/2605.16457v3) assigns
  actual previous/generated content with source-use and conflict resolution.
  It tackles provenance and duplication. Its discrete appearance tokens differ
  from our contextual JEPA targets; a whole feature cannot be assumed valid at a
  new location. Content-preserving recontextualization remains a learning question.
- **Structured transition operators:** [Finn et al.](https://arxiv.org/html/1605.07157v2)
  learn motion transformations and composite previous appearance, rather than
  recreating it independently. A learned transform/compositor is a generic
  alternative to signed endpoint edits. Soft advection still permits blur and
  splitting; the pixel experiment does not validate our continuous latent variant.
- **Persistent entities with inferred innovations:**
  [LPWM](https://arxiv.org/html/2603.04553v1) separates particle attributes and
  continuous per-particle actions with an inverse model and conditional policy.
  [FLAM](https://arxiv.org/html/2602.16229v2) factors inverse/forward action models.
  Both show useful factorization, while their reconstruction machinery and
  actionless-data setting differ. Our previous collapsed implementation does not
  reject this family. A concurrent S4 audit strengthens that qualification:
  on the same26 FIT futures, supervised flat inputs classify26/26 by500 updates,
  mean-pooled inputs classify5/26, and the existing attention-posterior classifies
  16/26 at3,000. Its changed-tile attention mass0.0413 matches uniform pooling,
  while futures differ in3.35 tiles on average. These are memorization/separability
  controls, with different flat/pooled capacities; they do not demonstrate a
  repaired unsupervised posterior or learned prior. They do show that spatial
  distinctions are available and poorly accessed by this posterior implementation.
  Evidence:`learned_innovation/audit_channel_{D,E,B3000}.json`, source
  `audit_s4_channel.py`, commit`3960e573`.
  [Stochastic MuZero's author account](https://www.julian.ac/blog/2022/05/15/planning-in-stochastic-environments-with-a-learned-model/)
  demonstrates afterstate/chance learning with future-observation codes and
  value/policy/reward supervision, rather than visual-state fidelity guarantees.
- **Direct joint distributional generation:**
  [JEDI](https://arxiv.org/html/2605.13013v1) learns predictive representations
  jointly with latent denoising, keeping representation/objective explanations
  alive. [EAR](https://arxiv.org/html/2505.07812v1) supports energy-trained
  continuous generation, but uses conditional autoregressive/refinement structure
  and reports that reduced generator LR stabilizes training. Neither validates
  our one-pass residual decoder or sparse computation.
  [Flow-JEPA](https://arxiv.org/html/2608.29029v3) explicitly leaves trajectory
  prediction versus stochastic-flow contributions unisolated.

The strongest structural candidate, if optimization stalls, is retaining content
identity while sampling its transition and recomputing compatible context and
consequences. This is a research principle shared by several families; the
current evidence does not choose a particular realization.

## Smallest next training experiment to discuss

One matched continuation of both existing arms,3,000→6,000 updates, with no
architecture, representation, objective, data, numerical-support or LR change.
Restore model weights, Adam moments, all RNGs, sampler and global update. Current
LR is1e-4 after1,000 warmup updates; no schedule decay/restart is involved.
This extends an unchanged training prefix, unlike an intervention that bolts a
different objective onto a pretrained world.

The original trainer/criterion hard-pin3,000. An extension therefore needs a new
follow-up contract/runner which imports the frozen scientific implementation;
it must preserve the original negative and caches. Endpoint6,000 is fixed,
with4,500 as a curve diagnostic rather than selectable evidence. No unconditional
second seed or fresh-data collection. Original validity/movement/joint-law/static
criteria and fresh reservation remain unchanged.

The question is whether additional optimization reduces invalid outcomes while
movement, event-law fidelity and paired endpoint dependence improve under both
readers. A lower score, fewer duplicates offset by more disappearance, or a
single context's successful packet switch cannot establish repair.

If score improves while physical validity/dependence do not, the tested extension
fails the budget-only repair hypothesis. If some semantic metrics improve but
the original criteria still fail, record partial learning and an unsuccessful6k
endpoint; do not launch indefinite extensions. A negative cannot identify whether
the remaining limitation is representation, objective, parameterization or their
optimization. That would justify a targeted separating intervention rather than
an architecture search. This continuation is recommended for review; it is not
running.
