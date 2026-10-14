# Paired continuous event completion — preregistration

2026-10-14. Saved before implementation or optimization. The user authorized
exactly two matched arms. This study tests whether source-associated endpoint
factorization makes coherent stochastic transitions learnable from observations.
It does not test H16, actor learning, nuisance invariance, long memory, persistent
epistemic information, adaptive computation, or identified minimal physical causes.

## Intervention and matching

- `direct`: the existing `direct_model.DirectWorld`, unchanged, using energy loss.
- `paired`: the same parent and causal corrt/fmamba backbone, with its stochastic
  completion output replaced by a paired contextual event editor.
- Both start from the original D parent recorded by `direct_train.PARENT`, not
  the completed direct-generator weights. Frozen Raw 81x192 observation tokens,
  TRAIN ledger, all actions, input-only brightness selection, batch16, L16,
  15 targets/window, 3,000 updates, optimizer/LR/precision and data/noise draws
  remain those of `direct_train.py`. Initial shared weights use seed7; window
  seed20261014 and noise seed2026101401. No simulator labels, future event masks,
  semantic readers, codebooks, inverse encoders or additional training losses.
- Both use the unchanged first-power global/local energy distance and two
  independent draws per update. Historical direct files and results are preserved.

For source i, retain the existing width256 noise/attention/FF trunk (local4 and
shared4 Gaussian coordinates). Its continuous packet e_i produces gate g_i and
soft weights p_ij over self and the existing four grid neighbors. Out-of-grid
routes are masked. There is no new chance bottleneck, assignment solver, hard
selection, species rule or entropy/gate penalty. The same packet/gate/routing
drives a source edit and its receiving edits:

    output_k = LN[b_k - g_k(1-p_kk) R(h_k,e_k)
                         + sum_(i!=k) g_i p_ik A(h_i,h_k,e_i,offset_ik)]

b is the existing parent copy/generation prediction without stochastic completion
(before the final LN). A shared role-conditioned contextual endpoint decoder
uses the event carrier and the recipient's deterministic belief; hidden width64
keeps output-head capacity close to the original linear generation readout.
Actual parameter counts and training/inference time/memory are reported. Shared
parent and stochastic trunk tensors must match exactly at initialization. The
new readout cannot have identical initial outputs, but both independent noises
must have live post-LN influence and gradients. There is no independent
noise-conditioned destination-generation bypass.

This is a change in endpoint factorization, not an isolated noise-sharing test.
Soft splitting, endpoint collisions and wrong contextual edits remain possible.
No conservation of arbitrary JEPA feature sums is claimed.

## Reference and coordinate limitation

Before the main run, evaluate the original parent and initialized paired world's
event-off reference on the existing panel with both frozen readers. Report mob
presence/counts, absent/multiple sources, static fidelity and camera alignment;
do not assume the deterministic reference is a physically valid afterstate.
Repeat event-off reference evaluation at milestones and final checkpoint.
Decompose sample-mean-minus-reference (systematic correction) from sampled
deviations around that mean, and report results separately on reference-valid
and reference-invalid clean sources. Preserve every root in the primary reading.
If reference-valid sources do not support the event intervention, that mechanism
claim is inconclusive; improved repairs of a bad baseline are not causal movement
evidence. No baseline, cohort or loss is changed after viewing those readings.

The strict14 exploratory source roots are camera-stationary. The deterministic
reference predicts destination screen coordinates whereas source-associated
packets initially use grid slots. We introduce no additional camera-registration
repair. All104 bright roots stay in distribution/static/camera reporting; report
scroll strata. A positive establishes local stationary-camera coordination only.

## Binding result and falsification

Final3,000-update checkpoint only; milestones0/500/1500 describe optimization,
cannot select the primary result. Generate64 draws/root with the old evaluation
seed rule and batch4. Use the frozen real and independent bank readers, the old
real-fan/source eligibility and episode-cluster paired bootstrap (2,000 draws,
seed20261014). Pin historical criteria and all source/input hashes before launch.

Success requires all of:

1. Joint scored35 decoded occupancy energy improves: upper95(paired-direct)<0,
   with independent-reader and clean-source sensitivity as in the original study.
2. Valid source outcomes increase: lower95(paired-direct)>0 on the clean source
   cohort. The paired arm also meets the ORIGINAL absolute absent+multiple,
   duplication and camera caps, not merely improvement from a broken control.
3. Valid movement increases: lower95(paired-direct)>0 on clean cow sources.
   Cow joint source/destination event Brier improves (upper95 difference<0),
   and report per-root move/stay/direction frequencies against independent real
   branches. Hidden-phase mob fans are not exact partial-history belief targets.
4. ORIGINAL actual-alternative coverage, static fidelity and reader-sensitivity
   criteria hold. The historical numeric margins are not loosened or re-fit.

Lower energy with invalid worlds, or validity gained by removing movement,
fails. Numerical/noise/readout failures or inadequate coverage are inconclusive.
Still materially improving held curves make budget-sufficiency claims
inconclusive; do not reinterpret a failure as impossibility of relational models.

## Mechanism and selectivity diagnostics (no additional training arms)

- Report routing max weight, entropy and split mass, event gates, arriving mass
  and conflicts; stratify by decoded valid move, stay and invalid outcomes.
  Routing concentration alone is not evidence of physical correspondence.
- At fixed context, freeze every packet after spatial attention; replace exactly
  one source's full packet from an independent draw. Measure source clearing and
  destination creation jointly with both readers, compare with untouched cells,
  and preserve all interventions rather than selecting examples. Also break the
  removal/arrival tie as an inference-only sensitivity. Changing input Gaussian
  noise before attention is not an isolated source-event intervention.
- Independently shuffle cells across sampled outcomes, preserving marginals.
  Measure the paired arm's dependence contribution to validity relative to the
  original generator's measured2.73 points and real fan's45.90 points. This is a
  mechanism measurement, not a new chosen training/selection objective.
- Report per-cell sample variance, stable/uncertain real-fan ratios, semantic
  alternative variation, stable mean error and static corruption. Lower spread
  produced by collapse is failure. Stable latent variation is not automatically
  nuisance. Continuous gate magnitudes do not prove sparse computation.

## Confirmation, restart and records

Existing14 sources are exploratory. Collect genuinely fresh reserved seeds only
if all binding exploratory criteria pass. Preserve the original reservation
2026101400, input-only selection, frozen readers,64 generated/32 real branches,
minimum128 bright roots and minimum20 variable clean roots INCLUDING20 cow
roots, first qualifying episode-boundary stop, max512 bright roots/1536 episodes.
Fresh data cannot tune weights, checkpoint selection, thresholds or criteria.

Save both models/optimizers, RNG states and paired window/noise ledger atomically
every25 updates; verify interrupted versus uninterrupted resume at actual B16.
Save evaluations by atomic input/checkpoint/shape-bound batches and diagnostics
by resumable units. Use durable process supervision; outputs/logs live under
`artifacts/eda/oracle_v1/paired_event`, not NOTEBOOK. Agents write only concise
experimental entries to NOTEBOOK. Preserve negative results and all source pins.
