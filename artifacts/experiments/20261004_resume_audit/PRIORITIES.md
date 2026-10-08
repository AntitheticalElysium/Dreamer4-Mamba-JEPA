# October 4: why the pending runs matter, and what has actually been localized

This is an independent code/artifact/notebook review, not a new performance gate. Running jobs are retained.
Historical results use different panels and readout contracts; numbers below are compared only within their original
matched contrasts. No actor result follows from a fork-supervised probe or an oracle substitution.

## Pending comparisons

**E17 stage 2 is a context-use diagnostic and an attempted context repair.** A16/M16 continue their respective 36k parents
for 6k updates on the shared Raw long ledger, L16, batch 40, seeds 7/8. It asks whether terrain sightings aged 6–15 are
actually used, whether Mamba has a recall advantage, and whether it survives self-feeding and improves decisions. The
matching attention arms are necessary controls. A16 is trained at both seeds; M16 and the final evaluations are pending.
The supervised final input length is **15**, not 16 (window_proof.json); queued evaluations were corrected earlier.

The empirical incentive is bounded in this sample: perfect selective terrain memory removes 2.1% of copy-world error
at lookback 5 and 3.9% at 15. These are clipped selective-copy statistics, not model-capacity ceilings. E17 does not
isolate context length alone against its old short parent: it also changes the ledger, time table and supervised
transition count per update (40×15 rather than 40×5), adds updates and starts a fresh optimizer from parent weights.
Within a fixed long-trained world, w15/w5 is a context intervention,
but it also changes positions/scan length. Content swaps at fixed length are the cleaner use-of-history diagnostic.
Health-layout defects are retained; a health negative cannot be read as proof that longer memory is useless.

**E18 is a repair ablation for the measured coordinate-registration failure.** Teacher fcanvas, 36k, seeds 7/8, same
Raw six-frame pool/loss as the 36k fmamba comparator. At 6k none of the backbones has learned much recall (capture about
0.18–0.21), so the old short fcanvas comparison cannot settle the mature memory question. Rules ask for moved-slot
capture ≥fmamba+0.15, same-slot capture ≥fmamba−0.05 and unseen-cell error within 5%, at both seeds. This is justified
by same-slot/moved-slot measurements plus the independent content/carry interventions in FINDINGS.md, rather than a
generic architectural preference. It is not a health repair or a full long-context canvas trial.

Important implementation limit: historical fcanvas freezes only SSM updates while a cell is absent; its convolution
advances. Both carries contribute to the measured recall pathway. A full-hold inference intervention on the earlier
6k checkpoint had no resolved aggregate benefit (−0.000088 /V [−0.000877,+0.000521]); that does not establish its
effect on a mature moved-slot retrieval stratum. Preserve this limitation in interpreting E18, without changing its
implementation midway. Both evaluators rebuild the state from a finite window. Streaming persistent-carry actor
inference is a separate, as-yet-unvalidated capability.

## Audit of the ten claims

| Claim | Numerical evidence | What can be said about why |
|---|---|---|
| 1. Sparse deterministic consequences are starved | At s7/18k, strict-consequence head-gradient share 0.11% under L1; generate weight 0.029, copy 0.971. End-to-end attempt-mask+skip raises catch 0.002→0.991 but all-token error 0.149→0.256. At 100k, plain s7 catch 0.991 and s8 0.877; s8 table catch remains 0. | Objective allocation and backbone/readout access have intervention support. Rarity alone is not a complete explanation of the seed/order differences. L2 head-only and simple resampling were negatives. These strict labels exclude tree mining; simulator-labelled tree removal is measured separately. |
| 2. Health/HUD dynamics fail | Deterministic damage catch 0.006–0.134 through 100k, fresh arrivals 0/31 at s7 and at most 1/31 at s8, recovery/starvation 0. Corrected E16 posterior: ≥2 damage 0/305, −1 0/32, recovery 0/112. Real-HUD replacement moves H16 real-fitted scores 0.597→0.688 / 0.600→0.687 / 0.585→0.685 / 0.600→0.688 (94–97% of transfer gap). | Drawn health is demonstrably deficient. The oracle identifies a real-fitted readout's dependence on **all HUD tokens**, not health alone or an actor's achievable gain. HUD as a whole is not unlearned: E16 improves aggregate HUD reconstruction by 79.2%. |
| 3. Terminal-position shortcut | 78% of ≥2 training drops are at row4: main [276,258,300,298,339], terminal [185,195,167,158,6172]. Same four frames: attention drawn-drop rate 0 at rows0–3, 1.34% at rows1–4. Repeating just the current frame five times: attention/Mamba draw 1.37%/1.25%, comparable to true-history 1.18%/0.93%. | Controlled dependence on time-row/scan length is established; historical content is unnecessary for these drops. The sampler is a concrete causal candidate. Its contribution must be separated by a de-alignment-only retrain. Distinct from canonical H2's generation-depth alias, although the shortcut family is the same. |
| 4. Delta does not reconstruct ordinary health | Held pre-VQ hit AUC 0.634, code 0.548, decoder h 0.772; ordinary −2 token error 75.4→66.1 with true-future posterior, generate weight 0.09. Terminal −2 error 245.0→16.5. Posterior damage readout still 0/305 on the full panel. | The posterior/reconstruction bottleneck is localized before prior prediction. It is not fixed by more prior samples. Saying capacity is allocated wrongly describes a measured asymmetry; rarity, loss, magnitude and position are not yet independently attributed. E16 is uniform latent **L1**, not MSE. |
| 5. Unseen terrain loses diversity | At H16, newly revealed grass shares 0.611/0.663 versus true 0.483/0.493; entropy 1.64/1.52 versus 2.29/2.25. A local deterministic predictor matches the world (accuracy 0.756/0.760). Independent categorical sampling restores an aligned class histogram but known-content accuracy falls 0.942→0.467 at H16. | Hidden terrain induces conditional uncertainty, although the simulator's map is fixed. This is not proof that an arbitrary history-based deterministic predictor cannot improve; distinguish never-seen cells from previously seen, currently hidden ones. L1 targets a coordinatewise median, not an MSE mean. Joint generation conditional on reliable memory is motivated; it has not passed here. |
| 6. Screen-slot memory | Fixed-length target-sighting swaps raise Mamba same-slot error by +109.202 [92.734,126.301] / +81.129 [66.567,96.026] versus unrelated swaps; moved-slot contrasts unresolved. Donor pull 0.249/0.257 falls to 0.00135/0.00100 when both target carries are reset; unrelated resets retain 0.249/0.257. | Actual content use and its local conv+SSM storage pathway are demonstrated. Reliable moved-slot retrieval is not demonstrated. Spatial attention can route information, so this is a learned pathway limitation, not an architectural impossibility or a long-SSM-only win. |
| 7. Local errors cause trajectory drift | On 36k worlds, 85–87% of false scrolls have a truly blocking target drawn passable. 18k→36k reduces ever-wrong position 0.449→0.298 at s7. Substitution analyses isolate consequence/scroll contributions. 50k→100k position rates are 0.283→0.269 / 0.284→0.270 while H16 improvement is unresolved at s8. | Content→movement→camera→rollout feedback is backed by interventions and traces. Rollout training was negative; its evaluation-position mismatch makes “self-fed training cannot work” unjustified. More budget does not remove health failure or establish H16 decision gain. |
| 8. Decision importance differs from latent geometry | Tree–grass and stone–path squared class-mean distances 39.6/106.1, separations 1.29/1.20 versus lava–grass 5.84. Strict consequence cut >120 misses tree mining. These are contextual-token squared-distance statistics. | Semantic proximity/measurement failure is established. A causal claim that these distances make the optimizer produce false scrolls is not isolated. The squared statistics are not the actual per-token L1 objective. Whitening's old universal explanation was withdrawn; do not resurrect it. |
| 9. Readout is too restricted | Attention trajectory versus final snapshot 0.645/0.629 and 0.646/0.629, resolved +0.017 at both seeds. Newly completed M6 s7: 0.644105/0.624918, +0.019187 [0.005559,0.033441]. One-real-future reference 0.679366 on the same DEV-B roots. | Reading the trajectory helps this probe. Root-aware comparisons also change input dimensionality; do not attribute all gains to context without a capacity control. All are fork-supervised, fixed-continuation readouts, not the model's own deployed actor. |
| 10. Final control unvalidated | No imagination-trained LeWM actor result. Canonical Raw H2 did not pass; canonical TC never bridged. E17/E18 use the Raw encoder's patches, not a completed repaired Raw/TC pipeline. | This is an untested outcome, not an identified physical failure or proof of eventual actor improvement. |

## Shared causes versus independent defects

There is no established single cause. Three interacting families are supported:

1. **Training priorities and support:** rare effects contribute little to bulk prediction, while terminal rows give an
   easier cue. Health fails even with the answer available to the stochastic posterior. This links 1–4; weak physics
   also affects 7 and decision readouts. Geometry can aggravate it, but its independent contribution is unmeasured.
2. **State registration and partial observability:** previously observed terrain needs coordinate-aware retrieval;
   genuinely never-observed terrain needs a conditional distribution. A failed memory should not create unnecessary
   stochasticity on known content. This links 5–7, with different repairs for known and unknown cells.
3. **Readout/control contract:** observed/generated decoder mismatch, context/depth shortcuts and training-label
   selection can produce apparent model failures or apparent probe wins. Better diagnostic scores are not an actor.

The old repeated mistakes to avoid are real-fitted probes as information ceilings, non-significance as equivalence,
global AUC as action choice, outcome-only labels as a ranking objective, oracle future substitutions as attainable
performance, and an active unit or old log row as live compute. “U loses only 0.006, unresolved” did not prove retention.
The previous W/isotropy/SLEEP explanation also needs its later notebook corrections, not the original headline.

## Priority after the booked comparisons

Keep E17/E18 running. Their negative and positive results answer distinct Mamba questions. Do not open another broad
architecture campaign while the reconstruction bottleneck remains. E19 should be separated into matched continuations:

- baseline recipe, same additional update budget;
- death de-alignment alone;
- identical de-aligned batches plus health-context dose.

This separates the layout contribution from the dose. If their interaction later matters, a dose-only arm completes
the factorial; it is not necessary to call the first three a full factorial. Check predictable fresh arrivals, ordinary
hits at every position, recovery/starvation, calibration and false hits. Event-positive-only doses already failed:
catch 0.966/0.946, hallucination 0.297/0.240 versus matching-attempt negatives 0.628/0.653, hallucination 0.022/0.027.
Matching negatives are necessary; target-dependent future-context selection still requires calibration checks rather
than assuming its conditional prior is unchanged. The proposed zombie mask does not cover every damage mechanism.

For E16, first require ordinary true-future posterior reconstruction. A more elaborate stochastic prior cannot
recover a consequence discarded by the posterior/decoder. If deterministic reconstruction learns predictable hits
but not ambiguous ones, a conditional stochastic intervention then answers a different, well-separated question.
Next compare an actual imagination-trained actor with its matched BC control; keep model exploit/false-consequence
checks. The purpose of diagnosis is to make this control test interpretable, not to accumulate proxy passes.

## Literature sanity check

- [Delta-IRIS](https://arxiv.org/html/2406.19320v1), §2.2 / appendix B: decoder reconstructability comes before prior
  prediction. It combines L1, L2 and max-pixel losses, unlike our latent-L1 adaptation. Without max-pixel, average
  L2 improves 0.000185→0.000178 while worst-pixel error worsens 0.018→0.031. This supports separating aggregate fidelity
  from local fidelity, not importing its coefficient or treating the paper as proof of our loss diagnosis.
- [Po et al.](https://arxiv.org/html/2505.20171v1), §5.4 / §6: block-size-1 SSIM 0.766 versus full model 0.855;
  their long-context training matters, and they explicitly do not handle memory beyond trained context effectively.
  fmamba is a related block-size-1 variant, not their whole method. This supports E17, not universal Mamba superiority.
- [TrajGRU](https://arxiv.org/abs/1706.03458) supplies a real precedent for motion-dependent recurrent connections;
  it does not validate our canvas mechanics, carry hold or learned scroll estimate.
- [VaGraM](https://arxiv.org/abs/2204.01464) distinguishes prediction loss from control value and weights by learned
  value gradients. The notebook's labelled health mask is a diagnostic surrogate, not the method or an actor result.
- PROPOSAL.md remains a hypothesis/design history. Its old fatal-direction 0.495 is decoder alignment, not absence of
  generated information; the old whitening-wide causal explanation has later counterevidence. Current local
  interventions have priority over a proposal's literature-derived expectation.

## Resume amendment: implemented without interrupting active jobs

Before this amendment, check_h16_traj opened feature files with w+, saved no optimizer/RNG/best-state progress, deleted
features after each world and did not skip completed worlds. A final JSON did not make it resumable.

New launches journal complete root batches (hashes, flush/fsync, atomic progress), save head state every 200 updates,
retain best validation state and CPU/CUDA/batch RNG, and persist each E16 draw and its generator/end-probability state.
Complete evaluations skip forward calls. Weights, actual inputs/labels, numeric sources, runtime and window are bound.
Corrupt storage is rejected; different contracts cannot reuse partial files. Raw decision scores/rows and features are
retained. Chunked standardization is numerically identical on CPU and avoids the full-matrix float32 allocation.
Long-window attention features use batch4 (68 branches), matching Mamba, under the lane's 3 GiB admission; short-window
attention retains batch16. This conservative memory cap is not a measured CUDA peak or a training intervention.
Legacy unbound reports are preserved; use a different --result-dir rather than overwriting them.

The queued check_damage, check_recall and teval now also checkpoint expensive root batches or recall accumulators,
skip completed worlds, seal contracts and preserve partial evidence. Teacher/rollout mathematics and metrics are
unchanged. check_h16_traj's continuation AUC additionally now averages ties (constant scores 0.5); historical continuation
AUCs from the running old process remain legacy measurements. Primary safe-choice contrasts are unaffected by this fix.

Evidence: verify_h16_resume.json (interrupted/uninterrupted and legacy head/feature/optimizer max abs differences all 0,
matching RNG/best states; changed-layout/contract and corrupted-cache/state controls reject),
verify_queued_eval_resume.json (original numerical parity and interruption parity for damage, recall pool/futures/
imagined and teval; completed forward calls 0). Tests use CPU fixtures; CUDA bitwise parity is not claimed. Original
sources are archived alongside these checks. Default training cadence remains unchanged; E17/E18 already save full
training state each 1000 updates.

The **already-running M6 H16 process remains the old, non-resumable process**. Source edits cannot retrofit its memory.
It is deliberately not stopped. The new cache namespaces cannot collide with its cleanup glob. Completed s7 report
remains valid for its primary comparison and is retained. New E17/E18 evaluator subprocesses load the amended sources.
