# M0 implementation review — 17c2a39c, then f4d1b672

**Verdict: keep production P1 unstarted for one bounded correction pass.** The data,
recurrent-state, gradient-routing, cache, resume and resource checks largely hold up.
The significant model issue is a mismatch between the trained denoiser and the
observation returned by the sampler. S1 also has several metric/contract defects.
The proposed systemd command needs an explicit repository working directory.

This review reads the design closure e81302d6, implementation 17c2a39c and evaluator
f4d1b672 in that order. It does not change canonical code or design, extend budgets,
launch production P1, or claim learned semantic retention from unit checks. Bounded
mechanical updates use audit seed 1897, not production world seed 1 or 2.

```text
real frame -> encoder output map A -> target z
                                      |
                           train raw denoiser to return z
                                      |
                          deployed sampler applies A again
                                      |
                               A(z) differs from z
                                      |
                             belief reads shifted evidence
```

## Findings to resolve

### R1 — High, before freezing P1: the deployed output map changes a perfect prediction

Locations: model.py:24-26,416-432; eval/s1.py:337; LOSSES.md:32-40.

The encoder emits `A(raw) = 3*tanh(layer_norm(raw)/3)`. G1 trains the preconditioned
clean estimate against that emitted target z, without applying A to the estimate.
Sampling then applies A to the clean estimate. A is not idempotent. This is a
composition problem in the agreed spec, which the implementation follows faithfully.

With actual initial encoder outputs on real dev frames, another application of A
changes periphery/glimpse/lighting tokens by:

| Type | RMS component change | Maximum component change |
|---|---:|---:|
| periphery | .0296053 | .0984290 |
| glimpse | .0290616 | .1357235 |
| lighting | .0213426 | .0423509 |

The stronger separating check supplies an exact raw denoiser for a deterministic
conditional target. Its training error is zero; its returned sample differs from
the target by up to **.1156863**. Under identical draws, replacing only the extra
map with a range clamp gives maximum error **2.98e-8**. This is a zero-loss interface
test, not evidence that a learned model has acquired the target or that actor
performance will improve by a specified amount.

Why the citations do not settle it:

- EDM Appendix B.3 derives the denoiser as the noise-conditional mean; B.6 derives
  the raw-output loss used here. The sampler must use that estimated denoiser.
- V-JEPA 2-AC normalizes predictions **before computing its training loss** as well
  as in rollouts (vendored train.py:417-440). It does not establish equivalence for
  normalizing only after a different diffusion objective.
- A box clamp leaves every already valid target unchanged, and also leaves an ideal
  conditional mean inside that box unchanged. LayerNorm plus tanh does neither.

[EDM](https://arxiv.org/abs/2206.00364),
[V-JEPA 2](https://arxiv.org/html/2506.09985v1),
[V-JEPA training code](https://github.com/facebookresearch/vjepa2/blob/main/app/vjepa_droid/train.py#L408-L440).

**Smallest recommended correction:** retain the causal encoder normalization, use an
idempotent range clamp for the sampled denoised estimate, and share that definition
between model.py and S1. An alternative is to include the output map in the training
objective, but that changes the declared loss. Add the zero-error conditional check
to tests, then rerun sampler parity and the S1 generation smoke. The existing sampler
equation test monkeypatches out_map to identity, so it cannot catch this mismatch.

### R2 — Execution blocker: the quoted systemd launch cannot import the package

I ran a harmless transient user unit with the same executable convention, using
`-c` to print only cwd and module visibility:

```text
cwd: /home/antithetical
saccade: None
```

An explicit working directory and absolute interpreter path produced:

```text
cwd: /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
saccade: .../saccade/__init__.py
```

The user manager defaults to the user's home directory, as documented in
[systemd.exec](https://www.freedesktop.org/software/systemd/man/latest/systemd.exec.html#WorkingDirectory=).
No training command was executed. Once the review corrections pass, the command
needs this shape, run from the repo root:

```sh
systemd-run --user --working-directory="$PWD" -p MemoryMax=16G \
  "$PWD/.venv/bin/python" -m saccade.train p1 1
```

### R3 — Medium, before S1: AP depends on input order when scores tie

Location: eval/s1.py:118-120. Stable sorting preserves an arbitrary order within each
score tie, then treats every item as a new threshold. A constant-score predictor on
four balanced labels returns AP 1.0 with positives first and .416667 with negatives
first. Threshold-based AP is .5 in both cases.

Use a threshold-grouped implementation or a standard metric; add constant-score and
permutation-invariance regression checks. The affected S1 outputs include achievement
and departed-entity AP. Their actual trained-score errors are not yet measured.
[AP definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html).

The historical A9b metric uses the same item-ranked convention. I checked the
reproduced teacher rather than assuming a large effect:

| Teacher evaluation | Historical item-ranked AP | Threshold-grouped AP |
|---|---:|---:|
| day | .9430604001 | .9430835717 |
| night | .6120657088 | .6120662291 |

Thus this issue does not materially change the rounded teacher findings. It remains
a real S1 metric defect, particularly for constant or heavily tied probe outputs.

### R4 — Medium, before S1: the classifier does not receive the declared history

Locations: eval/s1.py:350-360; TEST_SPEC.md:61-62.

The classifier tuple contains one previous periphery/lighting percept, action,
fixation, next percept and outcomes. It omits older percepts, previous glimpses,
reward/termination history, and belief. In a direct feature check, changing all those
omitted inputs leaves the 3252-dimensional classifier feature vector **identical**.

It is a valid two-sample test of that smaller feature space; it cannot establish
correct conditioning on information excluded from its inputs. A generator that
ignores older history can match the same last-frame marginal.

Include the common conditioning information actually supplied to the generator
(recent percepts and predicted belief, with the associated action/fixation/signals),
or explicitly narrow the interpretation and provide a separate history-dependent
check. Retain physical-episode splits. The C2ST paper's interpretation is tied to
the supplied sample representation; it does not recover missing conditioning.
[C2ST](https://arxiv.org/html/1610.06545v2).

Also equalize representation precision in compared outcomes: encode() saves real
tokens as FP16 while generated tokens remain FP32. No measured classifier failure
is attributed to that difference here, but storage rounding should not define the
two classes' preprocessing differently.

### R5 — Medium, before interpreting memory: departed labels add hidden survival

Locations: eval/s1.py:192-216; TEST_SPEC.md:53-60.

The documented historical-memory question is whether an entity was seen recently
and is now out of view. The implementation additionally requires its simulator slot
to remain active, and clears the remembered sighting when its hidden mask turns off.
Exit side also uses its first out-of-view simulator position.

This is an alive-and-unseen tracking target, not a pure memory-of-observation target.
It can be a useful separate inference diagnostic. Poor performance alone cannot
identify forgetting: survival and, in some cases, exit direction are unobserved.

For context, on the full dev split I selected frames where a kind was seen within
64 steps and none of that kind is currently visible. Under the implementation's
additional lifecycle/identity conditions, the target is "none" this often:

| Kind | Observable recent-sighting/now-absent frames | Current target none |
|---|---:|---:|
| zombie | 80,008 | 48,914 (61.14%) |
| cow | 136,106 | 44,688 (32.83%) |
| skeleton | 105,438 | 50,155 (47.57%) |
| arrow | 104,103 | 93,721 (90.03%) |

These are two explicitly different target definitions; the table is not a claim
that those simulator labels are factually wrong. Add the observation-history target
needed for the declared memory question, and name survival/identity inference
separately. Neither requires changing M0's architecture.

### R6 — Medium: unlock calibration compares two different events

Locations: eval/s1.py:63,451-460.

The model probability is P(reward > .45); the Brier/ECE target is achievement unlock.
An unlock combined with damage can have net reward <=.45. On all dev transitions:

- 304,898 transitions, 10,528 unlocks;
- 10,413 rewards above .45;
- **115 unlocks missed**, no false positives: threshold recall **98.9077%**;
- even a perfect reward predictor gets unavoidable unlock Brier .000377175 under
  the current event mismatch.

The overall agreement is high, but the events are not identical. Either score a
clearly named large-reward event against its own target or construct the actual
unlock probability using the health-change information in the evaluation labels.
Keep the S0-reference comparison on the same event. Reward bin NLL is unaffected.

### R7 — Medium, before handing S1 to P2: persist the frozen reader

S1 retains `reader` only in memory (main():404-405) and writes a metrics JSON at
line 468. It never saves the fitted reader weights. S2 is specified to use the same
frozen reader after P1. Save that reader with its input schema, probe seed and
encoder-checkpoint identity. Retraining a reader later is not literally reusing
the fitted reader and unnecessarily weakens reproducibility.

The full S1 evaluation on a P1-trained checkpoint remains future work. The smoke
and code inspection do not validate calibration or retention of a trained world.

## Smaller corrections and completeness

1. **Duplicate YAML key:** `heads.change_map` occurs twice (lines 108 and 110).
   PyYAML silently retains only the second, dropping `target` and `rows`. The current
   code hardcodes the intended teacher/map scope, so this does not alter this run,
   but merge the entries before treating YAML as the complete experiment definition.
2. **Resume identity:** train.py:179-184 loads latest.pt without checking its saved
   configuration or world seed against the requested run. Add a small consistency
   check; record source/manifest identity alongside cfg. No registry is needed.
3. **Unmeasured blockers:** perception_blocker skips None metrics without returning
   an unmeasured status. The full-dev kind supports are present, so this is not a
   demonstrated default-run false pass; keep "unmeasured" distinct from "passed".
4. **Test claim:** the whole-sampler test resets at frame 3 of 6 and never exercises
   a history longer than four. Its reference truncates to four when the model's
   composed reach is eight. My longer check validates the implementation, but the
   regression test should retain that coverage. P2 gradient checks cannot yet have
   passed: P2 is intentionally not implemented.
5. **Diagnostics:** the promised per-route gradient monitoring and pre-normalization
   token statistics are not in the P1 loop. It logs total gradient norm/losses, and
   monitor() gives post-normalization effective rank. Either add bounded periodic
   diagnostics or accurately state this implementation's coverage.
6. **Source metadata:** PAPERS.lock gives JEDI's arXiv ID the wrong title. The actual
   pinned PDF has the correct online-MBRL title and matches its recorded SHA256;
   this is a metadata typo, not evidence that the wrong PDF was used.

## Commit-by-commit assessment

### e81302d6 — design closure used by the implementation

The previous causal-BatchNorm concern is resolved by per-token normalization; frame
input/target detachment is explicit; EDM weighting, cache entry, reward representatives,
P2 G1+G3, padded-window coverage and actor/BC details are pinned. The extra output-map
composition in R1 is the remaining design-level mismatch.

### 17c2a39c — data, model, P1 and tests

**Data/targets verified:** 7672 train episodes / 2,439,791 frames after the fixed
397-episode monitor holdout; split episode IDs are disjoint. Target shape and all
episode-first NaN rows agree with that order. Teacher refit with LightGBM 4.6.0
reproduces every saved printed score, including AP .9430604001/.6120657088. Six
day/night snippets have zero feature difference against the original cell loop;
44,586 sampled stored targets from three whole episodes agree exactly at FP16.
Teacher/model target hashes match CHECKS.md. Simulator labels stay in evaluation.
Final-split observation/outcome examples were not scored.

**Model/P1 verified:** causal frame-local perception and bounded tokens; local Mamba
sequence IDs, convolution reset and incoming-state ownership; prefix memory alignment
without rescanning old frames; declared G1 detach routing and grounded gradient paths;
EDM coefficients/raw-output weighting and churn-adjusted schedule; reward value
decoding; AdamW parameter groups and encoder-rate grouping; frame-indexed schedule;
serialized recurrent/cache/stream/RNG state. These findings do not establish learned
entity retention or an adequate belief state.

The independent extended kernel test uses 513 tokens, multiple resets around kernel
chunk boundaries, nonzero incoming state and backward:

- max Triton/reference output difference .000767425;
- max final SSM difference 2.96e-5; convolution carry difference 0;
- input/carry gradient differences <=1.25e-6;
- reset stream's gradients to incoming convolution and SSM state exactly zero.

With nontrivial denoiser gates and heads, activation checkpoint on/off gives identical
loss terms and max parameter-gradient difference **2.33e-10** (FP32 check). Pixel
likelihoods sum to one within 1.63e-5 over a parameter grid, with finite gradients.

**Cached sampling independently verified:** 15 frames, a long pre-reset segment,
branch duplication, cache save/load and reset; identical draws:

| Comparison | Max sample component difference |
|---|---:|
| cached vs full recomputation | 2.27e-6 |
| cached vs eight-frame recomputation | 2.58e-6 |
| cached vs four-frame recomputation | .00107360 |

This supports the production cache and exposes the short test's reference limitation.
The same tests execute the current output map; agreement between two samplers does
not resolve R1's agreement with the training objective.

**Resume:** all 17 tests pass independently on CUDA in 38.92s. Comparing the resulting
uninterrupted/resumed checkpoint models gives max state-dict difference **0**, stronger
than the regression's 1e-3 tolerance. The tiny test ends at update 3 / 1536 frames
because its toy frame cap is exceeded by the long-chunk update; it is not four
production-sized updates or a restart in a separate process.

**Resources reproduced:** 14,439,811 parameters. Audit seed 1897, all real-data losses,
optimizer state, BF16, activation checkpointing, carried states, three warmups plus
five measured updates per shape:

| Shape | Seconds/update | Peak allocated GiB |
|---|---:|---:|
| 16 x 64 | 1.3724 | 4.0059 |
| 4 microbatches x 4 x 256 | 5.1551 | 3.8269 |

That is roughly 9.24 h for the declared P1 update counts, excluding monitor/checkpoint
overhead. This shorter check corroborates the stated estimate; it is not an independent
200-update endurance result. The audit uses a fixed 1e-4 learning rate (encoder x.3)
for resource measurement, not the long-run warmup/cosine trajectory. Sixteen bounded
audit updates are not production P1 and establish no learning quality.

### f4d1b672 — S1

The evaluator has sensible physical-episode splits, checkpoint-matched untrained
encoder initialization, kind-specific precision/recall, outcome likelihoods and
current/eight-step controls with actions and signals. R3-R7 prevent treating it as
finished scientific validation. Fix these before freezing readings; keep the original
budgets and held-out final split.

## The categorical reward change

**I accept it for M0.** A categorical over the declared bins is a proper discrete
likelihood, can sample, and uses explicit train-bin representatives. The existing
tests verify exact 0/1 and empirical sampling frequencies. DIAMOND's reward head
is categorical over sign-clipped rewards; JEDI uses cross-entropy. Those support the
family, not the specific 256-bin symlog quantizer, which is M0's declared adaptation.
[DIAMOND reward implementation](https://github.com/eloialonso/diamond/blob/main/src/models/rew_end_model.py#L64-L71),
[JEDI](https://arxiv.org/html/2605.13013v1).

I recovered the original mixture Outcome class and the diagnostic output from the
local Claude session record. The historical output contains reward-head norms
**322.716** and **787.445**, versus sub-unit norms for the other reported modules;
this supports the gradient-domination observation. The committed categorical overfit
log supports the reported loss reductions. I did not rerun the full historical
mixture/control training comparison, so these are artifact-verified historical facts.

Two wording corrections matter:

- CE gradients are bounded with respect to logits, not every network parameter.
- A clip factor of .001 does not imply an Adam parameter update .001 as large.
  In a constant-scale sanity check, multiplying a gradient by .001 changes the
  first Adam update by a factor **.999990**, because moment normalization cancels
  nearly all the scale. Time-varying clipping and old moment estimates can still
  cause stalls. The precise "1000-fold updates" claim requires measured parameter
  deltas, not only raw gradient norms.

The practical implementation correction does not need to establish that all logistic
mixtures fail, nor justify another optimization campaign before proceeding.

## Literature/source check

See PAPERS_READ.md for the specific sections and what transfers. The relevant result
is not "a paper did something similar, so M0 works":

- EDM's raw loss and sampler must agree on the denoiser (R1).
- V-JEPA normalizes predictions inside its loss, a distinction material to R1.
- Dreamer 4 supports the factorized transformer structure; M0's width/window,
  memory conditioning and omission of logit soft-capping are declared adaptations.
- HarmonyDream supports the implemented rectified group formula; it does not
  guarantee balanced gradients or protection of rare entities.
- PixelCNN++ supports the discretized logistic likelihood and scale floor, not
  an assertion that a mixture reward head cannot work.
- C2ST tests distributions of the supplied features, not omitted history.

## Completion boundary

Before production P1: reconcile R1 and its design wording; freeze corrected S1
definitions/metrics and reader persistence; merge YAML duplicates; make resume identity
explicit; use the verified launch working directory. Rerun the affected unit/parity
checks and the bounded S1 smoke. No longer training budget or architecture search is
needed to close these findings. P2/P3 implementation remains correctly deferred.

**LightGBM:** pin `lightgbm==4.6.0` persistently in the project dependency setup and
install it into .venv. This is routine reproducibility work, not an architecture
decision. This audit used an isolated /tmp install, leaving .venv unchanged.

Evidence files: existing_tests.log, cpu_contract_results.json, gpu_contract_results.json,
long_sampler_results.json, data_evaluation_results.json, teacher_AP_ties.json,
pixel_likelihood_results.json, recipe_profile.json, resume_exactness.json, and the
associated reproduction scripts. Historical mixture code/output are saved separately.
Canonical source hashes are recorded and rechecked in validation.json.
