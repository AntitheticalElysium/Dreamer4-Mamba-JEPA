# Full-frame change correction and history-dependent timer test

2026-11-08. User-authorized correction and bounded diagnostics; no overnight training.

## What changed

Canonical change grounding no longer excludes the bottom two screen rows or the player. Every cell has the same
two targets: compare with its previous screen position, and compare with the estimated dominant translation.
There is no semantic selector assigning one target to a known region. Translation is estimated on the complete
image using soft temporal variability at both source and destination. Only geometric source validity is masked;
the identity comparison covers every cell. The existing A9b detector supplies soft pair scores; exact equal
patches score zero, removing its measured content-dependent identity bias. No synthetic refit was adopted.

The shared head now emits [9,9,2], versus [7,9]. This adds 833 parameters (position rows and one output channel),
not another network. BCE is averaged over valid cell/comparison pairs inside the existing G2 group. The identity comparison includes
camera-induced apparent changes; its optimization pressure is therefore new, not just a restored HUD mask. No change to
Mamba, perception, the denoiser, sampler, reward heads, fixation policy or corpus. A new initialization naturally
has a different RNG draw sequence around the enlarged auxiliary: do not call a full old-F4 versus new run an
isolated HUD-mask comparison. A mask-only learning test needs identical NEW architecture, initialization, targets
and streams in both arms, with only the historical exclusion applied in the experimental control.

New targets use a separate versioned store, shape/episode-order/mode validation and atomic completion. Legacy
7x9 arrays are rejected. Historical F4 model weights can still be loaded for evaluation: absent auxiliary position
rows are initialized to zero and its single output is copied into both channels. This is not permission to resume
an old optimizer/recipe. Historical checkpoints and old target files are untouched.

## Qualification on disjoint dev episodes 480--559

Same rows, same simulator-derived truth for both methods. Simulator labels are evaluation only.

| Detector quality | Old map control | New translation comparison |
|---|---:|---:|
| Day AP, 725,588 cells | 0.948776 | 0.951819 |
| Night AP, 321,164 cells | 0.594413 | 0.594218 |

Same-position comparison, 17,839 transitions per body value:

| Value | Actual changes | AP | Recall at 0.5 | False positives on unchanged values |
|---|---:|---:|---:|---:|
| Food | 630 | 1.000 | 1.000 | 0 |
| Drink | 807 | 1.000 | 1.000 | 0 |
| Energy | 605 | 1.000 | 1.000 | 0 |
| Health | 639 | 1.000 | 0.885759 | 0 |

Health's score remains under-confident on some real changes. AP1 is separation, NOT calibrated probabilities or
perfect recall at 0.5. This inherited limitation is retained rather than hidden by a label-tuned threshold.
The first three body values relevant to the missing timer cues qualify at the original 0.5 threshold.
Raw output: QUALIFICATION.json and qualification_arrays.npz.

## Rejected approaches, preserved

- Applying the old detector to unchanged HUD cells without the exact-match correction produces false positives
  on roughly 19--37% of unchanged food/drink/energy values in the first 160-episode slice (AUDIT.json).
- Selecting one local identity/translation winner loses independent changes. In dev episodes320--479, matched
  day AP falls .937530 -> .852707 and night .552300 -> .426455 (SINGLE_PLANE_QUALIFICATION.json).
- Adding synthetic sparse-change pairs to the detector improved HUD recognition but damaged map detection. Both
  noisy and clean refits were rejected; FIRST_TEACHER_HELDOUT.json, REJECTED_FACTORIAL.json and
  CLEAN_TEACHER_CONTROL.json preserve their outcomes. No candidate detector weights entered canonical training.
- An unweighted whole-frame camera fit is strongly biased toward screen-fixed content. The engineering control
  on80 dev episodes gives72.4% shift accuracy versus98.43% for the historical cropped fit. Soft temporal support
  at BOTH source and target recovers98.34% without region labels (MOTION.json). Local winner selection is still
  rejected even with the better global estimate; both comparisons are supervised instead.

## Timer test: real observations establish the missing-history requirement

The exact-history search found a real DEV example: episode376, histories ending at states654 and658.
All eight recent FULL-resolution frames match byte for byte. Incoming actions, rewards and terminations and the
next action also match. In the native evaluation, fixations were matched and the resulting recent percepts were
asserted bitwise equal. Current drink is5 in both.

The last visible drink decrement was at state638. Sixteen awake ticks have elapsed at654, twenty at658. Replayed
hidden thirst is16 and20 respectively. The actual next drink is5 and4: recent evidence alone cannot distinguish
these outcomes, while the older visible reset plus the intervening history can. This is a natural exact alias,
not a lossy-feature probe, nearest neighbour or counterfactual cache swap.

Frozen F4,128 futures per history, same generation randomness:

| History | Correct next drink | Generated decrement |
|---|---:|---:|
| Due, t658 | 4 | 2/128 = .015625 |
| Not due, t654 | 5 | 2/128 = .015625 |

The digit reader was trained on other physical episodes, scores100% on its held-out real data and both actual
successors. An independent geometry check agrees: generated drink tokens are typically only0.032 true digit-steps
from the real unchanged5 token, and nearest-real-endpoint classification also reports2/128 decrements in each.
Mean projection along the actual5->4 direction is.01545/.01507; these projections are NOT event probabilities.
The output is near copying the old digit, not an ambiguous reader score. NATURAL_PAIR.json, NATURAL_GEOMETRY.json.

This single naturally occurring alias proves a failure on this history, not a population prevalence estimate.
The native denoiser cache includes past Mamba exports, so this evaluates the complete historical-information route,
not a current-memory-only test. It does NOT distinguish information lost in memory from information present but
unused by the denoiser.

## Controlled replication

24 real dev backgrounds, two simulator hunger phases per background, same NOOP actions and environment keys.
Each branch has a visible food decrement at a different earlier step. Later, the last eight full frames, incoming
rewards/terminations and actions match exactly, but the next food decrement differs deterministically.
Every base scene replays the stored dev frame exactly and each true successor is read correctly.

Native F4,32 futures per member (1,536 total): decrement probability .059896 when due versus .058594 when not.
Mean paired contrast .001302; background-bootstrap95% interval [0,.003906], conditional on the fixed generation draws, against true contrast1.
This supports the natural example but is a constructed diagnostic family, not an unbiased dev rate.
PHASE_BUILD.json, PHASE_RESULTS.json and their saved bank/futures preserve the inputs and outputs.

## What this says about the missing loss

The HUD does not display the hidden clocks. It displays quantities whose changes provide some reset cues.
The omission removed a fixed-target grounding signal for these cues and could have contributed to poor retention.
However, ChangeMap reads BOTH actual percepts, directly: it can recognize a change without Mamba forecasting or
remembering phase. Adding those targets does not logically force a timer representation in Mamba.

Existing M4 already reads changes perfectly from raw HUD tokens but poorly from the global embedding for several
values. The new exact-history tests establish that older cues really matter and that native F4 does not exploit
them in these cases. They do NOT establish which omitted loss caused that failure. Historical causal attribution
requires matched learning; neither a larger gradient nor a decoder refit would establish it.

## Mechanical checks and preparation

- Full suite:77 CPU/GPU tests pass, including genuine target construction, per-loss gradient routes and exact resume. The HUD-only loss check
  reaches perception, Mamba and its percept projection, with no contribution from map targets.
- Full-frame tests cover arbitrary overlay positions, the former player location, bottom rows, geometric validity,
  biased-detector identity control and rejection of legacy target arrays.
- Historical F4 perception, belief and sampled futures remain BITWISE equal after loading into the corrected code
  (LEGACY_CHECK.json). The enlarged auxiliary's own BF16 logits can differ by one rounding unit; it is not used for
  generation. Do not claim the old auxiliary itself is bitwise equal.
- One test fixture exposed LightGBM4.6.0's loaded-Booster reset_parameter crash; a standalone subprocess reproduced
  exit139. The fixture no longer calls that API. Production prediction never used it. The final complete suite passes.
- Full-store construction reuses old scores ONLY when the detector's source patch and destination patch match
  exactly. The old cache checksum and data configuration are bound to F4's checkpoint (LEGACY_CACHE_IDENTITY.json).
  Direct canonical recomputation checks and completion metadata reside in the new store's change_targets.json.

No new world training or timer repair claim. The correction must be learned before it can change F4's behaviour.

## Full target store completed

`artifacts/saccade_full_frame_change/targets/change_targets.npy`: [2,439,791,9,9,2], covering7,672 training episodes
and2,432,119 transitions. Construction finished in1,688seconds. All first frames are undefined; every identity-mode
cell is supervised on every transition, including the fixed player's location. Geometric validity limits only
translation-mode sources. Old artifacts retain F4's bound checksums.

Every true body-value change in the FULL training split has a nonzero same-position target:
health88,655, food86,693, drink125,155 and energy88,631. There are zero nonzero same-position targets on unchanged
body values. At threshold0.5, all food/drink/energy changes qualify; health qualifies77,613/88,655 and remains
under-confident on the rest. These audits use simulator labels ONLY after target construction.

Target densities above0.5 are3.793% for identity comparisons and8.117% for translation comparisons; they describe
different pairwise tasks. Do not interpret the latter as an8.117% world-event rate: a screen-fixed overlay compared
under translation can differ while its displayed value is unchanged. Both outputs are generic grounding targets,
not semantic event annotations. New optimization pressure and downstream benefit remain to be measured by training.

Direct canonical recomputation of three complete episodes matches the cache-assisted store bit for bit. Full
coverage audit is STORE_COVERAGE.json; construction/checksums are in the store's change_targets.json and PINS.json.
