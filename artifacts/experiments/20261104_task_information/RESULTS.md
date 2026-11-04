# Task-relevant incremental fine information — completed 2026-11-04

**Outcome:** one natural-frame, task-relevant fine-readout witness survived the
controls. A robust, independently confirmed fine-only capability was NOT
established. Do not use this campaign to declare the mixer, Mamba or denoiser
broken, or to justify a particular repair.

Frozen F1.6 checkpoint10:
`34bda77fe5ec5226c4f003301c972d918e2d4a4a554a5a6f279fcabbe922b045`.
World, sensing, chooser and architecture parameters were unchanged. Supervised
diagnostic readers were fitted; simulator labels never trained the world.
F2 scientific results were not inspected for this campaign.

## What the question requires

Keep three claims separate:

1. Fine evidence helps identify a task fact beyond the tested coarse/history
   observers.
2. That fact changes health/reward under matched actions and random keys in the
   actual environment.
3. The native belief and generator use that evidence.

This campaign investigates 1 and 2. It deliberately does not require native
generation to already master the fact, and does not establish 3. A fine latent
being predictable or decodable is not alone a task-value demonstration.

## Broad conditional readers, and the falsification that matters

Data: first400 training episodes (360 fit,40 calibration); first400 dev episodes
(250 discovery,150 confirmation). Natural genuine random crops yield992,577
training and1,004,976 dev map queries over128,564/130,261 frames. Confirmation
night subset:138,219 queries in102 episodes.

Baseline inputs include the FULL current coarse scene and lighting, eight full
past coarse scenes, native prior Mamba output, past actions/outcomes, full raw
3x3 coarse inputs over the same current/history frames, query/gaze positions,
and native pre-glimpse TSP pixels. The current fine crop and current corrected
belief are excluded. Prior native memory is allowed to contain earlier glimpses.
This is not an unlimited exact tracker or a readout of every raw recurrent-carry
element. Histories are strictly prior and remain within the source episode.

Two reader parameterizations x two seeds; identical1200-update budgets and
class-balanced diagnostic training; natural-frequency calibration on the40
held-out TRAIN episodes. Equal-parameter arms get zero extra input, native
fine tokens or real fine pixels. A later, explicitly declared control supplies
the full TSP crop in the extra branch; its baseline is chosen on TRAIN
calibration only. All original outputs are retained.

The strengthened four-reader comparison shows a confirmation/night kind NLL
benefit of0.002216 (95% episode-bootstrap interval0.001608–0.002884), from a
baseline0.049547. Brier benefit0.000887 (0.000642–0.001156), from0.020282.
However, threat-vs-background and arrow-heading gains fail the stricter
replication rule. The raw-pixel arm also does not establish a consistent gain.
See `semantic_results_strong.json`; original metrics are retained separately.

**These conditional-reader gains are not the final prerequisite.** The existing
small S1 LOCAL coarse reader correctly identifies all24 initially selected
threats at near-certainty (true-kind scores at least0.9998). Thus larger baseline
readers were weaker on those examples. Their simulator consequences remain
real but do not establish fine necessity. `local_source_reader.json` records
this falsification. Do not cite the initial24 sources or the barrier example
as successful incremental-fine examples.

## Targeted genuine acquisition, with existing stronger controls

No new readers were fitted after that falsification. Search all current map
cells for night hostiles with S1 true-kind score<0.8. There are4,915 eligible
queries. Take at most2,000 distinct source frames, prioritizing no visibility
in the preceding8 steps and then corpus order. Generate a genuine focused
glimpse from each recorded frame, retaining its coarse input and native prior.
This is an enriched acquisition search, not an estimate of prevalence or a
deployable fixation policy. Selection uses current labels, never future damage.

| Control | Result |
|---|---:|
| Four wide fine-augmented readers must all beat the strongest five coarse instruments | 0/2,000 |
| Existing local fine readers both correct >=0.9; all five coarse instruments wrong and true-kind score<0.5 | 356/2,000 |
| Add the existing **matched class-balanced local coarse** reader p_0 | **144/2,000** |

The sixth coarse control prevents different training class priors from supplying
the apparent gain. Final144 split130 discovery/14 confirmation; kinds89 zombie,
35 skeleton,20 arrow. Select the first qualifying source in up to12 different
episodes per stratum:12 discovery and11 confirmation sources. This is a
post-result search with logged controls, not a pristine confirmatory study.
Local softmax scores are class-balanced classifier scores, not calibrated
environmental posterior probabilities.

Fresh exact replay and two-sided checks:

- Source frame exactly equals the stored uint8 frame.
- Remove only the selected foreground: mask false, health zero, and update
  mob occupancy only where necessary; arrows do not own mob-map entries.
- Same gaze and prior for factual/removed frames. Recompute both complete
  current coarse scenes, lighting, TSP and fine crops genuinely.
- Require all six coarse instruments still wrong on the factual source; both
  fine instruments correct on the factual source **and** the removed background.

Only **4/23** pass, all discovery sources; **0/11 confirmation** sources pass.
The other19 often still call some entity on the removed background. They are
not accepted as successful fine-evidence witnesses. All probabilities and
rejections are saved in `focus_paired_readers.json`.

For the23 source pairs, actual raw coarse cache parity is exact; lighting
parity is exact; maximum coarse-token difference under fresh BF16 batching is
0.022583. Final reader qualification is made on fresh encodings, not assumed
from cache. Current fine content is explicitly varied while asserting coarse
reader features are unchanged.

## Simulator task relevance — all fixed plans retained

For every selected source, factual and removed worlds receive identical six-step
constant plans NOOP/LEFT/RIGHT/UP/DOWN/DO under32 common random-key sequences.
Termination absorbs. The simulator has auto-reset disabled. Record cumulative
health decrements, actual environmental reward, termination and final position.
No native forecast or recorded reactive future action supplies these outcomes.

| Paired-reader survivor | Kind | Largest paired mean damage difference over the six declared plans |
|---|---|---:|
| Episode2, t227 | skeleton | 0 |
| **Episode7, t221** | **skeleton** | **0.375** |
| Episode42, t217 | arrow | 0 |
| Episode62, t497 | zombie | 0 |

Only episode7 meets both observation and six-step consequence checks. Its
skeleton is3 cells above and1 right of the player, light0.01975. Both local
coarse readers and all four rich coarse/history observers call the source
background. Their fresh skeleton scores range0.00048–0.01659; both fine
readers call skeleton (scores0.99941/0.99988). Both fine readers call background
after removal. Native pre-glimpse TSP chooses the absent alternative for both
inputs in the source contrast check; it does not already distinguish this pair.

Its simulator slot had been active since t199 and was not visible between
t199 and t220. Earlier uses/activations of that slot were visible, last at t93.
Do not call this a never-seen simulator identity or assume identity across slot
reuse/revival. All coarse observers receive the actual native prior state.

Report NOOP explicitly, rather than presenting a selected maximum as value:

| Fixed six-step plan | Mean cumulative damage, present / removed | Keys where damage differs |
|---|---:|---:|
| NOOP | **0.3125 / 0** | **5/32**, each difference2 health |
| LEFT | 1.3125 / 1.3125 | 0/32 |
| RIGHT | 0 / 0 | 0/32 |
| UP | 0.375 / 0 | 6/32 |
| DOWN | 0 / 0 | 0/32 |
| DO | 0.3125 / 0 | 5/32 |

NOOP reward is -0.03125 /0. No death in these branches. This demonstrates
causal relevance to a health forecast, not an optimal-action flip, actor
improvement or population expected value of fixation. The removed world is a
current-state intervention, not a claim about two identical feasible pasts.

`natural_witness.png` displays the unchanged natural image, the exact3x3 coarse
sensor crop, the full-resolution crop and the removed crop. No brightness or
contrast manipulation is applied.

## Reliability check limits the positive finding

Render each of the four fixed source pairs under64 fresh common night-static
keys. No refitting, replacement sources or expanded search follows this check.

| Source | Local coarse S1/p_0: present recall | Local fine g_0/g_1: present recall |
|---|---:|---:|
| Episode2 | 92.2% /87.5% | 96.9% /100% |
| **Episode7** | **87.5% /75.0%** | **68.8% /89.1%** |
| Episode42 | 65.6% /92.2% | 96.9% /96.9% |
| Episode62 | 32.8% /25.0% | 81.2% /96.9% |

These are noise-resampling instrument measurements, not independent episodes.
For episode7, S1 correctly recognizes the removed background64/64, while fine
readers do so55/64 and48/64. Thus its natural coarse error is an unfortunate
rendering realization, not a stable claim that coarse vision can never see that
skeleton. Fine is not uniformly more reliable. Episode62 shows the clearest
repeated fine-readout advantage, but none of the declared six-step action plans
shows a health/reward difference from removing that zombie.

All four native TSP contrast checks choose absence for both alternatives.
Fine reconstruction separates both alternatives for episodes2/62, but not7/42;
for episode7 its present-side contrast error is almost tied. The source fine
classifiers still distinguish the fact. Decoder fidelity, latent readability
and a native belief update are distinct measurements.

## Other retained negative evidence

- Exact coarse-sensor alias search across1,000 natural newly exposed night ore
  cells and stone/coal/iron/diamond alternatives found **zero** exact raw coarse
  aliases. This is not proof no alias exists elsewhere.
- Previous campaign recovered70,208/70,208 HUD values from coarse tokens and
  found no task-relevant coarse misses in17,518 local resource/blockage frames.
  Those easy cue families cannot be treated as established fine-required tasks.

## What is and is not established

**Established:** a focused real fine observation can correct a task-relevant
mistake made by all the tested coarse/history observers on a natural frame.
That observed fact has a matched simulator damage effect. This is a legitimate
small witness of accessible, useful fine evidence under finite observers.

**Not established:** information-theoretic absence from every possible coarse
decoder or exact tracker; a robust independently confirmed task where fine is
necessary; a population policy benefit; native acquisition/retention/use of the
cue; or which component should be repaired. The broad prerequisite remains
unsettled. A coarse classifier error alone is not an information-loss proof.

The campaign stops here for discussion. No world run, pathway repair, further
candidate scan or background follow-up is queued.

## Evidence, checks and reproduction

- `PLAN.md` preserves initial declaration and explicitly labeled later controls.
- `SUMMARY.json` aggregates final sources and all six-plan outcomes.
- `focus_balanced_results.json`, `focus_paired_readers.json`,
  `balanced_consequences.pt`, `witness_checks.json`, `witness_arrays.pt` preserve
  the final natural, paired and independent-render results.
- Initial fits, coarse controls, rejected examples and all negative searches
  remain in this directory. `SOURCE_PINS.json` predates full fitting;
  `FINAL_PINS.json` records the final source/artifact hashes. The fit source was
  later extended with the separately reported coarse_tsp control, and PLAN was
  appended; the original pins were not overwritten.
- History-index check passes1,027,688 prior same-episode indices. All23 pairs
  have complete `[2,6,32,6]` finite outputs and no simulator-source rejection.
  Final Python sources parse. World source hash matches the pre-fit pin.
- GPU reproduction (when free), from repo root: `python -m
  artifacts.experiments.20261104_task_information.focus --local --balanced-coarse`,
  then `focus_pairs` and `witness`. CPU simulator: the same module prefix plus
  `consequence --focus --balanced-coarse`. Reporting: `summarize`.

The relevance criterion is consistent with [Sprague & Ballard's task-based eye
movement model](https://proceedings.neurips.cc/paper_files/paper/2003/hash/da4902cb0bc38210839714ebdcf0efc3-Abstract.html)
and [EPIG's emphasis on target predictions](https://proceedings.mlr.press/v206/bickfordsmith23a.html).
Neither paper establishes a Saccade sensing benefit; only the measurements above
address that question here.
