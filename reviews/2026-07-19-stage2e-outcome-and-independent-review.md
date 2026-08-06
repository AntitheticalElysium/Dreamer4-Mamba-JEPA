# Stage-2E outcome and independent review

Date: 2026-07-19

Status: **VALID NEGATIVE — global temperature/zero-bin calibration rejected**

## Executive verdict

The frozen C-LR calibration experiment is valid after one caught and repaired
device-parity defect. The CAL-selected arm was fixed before any selected-arm
DEV result was observed. It then failed 4 of 11 registered gate conditions.

The selected two-scalar calibrator:

- reduces C-LR's mean absolute predicted return on truly zero-return fork
  suffixes from `.06404` to `.04177`;
- does **not** reach the required A-relative budget: its delta is
  `+.03230 [.02432, .04041]`, entirely above the `+.02` ceiling;
- improves K8 signed Pearson by `+.01634 [.00456, .02642]`;
- significantly worsens K8 event MAE by
  `+.01334 [.00554, .02294]` and lowers event magnitude;
- changes the selected fork outcome at only one of 21 informative anchors.

This is a useful negative. C-LR's deployment failure is not removable by one
global temperature and one global zero-bin offset under the registered local
categorical operator. No alternative Stage-2E arm may now be selected on the
spent DEV set.

FINAL, planner execution, Mamba transfer, replication, actor/critic training,
reliability weighting, and online policy training remain **NO-GO**.

## 1. Chain of custody

### Immutable inputs

| Item | SHA-256 |
|---|---|
| C-LR checkpoint | `60657857d5eb811e2ce2dc66ec953301c4865e3ac7a203ca2e5dca3c237e5bae` |
| C-LR full state | `93509072da3bf55c21e1e83b023ab47aa3cc49af52d4c2cac0121ceca72afe49` |
| CAL data | `709e9646ce5ee1cf36ef4118f6b5d4482751a300b8c97186929af6f0271b27ad` |
| DEV natural | `5335cf6133ab16aa1f0ec3f6bd6c3a506c706424985ad694d002026a22ea175e` |
| DEV terminal | `14732eb37f475d38d2aa91834bd64b5ce04598398d28a348452922b303018ccf` |
| DEV fork bundle | `d570ae8d82592e9153d1db3025ce3f4bdbd125e370c838f58cb882ab33bafdb8` |
| Stage-2C raw control | `e67fd07706bb458b94924678f8c43b1f01fd5d44182e7139bde6123ea596b4a5` |

### Commit order

1. `8f5f1c5` implemented the calibration math, split-safe fit/evaluation
   runners, paired gate, and tests.
2. The CAL-only fit ran on a clean tree and was recomputed a second time.
   Every logit digest, fitted scalar, optimizer result, and selected arm
   matched exactly.
3. `c25e4cb` committed the fit artifact before DEV.
4. `e2b6437` pinned the committed fit SHA in the DEV evaluator.
5. The first DEV invocation stopped in the E-I reconstruction assertion
   before reporting or evaluating any non-identity arm.
6. `e2b40ca` repaired the canonical-device identity path and added a CUDA
   regression. No fitted parameter, selection rule, gate, threshold, or model
   value changed.
7. The clean rerun produced the report/raw artifacts, followed by the
   preregistered paired analysis.

This ordering preserves the purpose of the split. The only information seen
before the repair was that E-I differed at floating-point bit level; no
selected-arm metric was available and no experimental choice was changed.

### Outcome artifacts

| Artifact | SHA-256 |
|---|---|
| `stage2e_calibration_fit.json` | `6c9f436fb64e1c6b92fa9cc3b351e24b4a49063cc430145f29a519a874351a0d` |
| `stage2e_report.json` | `4e0320980cd9634df2b34d4aec33e123754afd8bf3dc911debcb3cfee8250d6e` |
| `stage2e_raw.json` | `d08160e39ed621febeee888e78617698428f982d86b2f73571ced71aa8bf019d` |
| `stage2e_analysis.json` | `5a58adf1cedc10331baba0888061e166ee8d2a10c8d74f7e4a76ab36f32a8542` |

The evaluator records commit `e2b40cad6daa7c5a037ab3af46f1d9aab89d4191`,
source digest
`5a06f7219e286023ceed8d5cb597a84ac410f370c58915c2bf49f8a438d04e67`,
and script SHA
`15cbdc7f66850ff0fb9ca36d4b9b32f9a892489cb163869797407968b326f876`.

## 2. Implementation audit and repaired defect

The model, target construction, transition indexing, calibration equations,
selection rule, and split boundary reproduce the registered protocol:

- CAL fitting imports no Stage-2 DEV manifest, Stage-2C raw artifact, or FINAL
  path.
- the 3,262 CAL targets contain 140 events and are repeated at exactly
  K0/K1/K2/K4/K8, for 16,310 categorical examples;
- E-T/E-TZ parameterize `T=exp(t)>0`;
- E-Z/E-TZ add a scalar only to exact center bin 127;
- fitting is deterministic full-batch float64 LBFGS from identity;
- selection uses only ordinary unweighted CAL NLL;
- model state before and after CAL and DEV is exactly
  `93509072...e49`;
- continuation and latent rows are copied unchanged only after the state and
  identity controls pass.

One defect was caught by the required E-I assertion. The new collector copied
float32 logits to CPU and decoded there, while canonical Stage-2C decoded the
same logits on CUDA. The checkpoint plus the original evaluator reproduced
the committed Stage-2C rows bit-for-bit. CPU decoding differed on 1,628–1,681
of 1,994 rows per depth, but only by CPU/CUDA reduction rounding:

| Depth | maximum absolute difference | mean absolute difference |
|---:|---:|---:|
| K0 | `3.58e-7` | `1.88e-9` |
| K1 | `2.98e-7` | `1.65e-9` |
| K2 | `1.79e-7` | `1.51e-9` |
| K4 | `1.79e-7` | `1.65e-9` |
| K8 | `1.79e-7` | `1.73e-9` |

The repair moves collected logits back to the registered inference device for
DEV decode/NLL. A CUDA regression now proves this path is bit-identical to
direct canonical CUDA decoding. After repair:

- E-I natural reward predictions equal committed C-LR exactly;
- E-I fork rows equal committed C-LR exactly;
- C-LR state before/after evaluation is bit-identical.

This was an over-strict control doing useful work: the numerical discrepancy
was scientifically immaterial, but accepting it silently would have made the
claimed exact reconstruction false.

## 3. CAL fit

| Arm | Temperature | Zero-bin bias | CAL NLL |
|---|---:|---:|---:|
| E-I | `1.000000` | `0` | `.195777025` |
| E-T | `1.073833` | `0` | `.194617957` |
| E-Z | `1.000000` | `.117344` | `.195562974` |
| **E-TZ** | **`1.499867`** | **`1.255273`** | **`.187375935`** |

E-TZ was therefore selected without DEV access. Its NLL improvement over
identity is `.00840109`, comfortably above the registered `1e-6` minimum.

An independent CAL decomposition explains, but does not alter, this choice.
Only 700 of 16,310 repeated examples (`4.2918%`) are reward events:

| Arm | zero-row NLL | event-row NLL | decoded `|r|`, zero | decoded `|r|`, event |
|---|---:|---:|---:|---:|
| E-I | `.04307` | `3.60118` | `.00795` | `.08003` |
| E-T | `.05144` | `3.38745` | `.00881` | `.08234` |
| E-Z | `.03886` | `3.68995` | `.00731` | `.07704` |
| E-TZ | `.03768` | `3.52550` | `.00523` | `.05452` |

Unweighted categorical NLL prefers a compromise dominated by zero rows.
Moreover, better categorical NLL does not imply better decoded conditional
mean magnitude under `symexp(E[symlog])`. This mismatch was a hypothesis of
Stage-2E; it is now measured rather than assumed.

## 4. DEV result

### Natural reward readouts

| Depth / metric | A | C-LR / E-I | selected E-TZ |
|---|---:|---:|---:|
| K0 event AUROC | `.90015` | `.90574` | `.89915` |
| K0 Pearson | `.73281` | `.71082` | `.68747` |
| K0 zero MAE | `.00421` | `.00357` | `.00261` |
| K1 event AUROC | `.80550` | `.82237` | `.81627` |
| K1 Pearson | `.57167` | `.57811` | `.56094` |
| K1 zero MAE | `.00276` | `.00585` | `.00412` |
| K8 event AUROC | `.67114` | `.73594` | `.72597` |
| K8 average precision | `.11889` | `.12368` | `.11940` |
| K8 Pearson | `.16146` | `.18915` | `.20549` |
| K8 event MAE | `.45912` | `.43483` | `.44817` |
| K8 decoded event `|r|` | `.00570` | `.03369` | `.01963` |

The selected arm trades discrimination and magnitude for zero calibration.
Against C-LR at K8:

- Pearson improves `+.01634 [.00456, .02642]`;
- event MAE worsens `+.01334 [.00554, .02294]`;
- decoded event magnitude falls
  `-.01406 [-.02391, -.00600]`;
- event AUROC changes
  `-.00997 [-.02803, .01106]`.

### Fork readouts

| Metric | A | C-LR / E-I | selected E-TZ |
|---|---:|---:|---:|
| chosen-minus-random | `.27698` | `.27540` | `.32302` |
| regret | `.12857` | `.13016` | `.08254` |
| zero-suffix absolute return | `.00947` | `.06404` | `.04177` |
| zero-suffix abs. gated return | `.00944` | `.06319` | `.04126` |

The apparent ranking gain is not broad. E-TZ changes the selected outcome at
only one of 21 informative anchors, improving environment seed 147. The
paired advantage against C-LR is
`+.04762 [.00000, .18750]`; it does not establish a general gain.

The primary false-return contrast against A is
`+.03230 [.02432, .04041]`. Both its point and entire interval exceed the
registered `+.02` budget.

## 5. Gate audit

Seven conditions pass and four fail:

1. **FAIL:** zero-suffix absolute return stays within A + `.02`;
2. **FAIL:** K8 AUROC/AP/Pearson/event-MAE point estimates all preserve C-LR;
3. **FAIL:** K8 paired metrics show no significant harm—event MAE
   significantly worsens;
4. **FAIL:** K1 zero-MAE does not significantly worsen versus A.

The analysis correctly:

- gates only the CAL-selected E-TZ arm;
- keeps all other arms transparent but non-selectable;
- resamples reward/continuation/latent rows by episode and fork rows by
  environment seed;
- uses paired indices across arms;
- fails closed on the conjunction of registered conditions.

The rejection does not depend on a borderline confidence interval. The
false-return point and lower CI exceed the budget, while K8 event-MAE harm has
a CI entirely above zero.

## 6. What this result licenses

### Supported

- C-LR contains useful action-conditioned reward geometry: it retains the
  Stage-2C K8 discrimination and fork ranking evidence.
- Its false-return problem is not merely a single global logit scale or zero
  prior that can be repaired while preserving all required readouts.
- Ordinary unweighted categorical NLL is not an adequate selector for the
  downstream decoded-mean trade-off in this sparse-reward regime.
- The local DRAMA/unofficial-D4-JAX-style categorical parameterization remains
  a live causal suspect, but is not proven to be the cause.

### Not supported

- “Calibration repaired C-LR.”
- “The reward distribution is definitely the sole issue.”
- “A post-hoc DreamerV3 decoder should now be tried.” Its targets,
  interpolation space, and decode operator differ; swapping only the decoder
  would be source-incoherent.
- “Choose E-T because its transparent DEV row looks better.” That would be
  post-selection on spent DEV and is forbidden.
- “The one changed fork anchor proves planning improved.”

## 7. Decision and next controlled question

Stage-2E's registered route is adopted:

- **global temperature/zero-bin calibration: NO-GO**;
- **more threshold or arm sweeps on DEV: NO-GO**;
- **current C-LR deployment/planner: NO-GO**;
- **FINAL access: NO-GO**.

The next smallest source-backed causal question is a matched reward-operator
training control:

1. local uniform-symlog target plus `symexp(E[symlog])`, versus
2. DreamerV3/CDP original-reward-space interpolation over symexp-spaced
   support plus `E[reward]`.

That comparison must change loss and decode together and retrain matched
models from the same initialization/schedule. A post-hoc decoder swap is not
admissible. It should remain GRU-505, DEV-only, with Mamba/FINAL/planner
blocked, and must retain the existing C-LR local checkpoint as the exact
reference or reproduce it bit-for-bit before attribution.

This control can tell us whether the categorical operator contributes to the
calibration/discrimination trade-off. It cannot by itself establish true
Dreamer-4 alignment: Dreamer 4 additionally uses task-conditioned agent
tokens, MTP length eight, and distance-specific outputs.

