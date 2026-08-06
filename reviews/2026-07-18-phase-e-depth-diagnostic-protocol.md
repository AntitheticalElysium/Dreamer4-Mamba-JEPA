# Phase-E same-target depth diagnostic

Status: **registered before execution on 2026-07-18**. This is a fixed-model
diagnostic, not an architecture selection or a replacement planner gate. It
cannot turn the Phase-E planner NO-GO into a GO.

## Why this diagnostic is necessary

The original Phase-E imagined-horizon sample contains only one reward event at
both h1 and h8, no event at h2, and two events at h4. Moreover, each horizon
scores a different transition. Those rows are sufficient for a conservative
NO-GO under the registered gate, but they cannot identify h1-to-h8 degradation
or distinguish reward-head supervision from imagined-state distribution shift.

## Fixed inputs

- Models: the six committed full-grid/no-bypass checkpoints only
  (GRU and Mamba-2, seeds 505/606/707). No fitting or model selection.
- Data: every transition in `data/heldout_20ep_v1.pt` that admits an
  eight-frame real prefix and eight following transitions. The data and every
  checkpoint are SHA-256 pinned in the output.
- Common target set: the exact same reward transitions and labels at every
  evaluated depth. Episode identity is retained as the uncertainty cluster.

## Suffix-replacement construction

For target transition `j`, start from the same eight real observations
`o[j-14:j-6]`. Advance to the same final transition using the same actions and
the same total of 16 temporal updates, replacing only the final `K` transitions
with autoregressive imagination:

- K=0: eight further real observations (teacher-forced oracle state);
- K=1,2,4,8: `8-K` further real observations, then K imagined transitions.

Thus every depth predicts the same `r[j]`; only the length of the generated
suffix changes. Timing is `(o_t, a_t) -> (o_{t+1}, r_t)` in replay-array
indices, equivalent to the project specification's
`(s_t, a_t) -> (s_{t+1}, r_{t+1})` notation.

## Readouts

Per checkpoint and K:

- reward-event AUROC and average precision, with episode-cluster bootstrap
  intervals;
- two-hot NLL and decoded MAE overall, on events, and on zero-reward steps;
- signed reward Pearson/Spearman;
- decoded positive/negative/zero reward means;
- cosine drift of final post-temporal context relative to the K=0 real state;
- peak allocated/reserved VRAM for the diagnostic.

Raw per-target decoded predictions are retained separately.

## Outcome-independent routing

1. If K=0 is weak, task-state sufficiency or reward supervision is binding;
   event-focused sampling/reweighting remains a first-line candidate.
2. If K=0 is strong and metrics deteriorate as K grows, imagined-state
   distribution shift is established. The first training control must expose
   task heads to rolled-out states and supervise each generated step; simple
   event reweighting alone is not a mechanism-matched fix.
3. If K=0 and imagined K remain strong but ranking fails, candidate-set/ranking
   coverage and reward calibration become the next diagnosis.
4. Compare both backends under the identical rows. Backend deployment remains
   a separate decision; a Mamba-trained model cannot be swapped to a GRU at
   deployment without a separately trained/distilled model and new gates.

## Continuation-depth supplement

Status: **registered after the reward-depth result was inspected, but before
this supplement was executed on 2026-07-18**. This is explicitly a follow-up,
not part of the original reward diagnostic and not a post-hoc route to planner
GO.

The original G-E3 evaluation measures continuation only on teacher-forced real
states. The planner, however, multiplies rewards by continuation probabilities
produced after generated transitions. The same suffix-replacement construction
will therefore be reused with the continuation target for the identical final
transition at every K. The readout is:

- terminal AUROC and average precision, with episode-cluster bootstrap
  intervals;
- continuation Brier score and skill relative to the empirical constant
  predictor;
- binary NLL overall and separately for terminal/non-terminal transitions;
- mean predicted termination probability on terminal/non-terminal targets and
  recall at a 0.5 termination threshold.

All 20 held-out episodes and every eligible transition are included. The six
fixed full-grid checkpoints, target rows, actions, prefix length, depths, and
hash/provenance requirements are unchanged. If real-state continuation is
useful but degrades with K, the next training control must supervise the
continuation head on rolled-out states as well as the reward head.

Pre-execution data-contract amendment: the first attempted run stopped at its
terminal-count assertion before loading any model. Inspection showed that 14
held-out episodes end with a recorded terminal and six were capped at 200
transitions with continuation still equal to one. The fixed assertion is
therefore 14, not 20. All 3,262 eligible transitions and all 20 episode
clusters remain in the analysis; capped episodes are not relabelled.

## Frozen-context task-information probe

Status: **registered after both same-target depth results, before probe fitting
on 2026-07-18**. This probe diagnoses the intervention; it cannot select an
architecture, alter a checkpoint, or grant planner GO.

Question: are the existing task heads merely out of distribution on generated
states, or has the generated context itself lost task-relevant information?

For each fixed checkpoint and each K independently, freeze the complete world
model and extract its final 64-dimensional pooled context under the same
suffix-replacement construction. Fit fresh binary MLP probes with the same
LayerNorm -> 2D hidden -> SiLU shape as the continuation head for:

1. reward event versus zero;
2. positive versus negative reward, restricted to reward events;
3. terminal versus non-terminal.

Probe fitting uses only the 40K training replay. Each task uses every eligible
minority example and a deterministically sampled equal-size comparison class.
The held-out 20-episode cache is evaluation-only. Hyperparameters are fixed
before fitting: full-batch AdamW, learning rate 1e-3, weight decay 1e-4, 300
updates, identical initialization per task and depth. Report train and held-out
AUROC; held-out AUROC/AP use episode-cluster bootstrap intervals. Raw held-out
logits and all selected training row identities are retained and hash-pinned.

Interpretation:

- recovery by a fresh probe at imagined K supports task-head covariate shift
  and licenses a head-supervision control;
- failure of a fresh probe despite fitting the training contexts means the
  frozen generated context has lost transferable task information, so head
  reweighting alone is not a mechanism-matched fix;
- mixed results route reward-event, reward-sign, and continuation supervision
  separately. This is a separability upper bound, not proof that a calibrated
  planner reward head has been learned.

## Outcomes

Completed 2026-07-18. Every filed scalar below was independently recomputed
from the raw-row artifacts after the run; maximum discrepancy was numerical
roundoff. The complete compact CUDA suite is reported in the companion audit.

### Reward on common targets

Mean over the three fixed checkpoints in each family:

| family | metric | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | event AUROC | .877 | .780 | .701 | .662 | .637 |
| Mamba-2 | event AP | .508 | .280 | .135 | .093 | .074 |
| Mamba-2 | signed Pearson | .663 | .514 | .164 | .096 | .048 |
| GRU | event AUROC | .888 | .812 | .754 | .724 | .648 |
| GRU | event AP | .435 | .269 | .151 | .132 | .098 |
| GRU | signed Pearson | .634 | .457 | .207 | .174 | .142 |

There are 3,262 identical targets, 140 reward events, and 20 episode
clusters at every K. Every checkpoint degrades as real suffix states are
replaced by generated states. This validates distribution shift, but refutes
the original 56-window backend narrative: Mamba seed 707 is not inverted at
K1 (AUROC .754, not .055), and GRU does not collapse to chance at K8 (family
mean .648, not .503). Those original readings each depended on a single
positive example at the scored horizon.

At K0, events already contribute 89.6% of Mamba and 84.1% of GRU reward NLL
despite their 4.29% frequency. Mean absolute decoded event reward falls from
.229/.227 (Mamba/GRU) at K0 to .0048/.0193 at K8. The immediate measured
failure is not that zero-reward steps dominate the converged loss; it is that
the deployed reward mapping shrinks generated states toward zero.

Artifacts:

- `reviews/artifacts/phase_e_same_target_depth.json`
  SHA-256 `d3dbd243b814fb3495a9bb77812de1da8480ba4a772beafc810ecafb9cb96fb8`
- `reviews/artifacts/phase_e_same_target_rows.json`
  SHA-256 `2f3bb307f9a05d7acf0018829e08141af35c144d5290b2bff4deffd777091f8c`
- evaluator SHA-256
  `9a4100fbfef256550c7eed88817c4df4d13bf0198dcf4574dc06c1841068f39f`

### Continuation on common targets

Mean over the same fixed checkpoints:

| family | metric | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | terminal AUROC | .954 | .873 | .850 | .694 | .568 |
| Mamba-2 | Brier skill | .330 | .012 | .003 | -.003 | -.004 |
| Mamba-2 | mean P(term) on terminals | .277 | .009 | .004 | .0006 | .00009 |
| GRU | terminal AUROC | .946 | .950 | .944 | .932 | .669 |
| GRU | Brier skill | .129 | .037 | .031 | .008 | -.004 |
| GRU | mean P(term) on terminals | .086 | .027 | .022 | .0067 | .00010 |

There are 14 recorded terminal targets; six held-out episodes end at the
200-step collection cap without being relabelled terminal. The registered
G-E3 result is therefore accurately described as a teacher-forced real-state
ranking pass. It is not an imagined-continuation or calibrated planner pass:
after one generated transition, terminal probabilities are already near zero,
and by K8 both families have negative Brier skill against the empirical
constant predictor.

Artifacts:

- `reviews/artifacts/phase_e_same_target_continuation.json`
  SHA-256 `e0a470d7b5d2b5e4d93893bd0b6bc868f269d7fb7b0470e71f84f008c22ea762`
- `reviews/artifacts/phase_e_same_target_continuation_rows.json`
  SHA-256 `21be653be57c6b05c6d1e36243d8ac35b6e4380982a02f6938c9a76ad624bc18`
- evaluator SHA-256
  `5f4ee121e65d48dbad0a613d83fc2b16d234553823f52e29ae8bfc72285350f0`

### Does task information survive in generated contexts?

Yes, partially. Fresh depth-specific probes trained only on the training
replay recover substantial held-out separation:

| family | probe | K0 | K1 | K2 | K4 | K8 |
|---|---|---:|---:|---:|---:|---:|
| Mamba-2 | reward event AUROC | .909 | .800 | .753 | .711 | .707 |
| Mamba-2 | reward sign AUROC | .810 | .791 | .736 | .719 | .708 |
| Mamba-2 | terminal AUROC | .968 | .878 | .871 | .878 | .830 |
| GRU | reward event AUROC | .915 | .827 | .805 | .790 | .734 |
| GRU | reward sign AUROC | .824 | .787 | .794 | .757 | .745 |
| GRU | terminal AUROC | .958 | .945 | .937 | .932 | .858 |

All six K8 event, sign, and terminal probe confidence intervals exclude .5.
Thus generated contexts are degraded but not task-blind. The strong gap
between these separability upper bounds and the deployed heads supports
task-head covariate shift as a binding mechanism. It licenses a controlled
generated-state task-supervision experiment; it does not prove that one
shared, naturally calibrated reward head will recover, because these probes
are depth-specific and class-balanced.

Artifacts:

- `reviews/artifacts/phase_e_context_probe.json`
  SHA-256 `c676a3fe84e173e25ef9f58a587fb633a48a997e7c0fb07af254af0f0eeaf7ed`
- `reviews/artifacts/phase_e_context_probe_rows.json`
  SHA-256 `323b6717a683f8ce50a572ef45a2ce7af3fa890cf10f3a1f125884ceb0a63943`
- evaluator SHA-256
  `641a7049c97bc00e2d3eb31d05a6d64810cf5aeb4dd9ebd01743bb59966e5161`

### Routing

Planner remains **NO-GO**. The first mechanism-matched lever is per-step
reward and continuation supervision on generated states, initially as
head-only adaptation on the frozen checkpoints. Event-focused sampling or an
auxiliary event/sign objective is a separate factorial control. Raising K
from 2 to 5 before implementing per-step latent and task targets would
preserve the current final-only objective and would not be an SPR-like test.
