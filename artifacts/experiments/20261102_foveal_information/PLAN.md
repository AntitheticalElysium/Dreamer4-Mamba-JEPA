# Conditional fine information and belief uptake — bounded native campaign

2026-11-02. User authorized this campaign after UP­TAKE_RESULTS.md: primary priority is
to distinguish redundant fine evidence from failure to acquire/export/use it. No P1-F2
recipe or overnight launch is selected. The corpus is held fixed.

## Primary conditional-readout experiment

Frozen native F1.5 checkpoint 10. Same real sensor stream and full-prefix belief replay
as S1; labels train diagnostic instruments only. Training: first 360 of train-400;
instrument validation: last 40; dev: first 250 episodes (not the final split).

Each query is a map cell inside the actual uniformly chosen real crop. Baseline inputs:
current FULL 81-cell periphery and lighting; the prior emitted belief; the previous
eight physical percepts, their emitted beliefs, actions, arrived outcomes and validity;
current arrived reward/termination, recorded action, fixation and queried-cell identity.
This covers the physical predictor's accessible coarse context, not an exact Bayesian
belief or the entirety of Mamba's internal carry. Action is supplied to every arm so
its hidden-information content cannot uniquely favor the after-action arm.

Six matched readouts differ only in a padded 1,024-dimensional supplement:

* baseline: zeros;
* fine: the queried real fine token (64 dims);
* input: the actual crop-value contribution to the native percept embedding (256 dims),
  isolated algebraically from known periphery, lighting and fixation-mask contributions;
* corrected: native emitted belief after the percept token (256 dims);
* predicted: native emitted belief after the action token (256 dims);
* memory: native four projected memory tokens (1,024 dims).

All receive identical baseline inputs, parent episodes, minibatches, update counts and
learning rates. Two seeds and two fixed reader designs: concatenated 256-256 MLP;
query-gated reader with a separate supplement projection and learned cell-dependent gate.
The second addresses spatial-binding difficulty in a generic concatenated reader.
Each: AdamW .001, weight decay .0001, 2,400 updates, balanced kind batches 256 per kind.
No reader-dependent early stopping or budget extension. Report confusion matrices,
per-kind recall/false positives and macro (class-balanced) cross-entropy on held-out
episodes, day/night separately; raw softmaxes are not calibrated natural event forecasts.
Include local current periphery explicitly so the baseline need not learn a screen lookup.
Use fixed field scaling, not label-dependent input transformations.

Primary comparison: fine versus baseline. Only if BOTH reader families reliably exploit
fine evidence can a negative updated-belief comparison be informative. A persistent gap
between raw-fine and later representations is an ACCESSIBILITY gap, not proof of erased
information. If the families disagree, report instrument dependence.

Re-read the old 595-scene / 49-crop native intervention with these conditional readers.
Select useful evidence by current-frame reader results only, not by future prediction
loss. Pair crops at the same screen row. Reuse saved native losses; no new independent
future is point-matched against the recording. Bootstrap by episode, seeds reported
separately. Test whether acquired evidence changes the conditional readout after update.

## Conditional localization, if needed

If raw fine evidence helps but the exported predicted belief does not, one further
bounded state-readout test is allowed: 1,200 queries per kind from train-360, 200 per
kind from the held-out 40 (fewer if unavailable). Save actual native post-action carry
at these real frames. Fit a small learned query token plus classifier through the
FROZEN Mamba, beside identical-pool raw-fine/exported-belief controls. This is a readout
instrument, not a proposed production token or world intervention. A positive result
can establish that carry information is accessible despite a weak standard emission;
a negative result cannot establish information-theoretic loss. Fixed 2,400 updates,
two seeds, no architecture/loss changes and no automatic further localization branch.

## Chooser research (secondary)

Verify H3's actual prediction/ranking/learning results before claiming head adequacy.
Compare (a) endogenous teacher-score distillation; (b) direct action-conditioned coarse
expectation/error heads; (c) reuse of existing per-cell native training losses; (d)
learned sensing policies / amortized acquisition; (e) controllability/inverse dynamics.
Distinguish the score target, pre-crop inputs, feedback to the world, drift, noise and
runtime. Critically: removing denoiser calls alone may leave sequential perception and
Mamba collection overhead. A cheap P1 selector must have a defensible batched path.

Read primary papers and retain provenance. Do not fit every candidate or pick a launch
recipe from proxies. Bring the most promising candidates and a matched native validation
plan back to the user after the primary experiments. No new corpus work this campaign.

## Pre-run support correction (17:39, before any conditional reader fit)

To interpret an empty-control crop at the SAME queried entity, the export reader must
have training support for queries outside its selected crop. Therefore every map cell
is eligible, with a crop-availability indicator supplied to ALL arms. Half of each
kind-balanced training batch is drawn inside the crop and half outside (sampling with
replacement). Fine supplement is zero outside the crop; comparisons establishing fine
information use INSIDE-crop queries only. Outside-crop evaluations are reported separately.
This changes sampling support, not the native world or its histories. It avoids treating
an out-of-support export-reader failure as a memory failure. Baseline is given the fixed
crop identity and availability in every arm. Histories include the previous eight full
periphery/lighting percepts, predicted beliefs and preceding actions, arrived outcomes,
and valid-history flags. Full native SSM carry remains outside the baseline reader.

Carry-readout mechanics, specified before that instrument runs: its per-kind query pool
is split equally inside/outside the crop, as in the primary readers; controls have the
same pool. Minibatch 50 per kind (250), 2,400 updates, two seeds. The query is an 81-by-256
learned embedding passed for one step through FROZEN native Mamba; a flat 256-256 reader
gets its emitted vector plus the identical baseline. Compare baseline, raw fine and
standard post-action emission on this same smaller pool. Saved carry includes BOTH the
short convolution buffer and SSM state; a positive query result does not establish long
memory or isolate the SSM. Check replay against original native emissions before fitting.
