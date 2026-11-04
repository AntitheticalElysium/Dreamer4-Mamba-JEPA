# F1.5/F1.6 frozen retention audit

User-authorized2026-11-03. Protocol: [PLAN.md](PLAN.md).

Both-world predictor and conditional semantic-reader smokes passed, including
source/time/gaze parity, future-target exclusion, actual nonlinear/full mixed
inputs, zero unavailable fine evidence, frozen world weights and native genuine-
look replay (.00595 maximum smoke RMS). Reporting checks recover known25% gain,
−5pp paired difference and exact zero for identical worlds. Initial allocator
warning is kept in smoke_initial.log; duplicate immutable GPU history was removed
before the final smoke. No scientific settings or world parameters changed.

Launched as `saccade-fusion-retention-audit.service`, invocation
`53c699c0daf4487c8993c305ace1b6f6`. One-shot44 diagnostic fits:22 future-fine
predictors plus22 conditional semantic readers, including two seeds and both
worlds. Local/raw identity arms in F1.5 alias exactly. Fixed2400 updates per fit;
no success-based extension or checkpoint selection.92 sources/inputs pinned.

- Live phase: RUN_STATUS.json (worker); detailed phase: STATUS.json (audit).
- Fits and replay: run.log, per-world fit.json/checks.json/identity.json.
- Automatic numerical analysis: report.log and RESULTS.json.
- World parameters stay frozen. No P1-F2/P2/actor follows this worker.

The auxiliary future-fine task does not receive future coarse evidence. Strong
diagnostic access does not automatically establish unused value under native
cascade conditioning. Native use, representation access and sufficient learning
pressure remain distinct questions. Stop for discussion after these readings.

## Result (written 2026-11-04 by Claude from `RESULTS.json`; Codex recorded no narrative for this audit)

Future-fine MSE reduction against the matched coarse/history baseline, confirmation cohort (1,824 frames, 102
episodes), primary stratum (night, entity in crop, no scroll), both reader seeds:

| Supplement | F1.5 crop / entity cell | F1.6 crop / entity cell |
|---|---:|---:|
| raw fine tokens | +38.9 / +42.2% ; +26.4 / +25.9% | +45.8 / +45.7% ; +31.1 / +30.5% |
| local joint (F1.6's fused token; F1.5 aliases raw) | = raw | +22.4 / +22.1% ; +11.4 / +11.7% |
| full mixed 256-D input | -0.2 / +1.0% ; +0.8 / -0.9% | +0.2 / +0.2% ; -0.8 / +0.6% |
| post-action emission | -0.3 / -4.9% ; +0.6 / -3.2% | +2.2 / +2.2% ; +1.1 / +1.2% |
| projected memory | -0.8 / +1.2% ; -0.8 / +0.1% | +2.0 / +0.7% ; +1.0 / +1.2% |

Reading: the fused token keeps about half of the raw fine tokens' predictive value; the full mixed 256-D input
carries essentially none of it in either world (the access loss is at the global frame projection, not removed by
the local fusion); F1.6's emissions / memory carry a small, mostly positive amount (paired F1.6 - F1.5: post-action
+2.5 [+0.7, +4.1] / +7.1 [+5.6, +8.7] pp, memory +2.8 [+1.3, +4.3] / -0.5 [-2.2, +1.2] pp, crop target). F1.6's fine
tokens live at a different scale (base error .052 vs .0095), so only within-world percentages compare. Same limits
as the README: the forecast lacks the next coarse frame, and access is not native use. The next audit
(`../20261103_fusion_projection`) tests the global projection directly.
Correction (11-04, Codex's review): these are reductions of these frozen readers' next-fine error, not fractions of
information, and "carries essentially none" holds for these readers only. The projection audit recodes the component
the projection retains and recovers +4.8 / +7.2% whole-crop error (two reader seeds; no entity-cell rescue): an
accessibility deficit with some surviving information, not demonstrated deletion.
