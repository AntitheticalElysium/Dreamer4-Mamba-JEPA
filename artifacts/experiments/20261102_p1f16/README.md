# P1-F1.6: mandatory local joint fusion

Approved2026-11-02. From scratch, world seed1,25.6M frames. Uniform fixations,
faithful decoder, factorised denoiser and cascade unchanged from F1.5. New joint
88->128->64 local MLP replaces raw fine-to-belief canvas; coarse path stays.
No P2, actor or learned-fixation job follows.

Completed training and the full queued evaluation. Reviewed2026-11-03:
[detailed results](../20261103_p1f16_review/RESULTS.md). Several consequence and
reconstruction gains, worse risky damage/death, no demonstrated native useful-look
repair, including the bounded matched source-local follow-up. No global upgrade
or P2/S2 pass is claimed. The source/launch protocol below remains historical.

Protocol: [PLAN.md](PLAN.md). Sources and limits: [RESEARCH.md](RESEARCH.md).
Pre-launch evidence: [CHECKS.md](CHECKS.md), [checks.json](checks.json).

One-shot worker: `python -m artifacts.experiments.20261102_p1f16.run`.
Unit: `saccade-p1f16.service`; no timer, restart loop or automatic scientific retry.
`STATUS.json` records training/evaluation/completion or failure. Source pins and
the source archive preserve the exact implementation. Source drift blocks
evaluation rather than silently changing its estimators.

- Training log: `training.log`; metric records/checkpoints: `run/log.jsonl`.
- Retain `run/ckpt_01.pt` through `ckpt_10.pt`, plus `latest.pt`.
- After successful full-budget training, select by the existing monitor rule and
  write `run/s1/selection.json`. No outcome-based checkpoint choice.
- Automatic same-protocol evaluation: `evaluation.log`, `run/s1/`, `readings/`.
  Full S1; paired actor interface/random and guided gaze; M0.5 and geometry;
  pixel decoder/noise/counter panels; cascade sampler factorial; query consistency/
  conditioning; CPU glimpse uptake; locus; paired Codex summary versus F1.5.

The scripts are reused unchanged behind path/label wrappers. S1 fits this world's
own readers. Euler3/churn1 remains primary. F1.5 is the paired control; its saved
Codex contexts/results remain untouched. One seed limits generalization.
