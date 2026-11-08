# Learned fixation: bounded chooser development

Protocol: [PLAN.md](PLAN.md). No production code or world-training recipe changes.

**Complete.** Full interpretation, exact-replay checks and grounded/paired analyses:
[DISSECTION.md](DISSECTION.md). F1.6 performance remains outside this campaign.

* `POLICY_RESULTS.md`, `policy_results.json`: completed cached CPU policy ablation.
* `STATUS.json`: one-shot GPU worker waiting/running/complete/failed.
* `gpu_smoke.json`, `cpu_checks.log`: correctness evidence, before the full fits.
* `e1.log`, `e2.log`, `e3.log`: fixed-budget head fits, independently authorized.
* `e*_*.npz/json`: per-context forecasts/scores/errors and paired summaries.
* `RESULTS.md`: tables generated when the three head experiments finish.

The worker waits for the *whole* P1-F1.6 evaluation (including CPU uptake), checks GPU
occupancy, then runs a GPU smoke and E1 -> E2 -> E3. It stops on an infrastructure
failure or45min worker cap. It never controls another service/process. No new world
training, policy selection, scientific extension or recurring monitor follows.

Commands from repo root:

```
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q artifacts/experiments/20261103_chooser_development/test_instruments.py
CUDA_VISIBLE_DEVICES= .venv/bin/python -m artifacts.experiments.20261103_chooser_development.policy
.venv/bin/python -m artifacts.experiments.20261103_chooser_development.worker --dry-run
```

The CPU ablation is allocation on fixed recorded histories, not policy-specific
histories or evidence of learning benefit. Its negative/positive readings do not
gate the other experiments. World1/F1.5 native source and all original outputs
remain unchanged. E3's full-belief input is diagnostic, not adopted for collection.
