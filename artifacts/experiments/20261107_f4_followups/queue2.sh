#!/bin/sh
# Follow-ups 2-4 (PLAN.md) after follow-up 1's unit, one GPU job at a time; stops at the first failure.
set -e
# resumes after queue.sh stopped at flex_bench correct (OOM at 8 rows; now 4)
D=artifacts/experiments/20261107_f4_followups
M=artifacts.experiments.20261107_f4_followups
.venv/bin/python -u -m $M.flex_bench correct > $D/flex_correct.log 2>&1
.venv/bin/python -u -m $M.flex_bench det > $D/flex_det.log 2>&1
.venv/bin/python -u -m $M.flex_bench cost > $D/flex_cost.log 2>&1
echo done > $D/queue.done
