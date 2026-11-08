#!/bin/sh
# Follow-ups 2-4 (PLAN.md) after follow-up 1's unit, one GPU job at a time; stops at the first failure.
set -e
while systemctl --user is-active -q f4-followup1; do sleep 30; done
D=artifacts/experiments/20261107_f4_followups
M=artifacts.experiments.20261107_f4_followups
.venv/bin/python -u -m $M.routes_f4 routes > $D/routes.log 2>&1
.venv/bin/python -u -m $M.routes_f4 movement > $D/movement.log 2>&1
.venv/bin/python -u -m $M.flex_bench correct > $D/flex_correct.log 2>&1
.venv/bin/python -u -m $M.flex_bench det > $D/flex_det.log 2>&1
.venv/bin/python -u -m $M.flex_bench cost > $D/flex_cost.log 2>&1
echo done > $D/queue.done
