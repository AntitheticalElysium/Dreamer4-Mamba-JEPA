#!/bin/sh
# T3 queue (PLAN.md): instrument R0 on f3, a 3-update smoke, then the three arms and their evaluations, one GPU job at
# a time. Stops at the first failure (set -e). Run from the repository root.
set -e
export MAMBA_DETERMINISTIC=1
PY=.venv/bin/python
M=artifacts.experiments.20261106_decision_levels.finetune
D=artifacts/experiments/20261106_decision_levels
$PY -u -m $M evaluate f3 > $D/eval_f3.log 2>&1
$PY -u -m $M train H --smoke 3 > $D/train_smoke.log 2>&1
for ARM in C H F; do
  $PY -u -m $M train $ARM > $D/train_$ARM.log 2>&1
  $PY -u -m $M evaluate $ARM > $D/eval_$ARM.log 2>&1
done
echo done > $D/queue.done
