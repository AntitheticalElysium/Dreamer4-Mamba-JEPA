#!/bin/sh
# T3 queue, second launch (the first stopped at the smoke: the controller needed the original run's gaze trace;
# fixed by loading its learned state with empty trace / log bookkeeping). R0 already passed (readings/eval_f3.json).
set -e
export MAMBA_DETERMINISTIC=1
PY=.venv/bin/python
M=artifacts.experiments.20261106_decision_levels.finetune
D=artifacts/experiments/20261106_decision_levels
$PY -u -m $M train H --smoke 3 > $D/train_smoke.log 2>&1
for ARM in C H F; do
  $PY -u -m $M train $ARM > $D/train_$ARM.log 2>&1
  $PY -u -m $M evaluate $ARM > $D/eval_$ARM.log 2>&1
done
echo done > $D/queue.done
