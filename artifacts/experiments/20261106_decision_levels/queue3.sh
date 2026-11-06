#!/bin/sh
# Third launch: dl-queue2 was stopped after arm C's evaluation to run the crafting measurement (PLAN.md, last section);
# arms H and F then run unchanged from scratch. A crafting failure does not stop H / F.
export MAMBA_DETERMINISTIC=1
PY=.venv/bin/python
D=artifacts/experiments/20261106_decision_levels
$PY -u -m artifacts.experiments.20261106_decision_levels.crafting --smoke > $D/crafting_smoke.log 2>&1 && \
  $PY -u -m artifacts.experiments.20261106_decision_levels.crafting > $D/crafting.log 2>&1
set -e
M=artifacts.experiments.20261106_decision_levels.finetune
for ARM in H F; do
  $PY -u -m $M train $ARM > $D/train_$ARM.log 2>&1
  $PY -u -m $M evaluate $ARM > $D/eval_$ARM.log 2>&1
done
echo done > $D/queue.done
