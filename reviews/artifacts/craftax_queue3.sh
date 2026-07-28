#!/usr/bin/env bash
# Component decomposition: WHICH part of JEPA training removes the task state?
#   H. jepa_weight=0 -> encoder trained ONLY by reward/continuation heads.
#      If the decay vanishes, the self-prediction term is the cause.
#      If it persists, the heads/dynamics are.
#   I. T-BASE time-course -> does reconstruction shed the same information,
#      more slowly? Tests the stage-E/F "leading reading" directly.
# Both are DIAGNOSTICS on existing code paths; neither proposes a change.
set -u
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
LOGDIR=outputs/d4_mamba_jepa/queue
PY=.venv/bin/python

stage () {
  local name="$1"; shift
  echo "=== [$(date -u +%H:%M:%S)] START $name ===" | tee -a "$LOGDIR/queue.log"
  if "$@" > "$LOGDIR/$name.log" 2>&1; then
    echo "=== [$(date -u +%H:%M:%S)] OK    $name ===" | tee -a "$LOGDIR/queue.log"
  else
    echo "=== [$(date -u +%H:%M:%S)] FAIL  $name ===" | tee -a "$LOGDIR/queue.log"
  fi
}

stage h_timecourse_jepaweight0 \
  $PY -u reviews/artifacts/craftax_timecourse.py --jepa-weight 0.0 \
    --output reviews/artifacts/craftax_timecourse_jepaweight0.json

echo "=== [$(date -u +%H:%M:%S)] QUEUE3 COMPLETE ===" | tee -a "$LOGDIR/queue.log"
