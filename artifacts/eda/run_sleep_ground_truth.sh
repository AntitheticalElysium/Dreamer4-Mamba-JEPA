#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in attention mamba; do
  echo "[$(date +%H:%M:%S)] == sleep ground truth $arm"
  $PY check_sleep_ground_truth.py --arm "$arm" --episodes 64 --states 32 \
    --branch-steps 250 --alternatives 3 \
    || { echo "[$(date +%H:%M:%S)] == FAILED $arm"; exit 1; }
done
echo "[$(date +%H:%M:%S)] == sleep ground truth complete"
