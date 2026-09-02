#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in attention mamba; do
  echo "[$(date +%H:%M:%S)] == sleep localization $arm"
  $PY check_sleep_localization.py --arm "$arm" --episodes 64 --samples 8 \
    || { echo "[$(date +%H:%M:%S)] == FAILED $arm"; exit 1; }
done
echo "[$(date +%H:%M:%S)] == sleep localization complete"
