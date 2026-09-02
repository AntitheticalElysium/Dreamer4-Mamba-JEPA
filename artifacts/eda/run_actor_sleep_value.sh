#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for arm in attention mamba; do
  for tag in "" _10k; do
    echo "[$(date +%H:%M:%S)] == actor sleep value $arm$tag"
    $PY check_actor_sleep_value.py --arm "$arm" --tag "$tag" --episodes 64 \
      --branch-steps 300 || { echo "[$(date +%H:%M:%S)] == FAILED $arm$tag"; exit 1; }
  done
done
echo "[$(date +%H:%M:%S)] == actor sleep value complete"
