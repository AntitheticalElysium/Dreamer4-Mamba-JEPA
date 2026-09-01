#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
R=../..; PY=$R/.venv/bin/python
export PYTHONPATH=$R PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say() { echo "[$(date +%H:%M:%S)] == $*"; }
for arm in attention mamba; do
  if [ -f "v2_phase3_${arm}/phase3_final.pt" ]; then say "phase 3 $arm already done"; continue; fi
  say "phase 3 $arm"
  $PY run_v2_phase3.py --arm "$arm" || { say "PHASE3 $arm FAILED"; exit 1; }
done
say "both actors trained"
