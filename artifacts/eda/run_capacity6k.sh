#!/usr/bin/env bash
# Both arms to 6,000 steps under one consistent curriculum, then probe every
# milestone. Resumable: an interrupt costs at most 500 steps.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi; }

MILES="500 1500 3000 4500 6000"
[ -f capacity6k/n32d16_s0/training_report.json ] || \
  stage 80_train_32x16_6k "$PY" train_bottleneck_arm.py --n-latents 32 --d-bottleneck 16 \
    --batch 2 --steps 6000 --milestones $MILES --out capacity6k/n32d16_s0
[ -f capacity6k/n64d16_s0/training_report.json ] || \
  stage 81_train_64x16_6k "$PY" train_bottleneck_arm.py --n-latents 64 --d-bottleneck 16 \
    --batch 2 --steps 6000 --milestones $MILES --out capacity6k/n64d16_s0
echo "capacity 6k arms finished $(date -Is)" >> triage.log
