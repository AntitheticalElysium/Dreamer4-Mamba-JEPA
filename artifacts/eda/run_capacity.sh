#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi; }
[ -f capacity/n32d16_s0/training_report.json ] || \
  stage 71_train_32x16 "$PY" train_bottleneck_arm.py --n-latents 32 --d-bottleneck 16 \
    --batch 2 --steps 3000 --out capacity/n32d16_s0
[ -f capacity/n64d16_s0/training_report.json ] || \
  stage 72_train_64x16 "$PY" train_bottleneck_arm.py --n-latents 64 --d-bottleneck 16 \
    --batch 2 --steps 3000 --out capacity/n64d16_s0
echo "capacity arms finished $(date -Is)" >> triage.log
