#!/usr/bin/env bash
# Phase-1A replication at a second tokenizer seed. Both geometries from step 0 to
# 6,000 under the identical 6k curriculum used for seed 1, so the comparison is a
# true replication rather than a differently-scheduled run.
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
[ -f capacity6k/n32d16_s1/training_report.json ] || \
  stage 90_seed2_32x16 "$PY" train_bottleneck_arm.py --n-latents 32 --d-bottleneck 16 \
    --batch 2 --steps 6000 --milestones $MILES --seed-offset 1 --out capacity6k/n32d16_s1
[ -f capacity6k/n64d16_s1/training_report.json ] || \
  stage 91_seed2_64x16 "$PY" train_bottleneck_arm.py --n-latents 64 --d-bottleneck 16 \
    --batch 2 --steps 6000 --milestones $MILES --seed-offset 1 --out capacity6k/n64d16_s1
echo "seed2 arms finished $(date -Is)" >> triage.log
