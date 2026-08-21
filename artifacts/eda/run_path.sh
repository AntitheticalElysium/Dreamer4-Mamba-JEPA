#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
echo "=== 50_predictor_path  $(date -Is) ===" | tee -a triage.log
s=$SECONDS
if "$PY" probe_predictor_path.py >>logs/50_predictor_path.log 2>&1; then
  echo "    ok    $((SECONDS-s))s" | tee -a triage.log
else
  echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 logs/50_predictor_path.log | tee -a triage.log
fi
echo "path probe finished $(date -Is)" >> triage.log
