#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
until grep -q "coverage A/B finished" triage.log; do sleep 30; done
echo "=== O1_coverage_eval  $(date -Is) ===" | tee -a triage.log
if "$PY" evaluate_coverage_ab.py >>logs/O1_coverage_eval.log 2>&1; then
  echo "    ok" | tee -a triage.log
else
  echo "    FAIL" | tee -a triage.log; tail -25 logs/O1_coverage_eval.log | tee -a triage.log
fi
echo "coverage eval finished $(date -Is)" >> triage.log
