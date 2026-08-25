#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
until grep -q "production 1b finished" triage.log; do sleep 60; done
[ -f production_1b/world.pt ] || { echo "no world.pt; training failed" | tee -a triage.log; exit 1; }
echo "=== P2_production_eval  $(date -Is) ===" | tee -a triage.log
if "$PY" evaluate_production_1b.py >>logs/P2_production_eval.log 2>&1; then
  echo "    ok" | tee -a triage.log
else
  echo "    FAIL" | tee -a triage.log; tail -30 logs/P2_production_eval.log | tee -a triage.log
fi
echo "production eval finished $(date -Is)" >> triage.log
