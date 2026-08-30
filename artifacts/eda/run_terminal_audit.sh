#!/usr/bin/env bash
# Full replay census of TRAIN terminal tails: which contain an action-conditioned
# decision and which are unavoidable. Resumable per shard.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
mkdir -p logs
echo "=== S1_terminal_audit  $(date -Is) ===" | tee -a triage.log
s=$SECONDS
if "$PY" audit_terminal_tails.py >>logs/S1_terminal_audit.log 2>&1; then
  echo "    ok    $((SECONDS-s))s" | tee -a triage.log
else
  echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 logs/S1_terminal_audit.log | tee -a triage.log
fi
echo "terminal audit finished $(date -Is)" >> triage.log
