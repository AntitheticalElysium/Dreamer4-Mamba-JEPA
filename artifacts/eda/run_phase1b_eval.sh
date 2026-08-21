#!/usr/bin/env bash
# Waits on the artifacts themselves, not on a log marker -- a stale marker from an
# earlier failed run let this exit immediately once already.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
while [ ! -f phase1b_s1_n32/training_report.json ] || \
      [ ! -f phase1b_s1_n64/training_report.json ]; do sleep 60; done
for m in 5000 10000 20000; do
  a="phase1b_s1_n32/world_$(printf %06d $m).pt"; b="phase1b_s1_n64/world_$(printf %06d $m).pt"
  if [ -f "$a" ] && [ -f "$b" ]; then
    echo "=== 97_eval_phase1b_$m  $(date -Is) ===" | tee -a triage.log
    if $PY evaluate_phase1b_fork.py --milestone $m --suffix s1 >>logs/97_eval_phase1b.log 2>&1
    then echo "    ok" | tee -a triage.log
    else echo "    FAIL" | tee -a triage.log; tail -15 logs/97_eval_phase1b.log | tee -a triage.log; fi
  fi
done
echo "PHASE1B EVAL COMPLETE $(date -Is)" >> triage.log
