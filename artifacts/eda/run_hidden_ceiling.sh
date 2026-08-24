#!/usr/bin/env bash
# Post-mixer hidden-decoder ceiling on both mixer seeds at 20k. 20000 decoder steps so
# no arm is undertrained -- that confound is what made the earlier pre-pool tap look
# information-limited when it was merely underfit.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
for suf in abm0 abm1; do
  [ -f "hidden_ceiling_${suf}_020000.json" ] || \
    stage "L1_hidden_${suf}" "$PY" probe_hidden_ceiling.py --suffix "$suf" --steps 20000
done
echo "hidden ceiling finished $(date -Is)" >> triage.log
