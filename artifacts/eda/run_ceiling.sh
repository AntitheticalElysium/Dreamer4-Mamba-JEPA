#!/usr/bin/env bash
# Frozen-decoder ceiling: is the counterfactual detail reachable from the backbone?
# Two taps, because production pools before the candidate action exists, so a pooled
# failure alone cannot separate "absent upstream" from "discarded by self.pool".
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

for tap in pooled prepool; do
  stage "I1_ceiling_${tap}" "$PY" probe_decoder_ceiling.py --tap "$tap" --steps 12000
done
echo "decoder ceiling finished $(date -Is)" >> triage.log
