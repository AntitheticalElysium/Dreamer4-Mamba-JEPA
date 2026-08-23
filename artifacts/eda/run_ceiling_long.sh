#!/usr/bin/env bash
# The 12k comparison was confounded: prepool's TRAINING loss was 2.1x pooled's, so it
# was underfit rather than information-limited, and both were still descending. Same
# budget, 3x longer, so the taps are compared near convergence rather than at equal
# step count but unequal fit.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
for tap in prepool pooled; do
  stage "I2_long_${tap}" "$PY" probe_decoder_ceiling.py --tap "$tap" --steps 36000
done
echo "long ceiling finished $(date -Is)" >> triage.log
