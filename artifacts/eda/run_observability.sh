#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() {
  local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log
  local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -20 "logs/$n.log" | tee -a triage.log; fi
}
[ -f state_features/features.pt ] && [ "$(stat -c%s state_features/features.pt)" -gt 50000000 ] || \
  stage 40_extract_features "$PY" extract_state_features.py --seeds 512
stage 41_probe_observability "$PY" probe_observability.py
echo "observability finished $(date -Is)" >> triage.log
