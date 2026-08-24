#!/usr/bin/env bash
# Capacity gradient. The oracle beats production Direct on two counts at once: more
# capacity with full cross-token attention, and a pure all-17 objective where
# production carries ordinary + lambda*fork. If depth 1 already matches depth 4, the
# gain is the objective and not the capacity, and "the head lacks capacity" does not
# follow from the oracle result.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
until grep -q "long ceiling finished" triage.log; do sleep 30; done
stage I3_depth1_pooled "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000 --depth 1
echo "depth gradient finished $(date -Is)" >> triage.log
