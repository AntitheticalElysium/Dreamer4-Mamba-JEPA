#!/usr/bin/env bash
# Remainder of the ceiling experiment after the overnight interruption. Every stage is
# skip-if-done, so relaunching this verbatim is safe and costs nothing for work already
# finished.
#
#   I4_scale_1825  the missing scaling rung, between 912 and the full 3651
#   I5_depth0      production's own head shape on the pure all-17 objective, which
#                  splits the oracle's +0.107 over production between cross-token
#                  action interaction and the objective
#   I6_depth4      regenerates the depth-4 pooled JSON that the depth-1 run overwrote
#                  before the _d{depth} filename suffix existed
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }

[ -f decoder_ceiling_abt0_pooled_fit1825.json ] || \
  stage I4_scale_1825 "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000 --fit-roots 1825
[ -f decoder_ceiling_abt0_pooled_heldout_d0.json ] || \
  stage I5_depth0 "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000 --depth 0
[ -f decoder_ceiling_abt0_pooled_heldout.json ] || \
  stage I6_depth4 "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000
echo "ceiling remainder finished $(date -Is)" >> triage.log
