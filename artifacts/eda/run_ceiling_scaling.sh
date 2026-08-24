#!/usr/bin/env bash
# Is 0.469 a feature ceiling or a data ceiling? The winning oracle overfits 3,651 fit
# roots by 8.9x (train 0.00325, test 0.02884), which is the same memorisation
# signature the earlier paired-coverage experiment hit. If R_delta climbs with root
# count the ceiling is data-limited and says nothing about the features; if it is flat
# the features really are the limit.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
stage() { local n="$1"; shift
  echo "=== $n  $(date -Is) ===" | tee -a triage.log; local s=$SECONDS
  if "$@" >>"logs/$n.log" 2>&1; then echo "    ok    $((SECONDS-s))s" | tee -a triage.log
  else echo "    FAIL  $((SECONDS-s))s" | tee -a triage.log; tail -25 "logs/$n.log" | tee -a triage.log; fi; }
until grep -q "depth gradient finished" triage.log; do sleep 30; done
for r in 456 912 1825; do
  stage "I4_scale_${r}" "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000 --fit-roots "$r"
done
echo "ceiling scaling finished $(date -Is)" >> triage.log
