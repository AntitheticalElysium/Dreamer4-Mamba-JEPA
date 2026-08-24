#!/usr/bin/env bash
# Production's own head shape, trained on the pure all-17 objective from cached
# features. Depth 1 already reaches 0.418 against production's 0.311, but it has both
# cross-token attention AND a purer objective. This arm has the objective and not the
# attention, so it attributes the +0.107 to one or the other.
set -u
cd "$(dirname "$0")"
PY=/home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA/.venv/bin/python
export JAX_PLATFORMS=cpu
mkdir -p logs
until grep -q "ceiling scaling finished" triage.log; do sleep 30; done
echo "=== I5_depth0_pooled  $(date -Is) ===" | tee -a triage.log
if "$PY" probe_decoder_ceiling.py --tap pooled --steps 36000 --depth 0 \
     >>logs/I5_depth0_pooled.log 2>&1; then echo "    ok" | tee -a triage.log
else echo "    FAIL" | tee -a triage.log; tail -20 logs/I5_depth0_pooled.log | tee -a triage.log; fi
echo "depth0 control finished $(date -Is)" >> triage.log
