#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_LONG_SMOKE_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
[ -f artifacts/experiments/20260929_mamba_integration/cache_audit.json ] || { echo 'cache audit missing'; exit 2; }
job mamba_transport_equivalence 3500 "$PY" artifacts/experiments/20260929_mamba_integration/transport_equivalence.py
echo "$(date '+%F %T') LANE_MAMBA_TRANSPORT_EQ_DONE" >> "$LOGDIR/lanes.log"
