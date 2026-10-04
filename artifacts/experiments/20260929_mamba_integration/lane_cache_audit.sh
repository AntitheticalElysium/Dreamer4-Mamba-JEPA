#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
if [ ! -f artifacts/experiments/20260929_mamba_integration/cache_audit.json ]; then
  job mamba_cache_audit 0 /usr/bin/env JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 nice -n 10 "$PY" artifacts/experiments/20260929_mamba_integration/cache_audit.py
fi
echo "$(date '+%F %T') LANE_MAMBA_CACHE_AUDIT_DONE" >> "$LOGDIR/lanes.log"
