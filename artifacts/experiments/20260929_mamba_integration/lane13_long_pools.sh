#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
export JAX_PLATFORMS=cpu
A=artifacts/eda/levers_mamba_long_pools_v1
for pool in raw ldad1 ldad10; do
  [ -f "$A/$pool/manifest.json" ] || job mamba_long_pool_${pool} 1800 "$PY" artifacts/experiments/20260929_mamba_integration/long_pool.py --pool "$pool" --batch 8
  [ -f "$A/$pool/manifest.json" ] || { echo "missing completed $pool long pool"; exit 2; }
done
echo "$(date '+%F %T') LANE_MAMBA_LONG_POOLS_DONE" >> "$LOGDIR/lanes.log"
