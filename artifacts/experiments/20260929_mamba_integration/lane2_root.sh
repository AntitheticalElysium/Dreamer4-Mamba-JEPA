#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_VERIFY_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
job mamba_integration_root_patch 2800 "$PY" artifacts/experiments/20260929_mamba_integration/root_patch.py
echo "$(date '+%F %T') LANE_MAMBA_ROOT_DONE" >> "$LOGDIR/lanes.log"
