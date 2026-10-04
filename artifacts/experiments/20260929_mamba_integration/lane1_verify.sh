#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_0_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
job mamba_integration_verify 3600 "$PY" artifacts/experiments/20260929_mamba_integration/verify_short.py
echo "$(date '+%F %T') LANE_MAMBA_VERIFY_DONE" >> "$LOGDIR/lanes.log"
