#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_EVAL6K_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
job mamba_long_smoke 3600 "$PY" artifacts/experiments/20260929_mamba_integration/long_smoke.py
echo "$(date '+%F %T') LANE_MAMBA_LONG_SMOKE_DONE" >> "$LOGDIR/lanes.log"
