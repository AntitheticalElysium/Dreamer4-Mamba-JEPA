#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
job mamba_integration_accept 3600 "$PY" artifacts/experiments/20260929_mamba_integration/verify_accept.py
echo "$(date '+%F %T') LANE_MAMBA_VERIFY_DONE" >> "$LOGDIR/lanes.log"
