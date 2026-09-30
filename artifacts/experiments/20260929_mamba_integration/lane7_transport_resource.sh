#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_TRANSPORT_EQ_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
job mamba_transport_resource 3500 "$PY" artifacts/experiments/20260929_mamba_integration/transport_resource.py
echo "$(date '+%F %T') LANE_MAMBA_TRANSPORT_RESOURCE_DONE" >> "$LOGDIR/lanes.log"
