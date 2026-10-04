#!/usr/bin/env bash
set -euo pipefail
source artifacts/experiments/20260927_levers/lib.sh
until grep -q 'LANE_MAMBA_TRANSPORT_RESOURCE_DONE' "$LOGDIR/lanes.log" 2>/dev/null; do sleep 20; done
export JAX_PLATFORMS=cpu
job mamba_learned_gate_rollout 3500 "$PY" artifacts/experiments/20260929_mamba_integration/learned_gate_rollout.py
echo "$(date '+%F %T') LANE_MAMBA_LEARNED_GATE_DONE" >> "$LOGDIR/lanes.log"
