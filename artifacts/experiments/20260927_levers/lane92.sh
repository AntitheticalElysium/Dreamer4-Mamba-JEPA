#!/usr/bin/env bash
# Frozen candidate-family diagnostic; no new model training.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e19_router_oracle_fmamba 1600 $PY -B "$L/e19_router_oracle.py"
echo "$(date '+%F %T') LANE92_DONE" >> "$LOGDIR/lanes.log"
