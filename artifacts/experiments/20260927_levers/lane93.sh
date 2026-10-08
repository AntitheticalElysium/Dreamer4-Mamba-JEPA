#!/usr/bin/env bash
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e19_incoming_exposure_cpu 0 $PY -B "$L/e19_incoming_exposure.py"
echo "$(date '+%F %T') LANE93_DONE" >> "$LOGDIR/lanes.log"
