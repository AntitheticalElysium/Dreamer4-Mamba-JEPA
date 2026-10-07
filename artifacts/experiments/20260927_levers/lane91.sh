#!/usr/bin/env bash
# Independent CPU census/readout work while canvas endpoint forwards run.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
job e19_exposure_census_cpu 0 $PY -B "$L/e19_exposure_census.py"
job e19_health_conditioning_cpu 0 $PY -B "$L/e19_health_conditioning.py"
echo "$(date '+%F %T') LANE91_DONE" >> "$LOGDIR/lanes.log"
