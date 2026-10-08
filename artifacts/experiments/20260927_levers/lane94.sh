#!/usr/bin/env bash
# Frozen completed canvas: repeat the existing component contrasts at 36k.
source artifacts/experiments/20260927_levers/lib.sh
set -eo pipefail
R=artifacts/experiments/20261005_recovery/recovery_inputs
job e18_access_s7_at36000 0 $PY -B "$R/e18_access_s7_36k.py"
job e18_carry_s7_at36000 0 $PY -B "$R/e18_carry_s7_36k.py"
job e18_reroute_s7_at36000 0 $PY -B "$R/e18_reroute_s7_36k.py"
echo "$(date '+%F %T') LANE94_DONE" >> "$LOGDIR/lanes.log"
