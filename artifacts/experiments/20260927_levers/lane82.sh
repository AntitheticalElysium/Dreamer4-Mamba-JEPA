#!/usr/bin/env bash
source artifacts/experiments/20260927_levers/lib.sh
job e19_router_s7 1300 $PY -B $L/e19_diagnose.py artifacts/eda/levers_tworlds_v1/e19_C_s7_fmamba_from36000.pt artifacts/eda/levers_tworlds_v1/e19_B_s7_fmamba_from36000.pt
echo "$(date '+%F %T') LANE82_DONE" >> "$LOGDIR/lanes.log"
