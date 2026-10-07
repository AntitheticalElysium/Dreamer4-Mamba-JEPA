#!/usr/bin/env bash
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
source artifacts/experiments/20260927_levers/lib.sh
"$PY" -B "$L/e17_health_verify.py" >> "$LOGDIR/e17_health_time_true.log" 2>&1
job e17_health_time_true 1850 "$PY" -B "$L/e17_health_time_true.py"
