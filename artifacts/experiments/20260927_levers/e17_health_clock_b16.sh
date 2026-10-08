#!/usr/bin/env bash
# Original inference-batch dimensions; preserve the failed batch4 trial.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
source artifacts/experiments/20260927_levers/lib.sh
"$PY" -B "$L/e17_health_verify.py" >> "$LOGDIR/e17_health_clock_b16.log" 2>&1
job e17_health_clock_b16 1850 "$PY" -B "$L/e17_health_clock_b16.py"
