#!/usr/bin/env bash
# Frozen health-clock lesion; source/input-bound atomic chunks resume after restart.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
source artifacts/experiments/20260927_levers/lib.sh
job e17_health_clock 1300 "$PY" -B "$L/e17_health_clock.py"
