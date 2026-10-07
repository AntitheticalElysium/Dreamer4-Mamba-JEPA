#!/usr/bin/env bash
# Frozen-world reader diagnosis. Re-running resumes committed optimizer states.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONUNBUFFERED=1
source artifacts/experiments/20260927_levers/lib.sh
job e17_h16_target_intervention 950 "$PY" -B "$L/e17_h16_target_intervention.py"
