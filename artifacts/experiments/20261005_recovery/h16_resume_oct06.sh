#!/usr/bin/env bash
# Resume only the already-declared H16 readouts; do not restart training lanes.
set -euo pipefail
cd /home/antithetical/EPITA/PERSO/DynamicHorizons-Mamba-JEPA
source artifacts/experiments/20260927_levers/lib.sh
W=artifacts/eda/levers_tworlds_v1
job e17_h16traj_w15 3000 "$PY" -B "$L/check_h16_traj.py" \
  "$W/corrt_rawlong_teacher_s7_L16b40_from36000.pt" \
  "$W/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt" \
  "$W/corrt_rawlong_teacher_s8_L16b40_from36000.pt" \
  "$W/corrt_rawlong_teacher_s8_fmamba_L16b40_from36000.pt" --window 15
# This short-world seed8 readout was already queued by h16_seed8.sh.
job oct05_h16_m6_s8 2800 "$PY" -B "$L/check_h16_traj.py" \
  "$W/corrt_raw_teacher_s8_fmamba_u36000.pt" \
  --result-dir artifacts/experiments/20261005_recovery/h16_results
echo "$(date '+%F %T') H16_OCT06_DONE" >> "$LOGDIR/lanes.log"
