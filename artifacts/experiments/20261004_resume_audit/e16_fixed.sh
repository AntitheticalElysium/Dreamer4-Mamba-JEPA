#!/usr/bin/env bash
# Finish the amended E16 evaluation on the completed seed-7 world/prior. No new world training.
source artifacts/experiments/20260927_levers/lib.sh
job oct04_e16_fixed_s7 2400 $PY -B $L/check_e16.py \
  artifacts/eda/levers_tworlds_v1/dworld_a_s7_scratch_u36000_prior_u20000.pt --samples 8
echo "$(date '+%F %T') OCT04_E16_FIXED_DONE" >> "$LOGDIR/lanes.log"
