#!/usr/bin/env bash
source artifacts/experiments/20260927_levers/lib.sh
# Hold this costly cache reconstruction until booked trainers/evaluations release the device.
while systemctl --user is-active --quiet d4mj-oct05-e17.service || systemctl --user is-active --quiet d4mj-oct05-e18.service; do
  sleep 30
done
job oct05_h16_m6_s8 2800 $PY -B $L/check_h16_traj.py \
  artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s8_fmamba_u36000.pt \
  --result-dir artifacts/experiments/20261005_recovery/h16_results
echo "$(date '+%F %T') OCT05_H16_SEED8_DONE" >> "$LOGDIR/lanes.log"
