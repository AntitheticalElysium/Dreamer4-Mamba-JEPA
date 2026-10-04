source artifacts/experiments/20260927_levers/lib.sh
# E13f (2026-10-01): consequence learning over training (consfit on 6k / 12k snapshots)
W=artifacts/eda/levers_tworlds_v1
job consfit_snap 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s8_u18000_at6000.pt $W/corrt_raw_teacher_s8_u18000_at12000.pt \
  $W/corrt_raw_teacher_s7_u18000_at6000.pt $W/corrt_raw_teacher_s7_u18000_at12000.pt
echo "$(date '+%F %T') LANE29E_DONE" >> $LOGDIR/lanes.log
