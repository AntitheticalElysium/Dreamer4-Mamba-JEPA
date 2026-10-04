source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_recall --futures on the 6-frame parents (DEV true trajectories; futures_replicates declared in its docstring).
W=artifacts/eda/levers_tworlds_v1
job check_recall_futures 2600 $PY $L/check_recall.py --futures $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_fmamba_u36000_at30000.pt $W/corrt_raw_teacher_s7_u100000.pt
echo "$(date '+%F %T') LANE70_DONE" >> $LOGDIR/lanes.log
