source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: check_zombie_cue (readings in its docstring) on the four budget-pair worlds.
W=artifacts/eda/levers_tworlds_v1
[ -f $LOGDIR/check_zombie_cue.done ] || { job check_zombie_cue 1300 $PY $L/check_zombie_cue.py $W/corrt_raw_teacher_s7_u18000.pt \
  $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_teacher_s8_u36000.pt && touch $LOGDIR/check_zombie_cue.done; }
echo "$(date '+%F %T') LANE47_DONE" >> $LOGDIR/lanes.log
