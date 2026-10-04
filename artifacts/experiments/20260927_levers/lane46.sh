source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: check_h16_subst (readings in its docstring) on the four budget-pair worlds, DEV split of deepeval.
W=artifacts/eda/levers_tworlds_v1
[ -f $LOGDIR/check_h16_subst.done ] || { job check_h16_subst 2400 $PY $L/check_h16_subst.py $W/corrt_raw_teacher_s7_u18000.pt \
  $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_teacher_s8_u36000.pt && touch $LOGDIR/check_h16_subst.done; }
echo "$(date '+%F %T') LANE46_DONE" >> $LOGDIR/lanes.log
