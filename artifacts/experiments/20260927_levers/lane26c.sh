source artifacts/experiments/20260927_levers/lib.sh
# E11d (2026-10-01): does known content drift faster next to the player? (adjdrift.py; reading declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
job adjdrift 2000 $PY $L/adjdrift.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE26C_DONE" >> $LOGDIR/lanes.log
