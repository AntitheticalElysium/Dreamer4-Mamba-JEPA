source artifacts/experiments/20260927_levers/lib.sh
# E11h (2026-10-01): where known content drift accumulates (scrolldrift.py; reading declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
job scrolldrift 2000 $PY $L/scrolldrift.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE26E_DONE" >> $LOGDIR/lanes.log
