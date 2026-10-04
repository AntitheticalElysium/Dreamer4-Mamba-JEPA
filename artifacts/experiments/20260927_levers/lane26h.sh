source artifacts/experiments/20260927_levers/lib.sh
# E11k (2026-10-01): known-content drift vs the light change per step (scrolldrift.py light split; reading in its docstring)
W=artifacts/eda/levers_tworlds_v1
job scrolldrift_light 2000 $PY $L/scrolldrift.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE26H_DONE" >> $LOGDIR/lanes.log
