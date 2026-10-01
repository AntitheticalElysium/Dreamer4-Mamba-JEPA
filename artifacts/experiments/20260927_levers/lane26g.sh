source artifacts/experiments/20260927_levers/lib.sh
# E11j (2026-10-01): known-content drift under hard (argmax) decoding (scrolldrift.py --hard; reading in its docstring)
W=artifacts/eda/levers_tworlds_v1
job scrolldrift_hard 2000 $PY $L/scrolldrift.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt --hard
echo "$(date '+%F %T') LANE26G_DONE" >> $LOGDIR/lanes.log
