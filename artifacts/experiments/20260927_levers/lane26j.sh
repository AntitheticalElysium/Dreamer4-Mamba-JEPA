source artifacts/experiments/20260927_levers/lib.sh
# E11c re-run with the 'slept' provenance (corrt worlds)
W=artifacts/eda/levers_tworlds_v1
job missedscroll_sleep 2000 $PY $L/missedscroll.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt $W/corrt_raw_selffed_s7.pt
echo "$(date '+%F %T') LANE26J_DONE" >> $LOGDIR/lanes.log
