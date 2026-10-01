source artifacts/experiments/20260927_levers/lib.sh
# E11o (2026-10-01): onset attribution of the decision-flipping target tile's corruption (onset.py, exploratory)
W=artifacts/eda/levers_tworlds_v1
job onset_joint 2000 $PY $L/onset.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE26L_DONE" >> $LOGDIR/lanes.log
