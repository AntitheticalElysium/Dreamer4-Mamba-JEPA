source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: check_rootaware (readings in its docstring) on the four budget-pair worlds, then its analysis.
W=artifacts/eda/levers_tworlds_v1
job check_rootaware 2400 $PY $L/check_rootaware.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s7_u36000.pt \
  $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_teacher_s8_u36000.pt
job check_rootaware_analyse 0 $PY $L/check_rootaware.py --analyse corrt_raw_teacher_s7_u18000:corrt_raw_teacher_s7_u36000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_teacher_s8_u36000
echo "$(date '+%F %T') LANE48_DONE" >> $LOGDIR/lanes.log
