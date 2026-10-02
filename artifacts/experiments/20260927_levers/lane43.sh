source artifacts/experiments/20260927_levers/lib.sh
# Item 1 (2026-10-02): head-fitting noise vs the budget worlds' gen1 regression next to zombies (check_headseeds.py; readings in
# its docstring, committed before it ran)
W=artifacts/eda/levers_tworlds_v1
job headseeds_fit 2800 $PY $L/check_headseeds.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s7_u36000.pt \
  $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_teacher_s8_u36000.pt
job headseeds_analyse 0 $PY $L/check_headseeds.py --analyse corrt_raw_teacher_s7_u18000:corrt_raw_teacher_s7_u36000 \
  corrt_raw_teacher_s8_u18000:corrt_raw_teacher_s8_u36000
echo "$(date '+%F %T') LANE43_DONE" >> $LOGDIR/lanes.log
