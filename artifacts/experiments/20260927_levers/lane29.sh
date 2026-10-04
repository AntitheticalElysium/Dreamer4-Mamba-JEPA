source artifacts/experiments/20260927_levers/lib.sh
# E13b (2026-10-01): are interaction consequences learned on TRAINING transitions? (consfit.py; readings in its docstring)
W=artifacts/eda/levers_tworlds_v1
job consfit 2000 $PY $L/consfit.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE29_DONE" >> $LOGDIR/lanes.log
