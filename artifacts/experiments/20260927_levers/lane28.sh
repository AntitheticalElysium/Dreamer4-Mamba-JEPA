source artifacts/experiments/20260927_levers/lib.sh
# E13 (2026-10-01): diagnosis by substitution of the imagined-error chain (subst16.py; readings declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
job subst16 2000 $PY $L/subst16.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt
echo "$(date '+%F %T') LANE28_DONE" >> $LOGDIR/lanes.log
