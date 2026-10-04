source artifacts/experiments/20260927_levers/lib.sh
# E13e (2026-10-01): consequence present-but-below-threshold or absent? (conscalib.py; readings in its docstring)
W=artifacts/eda/levers_tworlds_v1
job conscalib 2000 $PY $L/conscalib.py $W/categorical_raw_teacher_s7_K4096.pt $W/corrt_raw_teacher_s7_u18000.pt \
  $W/corrt_raw_teacher_s8_u18000.pt $W/direct_raw_suffix_s7.pt
echo "$(date '+%F %T') LANE29D_DONE" >> $LOGDIR/lanes.log
