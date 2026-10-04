source artifacts/experiments/20260927_levers/lib.sh
# E11i (2026-10-01): does the copy head lose confidence on its own outputs? (copyconf.py; reading declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
job copyconf 2000 $PY $L/copyconf.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/corrt_raw_suffix_s7_u18000.pt $W/corrt_raw_suffix_s8_u18000.pt $W/corrt_raw_selffed_s7.pt
echo "$(date '+%F %T') LANE26F_DONE" >> $LOGDIR/lanes.log
