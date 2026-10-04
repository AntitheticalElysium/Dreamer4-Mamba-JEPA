source artifacts/experiments/20260927_levers/lib.sh
# E13d (2026-10-01): stage probe of consequences in the world's backbone state (stageprobe.py; readings in its docstring)
W=artifacts/eda/levers_tworlds_v1
job stageprobe 2000 $PY $L/stageprobe.py $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt \
  $W/direct_raw_suffix_s7.pt $W/residual_raw_teacher_s7.pt $W/categorical_raw_teacher_s7_K4096.pt
echo "$(date '+%F %T') LANE29C_DONE" >> $LOGDIR/lanes.log
