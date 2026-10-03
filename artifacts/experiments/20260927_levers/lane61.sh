source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: Mamba diagnosis, context use (check_context; readings in its docstring), attention vs Mamba 36k teacher, seed 7.
W=artifacts/eda/levers_tworlds_v1
job check_context 2000 $PY $L/check_context.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt
echo "$(date '+%F %T') LANE61_DONE" >> $LOGDIR/lanes.log
