source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-03: check_h16_traj (readings in its docstring) on the 36k teacher worlds, seeds 7 and 8.
W=artifacts/eda/levers_tworlds_v1
[ -f $L/evals/h16traj/result.json ] || job check_h16_traj 2400 $PY $L/check_h16_traj.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s8_u36000.pt
echo "$(date '+%F %T') LANE50_DONE" >> $LOGDIR/lanes.log
