source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_position v2 (scan length vs history: w5_same, w5_rep1; readings in its docstring; run 1's conditions rerun).
W=artifacts/eda/levers_tworlds_v1
job check_position 2000 $PY $L/check_position.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt
echo "$(date '+%F %T') LANE66_DONE" >> $LOGDIR/lanes.log
