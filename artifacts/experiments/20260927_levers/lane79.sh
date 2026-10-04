source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: the thesis's Experiment 0 in the per-tile world (check_delta; readings delta_flags_error, delta_beyond_saliency,
# delta_vs_ensemble declared in its docstring): M6 s7 and s8.
W=artifacts/eda/levers_tworlds_v1
job check_delta 2000 $PY $L/check_delta.py $W/corrt_raw_teacher_s7_fmamba_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
echo "$(date '+%F %T') LANE79_DONE" >> $LOGDIR/lanes.log
