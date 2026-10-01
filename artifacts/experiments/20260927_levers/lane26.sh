source artifacts/experiments/20260927_levers/lib.sh
# E11c (2026-10-01): why the move decision fails in imagination (missedscroll.py; readings declared in its docstring)
W=artifacts/eda/levers_tworlds_v1
M=artifacts/eda/levers_mamba_integration_v1
until grep -qE "LANE25_DONE|FAILED driftanat" $LOGDIR/lanes.log; do sleep 60; done
[ -f $L/evals/missedscroll_corrt_raw_selffed_s7.json ] || job missedscroll 2000 $PY $L/missedscroll.py \
  $W/corrt_raw_teacher_s7_u18000.pt $W/corrt_raw_teacher_s8_u18000.pt $W/corrt_raw_suffix_s7_u18000.pt \
  $W/corrt_raw_suffix_s8_u18000.pt $M/int_corrg_raw_suffix_s7_fmamba_u18000.pt $W/corrt_raw_selffed_s7.pt
echo "$(date '+%F %T') LANE26_DONE" >> $LOGDIR/lanes.log
