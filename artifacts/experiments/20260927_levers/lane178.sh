source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 10:33: decision relevance by region (check_h16_subst, deepeval DEV, real_k heads): which part of the imagined state carries the
# decision gap, for the L16 worlds (window 15): M16 (L1) and D from scratch.
V=artifacts/eda/levers_tworlds_v1
job h16subst_L16 3000 $PY $L/check_h16_subst.py $V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --window 15
echo "$(date '+%F %T') LANE178_DONE" >> $LOGDIR/lanes.log
