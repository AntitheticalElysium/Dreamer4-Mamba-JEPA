source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 09:15: imagination substitution refined: restore only the zombie neighbourhoods vs an equal number of random cells (D).
job himag_zombie 1500 $PY $L/health_imagination.py artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE140_DONE" >> $LOGDIR/lanes.log
