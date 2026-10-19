source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 16:32: the ghost at a moved zombie's old cell, one step ahead (posterior Delta for E22Z / E22M), D vs E22Z vs E22M.
job ghost_zombie 1500 $PY $L/health_zombie.py Dg=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt E22Zg=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt E22Mg=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt
echo "$(date '+%F %T') LANE160_DONE" >> $LOGDIR/lanes.log
