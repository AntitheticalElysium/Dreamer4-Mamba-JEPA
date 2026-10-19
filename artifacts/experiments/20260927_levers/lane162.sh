source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 16:50: zombies entering the view (recall and false entries on scroll steps), one step ahead, D / E22Z / E22M.
until grep -q "LANE161_DONE" $LOGDIR/lanes.log; do sleep 30; done
job enter_zombie 1500 $PY $L/health_zombie.py De=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt E22Ze=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt E22Me=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt
echo "$(date '+%F %T') LANE162_DONE" >> $LOGDIR/lanes.log
