source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 16:47: where the extra zombie cells are, one step ahead (new / old / other candidates / elsewhere; true frame as reference).
until grep -q "LANE160_DONE" $LOGDIR/lanes.log; do sleep 30; done
job where_zombie 1500 $PY $L/health_zombie.py Dw=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt E22Zw=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_zwauto_L16_di_b16_u2000.pt E22Mw=artifacts/eda/levers_tworlds_v1/dworld_a_s7_from2000_revalways_mwauto_L16_di_b16_u2000.pt
echo "$(date '+%F %T') LANE161_DONE" >> $LOGDIR/lanes.log
