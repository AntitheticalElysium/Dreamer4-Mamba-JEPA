source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 03:05: D's food cost, diagnosis (health_food.py): teacher-forced fractional drift vs self-fed compounding, C0 vs D vs G.
job hfood 1100 $PY $L/health_food.py e21 C0=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt D=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt \
  G=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE133_DONE" >> $LOGDIR/lanes.log
