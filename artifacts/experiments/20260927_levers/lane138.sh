source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 09:00: zombies in one-step predictions (by true move type) and in imagination (health_zombie.py): D, C0, M16.
job hzombie 1500 $PY $L/health_zombie.py D=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt C0=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt \
  M16=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE138_DONE" >> $LOGDIR/lanes.log
