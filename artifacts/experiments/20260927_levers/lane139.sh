source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 09:08: zombie diagnostic with the calibration test (part 3): D and C0.
job hzombie_calib 1500 $PY $L/health_zombie.py D=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt C0=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE139_DONE" >> $LOGDIR/lanes.log
