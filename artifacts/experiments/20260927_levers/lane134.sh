source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 03:25: why D draws damage teacher-forced but not in imagination (health_imagination.py, substitution), D and C0.
job himag 1500 $PY $L/health_imagination.py artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE134_DONE" >> $LOGDIR/lanes.log
