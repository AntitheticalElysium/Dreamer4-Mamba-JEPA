source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 night: loss-of-plasticity signatures (health_plasticity.py metrics): fresh init, M6 s7 at 6k / 18k / 36k, M16 s7, C0, G.
job hplast_metrics 1500 $PY $L/health_plasticity.py metrics init artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_u36000_at6000.pt \
  artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_u36000_at18000.pt artifacts/eda/levers_tworlds_v1/corrt_raw_teacher_s7_fmamba_u36000.pt \
  artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt
echo "$(date '+%F %T') LANE117_DONE" >> $LOGDIR/lanes.log
