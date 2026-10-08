source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 02:45: E21 cost in L1 on teval's held roots and tf path (health_l1cost.py), beside teval's MSE / copy.
job hl1cost 1100 $PY $L/health_l1cost.py M16=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt C0=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt G=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt \
  GE=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt EP=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000.pt \
  D=artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --pairs M16:C0 C0:G C0:GE C0:EP C0:D
echo "$(date '+%F %T') LANE132_DONE" >> $LOGDIR/lanes.log
