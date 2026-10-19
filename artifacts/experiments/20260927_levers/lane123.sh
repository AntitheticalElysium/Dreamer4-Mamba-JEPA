source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 22:25: noise floor of the h63 ingredient AUCs (health_ci.py): M16, C0, G, GE at 6k (window 15), paired contrasts.
job hci_e21_6k 800 $PY $L/health_ci.py e21_6k M16=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  C0=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt G=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  GE=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt --pairs M16:C0 C0:G C0:GE G:GE
echo "$(date '+%F %T') LANE123_DONE" >> $LOGDIR/lanes.log
