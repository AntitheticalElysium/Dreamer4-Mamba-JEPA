source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 08:42: test 3 decisive intervals: from-scratch GE endpoint vs M16 and vs the continuation GE, rows 14 and 9.
job hci_scratch_ge_end_w15 800 $PY $L/health_ci.py scratch_ge_end_w15 M16=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  GEc=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt GEs=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000__w15__ext.pt --pairs M16:GEs GEc:GEs M16:GEc
job hci_scratch_ge_end_w10 800 $PY $L/health_ci.py scratch_ge_end_w10 M16=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt \
  GEc=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt GEs=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000__w10__ext.pt --pairs M16:GEs GEc:GEs M16:GEc
echo "$(date '+%F %T') LANE137_DONE" >> $LOGDIR/lanes.log
