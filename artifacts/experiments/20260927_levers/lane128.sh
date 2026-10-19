source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 00:24: paired intervals for E' (LP-FT) against C0 / G / GE and its own 2k / 4k (health_ci.py).
until grep -q "LANE127_DONE" $LOGDIR/lanes.log; do sleep 20; done
job hci_evlp 800 $PY $L/health_ci.py evlp C0=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  G=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt GE=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w15__ext.pt \
  EP2k=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at2000__w15__ext.pt EP4k=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at4000__w15__ext.pt EP=artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000__w15__ext.pt \
  --pairs C0:EP C0:EP4k GE:EP EP4k:EP
echo "$(date '+%F %T') LANE128_DONE" >> $LOGDIR/lanes.log
