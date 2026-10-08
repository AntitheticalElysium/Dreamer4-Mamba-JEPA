source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 B1 at the unbiased row: extended states at window 10 (output row 9) for M16 s7 and G, then the calibrated evidence.
job hext_w10_m16 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --window 10 --ext
job hext_w10_g 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt --window 10 --ext
job hevid_row9 1100 $PY $L/health_evidence.py row9 artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt artifacts/eda/health_chain_v1/corrt_rawlong_teacher_s7_fmamba_gl_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10__ext.pt
echo "$(date '+%F %T') LANE116_DONE" >> $LOGDIR/lanes.log
