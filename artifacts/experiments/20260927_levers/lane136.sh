source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 08:32: the from-scratch GE endpoint read now (lane122 re-extracts stage-1 snapshots first; same deterministic files).
job hext_early_corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000.pt --ext
job hext_early_w10_corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000.pt --window 10 --ext
job inflight_scratch_ge_end 600 $PY $L/health_inflight.py corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000
job row9_scratch_ge 600 $PY $L/health_row9.py corrt_rawlong_teacher_s7_fmamba_L16b40_from36000__w10 corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000__w10__ext corrt_rawlong_teacher_s7_fmamba_gl_ev_L16b40_from36000__w15__ext
echo "$(date '+%F %T') LANE136_DONE" >> $LOGDIR/lanes.log
