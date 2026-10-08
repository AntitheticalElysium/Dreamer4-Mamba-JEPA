source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 00:16: E' endpoint read now, beside D's training (lane121 re-extracts the same files after D; deterministic).
job hext_early_corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000.pt --ext
job inflight_evlp_end 600 $PY $L/health_inflight.py corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000
job hext_early_w10_corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000.pt --window 10 --ext
echo "$(date '+%F %T') LANE127_DONE" >> $LOGDIR/lanes.log
