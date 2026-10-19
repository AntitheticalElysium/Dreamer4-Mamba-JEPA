source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-08 23:15: in-flight check of E' (LP-FT) at 2k / 4k as its snapshots appear (lane121 extracts them only after D).
for U in 2000 4000; do
  until [ -f artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at$U.pt ]; do sleep 30; done; sleep 30
  job hext_corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at$U.pt --ext
  job inflight_evlp_at$U 600 $PY $L/health_inflight.py corrt_rawlong_teacher_s7_fmamba_ev_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_evlp_L16b40_from36000_at$U
done
echo "$(date '+%F %T') LANE126_DONE" >> $LOGDIR/lanes.log
