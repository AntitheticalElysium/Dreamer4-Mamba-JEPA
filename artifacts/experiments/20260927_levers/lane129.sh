source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 00:38: in-flight check of D (Delta-IRIS loss) at 2k / 4k, rows 14 and 9, as snapshots appear.
for U in 2000 4000; do
  until [ -f artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U.pt ]; do sleep 30; done; sleep 30
  job hext_corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U.pt --ext
  job inflight_di_at$U 600 $PY $L/health_inflight.py corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U
done
echo "$(date '+%F %T') LANE129_DONE" >> $LOGDIR/lanes.log
