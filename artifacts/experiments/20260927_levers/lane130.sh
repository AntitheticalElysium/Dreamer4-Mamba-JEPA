source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-09 00:53: D at row 9 (window 10) for its 2k / 4k snapshots, as they appear.
for U in 2000 4000; do
  until [ -f artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U.pt ]; do sleep 30; done; sleep 40
  job hext_w10_corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U 1100 $PY $L/health_chain.py world artifacts/eda/levers_tworlds_v1/corrt_rawlong_teacher_s7_fmamba_di_L16b40_fromcorrt_rawlong_teacher_s7_fmamba_L16b40_from36000_at$U.pt --window 10 --ext
done
echo "$(date '+%F %T') LANE130_DONE" >> $LOGDIR/lanes.log
