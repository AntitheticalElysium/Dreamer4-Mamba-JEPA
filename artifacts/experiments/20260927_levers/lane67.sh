source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: cross-scroll recall by backbone (check_recall; readings in its docstring): the 36k teacher pair (full vs fmamba, s7)
# and lane 9's matched 6k suffix arms (full, fattn, fmamba, fcanvas, fscan, s7).
W=artifacts/eda/levers_tworlds_v1
job check_recall 2000 $PY $L/check_recall.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_suffix_s7.pt $W/corrt_raw_suffix_s7_fattn.pt $W/corrt_raw_suffix_s7_fmamba.pt $W/corrt_raw_suffix_s7_fcanvas.pt $W/corrt_raw_suffix_s7_fscan.pt
echo "$(date '+%F %T') LANE67_DONE" >> $LOGDIR/lanes.log
