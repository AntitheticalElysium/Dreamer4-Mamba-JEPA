source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_recall --futures --imagined (self-fed recall; imagined_recall_edge declared in its docstring) on the 6-frame
# parents; plus the s8 36k pair (M6 s8 finished 13:20; reported, not part of the declared reading) and the recall curve's s8 36k point.
W=artifacts/eda/levers_tworlds_v1
job check_recall_imagined 2000 $PY $L/check_recall.py --futures --imagined $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_fmamba_u36000_at30000.pt \
  $W/corrt_raw_teacher_s8_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
job check_recall_s8_36k 2000 $PY $L/check_recall.py $W/corrt_raw_teacher_s8_u36000.pt $W/corrt_raw_teacher_s8_fmamba_u36000.pt
echo "$(date '+%F %T') LANE71_DONE" >> $LOGDIR/lanes.log
