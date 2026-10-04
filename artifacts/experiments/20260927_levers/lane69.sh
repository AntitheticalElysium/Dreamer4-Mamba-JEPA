source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-04: check_recall v2 split (same / moved view slot, age; slot_bias declared in its docstring): the s7 36k pair, the s8
# 30k pair, attention s7 100k.
W=artifacts/eda/levers_tworlds_v1
job check_recall_split 2000 $PY $L/check_recall.py $W/corrt_raw_teacher_s7_u36000.pt $W/corrt_raw_teacher_s7_fmamba_u36000.pt \
  $W/corrt_raw_teacher_s8_u36000_at30000.pt $W/corrt_raw_teacher_s8_fmamba_u36000_at30000.pt $W/corrt_raw_teacher_s7_u100000.pt
echo "$(date '+%F %T') LANE69_DONE" >> $LOGDIR/lanes.log
