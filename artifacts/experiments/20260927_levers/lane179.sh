source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 17:2x: loss currency on the GRADIENT side (gradanat --facts): how much of each objective's update direction goes to cells whose
# decoded facts are wrong vs already right, under both objectives, for the L16 worlds (M16, D scratch; 60 batches, TEST roots) and the
# stage-1 endpoints of the two training runs (M6 / DS1 at 6k and 36k). CPU test 11:2x-11:5x passed (2 batches; default path unchanged).
V=artifacts/eda/levers_tworlds_v1
job gradfacts_L16 2000 $PY $L/gradanat.py $V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --batches 60 --facts
job gradfacts_stage1 2000 $PY $L/gradanat.py $V/corrt_raw_teacher_s7_fmamba_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_u36000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000.pt --facts
echo "$(date '+%F %T') LANE179_DONE" >> $LOGDIR/lanes.log
