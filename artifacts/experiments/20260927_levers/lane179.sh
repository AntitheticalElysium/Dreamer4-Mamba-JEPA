source artifacts/experiments/20260927_levers/lib.sh
# 2026-10-10 17:2x / 18:1x: the gradient side of the currency question, after Codex's review of P-C.
#  gradfacts_*: how much of each objective's update goes to cells whose decoded facts are wrong vs already right (TEST roots).
#  gradadam_*:  P-C again, (a) preconditioned by each run's saved AdamW moments (the actual update, not the raw gradient),
#               (b) entering split into seen / never seen in the window. Endpoints only (the only saved optimizer states).
# CPU tests passed: --facts 11:2x-11:5x, --seen --adam 17:5x-18:1x (2 batches; objective rel_err 0; state weights == world).
V=artifacts/eda/levers_tworlds_v1
job gradfacts_L16 2000 $PY $L/gradanat.py $V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --batches 60 --facts
job gradadam_L16 2300 $PY $L/gradanat.py $V/corrt_rawlong_teacher_s7_fmamba_L16b40_from36000.pt $V/corrt_rawlong_teacher_s7_fmamba_di_L16b40_from36000.pt --batches 60 --seen --adam
job gradadam_stage1 2300 $PY $L/gradanat.py $V/corrt_raw_teacher_s7_fmamba_u36000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000.pt --seen --adam
job gradfacts_stage1 2000 $PY $L/gradanat.py $V/corrt_raw_teacher_s7_fmamba_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000_at6000.pt $V/corrt_raw_teacher_s7_fmamba_u36000.pt $V/corrt_raw_teacher_s7_fmamba_di_u36000.pt --facts
echo "$(date '+%F %T') LANE179_DONE" >> $LOGDIR/lanes.log
